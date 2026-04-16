import copy
import torch
import torch.nn as nn
from torch.func import functional_call, stack_module_state, vmap
from utils import MLP, StaticNormalizer, HsLoss, HsLoss1D

# ---- Complex-conjugate feature blocks ----
def form_complex_conjugate_block(omegas, delta_t):
    mu, omega = omegas[...,1], omegas[...,0]
    exp_m = torch.exp(mu * delta_t)
    c, s = torch.cos(omega * delta_t), torch.sin(omega * delta_t)
    return (exp_m[...,None,None] * torch.stack([
        torch.stack([c, -s], dim=-1),
        torch.stack([s,  c], dim=-1)
    ], dim=-2)).squeeze(1)

class ParallelOmegaNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.n_complex = args.num_complex
        self.n_real = args.num_real
        self.radius_mode = args.radius

        complex_in_dim = 1 if self.radius_mode else 2

        # --------------------------------------------------
        # template networks
        # --------------------------------------------------
        self.complex_template = MLP(
            complex_in_dim,
            2,
            args.omega_dim,
            args.gate_layers,
            stable=True
        )

        self.real_template = MLP(
            1,
            1,
            args.omega_dim,
            args.gate_layers,
            stable=True
        )

        # --------------------------------------------------
        # stacked states
        # --------------------------------------------------
        if self.n_complex > 0:
            (
                self.complex_params,
                self.complex_buffers,
                self.complex_name_map,
            ) = self._stack_module_state(
                self.complex_template,
                self.n_complex,
            )

        if self.n_real > 0:
            (
                self.real_params,
                self.real_buffers,
                self.real_name_map,
            ) = self._stack_module_state(
                self.real_template,
                self.n_real,
            )

    def _stack_module_state(self, template, num_models):
        """
        Create num_models copies of template and stack all parameters/buffers.

        Returns:
            param_dict: ParameterDict with safe parameter names
            buffers: stacked buffers
            reverse_name_map: maps safe names back to original names
        """
        models = [copy.deepcopy(template) for _ in range(num_models)]

        params, buffers = stack_module_state(models)

        param_dict = nn.ParameterDict()
        reverse_name_map = {}

        for orig_name, tensor in params.items():
            # nn.ParameterDict does not allow "." in keys
            safe_name = orig_name.replace(".", "__")

            param_dict[safe_name] = nn.Parameter(tensor)
            reverse_name_map[safe_name] = orig_name

        return param_dict, buffers, reverse_name_map

    def _recover_params(self, param_dict, reverse_name_map):
        """
        Recover original parameter names required by functional_call.
        """
        return {
            reverse_name_map[safe_name]: tensor
            for safe_name, tensor in param_dict.items()
        }

    def _batched_forward(
        self,
        template,
        param_dict,
        buffers,
        reverse_name_map,
        x,
    ):
        """
        Args:
            template: nn.Module template
            param_dict: ParameterDict with shape [n_models, ...]
            buffers: stacked buffers
            reverse_name_map: safe_name -> original_name
            x: [n_models, B, 1, input_dim]

        Returns:
            [n_models, B, 1, output_dim]
        """
        recovered_params = self._recover_params(
            param_dict,
            reverse_name_map,
        )

        def single_forward(single_params, single_buffers, single_x):
            return functional_call(
                template,
                (single_params, single_buffers),
                (single_x,),
            )

        out = vmap(
            single_forward,
            in_dims=(0, 0, 0),
            out_dims=0,
        )(
            recovered_params,
            buffers,
            x,
        )

        return out

    def forward(self, ycoords):
        """
        Args:
            ycoords: [B, 1, N]

        Returns:
            omega: [B, 1, 2*n_complex + n_real]
        """
        B = ycoords.shape[0]
        outputs = []

        # --------------------------------------------------
        # complex part
        # --------------------------------------------------
        if self.n_complex > 0:
            # [B, 1, 2*n_complex]
            complex_part = ycoords[..., : 2 * self.n_complex]

            # [B, 1, 2*n_complex] -> [B, n_complex, 2]
            complex_pairs = complex_part.view(B, self.n_complex, 2)

            if self.radius_mode:
                # [B, n_complex, 1]
                complex_inputs = (complex_pairs ** 2).sum(
                    dim=-1,
                    keepdim=True,
                )
            else:
                # [B, n_complex, 2]
                complex_inputs = complex_pairs

            # [B, n_complex, d] -> [n_complex, B, 1, d]
            complex_inputs = complex_inputs.transpose(0, 1).unsqueeze(2)

            # [n_complex, B, 1, 2]
            complex_out = self._batched_forward(
                self.complex_template,
                self.complex_params,
                self.complex_buffers,
                self.complex_name_map,
                complex_inputs,
            )

            # [n_complex, B, 1, 2] -> [B, n_complex, 2]
            complex_out = complex_out.squeeze(2).transpose(0, 1)

            # [B, n_complex, 2] -> [B, 1, 2*n_complex]
            complex_out = complex_out.reshape(B, 1, -1)

            outputs.append(complex_out)

        # --------------------------------------------------
        # real part
        # --------------------------------------------------
        if self.n_real > 0:
            # [B, 1, n_real]
            real_part = ycoords[..., 2 * self.n_complex :]

            # [B, 1, n_real] -> [B, n_real, 1]
            real_inputs = real_part.view(B, self.n_real, 1)

            # [B, n_real, 1] -> [n_real, B, 1, 1]
            real_inputs = real_inputs.transpose(0, 1).unsqueeze(2)

            # [n_real, B, 1, 1]
            real_out = self._batched_forward(
                self.real_template,
                self.real_params,
                self.real_buffers,
                self.real_name_map,
                real_inputs,
            )

            # [n_real, B, 1, 1] -> [B, n_real, 1]
            real_out = real_out.squeeze(2).transpose(0, 1)

            # [B, n_real, 1] -> [B, 1, n_real]
            real_out = real_out.reshape(B, 1, -1)

            outputs.append(real_out)

        if len(outputs) == 0:
            return ycoords.new_zeros(B, 1, 0)

        return torch.cat(outputs, dim=-1)

# ---- Koopman network ----
class model(nn.Module):
    def __init__(self, args, meta):
        super().__init__()

        self.pred_len = args.pred_len
        if args.norm:
            self.normalizer = StaticNormalizer(meta.mu, meta.std)
        else:
            self.normalizer = StaticNormalizer(0., 1.)
        self.encoder = MLP(args.enc_in, args.dynamic_dim, args.hidden_dim, args.hidden_layers, stable=args.stable)
        self.decoder = MLP(args.dynamic_dim, args.enc_in, args.hidden_dim, args.hidden_layers, stable=args.stable)

        self.omega_net = ParallelOmegaNet(args)
        self.delta_t = args.delta_t
        self.n_complex = args.num_complex
        self.n_real = args.num_real
        self.lin_fn = self._get_loss_fn(args)
        self.lin_loss = None
        self.step = None

    def _get_loss_fn(self, args):
        if args.loss == 'mse': return nn.MSELoss()
        if args.loss == 'hs1d': return HsLoss1D(rel=args.rel, k=args.k)
        return HsLoss(rel=args.rel, k=args.k)
    
    def forward(self, x):
        _, T, _ = x.shape
        x_norm = self.normalizer.encode(x)
        z = self.encoder(x_norm)
        v = z[:,:1]
        latent = [v]
        if self.step == None:
            self.step = T
        for _ in range(self.step-1):
            omegas = self.omega_net(v)
            v = self._advance(v, omegas)
            latent.append(v)
        latent = torch.cat(latent, dim=1)
        y_preds = self.decoder(latent)
        if self.training:
            self.lin_loss = self.lin_fn(latent[:, :T], z)
    
        return y_preds, z, latent

    def _advance(self, y, omegas):
        '''
        y:[batch, t, 2*c+r]
        omegas:[batch_size, t, 2*c+r]
        '''
        B, _, N = omegas.shape
        _, T, _ = y.shape
        parts = []
                # Complex-conjugate block part
        if self.n_complex > 0:
            omegas_c = omegas[..., :2*self.n_complex].reshape(B, -1, self.n_complex, 2)  # (B, T, n_complex, 2)
            y_c = y[..., :2*self.n_complex].reshape(B, T, self.n_complex, 2)  # (B, T, n_complex, 2)

            # form all complex conjugate blocks at once
            block = form_complex_conjugate_block(omegas_c, self.delta_t)  # → (B, n_complex, 2, 2)
            # einsum over all complex blocks in parallel
            parts_c = torch.einsum('bnij,btnj->btni', block, y_c)  # (B, T, n_complex, 2)
            parts.append(parts_c.reshape(B, T, 2*self.n_complex))


        # Real-feature part
        if self.n_real > 0:
            omegas_r = omegas[..., 2*self.n_complex:]  # (B, L, n_real)
            y_r = y[..., 2*self.n_complex:]  # (B, T, n_real)
            scale = torch.exp(omegas_r * self.delta_t)  # (B, L, n_real)
            parts_r = y_r * scale  # adjust broadcasting if different layout is used
            parts.append(parts_r)

        return torch.cat(parts, dim=-1)
    def update_steps(self, steps):
        self.step = steps
