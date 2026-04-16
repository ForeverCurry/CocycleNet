"""Operator utilities used by the models.

This module contains small neural building blocks and optimizer-like
components used to construct the model generators and gating networks.
"""

from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    '''
    Multilayer perceptron to encode/decode high dimension representation of sequential data
    '''
    def __init__(self, 
                 f_in, 
                 f_out, 
                 hidden_dim=128, 
                 hidden_layers=2, 
                 stable=False): 
        super(MLP, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.activation = nn.SiLU()

        self.stable = stable
        if hidden_layers>1:
            layers = [nn.Linear(self.f_in, self.hidden_dim), 
                  self.activation]
            for _ in range(self.hidden_layers-2):
                layers += [nn.Linear(self.hidden_dim, self.hidden_dim),
                        self.activation]
            
            layers += [nn.Linear(hidden_dim, f_out)]
        else:
            layers = [nn.Linear(self.f_in, self.f_out)]

        self.layers = nn.Sequential(*layers)
        self._init_weights()
      
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.activation_type == 'Silu':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif self.activation_type == 'tanh':
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if isinstance(self.layers[-1], nn.Linear) and self.stable:
            nn.init.normal_(self.layers[-1].weight, mean=0, std=0.01)
    def forward(self, x):
        # x:     B x S x f_in
        # y:     B x S x f_out
        y = self.layers(x)
        return y

class CayleyEvolution(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Register identity matrix as a buffer to avoid recreating it each forward
        self.register_buffer('I', torch.eye(dim))

    def forward(self, G, z):
        """Evolve latent state using Cayley-like integration.

        Args:
            G: Tensor of shape [B, D, D]. Continuous-time generator matrices
               (already multiplied by delta_t).
            z: Tensor of shape [B, D]. Current latent state.

        Returns:
            Next latent state tensor of shape [B, D].
        """
        
        batch_size = G.shape[0]
        I_batch = self.I.unsqueeze(0).expand(batch_size, -1, -1)
        
        # assemble linear operators for Cayley update
        A_left = I_batch - 0.5 * G
        A_right = I_batch + 0.5 * G

        # compute right-hand side: [B, D, D] @ [B, D, 1] -> [B, D, 1]
        b = torch.bmm(A_right, z.unsqueeze(-1))
        
        return torch.linalg.solve(A_left, b).squeeze(-1)
    def cayley_matrix(self, G):
        """Compute the Cayley transform of matrix G.

        The Cayley transform C = (I - 0.5 G)^{-1} (I + 0.5 G) maps a
        continuous-time generator matrix to a discrete-time transition
        approximation.

        Args:
            G: Tensor of shape [D, D] or [B, D, D].

        Returns:
            Tensor with the same shape as G representing the Cayley transform.
        """
        # Normalize input to batch format: accept either [D,D] or [B,D,D]
        if G.dim() == 2:
            # single matrix provided; add batch dimension
            G = G.unsqueeze(0)
            batch_size = 1
            single = True
        elif G.dim() == 3:
            batch_size = G.shape[0]
            single = False
        else:
            raise ValueError("G must be a 2D or 3D tensor")

        I_batch = self.I.unsqueeze(0).expand(batch_size, -1, -1)

        A_left = I_batch - 0.5 * G
        A_right = I_batch + 0.5 * G

        C = torch.linalg.solve(A_left, A_right)

        # If a single matrix was provided, remove the added batch dim
        if single:
            C = C.squeeze(0)
        return C
    
class complex_opt(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        if dim % 2==0:
            n = dim // 2
        else:
            raise ValueError
        
        self._cached_U = None
        self._cache_valid = False

        # Learnable complex matrix represented by separate real/imag parts
        self.real = nn.Parameter(torch.randn(n, n) / sqrt(n))
        self.imag = nn.Parameter(torch.randn(n, n) / sqrt(n))

    def get_matrix(self, force_recompute=False):
        if not self._cache_valid or force_recompute:
            top = torch.cat([self.real, -self.imag], dim=1)
            bottom = torch.cat([self.imag, self.real], dim=1)
            self._cached_U = torch.cat([top, bottom], dim=0)
            self._cache_valid = True
        return self._cached_U
    
    def invalidate_cache(self):
        self._cache_valid = False

    def eval(self):
        super().eval()
        self.invalidate_cache()
    
class real_opt(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Learnable real matrix parameter
        self.real = nn.Parameter(torch.randn(dim, dim) / sqrt(dim))

    def get_matrix(self, force_recompute=False):
        return self.real
    
    def invalidate_cache(self):
        self._cache_valid = False

    def eval(self):
        super().eval()
        self.invalidate_cache()

class Ada_opt(nn.Module):
    def __init__(self, embed_dim:int, hidden_dim:int, gate_layers:int, 
                  num_generators:int, complex:bool=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_generators = num_generators

        if not complex:
            self.generators = nn.ModuleList([real_opt(embed_dim) for _ in range(self.num_generators)])
        else:   
            self.generators = nn.ModuleList([complex_opt(embed_dim) for _ in range(self.num_generators)] )

        self.gate = MLP(embed_dim, self.num_generators, hidden_dim, gate_layers, stable=True)

        # Stochastic softmax module used during training for exploration
        self.softmax = NoisySoftmax()

    def get_expert_generators(self):
        return torch.stack([exp.get_matrix(force_recompute=True) for exp in self.generators], dim=0).contiguous()


    def get_matrix(self, z, delta_t, generators, x=None):
        x_ = z if x is None else torch.cat([x, z], dim=-1)

        gate_logits = self.gate(x_)
        if self.training:
            # During training use noisy softmax for exploration
            expert_probs = self.softmax(gate_logits)
            self.softmax.step_forward()
        else:
            expert_probs = F.softmax(gate_logits, dim=-1)

        K_soft = torch.einsum('be,edj->bdj', expert_probs, generators)

        return delta_t * K_soft, expert_probs  # [B, D, D]
 
class NoisySoftmax(nn.Module):
    def __init__(self, init_std=1.0, final_std=0.0, total_steps=15000):
        super().__init__()
        self.init_std = init_std
        self.final_std = final_std
        self.total_steps = total_steps
        self.register_buffer('step', torch.tensor(0))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply annealed Gaussian noise to logits, then softmax.

        Args:
            logits: Raw gate logits of shape (B, E).
            num_skew: Index to split logits when using cartesian decomposition.
        """
        current_std = self.get_current_std()
        noise = torch.randn_like(logits) * current_std
        noisy_logits = logits + noise
        return F.softmax(noisy_logits, dim=-1)

    def get_current_std(self):
        ratio = (self.step.float() / self.total_steps).clamp(0, 1)
        return self.init_std * (1 - ratio) + self.final_std * ratio

    def step_forward(self):
        self.step += 1