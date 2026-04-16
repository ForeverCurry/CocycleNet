import torch
import torch.nn as nn
from utils import MLP, CayleyEvolution, Ada_opt, HsLoss, HsLoss1D, StaticNormalizer

class Cocycle_block(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.opt = torch.compile(
            Ada_opt(args.dynamic_dim, args.hidden_dim, args.gate_layers, 
                    args.num_generators, args.complex), 
                    mode="reduce-overhead")
        self.cov_loss = None
        self.step = args.seq_len
        self.delta_t = args.delta_t
        self.cayley = CayleyEvolution(args.dynamic_dim)   
        
    def forward(self, x, step=None):
        B, _, D = x.shape     
        if step is not None:
            self.step = step

        generators = self.opt.get_expert_generators()
        z_all = torch.empty((B, self.step, D), device=x.device, dtype=x.dtype)
        z_all[:, 0] = x[:, 0]

        current_z = x[:, 0]
        for t in range(self.step-1):
            G, _ = self.opt.get_matrix(current_z, self.delta_t, generators)
            current_z = self.cayley(G, current_z)
            z_all[:, t+1] = current_z
        return z_all
    
    def get_transition(self):
        return self.cayley.cayley_matrix(self.opt.get_expert_generators())

class model(nn.Module):
    def __init__(self, args, meta):
        super().__init__()
        if args.norm:
            self.normalizer = StaticNormalizer(meta.mu, meta.std)
        else:
            self.normalizer = StaticNormalizer(0., 1.)
        # Encoder: MLP mapping input channels to latent dimension
        self.encoder = MLP(args.enc_in, args.dynamic_dim, args.hidden_dim,
                           args.hidden_layers, stable=args.stable)
        self.Cocycle = Cocycle_block(args)
        # Decoder: reconstruct from latent to original channel dimension
        self.decoder = MLP(args.dynamic_dim, args.enc_in, args.hidden_dim,
                           args.hidden_layers,  stable=args.stable)
        self.lin_fn = self._get_loss_fn(args)
        self.pred_step = None
        self.lin_loss = None
    def _get_loss_fn(self, args):
        if args.loss == 'mse': return nn.MSELoss()
        if args.loss == 'hs1d': return HsLoss1D(rel=args.rel, k=args.k)
        return HsLoss(rel=args.rel, k=args.k)
    
    def forward(self, x):
        # x: [batch, t, dim]
        _, T, _ = x.shape

        x_norm = self.normalizer.encode(x)
        z = self.encoder(x_norm) 
        latent = self.Cocycle(z, self.pred_step) 
        # Dynamic weighting mechanism: compute linearization loss if training
        if self.training:
            self.lin_loss = self.lin_fn(latent[:, :T], z)

        #### reconstruction
        y = self.decoder(latent)  

        # y = latent  
        return y, z, latent
    
    def update_steps(self, steps):
        self.pred_step = steps
    
