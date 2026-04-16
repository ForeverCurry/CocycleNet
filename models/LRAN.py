import torch
import torch.nn as nn
from utils import MLP, HsLoss, HsLoss1D, StaticNormalizer

class model(nn.Module):
    def __init__(self, args, meta):
        super().__init__()
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        if args.norm:
            self.normalizer = StaticNormalizer(meta.mu, meta.std)
        else:
            self.normalizer = StaticNormalizer(0., 1.)
        # Encoder MLP operating on input channels
        self.encoder = MLP(args.enc_in, args.dynamic_dim, args.hidden_dim,
                   args.hidden_layers, stable=args.stable)

        self.K = nn.Linear(args.dynamic_dim, args.dynamic_dim, bias=False)

        # Decoder MLP mapping latent back to input channels
        self.decoder = MLP(args.dynamic_dim, args.enc_in, args.hidden_dim,
                args.hidden_layers, stable=args.stable)
        self.lin_fn = self._get_loss_fn(args)
        self.step = None
        self.lin_loss = None

    def _get_loss_fn(self, args):
        if args.loss == 'mse': return nn.MSELoss()
        if args.loss == 'hs1d': return HsLoss1D(rel=args.rel, k=args.k)
        return HsLoss(rel=args.rel, k=args.k)
    
    def forward(self, x):
        # x: [batch, t, dim]

        B, T, C = x.shape
        x_norm = self.normalizer.encode(x)
        if self.step == None:
            self.step = T
        z = self.encoder(x_norm) # latent state
        v=z[:,:1]
        latent = [v]
        for _ in range(1,self.step):
            v = self.K(v)
            latent.append(v)
        latent = torch.cat(latent, dim=1)
        self.lin_loss = self.lin_fn(latent[:,:T], z)
        #### reconstruction
        y = self.decoder(latent)  
        return y, z, latent
    
    def update_steps(self, steps):
        self.step = steps