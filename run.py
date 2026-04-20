"""Command-line entrypoint for running CocycleNet experiments.

This script parses arguments and launches the trainer. It is safe to import
from other scripts (it will not execute training on import).
"""

import argparse
from argparse import Namespace
import yaml
import logging
import torch
torch.set_float32_matmul_precision('high')
from exp.exp_DS import trainer
import random
import numpy as np

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='CocycleNet for learning Dynamical Systems')

# basic config
parser.add_argument('--dataset', type=str, required=True, default='l63')
parser.add_argument('--model', type=str, required=True, default='CocycleNet',
                    help='model name in [CocycleNet, DeepKoopman, LRAN]')
parser.add_argument('--config', type=str, default=None, help='Path to JSON config file to override args')
# data loader
parser.add_argument('--root_path', type=str, default='./data/Pendulum/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='Pendulum_x.npy', help='data file')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--delta_t', type=float, default=0.1, help='integration step')
parser.add_argument('--downsample', type=int, default=1, help="Downsampling factor for the data" )
# forecasting task
parser.add_argument('--train_len', type=int, default=30, help='training sequence length')
parser.add_argument('--test_len', type=int, default=50, help='test sequence length')
parser.add_argument('--enc_in', type=int, default=2, help='encoder input size')

# CocycleNet
parser.add_argument('--num_generators', type=int, default=0, help='Number of basis generators')
parser.add_argument('--norm', action='store_true', help='whether to use normalization')
parser.add_argument('--dynamic_dim', type=int, default=128, help='latent dimension')
parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension of en/decoder')
parser.add_argument('--hidden_layers', type=int, default=2, help='number of hidden layers of en/decoder')
parser.add_argument('--gate_layers', type=int, default=2, help='number of hidden layers of gate')
parser.add_argument('--complex', action='store_true', help='whether to use the complex latent state space')

# Deepkoopman
parser.add_argument('--radius',action='store_true', help='whether to use radius')
parser.add_argument('--num_complex', type=int, help='The number of complex eigenvalues')
parser.add_argument('--num_real', type=int, default=0, help='The number of real eigenvalues')
parser.add_argument('--omega_dim', type=int, help='the latent dim of omega net')
parser.add_argument('--omega_layers', type=int, help='number of hidden layers of omeganet')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--stable', action='store_true', default=False, help='Use stable output MLP')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--lr_main', type=float, default=1e-3, help='optimizer learning rate')
parser.add_argument('--lr_expert', type=float, default=1e-3, help='optimizer learning rate of expert network')
parser.add_argument('--lr_gate', type=float, default=1e-3, help='optimizer learning rate of gate network')
parser.add_argument('--div_main', type=float, default=1e4, help='optimizer learning rate')
parser.add_argument('--div_expert', type=float, default=1e5, help='optimizer learning rate of expert network')
parser.add_argument('--div_gate', type=float, default=1e4, help='optimizer learning rate of gate network')
parser.add_argument('--loss', type=str, default='mse', help='loss function, [mse, hs1d, hs]')
parser.add_argument('--k', type=int, default=1, help='sobolev norm order')
parser.add_argument('--clip_grad', type=float, default=1.0, help="gradient clipping value")
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument("--decay_weight", type=float, default=0.00001)
parser.add_argument('--warm_pct', type=float, default=0.1, help='warm up percentage of training epochs')
parser.add_argument('--rel', action='store_true', help='whether to use the relative loss')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--seed', type=int, default=2025, help='random seed')

def _load_config(path: str, namespace: Namespace) -> Namespace:
    """Load a YAML config file and merge with parsed args."""
    try:
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load config {path}: {e}")
    for k, v in cfg.items():
        if hasattr(namespace, k):
            setattr(namespace, k, v)
    return namespace


def main():
    # parse args
    args = parser.parse_args()

    # optional json config overrides
    if getattr(args, 'config', None):
        args = _load_config(args.config, args)

    # setup basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    logger.info('Args in experiment: %s', args)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    fix_seed = int(args.seed)
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    if args.use_gpu:
        if args.use_multi_gpu:
            args.devices = args.devices.replace(' ', '')
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]
        else:
            torch.cuda.set_device(args.gpu)

    Exp = trainer

    args.setting = '{}_{}_{}_numg{}_l{}_lrm{}_lre{}_lrg{}_n{}_dyna{}_h{}_hl{}_gl{}_k{}_epo{}'.format(
        args.dataset,
        args.train_len,
        args.model,
        args.num_generators,
        args.loss,
        args.lr_main,
        args.lr_expert,
        args.lr_gate,
        args.norm, 
        args.dynamic_dim,
        args.hidden_dim,
        args.hidden_layers,
        args.gate_layers,
        args.k,
        args.train_epochs)

    Exp(args)  # set experiments

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
