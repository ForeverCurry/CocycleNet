# CocycleNet

This is the official codebase for the paper: [Stability-Guaranteed and Structure-Preserving Cocycle Networks for Learning Complex Dynamics].


## Introduction

CocycleNet is a **quasi-linear** and **MET theory-inspired** model for Learning Complex Dynamical Systems. 

<!-- - 
- Compared with the advanced but painstakingly trained deep forecasters, CocycleNet achieves state-of-the-art performance while saving **77.3%** training time and **76.0%** memory footprint.

<p align="center">
<img src="./figures/efficiency.png" height = "240" alt="" align=center />
</p>

- Focus on portraying ubiquitous **non-stationary** time series, CocycleNet shows **enhanced model capacity** empowered by the modern Koopman theory that naturally addresses the nonlinear evolution of real-world time series.
  
<p align="center">
<img src="./figures/motivation.png" height = "180" alt="" align=center />
</p>

- CocycleNet differs from the canonical Koopman Autoencoder without the reconstruction loss function to achieve **end-to-end predictive training**.
  
<p align="center">
<img src="./figures/architecture.png" height = "360" alt="" align=center />
</p> -->


<!-- ## Discussions

There are already several discussions about our paper, we appreciate a lot for their valuable comments and efforts: [[Official]](https://mp.weixin.qq.com/s/10PoA6n51Qok-nJT6_vkhA), [[Openreview]](https://openreview.net/forum?id=jsanMaAxZE), [[Zhihu]](https://www.zhihu.com/question/24189178/answer/3064876852). -->


## Preparation

1. Install Pytorch (>=2.0.0) and other necessary dependencies.
```
pip install -r requirements.txt
```
<!-- 2. All the six benchmark datasets can be obtained from [Google Drive](https://drive.google.com/file/d/1CC4ZrUD4EKncndzgy5PSTzOPSqcuyqqj/view?usp=sharing) or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b8f4a78a39874ac9893e/?dl=1). -->

## Training scripts

We provide the CocycleNet experiment scripts and hyperparameters of all benchmark datasets under the folder `./scripts`.

```bash
bash ./scripts/Pendulum_script/CocycleNet.sh
bash ./scripts/Lorenz_script/CocycleNet.sh
bash ./scripts/KS_script/CocycleNet.sh
```

## Quickstart

1. Create and activate a Python 3.8+ environment

```
python -m venv .venv
source .venv/bin/activate   # or .\\venv\\Scripts\\activate on Windows
pip install -r requirements.txt
```

2. Run an experiment (example)

```
python run.py --dataset Pendulum --model CocycleNet
```

3. Run smoke tests

```
pytest -q
```

## License

This repository is released under the MIT License. See [LICENSE](LICENSE) for details.


<!-- ## Citation

If you find this repo useful, please cite our paper. 

```
@article{liu2023CocycleNet,
  title={CocycleNet: Learning Non-stationary Time Series Dynamics with Koopman Predictors},
  author={Liu, Yong and Li, Chenyu and Wang, Jianmin and Long, Mingsheng},
  journal={arXiv preprint arXiv:2305.18803},
  year={2023}
}
``` -->

## Contact

If you have any questions or want to use the code, please contact:
* suliangyu0917@stu.xjtu.edu.cn
