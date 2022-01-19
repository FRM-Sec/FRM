# FRM: Federated Reputation Models for Secure Sybil Mitigation

### This repository is based on a fork of AAAI previous work, a benchmark for residual based Federated Learning: Attack-Resistant Federated Learning with Residual-based Reweighting

This is a PyTorch implementation of our [paper](https://arxiv.org/abs/1912.11464). We present a novel aggregation algorithm with residual-based reweighting to defend federated learning. Our aggregation algorithm combines repeated median regression with the reweighting scheme in iteratively reweighted least squares. Our experiments show that our aggregation algorithm outperforms other alternative algorithms in the presence of label-flipping, backdoor, and Gaussian noise attacks. We also provide theoretical guarantees for our aggregation algorithm.
  * This repository used code from [federated learning](https://github.com/shaoxiongji/federated-learning).
  * Previous Results: their algorithm can successfully defend Gaussian Noise Attacks, Label-Flipping Attacks and Backdoor Attacks. 

| # of attackers  | 0      | 1      | 2      | 3      | 4      | Average |
|-----------------|--------|--------|--------|--------|--------|---------|
| FedAvg          | 88.96% | 85.74% | 82.49% | 82.35% | 82.11% | 84.33%  |
| Median          | 88.11% | 87.69% | 87.15% | 85.85% | 82.01% | 86.16%  |
| Trimmed Mean    | 88.70% | 88.52% | 87.44% | 85.36% | 82.35% | 86.47%  |
| Repeated Median | 88.60% | 87.76% | 86.97% | 85.77% | 81.82% | 86.19%  |
| FoolsGold       | 9.70%  | 9.57%  | 10.72% | 11.42% | 9.98%  | 10.28%  |
| Ours            | 89.17% | 88.60% | 86.66% | 86.09% | 85.81% | 87.27%  |

*Results of label-flipping attacks on CIFAR-10 dataset with different numbers of attackers.*

![MNIST](images/MNIST.png?raw=true) ![backdoor](images/backdoor.png?raw=true)

*Results of label-flipping attacks on the MNIST dataset (left). Result of backdoor attack success rate
on CIFAR-10 (right).*


### Authors and License: Tianyue Chu and Alvaro Garcia-Recuero. Licensed at IMDEA NETWORKS under GNU Lesser General Public License v3.0.

## Introduction
Our framework provides secure Federated Learning because we make sure our model bounds updates into a linear regression boundary as the repeated median of this previous work, but with additional features such as a reputation model including freshness, historic of client, etc

## Citing our work:
If you find [Securing Federated Sensitive Topic Classification against
Poisoning Attacks] useful in your research, please consider citing:
```
@article{imdeadtg2022federatedSensitive,
    title={Securing Federated Sensitive Topic Classification against
Poisoning Attacks},
    author={},
    journal={},
    year={2022}
}
```


## Requirements: Software

1. Colab Pro.
2. Pytorch from [the offical repository](https://pytorch.org/).
3. Install packages: tensorboardX, etc.
```
pip install tensorboardX
```


## Preparation for Training & Testing
1. The full code will automatically download CIFAR dataset. This reposiroty is using the SURL dataset only.

2. Colab setup

- Mount drive first.
```
mount drive
```
- Test data paths.
- Install packages and pytorch compatible version with Colab.
- Test GPU is available.


## Usage
### Label-flipping attack experiments
Label Flipping attack on SURL
```
!pipenv run python main_nn.py --model URLNet --dataset URL --epochs 100 --gpu 0 --iid 0 --num_users 7 --num_attackers 3  --attack_label 0 --agg irls --reputation_active 1 --kappa 0.3 --a 0.5 
```

Change `--agg` tag to select aggregation algorithm and change `--num_attackers` to specify the number of attackers. Note that in our experiments the `--num_users` + `--num_attackers` comprises the 100% of users. Also, we add a new set of parameters, a boolean `--reputation_active`, `--a`, `--eta`, `--W` , `--a`, `--z`, `--s` as specified in options.py   

### Backdoor attack experiments

```
!pipenv run python main_nn.py --model URLNet --dataset URL --epochs 100 --gpu 0 --iid 0 --num_users 7 --num_attackers 3 --agg irls --reputation_active 1 --is_backdoor 1 --backdoor_label 0  --kappa 0.3 --a 0.5

```
