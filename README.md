# Welcome to our code repo for PGrad

See the image below for an illustration.
<p align='center'><img src="https://github.com/QData/PGrad/blob/main/figures/Framework.png" alt="Training Paradigm" width="700"/></p>

To get a quick introduction of PGrad, please check out our ICLR [Slides] (https://github.com/QData/PGrad/blob/main/PGrad_ICLR.pdf)

This is the official PyTorch implementation of our ICLR 2023 paper [PGrad: Learning Principal Gradients for Domain Generalization](https://openreview.net/pdf?id=CgCmwcfgEdH).

Machine learning models fail to perform when facing out-of-distribution (OOD) domains, a challenging task known as domain generalization (DG). In this work, we develop a novel DG training strategy, we call PGrad, to learn a robust gradient direction, improving models' generalization ability on unseen domains.  The proposed gradient aggregates the principal directions of a sampled roll-out optimization trajectory that measures the training dynamics across all training domains. PGrad's gradient design forces the DG training to ignore domain-dependent noise signals and updates all training domains with a robust direction covering main components of parameter dynamics.  We further improve PGrad via bijection-based computational refinement and directional plus length-based calibrations. Our theoretical proof connects PGrad to the spectral analysis of Hessian in training neural networks. 

## How to use our code

This repository contains code to reproduce the main results of our paper.

Our code is largely based on the DomainBed(https://github.com/facebookresearch/DomainBed).

To train a model: 

```sh
python -m domainbed.scripts.train\
       --data_dir=./domainbed/data/TerraIncognita/\
       --algorithm PGrad\
       --dataset PACS\
       --test_env 0
```

To lunch a sweep and reproduce the main results:

```sh
python -m domainbed.scripts.sweep launch\
       --data_dir=./domainbed/data/ \
       --output_dir ./results\
       --command_launcher multi_gpu\ 
       --algorithms  PGrad\
       --datasets OfficeHome TerraIncognita PACS VLCS\ 
       --n_hparams 2\
       --n_trials 3\
       --single_test_envs\
```

## Analysis code validating the effectiveness of the sequential training in PGrad

PGrad learns robust update direcitions from those principal components of the sampled parameter trajectories. The trajectories are sampled by performing sequential training on each training domains. We analyze in the paper that sequential training will reinforce robust directions; parallel training will enlarge domain-specific noises and suppress robust directions. 

For comparison, run PGradParallel with:

```sh
python -m domainbed.scripts.train\
       --data_dir=./domainbed/data/ \
       --algorithm PGradParallel\
       --dataset VLCS\
       --test_env 0
```

Using the test domain accuracy as the indicator, PGrad will consistently outperform ERM, while PGradParallel has the worst generalization ability.

## Reference

If you found our work helpful, we would appreciate if you considered citing the paper that is most relevant to your work:

```
@inproceedings{
       wang2023pgrad,
       title={{PG}rad: Learning Principal Gradients For Domain Generalization},
       author={Zhe Wang and Jake Grigsby and Yanjun Qi},
       booktitle={The Eleventh International Conference on Learning Representations },
       year={2023},
       url={https://openreview.net/forum?id=CgCmwcfgEdH}
       }
```



