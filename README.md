# Simple Implementation of ProGAN

This is a simple implementation of ProGAN from Nvidia. The link of thesis is [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196).

## Introduction

The directory structure should be the following:

```markdown
/ProGAN
|--/data
|--/images
|--/save
|--/workdir
|--dataset.py
|--network.py
|--test.py
|--train.py
```

* `/data`: This directory should include your dataset. Place images directly into it without any additional substructures.
* `/images`: This directory is used to generate sample images during training, allowing the user to monitor the training process.
* `/save`: Models are saved here during training. Save the model after every epoch.
* `/workdir`: This directory is where the current model is located.
* `dataset.py`: This is a tool for loading data from the `/data` directory.
* `network.py`: This file defines the structure of the network.
* `test.py`: This is a tool for testing the dataloader and model.
* `train.py`: This file is used to train the model.

## Installation

If you want to use this reposory, then you just need to clone it.

`git clone https://github.com/Keyan0412/ProGAN.git`

## Usage

All training settings is in `train.py`. If you want to use it, change the parameters and then run it.

## Potential Problems

When training, a warning will occur:

```python
/data/miniconda/envs/torch/lib/python3.10/site-packages/torch/autograd/graph.py:768: UserWarning: Attempting to run cuBLAS, but there was no current CUDA context! Attempting to set the primary context... (Triggered internally at /opt/conda/conda-bld/pytorch_1720165264854/work/aten/src/ATen/cuda/CublasHandlePool.cpp:135.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
```

Upon my verification, the aforementioned warning will not impact the speed of training.

If you set the size of batch as 1, the following warning will occur because of the mini-batch standard deviation block:

```python
/data/coding/network.py:152: UserWarning: std(): degrees of freedom is <= 0. Correction should be strictly less than the reduction factor (input numel divided by output numel). (Triggered internally at /opt/conda/conda-bld/pytorch_1720165264854/work/aten/src/ATen/native/ReduceOps.cpp:1808.)
  torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
```
