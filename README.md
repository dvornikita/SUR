# SUR: Selecting from Universal Representations
This repository contains the code to reproduce the few-shot classification experiment on MetaDataset carried out in [Selecting Relevant Features from a Universal Representation for Few-shot Learning](https://arxiv.org/abs/2003.09338).

## Dependencies
This code requires the following:
* Python 3.6 or greater
* PyTorch 1.0 or greater
* TensorFlow 1.14 or greater


## Installation
1. Clone or download this repository.
2. Configure Meta-Dataset:
    * Follow the the "User instructions" in the Meta-Dataset repository (https://github.com/google-research/meta-dataset) for "Installation" and "Downloading and converting datasets". Brace yourself, the full process would take around a day.
    * If you want to test out-of-domain behavior on additional datasets, namely, MNIST, CIFAR10, CIFAR100, follow the installation instructions in the [CNAPs repository](https://github.com/cambridge-mlg/cnaps) to get these datasets. This step is takes little time and we recommended to do it.

## Usage
Here is how to initialize, train and test our method:
#### Initialization

1. Before doing anything, first run the following commands.
    
    ```ulimit -n 50000```
    ```export META_DATASET_ROOT=<root directory of the cloned or downloaded Meta-Dataset repository>```
    ```export RECORDS=<the directory where tf-records of MetaDataset are stored>```
    
    Note the above commands need to be run every time you open a new command shell.
2. Enter the root directory of this project, i.e. the directory where this project was cloned or downloaded.
    
#### Getting the Feature Extractors
1. The easiest way is to download our pre-trained models and use them to obtain a universal set of features directly.
If that is what you want, execute the following command in the root directory of this project:
    ```wget http://thoth.inrialpes.fr/research/SUR/all_weights.zip && unzip all_weights.zip && rm all_weights.zip```
It will donwnload all the weights and place them in the `./weights` directory.

2. Alternatively, instead of using the pretrained models, one can train the models from scratch.
To train 8 independent feature extractors, run:
```./scripts/train_networks.sh```
And/or to train a parametric network family, run:
```./scripts/train_pnf.sh```


#### Testing
1. This step would run our SUR procedure to select appropriate features from a universal feature set.
To select from features obtained with different networks, run:
```python test.py --model.backbone=resnet18```
To select from features obtained with a parametric network family, run:
```python test.py --model.backbone=resnet18_pnf```
Note: If you train the models yourself, be sure you have trained the corresponding extractors.

#### Offline Testing (optional)
To speed up the testing procedure, one could first dump the features on the hard drive, and then use them for selection directly, without needing to run a CNN. To do so, follow the steps:
1. Dump test features extracted from the test episodes on your hard drive by running
```./scripts/dump_test_episodes.sh```

2. Test SUR offline. Depending on your desired feature extractor, run:
```python test_offline.py --model.backbone=resnet18``` or ```python test_offline.py --model.backbone=resnet18_pnf```

This step is useful for those who want to experiment with selection by SUR and want to avoid recomputing the same features every run.

## Expected Results
Below are the results extracted from our papers. The results will vary from run to run by a percent or two up or 
down due to the fact that the Meta-Dataset reader generates different tasks each run, due randomnes in training the networks and in SUR optimization.
The SUR method selects from 8 independently trained feature extractors, while SUR-pnf selectrs from outputs of a parametric
network family, which has fewer parameters. More details about that could be found in the original paper.

**Models trained on all datasets**

| Dataset       | SUR           | SUR-pnf      |
| ---           | ---           | ---          |
| Imagenet      | 56.3±1.1      | 56.4±1.2     |
| Omniglot      | 93.1±0.5      | 88.5±0.8     |
| Aircraft      | 85.4±0.7      | 79.5±0.8     |
| Birds         | 71.4±1.0      | 76.4±0.9     |
| Textures      | 71.5±0.8      | 73.1±0.7     |
| Quick Draw    | 81.3±0.6      | 75.7±0.7     |
| Fungi         | 63.1±1.0      | 48.2±0.9     |
| VGG Flower    | 82.8±0.7      | 90.6±0.5     |
| Traffic Signs | 70.4±0.8      | 65.1±0.8     |
| MSCOCO        | 52.4±1.1      | 52.1±1.0     |
| MNIST         | 94.3±0.4      | 93.2±0.4     |
| CIFAR10       | 66.8±0.9      | 66.4±0.8     |
| CIFAR100      | 56.6±1.0      | 57.1±1.0     |

## Citation
If you use this code, please cite our [Selecting Relevant Features from a Universal Representation for Few-shot Learning](https://arxiv.org/abs/2003.09338) paper:
```
@article{dvornik2020selecting,
  title={Selecting Relevant Features from a Universal Representation for Few-shot Classification},
  author={Dvornik, Nikita and Schmid, Cordelia and Mairal, Julien},
  journal={arXiv preprint arXiv:2003.09338},
  year={2020}
}
