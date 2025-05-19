# DLL

A public framework for Dendritic Localized Learning (DLL).

## Related Paper
* Dendritic Localized Learning: Toward Biologically Plausible Algorithm, [ICML 2025], (https://arxiv.org/pdf/2501.09976).


## Installation
To install DLL in a new conda environment:
```
conda create -n DLL python=[3.8, 3.9, 3.10]
conda activate DLL
git clone https://github.com/Lvchangze/Dendritic-Localized-Learning
cd Dendritic-Localized-Learning
pip install -r requirements.txt
```

If you would like to make changes and run your experiments, use:

`pip install -e .`

## Training
Taking the MLP_Model running on the MNIST dataset as an example:

```sh
python train_mlp.py --dataset mnist --channel 1 --batch_size 128 --num_class 10 --dim_list "[784, 1024, 512, 256, 10]" --layer_type "DLL_FCLayer" --sigma 1.0 --seed 42 --max_iter 200 --epochs 200 --learning_rate 5e-5 --scheduler_type "adam" --earlystop_steps 30
```
Taking the CNN_Model running on the subject prediction task as an example:

```sh
python train_cnn.py --dataset subj --filters 1,2,3 --layers 512,256,128 --learning_rate 0.001
```

Taking the RNN_Model running on the Harry Potter language modeling task as an example:

```sh
python train_rnn.py --dataset harrypotter --seq_len 32 --weight_learning_rate 0.001
```

You can change the parser as you want.

## Datasets

The CIFAR-10, CIFAR-100, SVHN, FashionMNIST and MNIST datasets are provided through `torchvision.datasets` and can be automatically downloaded and loaded via the corresponding interfaces in the `dataset.py` file. Simply set `download=True` and specify the data storage path to use them.
TinyImageNet can be downloaded from  (http://cs231n.stanford.edu/tiny-imagenet-200.zip). Due to its unique data structure, the dataset requires preprocessing via `tiny-imagenet_process.py` before use. Notably, the test set does not contain ground truth labels, so the validation set is typically used as the test set in experiments.
The Sst2 and Subj datasets are available for download from [Kaggle](www.kaggle.com) and are tokenized with GloVe 6B 300d embeddings. All data processing scripts are included in `dataset.py`.

Among the RNN datasets, HarryPotter can be downloaded automatically using `dataset.py`, while the Electricity, Exchange_rate, and PEMS-BAY datasets can be obtained from [Kaggle](www.kaggle.com).



The folder structure of this project is as follows:
```
Dendritic-Localized-Learning
│   README.md 
│   ...
│
└───tiny-imagenet-200
│   │   wnids.txt
│   │   words.txt
│   │
│   └───test
│   │   │   ...
│   │   
│   └───train
│	│	│
│   │   └───n01443537
│   │   │   ...
│   │   
│   └───val
│	│	│
│   │   └───n01443537
│   │   │   ...
```
## Acknowledgement
This repo is built upon (PredictiveCodingBackprop)[https://github.com/BerenMillidge/PredictiveCodingBackprop]. We greatly thank @BerenMillidge and @Tommaso Salvatori for their initial contribution and for their assistance in answering questions and providing guidance during the code implementation.