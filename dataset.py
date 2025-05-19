import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import pickle
from utils import MAE

class MinMaxScaler(object):
    def __call__(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        return (tensor - min_val) / (max_val - min_val)

def get_dataset(dataset, batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if dataset == "cifar10":
        # cifar10_transform = transforms.Compose([transforms.ToTensor()])
        trainset = torchvision.datasets.CIFAR10(root='./cifar10_data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)
        train_data = list(iter(trainloader))
        testset = torchvision.datasets.CIFAR10(root='./cifar10_data', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=True)
        test_data = list(iter(testloader))
    elif dataset == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root='./cifar100_data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)
        train_data = list(iter(trainloader))
        testset = torchvision.datasets.CIFAR100(root='./cifar100_data', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=True)
        test_data = list(iter(testloader))
    elif dataset == "svhn":
        # svhn_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728),(0.1201, 0.1231, 0.1052))])
        trainset = torchvision.datasets.SVHN(root='./svhn_data', split='train',
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)
        train_data = list(iter(trainloader))
        testset = torchvision.datasets.SVHN(root='./svhn_data', split='test',
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=True)
        test_data = list(iter(testloader))
    elif dataset == "mnist":
        mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))])
        trainset = torchvision.datasets.MNIST(root='./mnist_data', train=True,
                                                download=True, transform=mnist_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=False)
        train_data = list(iter(trainloader))
        testset = torchvision.datasets.MNIST(root='./mnist_data', train=False,
                                               download=True, transform=mnist_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False)
        test_data = list(iter(testloader))
    elif dataset == "fashionmnist":
        fashionmnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860, ), (0.3205, ))])
        trainset = torchvision.datasets.FashionMNIST(root='./fashionmnist_data', train=True,
                                                    download=True, transform=fashionmnist_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True)
        train_data = list(iter(trainloader))
        
        testset = torchvision.datasets.FashionMNIST(root='./fashionmnist_data', train=False,
                                                    download=True, transform=fashionmnist_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=True)
        test_data = list(iter(testloader))
    elif dataset == "tinyimagenet":
        tinyimagenet_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        data_root = './tiny-imagenet-200'

        trainset = datasets.ImageFolder(root=os.path.join(data_root, 'train'), transform=tinyimagenet_transform)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        train_data = list(iter(trainloader))

        testset = datasets.ImageFolder(root=os.path.join(data_root, 'val'), transform=tinyimagenet_transform)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
        test_data = list(iter(testloader))
    elif dataset == "sst2":
        train_tensor_path = "nlp_data/sst2/train_u_3v_sst2_glove6B300d_sent_len25.tensor_dataset"
        test_tensor_path = "nlp_data/sst2/test_u_3v_sst2_glove6B300d_sent_len25.tensor_dataset"
        
        if os.path.exists(train_tensor_path):
            with open(train_tensor_path, 'rb') as f:
                train_dataset = pickle.load(f)
        else:
            train_dataset = encode_tensor("train", bias=3, sent_length=25, dataset_name="sst2")
        
        if os.path.exists(test_tensor_path):
            with open(test_tensor_path, 'rb') as f:
                test_dataset = pickle.load(f)
        else:
            test_dataset = encode_tensor("test", bias=3, sent_length=25, dataset_name="sst2")

        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        train_data = list(iter(trainloader))
        test_data = list(iter(testloader))
    elif dataset == "subj":
        train_tensor_path = "nlp_data/subj/train_u_3v_subj_glove300d_sent_len35.tensor_dataset"
        test_tensor_path = "nlp_data/subj/test_u_3v_subj_glove300d_sent_len35.tensor_dataset"
        
        if os.path.exists(train_tensor_path):
            with open(train_tensor_path, 'rb') as f:
                train_dataset = pickle.load(f)
        else:
            train_dataset = encode_tensor("train", bias=3, sent_length=35, dataset_name="subj")
        
        if os.path.exists(test_tensor_path):
            with open(test_tensor_path, 'rb') as f:
                test_dataset = pickle.load(f)
        else:
            test_dataset = encode_tensor("test", bias=3, sent_length=35, dataset_name="subj")

        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        train_data = list(iter(trainloader))
        test_data = list(iter(testloader))
    else:
        raise Exception("dataset: " + str(dataset) + " not supported")

    print("Setup data:")
    print("Train: ",len(train_data))
    print("Test: ", len(test_data))
    return train_data, trainloader, test_data, testloader

def encode_tensor(data_type, bias, sent_length, dataset_name):
    # tokenize using glove.6B.300d
    glove_path = 'nlp_data/glove.6B.300d.txt'
    with open(glove_path, 'r', encoding='utf-8') as f:
        glove_embeddings = {}
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            glove_embeddings[word] = vector
    vocab_name="glove6B300d"
    embedding_dim=300
    import nltk
    from tqdm import tqdm
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    zero_embedding = np.array([0] * embedding_dim, dtype=float)
    mean_value = np.mean(list(glove_embeddings.values()))
    variance_value = np.var(list(glove_embeddings.values()))
    left_boundary = mean_value - bias * np.sqrt(variance_value)
    right_boundary = mean_value + bias * np.sqrt(variance_value)

    sample_list = []
    with open(f"{dataset_name}/{data_type}.txt", "r") as f:
        for line in f.readlines():
            temp = line.split('\t')
            sentence = temp[0].strip()
            label = int(temp[1])
            sample_list.append((sentence, label))

    # sample_list = get_samples_from_web(dataset_name, data_type)

    embedding_tuple_list = []
    for i in tqdm(range(len(sample_list))):
        sent_embedding = np.array([[0] * embedding_dim] * sent_length, dtype=float)
        text_list = nltk.word_tokenize(sample_list[i][0])
        label = sample_list[i][1]
        for j in range(sent_length):
            if j >= len(text_list):
                embedding_norm = zero_embedding # zero padding
            else:
                word = text_list[j]
                embedding = glove_embeddings[word] if word in glove_embeddings.keys() else zero_embedding
                # N(0, 1)
                embedding_n01 = (embedding - np.array([mean_value] * embedding_dim)) / np.array([np.sqrt(variance_value)] * embedding_dim)
                embedding_norm = np.array([0] * embedding_dim, dtype=float)
                for k in range(embedding_dim):
                    if embedding[k] < left_boundary:
                        embedding_norm[k] = -bias
                    elif embedding[k] > right_boundary:
                        embedding_norm[k] = bias
                    else:
                        embedding_norm[k] = embedding_n01[k]
                # add abs(left_embedding)
                embedding_norm = (embedding_norm + np.array([np.abs(bias)] * embedding_dim))/(bias * 2)
                # embedding_norm = np.clip(embedding_norm, a_min=0, a_max=1)
            sent_embedding[j] = embedding_norm
        # print(i, sent_embedding)
        embedding_tuple_list.append((torch.tensor(sent_embedding, dtype=float), label))

    dataset = TensorDataset(embedding_tuple_list)

    file_name = f"nlp_data/{dataset_name}/{data_type}_u_{bias}v_{dataset_name}_{vocab_name}_sent_len{sent_length}.tensor_dataset"
    if not os.path.exists(file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(dataset, f, -1)
    print(file_name)
    return dataset

def get_seq_dataset(dataset, batch_size, seq_len, verbose):
    class rnn_eval():
        def __init__(self, batch_size):
            self.batch_size = batch_size
            self.test = False
            self.no_improve_cnt = 0

        def reset(self, epoch_num):
            raise NotImplementedError("Subclasses should implement this method.")

        def train_reset(self, epoch_num):
            self.reset(epoch_num)
            self.test = False

        def test_reset(self, epoch_num):
            self.reset(epoch_num)
            self.test = True

        def summarize(self, p=print):
            raise NotImplementedError("Subclasses should implement this method.")

        def __call__(self, y, t):
            raise NotImplementedError("Subclasses should implement this method.")
        
    class acc_eval(rnn_eval):
        def __init__(self, batch_size):
            super().__init__(batch_size)
            self.reset(0)
            self.max_acc = 0
            self.max_acc_epoch = -1

        def reset(self, epoch_num):
            self.count = 0
            self.correct = 0
            self.epoch_num = epoch_num

        def summarize(self, p=print):
            accuracy = self.correct / self.count if self.count != 0 else 0
            p(f"Total Accuracy: {accuracy}")
            if self.test: 
                self.no_improve_cnt += 1
                if self.max_acc < accuracy:
                    self.max_acc = accuracy
                    self.max_acc_epoch = self.epoch_num
                    self.no_improve_cnt = 0
            return self.max_acc, self.max_acc_epoch, self.no_improve_cnt

        def __call__(self, y, t):
            accuracy = 0
            L, _, B = t.shape
            for i in range(len(y)): # this loop is over the seq_len
                for b in range(B):
                    if torch.argmax(t[i,:,b]) == torch.argmax(y[i][:,b]):
                        accuracy+=1
            if verbose: print(f"Batch Accuracy: {accuracy / (L * B) if L * B != 0 else 0}")
            self.correct += accuracy
            self.count += L * B
        
    class loss_eval(rnn_eval):
        def __init__(self, batch_size):
            super().__init__(batch_size)
            self.reset(0)
            self.min_mse_loss = 1e1000
            self.min_mse_loss_epoch = -1
            self.min_mae_loss = 1e1000
            self.min_mae_loss_epoch = -1

        def reset(self, epoch_num):
            self.total_mse_loss = 0
            self.total_mae_loss = 0
            self.batch_cnt = 0
            self.epoch_num = epoch_num
        
        def summarize(self, p=print):
            avg_mse_loss = self.total_mse_loss / self.batch_cnt
            avg_mae_loss = self.total_mae_loss / self.batch_cnt
            p(f"Total MSE Loss: {avg_mse_loss}\tTotal MAE Loss: {avg_mae_loss}")
            if self.test:
                self.no_improve_cnt += 1
                if self.min_mse_loss > avg_mse_loss:
                    self.min_mse_loss = avg_mse_loss
                    self.min_mse_loss_epoch = self.epoch_num
                    self.no_improve_cnt = 0
                if self.min_mae_loss > avg_mae_loss:
                    self.min_mae_loss = avg_mae_loss
                    self.min_mae_loss_epoch = self.epoch_num
                    self.no_improve_cnt = 0
            return self.min_mse_loss, self.min_mse_loss_epoch, self.no_improve_cnt

        def __call__(self, output_seq, target_seq):
            mse_loss = F.mse_loss(output_seq, target_seq).item()
            mae_loss = MAE(output_seq, target_seq).item()
            if verbose: print(f"Batch MSE Loss: {mse_loss}\tBatch MAE Loss: {mae_loss}")
            self.total_mse_loss += mse_loss
            self.total_mae_loss += mae_loss
            self.batch_cnt += 1

    file_info_dicts = {
        'electricity': 'rnn_data/electricity/electricity.txt',
        'exchange_rate': 'rnn_data/exchange_rate/exchange_rate.txt',
        'pems-bay': 'rnn_data/perms-bay/pems-bay.h5',
    }
    rnn_prefix = "rnn_data"
    os.makedirs(rnn_prefix, exist_ok=True)
    if dataset == "harrypotter":
        dir = os.path.join(rnn_prefix, "harrypotter")
        os.makedirs(dir, exist_ok=True)
        path_to_file = os.path.join(dir, 'harrypotter.txt')
        if not os.path.exists(path_to_file):
            import urllib.request
            print("downloading harrypotter.txt")
            url = 'https://raw.githubusercontent.com/LouisScorpio/datamining/refs/heads/master/tensorflow-program/nlp/word2vec/dataset/%E5%93%88%E5%88%A9%E6%B3%A2%E7%89%B91-7%E8%8B%B1%E6%96%87%E5%8E%9F%E7%89%88.txt'
            urllib.request.urlretrieve(url, path_to_file)
        text = open(path_to_file, 'rb').read().decode(encoding='gbk').replace("　", " ")

        class TextDataset(Dataset):
            def __init__(self, text, char2idx, seq_len):
                self.text = text
                self.char2idx = char2idx
                self.seq_len = seq_len
                self.data = self.create_sequences()

            def create_sequences(self):
                sequences = []
                for i in range(0, len(self.text), self.seq_len + 1):
                    input_seq = self.text[i:i + self.seq_len]
                    target_seq = self.text[i + 1:i + self.seq_len + 1]
                    if i + self.seq_len + 1 <= len(self.text):
                        sequences.append((input_seq, target_seq))
                return sequences

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                input_seq, target_seq = self.data[idx]
                input_tensor = torch.zeros((self.seq_len, len(self.char2idx)))
                target_tensor = torch.zeros((self.seq_len, len(self.char2idx)))
                for i, char in enumerate(input_seq):
                    input_tensor[i][self.char2idx[char]] = 1
                for i, char in enumerate(target_seq):
                    target_tensor[i][self.char2idx[char]] = 1
                return input_tensor, target_tensor

        vocab = sorted(set(text))
        char2idx = {u: i for i, u in enumerate(vocab)}
        idx2char = np.array(vocab)
        vocab_size = len(vocab)
        input_size, output_size = vocab_size, vocab_size

        split_ratio = 0.8
        split_index = int(len(text) * split_ratio)
        train_text = text[:split_index]
        test_text = text[split_index:]

        train_dataset = TextDataset(train_text, char2idx, seq_len)
        test_dataset = TextDataset(test_text, char2idx, seq_len)

        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        rnn_eval = acc_eval(batch_size)
    elif dataset in file_info_dicts: 
        file = file_info_dicts[dataset]
        if file.endswith(".txt"):
            with open(file) as f:
                raw_data = np.loadtxt(f, delimiter=",").astype(np.float32)
        elif file.endswith(".csv"):
            with open(file) as f:
                raw_data = np.loadtxt(
                    f, delimiter=",", skiprows=1, dtype=object
                )[:, 1:].astype(np.float32)
        elif file.endswith(".h5"):
            raw_data = (
                pd.read_hdf(file).reset_index().values[:, 1:].astype(np.float32)
            )
        dat = np.zeros(raw_data.shape, dtype=np.float32)
        n, m = dat.shape

        for i in range(raw_data.shape[1]):
            col_mean = np.mean(raw_data[:, i])
            col_std = np.std(raw_data[:, i])
            dat[:, i] = (raw_data[:, i] - col_mean) / (col_std + np.finfo(float).eps)

        input_size = dat.shape[1] - 1
        output_size = 1
        seqs = []
        dat = dat[:len(dat) - len(dat) % batch_size]
        for i in range(len(dat)):
            seq = dat[i:i + seq_len]
            if(len(seq)!=seq_len): continue
            seqs.append(seq)

        tensors=[]

        for seq in seqs:
            input_tensor = torch.tensor(seq[:,:-1])
            output_tensor = torch.zeros((seq_len, 1))
            output_tensor[:,0] = torch.tensor(seq[:,-1])
            tensors.append((input_tensor, output_tensor))

        split_ratio = 0.8
        split_index = int(len(tensors) * split_ratio)
        train_tensors = tensors[:split_index]
        test_tensors = tensors[split_index:]

        trainloader = DataLoader(train_tensors, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(test_tensors, batch_size=batch_size, shuffle=False)
        
        rnn_eval = loss_eval(batch_size)
    else:
        raise Exception("dataset: " + str(dataset) + " not supported")

    print("Setup data:")
    print("Train: ",len(trainloader.dataset))
    print("Test: ", len(testloader.dataset))
    return trainloader, testloader, input_size, output_size, rnn_eval


class RateDataset(Dataset):
    def __init__(self, data: str):
        super(RateDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        rate_code = self.data[index][0]
        label = int(self.data[index][1])
        return rate_code, label

class TxtDataset(Dataset):
    def __init__(self, data_path: str):
        super(TxtDataset, self).__init__()
        with open(data_path) as fin:
            self.lines = fin.readlines()
        

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index: int):
        line = self.lines[index]
        line = line.strip()
        temp = line.split('\t')
        sentence = temp[0]
        label = int(temp[1])
        return sentence, label
    
class TensorDataset(Dataset):
    def __init__(self, data: str):
        super(TensorDataset, self).__init__()
        self.data = [(d[0].float(), float(d[1])) for d in data]  # 将tensor转换为float32

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        embedding = self.data[index][0].unsqueeze(0)
        label = int(self.data[index][1])
        return embedding, label

def encode_tensor(data_type, bias, sent_length, dataset_name):
    # tokenize using glove.6B.300d
    glove_path = 'nlp_data/glove.6B.300d.txt'
    with open(glove_path, 'r', encoding='utf-8') as f:
        glove_embeddings = {}
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            glove_embeddings[word] = vector
    vocab_name="glove6B300d"
    embedding_dim=300
    import nltk
    from tqdm import tqdm
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    zero_embedding = np.array([0] * embedding_dim, dtype=float)
    mean_value = np.mean(list(glove_embeddings.values()))
    variance_value = np.var(list(glove_embeddings.values()))
    left_boundary = mean_value - bias * np.sqrt(variance_value)
    right_boundary = mean_value + bias * np.sqrt(variance_value)

    sample_list = []
    with open(f"{dataset_name}/{data_type}.txt", "r") as f:
        for line in f.readlines():
            temp = line.split('\t')
            sentence = temp[0].strip()
            label = int(temp[1])
            sample_list.append((sentence, label))

    # sample_list = get_samples_from_web(dataset_name, data_type)

    embedding_tuple_list = []
    for i in tqdm(range(len(sample_list))):
        sent_embedding = np.array([[0] * embedding_dim] * sent_length, dtype=float)
        text_list = nltk.word_tokenize(sample_list[i][0])
        label = sample_list[i][1]
        for j in range(sent_length):
            if j >= len(text_list):
                embedding_norm = zero_embedding # zero padding
            else:
                word = text_list[j]
                embedding = glove_embeddings[word] if word in glove_embeddings.keys() else zero_embedding
                # N(0, 1)
                embedding_n01 = (embedding - np.array([mean_value] * embedding_dim)) / np.array([np.sqrt(variance_value)] * embedding_dim)
                embedding_norm = np.array([0] * embedding_dim, dtype=float)
                for k in range(embedding_dim):
                    if embedding[k] < left_boundary:
                        embedding_norm[k] = -bias
                    elif embedding[k] > right_boundary:
                        embedding_norm[k] = bias
                    else:
                        embedding_norm[k] = embedding_n01[k]
                # add abs(left_embedding)
                embedding_norm = (embedding_norm + np.array([np.abs(bias)] * embedding_dim))/(bias * 2)
                # embedding_norm = np.clip(embedding_norm, a_min=0, a_max=1)
            sent_embedding[j] = embedding_norm
        # print(i, sent_embedding)
        embedding_tuple_list.append((torch.tensor(sent_embedding, dtype=float), label))

    dataset = TensorDataset(embedding_tuple_list)

    file_name = f"nlp_data/{dataset_name}/{data_type}_u_{bias}v_{dataset_name}_{vocab_name}_sent_len{sent_length}.tensor_dataset"
    if not os.path.exists(file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(dataset, f, -1)
    print(file_name)
    return dataset

def get_seq_dataset(dataset, batch_size, seq_len, verbose):
    class rnn_eval():
        def __init__(self, batch_size):
            self.batch_size = batch_size
            self.test = False
            self.no_improve_cnt = 0

        def reset(self, epoch_num):
            raise NotImplementedError("Subclasses should implement this method.")

        def train_reset(self, epoch_num):
            self.reset(epoch_num)
            self.test = False

        def test_reset(self, epoch_num):
            self.reset(epoch_num)
            self.test = True

        def summarize(self, p=print):
            raise NotImplementedError("Subclasses should implement this method.")

        def __call__(self, y, t):
            raise NotImplementedError("Subclasses should implement this method.")
        
    class acc_eval(rnn_eval):
        def __init__(self, batch_size):
            super().__init__(batch_size)
            self.reset(0)
            self.max_acc = 0
            self.max_acc_epoch = -1

        def reset(self, epoch_num):
            self.count = 0
            self.correct = 0
            self.epoch_num = epoch_num

        def summarize(self, p=print):
            accuracy = self.correct / self.count if self.count != 0 else 0
            p(f"Total Accuracy: {accuracy}")
            if self.test: 
                self.no_improve_cnt += 1
                if self.max_acc < accuracy:
                    self.max_acc = accuracy
                    self.max_acc_epoch = self.epoch_num
                    self.no_improve_cnt = 0
            return self.max_acc, self.max_acc_epoch, self.no_improve_cnt

        def __call__(self, y, t):
            accuracy = 0
            L, _, B = t.shape
            for i in range(len(y)): # this loop is over the seq_len
                for b in range(B):
                    if torch.argmax(t[i,:,b]) == torch.argmax(y[i][:,b]):
                        accuracy+=1
            if verbose: print(f"Batch Accuracy: {accuracy / (L * B) if L * B != 0 else 0}")
            self.correct += accuracy
            self.count += L * B
        
    class loss_eval(rnn_eval):
        def __init__(self, batch_size):
            super().__init__(batch_size)
            self.reset(0)
            self.min_mse_loss = 1e1000
            self.min_mse_loss_epoch = -1
            self.min_mae_loss = 1e1000
            self.min_mae_loss_epoch = -1

        def reset(self, epoch_num):
            self.total_mse_loss = 0
            self.total_mae_loss = 0
            self.batch_cnt = 0
            self.epoch_num = epoch_num
        
        def summarize(self, p=print):
            avg_mse_loss = self.total_mse_loss / self.batch_cnt
            avg_mae_loss = self.total_mae_loss / self.batch_cnt
            p(f"Total MSE Loss: {avg_mse_loss}\tTotal MAE Loss: {avg_mae_loss}")
            if self.test:
                self.no_improve_cnt += 1
                if self.min_mse_loss > avg_mse_loss:
                    self.min_mse_loss = avg_mse_loss
                    self.min_mse_loss_epoch = self.epoch_num
                    self.no_improve_cnt = 0
                if self.min_mae_loss > avg_mae_loss:
                    self.min_mae_loss = avg_mae_loss
                    self.min_mae_loss_epoch = self.epoch_num
                    self.no_improve_cnt = 0
            return self.min_mse_loss, self.min_mse_loss_epoch, self.no_improve_cnt

        def __call__(self, output_seq, target_seq):
            mse_loss = F.mse_loss(output_seq, target_seq).item()
            mae_loss = MAE(output_seq, target_seq).item()
            if verbose: print(f"Batch MSE Loss: {mse_loss}\tBatch MAE Loss: {mae_loss}")
            self.total_mse_loss += mse_loss
            self.total_mae_loss += mae_loss
            self.batch_cnt += 1

    file_info_dicts = {
        'electricity': 'rnn_data/electricity/electricity.txt',
        'exchange_rate': 'rnn_data/exchange_rate/exchange_rate.txt',
        'pems-bay': 'rnn_data/perms-bay/pems-bay.h5',
    }
    rnn_prefix = "rnn_data"
    os.makedirs(rnn_prefix, exist_ok=True)
    if dataset == "harrypotter":
        dir = os.path.join(rnn_prefix, "harrypotter")
        os.makedirs(dir, exist_ok=True)
        path_to_file = os.path.join(dir, 'harrypotter.txt')
        if not os.path.exists(path_to_file):
            import urllib.request
            print("downloading harrypotter.txt")
            url = 'https://raw.githubusercontent.com/LouisScorpio/datamining/refs/heads/master/tensorflow-program/nlp/word2vec/dataset/%E5%93%88%E5%88%A9%E6%B3%A2%E7%89%B91-7%E8%8B%B1%E6%96%87%E5%8E%9F%E7%89%88.txt'
            urllib.request.urlretrieve(url, path_to_file)
        text = open(path_to_file, 'rb').read().decode(encoding='gbk').replace("　", " ")

        class TextDataset(Dataset):
            def __init__(self, text, char2idx, seq_len):
                self.text = text
                self.char2idx = char2idx
                self.seq_len = seq_len
                self.data = self.create_sequences()

            def create_sequences(self):
                sequences = []
                for i in range(0, len(self.text), self.seq_len + 1):
                    input_seq = self.text[i:i + self.seq_len]
                    target_seq = self.text[i + 1:i + self.seq_len + 1]
                    if i + self.seq_len + 1 <= len(self.text):
                        sequences.append((input_seq, target_seq))
                return sequences

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                input_seq, target_seq = self.data[idx]
                input_tensor = torch.zeros((self.seq_len, len(self.char2idx)))
                target_tensor = torch.zeros((self.seq_len, len(self.char2idx)))
                for i, char in enumerate(input_seq):
                    input_tensor[i][self.char2idx[char]] = 1
                for i, char in enumerate(target_seq):
                    target_tensor[i][self.char2idx[char]] = 1
                return input_tensor, target_tensor

        vocab = sorted(set(text))
        char2idx = {u: i for i, u in enumerate(vocab)}
        idx2char = np.array(vocab)
        vocab_size = len(vocab)
        input_size, output_size = vocab_size, vocab_size

        split_ratio = 0.8
        split_index = int(len(text) * split_ratio)
        train_text = text[:split_index]
        test_text = text[split_index:]

        train_dataset = TextDataset(train_text, char2idx, seq_len)
        test_dataset = TextDataset(test_text, char2idx, seq_len)

        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        rnn_eval = acc_eval(batch_size)
    elif dataset in file_info_dicts: 
        file = file_info_dicts[dataset]
        if file.endswith(".txt"):
            with open(file) as f:
                raw_data = np.loadtxt(f, delimiter=",").astype(np.float32)
        elif file.endswith(".csv"):
            with open(file) as f:
                raw_data = np.loadtxt(
                    f, delimiter=",", skiprows=1, dtype=object
                )[:, 1:].astype(np.float32)
        elif file.endswith(".h5"):
            raw_data = (
                pd.read_hdf(file).reset_index().values[:, 1:].astype(np.float32)
            )
        dat = np.zeros(raw_data.shape, dtype=np.float32)
        n, m = dat.shape

        for i in range(raw_data.shape[1]):
            col_mean = np.mean(raw_data[:, i])
            col_std = np.std(raw_data[:, i])
            dat[:, i] = (raw_data[:, i] - col_mean) / (col_std + np.finfo(float).eps)

        input_size = dat.shape[1] - 1
        output_size = 1
        seqs = []
        dat = dat[:len(dat) - len(dat) % batch_size]
        for i in range(len(dat)):
            seq = dat[i:i + seq_len]
            if(len(seq)!=seq_len): continue
            seqs.append(seq)

        tensors=[]

        for seq in seqs:
            input_tensor = torch.tensor(seq[:,:-1])
            output_tensor = torch.zeros((seq_len, 1))
            output_tensor[:,0] = torch.tensor(seq[:,-1])
            tensors.append((input_tensor, output_tensor))

        split_ratio = 0.8
        split_index = int(len(tensors) * split_ratio)
        train_tensors = tensors[:split_index]
        test_tensors = tensors[split_index:]

        trainloader = DataLoader(train_tensors, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(test_tensors, batch_size=batch_size, shuffle=False)
        
        rnn_eval = loss_eval(batch_size)
    else:
        raise Exception("dataset: " + str(dataset) + " not supported")

    print("Setup data:")
    print("Train: ",len(trainloader.dataset))
    print("Test: ", len(testloader.dataset))
    return trainloader, testloader, input_size, output_size, rnn_eval


class RateDataset(Dataset):
    def __init__(self, data: str):
        super(RateDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        rate_code = self.data[index][0]
        label = int(self.data[index][1])
        return rate_code, label

class TxtDataset(Dataset):
    def __init__(self, data_path: str):
        super(TxtDataset, self).__init__()
        with open(data_path) as fin:
            self.lines = fin.readlines()
        

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index: int):
        line = self.lines[index]
        line = line.strip()
        temp = line.split('\t')
        sentence = temp[0]
        label = int(temp[1])
        return sentence, label
    
class TensorDataset(Dataset):
    def __init__(self, data: str):
        super(TensorDataset, self).__init__()
        self.data = [(d[0].float(), float(d[1])) for d in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        embedding = self.data[index][0].unsqueeze(0)
        label = int(self.data[index][1])
        return embedding, label