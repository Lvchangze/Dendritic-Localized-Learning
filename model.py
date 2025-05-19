import sys
from FC_layer import *
from CNN_layer import *
import torch
import torchvision.transforms as transforms
from utils import *

class MLP_Model(object):
    def __init__(self, args, device='cuda') -> None:
        self.args = args
        self.dim_list = args.dim_list
        self.build_layers()
        # self.print_layers()
        self.L = len(self.layers)
        self.max_iter = args.max_iter
        self.device = device
        self.u = [[] for _ in  range(self.L+1)]
        self.errors = [[] for _ in range(self.L+1)]
        self.x = [[] for _ in range(self.L+1)]
    
    def build_layers(self):
        if self.args.layer_type == "FCLayer":
            self.layers = [FCLayer(self.dim_list[i], self.dim_list[i+1], learning_rate=self.args.learning_rate, f=tanh, df=tanh_deriv, enable_adam=self.args.enable_adam, device=DEVICE) for i in range(len(self.dim_list)-2)]
            self.layers.append(FCLayer(self.dim_list[-2], self.dim_list[-1], learning_rate=self.args.learning_rate, f=softmax, df=linear_deriv, device=DEVICE, enable_adam=self.args.enable_adam))
        elif self.args.layer_type == "FA_FCLayer":
            self.layers = [FA_FCLayer(self.dim_list[i], self.dim_list[i+1], learning_rate=self.args.learning_rate, f=tanh, df=tanh_deriv) for i in range(len(self.dim_list)-2)]
            self.layers.append(FA_FCLayer(self.dim_list[-2], self.dim_list[-1], learning_rate=self.args.learning_rate, f=softmax, df=linear_deriv))
        elif self.args.layer_type == "Outer_W_FCLayer":
            self.layers = [Outer_W_FCLayer(self.dim_list[i], self.dim_list[i+1], learning_rate=self.args.learning_rate, f=tanh, df=tanh_deriv) for i in range(len(self.dim_list)-2)]
            self.layers[0] = Outer_W_FCLayer(self.dim_list[0], self.dim_list[1], learning_rate=self.args.learning_rate, f=linear, df=linear_deriv) # target: w * f(x) = w * x
            self.layers.append(FCLayer(self.dim_list[-2], self.dim_list[-1], learning_rate=self.args.learning_rate, f=softmax, df=linear_deriv)) # target: softmax(w * f(x)) = softmax(w * x)
        elif self.args.layer_type == "FA_Outer_W_FCLayer":
            self.layers = [FA_Outer_W_FCLayer(self.dim_list[i], self.dim_list[i+1], learning_rate=self.args.learning_rate, f=tanh, df=tanh_deriv) for i in range(len(self.dim_list)-2)]
            self.layers[0] = FA_Outer_W_FCLayer(self.dim_list[0], self.dim_list[1], learning_rate=self.args.learning_rate, f=linear, df=linear_deriv) # target: w * f(x) = w * x
            self.layers.append(FA_FCLayer(self.dim_list[-2], self.dim_list[-1], learning_rate=self.args.learning_rate, f=softmax, df=linear_deriv)) # target: softmax(w * f(x)) = softmax(w * x)
        elif self.args.layer_type == "DLL_FCLayer":
            self.layers = [DLL_FCLayer(self.dim_list[i], self.dim_list[i+1], learning_rate=self.args.learning_rate, f=tanh, df=tanh_deriv, enable_adam=self.args.enable_adam, device=DEVICE) for i in range(len(self.dim_list)-2)]
            self.layers[0] = DLL_FCLayer(self.dim_list[0], self.dim_list[1], learning_rate=self.args.learning_rate, f=tanh, df=tanh_deriv, enable_adam=self.args.enable_adam, device=DEVICE) # target: w * f(x) = w * x
            # self.layers.append(FA_FCLayer(self.dim_list[-2], self.dim_list[-1], learning_rate=self.args.learning_rate, f=softmax, df=linear_deriv)) # target: softmax(w * f(x)) = softmax(w * x)
            self.layers.append(DLL_FCLayer(self.dim_list[-2], self.dim_list[-1], learning_rate=self.args.learning_rate, f=softmax, df=linear_deriv, enable_adam=self.args.enable_adam, device=DEVICE)) # target: softmax(w * f(x)) = softmax(w * x)
          
    def forward(self, inputs):
        self.x[0] = inputs.clone()
        self.u[0] = inputs.clone()
        for i, l in enumerate(self.layers):
            self.x[i+1] = l.forward(self.x[i])
            self.u[i+1] = self.x[i+1].clone()
        return
        
    def iterate(self, target):
        self.x[-1] = target.clone()
        self.errors[-1] = -cross_entropy_deriv(out=self.u[-1], label=self.x[-1])
        
        self.errors[0] = (self.x[0] - self.u[0]) / self.args.sigma
        for i in reversed(range(len(self.layers))):
            if i != 0:
                self.errors[i] = self.layers[i].backward(self.errors[i+1], sigma=self.args.sigma)

        # for iter in range(self.max_iter):
        #     for i in reversed(range(len(self.layers))):
        #         self.errors[i] = (self.x[i] - self.u[i]) / self.args.sigma
                # print(f"self.errors[{i}]: {torch.mean(self.errors[i])}")
                if i != 0:
                    self.x[i] += 0.1 * 2 * (-self.errors[i] + (self.layers[i].backward(self.errors[i+1], sigma=self.args.sigma)))
            # print(f"iter {iter}, torch.mean(self.errors[-2]): {torch.mean(self.errors[-2])}")
        
        return
    
    def update_weights(self, epoch):
        epoch_limit = 0
        for (i, l) in enumerate(self.layers):
            l.update_weights(self.errors[i+1], sigma=self.args.sigma)
            if "Dendritic" in str(type(l).__name__) and epoch >= epoch_limit:
                l.update_theta(self.errors[i], self.errors[i+1], sigma=self.args.sigma)
                continue
        return
    
    def print_layers(self, print=print):
        for i, layer in enumerate(self.layers):
            print(f"Layer {i}: {layer}")
        return

class PC_CNN_Model(object):
    def __init__(self, args, device='cuda') -> None:
        self.args = args
        self.build_layers()
        # self.print_layers()
        self.L = len(self.layers)
        self.max_iter = args.max_iter
        self.device = device
        self.u = [[] for _ in  range(self.L+1)]
        self.errors = [[] for _ in range(self.L+1)]
        self.x = [[] for _ in range(self.L+1)]
    
    def build_layers(self):
        if self.args.layer_type == "ConvLayer":
            if self.args.dataset == "mnist":
                self.layers = [
                    ConvLayer(input_size=self.args.input_size,num_channels=self.args.channel,num_filters=64,kernel_size=5,learning_rate=self.args.learning_rate,f=tanh,df=tanh_deriv,enable_batchnorm = False,enable_adam=self.args.enable_adam),
                    MaxPool(2,device=DEVICE),
                    ConvLayer(12,64,128,3,self.args.learning_rate,tanh,tanh_deriv,enable_adam=self.args.enable_adam),
                    MaxPool(2,device=DEVICE),
                    ConvLayer(5,128,64,3,self.args.learning_rate,tanh,tanh_deriv,enable_adam=self.args.enable_adam),
                    ProjectionLayer(64,3,3,128,self.args.learning_rate,tanh,tanh_deriv,enable_adam=self.args.enable_adam),
                    FCLayer(128,self.args.num_class,self.args.learning_rate,softmax,linear_deriv,enable_adam=self.args.enable_adam)
                ]
            elif self.args.dataset == "fashionmnist":
                self.layers = [
                    ConvLayer(input_size=self.args.input_size,num_channels=self.args.channel,num_filters=64,kernel_size=5,learning_rate=self.args.learning_rate,f=tanh,df=tanh_deriv,enable_batchnorm = False,enable_adam=self.args.enable_adam),
                    MaxPool(2,device=DEVICE),
                    ConvLayer(12,64,128,3,self.args.learning_rate,tanh,tanh_deriv,enable_adam=self.args.enable_adam),
                    MaxPool(2,device=DEVICE),
                    ConvLayer(5,128,64,3,self.args.learning_rate,tanh,tanh_deriv,enable_adam=self.args.enable_adam),
                    ProjectionLayer(64,3,3,128,self.args.learning_rate,tanh,tanh_deriv,enable_adam=self.args.enable_adam),
                    FCLayer(128,self.args.num_class,self.args.learning_rate,softmax,linear_deriv,enable_adam=self.args.enable_adam)
                ]
            elif self.args.dataset == "svhn":
                self.layers = [
                    ConvLayer(input_size=self.args.input_size,num_channels=self.args.channel,num_filters=64,kernel_size=3,learning_rate=self.args.learning_rate,f=tanh,df=tanh_deriv,padding=1,enable_batchnorm = False,enable_adam=self.args.enable_adam),
                    MaxPool(2,device=DEVICE),
                    ConvLayer(16,64,128,3,self.args.learning_rate,tanh,tanh_deriv,1,enable_adam=self.args.enable_adam),
                    MaxPool(2,device=DEVICE),
                    ConvLayer(8,128,64,3,self.args.learning_rate,tanh,tanh_deriv,1,enable_adam=self.args.enable_adam),
                    ProjectionLayer(64,8,8,256,self.args.learning_rate,tanh,tanh_deriv,enable_adam=self.args.enable_adam),
                    FCLayer(256,self.args.num_class,self.args.learning_rate,softmax,linear_deriv,enable_adam=self.args.enable_adam)
                ]
            elif self.args.dataset == "cifar10":
                self.layers = [
                    ConvLayer(input_size=self.args.input_size,num_channels=self.args.channel,num_filters=64,kernel_size=3,learning_rate=self.args.learning_rate,f=tanh,df=tanh_deriv,padding=1,device=DEVICE,enable_batchnorm = False,enable_adam=self.args.enable_adam),
                    MaxPool(2,device=DEVICE),
                    ConvLayer(16,64,128,3,self.args.learning_rate,tanh,tanh_deriv,1,device=DEVICE,enable_adam=self.args.enable_adam),
                    MaxPool(2,device=DEVICE),
                    ConvLayer(8,128,64,3,self.args.learning_rate,tanh,tanh_deriv,1,device=DEVICE,enable_adam=self.args.enable_adam),
                    AvgPool(2,device=DEVICE),
                    ProjectionLayer(64,4,4,256,self.args.learning_rate,tanh,tanh_deriv,device=DEVICE,enable_adam=self.args.enable_adam),
                    FCLayer(256,self.args.num_class,self.args.learning_rate,softmax,linear_deriv,device=DEVICE,enable_adam=self.args.enable_adam)
                ]
        
    def forward(self, inputs):
        self.x[0] = inputs.clone()
        self.u[0] = inputs.clone()
        for i, l in enumerate(self.layers):
            # print(f"i, x[i]: {i}, {self.x[i].shape}")
            self.x[i+1] = l.forward(self.x[i])
            self.u[i+1] = self.x[i+1].clone()
        return
        
    def iterate(self, target):
        self.x[-1] = target.clone()
        self.errors[-1] = -cross_entropy_deriv(out=self.u[-1], label=self.x[-1])
        for iter in range(self.max_iter):
            for i in reversed(range(len(self.layers))):
                self.errors[i] = (self.x[i] - self.u[i]) / self.args.sigma
                if i != 0:
                    self.x[i] -= 0.1 * 2 * (self.errors[i] - (self.layers[i].backward(tanh(self.errors[i+1]), sigma=self.args.sigma)))
            # print(f"iter {iter }, torch.mean(self.errors[-2]): {torch.mean(self.errors[-2])}")
        return
    
    def update_weights(self,epoch):
        for (i, l) in enumerate(self.layers):
            l.update_weights(self.errors[i+1], sigma=self.args.sigma)
            if "Dendritic" in str(type(l).__name__) or "DLL" in str(type(l).__name__):
                l.update_theta(self.errors[i], tanh(self.errors[i+1]), sigma=self.args.sigma)
                continue
        return
    
    def print_layers(self, print=print):
        for i, layer in enumerate(self.layers):
            print(f"Layer {i}: {layer}")
        return

class DLL_CNN_Model(object):
    def __init__(self, args, device='cuda') -> None:
        self.args = args
        self.build_layers()
        # self.print_layers()
        self.L = len(self.layers)
        self.max_iter = args.max_iter
        self.device = device
        self.u = [[] for _ in  range(self.L+1)]
        self.errors = [[] for _ in range(self.L+1)]
        self.x = [[] for _ in range(self.L+1)]
    
    def build_layers(self):
        if self.args.layer_type == "DLL_OUTER_W_ConvLayer":
            self.layers = [
                # mnist:input_size=28, cifar10:input_size=32
                DLL_OUTER_W_ConvLayer(input_size=self.args.input_size,num_channels=self.args.channel,num_filters=32,kernel_size=5,learning_rate=self.args.learning_rate,f=tanh,df=tanh_deriv),
                MaxPool(2,device=DEVICE),
                DLL_OUTER_W_ConvLayer(12,32,64,3,self.args.learning_rate,tanh,tanh_deriv),
                MaxPool(2,device=DEVICE),
                DLL_OUTER_W_ConvLayer(5,64,16,3,self.args.learning_rate,tanh,tanh_deriv),
                DLL_OUTER_W_ProjectionLayer(16,3,3,200,self.args.learning_rate,tanh,tanh_deriv),
                DLL_FCLayer(200,self.args.num_class,self.args.learning_rate,softmax,linear_deriv)
            ]
        elif self.args.layer_type == "DLL_INNER_W_ConvLayer":
            if self.args.dataset == "mnist":
                self.layers = [
                    DLL_INNER_W_ConvLayer(input_size=self.args.input_size,num_channels=self.args.channel,num_filters=32,kernel_size=5,learning_rate=self.args.learning_rate,f=tanh,df=tanh_deriv,enable_batchnorm = False,enable_adam=self.args.enable_adam),
                    MaxPool(2,device=DEVICE),
                    DLL_INNER_W_ConvLayer(12,32,64,3,self.args.learning_rate,tanh,tanh_deriv,enable_batchnorm = False,enable_adam=self.args.enable_adam),
                    MaxPool(2,device=DEVICE),
                    DLL_INNER_W_ConvLayer(5,64,16,3,self.args.learning_rate,tanh,tanh_deriv,enable_batchnorm = False,enable_adam=self.args.enable_adam),
                    DLL_INNER_W_ProjectionLayer(16,3,3,200,self.args.learning_rate,tanh,tanh_deriv,enable_adam=self.args.enable_adam),
                    DLL_INNER_W_FCLayer(200,self.args.num_class,self.args.learning_rate,softmax,linear_deriv,enable_adam=self.args.enable_adam)
                ]
            elif self.args.dataset == "fashionmnist":
                self.layers = [
                    DLL_INNER_W_ConvLayer(input_size=self.args.input_size,num_channels=self.args.channel,num_filters=64,kernel_size=5,learning_rate=self.args.learning_rate,f=tanh,df=tanh_deriv,padding=0,enable_batchnorm = False,enable_adam=self.args.enable_adam),
                    MaxPool(2,device=DEVICE),
                    DLL_INNER_W_ConvLayer(12,64,128,3,self.args.learning_rate,tanh,tanh_deriv, 0,enable_batchnorm = False,enable_adam=self.args.enable_adam),
                    AvgPool(2,device=DEVICE),
                    DLL_INNER_W_ConvLayer(5,128,64,3,self.args.learning_rate,tanh,tanh_deriv, 0,enable_batchnorm = False,enable_adam=self.args.enable_adam),
                    DLL_INNER_W_ProjectionLayer(64,3,3,128,self.args.learning_rate,tanh,tanh_deriv,enable_adam=self.args.enable_adam),
                    DLL_INNER_W_FCLayer(128,self.args.num_class,self.args.learning_rate,softmax,linear_deriv,enable_adam=self.args.enable_adam)
                ]
            elif self.args.dataset == "cifar10":
                self.layers = [                    
                    DLL_INNER_W_ConvLayer(input_size=self.args.input_size,num_channels=self.args.channel,num_filters=64,kernel_size=3,learning_rate=self.args.learning_rate,f=tanh,df=tanh_deriv,padding=1,device=DEVICE, enable_batchnorm = False,enable_adam=self.args.enable_adam),
                    MaxPool(2,device=DEVICE),
                    DLL_INNER_W_ConvLayer(16,64,128,3,self.args.learning_rate,tanh,tanh_deriv, 1,device=DEVICE,enable_batchnorm = False,enable_adam=self.args.enable_adam),
                    MaxPool(2,device=DEVICE),
                    DLL_INNER_W_ConvLayer(8,128,64,3,self.args.learning_rate,tanh,tanh_deriv, 1,device=DEVICE,enable_batchnorm = False,enable_adam=self.args.enable_adam),
                    AvgPool(2,device=DEVICE),
                    DLL_INNER_W_ProjectionLayer(64,4,4,256,self.args.learning_rate,tanh,tanh_deriv,device=DEVICE,enable_adam=self.args.enable_adam),
                    DLL_INNER_W_FCLayer(256,self.args.num_class,self.args.learning_rate,softmax,linear_deriv,device=DEVICE,enable_adam=self.args.enable_adam)
                ]
            elif self.args.dataset == "cifar100":
                self.layers = [
                    DLL_INNER_W_ConvLayer(input_size=self.args.input_size, num_channels=self.args.channel, num_filters=64, kernel_size=3, learning_rate=self.args.learning_rate, f=tanh, df=tanh_deriv, padding=1, enable_batchnorm=False, enable_adam=self.args.enable_adam),
                    DLL_INNER_W_ConvLayer(32, 64, 64, 3, self.args.learning_rate, tanh, tanh_deriv, 1, enable_batchnorm=False, enable_adam=self.args.enable_adam),
                    MaxPool(2, device=DEVICE),

                    DLL_INNER_W_ConvLayer(16,64,128,3,self.args.learning_rate,tanh,tanh_deriv, 1,enable_batchnorm = False,enable_adam=self.args.enable_adam),
                    MaxPool(2,device=DEVICE),
                    DLL_INNER_W_ConvLayer(8,128,64,3,self.args.learning_rate,tanh,tanh_deriv, 1,enable_batchnorm = False,enable_adam=self.args.enable_adam),
                    AvgPool(2,device=DEVICE),
                    DLL_INNER_W_ProjectionLayer(64,4,4,1024,self.args.learning_rate,tanh,tanh_deriv,enable_adam=self.args.enable_adam),
                    DLL_INNER_W_FCLayer(1024,self.args.num_class,self.args.learning_rate,softmax,linear_deriv,enable_adam=self.args.enable_adam)
                ]

            elif self.args.dataset == "svhn":
                self.layers = [                    
                    DLL_INNER_W_ConvLayer(input_size=self.args.input_size,num_channels=self.args.channel,num_filters=64,kernel_size=3,learning_rate=self.args.learning_rate,f=tanh,df=tanh_deriv,padding=1,enable_batchnorm = False,enable_adam=self.args.enable_adam),
                    MaxPool(2,device=DEVICE),
                    DLL_INNER_W_ConvLayer(16,64,128,3,self.args.learning_rate,tanh,tanh_deriv, 1,enable_batchnorm = False,enable_adam=self.args.enable_adam),
                    MaxPool(2,device=DEVICE),
                    DLL_INNER_W_ConvLayer(8,128,64,3,self.args.learning_rate,tanh,tanh_deriv, 1,enable_batchnorm = False,enable_adam=self.args.enable_adam),
                    DLL_INNER_W_ProjectionLayer(64,8,8,256,self.args.learning_rate,tanh,tanh_deriv,enable_adam=self.args.enable_adam),
                    DLL_INNER_W_FCLayer(256,self.args.num_class,self.args.learning_rate,softmax,linear_deriv,enable_adam=self.args.enable_adam)
                ]
            elif self.args.dataset == "tinyimagenet":
                self.layers = [                    
                    DLL_INNER_W_ConvLayer(input_size=self.args.input_size, num_channels=self.args.channel, num_filters=64, kernel_size=3, learning_rate=self.args.learning_rate, f=tanh, df=tanh_deriv, padding=1, enable_batchnorm=False, enable_adam=self.args.enable_adam, enable_sgd=self.args.enable_sgd),
                    DLL_INNER_W_ConvLayer(64, 64, 64, 3, self.args.learning_rate, tanh, tanh_deriv, 1, enable_batchnorm=False, enable_adam=self.args.enable_adam, enable_sgd=self.args.enable_sgd),
                    MaxPool(2, device=DEVICE),

                    DLL_INNER_W_ConvLayer(32,64,128,3,self.args.learning_rate,tanh,tanh_deriv, 1,enable_batchnorm = False,enable_adam=self.args.enable_adam, enable_sgd=self.args.enable_sgd),
                    DLL_INNER_W_ConvLayer(32,128,128,3,self.args.learning_rate,tanh,tanh_deriv, 1,enable_batchnorm = False,enable_adam=self.args.enable_adam, enable_sgd=self.args.enable_sgd),
                    MaxPool(2,device=DEVICE),
                    DLL_INNER_W_ConvLayer(16,128,64,3,self.args.learning_rate,tanh,tanh_deriv, 1,enable_batchnorm = False,enable_adam=self.args.enable_adam, enable_sgd=self.args.enable_sgd),
                    AvgPool(2,device=DEVICE),
                    DLL_INNER_W_ProjectionLayer(64,8,8,1024,self.args.learning_rate,tanh,tanh_deriv,enable_adam=self.args.enable_adam, enable_sgd=self.args.enable_sgd),
                    DLL_INNER_W_FCLayer(1024,self.args.num_class,self.args.learning_rate,softmax,linear_deriv,enable_adam=self.args.enable_adam, enable_sgd=self.args.enable_sgd)
                ]
            elif self.args.dataset == "sst2":
                filters=[3,4,5]
                sentence_length=25
                label_num=2
                filter_num=100
                
                # 定义后续层
                self.layers = [
                    [
                        DLL_INNER_W_ConvLayer(sentence_length, 1, num_filters=filter_num, 
                            kernel_size=filter_size, learning_rate=self.args.learning_rate, 
                            f=tanh, df=tanh_deriv, padding=0, enable_batchnorm=False, 
                            enable_adam=self.args.enable_adam, input_size_c=300, kernel_size_c=300)
                        for filter_size in filters
                    ],
                    [
                        AvgPool((sentence_length - filter_size + 1, 1), device=DEVICE)
                        for filter_size in filters
                    ],
                    [
                        Tanh_layer()
                        for _ in filters
                    ],
                    DLL_INNER_W_ProjectionLayer(100,1,1,512,self.args.learning_rate,
                        tanh,tanh_deriv,enable_adam=self.args.enable_adam, concat_num=3),
                    DLL_INNER_W_FCLayer(512,128,self.args.learning_rate,
                        tanh,tanh_deriv,enable_adam=self.args.enable_adam),
                    DLL_INNER_W_FCLayer(128,label_num,self.args.learning_rate,
                        softmax,linear_deriv,enable_adam=self.args.enable_adam)
                ]
            elif self.args.dataset == "subj":
                filters=[int(i) for i in self.args.filters.split(',')]
                layers=[int(i) for i in self.args.layers.split(',')] + [self.args.num_class]
                filter_num=self.args.filter_num
                
                # 定义后续层
                self.layers = [
                    [
                        DLL_INNER_W_ConvLayer(self.args.input_size, self.args.channel, num_filters=filter_num, 
                            kernel_size=filter_size, learning_rate=self.args.learning_rate, 
                            f=tanh, df=tanh_deriv, padding=0, enable_batchnorm=False, 
                            enable_adam=self.args.enable_adam, input_size_c=self.args.input_size_c, kernel_size_c=self.args.input_size_c)
                        for filter_size in filters
                    ],
                    [
                        AvgPool((self.args.input_size - filter_size + 1, 1), device=DEVICE)
                        for filter_size in filters
                    ],
                    [
                        Tanh_layer()
                        for _ in filters
                    ],
                    DLL_INNER_W_ProjectionLayer(filter_num,1,1,layers[0],self.args.learning_rate,
                        tanh,tanh_deriv,enable_adam=self.args.enable_adam, concat_num=len(filters)),
                ] + \
                [
                    DLL_INNER_W_FCLayer(layers[i],layers[i+1],self.args.learning_rate,
                        tanh,tanh_deriv,enable_adam=self.args.enable_adam) for i in range(len(layers[:-1]))
                ]
        
    def forward(self, inputs):
        self.x[0] = inputs.clone()
        self.u[0] = inputs.clone()
        for i, l in enumerate(self.layers):
            if isinstance(l, list):
                self.x[i+1] = []
                for j, part in enumerate(l):
                    if isinstance(self.x[i], list):
                        self.x[i+1].append(part.forward(self.x[i][j]))
                    else:
                        self.x[i+1].append(part.forward(self.x[i]))
                self.u[i+1] = [t.clone() for t in self.x[i+1]]
            else:
                self.x[i+1] = l.forward(self.x[i])
                self.u[i+1] = self.x[i+1].clone()
        return self.x[-1]
        
    def iterate(self, target):
        self.x[-1] = target.clone()
        self.errors[-1] = -cross_entropy_deriv(out=self.u[-1], label=self.x[-1]) # B D

        self.errors[0] = (self.x[0] - self.u[0]) / self.args.sigma
        for i in reversed(range(1, len(self.layers))):
            if isinstance(self.layers[i], list):
                self.errors[i] = [part.backward(self.errors[i+1][j], sigma=self.args.sigma) for j, part in enumerate(self.layers[i])]
            else:
                self.errors[i] = self.layers[i].backward(self.errors[i+1], sigma=self.args.sigma)
        
        # for i in reversed(range(len(self.layers))):
        #     if i != 0:
        #         self.errors[i] = self.layers[i].backward(self.errors[i+1], sigma=self.args.sigma)
        # # for iter in range(self.max_iter):
        # #     for i in reversed(range(len(self.layers))):
        # #         self.errors[i] = (self.x[i] - self.u[i]) / self.args.sigma
        #         if i != 0:
        #             e = self.layers[i].backward(self.errors[i+1], sigma=self.args.sigma)
        #             # self.x[i] -= 0.2 * (self.errors[i] - e)
        #             self.x[i] -= self.errors[i] - e
        #     # print(f"iter {iter }, torch.mean(self.errors[-2]): {torch.mean(self.errors[-2])}")
        return
    
    def update_weights(self,epoch):
        epoch_limit = 0
        for (i, l) in enumerate(self.layers):
            l.update_weights(self.errors[i+1], sigma=self.args.sigma)
            if "Dendritic" in str(type(l).__name__) or "DLL" in str(type(l).__name__):
                if epoch >= epoch_limit: 
                    # self.errors[i].shape = l.input.shape, self.errors[i+1].shape = l.output.shape
                    l.update_theta(self.errors[i], self.errors[i+1], sigma=self.args.sigma)
                    continue
        return
    
    def print_layers(self, print=print):
        for i, layer in enumerate(self.layers):
            print(f"Layer {i}: {layer}")
        return


class BP_RNN_Model(object):
    def __init__(self, args, device='cuda') -> None:
        self.args = args
        self.device = device
        self.seq_len = args.seq_len
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.fn = args.fn
        self.fn_deriv = args.fn_deriv
        self.weight_learning_rate = args.weight_learning_rate
        self.clamp_val = 50
        #weights
        self.Wh = torch.empty([self.hidden_size, self.hidden_size]).normal_(mean=0.0,std=0.05).to(self.device)
        self.Wx = torch.empty([self.hidden_size, self.input_size]).normal_(mean=0.0,std=0.05).to(self.device)
        self.Wy = torch.empty([self.output_size, self.hidden_size]).normal_(mean=0.0,std=0.05).to(self.device)
        self.h0 = torch.empty([self.hidden_size, self.batch_size]).normal_(mean=0.0,std=0.05).to(self.device)

        self.hu = torch.empty([self.seq_len+1, self.hidden_size, self.batch_size]).to(self.device)
        self.y = torch.empty([self.seq_len, self.output_size, self.batch_size]).to(self.device)

    def forward(self, inputs_seq):
        # L V B
        self.inputs = inputs_seq.clone()
        self.hu[0] = self.h0
        for i, inp in enumerate(inputs_seq):
            self.hu[i+1] = self.fn(self.Wh @ self.hu[i] + self.Wx @ inp)
            self.y[i] = linear(self.Wy @ self.hu[i+1])
        return self.y
    
    def update_weights(self, target_seq, epoch_num):
        dhs = torch.zeros_like(self.hu).to(self.device)
        dys = torch.zeros_like(self.y).to(self.device)
        dWy = torch.zeros_like(self.Wy).to(self.device)
        dWx = torch.zeros_like(self.Wx).to(self.device)
        dWh = torch.zeros_like(self.Wh).to(self.device)

        for i, tar in reversed(list(enumerate(target_seq))):
            dys[i] = tar - self.y[i]
            dh = self.Wy.T @ (dys[i] * linear_deriv(self.Wy @ self.hu[i+1]))
            if i < len(target_seq) - 1:
                fn_deriv =  self.fn_deriv(self.Wh @ self.hu[i+1] + self.Wx @ self.inputs[i+1])
                dh += self.Wh.T @ (dhs[i+1] * fn_deriv)
            dhs[i]= dh
        for i,inp in reversed(list(enumerate(self.inputs))):
            fn_deriv = self.fn_deriv(self.Wh @ self.hu[i] + self.Wx @ inp)
            dWy += (dys[i] * linear_deriv(self.Wy @ self.hu[i+1])) @ self.hu[i+1].T
            dWx += (dhs[i] * fn_deriv) @ inp.T
            dWh += (dhs[i] * fn_deriv) @ self.hu[i].T
        self.Wy += self.weight_learning_rate * torch.clamp(dWy, -self.clamp_val, self.clamp_val)
        self.Wx += self.weight_learning_rate * torch.clamp(dWx, -self.clamp_val, self.clamp_val)
        self.Wh += self.weight_learning_rate * torch.clamp(dWh, -self.clamp_val, self.clamp_val)
        
class PC_RNN_Model(object):
    def __init__(self, args, device='cuda') -> None:
        self.args = args
        self.device = device
        self.seq_len = args.seq_len
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.fn = args.fn
        self.fn_deriv = args.fn_deriv
        self.weight_learning_rate = args.weight_learning_rate
        self.inference_learning_rate = args.inference_learning_rate
        self.inference_steps = args.inference_steps
        self.fixed_predictions = args.fixed_predictions
        self.clamp_val = 50
        self.accelerate = args.accelerate
        self.fa = args.fa
        #weights
        self.Wh = torch.empty([self.hidden_size, self.hidden_size]).normal_(mean=0.0,std=0.05).to(self.device)
        self.Wh_fixed = torch.empty([self.hidden_size, self.hidden_size]).normal_(mean=0.0,std=0.05).to(self.device)
        self.Wx = torch.empty([self.hidden_size, self.input_size]).normal_(mean=0.0,std=0.05).to(self.device)
        self.Wy = torch.empty([self.output_size, self.hidden_size]).normal_(mean=0.0,std=0.05).to(self.device)
        self.Wy_fixed = torch.empty([self.output_size, self.hidden_size]).normal_(mean=0.0,std=0.05).to(self.device)
        self.h0 = torch.empty([self.hidden_size, self.batch_size]).normal_(mean=0.0,std=0.05).to(self.device)

        self.hu = torch.empty([self.seq_len+1, self.hidden_size, self.batch_size]).to(self.device)
        self.hx = torch.zeros_like(self.hu)
        self.y = torch.empty([self.seq_len, self.output_size, self.batch_size]).to(self.device)

    def forward(self, inputs_seq):
        # L V B
        self.inputs = inputs_seq.clone()
        self.hu[0] = self.h0.clone()
        self.hx[0] = self.h0.clone()
        for i, inp in enumerate(inputs_seq):
            self.hu[i+1] = self.fn(self.Wh @ self.hu[i] + self.Wx @ inp)
            self.hx[i+1] = self.hu[i+1].clone()
            self.y[i] = linear(self.Wy @ self.hu[i+1])
        return self.y
    
    def update_weights(self, target_seq, epoch_num):
        with torch.no_grad():
            Wh = self.Wh if not self.fa else self.Wh_fixed
            Wy = self.Wy if not self.fa else self.Wy_fixed
            # the last dim of ehs is not used, because we don't need hx[0] - hu[0]
            ehs = torch.zeros_like(self.hu).to(self.device)
            eys = torch.zeros_like(self.y).to(self.device)
            for i, tar in reversed(list(enumerate(target_seq))):
                # for n in range(self.inference_steps):
                # for hx,hu,deltah,ehs, i+1 means time t
                # for eys,y, i means time t
                if self.accelerate and self.inference_steps >= 10: # ipc
                    eys[i] = tar - self.y[i]
                    deltah = Wy.T @ (eys[i] * linear_deriv(self.Wy @ self.hu[i+1]))
                    if i < len(target_seq)-1:
                        fn_deriv =  self.fn_deriv(self.Wh @ self.hu[i+1] + self.Wx @ self.inputs[i]) # current layer
                        deltah += Wh.T @ (ehs[i+1] * fn_deriv)
                    ehs[i] = deltah
                else:
                    for n in range(self.inference_steps):
                        # for hx,hu,deltah,ehs, i+1 means time t
                        # for eys,y, i means time t
                        eys[i] = tar - self.y[i]
                        if self.fixed_predictions == False:
                            self.hu[i+1] = self.fn(self.Wh @ self.hx[i] + self.Wx @ self.inputs[i])
                        ehs[i] = self.hx[i+1] - self.hu[i+1] # x-u
                        deltah = -ehs[i].clone()
                        deltah += Wy.T @ (eys[i] * linear_deriv(self.Wy @ self.hu[i+1]))
                        if i < len(target_seq)-1:
                            fn_deriv =  self.fn_deriv(self.Wh @ self.hu[i+1] + self.Wx @ self.inputs[i]) # current layer
                            deltah += Wh.T @ (ehs[i+1] * fn_deriv)
                        self.hx[i+1] += self.inference_learning_rate * deltah
                        if self.fixed_predictions == False:
                            self.y[i] = linear(self.Wy @ self.hx[i+1])
                        # print(f"iter {n} {self.hx[i+1][0]}")
                    
            dWy = torch.zeros_like(self.Wy).to(self.device)
            dWx = torch.zeros_like(self.Wx).to(self.device)
            dWh = torch.zeros_like(self.Wh).to(self.device)
            for i,inp in reversed(list(enumerate(self.inputs))):
                fn_deriv = self.fn_deriv(self.Wh @ self.hu[i] + self.Wx @ inp)
                dWy += (eys[i] * linear_deriv(self.Wy @ self.hu[i+1])) @ self.hu[i+1].T
                dWx += (ehs[i] * fn_deriv) @ inp.T
                dWh += (ehs[i] * fn_deriv) @ self.hu[i].T
            self.Wy += self.weight_learning_rate * torch.clamp(dWy, -self.clamp_val, self.clamp_val)
            self.Wx += self.weight_learning_rate * torch.clamp(dWx, -self.clamp_val, self.clamp_val)
            self.Wh += self.weight_learning_rate * torch.clamp(dWh, -self.clamp_val, self.clamp_val)

class DLL_RNN_Model(object):
    def __init__(self, args, device='cuda') -> None:
        self.args = args
        self.device = device
        self.seq_len = args.seq_len
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.fn = args.fn
        self.fn_deriv = args.fn_deriv
        self.weight_learning_rate = args.weight_learning_rate
        self.clamp_val = 50
        self.theta_update_discount = args.theta_update_discount
        self.noclamp = args.noclamp
        self.fix_theta_until = args.fix_theta_until
        # self.ecoefficient = args.ecoefficient
        self.noise = args.noise
        #weights
        self.Wh = torch.empty([self.hidden_size, self.hidden_size]).normal_(mean=0.0,std=0.05).to(self.device)
        self.Wx = torch.empty([self.hidden_size, self.input_size]).normal_(mean=0.0,std=0.05).to(self.device)
        self.Wy = torch.empty([self.output_size, self.hidden_size]).normal_(mean=0.0,std=0.05).to(self.device)
        self.h0 = torch.empty([self.hidden_size, self.batch_size]).normal_(mean=0.0,std=0.05).to(self.device)

        self.hu = torch.empty([self.seq_len+1, self.hidden_size, self.batch_size]).to(self.device)
        self.hx = torch.zeros_like(self.hu).to(self.device)
        self.y = torch.empty([self.seq_len, self.output_size, self.batch_size]).to(self.device)

        self.theta_h = torch.zeros_like(self.Wh).normal_(mean=0.0,std=0.05).to(self.device)
        self.theta_y = torch.zeros_like(self.Wy).normal_(mean=0.0,std=0.05).to(self.device)

    def forward(self, inputs_seq):
        with torch.no_grad():
            # L V B
            self.inputs = inputs_seq.clone()
            self.hu[0] = self.h0.clone()
            self.hx[0] = self.h0.clone()
            for i, inp in enumerate(inputs_seq):
                self.hu[i+1] = self.fn(self.Wh @ self.hu[i] + self.Wx @ inp)
                self.hx[i+1] = self.hu[i+1].clone()
                self.y[i] = linear(self.Wy @ self.hu[i+1])
            return self.y
    
    def update_weights(self, target_seq, epoch_num):
        with torch.no_grad():
            # the last dim of ehs is not used, because we don't need hx[0] - hu[0]
            ehs = torch.zeros_like(self.hu).to(self.device)
            eys = torch.zeros_like(self.y).to(self.device)
            for i, tar in reversed(list(enumerate(target_seq))):
                eys[i] = tar - self.y[i]
                deltah = self.theta_y.T @ (eys[i] * linear_deriv(self.Wy @ self.hu[i+1]))
                if i < len(target_seq)-1:
                    fn_deriv =  self.fn_deriv(self.Wh @ self.hu[i+1] + self.Wx @ self.inputs[i]) # current layer
                    deltah += self.theta_h.T @ (ehs[i+1] * fn_deriv)
                ehs[i] = deltah
            if self.noise: ehs = ehs * (1 + 2 * self.noise * (torch.rand_like(ehs) - 0.5))
            dWy = torch.zeros_like(self.Wy).to(self.device)
            dWx = torch.zeros_like(self.Wx).to(self.device)
            dWh = torch.zeros_like(self.Wh).to(self.device)
            dtheta_y_T = torch.zeros_like(self.Wy.T).to(self.device)
            dtheta_h_T = torch.zeros_like(self.Wh.T).to(self.device)
            for i,inp in reversed(list(enumerate(self.inputs))):
                fn_deriv = self.fn_deriv(self.Wh @ self.hu[i] + self.Wx @ inp)
                dWy += (eys[i] * linear_deriv(self.Wy @ self.hu[i+1])) @ self.hu[i+1].T
                dWx += (ehs[i] * fn_deriv) @ inp.T
                dWh += (ehs[i] * fn_deriv) @ self.hu[i].T
                dtheta_y_T -= ehs[i] @ (eys[i] * linear_deriv(self.Wy @ self.hu[i+1])).T
                if i>=1: dtheta_h_T -= ehs[i-1] @ (ehs[i] * fn_deriv).T
            if not self.noclamp:
                dWy = torch.clamp(dWy, -self.clamp_val, self.clamp_val)
                dWx = torch.clamp(dWx, -self.clamp_val, self.clamp_val)
                dWh = torch.clamp(dWh, -self.clamp_val, self.clamp_val)
                dtheta_y_T = torch.clamp(dtheta_y_T, -self.clamp_val, self.clamp_val)
                dtheta_h_T = torch.clamp(dtheta_h_T, -self.clamp_val, self.clamp_val)
            self.Wy += self.weight_learning_rate * dWy
            self.Wx += self.weight_learning_rate * dWx
            self.Wh += self.weight_learning_rate * dWh
            if epoch_num >= self.fix_theta_until:
                self.theta_y += self.weight_learning_rate * dtheta_y_T.T / self.theta_update_discount
                self.theta_h += self.weight_learning_rate * dtheta_h_T.T / self.theta_update_discount