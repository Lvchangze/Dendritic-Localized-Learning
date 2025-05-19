import torch
import torchvision.transforms as transforms
from utils import *

class ConvLayer(object):
    def __init__(self,input_size,num_channels,num_filters,kernel_size,learning_rate,f=tanh,df=tanh_deriv,padding=0,stride=1,device="cuda",enable_batchnorm = True,enable_adam = True, input_size_c=-1, kernel_size_c=-1):
        self.input_size = input_size
        self.input_size_r = input_size
        self.input_size_c = input_size_c if input_size_c !=-1 else input_size
        self.num_channels = num_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.kernel_size_r = kernel_size
        self.kernel_size_c = kernel_size_c if kernel_size_c !=-1 else kernel_size
        self.padding = padding
        self.stride = stride
        self.output_size_r = math.floor((self.input_size_r + (2 * self.padding) - self.kernel_size_r)/self.stride) + 1
        self.output_size_c = math.floor((self.input_size_c + (2 * self.padding) - self.kernel_size_c)/self.stride) + 1
        self.learning_rate = learning_rate
        self.f = f
        self.df = df
        self.device = device
        self.kernel= torch.empty(self.num_filters,self.num_channels,self.kernel_size_r,self.kernel_size_c).normal_(mean=0,std=0.05).to(self.device)
        self.unfold = nn.Unfold(kernel_size=(self.kernel_size_r,self.kernel_size_c),padding=self.padding,stride=self.stride).to(self.device)
        self.fold = nn.Fold(output_size=(self.input_size_r,self.input_size_c),kernel_size=(self.kernel_size_r,self.kernel_size_c),padding=self.padding,stride=self.stride).to(self.device)
        self.enable_batchnorm = enable_batchnorm
        self.batchnorm = BatchNorm2d
        self.enable_adam = enable_adam

        # Adam parameters for kernel
        self.beta1_kernel = 0.9
        self.beta2_kernel = 0.999
        self.eps_kernel = 1e-8
        self.m_kernel = torch.zeros_like(self.kernel)
        self.v_kernel = torch.zeros_like(self.kernel)
        self.t_kernel = 0  # Time step for kernel

    def forward(self,inp):
        batch_size = inp.shape[0] # B, C, H, W
        self.X_col = self.unfold(inp.clone())
        self.flat_weights = self.kernel.reshape(self.num_filters,-1)
        out = self.flat_weights @ self.X_col
        self.activations = out.reshape(batch_size, self.num_filters, self.output_size_r, self.output_size_c)
        if self.enable_batchnorm:
            return self.batchnorm(self.f(self.activations))
        else:
            return self.f(self.activations)

    def update_weights(self,e,sigma=1.0):
        batch_size = e.shape[0]
        fn_deriv = self.df(self.activations)
        e = e * fn_deriv
        self.dout = e.reshape(batch_size,self.num_filters,-1)
        dW = self.dout @ self.X_col.permute(0,2,1)
        dW = torch.sum(dW,dim=0)
        dW = dW.reshape((self.num_filters,self.num_channels,self.kernel_size_r,self.kernel_size_c))
        dW = torch.clamp(dW * 2/torch.pow(torch.tensor(sigma), torch.tensor(2.0)),-50,50)
        if self.enable_adam:
            self.t_kernel += 1
            self.m_kernel = self.beta1_kernel * self.m_kernel + (1 - self.beta1_kernel) * dW
            self.v_kernel = self.beta2_kernel * self.v_kernel + (1 - self.beta2_kernel) * (dW ** 2)
            m_hat_kernel = self.m_kernel / (1 - self.beta1_kernel ** self.t_kernel)
            v_hat_kernel = self.v_kernel / (1 - self.beta2_kernel ** self.t_kernel)
            adjusted_lr_kernel = self.learning_rate * m_hat_kernel / (torch.sqrt(v_hat_kernel) + self.eps_kernel)
            self.kernel += adjusted_lr_kernel
        else:
            # self.kernel -= self.learning_rate * dW
            self.kernel += self.learning_rate * dW
        return dW

    def backward(self,e,sigma=1.0):
        batch_size = e.shape[0]
        fn_deriv = self.df(self.activations)
        e = e * fn_deriv
        self.dout = e.reshape(batch_size,self.num_filters,-1)
        dX_col = self.flat_weights.T @ self.dout
        dX = self.fold(dX_col)
        dX = torch.clamp(dX/torch.pow(torch.tensor(sigma), torch.tensor(2.0)),-50,50)
        return dX
    
    def __repr__(self):
        return f"{self.__class__.__name__}(input_size={self.input_size_r},{self.input_size_c}, num_channels={self.num_channels}, num_filters={self.num_filters}, kernel_size={self.kernel_size_r},{self.kernel_size_c}, learning_rate={self.learning_rate}, f={self.f.__name__}, df={self.df.__name__}, padding={self.padding}, stride={self.stride}, device={self.device}, enable_batchnorm={self.enable_batchnorm}, enable_adam={self.enable_adam})"
    

class DLL_OUTER_W_ConvLayer(object):
    def __init__(self,input_size,num_channels,num_filters,kernel_size,learning_rate,f=tanh,df=tanh_deriv,padding=0,stride=1,device="cuda",enable_batchnorm=True):
        self.input_size = input_size
        self.num_channels = num_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.output_size = math.floor((self.input_size + (2 * self.padding) - self.kernel_size)/self.stride) +1
        self.learning_rate = learning_rate
        self.f = f
        self.df = df
        self.device = device
        self.kernel= torch.empty(self.num_filters,self.num_channels,self.kernel_size,self.kernel_size).normal_(mean=0,std=0.05).to(self.device)
        self.unfold = nn.Unfold(kernel_size=(self.kernel_size,self.kernel_size),padding=self.padding,stride=self.stride).to(self.device)
        self.fold = nn.Fold(output_size=(self.input_size,self.input_size),kernel_size=(self.kernel_size,self.kernel_size),padding=self.padding,stride=self.stride).to(self.device)
        self.theta = torch.empty(self.num_filters,self.num_channels,self.kernel_size,self.kernel_size).normal_(mean=0,std=0.05).to(self.device)
        # self.theta = generate_ones_and_minus_ones_matrix(self.input_size, self.output_size).to(self.device)
        self.enable_batchnorm = enable_batchnorm
        self.batchnorm = BatchNorm2d

    def forward(self,inp):
        batch_size = inp.shape[0]
        self.X_col = self.unfold(inp.clone())
        self.activations = self.X_col.clone()
        self.flat_weights = self.kernel.reshape(self.num_filters,-1)
        out = torch.matmul(self.flat_weights, self.f(self.X_col))
        if self.enable_batchnorm:
            return self.batchnorm(out.reshape(batch_size, self.num_filters, self.output_size, self.output_size))
        else:
            return out.reshape(batch_size, self.num_filters, self.output_size, self.output_size)

    def update_weights(self,e,sigma=1.0):
        # B = e.shape[0]
        fn = self.f(self.activations).reshape(self.X_col.shape[1], -1)
        e_col = e.reshape(self.num_filters,-1)
        dW = torch.matmul(e_col, fn.T)
        dW = torch.clamp(dW * 2/torch.pow(torch.tensor(sigma), torch.tensor(2.0)),-50,50)
        # self.kernel -= self.learning_rate * torch.clamp(dW * 2,-50,50)
        self.kernel += self.learning_rate * dW.reshape(self.kernel.shape)
        return dW

    def backward(self,e,sigma=1.0):
        B = e.shape[0]
        fn_deriv = self.df(self.activations)
        e_col = e.reshape(B, self.num_filters,-1)
        # # theta
        flat_theta = self.theta.reshape(self.num_filters, -1)
        dX_col = torch.matmul(flat_theta.T, e_col) * fn_deriv

        # outer W DLL-CNN
        # dX_col = torch.matmul(self.flat_weights.T, e_col) * fn_deriv

        dX = self.fold(dX_col)
        dX = torch.clamp(dX/torch.pow(torch.tensor(sigma), torch.tensor(2.0)),-50,50)
        return dX

    def update_theta(self, e1, e2, sigma=1.0):
        dx_col = self.fold(self.df(self.activations))
        flat_e2 = e2.reshape(-1,self.num_filters)
        dtheta = torch.matmul(self.unfold(dx_col * e1).reshape(self.num_channels*self.kernel_size*self.kernel_size,-1), flat_e2)
        dtheta = dtheta.reshape(self.theta.shape)
        self.theta += self.learning_rate / 10 * torch.clamp(dtheta*2/torch.pow(torch.tensor(sigma), torch.tensor(2.0)),-50,50)
        return
    
    def __repr__(self):
        return f"{self.__class__.__name__}(input_size={self.input_size}, num_channels={self.num_channels}, num_filters={self.num_filters}, kernel_size={self.kernel_size}, learning_rate={self.learning_rate}, f={self.f.__name__}, df={self.df.__name__}, padding={self.padding}, stride={self.stride}, device={self.device}, enable_batchnorm={self.enable_batchnorm})"
    
class DLL_INNER_W_ConvLayer(object):
    def __init__(self,input_size,num_channels,num_filters,kernel_size,learning_rate,f=tanh,df=tanh_deriv,padding=0,stride=1,device="cuda",enable_batchnorm = True,enable_adam = True, enable_sgd=False, input_size_c=-1, kernel_size_c=-1):
        self.input_size = input_size
        self.input_size_r = input_size
        self.input_size_c = input_size_c if input_size_c !=-1 else input_size
        self.num_channels = num_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.kernel_size_r = kernel_size
        self.kernel_size_c = kernel_size_c if kernel_size_c !=-1 else kernel_size
        self.padding = padding
        self.stride = stride
        self.output_size_r = math.floor((self.input_size_r + (2 * self.padding) - self.kernel_size_r)/self.stride) +1
        self.output_size_c = math.floor((self.input_size_c + (2 * self.padding) - self.kernel_size_c)/self.stride) +1
        self.learning_rate = learning_rate
        self.f = f
        self.df = df
        self.device = device
        self.kernel= torch.empty(self.num_filters,self.num_channels,self.kernel_size_r,self.kernel_size_c).normal_(mean=0,std=0.05).to(self.device)
        self.unfold = nn.Unfold(kernel_size=(self.kernel_size_r,self.kernel_size_c),padding=self.padding,stride=self.stride).to(self.device)
        self.fold = nn.Fold(output_size=(self.input_size_r,self.input_size_c),kernel_size=(self.kernel_size_r,self.kernel_size_c),padding=self.padding,stride=self.stride).to(self.device)
        self.theta = torch.empty(self.num_filters,self.num_channels,self.kernel_size_r,self.kernel_size_c).normal_(mean=0,std=0.05).to(self.device)
        # self.theta = generate_ones_and_minus_ones_matrix(self.input_size, self.output_size).to(self.device)
        self.batchnorm = BatchNorm2d
        self.enable_batchnorm = enable_batchnorm
        self.enable_adam = enable_adam

        # Adam parameters for kernel
        self.beta1_kernel = 0.9
        self.beta2_kernel = 0.999
        self.eps_kernel = 1e-8
        self.m_kernel = torch.zeros_like(self.kernel)
        self.v_kernel = torch.zeros_like(self.kernel)
        self.t_kernel = 0  # Time step for kernel
        
        # Adam parameters for theta
        self.beta1_theta = 0.9
        self.beta2_theta = 0.999
        self.eps_theta = 1e-8
        self.m_theta = torch.zeros_like(self.theta)
        self.v_theta = torch.zeros_like(self.theta)
        self.t_theta = 0  # Time step for theta

        self.enable_sgd = enable_sgd
        self.momentum = 0.9
        self.weight_decay = 5e-4
        if self.enable_sgd and self.momentum > 0:
            self.velocity_kernel = torch.zeros_like(self.kernel)
            self.velocity_theta = torch.zeros_like(self.theta)

    def forward(self,inp):
        batch_size = inp.shape[0] # B, C, H, W
        self.X_col = self.unfold(inp.clone())
        self.flat_weights = self.kernel.reshape(self.num_filters,-1)
        out = self.flat_weights @ self.X_col
        self.activations = out.reshape(batch_size, self.num_filters, self.output_size_r, self.output_size_c)
        if self.enable_batchnorm:
            return self.batchnorm(self.f(self.activations))
        else:
            return self.f(self.activations)

    def update_weights(self,e,sigma=1.0):
        batch_size = e.shape[0]
        fn_deriv = self.df(self.activations)
        e = e * fn_deriv
        self.dout = e.reshape(batch_size,self.num_filters,-1)
        dW = self.dout @ self.X_col.permute(0,2,1)
        dW = torch.sum(dW,dim=0)
        dW = dW.reshape((self.num_filters,self.num_channels,self.kernel_size_r,self.kernel_size_c))
        dW = torch.clamp(dW * 2/torch.pow(torch.tensor(sigma), torch.tensor(2.0)),-50,50)
                
        if self.enable_adam:
            self.t_kernel += 1
            self.m_kernel = self.beta1_kernel * self.m_kernel + (1 - self.beta1_kernel) * dW
            self.v_kernel = self.beta2_kernel * self.v_kernel + (1 - self.beta2_kernel) * (dW ** 2)
            m_hat_kernel = self.m_kernel / (1 - self.beta1_kernel ** self.t_kernel)
            v_hat_kernel = self.v_kernel / (1 - self.beta2_kernel ** self.t_kernel)
            adjusted_lr_kernel = self.learning_rate * m_hat_kernel / (torch.sqrt(v_hat_kernel) + self.eps_kernel)
            self.kernel += adjusted_lr_kernel
        elif self.enable_sgd:
            dW += self.weight_decay * self.kernel
            if self.momentum > 0:
                self.velocity_kernel = self.momentum * self.velocity_kernel - self.learning_rate * dW
                self.kernel += self.velocity_kernel
            else:
                self.kernel -= self.learning_rate * dW
        else:
            self.kernel += self.learning_rate * dW
        return dW

    def backward(self,e,sigma=1.0):
        batch_size = e.shape[0]
        fn_deriv = self.df(self.activations)
        e = e * fn_deriv
        self.dout = e.reshape(batch_size,self.num_filters,-1)
        self.flat_theta = self.theta.reshape(self.num_filters,-1)
        dX_col = self.flat_theta.T @ self.dout
        dX = self.fold(dX_col)
        dX = torch.clamp(dX/torch.pow(torch.tensor(sigma), torch.tensor(2.0)),-50,50)
        return dX

    def update_theta(self, e1, e2, sigma=1.0):
        fn_deriv = self.df(self.activations)
        dtheta = torch.matmul(self.unfold(e1).reshape(self.num_channels * self.kernel_size * self.kernel_size, -1),
                              (e2 * fn_deriv).reshape(-1, self.num_filters))
        dtheta = dtheta.reshape(self.theta.shape)
        dtheta = torch.clamp(dtheta * 2 / (sigma ** 2), -50, 50)

        if self.enable_adam:
            self.t_theta += 1
            self.m_theta = self.beta1_theta * self.m_theta + (1 - self.beta1_theta) * dtheta
            self.v_theta = self.beta2_theta * self.v_theta + (1 - self.beta2_theta) * (dtheta ** 2)
            m_hat_theta = self.m_theta / (1 - self.beta1_theta ** self.t_theta)
            v_hat_theta = self.v_theta / (1 - self.beta2_theta ** self.t_theta)
            adjusted_lr_theta = self.learning_rate / 10 * m_hat_theta / (torch.sqrt(v_hat_theta) + self.eps_theta)
            self.theta += adjusted_lr_theta
        elif self.enable_sgd:
            dtheta += self.weight_decay * self.theta
            if self.momentum > 0:
                self.velocity_theta = self.momentum * self.velocity_theta - self.learning_rate / 10 * dtheta
                self.theta += self.velocity_theta
            else:
                self.theta -= self.learning_rate / 10 * dtheta
        else:
            self.theta += self.learning_rate / 10 * dtheta
        return

    def __repr__(self):
        return f"{self.__class__.__name__}(input_size={self.input_size}, num_channels={self.num_channels}, num_filters={self.num_filters}, kernel_size={self.kernel_size}, learning_rate={self.learning_rate}, f={self.f.__name__}, df={self.df.__name__}, padding={self.padding}, stride={self.stride}, device={self.device}, enable_batchnorm={self.enable_batchnorm}, enable_adam={self.enable_adam}, enable_sgd={self.enable_sgd})"




class MaxPool(object):
    def __init__(self, kernel_size,device='cpu'):
        self.kernel_size = kernel_size
        self.device = device
        self.activations = torch.empty(1)

    def forward(self,x):
        out, self.idxs = F.max_pool2d(x, self.kernel_size,return_indices=True)
        return out

    def backward(self, y,sigma=1.0):
        return F.max_unpool2d(y,self.idxs, self.kernel_size)

    def update_weights(self,e,sigma=1.0):
        return 0
    
    def __repr__(self):
        return f"MaxPool(kernel_size={self.kernel_size}, device={self.device})"

class AvgPool(object):
    def __init__(self, kernel_size, device='cpu'):
        self.kernel_size = kernel_size
        self.device = device
        self.activations = torch.empty(1)

    def forward(self, x):
        self.B_in,self.C_in,self.H_in,self.W_in = x.shape
        return F.avg_pool2d(x,self.kernel_size)

    def backward(self, y,sigma=1.0):
        return  F.interpolate(y, scale_factor=self.kernel_size, mode='nearest')

    def update_weights(self,e,sigma=1.0):
        return 0
    
    def __repr__(self):
        return f"AvgPool(kernel_size={self.kernel_size}, device={self.device})"

class Tanh_layer(object):
    def __init__(self, device='cpu'):
        pass

    def forward(self, x):
        return tanh(x)

    def backward(self, y,sigma=1.0):
        return  tanh_deriv(y)

    def update_weights(self,e,sigma=1.0):
        return 0
    
    def __repr__(self):
        return f"Tanh_layer"