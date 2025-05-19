import torch
import torchvision.transforms as transforms
from utils import *

class FCLayer(object):
    def __init__(self, input_size, output_size, learning_rate=5e-4, f=sigmoid, df=sigmoid_deriv, device="cuda", enable_adam=True):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.f = f
        self.df = df
        self.device = device
        self.weights = torch.empty([self.input_size, self.output_size]).normal_(mean=0.0,std=0.05).to(self.device)

        self.enable_adam = enable_adam

        # Adam parameters for weights
        self.beta1_weights = 0.9
        self.beta2_weights = 0.999
        self.eps_weights = 1e-8
        self.m_weights = torch.zeros_like(self.weights)
        self.v_weights = torch.zeros_like(self.weights)
        self.t_weights = 0  # Time step for weights

    def forward(self, x):
        self.inp = x.clone()
        self.activations = torch.matmul(self.inp, self.weights)
        out = self.f(self.activations)
        return out

    def backward(self, e, sigma=1.0):
        self.fn_deriv = self.df(self.activations)
        out = torch.matmul(e * self.fn_deriv, self.weights.T)
        out = torch.clamp(out/torch.pow(torch.tensor(sigma), torch.tensor(2.0)),-50,50)
        return out

    def update_weights(self, e, sigma=1.0):
        self.fn_deriv = self.df(self.activations)
        dw = torch.matmul(self.inp.T, e * self.fn_deriv)
        dw = torch.clamp(dw * 2/torch.pow(torch.tensor(sigma), torch.tensor(2.0)),-50,50)
        if self.enable_adam:
            self.t_weights += 1
            self.m_weights = self.beta1_weights * self.m_weights + (1 - self.beta1_weights) * dw
            self.v_weights = self.beta2_weights * self.v_weights + (1 - self.beta2_weights) * (dw ** 2)
            m_hat_weights = self.m_weights / (1 - self.beta1_weights ** self.t_weights)
            v_hat_weights = self.v_weights / (1 - self.beta2_weights ** self.t_weights)
            adjusted_lr_weights = self.learning_rate * m_hat_weights / (torch.sqrt(v_hat_weights) + self.eps_weights)
            self.weights += adjusted_lr_weights
        else:
            self.weights += self.learning_rate * dw
        return
    
    def __repr__(self):
        return f"{self.__class__.__name__}(input_size={self.input_size}, output_size={self.output_size}, learning_rate={self.learning_rate}, f={self.f.__name__}, df={self.df.__name__}, device={self.device}, enable_adam={self.enable_adam})"
    
class ProjectionLayer(object):
    def __init__(self, C, H, W, output_size, learning_rate=5e-4, f=sigmoid, df=sigmoid_deriv, device="cuda", enable_adam=True):
        self.input_size = C * H * W
        self.C = C
        self.H = H
        self.W = W
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.f = f
        self.df = df
        self.device = device
        self.weights = torch.empty([self.input_size, self.output_size]).normal_(mean=0.0,std=0.05).to(self.device)
        self.enable_adam = enable_adam

        # Adam parameters for weights
        self.beta1_weights = 0.9
        self.beta2_weights = 0.999
        self.eps_weights = 1e-8
        self.m_weights = torch.zeros_like(self.weights)
        self.v_weights = torch.zeros_like(self.weights)
        self.t_weights = 0  # Time step for weights

    def forward(self, x):
        self.inp = x.detach().clone()
        B = self.inp.shape[0]
        self.inp = self.inp.reshape((B, -1))
        self.activations = torch.matmul(self.inp, self.weights)
        out = self.f(self.activations)
        return out

    def backward(self, e, sigma=1.0):
        self.fn_deriv = self.df(self.activations)
        out = torch.matmul(e * self.fn_deriv, self.weights.T)
        out = out.reshape((len(e), self.C, self.H, self.W))
        out = torch.clamp(out/torch.pow(torch.tensor(sigma), torch.tensor(2.0)),-50,50)
        return out

    def update_weights(self, e, sigma=1.0):
        self.fn_deriv = self.df(self.activations)
        dw = torch.matmul(self.inp.T, e * self.fn_deriv)
        dw = torch.clamp(dw * 2/torch.pow(torch.tensor(sigma), torch.tensor(2.0)),-50,50)
        if self.enable_adam:
            self.t_weights += 1
            self.m_weights = self.beta1_weights * self.m_weights + (1 - self.beta1_weights) * dw
            self.v_weights = self.beta2_weights * self.v_weights + (1 - self.beta2_weights) * (dw ** 2)
            m_hat_weights = self.m_weights / (1 - self.beta1_weights ** self.t_weights)
            v_hat_weights = self.v_weights / (1 - self.beta2_weights ** self.t_weights)
            adjusted_lr_weights = self.learning_rate * m_hat_weights / (torch.sqrt(v_hat_weights) + self.eps_weights)
            self.weights += adjusted_lr_weights
        else:
            self.weights += self.learning_rate * dw
        return
    
    def __repr__(self):
        return f"{self.__class__.__name__}(C={self.C}, H={self.H}, W={self.W}, output_size={self.output_size}, learning_rate={self.learning_rate}, f={self.f.__name__}, df={self.df.__name__}, device={self.device}, enable_adam={self.enable_adam})"


class DLL_OUTER_W_ProjectionLayer(object):
    def __init__(self, C, H, W, output_size, learning_rate=5e-4, f=tanh, df=tanh_deriv, device="cuda"):
        self.input_size = C * H * W
        self.C = C
        self.H = H
        self.W = W
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.f = f
        self.df = df
        self.device = device
        self.weights = torch.empty([self.input_size, self.output_size]).normal_(mean=0.0,std=0.05).to(self.device)
        self.theta = torch.empty([self.input_size, self.output_size]).normal_(mean=0.0, std=1).to(self.device)

    def forward(self, x):
        self.inp = x.detach().clone()
        batch_size = self.inp.shape[0]
        self.activations = self.inp.reshape((batch_size, -1))
        out = torch.matmul(
            self.f(
                self.activations
            ),self.weights)
        return out

    def backward(self, e, sigma=1.0):
        self.fn_deriv = self.df(self.activations)
        out = torch.matmul(e,self.theta.T) * self.fn_deriv
        out = out.reshape((len(e), self.C, self.H, self.W))
        out = torch.clamp(out/torch.pow(torch.tensor(sigma), torch.tensor(2.0)),-50,50)
        return out

    def update_weights(self, e, sigma=1.0):
        dw = torch.matmul(
            self.f(
                self.activations).T,
            e
        )
        self.weights += self.learning_rate * torch.clamp(dw * 2 / torch.pow(torch.tensor(sigma), torch.tensor(2.0)),
                                                         -50, 50)
        return

    def update_theta(self, e1, e2, sigma=1.0):
        batch_size = e1.shape[0]
        dtheta = torch.matmul((self.df(self.activations) * e1.reshape(batch_size, -1)).T, e2)
        self.theta += self.learning_rate / 10 * torch.clamp(dtheta*2/torch.pow(torch.tensor(sigma), torch.tensor(2.0)),-50,50)
        return
    
    def __repr__(self):
        return f"{self.__class__.__name__}(C={self.C}, H={self.H}, W={self.W}, output_size={self.output_size}, learning_rate={self.learning_rate}, f={self.f.__name__}, df={self.df.__name__}, device={self.device})"

    
class DLL_INNER_W_ProjectionLayer(object):
    def __init__(self, C, H, W, output_size, learning_rate=5e-4, f=tanh, df=tanh_deriv, device="cuda", enable_adam=True, enable_sgd=False):
        self.input_size = C * H * W
        self.C = C
        self.H = H
        self.W = W
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.f = f
        self.df = df
        self.device = device
        self.weights = torch.empty([self.input_size, self.output_size]).normal_(mean=0.0,std=0.05).to(self.device)
        self.theta = torch.empty([self.input_size, self.output_size]).normal_(mean=0.0, std=1).to(self.device)

        self.batchnorm = BatchNorm1d
        self.enable_adam = enable_adam

        # Adam parameters for weights
        self.beta1_weights = 0.9
        self.beta2_weights = 0.999
        self.eps_weights = 1e-8
        self.m_weights = torch.zeros_like(self.weights)
        self.v_weights = torch.zeros_like(self.weights)
        self.t_weights = 0  # Time step for weights
        
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
            self.velocity_weights = torch.zeros_like(self.weights)
            self.velocity_theta = torch.zeros_like(self.theta)

    def forward(self, x):
        self.inp_lengths = None
        if isinstance(x, list):
            self.inp = [t.detach().clone().reshape(t.shape[0], -1) for t in x]
            self.inp_lengths = [t.shape[1] for t in self.inp]
            self.inp = torch.cat(self.inp, dim=1)
        else:
            self.inp = x.detach().clone()
            batch_size = self.inp.shape[0]
            self.inp = self.inp.reshape((batch_size, -1))
        self.activations = torch.matmul(self.inp, self.weights)
        # return self.batchnorm(self.f(self.activations))
        return self.f(self.activations)

    def backward(self, e, sigma=1.0):
        self.fn_deriv = self.df(self.activations)
        out = torch.matmul(e * self.fn_deriv, self.theta.T)
        
        if self.inp_lengths:
            outs = []
            start_idx = 0
            for length in self.inp_lengths:
                end_idx = start_idx + length
                out_part = out[:, start_idx:end_idx]
                out_part = out_part.reshape((len(e), self.C, self.H, self.W))
                out_part = torch.clamp(out_part/torch.pow(torch.tensor(sigma), torch.tensor(2.0)), -50, 50)
                outs.append(out_part)
                start_idx = end_idx
            return outs
        else:
            out = out.reshape((len(e), self.C, self.H, self.W))
            out = torch.clamp(out/torch.pow(torch.tensor(sigma), torch.tensor(2.0)), -50, 50)
            return out

    def update_weights(self, e, sigma=1.0):
        self.fn_deriv = self.df(self.activations)
        dw = torch.matmul(self.inp.T, e * self.fn_deriv)
        dw = torch.clamp(dw * 2 / torch.pow(torch.tensor(sigma), torch.tensor(2.0)), -50, 50)
        if self.enable_adam:
            self.t_weights += 1
            self.m_weights = self.beta1_weights * self.m_weights + (1 - self.beta1_weights) * dw
            self.v_weights = self.beta2_weights * self.v_weights + (1 - self.beta2_weights) * (dw ** 2)
            m_hat_weights = self.m_weights / (1 - self.beta1_weights ** self.t_weights)
            v_hat_weights = self.v_weights / (1 - self.beta2_weights ** self.t_weights)
            adjusted_lr_weights = self.learning_rate * m_hat_weights / (torch.sqrt(v_hat_weights) + self.eps_weights)
            self.weights += adjusted_lr_weights
        elif self.enable_sgd:
            dw += self.weight_decay * self.weights
            if self.momentum > 0:
                self.velocity_weights = self.momentum * self.velocity_weights - self.learning_rate * dw
                self.weights += self.velocity_weights
            else:
                self.weights -= self.learning_rate * dw
        else:
            self.weights += self.learning_rate * dw
        return

    def update_theta(self, e1, e2, sigma=1.0):
        if isinstance(e1, list):
            e1 = torch.cat([t.detach().clone().reshape(t.shape[0], -1) for t in e1], dim=1)
        else:
            e1 = e1.reshape(e1.shape[0], -1)
        dtheta = torch.matmul(e1.T, self.df(self.activations) * e2)
        dtheta = torch.clamp(dtheta*2/torch.pow(torch.tensor(sigma), torch.tensor(2.0)),-50,50)
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
        return f"{self.__class__.__name__}(C={self.C}, H={self.H}, W={self.W}, output_size={self.output_size}, learning_rate={self.learning_rate}, f={self.f.__name__}, df={self.df.__name__}, device={self.device}, enable_adam={self.enable_adam}, enable_sgd={self.enable_sgd})"


# feedback alignment FCLayer
class FA_FCLayer(object):
    def __init__(self, input_size, output_size, learning_rate=5e-4, f=sigmoid, df=sigmoid_deriv, device="cuda"):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.f = f
        self.df = df
        self.device = device
        # self.fa_matrix = torch.empty([self.input_size, self.output_size]).normal_(mean=0.0,std=1).to(self.device)
        self.fa_matrix = generate_ones_and_minus_ones_matrix(self.input_size, self.output_size).to(self.device)
        self.weights = torch.empty([self.input_size, self.output_size]).normal_(mean=0.0,std=0.05).to(self.device)

    def forward(self, x):
        self.inp = x.clone()
        self.activations = torch.matmul(self.inp, self.weights)
        out = self.f(self.activations)
        return out

    def backward(self, e, sigma=1.0):
        self.fn_deriv = self.df(self.activations)
        out = torch.matmul(e * self.fn_deriv, self.fa_matrix.T)
        out = torch.clamp(out/torch.pow(torch.tensor(sigma), torch.tensor(2.0)),-50,50)
        return out

    def update_weights(self, e, sigma=1.0):
        self.fn_deriv = self.df(self.activations)
        dw = torch.matmul(self.inp.T, e * self.fn_deriv)
        self.weights += self.learning_rate * torch.clamp(dw * 2/torch.pow(torch.tensor(sigma), torch.tensor(2.0)),-50,50)
        return
    
    def __repr__(self):
        return f"{self.__class__.__name__}(input_size={self.input_size}, output_size={self.output_size}, learning_rate={self.learning_rate}, f={self.f.__name__}, df={self.df.__name__}, device={self.device})"

class Outer_W_FCLayer(object):
    def __init__(self, input_size, output_size, learning_rate=5e-4, f=sigmoid, df=sigmoid_deriv, device="cuda"):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.f = f
        self.df = df
        self.device = device
        self.weights = torch.empty([self.input_size, self.output_size]).normal_(mean=0.0,std=0.05).to(self.device)

    def forward(self, x):
        self.inp = x.clone()
        self.activations = self.inp.clone()
        out = torch.matmul(self.f(self.activations), self.weights)
        return out

    def backward(self, e, sigma=1.0):
        self.fn_deriv = self.df(self.activations)
        out = torch.matmul(e, self.weights.T) * self.fn_deriv
        out = torch.clamp(out/torch.pow(torch.tensor(sigma), torch.tensor(2.0)),-50,50)
        return out

    def update_weights(self, e, sigma=1.0):
        dw = torch.matmul(self.f(self.activations).T, e)
        self.weights += self.learning_rate * torch.clamp(dw*2/torch.pow(torch.tensor(sigma), torch.tensor(2.0)),-50,50)
        return
    
    def __repr__(self):
        return f"{self.__class__.__name__}(input_size={self.input_size}, output_size={self.output_size}, learning_rate={self.learning_rate}, f={self.f.__name__}, df={self.df.__name__}, device={self.device})"

class FA_Outer_W_FCLayer(object):
    def __init__(self, input_size, output_size, learning_rate=5e-4, f=sigmoid, df=sigmoid_deriv, device="cuda"):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.f = f
        self.df = df
        self.device = device
        # self.fa_matrix = torch.empty([self.input_size, self.output_size]).normal_(mean=0.0,std=0.1).to(self.device)
        self.fa_matrix = generate_ones_and_minus_ones_matrix(self.input_size, self.output_size).to(self.device)
        self.weights = torch.empty([self.input_size, self.output_size]).normal_(mean=0.0,std=0.05).to(self.device)

    def forward(self, x):
        self.inp = x.clone()
        self.activations = self.inp.clone()
        out = torch.matmul(self.f(self.activations), self.weights)
        return out

    def backward(self, e, sigma=1.0):
        self.fn_deriv = self.df(self.activations)
        out = torch.matmul(e, self.fa_matrix.T) * self.fn_deriv
        out = torch.clamp(out/torch.pow(torch.tensor(sigma), torch.tensor(2.0)),-50,50)
        return out

    def update_weights(self, e, sigma=1.0):
        dw = torch.matmul(self.f(self.activations).T, e)
        self.weights += self.learning_rate * torch.clamp(dw*2/torch.pow(torch.tensor(sigma), torch.tensor(2.0)),-50,50)
        return
    
    def __repr__(self):
        return f"{self.__class__.__name__}(input_size={self.input_size}, output_size={self.output_size}, learning_rate={self.learning_rate}, f={self.f.__name__}, df={self.df.__name__}, device={self.device})"

# default outer w
class DLL_FCLayer(object):
    def __init__(self, input_size, output_size, learning_rate=5e-4, f=sigmoid, df=sigmoid_deriv, device="cuda", enable_adam = True):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.f = f
        self.df = df
        self.device = device
        self.weights = torch.empty([self.input_size, self.output_size]).normal_(mean=0.0,std=0.05).to(self.device)
        self.theta = torch.empty([self.input_size, self.output_size]).normal_(mean=0.0,std=1).to(self.device)
        # self.theta = generate_ones_and_minus_ones_matrix(self.input_size, self.output_size).to(self.device)
        self.enable_adam = enable_adam

        # Adam parameters for weights
        self.beta1_weights = 0.9
        self.beta2_weights = 0.999
        self.eps_weights = 1e-8
        self.m_weights = torch.zeros_like(self.weights)
        self.v_weights = torch.zeros_like(self.weights)
        self.t_weights = 0  # Time step for weights
        
        # Adam parameters for theta
        self.beta1_theta = 0.9
        self.beta2_theta = 0.999
        self.eps_theta = 1e-8
        self.m_theta = torch.zeros_like(self.theta)
        self.v_theta = torch.zeros_like(self.theta)
        self.t_theta = 0  # Time step for theta

    def forward(self, x):
        self.inp = x.clone()
        self.activations = self.inp.clone()
        out = torch.matmul(self.f(self.activations), self.weights)
        return out
    
    def backward(self, e, sigma=1.0):
        self.fn_deriv = self.df(self.activations)
        out = torch.matmul(e, self.theta.T) * self.fn_deriv
        out = torch.clamp(out/torch.pow(torch.tensor(sigma), torch.tensor(2.0)),-50,50)
        return out

    def update_weights(self, e, sigma=1.0):
        dw = torch.matmul(self.f(self.activations).T, e)
        dw = torch.clamp(dw * 2 / torch.pow(torch.tensor(sigma), torch.tensor(2.0)), -50, 50)
        if self.enable_adam:
            self.t_weights += 1
            self.m_weights = self.beta1_weights * self.m_weights + (1 - self.beta1_weights) * dw
            self.v_weights = self.beta2_weights * self.v_weights + (1 - self.beta2_weights) * (dw ** 2)
            m_hat_weights = self.m_weights / (1 - self.beta1_weights ** self.t_weights)
            v_hat_weights = self.v_weights / (1 - self.beta2_weights ** self.t_weights)
            adjusted_lr_weights = self.learning_rate * m_hat_weights / (torch.sqrt(v_hat_weights) + self.eps_weights)
            self.weights += adjusted_lr_weights
        else:
            self.weights += self.learning_rate * dw
        return

    def update_theta(self, e1, e2, sigma=1.0):
        dtheta = torch.matmul((self.df(self.activations) * e1).T, e2)
        dtheta = torch.clamp(dtheta * 2 / torch.pow(torch.tensor(sigma), torch.tensor(2.0)), -50, 50)
        if self.enable_adam:
            self.t_theta += 1
            self.m_theta = self.beta1_theta * self.m_theta + (1 - self.beta1_theta) * dtheta
            self.v_theta = self.beta2_theta * self.v_theta + (1 - self.beta2_theta) * (dtheta ** 2)
            m_hat_theta = self.m_theta / (1 - self.beta1_theta ** self.t_theta)
            v_hat_theta = self.v_theta / (1 - self.beta2_theta ** self.t_theta)
            adjusted_lr_theta = self.learning_rate / 10 * m_hat_theta / (torch.sqrt(v_hat_theta) + self.eps_theta)
            self.theta += adjusted_lr_theta
        else:
            self.theta += self.learning_rate / 10 * dtheta
        return
    
    def __repr__(self):
        return f"{self.__class__.__name__}(input_size={self.input_size}, output_size={self.output_size}, learning_rate={self.learning_rate}, f={self.f.__name__}, df={self.df.__name__}, device={self.device}, enable_adam={self.enable_adam})"
    
class DLL_INNER_W_FCLayer(object):
    def __init__(self, input_size, output_size, learning_rate=5e-4, f=sigmoid, df=sigmoid_deriv, device="cuda", enable_adam=True, enable_sgd=False):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.f = f
        self.df = df
        self.device = device
        self.weights = torch.empty([self.input_size, self.output_size]).normal_(mean=0.0,std=0.05).to(self.device)
        self.theta = torch.empty([self.input_size, self.output_size]).normal_(mean=0.0,std=1).to(self.device)
        # self.theta = generate_ones_and_minus_ones_matrix(self.input_size, self.output_size).to(self.device)

        self.enable_adam = enable_adam

        # Adam parameters for weights
        self.beta1_weights = 0.9
        self.beta2_weights = 0.999
        self.eps_weights = 1e-8
        self.m_weights = torch.zeros_like(self.weights)
        self.v_weights = torch.zeros_like(self.weights)
        self.t_weights = 0  # Time step for weights
        
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
            self.velocity_weights = torch.zeros_like(self.weights)
            self.velocity_theta = torch.zeros_like(self.theta)

    def forward(self, x):
        self.inp = x.clone()
        # self.activations = self.inp.clone()
        self.activations = torch.matmul(self.inp, self.weights)
        self.activations = self.f(self.activations)

        return self.activations
    
    def backward(self, e, sigma=1.0):
        self.fn_deriv = self.df(self.activations)
        out = torch.matmul(e*self.fn_deriv, self.theta.T)
        out = torch.clamp(out/torch.pow(torch.tensor(sigma), torch.tensor(2.0)),-50,50)
        return out

    def update_weights(self, e, sigma=1.0):
        self.fn_deriv = self.df(self.activations)
        dw = torch.matmul(self.inp.T, e * self.fn_deriv)
        dw = torch.clamp(dw * 2 / torch.pow(torch.tensor(sigma), torch.tensor(2.0)), -50, 50)
        if self.enable_adam:
            self.t_weights += 1
            self.m_weights = self.beta1_weights * self.m_weights + (1 - self.beta1_weights) * dw
            self.v_weights = self.beta2_weights * self.v_weights + (1 - self.beta2_weights) * (dw ** 2)
            m_hat_weights = self.m_weights / (1 - self.beta1_weights ** self.t_weights)
            v_hat_weights = self.v_weights / (1 - self.beta2_weights ** self.t_weights)
            adjusted_lr_weights = self.learning_rate * m_hat_weights / (torch.sqrt(v_hat_weights) + self.eps_weights)
            self.weights += adjusted_lr_weights
        elif self.enable_sgd:
            dw += self.weight_decay * self.weights
            if self.momentum > 0:
                self.velocity_weights = self.momentum * self.velocity_weights - self.learning_rate * dw
                self.weights += self.velocity_weights
            else:
                self.weights -= self.learning_rate * dw
        else:
            self.weights += self.learning_rate * dw
        return

    def update_theta(self, e1, e2, sigma=1.0):
        batch_size = e1.shape[0]
        dtheta = torch.matmul((e1.reshape(batch_size, -1)).T, self.df(self.activations) * e2)
        dtheta = torch.clamp(dtheta*2/torch.pow(torch.tensor(sigma), torch.tensor(2.0)),-50,50)
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
        return

    def __repr__(self):
        return f"{self.__class__.__name__}(input_size={self.input_size}, output_size={self.output_size}, learning_rate={self.learning_rate}, f={self.f.__name__}, df={self.df.__name__}, device={self.device}, enable_adam={self.enable_adam}, enable_sgd={self.enable_sgd})"

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
