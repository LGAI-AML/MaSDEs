import torch
import torch.nn as nn
import numpy as np

def temporal_privacy(c_gap, r_gap, total):
    t = total
    r = r_gap
    c = c_gap
    maxi = divmod((t - 1 - r), c)[0] + 1
    nums = []
    for n in range(total):
        if n < c:
            nums.append(torch.tensor([0]))
        if n >= c:
            if n <= r:
                if n % c == 0:
                    nums.append(torch.arange(divmod(n, c)[0], -1, -1))
                else:
                    nums.append(torch.arange(divmod(n, c)[0], -1, -1))
            else:
                if n % c == 0:
                    nums.append(
                        torch.arange(
                            divmod(n, c)[0], divmod(n, c)[0] - int(r / c) - 1, -1
                        )
                    )
                else:
                    nums.append(
                        torch.arange(divmod(n, c)[0], divmod(n, c)[0] - int(r / c), -1)
                    )
    if c_gap < 3:
        for i in range(3):
            nums[i] = torch.cat((nums[i], torch.tensor([total + 1, total + 2])))
    return nums, maxi


class LipSwish(torch.nn.Module):
    def forward(self, x):
        return 0.909 * torch.nn.functional.silu(x)
    
class AdaIN(torch.nn.Module):
    def forward(self, X, Agent, I, eps = 1e-5):
        # X , Agent : B x Dim , B x Dim
        assert (X.size() == Agent.size())
        size = X.size()
        
        Agent = (Agent + 1) / I

        X_mean, X_std = X.mean(0, keepdim=True),  X.std(0, keepdim=True) + eps
        
        normalized_X = (X - X_mean.expand(
            size)) / X_std.expand(size)
        
        return Agent * normalized_X + Agent


class temporal_privacy_fcns(nn.Module):
    def __init__(self, args):
        super(temporal_privacy_fcns, self).__init__()
        self.args = args
        
        
        self.bottleneck_front = nn.Linear(2 * self.args.D + 3, self.args.hidden_dim)
        self.bottleneck_drift = nn.Linear(self.args.hidden_dim, self.args.D)
        # self.bottleneck_diffusion = nn.Linear(self.hidden_features, data_dim)
        self.bottleneck_weight = nn.Sequential(
            nn.Linear(self.args.hidden_dim, self.args.hidden_weight),
            nn.ReLU(),
            nn.Linear(self.args.hidden_weight, self.args.hidden_weight),
            nn.ReLU(),
            nn.Linear(self.args.hidden_weight, 1),
        )        
        
        self.AdaIN = AdaIN()       
        self.activation = LipSwish()
        self.softmax = nn.Softmax()
        
        nums, maxi = temporal_privacy(self.args.c_gap, self.args.r_gap, self.args.T_p)

        self.layers = []
        for i in range(self.args.T_p + 4):
            self.block = []
            
            for f in range(self.args.L):
                self.block.append(
                    nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
                    )
                self.block.append(self.activation)
            self.layers.append(nn.Sequential(*self.block))

        self.dense_layers = []
        
        for i in range(self.args.T_p):
            d = nums[i]
            self.denses = []
            for j in range(len(d)):
                self.denses.append(self.layers[d[j]])
            self.dense_layers.append(nn.Sequential(*self.denses))
        self.all_layers = nn.Sequential(*self.dense_layers)

    def forward(self, input, m, t, N, agent): 
        t_1 = torch.cos(t / self.args.T)
        t_2 = torch.sin(t / self.args.T)
        t_3 = t / self.args.T
        X = torch.cat((input, m, t_1, t_2, t_3), 1)
        X = self.bottleneck_front(X)
        
        if self.args.AdaIN == True:
            A_ = agent * torch.ones_like(X)
            X = self.AdaIN(X, A_, self.args.A)
            
        out = 0
        for k in range(len(self.dense_layers[N])):
            out += self.all_layers[N][k](X)

        X = out / float(len(self.dense_layers[N]))
        
        X_to_drift = self.bottleneck_drift(X)
        # X_to_diffusion = self.bottleneck_diffusion(X)
        X_to_diffusion = torch.ones_like(X_to_drift)
        X_to_weight = self.bottleneck_weight(X)
        return X_to_diffusion, X_to_drift, X_to_weight  
