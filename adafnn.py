import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def _inner_product(f1, f2, h):
    """    
    f1 - (B, T) : value of B functions, observed at T time points,
    f2 - (B, T) : same as f1
    h  - (T-1, ): interval between time points
    pay attention to dimension
    <f1, f2> = sum (h/2) (f1(t{j}) + f2(t{j+1}))
    """
    prod = f1 * f2 # (B, T = len(h) + 1)
    return torch.matmul((prod[:, :-1] + prod[:, 1:]), h.unsqueeze(dim=-1))/2    # shape: [B, 1]


def _l1(f, h):
    # f dimension : (B bases, T)
    B, T = f.size()
    return _inner_product(torch.abs(f), torch.ones((B, T)), h)


def _l2(f, h):
    # f dimension : (B bases, T )
    # output dimension - (B bases, 1)
    return torch.sqrt(_inner_product(f, f, h))


# def numeric_deriv(f, h, degree=2):
#     """
#     input:
#         f - (B, T) : value of B functions, observed at T time points,
#         h - (T-1, ): interval between time points
#         degree : 1 or 2, denoting first or second derivative
#     output:
#         deriv - (B, T-1) or (B, T-2)
#     """
#     assert degree in (1,2)
#     if degree == 1:
#         # first derivative: (f(x+h) - f(x))/h
#         deriv = (f[:, 1:] - f[:, :-1]) / h
#     else:
#         # second derivative(needs equal spacing): (f(x+2h) - 2f(x+h) +f(x)) / h^2
#         deriv = (f[:, 2:] - 2*f[:, 1:-1] + f[:, :-2]) / (h[:-1]**2)
#     return deriv


class LayerNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        # d is size of the dimension we want to normalize
        self.eps = eps
        self.alpha = nn.Parameter(torch.randn(d))
        self.beta = nn.Parameter(torch.randn(d))

    def forward(self, x):
        # x is a torch.Tensor of shape: [batch_size, d]
        # avg is the mean value of a layer
        avg = x.mean(dim=-1, keepdim=True)
        # std is the standard deviation of a layer (eps is added to prevent dividing by zero)
        std = x.std(dim=-1, keepdim=True) + self.eps
        return (x - avg) / std * self.alpha + self.beta


class MicroNN(nn.Module):
    def __init__(self, in_d=1, hidden=[4,4,4], dropout=0.1, activation=nn.ReLU()):
        """
            in_d      : input dimension, integer
            hidden    : hidden layer dimension, array of integers
            dropout   : dropout probability, a float between 0.0 and 1.0
            activation: activation function at each layer
        """
        super().__init__()
        self.activation = activation
        dim = [in_d] + hidden + [1]
        self.layers = nn.ModuleList([nn.Linear(dim[i-1], dim[i]) for i in range(1, len(dim))])
        self.ln = nn.ModuleList([LayerNorm(k) for k in hidden])
        self.dp = nn.ModuleList([nn.Dropout(dropout) for _ in range(len(hidden))])

    def forward(self, t):
        for i in range(len(self.layers)-1):
            t = self.layers[i](t)
            t = t + self.ln[i](t)   # skip connection
            t = self.activation(t)
            t = self.dp[i](t)       # apply dropout
        return self.layers[-1](t)   # linear activation at the last layer
    

class BasisLinearTransform(nn.Module):
    def __init__(self, n_base=4):
        """
        n_base: number of bases used
        n_weights: corresponds to the number of m1 in the above equation
        """
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_base, n_base))
    
    def forward(self, bases):
        """
        input:
            bases - [n_base, T]: Tensor containing "n_bases" bases observed at T time points
        
        output:
            new_bases - [n_base, T]: Tensor containing linearly transformed bases
        """
        new_bases = torch.matmul(self.weights, bases)   # shape: [n_base, T]
        return new_bases


class MyAdaFNN(nn.Module):
    def __init__(self, n_base=4, base_hidden=[64, 64, 64], tseq=torch.linspace(0, 1, 101),
                 dropout=0.1, lambda1=0.0, lambda2=0.0, device=None):
        """
        n_base      : number of basis nodes, integer
        base_hidden : hidden layers used in each basis node, array of integers
        tseq        : observation time points, must be torch.Tensor
        dropout     : dropout probability
        lambda1     : penalty of L1 regularization, a positive real number
        lambda2     : penalty of L2 regularization, a positive real number
        device      : device for the training
        """
        super().__init__()
        self.n_base = n_base
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.device = device
        # time points
        self.tseq = tseq.to(device)
        # interval between time points (used as weights in numerical integration)
        self.h = self.tseq[1:] - self.tseq[:-1]
        # instantiate each basis node in the basis layer
        self.BL = nn.ModuleList([MicroNN(1, hidden=base_hidden, dropout=dropout, activation=nn.SELU())
                                 for _ in range(n_base)])
        # instantiate the subsequent network
        self.linear_trans = BasisLinearTransform(n_base)

    def forward(self, x):
        batch_size, T = x.size()
        assert T == self.h.size()[0] + 1
        tseq = self.tseq.unsqueeze(dim=-1)    # shape: [T, 1]
        # evaluate the current basis nodes at time points
        bases = [basis(tseq).transpose(-1, -2) for basis in self.BL]    # each element is a basis with shape: [1, T]
        bases = self.linear_trans(torch.cat(bases, dim=0))              # linear transformation of basis, shape: [n_base, T]
        """
        compute each basis node's L2 norm
        normalize basis nodes
        """
        l2_norm = _l2(bases, self.h).detach()    # shape: [n_base, 1]
        self.normalized_bases = [bases[[i]] / (l2_norm[i, 0] + 1e-6) for i in range(self.n_base)]  # each element is a basis with shape: [1, T]
        # compute each score <basis_i, f> 
        score = torch.cat([_inner_product(b.repeat((batch_size, 1)), x, self.h)     # shape: [batch_size, 1]
                           for b in self.normalized_bases], dim=-1) # score shape: [batch_size, n_base]
        # multiply scores to basis (Parceval's Theorem)
        out = torch.matmul(score, torch.cat(self.normalized_bases, dim=0))    # shape: [batch_size, T]
        return out

    def R1(self, l1_k):
        """
        L1 regularization
        l1_k : number of basis nodes to regularize, integer        
        """
        if self.lambda1 == 0: return torch.zeros(1).to(self.device)
        # sample l1_k basis nodes to regularize
        selected = np.random.choice(self.n_base, min(l1_k, self.n_base), replace=False)
        selected_bases = torch.cat([self.normalized_bases[i] for i in selected], dim=0) # (k, J)
        return self.lambda1 * torch.mean(_l1(selected_bases, self.h))

    def R2(self, l2_pairs=None):
        """
        L2 regularization
        l2_pairs : number of pairs to regularize, integer  
        """
        if self.lambda2 == 0 or self.n_base == 1: return torch.zeros(1).to(self.device)
        # k = min(l2_pairs, self.n_base * (self.n_base - 1) // 2)
        # f1, f2 = [None] * k, [None] * k
        # for i in range(k):
        #     a, b = np.random.choice(self.n_base, 2, replace=False)
        #     f1[i], f2[i] = self.normalized_bases[a], self.normalized_bases[b]
        # return self.lambda2 * torch.mean(torch.abs(_inner_product(torch.cat(f1, dim=0),
        #                                                           torch.cat(f2, dim=0),
        #                                                           self.h)))
        reg = 0
        count = 0
        for i in torch.combinations(torch.arange(self.n_base)):
            count += 1
            ind1, ind2 = i[0].item(), i[1].item()
            inner_prod = _inner_product(self.normalized_bases[ind1], self.normalized_bases[ind2], self.h)
            reg += 0
        return self.lambda2 * (reg / count)

    # def R3(self):
    #     deriv = numeric_deriv(self.normalized_bases, self.h, degree=2)  # [batch_size, T-2]

    def check_orthonormality(self):
        for i in torch.combinations(torch.arange(self.n_base)):
            ind1, ind2 = i[0].item(), i[1].item()
            inner_prod = _inner_product(self.normalized_bases[ind1], self.normalized_bases[ind2], self.h).detach().item()
            print(f"Inner Product between basis {ind1+1} and basis {ind2+1}: {inner_prod}")

    def plot_bases(self):
        if self.n_base % 2 == 0: # n_base is even
            fig, ax = plt.subplots((self.n_base // 2), 2, figsize=(self.n_base, self.n_base*2))
        else:
            fig, ax = plt.subplots((self.n_base // 2) + 1, 2, figsize=(self.n_base, self.n_base*2))
        plt.setp(ax, ylim=(-3,3))
        for i in range(self.n_base):
            ax[i//2, i%2].plot(self.tseq.detach().cpu(), self.normalized_bases[i].detach().cpu().squeeze())