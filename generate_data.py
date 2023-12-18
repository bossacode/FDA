import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import Matern
from scipy.linalg import sqrtm
from math import sqrt
import os


# First, we construct 10 base functions: $\cos(k \pi t)$ for $k = 1, \dots, 10$
# For each data, we sum 3 randomly sampled base functions and add Gaussian noise from $N(0, \sigma^2)$
def generate_my_data(tseq, sig=0.5, n=4000, scale=True):
    """
    tseq: timepoints
    sig: std of Gaussian distribution
    n: size of data
    scale: whether to standardize the data
    """
    base_func = [torch.cos(i * torch.pi * tseq) for i in range(10)]
    data_list = []
    for i in range(n):
        ind = np.random.choice(range(10), 3, replace=False)
        data = base_func[ind[0]] + base_func[ind[1]] + base_func[ind[2]]
        data_list.append(data)
    original_data = torch.stack(data_list, dim=0)
    noise_data = original_data + sig * torch.randn(n, len(tseq))    # add gaussian noise from N(0, sig^2)

    # standardize data
    if scale:
        scaler = StandardScaler()
        original_data = torch.from_numpy(scaler.fit_transform(original_data.numpy()))
        noise_data = torch.from_numpy(scaler.fit_transform(noise_data.numpy()))

    if not os.path.exists("data/mydata/"):
        os.makedirs("data/mydata/")
    torch.save(original_data, "data/mydata/original_data.pt")
    torch.save(noise_data, "data/mydata/noise_data.pt")


def generate_matern_data(tseq, nu=0.5, sig=1, n=4000, scale=True):
    """
    tseq: timepoints
    nu:
    sig: std of Gaussian distribution
    n: size of data
    scale: whether to standardize the data
    """
    tseq = tseq.numpy()
    h = tseq[1:] - tseq[:-1]    # length of intervals

    # Matern Process with nu = 1/2
    kernel = sig**2 * Matern(nu=nu)
    cov_mat = kernel(tseq.reshape(T,1), tseq.reshape(T,1))

    z = np.random.randn(n, T)
    data = z @ sqrtm(cov_mat)   # [data_size, T]
    
    # standardize data
    if scale:
        scaler = StandardScaler()
        data = torch.from_numpy(scaler.fit_transform(data)).float()

    if not os.path.exists("data/matern/"):
        os.makedirs("data/matern/")
    torch.save(data, "data/matern/data.pt")


def generate_adafnn_data(tseq, sig=1, n=4000, scale=True):
    """
    tseq: timepoints
    sig: std of Gaussian distribution
    n: size of data
    scale: whether to standardize the data
    """
    K = 5
    # case 3 of AdaFNN paper
    z = torch.ones(n, K)
    z[:, 0] = z[:, 2] = 5
    # z[:, 4] = z[:, 9] = 3
    z[:, 4] = 3
    r = (sqrt(3) + sqrt(3)) * torch.rand(n, K) - sqrt(3)   # uniform from [-sqrt(3), sqrt(3)]
    c = z*r
    phi = torch.stack([torch.ones(len(tseq)) if k == 1
                       else sqrt(2) * torch.cos((k-1) * torch.pi * tseq)
                       for k in range(1, K+1)])
    original_data = torch.matmul(c, phi)
    noise_data = original_data + sig * torch.randn(n, len(tseq))    # add gaussian noise from N(0, sig^2)

    # standardize data
    if scale:
        scaler = StandardScaler()
        original_data = torch.from_numpy(scaler.fit_transform(original_data.numpy()))
        noise_data = torch.from_numpy(scaler.fit_transform(noise_data.numpy()))

    if not os.path.exists("data/adafnn/"):
        os.makedirs("data/adafnn/")
    torch.save(original_data, "data/adafnn/original_data.pt")
    torch.save(noise_data, "data/adafnn/noise_data.pt")


if __name__ == "__main__":
    np.random.seed(123)
    torch.manual_seed(123)

    T = 101 # number of timepoints
    tseq = torch.linspace(0, 1, T)  # timepoints
    n = 10000   # number of data

    if not os.path.exists("data/"):
        os.makedirs("data/")

    generate_my_data(tseq, n=n)
    generate_matern_data(tseq, n=n)
    generate_adafnn_data(tseq, n=n)
