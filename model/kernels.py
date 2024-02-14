import torch
import numpy as np
import scipy
import math

@torch.no_grad()
def get_gaussian_kernel_np(t, kernel_size: int):
        # when t=0, the returned kernel is a impulse function
        if isinstance(t, torch.Tensor):
                t = t.cpu()
        assert kernel_size % 2 == 1 and kernel_size > 0, f"invalid kernel_size={kernel_size}"
        ns = np.arange(-(kernel_size//2), kernel_size//2+1)
        kernel = math.exp(-t) * scipy.special.iv(ns, t)
        return torch.tensor(kernel).float()

def get_gaussian_kernel(t, kernel_size: int):
        #ic(sigma, kernel_size)
        ns =torch.arange(-(kernel_size//2), kernel_size//2+1, device=t.device, dtype=torch.float32)
        exponent = - 0.5 * (ns / max(t,0.0001)) * (ns / max(t,0.0001))
        kernel = 1/(max(t, 0.0001)*math.sqrt(2*math.pi)) * torch.exp(exponent)
        kernel =  torch.clamp(kernel, max=1.0)
        return kernel

@torch.no_grad()
def get_average_kernel(t, kernel_size: int):
        # to be consistent with gaussian kernel
        # we should return impulse when t = 0
        if kernel_size % 2 == 0:
                kernel_size +=1
        if isinstance(t, torch.Tensor):
                t = t.item()
        t0 = min(math.floor(t),kernel_size//2)
        kernel0 = torch.zeros(kernel_size)
        kernel0[kernel_size//2-t0:kernel_size//2+t0+1] = 1 / (t0*2 + 1)

        t1 = min(math.ceil(t),kernel_size//2)
        kernel1 = torch.zeros(kernel_size)
        kernel1[kernel_size//2-t1:kernel_size//2+t1+1] = 1 / (t1*2+1)

        kernel = (t % 1.0) * kernel1 + (1-t%1.0)*kernel0
        return kernel

# return kernel tensor on the same device as external_sigma
def get_gaussian_diff_kernel(kernel_size: int, sigma_scale=None, external_sigma=None):
        assert external_sigma != None , "external_sigma is not defined"

        if external_sigma is not None:
            sigma = external_sigma * (1 if sigma_scale is None else sigma_scale) # scaling parameter for combined kernel
        else:
            error("gaussian_diff_kernel_sigma is not defined")
        #ic(sigma, kernel_size)
        ns =torch.arange(-(kernel_size//2), kernel_size//2+1).to(device=sigma.device, dtype=torch.float32)
        exponent = - 0.5 * (ns / max(sigma,0.001)) * (ns / max(sigma,0.001))
        kernel = 1/(max(sigma, 0.0001)*math.sqrt(2*math.pi)) * torch.exp(exponent)
        kernel = torch.clamp(kernel, max=1.0)
        return kernel

# return kernel tensor on the same device as external_sigma
def get_batch_gaussian_diff_kernel(kernel_size: int, sigma_scale=None, external_sigma=None):
        # external_sigma is tensorf of shape (n_comp, )
        # return (n_comp, kernel_size), each row corresponds to a kernel from a single external_sigma
        if external_sigma is not None:
            sigma = external_sigma * (1 if sigma_scale is None else sigma_scale) # scaling parameter for combined kernel
        else:
            error("gaussian_diff_kernel_sigma is not defined")

        ns =torch.arange(-(kernel_size//2), kernel_size//2+1, device=sigma.device, dtype=torch.float32)
        ns = ns.unsqueeze(0)
        sigma = sigma.unsqueeze(1)
        one = torch.ones(1, device=sigma.device, dtype=torch.float32)
        eps = torch.tensor(0.0001, device=sigma.device, dtype=torch.float32)
        exponent = -0.5 * ((ns / torch.clamp(sigma, min=0.0001)) ** 2)
        kernel = 1 / (torch.clamp(sigma, min=0.0001) * math.sqrt(2*math.pi)) * torch.exp(exponent)
        kernel = torch.clamp(kernel, max=1.0)
        return kernel
