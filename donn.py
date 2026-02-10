import torch, math
from torch.nn import functional as F, init
from torch import nn
import math
from typing import List, Tuple

def cReLU(x):
    return torch.complex(torch.relu(torch.real(x)),
                         torch.relu(torch.imag(x)))

def cLeakyReLU(x):
    return torch.complex(torch.nn.LeakyReLU()(torch.real(x)),
                         torch.nn.LeakyReLU()(torch.imag(x)))

class MuHopf(nn.Module):
    def __init__(self, input_dim: int, dt: float, num_steps: int,
                 min_omega: float, max_omega: float, device="cpu"):

        super(MuHopf, self).__init__()
        self.omegas = nn.Parameter(torch.rand(input_dim, device=device))
        self.min_omega = min_omega
        self.max_omega = max_omega
        self.dt = dt
        self.mu0 = 1.0
        self.num_steps = num_steps

    def forward(self, x: torch.Tensor, r: torch.Tensor,
                phi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        assert r.size() == x.size()
        assert x.size() == phi.size()

        output = torch.jit.annotate(List[torch.Tensor], [])

        omega_intl = self.max_omega - self.min_omega
        omegas = torch.sigmoid(self.omegas) * omega_intl + self.min_omega
        omegas = omegas * (2*torch.pi)

        for t in range(self.num_steps):
            mu = 5 * (self.mu0 + x)
            r = r + ((mu - (r**2)) * r) * self.dt
            phi = phi + omegas * self.dt
            z_r = r * torch.cos(phi)
            z_i = r * torch.sin(phi)
            z_t = torch.complex(z_r, z_i)
            output.append(z_t)

        return torch.stack(output, dim=0), r, phi

class ResHopf(torch.nn.Module):
    
    def __init__(self, units, min_omega, max_omega, dt, 
                 train_omegas=True, input_scaler=5.0, 
                 *args, **kwargs) -> None:
        
        super(ResHopf, self).__init__(*args, **kwargs)
        
        self.units = units
        self.min_omega = min_omega
        self.max_omega = max_omega
        self.dt = dt

        self.mu0 = torch.tensor(1.0)
        self.beta1 = 1.0
        self.input_scaler = input_scaler
        self.train_omegas = train_omegas

        self.omegas = nn.Parameter(torch.rand((1, units)))

    def forward(self, X, r, phi):
        
        Xr = torch.real(X)
        Xi = torch.imag(X)
        assert Xr.shape == Xi.shape
        assert r.shape == Xr[0].shape
        assert r.shape == phi.shape 
        
        r_save = torch.zeros(Xr.shape)
        phi_save = torch.zeros(Xr.shape)
        
        omega_intl = self.max_omega - self.min_omega
        omegas = torch.sigmoid(self.omegas) * omega_intl + self.min_omega
        omegas = omegas * (2*torch.pi)
        
        for t in range(X.shape[0]):
            input_r = self.input_scaler*Xr[t]*torch.cos(phi)
            input_phi = self.input_scaler*Xi[t]*torch.sin(phi)
            r = r + ((self.mu0 - self.beta1*(r**2)) * r + input_r) * self.dt
            phi = phi + (omegas - input_phi) * self.dt            
            r_save[t] = r
            phi_save[t] = phi
        
        z_r = r_save * torch.cos(phi_save)
        z_i = r_save * torch.sin(phi_save)

        return torch.complex(z_r, z_i), r, phi

class SingledtResHopf(torch.nn.Module):
    
    def __init__(self, units, min_omega, max_omega, dt, 
                 train_omegas=True, 
                 input_scaler=5.0, *args, **kwargs) -> None:

        super(SingledtResHopf, self).__init__(*args, **kwargs)

        self.units = units
        self.min_omega = min_omega
        self.max_omega = max_omega
        self.dt = dt

        self.mu0 = torch.tensor(1.0)
        self.beta1 = 1.0
        self.input_scaler = input_scaler
        self.train_omegas = train_omegas

        self.omegas = torch.nn.Parameter(torch.randn((1, units)))

    def forward(self, X, r, phi):
        
        Xr = torch.real(X)
        Xi = torch.imag(X)
        assert Xr.shape == Xi.shape
        assert r.shape == Xr.shape
        assert r.shape == phi.shape 
        
        omega_intl = self.max_omega - self.min_omega
        omegas = torch.sigmoid(self.omegas) * omega_intl + self.min_omega
        omegas = omegas * (2*torch.pi)
        
        input_r = self.input_scaler*Xr*torch.cos(phi)
        input_phi = self.input_scaler*Xi*torch.sin(phi)
        r = r + ((self.mu0 - self.beta1*(r**2)) * r + input_r) * self.dt
        phi = phi + (omegas - input_phi) * self.dt            

        z_r = r * torch.cos(phi)
        z_i = r * torch.sin(phi)

        return torch.complex(z_r, z_i), r, phi

