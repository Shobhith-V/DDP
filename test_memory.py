import torch
from torchdiffeq import odeint_adjoint

class ODE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.tensor([1.0]))
        self.drive = None
    
    def forward(self, t, y):
        # drive has shape (10)
        idx = int(t * 10)
        idx = min(idx, 9)
        return -self.p * y + self.drive[idx]

ode = ODE()
drive = torch.ones(10, requires_grad=True)
ode.drive = drive

y0 = torch.tensor([1.0])
t = torch.linspace(0, 1, 10)

sol = odeint_adjoint(
    ode, 
    y0, 
    t, 
    adjoint_options={}, 
    adjoint_params=(ode.p, drive)
)
loss = sol.sum()
loss.backward()

print('Drive grad:', drive.grad)
print('Param grad:', ode.p.grad)
