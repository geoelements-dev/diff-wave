import torch
import matplotlib.pyplot as plt

# Set up an n-point uniform mesh
n = 1000
dx = 1.0/(n-1)
x0 = torch.linspace(0.0,1.0,n)

def forward(c):
    dt = 5e-4
    # Sanity check the physical constants
    C = c*dt/dx
    C2 = C*C
    # print("CFL constant is {0} (should be < 1 for stability)".format(C))

    # Set up initial conditions
    u0 = torch.exp(-(5*(x0-0.5))**2)
    u1 = torch.exp(-(5*(x0-0.5-c*dt))**2)
    a = []

    # Space for time steps
    for i in range(5000):
        u2 = torch.zeros(n)    
        # Shift right to get j - 1 and set the first value to 0
        u1p = torch.roll(u1, 1)
        u1p[0] = 0
        # Shift left to get j + 1 and set the last value to 0
        u1n = torch.roll(u1, -1)
        u1n[n - 1] = 0
        # Central difference in space and finite difference in time
        #  u2[j] = 2*u1[j] -u0[j] + C2*(u1[j-1]-2*u1[j]+u1[j+1])
        u2 = 2 * u1 - u0 + C2*(u1p -2*u1 +u1n)
        if i % 10 == 0:
            a.append(u2)    
        u0 = u1
        u1 = u2
    
    return u2

# Speed of sound, space step, time step
# ctarget = torch.ones(n) * 0.9 # constant model
ctarget = torch.linspace(0.9, 1.0, n) # linear model

target = forward(ctarget)

def wave_optimize(c):
    u2 = forward(c)
    return torch.norm(target - u2)


# Optimize the wave velocity
# c = torch.ones(n) * 0.85 # Constant model
c = torch.linspace(0.85, 1.0, n) # Linear model

c.requires_grad = True

# LBGFS Optimizer
# optim = torch.optim.LBFGS([c], history_size=10, max_iter=100, line_search_fn="strong_wolfe")
# optim = torch.optim.SGD([c], lr=1e-4, momentum=1e-3)
optim = torch.optim.Adam([c], lr=1e-4)

# A closure that reevaluates the model and returns the loss.
def closure():
    optim.zero_grad()
    loss = wave_optimize(c)
    loss.backward()
    return loss

def optimize(iter):
    # Optimize
    for i in range(iter):
        optim.step(closure)
        loss = wave_optimize(c)
        print('Step: {} loss: {}, wave velocity min {} max {} avg {}'.format(i, loss.item(), torch.min(c), torch.max(c), torch.sum(c)/len(c)))

# For LBFGS Optimizer use n_optim_steps = 1
n_optim_steps = 1000
optimize(n_optim_steps)

# Velocity profile
plt.plot(c.detach().numpy(), 'r--', label='velocity profile')
plt.plot(ctarget.detach().numpy(), 'c', label='Target velocity profile')

# Waves
wave = forward(c)

plt.plot(wave.detach().numpy(), 'g-.', label='solution')
plt.plot(target.detach().numpy(), 'b:', label='target')
plt.legend()
plt.savefig("waves.png")