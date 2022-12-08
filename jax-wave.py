import jax.numpy as jnp
from jax import grad, jit, vmap, lax
import jax.scipy as jsp
import jax.scipy.optimize as jsp_opt
import optax 
import jaxopt
from jaxopt import ScipyBoundedMinimize
import matplotlib.pyplot as plt


# Set up an n-point uniform mesh
n = 1000
dx = 1.0/(n-1)
x0 = jnp.linspace(0.0,1.0,n)

@jit
def wave_propagation(params):
    c = params
    dt = 5e-4
    # Sanity check the physical constants
    C = c*dt/dx
    C2 = C*C
    # C should be < 1 for stability

    # Set up initial conditions
    u0 = jnp.exp(-(5*(x0-0.5))**2)
    u1 = jnp.exp(-(5*(x0-0.5-c*dt))**2)
    u2 = jnp.zeros(n)
    def step(i, carry):
        u0, u1, u2 = carry
        # Shift right to get j - 1 and set the first value to 0
        u1p = jnp.roll(u1, 1)
        u1p = u1p.at[0].set(0)
        # Shift left to get j + 1 and set the last value to 0
        u1n = jnp.roll(u1, -1)
        u1n = u1n.at[n - 1].set(0)
        # Central difference in space and finite difference in time
        #  u2[j] = 2*u1[j] -u0[j] + C2*(u1[j-1]-2*u1[j]+u1[j+1])
        u2 = 2 * u1 - u0 + C2*(u1p -2*u1 +u1n)
        u0 = u1
        u1 = u2
        return (u0, u1, u2)
    # Space for time steps
    u0, u1, u2 = lax.fori_loop(0, 5000, step, (u0, u1, u2))

    return u2

# Assign target
# ctarget = jnp.ones(n) * 1.0 # constant model
ctarget = jnp.linspace(0.9, 1.0, n) # Linear model
target = wave_propagation(ctarget)

@jit
def compute_loss(c):
    u2 = wave_propagation(c)
    return jnp.linalg.norm(u2 - target)

# Gradient of forward problem
df = grad(compute_loss)

# Return value and gradient
def value_grad(c):
  return compute_loss(c), df(c)

# Optimizers
def optimizer(params, niter):
  # Initialize parameters of the model + optimizer.
  start_learning_rate = 1e-3
  optimizer = optax.adam(start_learning_rate)
  opt_state = optimizer.init(params)

  # A simple update loop.
  for _ in range(niter):
    grads = grad(compute_loss)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
  return params

# BFGS Optimizer
def bfgs_optimizer(params, niter):
  # opt= jaxopt.BFGS(fun=compute_loss, maxiter=niter)
  opt= jaxopt.BFGS(fun=value_grad, value_and_grad=True, maxiter=niter)
  res = opt.run(init_params=params)
  result, state = res
  return result

# params = jnp.ones(n) * 0.85 # Constant model
params = jnp.linspace(0.85, 1.0, n) # Linear model
result = bfgs_optimizer(params, 100)
# result = optimizer(params, 1000)

# Velocity profile
plt.plot(result, 'r--', label='velocity profile')
plt.plot(ctarget, 'c', label='Target velocity profile')

# Waves
wave = wave_propagation(result)

plt.plot(wave, 'g-.', label='solution')
plt.plot(target, 'b:', label='target')
plt.legend()
plt.savefig("waves.png")

# Newton Raphson
# c = 0.975
# for i in range(20):
#   print("Iteration {} c {}".format(i, c))
#   h = forward(c)/df(c)
#   c = c - h
#   if abs(h) < 1e-5:
#     break