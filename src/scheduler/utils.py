# src/scheduler/utils.py
import holoviews as hv
import numpy as np
# One cycle learning rate range finder multiplicative factor computation
"""
Reference: [sgugger](https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html)

# Setup:
Over a single epoch, plot the losses as updating the gradient descent's learning rate at
each mini_batch iteration. 

- start with initial lr, lr_0
- at each mini_batch, update the lr to lr_{t+1} = lr_{t} * f
    - f: multiplicative lr update factor
- at the end of the iterations for an epoch, we would like the learning rate have tested 
    values in range [ lr_0, lr_t ]. 
    
This means, given `lr_0`, `lr_t`, `number of updates` (ie. `t` -- in this case, number of mini_batch updates in a single epoch), we need to find the factor `f`. 

# Math:
Let a := lr_0, b := lr_t
b = f^t * a
f^t = b/a
log(f**t, base=f) = log(b/a, base=f)
t = log(b/a, base=f) <--- Eqn.(1)

Equivalently,
f = (b/a)^(1/t) <--- Eqn.(2)
"""
def get_mult_factor(x_0, x_n, n):
    """
    x_0 (float): initial value
    x_n (float): final value
    n (int): number of updates to get to x_n from x_1 via mulfiplicative factor f
    
    Returns:
    f (float): multiplicative factor st. x_{t+1} = f * x_{t}
    """
    return (x_n/x_0)**(1./n)


def show_lr_generator(lr_gen, n_iters):
    
    xs = np.arange(n_iters)
    ys = [lr_gen(x) for x in xs]
    return hv.Curve((xs, ys)).opts(width=400)


