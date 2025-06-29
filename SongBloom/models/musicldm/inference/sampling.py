import torch
import math
from tqdm import trange, tqdm

# Define the noise schedule and sampling loop
def get_alphas_sigmas(t):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

def alpha_sigma_to_t(alpha, sigma):
    """Returns a timestep, given the scaling factors for the clean image and for
    the noise."""
    return torch.atan2(sigma, alpha) / math.pi * 2

def t_to_alpha_sigma(t):
    """Returns the scaling factors for the clean image and for the noise, given
    a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

@torch.no_grad()
def sample_discrete_euler_with_temperature(model, x, steps, temperature=1.0, sigma_max=1.0, prog_bar=False, **extra_args):
    """Draws samples from a model given starting noise. Euler method"""

    # Make tensor of ones to broadcast the single t values
    ts = x.new_ones([x.shape[0]])
    noise = x

    # Create the noise schedule
    t = torch.linspace(sigma_max, 0, steps + 1)
    # all = {}
    x = torch.zeros_like(noise)
    if temperature >= sigma_max:
        x = noise
        
    #alphas, sigmas = 1-t, t
    iterator = tqdm(zip(t[:-1], t[1:]), total=steps) if prog_bar else zip(t[:-1], t[1:])
    for t_curr, t_prev in iterator:
            # Broadcast the current timestep to the correct shape
                
            t_curr_tensor = t_curr * torch.ones(
                (x.shape[0],), dtype=x.dtype, device=x.device
            )
            dt = t_prev - t_curr  # we solve backwards in our formulation
            v = model(x, t_curr_tensor, **extra_args)
            # all[t_curr.item()] = x-t_curr*v
            if  t_curr > temperature and t_prev <= temperature:
                x_0 = x - v
                x = (1-t_prev) * x_0 + t_prev * noise
            else:
                x = x + dt * v #.denoise(x, denoiser, t_curr_tensor, cond, uc)
            
    # If we are on the last timestep, output the denoised image
    return x #, all 