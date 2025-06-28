import torch
import math
from tqdm import trange, tqdm

# import k_diffusion as K

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
def sample_discrete_euler(model, x, steps, sigma_max=1.0, prog_bar=False, **extra_args):
    """Draws samples from a model given starting noise. Euler method"""

    # Make tensor of ones to broadcast the single t values
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(sigma_max, 0, steps + 1)
    # all = {}

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
            x = x + dt * v #.denoise(x, denoiser, t_curr_tensor, cond, uc)
            
    # If we are on the last timestep, output the denoised image
    return x #, all

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

import numpy as np

@torch.no_grad()
def sample_discrete_euler_spiral(model, x, steps, temperature=1.0, sigma_max=1.0, 
                                 num_paths=3, spiral_strength=0.1, prog_bar=False, 
                                 combination_method='weighted_avg', **extra_args):
    """
    Multi-path spiral sampling using Euler method.
    Each path follows a slightly different spiral trajectory through noise space.
    
    Args:
        model: The diffusion model
        x: Starting noise
        steps: Number of denoising steps
        temperature: Temperature parameter from original sampler
        sigma_max: Maximum noise level
        num_paths: Number of spiral paths to sample
        spiral_strength: How much the spiral affects the trajectory (0.0 = no spiral, 1.0 = strong spiral)
        prog_bar: Show progress bar
        combination_method: 'weighted_avg', 'best_path', or 'ensemble'
        **extra_args: Additional arguments for the model
    
    Returns:
        Combined result from all spiral paths
    """
    
    def create_spiral_schedule(steps, sigma_max, phase_offset=0, spiral_freq=2):
        """Create a spiral-modulated time schedule"""
        # Base linear schedule
        t_linear = torch.linspace(sigma_max, 0, steps + 1)
        
        # Create spiral modulation
        theta = torch.linspace(0, spiral_freq * 2 * np.pi, steps + 1) + phase_offset
        spiral_mod = spiral_strength * torch.sin(theta) * t_linear / sigma_max
        
        # Apply spiral modulation (keep within bounds)
        t_spiral = t_linear + spiral_mod
        t_spiral = torch.clamp(t_spiral, 0, sigma_max)
        
        return t_spiral
    
    def sample_single_path(noise, t_schedule, path_id):
        """Sample a single spiral path"""
        ts = noise.new_ones([noise.shape[0]])
        x_path = torch.zeros_like(noise)
        
        if temperature >= sigma_max:
            x_path = noise.clone()
        
        path_results = []
        iterator = tqdm(zip(t_schedule[:-1], t_schedule[1:]), 
                       total=steps, desc=f"Path {path_id+1}") if prog_bar else zip(t_schedule[:-1], t_schedule[1:])
        
        for t_curr, t_prev in iterator:
            # Broadcast the current timestep
            t_curr_tensor = t_curr * torch.ones(
                (x_path.shape[0],), dtype=x_path.dtype, device=x_path.device
            )
            dt = t_prev - t_curr
            
            # Get model prediction
            v = model(x_path, t_curr_tensor, **extra_args)
            
            # Apply temperature switching logic from original
            if t_curr > temperature and t_prev <= temperature:
                x_0 = x_path - v
                x_path = (1-t_prev) * x_0 + t_prev * noise
            else:
                x_path = x_path + dt * v
                
            # Store intermediate results for potential analysis
            if len(path_results) < 5:  # Store first few steps
                path_results.append(x_path.clone())
        
        return x_path, path_results
    
    # Generate different spiral paths
    all_paths = []
    all_intermediates = []
    
    for path_idx in range(num_paths):
        # Create unique phase offset for each path
        phase_offset = path_idx * (2 * np.pi / num_paths)
        
        # Create spiral schedule for this path
        t_schedule = create_spiral_schedule(steps, sigma_max, phase_offset)
        
        # Sample this path
        path_result, intermediates = sample_single_path(x, t_schedule, path_idx)
        all_paths.append(path_result)
        all_intermediates.append(intermediates)
    
    # Combine paths based on chosen method
    if combination_method == 'weighted_avg':
        # Simple average of all paths
        final_result = torch.stack(all_paths).mean(dim=0)
        
    elif combination_method == 'best_path':
        # Select path with lowest variance (most stable)
        path_variances = [torch.var(path).item() for path in all_paths]
        best_idx = np.argmin(path_variances)
        final_result = all_paths[best_idx]
        
    elif combination_method == 'ensemble':
        # Weighted combination based on path quality metrics
        weights = []
        for path in all_paths:
            # Simple quality metric: negative variance (lower variance = higher weight)
            quality = 1.0 / (1.0 + torch.var(path).item())
            weights.append(quality)
        
        weights = torch.tensor(weights)
        weights = weights / weights.sum()
        
        # Weighted combination
        final_result = torch.zeros_like(all_paths[0])
        for i, (path, weight) in enumerate(zip(all_paths, weights)):
            final_result += weight * path
    
    else:
        # Fallback to simple average
        final_result = torch.stack(all_paths).mean(dim=0)
    
    return final_result

@torch.no_grad()
def sample_discrete_euler_with_temperature_pingpong(model, x, steps, temperature=1.0, sigma_max=1.0, 
                                                      ping_pong_amplitude=0.1, prog_bar=False, **extra_args):
    """Alternative ping pong implementation with amplitude-based oscillation"""

    # Make tensor of ones to broadcast the single t values
    ts = x.new_ones([x.shape[0]])
    noise = x

    # Create the noise schedule
    t = torch.linspace(sigma_max, 0, steps + 1)
    x = torch.zeros_like(noise)
    if temperature >= sigma_max:
        x = noise
        
    iterator = tqdm(zip(t[:-1], t[1:]), total=steps) if prog_bar else zip(t[:-1], t[1:])
    
    for step_idx, (t_curr, t_prev) in enumerate(iterator):
        # Broadcast the current timestep to the correct shape
        t_curr_tensor = t_curr * torch.ones(
            (x.shape[0],), dtype=x.dtype, device=x.device
        )
        dt = t_prev - t_curr  # we solve backwards in our formulation
        
        # Get velocity at current point
        v = model(x, t_curr_tensor, **extra_args)
        
        # Ping pong: add oscillation to the step
        ping_pong_factor = ping_pong_amplitude * torch.sin(torch.tensor(step_idx * 2 * 3.14159 / 4))
        effective_dt = dt * (1 + ping_pong_factor)
        
        # Take the step
        x_new = x + effective_dt * v
        
        # Optional: Add a corrector step for better stability
        if ping_pong_amplitude > 0:
            t_new_tensor = t_prev * torch.ones(
                (x.shape[0],), dtype=x.dtype, device=x.device
            )
            v_corrector = model(x_new, t_new_tensor, **extra_args)
            x = x + 0.5 * dt * (v + v_corrector)  # Heun's method style correction
        else:
            x = x_new
        
        # Temperature-based noise injection
        if t_curr > temperature and t_prev <= temperature:
            x_0 = x - v
            x = (1 - t_prev) * x_0 + t_prev * noise
            
    return x

@torch.no_grad()
def sample(model, x, steps, eta, prog_bar=False, **extra_args):
    """Draws samples from a model given starting noise. v-diffusion"""
    ts = x.new_ones([x.shape[0]])
    origin_dtype = x.dtype
    # Create the noise schedule
    t = torch.linspace(1, 0, steps + 1)[:-1]

    alphas, sigmas = get_alphas_sigmas(t)

    # The sampling loop
    bar = trange if prog_bar else range
    for i in bar(steps):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            v = model(x, ts * t[i], **extra_args).float()

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < steps - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * adjusted_sigma
            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma

    # If we are on the last timestep, output the denoised image
    return pred.to(origin_dtype)

# Soft mask inpainting is just shrinking hard (binary) mask inpainting
# Given a float-valued soft mask (values between 0 and 1), get the binary mask for this particular step
def get_bmask(i, steps, mask):
    strength = (i+1)/(steps)
    # convert to binary mask
    bmask = torch.where(mask<=strength,1,0)
    return bmask

def make_cond_model_fn(model, cond_fn):
    def cond_model_fn(x, sigma, **kwargs):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            denoised = model(x, sigma, **kwargs)
            cond_grad = cond_fn(x, sigma, denoised=denoised, **kwargs).detach()
            cond_denoised = denoised.detach() + cond_grad * K.utils.append_dims(sigma**2, x.ndim)
        return cond_denoised
    return cond_model_fn

# Uses k-diffusion from https://github.com/crowsonkb/k-diffusion
# init_data is init_audio as latents (if this is latent diffusion)
# For sampling, set both init_data and mask to None
# For variations, set init_data 
# For inpainting, set both init_data & mask 
def sample_k(
        model_fn, 
        noise, 
        init_data=None,
        mask=None,
        steps=100, 
        sampler_type="dpmpp-2m-sde", 
        sigma_min=0.5, 
        sigma_max=50, 
        rho=1.0, device="cuda", 
        callback=None, 
        cond_fn=None,
        **extra_args
    ):

    denoiser = K.external.VDenoiser(model_fn)

    if cond_fn is not None:
        denoiser = make_cond_model_fn(denoiser, cond_fn)

    # Make the list of sigmas. Sigma values are scalars related to the amount of noise each denoising step has
    sigmas = K.sampling.get_sigmas_polyexponential(steps, sigma_min, sigma_max, rho, device=device)
    # Scale the initial noise by sigma 
    noise = noise * sigmas[0]

    wrapped_callback = callback

    if mask is None and init_data is not None:
        # VARIATION (no inpainting)
        # set the initial latent to the init_data, and noise it with initial sigma
        x = init_data + noise 
    elif mask is not None and init_data is not None:
        # INPAINTING
        bmask = get_bmask(0, steps, mask)
        # initial noising
        input_noised = init_data + noise
        # set the initial latent to a mix of init_data and noise, based on step 0's binary mask
        x = input_noised * bmask + noise * (1-bmask)
        # define the inpainting callback function (Note: side effects, it mutates x)
        # See https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py#L596C13-L596C105
        # callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        # This is called immediately after `denoised = model(x, sigmas[i] * s_in, **extra_args)`
        def inpainting_callback(args):
            i = args["i"]
            x = args["x"]
            sigma = args["sigma"]
            #denoised = args["denoised"]
            # noise the init_data input with this step's appropriate amount of noise
            input_noised = init_data + torch.randn_like(init_data) * sigma
            # shrinking hard mask
            bmask = get_bmask(i, steps, mask)
            # mix input_noise with x, using binary mask
            new_x = input_noised * bmask + x * (1-bmask)
            # mutate x
            x[:,:,:] = new_x[:,:,:]
        # wrap together the inpainting callback and the user-submitted callback. 
        if callback is None: 
            wrapped_callback = inpainting_callback
        else:
            wrapped_callback = lambda args: (inpainting_callback(args), callback(args))
    else:
        # SAMPLING
        # set the initial latent to noise
        x = noise


    with torch.cuda.amp.autocast():
        if sampler_type == "k-heun":
            return K.sampling.sample_heun(denoiser, x, sigmas, disable=False, callback=wrapped_callback, extra_args=extra_args)
        elif sampler_type == "k-lms":
            return K.sampling.sample_lms(denoiser, x, sigmas, disable=False, callback=wrapped_callback, extra_args=extra_args)
        elif sampler_type == "k-dpmpp-2s-ancestral":
            return K.sampling.sample_dpmpp_2s_ancestral(denoiser, x, sigmas, disable=False, callback=wrapped_callback, extra_args=extra_args)
        elif sampler_type == "k-dpm-2":
            return K.sampling.sample_dpm_2(denoiser, x, sigmas, disable=False, callback=wrapped_callback, extra_args=extra_args)
        elif sampler_type == "k-dpm-fast":
            return K.sampling.sample_dpm_fast(denoiser, x, sigma_min, sigma_max, steps, disable=False, callback=wrapped_callback, extra_args=extra_args)
        elif sampler_type == "k-dpm-adaptive":
            return K.sampling.sample_dpm_adaptive(denoiser, x, sigma_min, sigma_max, rtol=0.01, atol=0.01, disable=False, callback=wrapped_callback, extra_args=extra_args)
        elif sampler_type == "dpmpp-2m-sde":
            return K.sampling.sample_dpmpp_2m_sde(denoiser, x, sigmas, disable=False, callback=wrapped_callback, extra_args=extra_args)
        elif sampler_type == "dpmpp-3m-sde":
            return K.sampling.sample_dpmpp_3m_sde(denoiser, x, sigmas, disable=False, callback=wrapped_callback, extra_args=extra_args)

# Uses discrete Euler sampling for rectified flow models
# init_data is init_audio as latents (if this is latent diffusion)
# For sampling, set both init_data and mask to None
# For variations, set init_data 
# For inpainting, set both init_data & mask 
def sample_rf(
        model_fn, 
        noise, 
        init_data=None,
        steps=100, 
        sigma_max=1,
        device="cuda", 
        callback=None, 
        cond_fn=None,
        **extra_args
    ):

    if sigma_max > 1:
        sigma_max = 1

    if cond_fn is not None:
        denoiser = make_cond_model_fn(denoiser, cond_fn)

    wrapped_callback = callback

    if init_data is not None:
        # VARIATION (no inpainting)
        # Interpolate the init data and the noise for init audio
        x = init_data * (1 - sigma_max) + noise * sigma_max
    else:
        # SAMPLING
        # set the initial latent to noise
        x = noise

    with torch.cuda.amp.autocast():
        # TODO: Add callback support
        #return sample_discrete_euler(model_fn, x, steps, sigma_max, callback=wrapped_callback, **extra_args)
        return sample_discrete_euler(model_fn, x, steps, sigma_max, **extra_args)