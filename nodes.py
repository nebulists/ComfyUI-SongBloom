import os
import sys
import re
import torch
import gc
import torchaudio
import json
import tqdm
from comfy.utils import ProgressBar
import numpy as np
from typing import Tuple, Dict, Any, Optional
import folder_paths
from omegaconf import OmegaConf, DictConfig

# Import ComfyUI interruption exception
try:
    from comfy.model_management import InterruptProcessingException
except ImportError:
    # Fallback if not available
    InterruptProcessingException = Exception

# Disable flash attention for compatibility
os.environ['DISABLE_FLASH_ATTN'] = "1"
model_name = "songbloom_full_150s"

# Global flag to track if resolvers are registered
_RESOLVERS_REGISTERED = False

# Global cache for loaded models
_MODEL_CACHE = {}

def register_omegaconf_resolvers(cache_dir: str):
    """Register OmegaConf resolvers globally to avoid conflicts"""
    global _RESOLVERS_REGISTERED
    
    print(f"Registering OmegaConf resolvers with cache_dir: {cache_dir}")
    
    # Clear existing resolvers if they exist
    try:
        OmegaConf.clear_resolver("dynamic_path")
        print("Cleared existing dynamic_path resolver")
    except:
        pass
    
    resolvers = {
        "eval": lambda x: eval(x),
        "concat": lambda *x: [xxx for xx in x for xxx in xx],
        "get_fname": lambda x: os.path.splitext(os.path.basename(x))[0],
        "load_yaml": lambda x: OmegaConf.load(x),
        "dynamic_path": lambda x: x.replace("???", cache_dir)
    }
    
    for name, resolver in resolvers.items():
        try:
            OmegaConf.register_new_resolver(name, resolver)
            print(f"Registered resolver: {name}")
        except ValueError:
            # Resolver already exists, skip
            print(f"Resolver {name} already exists, skipping...")
            pass
    
    _RESOLVERS_REGISTERED = True

try:
    from .SongBloom.models.songbloom.songbloom_pl import SongBloom_Sampler
except ImportError as e:
    print(f"Warning: Could not import SongBloom: {e}")
    SongBloom_Sampler = None
    symbols = []
    LABELS = {}

# Helper to wrap a raw SongBloom model into the expected dict format

def wrap_songbloom_model(model):
    # If already a dict, return as is
    if isinstance(model, dict) and "model" in model:
        return model
    # If it's a SongBloom_Sampler or SongBloom_PL, wrap it
    # Try to extract metadata, fallback to defaults if missing
    sample_rate = getattr(model, "sample_rate", 48000)
    frame_rate = getattr(model, "frame_rate", 25)
    max_duration = getattr(model, "max_duration", 30.0)
    dtype = getattr(model, "dtype", torch.float32)
    lyric_processor_key = getattr(model, "lyric_processor_key", "phoneme")
    return {
        "model": model,
        "sample_rate": sample_rate,
        "frame_rate": frame_rate,
        "max_duration": max_duration,
        "dtype": dtype,
        "lyric_processor_key": lyric_processor_key,
    }

class SongBloomModelLoader:
    """
    Node to load SongBloom model and VAE together
    """
    
    @staticmethod
    def _list_safetensors():
        """List available .safetensors files in the checkpoints directory."""
        checkpoints_dir = os.path.join(folder_paths.models_dir, "checkpoints")
        if not os.path.exists(checkpoints_dir):
            return []
        return [f for f in os.listdir(checkpoints_dir) if f.endswith('.safetensors')]

    @staticmethod
    def _list_vae_safetensors():
        """List available .safetensors files in the VAE directory."""
        vae_dir = os.path.join(folder_paths.models_dir, "vae")
        if not os.path.exists(vae_dir):
            return []
        return [f for f in os.listdir(vae_dir) if f.endswith('.safetensors')]

    @classmethod
    def INPUT_TYPES(cls):
        safetensor_files = cls._list_safetensors()
        safetensor_files = safetensor_files if safetensor_files else ["None found"]
        vae_files = cls._list_vae_safetensors()
        vae_files = vae_files if vae_files else ["None found"]
        return {
            "required": {
                "checkpoint": (safetensor_files, {"default": safetensor_files[0], "tooltip": "Pick a .safetensors checkpoint from models/checkpoints."}),
                "vae": (vae_files, {"default": vae_files[0], "tooltip": "Pick a .safetensors VAE from models/vae."}),
                "dtype": (["float32", "bfloat16"], {"default": "bfloat16"}),
                "prompt_len": ("INT", {"default": 10, "min": 1, "max": 60, "step": 1}),
            },
            "optional": {
                "lyric_processor": (["pinyin", "phoneme", "none"], {"default": "phoneme"}),
            }
        }
    
    RETURN_TYPES = ("SONGBLOOM_MODEL", "VAE",)
    RETURN_NAMES = ("model", "vae",)
    FUNCTION = "load_model"
    CATEGORY = "audio/songbloom"
    
    def __init__(self):
        self.cache_dir = os.path.join(folder_paths.models_dir , "songbloom")
        os.makedirs(self.cache_dir, exist_ok=True)
        # Set config directory for resolvers
        self.config_dir = os.path.join(os.path.dirname(__file__), "SongBloom", "config")
    
    def load_config(self, cfg_file: str) -> DictConfig:
        """Load configuration file with resolvers"""
        register_omegaconf_resolvers(self.config_dir)
        raw_cfg = OmegaConf.load(open(cfg_file, 'r'))
        return raw_cfg
    
    def load_vae(self, vae: str):
        """Load VAE from safetensors file"""
        try:
            from .SongBloom.models.vae_frontend import StableVAE
        except ImportError as e:
            raise RuntimeError(f"Could not import SongBloom VAE code: {e}")
        vae_dir = os.path.join(folder_paths.models_dir, "vae")
        vae_path = os.path.join(vae_dir, vae)
        config_path = os.path.join(self.config_dir, "stable_audio_1920_vae.json")
        vae = StableVAE(vae_ckpt=None, vae_cfg=config_path, sr=48000, vae_safetensor_path=vae_path)
        return vae
    
    def load_model(self, dtype: str, checkpoint: str, vae: str, lyric_processor: str = "phoneme", prompt_len: int = 10, **kwargs):
        try:
            # Clear all cached models when loader is executed
            global _MODEL_CACHE
            _MODEL_CACHE.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Cleared all cached SongBloom models")
            
            print(f"Preparing to load SongBloom model: {model_name}")
            
            # Create a unique cache key for this model configuration
            cache_key = f"{checkpoint}_{vae}_{dtype}_{lyric_processor}_{prompt_len}"
            
            # Check if model is already cached
            if cache_key in _MODEL_CACHE:
                print(f"Using cached model for key: {cache_key}")
                cached_result = _MODEL_CACHE[cache_key]
                return (cached_result["model_config"], cached_result["vae"])
            
            # Load VAE first
            print(f"Loading VAE: {vae}")
            vae = self.load_vae(vae)
            
            # All files are local now
            # Use config_dir from instance variable
            cfg_path = os.path.join(self.config_dir, f"{model_name}.yaml")
            vae_cfg_path = os.path.join(self.config_dir, "stable_audio_1920_vae.json")
            g2p_path = os.path.join(self.config_dir, "vocab_g2p.yaml")
            model_safetensor = os.path.join(folder_paths.models_dir, "checkpoints", checkpoint)
            safetensor_path = model_safetensor
            
            cfg = self.load_config(cfg_path)
            if hasattr(cfg, 'train_dataset'):
                cfg.train_dataset.lyric_processor = lyric_processor
            
            # Convert dtype string to torch dtype
            if dtype == 'bfloat16':
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32
            
            # Build the model
            if SongBloom_Sampler is None:
                raise RuntimeError("SongBloom package not found. Please ensure it's installed in the node directory.")
            
            print(f"Loading new model with dtype: {dtype}")
            model = SongBloom_Sampler.build_from_trainer(cfg, vae=vae, strict=True, dtype=torch_dtype, safetensor_path=safetensor_path, external_vae=True)
            model.prompt_duration = cfg.sr * prompt_len
            if hasattr(cfg, 'inference') and cfg.inference:
                model.set_generation_params(**cfg.inference)
            
            # Create model config with loaded model
            model_config = {
                "model": model,
                "cfg": cfg,
                "vae_cfg_path": vae_cfg_path,
                "g2p_path": g2p_path,
                "safetensor_path": safetensor_path,
                "dtype": dtype,
                "prompt_len": prompt_len
            }
            
            # Cache both model config and VAE
            _MODEL_CACHE[cache_key] = {
                "model_config": model_config,
                "vae": vae
            }
            print(f"Cached model and VAE with key: {cache_key}")
            
            return (model_config, vae)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to load SongBloom model: {e}")


class SongBloomGenerate:
    """
    Node to generate music using SongBloom model
    """
      
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SONGBLOOM_MODEL",),
                "lyrics": ("STRING", {"multiline": True, "default": "Hello world, this is a test song"}),
                "audio": ("AUDIO",),
            },
            "optional": {
                "cfg_coef": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.5, "step": 0.01}),
                "diff_temp": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.5, "step": 0.01}),
                "steps": ("INT", {"default": 36, "min": 1, "max": 200, "step": 1}),
                "use_sampling": ("BOOLEAN", {"default": True}),
                "dit_cfg_type": (["h", "global"], {"default": "h"}),
                "top_k": ("INT", {"default": 100, "min": 1, "max": 1000, "step": 1}),
                "max_duration": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 300.0, "step": 1.0}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**32-1}),
                "sampling_method": (["discrete_temperature", "spiral", "pingpong"], {"default": "discrete_temperature"}),
                "spiral_num_paths": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1, "tooltip": "Number of spiral paths (only used with spiral sampling)"}),
                "spiral_strength": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Spiral strength (only used with spiral sampling)"}),
                "spiral_combination": (["weighted_avg", "best_path", "ensemble"], {"default": "weighted_avg", "tooltip": "How to combine spiral paths (only used with spiral sampling)"}),
                "pingpong_amplitude": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Ping-pong oscillation amplitude (only used with pingpong sampling)"}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "generate"
    CATEGORY = "audio/songbloom"
    OUTPUT_NODE = True    

    def generate(self, model: dict, lyrics: str, audio: dict, 
                cfg_coef: float = 1.5, temperature: float = 0.9, diff_temp: float = 0.95, steps: int = 50, 
                use_sampling: bool = True, dit_cfg_type: str = "h", top_k: int = 200,  max_duration: float = 30.0, 
                seed: int = -1, sampling_method: str = "discrete_temperature", 
                spiral_num_paths: int = 3, spiral_strength: float = 0.1, spiral_combination: str = "weighted_avg",
                pingpong_amplitude: float = 0.1):
        """Generate music using SongBloom"""
        try:
            # Use the cached model from the loader
            songbloom_model = model["model"]
            print("Model type:", type(songbloom_model))
            model_sample_rate = songbloom_model.sample_rate
            
            # Use dtype from model_config, not from VAE
            dtype = model.get('dtype', 'float32')
            if dtype == 'float32':
                model_dtype = torch.float32
            elif dtype == 'bfloat16':
                model_dtype = torch.bfloat16
            else:
                model_dtype = torch.float32
                
            # Set random seed if specified
            if seed != -1:
                torch.manual_seed(seed)
                np.random.seed(seed)
                
            #lyrics = lyrics.replace('\r', ' ').replace('\n', ' , ').lower()
            lyrics = '\n'.join(line.strip() for line in lyrics.split('\n'))
            lyrics = re.sub(r'\n\s*\n', r' , ', lyrics)
            lyrics = re.sub(r'(?<!\])\n', ', ', lyrics)
            lyrics = lyrics.replace("[", " [").replace("]", "] ")
            lyrics = re.sub(r' {2,}', ' ', lyrics)
            processed_lyrics = songbloom_model._process_lyric(lyrics)
            
            # Process input audio and prepare attributes
            waveform = audio["waveform"]  # Shape: [batch, channels, samples]
            sample_rate = audio["sample_rate"]
            # Take first item from batch and convert to model's expected format
            prompt_wav = waveform[0]  # Shape: [channels, samples]
            # Resample if necessary
            if sample_rate != model_sample_rate:
                prompt_wav = torchaudio.functional.resample(
                    prompt_wav, sample_rate, model_sample_rate
                )
            # Convert to mono if stereo
            if prompt_wav.shape[0] > 1:
                prompt_wav = prompt_wav.mean(dim=0, keepdim=True)
            # Convert to correct dtype from loader
            prompt_wav = prompt_wav.to(dtype=model_dtype)
            # Limit prompt duration (10 seconds max as in original code)
            max_prompt_samples = songbloom_model.prompt_duration
            print(f"\nprompt limit: {max_prompt_samples}")
            if prompt_wav.shape[-1] > max_prompt_samples:
                prompt_wav = prompt_wav[..., :max_prompt_samples]
            # Prepare attributes for generation
            attributes, _ = songbloom_model._prepare_tokens_and_attributes(
                conditions={"lyrics": [processed_lyrics], "prompt_wav": [prompt_wav]}, 
                prompt=None, prompt_tokens=None
            )
            # Set generation parameters
            max_frames = int(max_duration * songbloom_model.frame_rate)
            generation_params = {
                "cfg_coef": cfg_coef,
                "temp": temperature,
                "diff_temp": diff_temp,
                "top_k": top_k,
                "penalty_repeat": True,
                "penalty_window": 50,
                "steps": steps,
                "dit_cfg_type": dit_cfg_type,
                "use_sampling": use_sampling,
                "max_frames": max_frames,
                "sampling_method": sampling_method,
                "spiral_num_paths": spiral_num_paths,
                "spiral_strength": spiral_strength,
                "spiral_combination": spiral_combination,
                "pingpong_amplitude": pingpong_amplitude
            }
            songbloom_model.set_generation_params(**generation_params)
            print(f"Generating music with processed lyrics: {processed_lyrics[:50]}...")
            print(f"Prompt audio shape: {prompt_wav.shape}, Sample rate: {model_sample_rate}")
            print(f"Generation params: {generation_params}")
            # Generate music - get latent representation only
            with torch.no_grad():
                latent_seq, token_seq = songbloom_model.diffusion.generate(None, attributes, **generation_params)
            # Convert latent to ComfyUI LATENT format
            if latent_seq.dim() == 2:
                latent_seq = latent_seq.unsqueeze(0)  # Add batch dimension
            print(f"Generated latent shape: {latent_seq.shape}")
            output_latent = {
                "samples": latent_seq.float().cpu()
            }
            return (output_latent,)
        except Exception as e:
            # Check if this is an interruption-related exception
            if isinstance(e, InterruptProcessingException):
                # Re-raise interruption exceptions without wrapping them
                raise
            elif isinstance(e, KeyboardInterrupt):
                # Re-raise keyboard interrupts without wrapping them
                raise
            else:
                # For other exceptions, wrap them in RuntimeError with proper message
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Failed to generate music: {e}")


class SongBloomDecoder:
    """
    Node to decode SongBloom latents to audio with chunked processing for memory efficiency
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("VAE",),
                "latent": ("LATENT",),
            },
            "optional": {
                "chunk_size": ("INT", {"default": 1000, "min": 10, "max": 5000, "step": 10, 
                                     "tooltip": "Number of latent frames to process at once. Lower values use less memory."}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "decode"
    CATEGORY = "audio/songbloom"
    
    def decode(self, model: Any, latent: Dict[str, torch.Tensor], 
               chunk_size: int = 100, overlap: int = 0):
        """Decode latent representation to audio using chunked processing"""
        try:
            # model is a StableVAE instance
            vae_model = model  # No wrapping needed
            # Assume sample_rate is stored in the VAE or passed in
            model_sample_rate = getattr(vae_model, "sr", 48000)

            # Get latent samples
            latent_samples = latent["samples"]  # Shape: [batch, channels, time]

            # Move to model device and dtype
            device = next(vae_model.parameters()).device
            vae_dtype = next(vae_model.parameters()).dtype
            latent_samples = latent_samples.to(device=device, dtype=vae_dtype)

            batch_size, channels, time_frames = latent_samples.shape

            print(f"Decoding latent shape: {latent_samples.shape} using chunks of size {chunk_size} with overlap {overlap}")

            if overlap >= chunk_size:
                overlap = chunk_size // 2
                print(f"Warning: overlap too large, reduced to {overlap}")

            if time_frames <= chunk_size:
                print("Sequence short enough, decoding without chunking")
                with torch.no_grad():
                    decoded_audio = vae_model.decode(latent_samples)
            else:
                decoded_chunks = []
                step_size = chunk_size - overlap

                for start_idx in range(0, time_frames, step_size):
                    end_idx = min(start_idx + chunk_size, time_frames)
                    print(f"Decoding chunk {start_idx}:{end_idx} ({end_idx - start_idx} frames)")
                    chunk_latent = latent_samples[:, :, start_idx:end_idx]
                    with torch.no_grad():
                        chunk_audio = vae_model.decode(chunk_latent)
                    if start_idx == 0:
                        decoded_chunks.append(chunk_audio)
                    else:
                        audio_overlap_samples = self._calculate_audio_overlap(
                            overlap, chunk_audio.shape[-1], chunk_size
                        )
                        if audio_overlap_samples > 0 and chunk_audio.shape[-1] > audio_overlap_samples:
                            chunk_audio = chunk_audio[:, :, audio_overlap_samples:]
                        decoded_chunks.append(chunk_audio)
                    del chunk_audio

                print(f"Concatenating {len(decoded_chunks)} decoded chunks")
                decoded_audio = torch.cat(decoded_chunks, dim=-1)
                del decoded_chunks

            if decoded_audio.dim() == 2:
                decoded_audio = decoded_audio.unsqueeze(0)  # Add batch dimension

            decoded_audio = decoded_audio.float().cpu()
            print(f"Final decoded audio shape: {decoded_audio.shape}")

            output_audio = {
                "waveform": decoded_audio,
                "sample_rate": model_sample_rate
            }

            return (output_audio,)

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to decode latent: {e}")
    
    def _calculate_audio_overlap(self, latent_overlap: int, audio_samples: int, latent_chunk_size: int) -> int:
        """Calculate the number of audio samples corresponding to latent overlap"""
        # This assumes a linear relationship between latent frames and audio samples
        # You may need to adjust this based on the specific VAE architecture
        if latent_chunk_size > 0:
            samples_per_latent_frame = audio_samples / latent_chunk_size
            return int(latent_overlap * samples_per_latent_frame)
        return 0


class SongBloomVAEEncoder:
    """
    Node to encode audio to SongBloom latents using the VAE (with optional chunked processing)
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("VAE",),
                "audio": ("AUDIO",),
            },
            "optional": {
                "chunked": ("BOOLEAN", {"default": False, "tooltip": "Enable chunked encoding for long audio."}),
                "chunk_size": ("INT", {"default": 1000, "min": 10, "max": 5000, "step": 10, "tooltip": "Chunk size in latent frames (only used if chunked)."}),
                "overlap": ("INT", {"default": 0, "min": 0, "max": 500, "step": 1, "tooltip": "Overlap in latent frames (only used if chunked)."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "encode"
    CATEGORY = "audio/songbloom"

    def encode(self, model: Any, audio: dict, chunked: bool = False, chunk_size: int = 1000, overlap: int = 0):
        try:
            vae_model = model
            waveform = audio["waveform"]  # [batch, channels, samples]
            sample_rate = audio["sample_rate"]
            # Resample to 48000Hz if needed
            target_sr = 48000
            if sample_rate != target_sr:
                import torchaudio
                # waveform: [batch, channels, samples]
                # Resample each batch separately
                resampled = []
                for w in waveform:
                    resampled.append(torchaudio.functional.resample(w, sample_rate, target_sr))
                # Pad/truncate to the same length
                min_len = min(w.shape[-1] for w in resampled)
                resampled = [w[..., :min_len] for w in resampled]
                waveform = torch.stack(resampled, dim=0)
                sample_rate = target_sr
            # Move to model device and dtype
            device = next(vae_model.parameters()).device
            vae_dtype = next(vae_model.parameters()).dtype
            waveform = waveform.to(device=device, dtype=vae_dtype)
            # Use encode_audio for chunked, encode for non-chunked
            if chunked:
                latent = vae_model.vae.encode_audio(waveform, chunked=True, chunk_size=chunk_size, overlap=overlap)
            else:
                latent = vae_model.encode(waveform)
            if isinstance(latent, tuple):
                latent = latent[0]
            if latent.dim() == 2:
                latent = latent.unsqueeze(0)
            output_latent = {"samples": latent.float().cpu()}
            return (output_latent,)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to encode audio to latent: {e}")


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "SongBloomModelLoader": SongBloomModelLoader,
    "SongBloomGenerate": SongBloomGenerate,
    "SongBloomDecoder": SongBloomDecoder,
    "SongBloomVAEEncoder": SongBloomVAEEncoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SongBloomModelLoader": "SongBloom Model Loader",
    "SongBloomGenerate": "SongBloom Generate",
    "SongBloomDecoder": "SongBloom Decoder",
    "SongBloomVAEEncoder": "SongBloom VAE Encoder",
}