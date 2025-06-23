import os
import sys
import re
import torch
import torchaudio
import json
import tqdm
from comfy.utils import ProgressBar
import numpy as np
from typing import Tuple, Dict, Any, Optional
import folder_paths
from omegaconf import OmegaConf, DictConfig
from huggingface_hub import hf_hub_download

# Disable flash attention for compatibility
os.environ['DISABLE_FLASH_ATTN'] = "1"
repo_id = "CypressYang/SongBloom"
model_name = "songbloom_full_150s"

# Global flag to track if resolvers are registered
_RESOLVERS_REGISTERED = False

def register_omegaconf_resolvers(cache_dir: str):
    """Register OmegaConf resolvers globally to avoid conflicts"""
    global _RESOLVERS_REGISTERED
    if _RESOLVERS_REGISTERED:
        print("Resolvers already registered, skipping...")
        return
    
    print(f"Registering OmegaConf resolvers with cache_dir: {cache_dir}")
    
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
    from .SongBloom.g2p.lyric_common import key2processor
except ImportError as e:
    print(f"Warning: Could not import SongBloom: {e}")
    SongBloom_Sampler = None
    key2processor = {}
    symbols = []
    LABELS = {}

class SongBloomModelLoader:
    """
    Node to load SongBloom model and VAE
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dtype": (["float32", "bfloat16"], {"default": "bfloat16"}),
            },
            "optional": {
                "lyric_processor": (["pinyin", "phoneme", "none"], {"default": "phoneme"}),
            }
        }
    
    RETURN_TYPES = ("SONGBLOOM_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "audio/songbloom"
    
    def __init__(self):
        self.cache_dir = os.path.join(folder_paths.models_dir , "songbloom")
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def hf_download(self, repo_id: str, model_name: str, force_redownload: bool = False):
        """Download model files from Hugging Face Hub with progress bars"""
        
        download_kwargs = {"force_download": force_redownload} if force_redownload else {}
        
        # File list with descriptions for better progress tracking
        files_to_download = [
            (f"{model_name}.yaml", "Model config"),
            (f"{model_name}.pt", "Model checkpoint"),
            ("stable_audio_1920_vae.json", "VAE config"),
            ("autoencoder_music_dsp1920.ckpt", "VAE checkpoint"),
            ("vocab_g2p.yaml", "G2P vocabulary")
        ]
        
        downloaded_paths = []
        
        try:
            print(f"Downloading model files from {repo_id}...")
            
            pbar = ProgressBar(len(files_to_download))
            for filename, description in files_to_download:
                print(f"\nDownloading {description}: {filename}")
                
                path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=self.cache_dir,
                    **download_kwargs
                )
                
                downloaded_paths.append(path)
                pbar.update(1)
            
            print(f"\n All files downloaded successfully to {self.cache_dir}")
            return tuple(downloaded_paths)
            
        except Exception as e:
            raise RuntimeError(f"Failed to download model files: {e}")
    
    def load_config(self, cfg_file: str) -> DictConfig:
        """Load configuration file with resolvers"""
        # Register resolvers globally
        register_omegaconf_resolvers(self.cache_dir)
        
        # Load the raw YAML first
        raw_cfg = OmegaConf.load(open(cfg_file, 'r'))
        
        # Print the raw configuration to see unresolved values
        #print("Raw YAML configuration (before resolver processing):")
        #print(OmegaConf.to_yaml(raw_cfg))
        
        return raw_cfg
    
    def load_model(self, dtype: str, lyric_processor: str = "phoneme"):
        """Load SongBloom model"""
        if SongBloom_Sampler is None:
            raise RuntimeError("SongBloom package not found. Please ensure it's installed in the node directory.")
        
        try:
            # Download model files
            print(f"Downloading SongBloom model: {model_name}")
            cfg_path, ckpt_path, vae_cfg_path, vae_ckpt_path, g2p_path = self.hf_download(
                repo_id, model_name
            )
            
            # Load configuration
            cfg = self.load_config(cfg_path)
            
            # Override lyric processor in config if specified
            if hasattr(cfg, 'train_dataset'):
                cfg.train_dataset.lyric_processor = lyric_processor
                print(f"Set lyric processor to: {lyric_processor}")
            
            # Print the loaded YAML configuration
            #print("Loaded YAML configuration:")
            #print(OmegaConf.to_yaml(cfg))
            
            # Set dtype
            torch_dtype = torch.float32 if dtype == 'float32' else torch.bfloat16
            
            # Build model
            print("Loading SongBloom model...")
            model = SongBloom_Sampler.build_from_trainer(cfg, strict=True, dtype=torch_dtype)
            
            # Set generation parameters from config
            if hasattr(cfg, 'inference') and cfg.inference:
                model.set_generation_params(**cfg.inference)
            
            # Verify lyric processor is working
            print(f"Lyric processor key: {model.lyric_processor_key}")
            print(f"Available processors: {list(key2processor.keys())}")
            
            print(f"SongBloom model loaded successfully. Sample rate: {model.sample_rate}")
            
            # Store model info for the generation node
            model_info = {
                "model": model,
                "sample_rate": model.sample_rate,
                "frame_rate": model.frame_rate,
                "max_duration": model.max_duration,
                "dtype": torch_dtype,
                "lyric_processor_key": model.lyric_processor_key
            }

            print(model_info)
            
            return (model_info,)
            
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
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.1}),
                "steps": ("INT", {"default": 36, "min": 1, "max": 200, "step": 1}),
                "dit_cfg_type": (["h", "m", "l"], {"default": "h"}),
                "use_sampling": ("BOOLEAN", {"default": True}),
                "top_k": ("INT", {"default": 100, "min": 1, "max": 1000, "step": 1}),
                "max_duration": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 300.0, "step": 1.0}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**32-1}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "generate"
    CATEGORY = "audio/songbloom"
    OUTPUT_NODE = True    
    
    def generate(self, model: Dict[str, Any], lyrics: str, audio: Dict[str, torch.Tensor], 
                cfg_coef: float = 1.5, temperature: float = 0.9, steps: int = 50, dit_cfg_type: str = "h", 
                use_sampling: bool = True, top_k: int = 200, max_duration: float = 30.0, 
                seed: int = -1):
        """Generate music using SongBloom"""
        
        try:
            print("Model keys:", model.keys())
            print("Model 'model' type:", type(model.get("model")))

            songbloom_model = model["model"]
            model_sample_rate = model["sample_rate"]
            model_dtype = model["dtype"]
            
            # Set random seed if specified
            if seed != -1:
                torch.manual_seed(seed)
                np.random.seed(seed)
            
            lyrics = lyrics.replace('\r', ' ').replace('\n', ' , ').lower()
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
            
            # Convert to model dtype
            prompt_wav = prompt_wav.to(dtype=model_dtype)
            
            # Limit prompt duration (10 seconds max as in original code)
            max_prompt_samples = 10 * model_sample_rate
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
                "diff_temp": 0.95,
                "top_k": top_k,
                "penalty_repeat": True,
                "penalty_window": 50,
                "steps": steps,
                "dit_cfg_type": dit_cfg_type,
                "use_sampling": use_sampling,
                "max_frames": max_frames
            }
            
            songbloom_model.set_generation_params(**generation_params)
            
            print(f"Generating music with processed lyrics: {processed_lyrics[:50]}...")
            print(f"Prompt audio shape: {prompt_wav.shape}, Sample rate: {model_sample_rate}")
            print(f"Generation params: {generation_params}")
            
            # Generate music - get latent representation only
            with torch.no_grad():
                latent_seq, token_seq = songbloom_model.diffusion.generate(None, attributes, **generation_params)
            
            # Convert latent to ComfyUI LATENT format
            # latent_seq is expected to be [batch, channels, time]
            if latent_seq.dim() == 2:
                latent_seq = latent_seq.unsqueeze(0)  # Add batch dimension
            
            print(f"Generated latent shape: {latent_seq.shape}")
            
            # ComfyUI LATENT format expects a dictionary with "samples" key
            output_latent = {
                "samples": latent_seq.float().cpu()
            }
            
            return (output_latent,)
            
        except Exception as e:
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
                "model": ("SONGBLOOM_MODEL",),
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
    
    def decode(self, model: Dict[str, Any], latent: Dict[str, torch.Tensor], 
               chunk_size: int = 100, overlap: int = 0):
        """Decode latent representation to audio using chunked processing"""
        
        try:
            songbloom_model = model["model"]
            model_sample_rate = model["sample_rate"]
            
            # Get latent samples
            latent_samples = latent["samples"]  # Shape: [batch, channels, time]
            
            # Move to model device and dtype
            device = next(iter(songbloom_model.diffusion.parameters())).device
            latent_samples = latent_samples.to(device=device, dtype=model["dtype"])
            
            batch_size, channels, time_frames = latent_samples.shape
            
            print(f"Decoding latent shape: {latent_samples.shape} using chunks of size {chunk_size} with overlap {overlap}")
            
            # Validate parameters
            if overlap >= chunk_size:
                overlap = chunk_size // 2
                print(f"Warning: overlap too large, reduced to {overlap}")
            
            # If the sequence is short enough, decode without chunking
            if time_frames <= chunk_size:
                print("Sequence short enough, decoding without chunking")
                with torch.no_grad():
                    decoded_audio = songbloom_model.compression_model.decode(latent_samples)
            else:
                # Chunked decoding
                decoded_chunks = []
                step_size = chunk_size - overlap
                
                for start_idx in range(0, time_frames, step_size):
                    end_idx = min(start_idx + chunk_size, time_frames)
                    
                    print(f"Decoding chunk {start_idx}:{end_idx} ({end_idx - start_idx} frames)")
                    
                    # Extract chunk
                    chunk_latent = latent_samples[:, :, start_idx:end_idx]
                    
                    # Decode chunk
                    with torch.no_grad():
                        chunk_audio = songbloom_model.compression_model.decode(chunk_latent)
                    
                    # Handle overlapping regions
                    if start_idx == 0:
                        # First chunk - keep everything
                        decoded_chunks.append(chunk_audio)
                    else:
                        # Subsequent chunks - remove overlap from beginning
                        # Calculate overlap in audio samples
                        audio_overlap_samples = self._calculate_audio_overlap(
                            overlap, chunk_audio.shape[-1], chunk_size
                        )
                        
                        if audio_overlap_samples > 0 and chunk_audio.shape[-1] > audio_overlap_samples:
                            chunk_audio = chunk_audio[:, :, audio_overlap_samples:]
                        
                        decoded_chunks.append(chunk_audio)
                    
                    # Clear GPU memory
                    del chunk_audio
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Concatenate all chunks
                print(f"Concatenating {len(decoded_chunks)} decoded chunks")
                decoded_audio = torch.cat(decoded_chunks, dim=-1)
                
                # Clean up
                del decoded_chunks
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Convert output to ComfyUI audio format
            # decoded_audio is expected to be [batch, channels, samples]
            if decoded_audio.dim() == 2:
                decoded_audio = decoded_audio.unsqueeze(0)  # Add batch dimension
            
            # Ensure float32 for ComfyUI compatibility
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


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "SongBloomModelLoader": SongBloomModelLoader,
    "SongBloomGenerate": SongBloomGenerate,
    "SongBloomDecoder": SongBloomDecoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SongBloomModelLoader": "SongBloom Model Loader",
    "SongBloomGenerate": "SongBloom Generate",
    "SongBloomDecoder": "SongBloom Decoder",
}