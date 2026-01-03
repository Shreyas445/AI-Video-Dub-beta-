import torch
import gc
import os

def clear_gpu_memory():
    """
    Clears GPU memory by collecting garbage and emptying the CUDA cache.
    This is critical for running multiple large models on a 6GB VRAM GPU.
    """
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("GPU memory cleared.")

def get_output_path(input_path, suffix, output_dir=None):
    """
    Generates an output path based on the input filename.
    """
    if output_dir is None:
        output_dir = os.path.dirname(input_path)
    
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(output_dir, f"{base_name}_{suffix}")
