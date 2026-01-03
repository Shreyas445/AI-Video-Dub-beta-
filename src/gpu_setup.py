import os
import sys
import ctypes
import importlib.util

def setup_gpu_environment():
    print(f"DEBUG: Setting up GPU environment...")
    print(f"DEBUG: Python: {sys.version}")
    print(f"DEBUG: Platform: {sys.platform}")

    # 1. Register DLL Directories for nvidia-cudnn and nvidia-cublas
    try:
        # Helper to find and register package paths
        def register_package_dlls(package_name):
            spec = importlib.util.find_spec(package_name)
            if spec and spec.submodule_search_locations:
                package_dir = list(spec.submodule_search_locations)[0]
                print(f"DEBUG: {package_name} found at: {package_dir}")
                
                if os.name == 'nt':
                    try:
                        os.add_dll_directory(package_dir)
                        # Specific subdirs often found in nvidia packages
                        bin_dir = os.path.join(package_dir, "bin")
                        lib_dir = os.path.join(package_dir, "lib")
                        if os.path.exists(bin_dir):
                            os.add_dll_directory(bin_dir)
                        if os.path.exists(lib_dir):
                            os.add_dll_directory(lib_dir)
                        print(f"DEBUG: Added DLL directories for {package_name}")
                        return package_dir
                    except Exception as e:
                        print(f"DEBUG: Error adding DLL directories for {package_name}: {e}")
            else:
                print(f"DEBUG: {package_name} path not found.")
            return None

        cudnn_dir = register_package_dlls("nvidia.cudnn")
        cublas_dir = register_package_dlls("nvidia.cublas")
        
        # 2. Register Torch Lib as well (Crucial for zlibwapi, cudart, nvrtc)
        import torch
        torch_lib_path = os.path.join(os.path.dirname(os.path.abspath(torch.__file__)), "lib")
        if os.path.exists(torch_lib_path):
             os.add_dll_directory(torch_lib_path)
             print(f"DEBUG: Added Torch Lib to DLL search: {torch_lib_path}")
        
    except ImportError:
        print("DEBUG: NVIDIA pip packages NOT found.")
    except Exception as e:
        print(f"DEBUG: Unexpected error during directory registration: {e}")

    # 3. Explicitly Preload DLLs via ctypes (The 'Nuclear Option')
    print("\nDEBUG: Attempting to load CUDA/CuDNN DLLs via ctypes...")
    
    # Common DLL names for CUDA 11.x / CuDNN 8.x
    dlls_to_check = [
        "zlibwapi.dll",          # Critical dependency often missing
        "cudart64_110.dll",      # CUDA Runtime
        "cublas64_11.dll",       # CuBLAS
        "cublasLt64_11.dll",     # CuBLAS Lt
        "cudnn64_8.dll",         # CuDNN Main
        "cudnn_ops_infer64_8.dll", # CuDNN Ops
        "cudnn_cnn_infer64_8.dll"  # CuDNN CNN
    ]

    # Also try to find zlibwapi specifically in the nvidia.cudnn directory if it exists there
    # (Sometimes it's bundled, sometimes not. If not, we might need to rely on Torch's copy again or download it)
    
    for dll in dlls_to_check:
        try:
            # Try loading by name (relies on add_dll_directory having worked)
            ctypes.CDLL(dll)
            print(f"DEBUG: [OK] {dll} loaded.")
        except Exception as e:
            print(f"DEBUG: [FAIL] {dll} failed to load directly: {e}")
            
            # Fallback: Try finding it in the package directories specifically
            found_fallback = False
            if cudnn_dir:
                guess_path = os.path.join(cudnn_dir, "bin", dll) # bin is common for DLLs
                if not os.path.exists(guess_path):
                     guess_path = os.path.join(cudnn_dir, "lib", dll)
                if not os.path.exists(guess_path):
                     guess_path = os.path.join(cudnn_dir, dll)
                     
                if os.path.exists(guess_path):
                    try:
                        ctypes.CDLL(guess_path)
                        print(f"DEBUG: [OK] {dll} loaded from absolute path: {guess_path}")
                        found_fallback = True
                    except Exception as ex:
                        print(f"DEBUG: [FAIL] Failed to load from absolute path {guess_path}: {ex}")

            if not found_fallback and cublas_dir and 'cublas' in dll:
                 # Similar checks for cublas
                 pass # Logic is similar, but let's see if generic load works first

if __name__ == "__main__":
    setup_gpu_environment()
