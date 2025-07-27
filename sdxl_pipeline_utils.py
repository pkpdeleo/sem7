# sdxl_pipeline_utils.py
import torch
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLImg2ImgPipeline, # Needed for refiner loading
    ControlNetModel,
    AutoencoderKL,
    UniPCMultistepScheduler
)
from controlnet_aux import CannyDetector, MidasDetector # Add others like OpenPoseDetector if needed
import time
import gc # Garbage collector

# Import config settings
from config import (
    DEVICE, DTYPE_VAE, DTYPE_CONTROL, DTYPE_UNET, DTYPE_REFINER,
    PREPROCESSOR_ANNOTATOR_REPO
)

def get_sdxl_controlnet_preprocessor(control_type):
    """Gets the appropriate preprocessor instance for a given control type."""
    print(f"Loading preprocessor for control type: {control_type}")
    try:
        if control_type == "canny":
            return CannyDetector()
        elif control_type == "depth":
            # Midas uses external annotator models
            print(f"  Loading MidasDetector (from {PREPROCESSOR_ANNOTATOR_REPO})")
            # Model loading within the detector happens on first use or via from_pretrained
            detector = MidasDetector.from_pretrained(PREPROCESSOR_ANNOTATOR_REPO)
            # Move detector's internal model to CPU initially if offloading, to be moved later
            if hasattr(detector, 'model') and hasattr(detector.model, 'to'):
                 detector.model.to('cpu') # Keep preprocessor models on CPU initially
            print("  MidasDetector loaded.")
            return detector
        # Add other control types here, e.g.:
        # elif control_type == "openpose":
        #     from controlnet_aux import OpenposeDetector
        #     print(f"  Loading OpenposeDetector (from {PREPROCESSOR_ANNOTATOR_REPO})")
        #     detector = OpenposeDetector.from_pretrained(PREPROCESSOR_ANNOTATOR_REPO)
        #     if hasattr(detector, 'body_estimation') and hasattr(detector.body_estimation, 'model') and hasattr(detector.body_estimation.model, 'to'):
        #         detector.body_estimation.model.to('cpu')
        #     if hasattr(detector, 'hand_estimation') and hasattr(detector.hand_estimation, 'model') and hasattr(detector.hand_estimation.model, 'to'):
        #         detector.hand_estimation.model.to('cpu')
        #     # Face detector might not have a complex model needing movement
        #     print("  OpenposeDetector loaded.")
        #     return detector
        else:
            raise ValueError(f"Unsupported control_type for preprocessor: {control_type}")
    except Exception as e:
        print(f"Error loading preprocessor for {control_type}: {e}")
        raise # Re-raise to indicate failure

def load_sdxl_controlnet_pipeline(base_model_id, refiner_model_id, vae_model_id,
                                controlnet_items: list, # List of tuples: (id, type_name)
                                enable_cpu_offload=False, enable_attention_slicing=False):
    """
    Loads the Stable Diffusion XL Multi-ControlNet pipeline with VAE and Refiner.

    Args:
        base_model_id (str): Hugging Face ID for the base SDXL model.
        refiner_model_id (str): Hugging Face ID for the SDXL refiner model.
        vae_model_id (str): Hugging Face ID for the VAE model.
        controlnet_items (list): List of tuples, where each tuple is (controlnet_model_id, controlnet_type_name).
        enable_cpu_offload (bool): Whether to enable model CPU offloading.
        enable_attention_slicing (bool): Whether to enable attention slicing.

    Returns:
        tuple: (StableDiffusionXLControlNetPipeline, StableDiffusionXLImg2ImgPipeline or None)
               Returns the main pipeline and the refiner pipeline (if loaded successfully).
    """
    print("\n--- Loading SDXL Multi-ControlNet Pipeline ---")
    start_time = time.time()

    pipe = None
    refiner_pipe = None
    controlnets = []
    controlnet_model_ids = [item[0] for item in controlnet_items] # Extract IDs for loading

    # 1. Load ControlNets
    print("Loading ControlNet models...")
    for cn_id, cn_type in controlnet_items:
        print(f"  Loading ControlNet ({cn_type}): {cn_id}")
        try:
            # Try loading with safetensors first, falling back to bin
            cn = ControlNetModel.from_pretrained(cn_id, torch_dtype=DTYPE_CONTROL, use_safetensors=True)
            print(f"    Loaded {cn_id} (safetensors)")
            controlnets.append(cn)
        except EnvironmentError: # Safetensors not found or other loading issue
            print(f"    Safetensors not found for {cn_id}, trying .bin")
            try:
                cn = ControlNetModel.from_pretrained(cn_id, torch_dtype=DTYPE_CONTROL, use_safetensors=False)
                print(f"    Loaded {cn_id} (.bin)")
                controlnets.append(cn)
            except Exception as e_bin:
                print(f"    Failed to load {cn_id} (.bin): {e_bin}")
                # Decide how to handle failure: raise error, skip, etc.
                # Raising error is safer to ensure all requested ControlNets load.
                raise RuntimeError(f"Could not load ControlNet model: {cn_id}") from e_bin
        except Exception as e:
            print(f"  Error loading ControlNet {cn_id}: {e}")
            raise RuntimeError(f"Could not load ControlNet model: {cn_id}") from e

    print(f"Loaded {len(controlnets)} ControlNet model(s).")

    # 2. Load VAE
    print(f"Loading VAE: {vae_model_id}")
    try:
        vae = AutoencoderKL.from_pretrained(vae_model_id, torch_dtype=DTYPE_VAE, use_safetensors=True)
        print(f"  Loaded VAE {vae_model_id} (safetensors)")
    except EnvironmentError:
        print(f"  Safetensors not found for VAE {vae_model_id}, trying .bin")
        try:
            vae = AutoencoderKL.from_pretrained(vae_model_id, torch_dtype=DTYPE_VAE, use_safetensors=False)
            print(f"  Loaded VAE {vae_model_id} (.bin)")
        except Exception as e_vae_bin:
            print(f"  Failed to load VAE {vae_model_id} (.bin): {e_vae_bin}")
            raise RuntimeError(f"Could not load VAE model: {vae_model_id}") from e_vae_bin
    except Exception as e_vae:
        print(f"  Error loading VAE {vae_model_id}: {e_vae}")
        raise RuntimeError(f"Could not load VAE model: {vae_model_id}") from e_vae

    # 3. Load Refiner Pipeline (separately, as it's Img2Img based)
    # Load refiner *before* the main pipeline to potentially offload it first if needed
    print(f"Loading Refiner Pipeline: {refiner_model_id}")
    try:
        refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            refiner_model_id,
            vae=vae, # Refiner shares the same VAE
            torch_dtype=DTYPE_REFINER,
            use_safetensors=True,
            variant="fp16" if DTYPE_REFINER == torch.float16 else None,
            add_watermarker=False,
        )
        print(f"  Loaded Refiner {refiner_model_id} (safetensors)")
    except EnvironmentError:
        print(f"  Safetensors not found for Refiner {refiner_model_id}, trying .bin")
        try:
            refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                refiner_model_id,
                vae=vae,
                torch_dtype=DTYPE_REFINER,
                use_safetensors=False,
                variant="fp16" if DTYPE_REFINER == torch.float16 else None,
                add_watermarker=False,
            )
            print(f"  Loaded Refiner {refiner_model_id} (.bin)")
        except Exception as e_refiner_bin:
            print(f"  Failed to load Refiner {refiner_model_id} (.bin): {e_refiner_bin}")
            # Allow proceeding without refiner? Or raise error? Raising is safer.
            raise RuntimeError(f"Could not load Refiner model: {refiner_model_id}") from e_refiner_bin
    except Exception as e_refiner:
        print(f"  Error loading Refiner {refiner_model_id}: {e_refiner}")
        raise RuntimeError(f"Could not load Refiner model: {refiner_model_id}") from e_refiner

    # 4. Load Base ControlNet Pipeline
    print(f"Loading Base SDXL ControlNet Pipeline: {base_model_id}")
    try:
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_model_id,
            controlnet=controlnets, # Pass the list of loaded ControlNet models
            vae=vae,
            torch_dtype=DTYPE_UNET,
            use_safetensors=True,
            variant="fp16" if DTYPE_UNET == torch.float16 else None,
            add_watermarker=False, # Optional: disable watermarker
        )
        print(f"  Loaded Base {base_model_id} (safetensors)")
    except EnvironmentError:
        print(f"  Safetensors not found for Base {base_model_id}, trying .bin")
        try:
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                base_model_id,
                controlnet=controlnets,
                vae=vae,
                torch_dtype=DTYPE_UNET,
                use_safetensors=False,
                variant="fp16" if DTYPE_UNET == torch.float16 else None,
                add_watermarker=False,
            )
            print(f"  Loaded Base {base_model_id} (.bin)")
        except Exception as e_base_bin:
            print(f"  Failed to load Base {base_model_id} (.bin): {e_base_bin}")
            raise RuntimeError(f"Could not load Base SDXL model: {base_model_id}") from e_base_bin
    except Exception as e_base:
        print(f"  Error loading Base {base_model_id}: {e_base}")
        raise RuntimeError(f"Could not load Base SDXL model: {base_model_id}") from e_base

    # 5. Apply Optimizations
    print("Applying optimizations...")
    # Use a faster scheduler (UniPC is a good choice)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # Refiner typically uses the same scheduler type
    if refiner_pipe:
         refiner_pipe.scheduler = UniPCMultistepScheduler.from_config(refiner_pipe.scheduler.config)
    print("  Using UniPCMultistepScheduler.")

    # Optional: Attention Slicing (Saves VRAM, can be slightly slower)
    if enable_attention_slicing:
        print("  Enabling Attention Slicing...")
        pipe.enable_attention_slicing()
        if refiner_pipe:
            refiner_pipe.enable_attention_slicing()

    # CPU Offloading (Most significant VRAM saving)
    if enable_cpu_offload and DEVICE == 'cuda':
        print("  Enabling Model CPU Offload (requires 'accelerate')...")
        try:
            # Offload refiner first if it exists
            if refiner_pipe:
                print("    Offloading Refiner pipeline...")
                refiner_pipe.enable_model_cpu_offload()
                gc.collect() # Collect garbage after potential large model movement
                torch.cuda.empty_cache() # Clear cache
                print("    Refiner offloaded.")
                time.sleep(1) # Small delay might help prevent race conditions

            print("    Offloading Base pipeline...")
            pipe.enable_model_cpu_offload()
            gc.collect()
            torch.cuda.empty_cache()
            print("    Base pipeline offloaded.")
            print("  CPU Offload enabled successfully.")

        except ImportError:
            print("  Warning: 'accelerate' library not found. CPU offload unavailable. Moving models to GPU.")
            enable_cpu_offload = False # Disable flag if library missing
            pipe.to(DEVICE)
            if refiner_pipe: refiner_pipe.to(DEVICE)
        except Exception as e_offload:
            print(f"  Warning: Failed to enable CPU offload: {e_offload}. Moving models to GPU.")
            enable_cpu_offload = False # Disable flag on error
            pipe.to(DEVICE)
            if refiner_pipe: refiner_pipe.to(DEVICE)

    elif DEVICE == 'cuda':
        print(f"  Moving pipeline components to GPU ({DEVICE})...")
        pipe.to(DEVICE)
        if refiner_pipe:
            refiner_pipe.to(DEVICE)
        print("  Pipeline components moved to GPU.")
    else:
        print("  Running on CPU (expect very slow performance).")

    end_time = time.time()
    print(f"--- SDXL Pipeline Loading Complete (Time: {end_time - start_time:.2f} seconds) ---")

    return pipe, refiner_pipe