# generation.py
import torch
from PIL import Image, ImageOps
import numpy as np
import time
import os
import gc

# Import config settings
from config import DEVICE

def preprocess_control_image(image_pil, preprocessor, control_type, target_size, device_for_preprocessor):
    """Applies preprocessing and resizes the control image."""
    print(f"  Preprocessing for {control_type}...")
    # Move preprocessor model to the designated device (GPU or CPU if offloading)
    preprocessor_needs_move = False
    original_device = 'cpu' # Default assumption
    if hasattr(preprocessor, 'model') and hasattr(preprocessor.model, 'to'):
        # Check current device before moving
        # Note: device attribute might not exist directly on all models
        try: original_device = next(preprocessor.model.parameters()).device
        except: pass # Ignore if device check fails

        if original_device != device_for_preprocessor:
            preprocessor.model.to(device_for_preprocessor)
            preprocessor_needs_move = True
            # print(f"    Moved {control_type} preprocessor model to {device_for_preprocessor}")

    # Run preprocessing
    # Handle specific args if needed (e.g., Midas resolution)
    if control_type == 'depth' and hasattr(preprocessor, 'config') and 'image_resolution' in preprocessor.config:
         # Example MidasDetector handling (check controlnet_aux docs for specifics)
         control_image = preprocessor(image_pil, detect_resolution=target_size[0], image_resolution=target_size[1])
    else:
         control_image = preprocessor(image_pil) # Default call for Canny etc.

    # Ensure PIL format and RGB
    if isinstance(control_image, np.ndarray):
        control_image = Image.fromarray(control_image)
    control_image = control_image.convert("RGB") # Ensure 3 channels

    print(f"    Raw {control_type} map size: {control_image.size}")

    # --- Explicitly Resize Control Map ---
    if control_image.size != target_size:
        print(f"    Resizing {control_type} map from {control_image.size} to {target_size}...")
        # Use a high-quality resampling filter
        control_image = control_image.resize(target_size, Image.Resampling.LANCZOS)
        print(f"    Resized {control_type} map size: {control_image.size}")

    # Move preprocessor model back to CPU if it was moved and offloading is active
    if preprocessor_needs_move and device_for_preprocessor != 'cpu':
        if 'cuda' in DEVICE and torch.cuda.is_available() and hasattr(torch.cuda, 'get_device_properties') and torch.cuda.get_device_properties(0).total_memory / (1024**3) < 15 : # Example low VRAM check for aggressive offload
             preprocessor.model.to('cpu')
             # print(f"    Moved {control_type} preprocessor model back to CPU")
        elif 'cuda' in DEVICE:
            pass # Keep on GPU if enough VRAM and not specifically offloading

    gc.collect() # Clean up memory after preprocessor use
    if DEVICE == 'cuda': torch.cuda.empty_cache()

    return control_image


@torch.no_grad() # Ensure no gradients are computed during generation
def generate_sdxl_controlled_image(
    pipe, # Base SDXL ControlNet pipeline
    refiner_pipe, # SDXL Refiner pipeline
    preprocessors: dict, # Dict: {'control_type': preprocessor_instance}
    prompt: str,
    negative_prompt: str,
    conditioning_image_pils: list, # List of PIL images for conditioning
    control_types: list, # List of strings ('depth', 'canny', etc.)
    controlnet_scales: list, # List of floats for scale
    output_path: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = None,
    image_resolution: int = 1024,
    refiner_steps_ratio: float = 0.2 # Ratio of steps for refiner
    ):
    """Generates an image using the SDXL ControlNet pipeline and Refiner."""
    print("\n--- Starting SDXL Controlled Image Generation ---")
    print(f"Prompt: {prompt}")
    print(f"Negative Prompt: {negative_prompt}")
    print(f"Output path: {output_path}")
    print(f"Resolution: {image_resolution}x{image_resolution}")
    print(f"Steps: {num_inference_steps}, Guidance: {guidance_scale}")
    print(f"Control Types: {control_types}")
    print(f"Control Scales: {controlnet_scales}")
    print(f"Refiner Steps Ratio: {refiner_steps_ratio}")
    print(f"Seed: {seed}")

    if not (0 < refiner_steps_ratio < 1):
        print("Error: refiner_steps_ratio must be between 0 and 1 (exclusive). Using default 0.2")
        refiner_steps_ratio = 0.2

    if len(conditioning_image_pils) != len(control_types) or len(control_types) != len(controlnet_scales):
        print("Error: Mismatch between number of conditioning images, control types, and scales.")
        print(f"  Images: {len(conditioning_image_pils)}, Types: {len(control_types)}, Scales: {len(controlnet_scales)}")
        return None, []

    # --- 1. Prepare Conditioning Images ---
    control_images_processed = []
    control_images_saved = {} # To save maps later
    print("Preprocessing conditioning images...")
    target_size = (image_resolution, image_resolution) # W, H for resize, usually H, W for models

    # Determine where to run preprocessors (CPU if main pipe is offloaded, else GPU)
    # Check if pipe has 'hf_device_map' which indicates accelerate offload/distribution
    is_offloaded = hasattr(pipe, 'hf_device_map') and pipe.hf_device_map is not None
    device_for_preprocessor = 'cpu' if is_offloaded else DEVICE
    print(f"Device for preprocessors: {device_for_preprocessor}")


    for i, (img_pil, ctype) in enumerate(zip(conditioning_image_pils, control_types)):
        if img_pil is None:
            print(f"Error: Conditioning image {i+1} for '{ctype}' is None.")
            return None, []
        if ctype not in preprocessors:
            print(f"Error: No preprocessor found for control type '{ctype}'.")
            return None, []

        try:
            control_image = preprocess_control_image(
                img_pil, preprocessors[ctype], ctype, target_size, device_for_preprocessor
            )
            control_images_processed.append(control_image)
            control_images_saved[ctype] = control_image # Store for saving later
        except Exception as e:
            print(f"Error during preprocessing for {ctype}: {e}")
            import traceback
            traceback.print_exc()
            return None, []

    # Sanity check sizes before passing to pipeline
    if not all(img.size == target_size for img in control_images_processed):
        print(f"Error: Not all control images were correctly resized to the target {target_size}.")
        print(f"Actual sizes: {[img.size for img in control_images_processed]}")
        return None, list(control_images_saved.values()) # Return maps even if generation fails

    # --- 2. Setup Generator ---
    generator = None
    if seed is not None:
        print(f"Using seed: {seed}")
        # Seed needs to be generated on the correct device ('cuda' or 'cpu')
        generator_device = 'cpu' # Default for CPU or if offloading (generator state is small)
        if DEVICE == 'cuda' and not is_offloaded:
             generator_device = DEVICE
        generator = torch.Generator(device=generator_device).manual_seed(seed)


    # --- 3. Run Base Pipeline ---
    print("Running SDXL ControlNet pipeline (Base)...")
    start_time_base = time.time()
    image_latents = None

    # Determine the split point for base/refiner
    # denoising_end controls how much of the process the base model handles
    denoising_end_for_base = 1.0 - refiner_steps_ratio
    # denoising_start for refiner will be the same value

    try:
        # Check device of the main pipe (can change with offloading)
        pipe_device = pipe.device if hasattr(pipe, 'device') else DEVICE
        print(f"  Base pipeline expected device: {pipe_device}")

        # Run Base Pipe, outputting latents
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_images_processed, # List of PIL control images
            controlnet_conditioning_scale=controlnet_scales, # List of scales
            num_inference_steps=num_inference_steps, # Total steps
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="latent", # Get latents for refiner
            denoising_end=denoising_end_for_base # Stop base partway
        )
        image_latents = output.images
        end_time_base = time.time()
        print(f"Base pipeline completed in {end_time_base - start_time_base:.2f} seconds.")

    except Exception as e_base:
        print(f"\nError during Base pipeline execution: {e_base}")
        import traceback
        traceback.print_exc()
        if "out of memory" in str(e_base).lower() and DEVICE == 'cuda':
            print("CUDA out of memory during base pass. Suggestions:")
            print(" - Ensure CPU offload is enabled (--enable_cpu_offload)")
            print(" - Reduce resolution (--resolution)")
            print(" - Enable attention slicing (--enable_attention_slicing)")
            print(" - Use fewer/less demanding ControlNets")
            print(" - Close other GPU-heavy applications")
            torch.cuda.empty_cache()
        return None, list(control_images_saved.values())

    # --- 4. Run Refiner Pipeline ---
    if refiner_pipe and refiner_steps_ratio > 0:
        print("Running SDXL Refiner pipeline...")
        start_time_refiner = time.time()
        output_image = None

        try:
            # Check refiner device and move latents if necessary
            refiner_device = refiner_pipe.device if hasattr(refiner_pipe, 'device') else DEVICE
            print(f"  Refiner pipeline expected device: {refiner_device}")
            if image_latents.device != refiner_device:
                print(f"  Moving latents from {image_latents.device} to {refiner_device} for refiner.")
                image_latents = image_latents.to(refiner_device)

            # Re-use generator if seeded, otherwise None
            generator_refiner = generator # Pass the same generator object

            # Run Refiner Pipe
            output_image = refiner_pipe(
                prompt=prompt, # Refiner uses prompts too
                negative_prompt=negative_prompt,
                image=image_latents, # Pass latents from base
                num_inference_steps=num_inference_steps, # Total steps
                guidance_scale=guidance_scale, # Can sometimes be lower for refiner, but often same
                denoising_start=denoising_end_for_base, # Start refining from where base left off
                generator=generator_refiner,
                # output_type="pil" # Default for refiner is PIL
            ).images[0]

            end_time_refiner = time.time()
            print(f"Refiner pipeline completed in {end_time_refiner - start_time_refiner:.2f} seconds.")

        except Exception as e_refiner:
            print(f"\nError during Refiner pipeline execution: {e_refiner}")
            import traceback
            traceback.print_exc()
            if "out of memory" in str(e_refiner).lower() and DEVICE == 'cuda':
                 print("CUDA out of memory during refiner pass. Check suggestions for base pass.")
                 torch.cuda.empty_cache()
            # Fallback: try decoding latents from base pipe if refiner fails?
            print("Refiner failed. Attempting to decode latents from base pipe...")
            try:
                 # Ensure latents are on the correct device for the VAE decoder part of the main pipe
                 vae_device = pipe.vae.device if hasattr(pipe, 'vae') else DEVICE
                 if image_latents.device != vae_device:
                     image_latents = image_latents.to(vae_device)
                 # Scale latents before decoding (standard practice)
                 image_latents = image_latents / pipe.vae.config.scaling_factor
                 with torch.no_grad(): # Ensure VAE decoding runs without gradients
                    output_image = pipe.vae.decode(image_latents, return_dict=False)[0]
                 # Convert to PIL
                 output_image = pipe.image_processor.postprocess(output_image, output_type="pil")[0]
                 print("Successfully decoded latents from base pipe as fallback.")
            except Exception as e_decode:
                 print(f"Error decoding latents from base pipe: {e_decode}")
                 return None, list(control_images_saved.values()) # Give up if decoding also fails
    else:
        # If no refiner or ratio is 0, decode latents from base pipe directly
        print("Skipping Refiner. Decoding latents from base pipeline...")
        try:
            # Ensure latents are on the correct device for the VAE decoder
            vae_device = pipe.vae.device if hasattr(pipe, 'vae') else DEVICE
            if image_latents.device != vae_device:
                print(f"  Moving latents from {image_latents.device} to {vae_device} for VAE decoding.")
                image_latents = image_latents.to(vae_device)

            # Scale latents before decoding
            image_latents = image_latents / pipe.vae.config.scaling_factor
            with torch.no_grad(): # Ensure VAE decoding runs without gradients
                output_image = pipe.vae.decode(image_latents, return_dict=False)[0]
            # Convert to PIL
            output_image = pipe.image_processor.postprocess(output_image, output_type="pil")[0]
            print("Latents decoded successfully.")

        except Exception as e_decode_no_refiner:
            print(f"\nError decoding latents from base pipe (no refiner): {e_decode_no_refiner}")
            import traceback
            traceback.print_exc()
            return None, list(control_images_saved.values())

    # --- 5. Save Output ---
    try:
        output_image.save(output_path)
        print(f"Generated image saved successfully to: {output_path}")
    except Exception as e_save:
        print(f"Error saving generated image to {output_path}: {e_save}")
        # Still return the image object if saving failed
        return output_image, list(control_images_saved.values())

    print("--- Generation Complete ---")
    return output_image, list(control_images_saved.values()) # Return PIL image and list of control PIL images