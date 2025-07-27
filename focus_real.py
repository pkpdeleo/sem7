# --- Complex Image RAG with SDXL, Refiner, and Multi-ControlNet (focus_real.py) ---
# --- Includes fix for control map resizing ---
import torch
from transformers import CLIPModel, CLIPProcessor
import faiss
import numpy as np
from PIL import Image
import os
import requests
from io import BytesIO
import pickle
import argparse
# Ensure opencv is installed for preprocessors even if not used directly here
import cv2 # Keep import for dependency checks or potential fallback
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL, UniPCMultistepScheduler
from controlnet_aux import CannyDetector, MidasDetector # Using Midas for Depth
import time
from huggingface_hub import hf_hub_download # For potentially loading VAE separately

# --- Configuration ---
# Models (Using SDXL)
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
BASE_SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
REFINER_SDXL_MODEL = "stabilityai/stable-diffusion-xl-refiner-1.0"
# SDXL ControlNets (Make sure these are compatible diffusers models)
# Using diffusers community models as examples, check for official/updated ones
CONTROLNET_MODEL_DEPTH_SDXL = "diffusers/controlnet-depth-sdxl-1.0"
CONTROLNET_MODEL_CANNY_SDXL = "diffusers/controlnet-canny-sdxl-1.0"
# SDXL VAE (Often improves results, especially if base model doesn't include it)
VAE_SDXL_MODEL = "madebyollin/sdxl-vae-fp16-fix" # Using fp16 fix VAE

# Files
DEFAULT_IMAGE_FOLDER = "image_knowledge_base_xl" # Use separate folder/index
VECTOR_DB_INDEX_PATH = "image_rag_sdxl_controlnet.faiss"
IMAGE_FILENAMES_PATH = "image_filenames_sdxl_controlnet.pkl"
DEFAULT_OUTPUT_PATH = "generated_sdxl_realistic.png"

# Parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# SDXL runs best in float16 on GPU
DTYPE_VAE = torch.float16 # VAE often needs fp16/bf16
DTYPE_CONTROL = torch.float16
DTYPE_UNET = torch.float16 # Base model components
DTYPE_REFINER = torch.float16

print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    try:
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        if torch.cuda.get_device_properties(0).total_memory / (1024**3) < 15:
            print("Warning: Less than 15GB VRAM detected. SDXL Multi-ControlNet + Refiner may fail due to OOM.")
    except Exception as e:
        print(f"Could not get GPU details: {e}")

# --- Core RAG Components (Mostly unchanged, ensure compatibility) ---

clip_model = None
clip_processor = None

def load_clip_models():
    global clip_model, clip_processor
    if clip_model is None:
        print(f"Loading CLIP model: {CLIP_MODEL_NAME}")
        clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE) # Keep CLIP on primary device
    if clip_processor is None:
        print(f"Loading CLIP processor: {CLIP_MODEL_NAME}")
        clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    return clip_model, clip_processor

def initialize_vector_db(embedding_dimension):
    return faiss.IndexFlatL2(embedding_dimension)

def add_embeddings_to_index(index, embeddings):
    index.add(embeddings.astype('float32'))

def save_vector_db_index(index, index_path):
    faiss.write_index(index, index_path)
    print(f"Vector index SAVED to: {index_path}")

def load_vector_db_index(index_path):
    if os.path.exists(index_path):
        print(f"Loading vector index from: {index_path}")
        try:
            return faiss.read_index(index_path)
        except Exception as e:
            print(f"Error loading FAISS index: {e}. Will re-index if needed.")
            return None
    else:
        print("Vector index file not found.")
        return None

def save_image_filenames(filenames, filenames_path):
    with open(filenames_path, 'wb') as f:
        pickle.dump(filenames, f)
    print(f"Image filenames SAVED to: {filenames_path}")

def load_image_filenames(filenames_path):
    if os.path.exists(filenames_path):
        print(f"Loading image filenames from: {filenames_path}")
        try:
            with open(filenames_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading filenames pickle: {e}. Will re-index if needed.")
            return None
    else:
        print("Image filenames file not found.")
        return None

def load_image(image_path):
    """Loads an image from path or URL into PIL format."""
    try:
        if image_path.startswith("http://") or image_path.startswith("https://"):
            response = requests.get(image_path, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
        elif os.path.exists(image_path):
            img = Image.open(image_path).convert("RGB")
        else:
            # Handle case where path stored in pickle is relative but execution dir changed
            print(f"Warning: Image path '{image_path}' not found directly.")
            # Optionally, try finding it relative to the script location or index file location
            # This part can get complex depending on setup. Returning None for now.
            base_dir = os.path.dirname(os.path.abspath(__file__)) # If paths relative to script
            potential_path = os.path.join(base_dir, image_path)
            if os.path.exists(potential_path):
                print(f"Found image at potential path: {potential_path}")
                img = Image.open(potential_path).convert("RGB")
                return img
            else:
                print(f"Image path does not exist: {image_path}")
                return None
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def get_clip_image_embedding(image_pil, model, processor):
    """Generates CLIP embedding for a PIL image."""
    if image_pil is None: return None
    try:
        # Ensure CLIP model is on the primary device for this step
        inputs = processor(images=image_pil, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
        embedding = outputs.cpu().numpy()
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding, axis=-1, keepdims=True)
        return embedding
    except Exception as e:
        print(f"Error generating CLIP embedding: {e}")
        return None

def get_clip_text_embedding(query_text, model, processor):
    """Generates CLIP embedding for a text query."""
    try:
        # Ensure CLIP model is on the primary device
        inputs = processor(text=[query_text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.get_text_features(**inputs)
        embedding = outputs.cpu().numpy()
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding, axis=-1, keepdims=True)
        return embedding
    except Exception as e:
        print(f"Error generating text embedding: {e}")
        return None

def index_image_knowledge_base(image_folder, index_path, filenames_path):
    """Indexes images in a folder and saves embeddings and filenames."""
    print(f"Starting indexing process for folder: {image_folder}")
    model, processor = load_clip_models()
    image_paths = []
    # Store paths relative to the image_folder if possible, or absolute paths
    abs_image_folder = os.path.abspath(image_folder)

    for root, _, files in os.walk(abs_image_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                full_path = os.path.join(root, file)
                # Store absolute path for robustness if script/data moves
                image_paths.append(full_path)
                # Or store relative path: image_paths.append(os.path.relpath(full_path, abs_image_folder))

    if not image_paths:
        print(f"No images found in folder: {image_folder}")
        return None, None # Indicate indexing failure

    embeddings_list = []
    image_filenames_stored = [] # Filenames actually stored in pickle
    processed_count = 0
    total_images = len(image_paths)

    for i, img_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{total_images}: {os.path.basename(img_path)} ...", end='\r')
        # Pass the absolute path to load_image
        image_pil = load_image(img_path)
        if image_pil:
            embedding = get_clip_image_embedding(image_pil, model, processor)
            if embedding is not None:
                embeddings_list.append(embedding)
                # Store the same path format (absolute/relative) used for loading
                image_filenames_stored.append(img_path)
                processed_count += 1
            else: print(f"\nSkipping {img_path} due to embedding error.")
        else: print(f"\nSkipping {img_path} due to loading error.")
        # Optional: Clear CUDA cache periodically if memory issues arise
        if DEVICE == 'cuda' and (i + 1) % 100 == 0:
            torch.cuda.empty_cache()


    print(f"\nSuccessfully processed {processed_count}/{total_images} images.")

    if not embeddings_list:
        print("No valid image embeddings were generated. Index not created.")
        return None, None # Indicate indexing failure

    embeddings_matrix = np.concatenate(embeddings_list, axis=0)
    embedding_dimension = embeddings_matrix.shape[1]

    index = initialize_vector_db(embedding_dimension)
    add_embeddings_to_index(index, embeddings_matrix)
    save_vector_db_index(index, index_path)
    save_image_filenames(image_filenames_stored, filenames_path) # Save the list of paths

    print(f"Indexed {len(image_filenames_stored)} images.")
    return index, image_filenames_stored

def retrieve_relevant_images(query_embedding, index, image_filenames, top_k=2):
    """Retrieves top_k most relevant image filenames based on query embedding."""
    print(f"\n--- Retrieving top {top_k} relevant images ---")
    if index is None or index.ntotal == 0: print("Vector database index is empty or not loaded."); return []
    if image_filenames is None: print("Image filenames not loaded."); return []
    if index.ntotal != len(image_filenames): print(f"Warning: FAISS index size ({index.ntotal}) != filenames ({len(image_filenames)}). Re-index recommended.")

    try:
        print("Searching FAISS index...")
        distances, indices = index.search(query_embedding.astype('float32'), min(top_k, index.ntotal)) # Ensure top_k <= index size
        print(f"FAISS search complete. Found indices: {indices[0]}")
    except Exception as e:
        print(f"Error during FAISS index search: {e}")
        return []

    retrieved_filenames = []
    retrieved_distances = []
    max_valid_index = len(image_filenames) - 1

    for i, idx in enumerate(indices[0]):
        if 0 <= idx <= max_valid_index:
            retrieved_filenames.append(image_filenames[idx]) # Get path from loaded list
            retrieved_distances.append(distances[0][i])
        else:
             print(f"Warning: Retrieved index {idx} is out of bounds (max valid: {max_valid_index}). Skipping.")

    print(f"Retrieved {len(retrieved_filenames)} filenames:")
    for fname, dist in zip(retrieved_filenames, retrieved_distances):
        # Display basename for cleaner logs
        print(f"  - {os.path.basename(fname)} (Distance: {dist:.4f})")
    print("--- Retrieval complete ---")
    return retrieved_filenames # Return the list of paths

# --- SDXL ControlNet Components ---

def get_sdxl_controlnet_preprocessor(control_type):
    """Gets the appropriate preprocessor for the control type."""
    print(f"Loading SDXL preprocessor for control type: {control_type}")
    # Make preprocessor loading robust
    try:
        if control_type == "canny":
            return CannyDetector()
        elif control_type == "depth":
            # Using MiDaS via controlnet_aux - uses lllyasviel/Annotators by default
            print("  Loading MidasDetector (may download annotator models)...")
            # Ensure model is loaded to correct device within the generation function later if needed
            detector = MidasDetector.from_pretrained("lllyasviel/Annotators")
            print("  MidasDetector loaded.")
            return detector
        else:
            raise ValueError(f"Unsupported control_type for SDXL: {control_type}")
    except Exception as e:
        print(f"Error loading preprocessor for {control_type}: {e}")
        raise # Re-raise error to stop execution if preprocessor fails

def load_sdxl_controlnet_pipeline(base_model_id, refiner_model_id, vae_model_id, controlnet_model_ids: list, enable_cpu_offload=False):
    """Loads the Stable Diffusion XL ControlNet pipeline with Refiner."""
    print("\n--- Loading SDXL Multi-ControlNet Pipeline ---")

    # Load ControlNets
    print("Loading ControlNet models...")
    controlnets = []
    for model_id in controlnet_model_ids:
         try:
              print(f"  Loading: {model_id}")
              # Try loading with safetensors first
              cn = ControlNetModel.from_pretrained(model_id, torch_dtype=DTYPE_CONTROL, use_safetensors=True)
              controlnets.append(cn)
         except EnvironmentError: # Catch errors if .safetensors doesn't exist
              print(f"  Safetensors not found for {model_id}, trying .bin")
              try:
                    cn = ControlNetModel.from_pretrained(model_id, torch_dtype=DTYPE_CONTROL, use_safetensors=False)
                    controlnets.append(cn)
              except Exception as e2:
                    print(f"Failed loading {model_id} (bin): {e2}")
                    raise
         except Exception as e:
              print(f"Error loading ControlNet {model_id}: {e}")
              raise # Re-raise the error if loading fails other than safetensors format

    print(f"Loaded {len(controlnets)} ControlNet model(s).")

    # Load VAE
    print(f"Loading VAE: {vae_model_id}")
    try:
        vae = AutoencoderKL.from_pretrained(vae_model_id, torch_dtype=DTYPE_VAE, use_safetensors=True)
    except EnvironmentError:
        print(f"  Safetensors not found for VAE {vae_model_id}, trying .bin")
        try:
            vae = AutoencoderKL.from_pretrained(vae_model_id, torch_dtype=DTYPE_VAE, use_safetensors=False)
        except Exception as e2:
            print(f"Failed loading VAE {vae_model_id} (bin): {e2}")
            raise
    except Exception as e:
        print(f"Error loading VAE {vae_model_id}: {e}")
        raise # Re-raise if VAE loading fails

    # Load Base and Refiner models into the pipeline
    print(f"Loading Base SDXL: {base_model_id}")
    # Refiner is loaded separately if needed, or handled by the pipeline constructor
    # Depending on diffusers version, refiner might be passed here or loaded later
    print(f"Loading Refiner SDXL: {refiner_model_id}")
    # Load refiner first to potentially offload it if needed
    # This part might vary based on exact diffusers version and how it handles refiners
    refiner = None
    try:
        from diffusers import StableDiffusionXLImg2ImgPipeline # Check if needed
        refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            refiner_model_id,
            vae=vae, # Refiner typically uses the same VAE
            torch_dtype=DTYPE_REFINER,
            use_safetensors=True,
            variant="fp16" if DTYPE_REFINER == torch.float16 else None,
        )
        print("Refiner loaded separately.")
    except EnvironmentError:
         print(f"  Safetensors not found for Refiner {refiner_model_id}, trying .bin")
         try:
            refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                refiner_model_id,
                vae=vae,
                torch_dtype=DTYPE_REFINER,
                use_safetensors=False,
                variant="fp16" if DTYPE_REFINER == torch.float16 else None,
            )
            print("Refiner loaded separately.")
         except Exception as e_refiner:
            print(f"Could not load Refiner separately: {e_refiner}. Pipeline might handle it internally or fail.")
            refiner = None # Ensure refiner is None if loading failed
    except Exception as e_refiner:
        print(f"Could not load Refiner separately: {e_refiner}. Pipeline might handle it internally or fail.")
        refiner = None


    print(f"Instantiating main pipeline: {base_model_id}")
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        base_model_id,
        controlnet=controlnets, # Pass list of ControlNets
        vae=vae,
        # refiner=refiner, # Pass refiner if pipeline constructor supports it directly
        torch_dtype=DTYPE_UNET,
        use_safetensors=True,
        variant="fp16" if DTYPE_UNET == torch.float16 else None,
        add_watermarker=False, # Optional: disable watermarker
    )


    # Optimization: Faster Scheduler
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # VRAM Optimization (Essential for many users)
    if enable_cpu_offload and DEVICE == 'cuda':
        print("Enabling model CPU offload (requires 'accelerate')...")
        try:
            # Offload pipeline components first
            pipe.enable_model_cpu_offload()
            # Offload separately loaded refiner if it exists
            if refiner is not None:
                 print("Offloading refiner...")
                 refiner.enable_model_cpu_offload()
            print("CPU offload enabled.")
        except ImportError:
            print("Warning: 'accelerate' library not found. CPU offload unavailable.")
        except Exception as e:
            print(f"Warning: Failed to enable CPU offload: {e}")
    elif DEVICE == 'cuda':
        print("Moving pipeline components to GPU...")
        pipe = pipe.to(DEVICE)
        if refiner is not None:
             refiner = refiner.to(DEVICE)
        print("Pipeline components moved to GPU.")
    else:
         print("Running on CPU (will be very slow).")


    print("--- SDXL Pipeline Loading Complete ---")
    # Return both pipe and the potentially separately loaded refiner
    return pipe, refiner

def generate_sdxl_controlled_image(pipe, refiner_pipe, preprocessors: dict, prompt, negative_prompt,
                                   conditioning_image_pils: list, control_types: list, controlnet_scales: list,
                                   output_path, num_inference_steps=50, guidance_scale=7.5,
                                   seed=None, image_resolution=1024,
                                   refiner_steps_ratio=0.2): # Ratio of steps for refiner

    """Generates an image using the SDXL ControlNet pipeline and Refiner."""
    print("\n--- Starting SDXL Controlled Image Generation ---")
    # ... (print parameters as before) ...
    print(f"Prompt: {prompt}")
    print(f"Negative Prompt: {negative_prompt}")
    print(f"Output path: {output_path}")
    print(f"Resolution: {image_resolution}x{image_resolution}")
    print(f"Steps: {num_inference_steps}, Guidance: {guidance_scale}")
    print(f"Control Types: {control_types}")
    print(f"Control Scales: {controlnet_scales}")
    print(f"Refiner Steps Ratio: {refiner_steps_ratio}")


    if len(conditioning_image_pils) != len(control_types) or len(control_types) != len(controlnet_scales):
        print("Error: Mismatch between conditioning images, control types, and scales.")
        return None, []

    # --- Prepare conditioning images ---
    control_images_pil = []
    print("Preprocessing conditioning images...")
    target_size = (image_resolution, image_resolution) # Define target size H, W

    for i, (img_pil, ctype) in enumerate(zip(conditioning_image_pils, control_types)):
        if ctype not in preprocessors:
            print(f"Error: No preprocessor found for control type '{ctype}'.")
            return None, []
        try:
            print(f"  Processing for {ctype}...")
            preprocessor = preprocessors[ctype]
            # Move preprocessor model to device if it has parameters (like MidasDetector)
            # Handle potential device placement with offloading
            preprocessor_device = DEVICE
            if hasattr(pipe, 'device') and 'offload' in str(pipe.device):
                preprocessor_device = 'cpu' # Assume preprocessor runs on CPU if offloading main pipe

            if hasattr(preprocessor, 'model') and hasattr(preprocessor.model, 'to'):
                 preprocessor.model.to(preprocessor_device)
                 if preprocessor_device != DEVICE: print(f"   (Preprocessor temporarily on {preprocessor_device})")


            # Run preprocessing
            # Some preprocessors might need specific handling or different parameters
            if ctype == 'depth' and isinstance(preprocessor, MidasDetector):
                 # MidasDetector might need specific handling for resolution if defaults aren't ideal
                 control_image = preprocessor(img_pil, detect_resolution=image_resolution, image_resolution=image_resolution)
            else:
                 # Default call for CannyDetector etc.
                 control_image = preprocessor(img_pil) # Canny usually doesn't need resolution args


            # Ensure PIL RGB format
            if isinstance(control_image, np.ndarray):
                 control_image = Image.fromarray(control_image)
            control_image = control_image.convert("RGB") # Ensure 3 channels

            # --- Explicitly Resize Control Map ---
            print(f"  Raw {ctype} map size: {control_image.size}")
            if control_image.size != target_size:
                print(f"  Resizing {ctype} map from {control_image.size} to {target_size}...")
                # Use a high-quality resampling filter like LANCZOS or BICUBIC
                control_image = control_image.resize(target_size, Image.Resampling.LANCZOS)
            # --- End Resize ---

            control_images_pil.append(control_image)
            print(f"  Final {ctype} map size: {control_image.size}")

            # Move preprocessor model back to CPU if it was moved and offloading active
            if hasattr(preprocessor, 'model') and hasattr(preprocessor.model, 'to') and preprocessor_device != 'cpu':
                if hasattr(pipe, 'device') and 'offload' in str(pipe.device):
                    preprocessor.model.to('cpu')
                    print("   (Preprocessor moved back to CPU)")


        except Exception as e:
            print(f"Error during preprocessing for {ctype}: {e}")
            # Print traceback for debugging: import traceback; traceback.print_exc()
            return None, [] # Exit if preprocessing fails

    # --- Sanity check sizes before passing to pipeline ---
    if not all(img.size == target_size for img in control_images_pil):
        print(f"Error: Not all control images were correctly resized to the target {target_size}.")
        print(f"Actual sizes: {[img.size for img in control_images_pil]}")
        return None, control_images_pil # Prevent running pipeline with mismatched sizes

    # Set seed for reproducibility
    generator = None
    if seed is not None:
        print(f"Using seed: {seed}")
        # Seed needs to be generated on the correct device if using GPU
        generator = torch.Generator(device=DEVICE if DEVICE=='cuda' else 'cpu').manual_seed(seed)


    # Determine split point for base/refiner steps using denoising_end
    denoising_end_for_base = 1.0 - refiner_steps_ratio

    # Run the pipeline
    print("Running SDXL ControlNet pipeline (Base)...")
    start_time = time.time()
    image_latents = None
    output_image = None

    try:
        # Run Base Pipe first, outputting latents
        # Need to manage device placement if offloading
        pipe_device = pipe.device if hasattr(pipe, 'device') else DEVICE
        print(f"Base pipeline running on device: {pipe_device}")

        image_latents = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_images_pil, # List of PIL control images
            num_inference_steps=num_inference_steps, # Total steps (refiner adjusts based on denoising)
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_scales, # List of scales
            generator=generator,
            output_type="latent", # Get latents for refiner
            denoising_end=denoising_end_for_base # Run base until this fraction
        ).images # Output is latents here

        print("Base pipeline complete. Running Refiner...")

        # Ensure latents are on the correct device for the refiner
        refiner_device = refiner_pipe.device if hasattr(refiner_pipe, 'device') else DEVICE
        print(f"Refiner pipeline running on device: {refiner_device}")

        # If latents are on CPU and refiner on GPU, move them
        if image_latents.device != refiner_device:
            print(f"Moving latents from {image_latents.device} to {refiner_device} for refiner.")
            image_latents = image_latents.to(refiner_device)


        # Run Refiner Pipe
        # Reset generator seed if needed for consistent refiner step? Usually not needed.
        # generator = torch.Generator(device=DEVICE if DEVICE=='cuda' else 'cpu').manual_seed(seed) if seed is not None else None

        output_image = refiner_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt, # Refiner usually uses prompts too
            image=image_latents, # Pass latents from base
            num_inference_steps=num_inference_steps, # Total steps
            guidance_scale=guidance_scale, # Often same cfg as base
            # Refiner starts from where base left off
            denoising_start=denoising_end_for_base,
            # generator=generator, # Pass generator if seeding refiner separately
        ).images[0]


        end_time = time.time()
        print(f"Pipeline execution time (Base+Refiner): {end_time - start_time:.2f} seconds")

        # Save the generated image
        output_image.save(output_path)
        print(f"Generated image saved to: {output_path}")
        print("--- Generation Complete ---")
        return output_image, control_images_pil # Return generated and control images

    except Exception as e:
        print(f"\nError during image generation pipeline: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        if "out of memory" in str(e).lower() and DEVICE == 'cuda':
             print("CUDA out of memory. Try enabling CPU offload (--enable_cpu_offload), reducing resolution, or using fewer ControlNets.")
             torch.cuda.empty_cache()
        # Consider re-raising for debugging: raise e
        return None, control_images_pil


# --- Main Execution Logic ---
def main(args):
    # --- 1. Setup & Load/Index Data ---
    if not os.path.exists(args.image_folder):
        print(f"Image folder '{args.image_folder}' not found. Please create it and add images.")
        # Optionally, create it:
        # try:
        #     os.makedirs(args.image_folder)
        #     print(f"Created image folder: {args.image_folder}")
        # except OSError as e:
        #     print(f"Error creating image folder {args.image_folder}: {e}")
        #     return # Exit if cannot create
        # return # Exit anyway, needs images added manually
        print("Please ensure the image folder exists and contains images before running.")
        return

    index = None
    image_filenames = None
    if not args.reindex:
        index = load_vector_db_index(args.index_path)
        image_filenames = load_image_filenames(args.filenames_path)

    if index is None or image_filenames is None or args.reindex:
        if args.reindex: print("Re-indexing requested.")
        else: print("Index or filenames not found/loaded. Starting indexing...")
        # Make sure the folder exists before indexing
        if not os.path.isdir(args.image_folder):
            print(f"Error: Image folder '{args.image_folder}' not found or is not a directory. Cannot index.")
            return
        index, image_filenames = index_image_knowledge_base(args.image_folder, args.index_path, args.filenames_path)
        if index is None or image_filenames is None: print("Indexing failed. Cannot proceed."); return
    else:
        print("Successfully loaded existing index and filenames.")
        if index.ntotal != len(image_filenames): print(f"Warning: Index size ({index.ntotal}) != filenames ({len(image_filenames)}) mismatch. Re-indexing is recommended.")

    # --- 2. Load Models (CLIP only here, SDXL pipeline loaded later) ---
    clip_model_instance, clip_processor_instance = load_clip_models()

    # --- 3. Process Query and Retrieve ---
    print(f"\nProcessing query: '{args.query}'")
    query_embedding = get_clip_text_embedding(args.query, clip_model_instance, clip_processor_instance)
    if query_embedding is None: print("Failed to generate query embedding."); return

    # Retrieve Top 2 for Depth and Canny
    retrieved_image_paths = retrieve_relevant_images(query_embedding, index, image_filenames, top_k=2)

    if not retrieved_image_paths: print("No relevant images found."); return
    if len(retrieved_image_paths) < 2:
        print("Warning: Found fewer than 2 relevant images. Using the first image for both Depth and Canny.")
        # Ensure we have two paths, even if identical
        while len(retrieved_image_paths) < 2:
             retrieved_image_paths.append(retrieved_image_paths[0])


    # --- 4. Prepare Conditioning Inputs ---
    conditioning_paths = [retrieved_image_paths[0], retrieved_image_paths[1]] # Img 0 for Depth, Img 1 for Canny
    control_types = ["depth", "canny"]
    controlnet_scales = [args.depth_scale, args.canny_scale]
    controlnet_model_ids = [CONTROLNET_MODEL_DEPTH_SDXL, CONTROLNET_MODEL_CANNY_SDXL]

    print(f"\nSelected conditioning images:")
    print(f"  Depth Control Source: {os.path.basename(conditioning_paths[0])}")
    print(f"  Canny Control Source: {os.path.basename(conditioning_paths[1])}")

    conditioning_pils = [load_image(p) for p in conditioning_paths]
    if None in conditioning_pils: print("Failed to load one or more conditioning images."); return

    # --- 5. Load SDXL Pipeline & Preprocessors ---
    pipe = None
    refiner = None
    preprocessors = {}
    try:
        pipe, refiner = load_sdxl_controlnet_pipeline(
             args.base_model, args.refiner_model, args.vae_model, controlnet_model_ids, args.enable_cpu_offload
        )
        preprocessors = {
            "depth": get_sdxl_controlnet_preprocessor("depth"),
            "canny": get_sdxl_controlnet_preprocessor("canny")
        }
        if refiner is None:
            print("Warning: Refiner pipe could not be loaded separately. Check pipeline implementation if refinement is expected.")
            # Decide how to proceed - maybe run without refiner or try integrated approach if pipe supports it
            # For now, let's try to proceed, generate_sdxl_controlled_image might handle refiner=None

    except Exception as e:
        print(f"FATAL: Failed to load SDXL pipeline or preprocessors: {e}")
        import traceback
        traceback.print_exc()
        return # Cannot proceed if pipeline fails

    # --- 6. Generate Image ---
    # Use a more detailed prompt for realism
    realistic_prompt = f"photorealistic DSLR photo, {args.query}, sharp focus, high detail, intricate details, natural lighting, cinematic composition"
    negative_prompt = args.negative_prompt + ", painting, drawing, illustration, sketch, cartoon, anime, 3D render, CGI, unrealistic, artificial, blurry, low quality, noisy, text, signature, watermark, disfigured, mutated, deformed, worst quality, low quality, jpeg artifacts"

    # Check if refiner pipe is available before calling generation
    if refiner is None:
         print("Error: Refiner pipeline is not available. Cannot proceed with refiner step.")
         # Optionally add logic here to run *only* the base pipe if that's desired fallback
         return

    generated_img, control_imgs_pil = generate_sdxl_controlled_image(
        pipe=pipe,
        refiner_pipe=refiner, # Pass the loaded refiner pipe
        preprocessors=preprocessors,
        prompt=realistic_prompt,
        negative_prompt=negative_prompt,
        conditioning_image_pils=conditioning_pils,
        control_types=control_types,
        controlnet_scales=controlnet_scales,
        output_path=args.output_path,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        image_resolution=args.resolution,
        refiner_steps_ratio=args.refiner_ratio
    )

    # Optional: Save the control maps for inspection
    if generated_img and control_imgs_pil and len(control_imgs_pil) == 2:
         try:
              depth_map_path = os.path.splitext(args.output_path)[0] + "_control_depth.png"
              canny_map_path = os.path.splitext(args.output_path)[0] + "_control_canny.png"
              control_imgs_pil[0].save(depth_map_path) # Depth map
              control_imgs_pil[1].save(canny_map_path) # Canny map
              print(f"Control maps saved to: {depth_map_path}, {canny_map_path}")
         except Exception as e:
              print(f"Warning: Failed to save control maps: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complex Image RAG with SDXL, Refiner, and Multi-ControlNet")

    # RAG Arguments
    parser.add_argument("query", type=str, help="Text query to search for relevant images and use as base for generation prompt.")
    parser.add_argument("--image_folder", type=str, default=DEFAULT_IMAGE_FOLDER, help="Folder containing the image knowledge base.")
    parser.add_argument("--index_path", type=str, default=VECTOR_DB_INDEX_PATH, help="Path to save/load the FAISS index file.")
    parser.add_argument("--filenames_path", type=str, default=IMAGE_FILENAMES_PATH, help="Path to save/load the image filenames pickle file.")
    parser.add_argument("--reindex", action="store_true", help="Force re-indexing of the image folder.")
    # top_k is fixed at 2 for Depth+Canny

    # Model Arguments (SDXL)
    parser.add_argument("--base_model", type=str, default=BASE_SDXL_MODEL, help="Base SDXL model ID.")
    parser.add_argument("--refiner_model", type=str, default=REFINER_SDXL_MODEL, help="Refiner SDXL model ID.")
    parser.add_argument("--vae_model", type=str, default=VAE_SDXL_MODEL, help="VAE model ID (SDXL specific).")
    # ControlNet model IDs are currently hardcoded based on type

    # Generation Arguments
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH, help="Path to save the generated image.")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt additions.")
    parser.add_argument("--num_steps", type=int, default=50, help="Total number of diffusion inference steps (base + refiner).")
    parser.add_argument("--guidance_scale", "-cfg", type=float, default=7.0, help="Guidance scale (for classifier-free guidance).")
    parser.add_argument("--depth_scale", type=float, default=0.6, help="Conditioning scale for the Depth ControlNet.")
    parser.add_argument("--canny_scale", type=float, default=0.4, help="Conditioning scale for the Canny ControlNet.")
    parser.add_argument("--refiner_ratio", type=float, default=0.2, help="Fraction of total steps dedicated to the refiner (e.g., 0.2 means refiner runs for last 20% of steps). Should be > 0 and < 1.")
    parser.add_argument("--resolution", type=int, default=1024, help="Resolution (height and width) for SDXL generation.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (optional).")
    parser.add_argument("--enable_cpu_offload", action="store_true", help="Enable model CPU offloading to save VRAM (requires accelerate).")


    args = parser.parse_args()

    # Validate refiner_ratio
    if not (0 < args.refiner_ratio < 1):
         print("Error: --refiner_ratio must be between 0 and 1 (exclusive).")
         exit()

    # --- Run Main Pipeline ---
    main(args)