# main.py
import argparse
import os
import sys
import torch
import gc

# -- Add project root to Python path ---
# This allows importing modules from the project directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ---

# Import from project modules
import config
import rag_utils
import sdxl_pipeline_utils
import generation

def check_gpu_memory(required_gb=15):
    """Checks available VRAM if CUDA is available."""
    if config.DEVICE == "cuda":
        try:
            gpu_props = torch.cuda.get_device_properties(0)
            total_mem_gb = gpu_props.total_memory / (1024**3)
            print(f"GPU Name: {gpu_props.name}")
            print(f"Total VRAM: {total_mem_gb:.2f} GB")
            if total_mem_gb < required_gb:
                 print(f"Warning: Detected less than {required_gb}GB VRAM ({total_mem_gb:.2f}GB).")
                 print("         SDXL Multi-ControlNet + Refiner can be demanding.")
                 print("         Enable CPU offload (--enable_cpu_offload) if you encounter OOM errors.")
                 return False # Indicate low memory
            return True # Indicate sufficient memory detected
        except Exception as e:
            print(f"Could not get GPU details: {e}")
            return False # Assume insufficient if check fails
    else:
        print("CUDA not available. Running on CPU.")
        return False # Not relevant for CPU

def main(args):
    """Main function to run the RAG + SDXL generation pipeline."""
    start_time_main = rag_utils.time.time() # Use time from rag_utils or import time here

    print("--- Starting Image RAG SDXL Pipeline ---")
    print(f"Using device: {config.DEVICE}")
    check_gpu_memory() # Check VRAM and print info/warning

    # --- 1. Setup Image Knowledge Base (Index/Load) ---
    index_exists = os.path.exists(args.index_path)
    filenames_exist = os.path.exists(args.filenames_path)

    rag_index = None
    image_filenames = None

    if args.reindex or not index_exists or not filenames_exist:
        if args.reindex:
            print("Re-indexing requested via command line.")
        elif not index_exists:
            print(f"FAISS index file not found at {args.index_path}. Indexing required.")
        else: # filenames must not exist
            print(f"Image filenames file not found at {args.filenames_path}. Indexing required.")

        # Ensure the image folder exists before attempting to index
        if not os.path.isdir(args.image_folder):
             print(f"Error: Image folder '{args.image_folder}' not found or is not a directory.")
             print("Please create the folder, add images, and run again, or specify correct --image_folder.")
             return # Stop execution

        # Perform indexing
        rag_index, image_filenames = rag_utils.index_image_knowledge_base(
            args.image_folder, args.index_path, args.filenames_path
        )
        if rag_index is None or image_filenames is None:
            print("Indexing failed. Cannot proceed.")
            return # Stop execution
    else:
        print("Found existing index and filename files. Loading...")
        rag_index = rag_utils.load_vector_db_index(args.index_path)
        image_filenames = rag_utils.load_image_filenames(args.filenames_path)

        if rag_index is None or image_filenames is None:
            print("Failed to load existing index or filenames. Please check file integrity or try re-indexing.")
            return # Stop execution

        # Sanity check: Compare index size and filenames list length
        if rag_index.ntotal != len(image_filenames):
            print(f"Warning: Mismatch between FAISS index size ({rag_index.ntotal}) and number of filenames ({len(image_filenames)}).")
            print("This might lead to incorrect image retrieval. Re-indexing with --reindex is strongly recommended.")
            # Allow proceeding but with a strong warning

    print("--- RAG Setup Complete ---")

    # --- 2. Process Query and Retrieve Relevant Images ---
    print(f"\nProcessing query: '{args.query}'")
    # Load CLIP models (cached in rag_utils)
    try:
        rag_utils.load_clip_models()
    except Exception as e:
        print(f"Fatal: Could not load CLIP model/processor. Error: {e}")
        return

    # Get text embedding for the query
    query_embedding = rag_utils.get_clip_text_embedding(args.query)
    if query_embedding is None:
        print("Failed to generate query embedding. Cannot retrieve images.")
        return

    # Retrieve top K images (K=2 for Depth + Canny in this setup)
    retrieved_image_paths = rag_utils.retrieve_relevant_images(
        query_embedding, rag_index, image_filenames, top_k=config.DEFAULT_TOP_K_RAG
    )

    if not retrieved_image_paths:
        print("No relevant images found in the knowledge base for the query.")
        return
    elif len(retrieved_image_paths) < config.DEFAULT_TOP_K_RAG:
        print(f"Warning: Found only {len(retrieved_image_paths)} relevant image(s), but need {config.DEFAULT_TOP_K_RAG}.")
        print("Using the first image for all control types.")
        # Duplicate the first path to fill the required slots
        first_path = retrieved_image_paths[0]
        while len(retrieved_image_paths) < config.DEFAULT_TOP_K_RAG:
            retrieved_image_paths.append(first_path)

    # Assign retrieved images to control types (customize if using more controls)
    # For this example: Image 0 -> Depth, Image 1 -> Canny
    control_image_map = {
        "depth": retrieved_image_paths[0],
        "canny": retrieved_image_paths[1]
        # Add more mappings here if using more control types and retrieving more images
        # "openpose": retrieved_image_paths[2] # Example
    }
    control_types_to_use = ["depth", "canny"] # Define which controls are active

    print("\nSelected conditioning images for ControlNets:")
    for ctype, path in control_image_map.items():
         if ctype in control_types_to_use:
            print(f"  - {ctype.capitalize()} Control Source: {os.path.basename(path)}")

    # Load the actual PIL images
    conditioning_pils = []
    control_paths_ordered = [] # Keep track of order for generation function
    for ctype in control_types_to_use:
        path = control_image_map[ctype]
        img = rag_utils.load_image(path)
        if img is None:
            print(f"Error: Failed to load conditioning image for {ctype} from {path}. Cannot proceed.")
            return
        conditioning_pils.append(img)
        control_paths_ordered.append(path)


    # --- 3. Load SDXL Pipeline and Preprocessors ---
    # Define which ControlNets to load based on control_types_to_use
    controlnet_items_to_load = []
    controlnet_scales = [] # Scales corresponding to control_types_to_use
    if "depth" in control_types_to_use:
        controlnet_items_to_load.append((config.CONTROLNET_MODEL_DEPTH_SDXL, "depth"))
        controlnet_scales.append(args.depth_scale)
    if "canny" in control_types_to_use:
        controlnet_items_to_load.append((config.CONTROLNET_MODEL_CANNY_SDXL, "canny"))
        controlnet_scales.append(args.canny_scale)
    # Add other ControlNets here based on args or logic
    # if "openpose" in control_types_to_use:
    #    controlnet_items_to_load.append((config.CONTROLNET_MODEL_POSE_SDXL, "openpose"))
    #    controlnet_scales.append(args.pose_scale) # Assuming args.pose_scale exists

    # Clear some memory before loading heavy models
    del rag_index, image_filenames, query_embedding, retrieved_image_paths
    gc.collect()
    if config.DEVICE == 'cuda': torch.cuda.empty_cache()

    pipe = None
    refiner = None
    preprocessors = {}
    try:
        # Load the main diffusion pipelines
        pipe, refiner = sdxl_pipeline_utils.load_sdxl_controlnet_pipeline(
            args.base_model, args.refiner_model, args.vae_model,
            controlnet_items_to_load,
            args.enable_cpu_offload,
            args.enable_attention_slicing
        )

        # Load necessary preprocessors
        for ctype in control_types_to_use:
            preprocessors[ctype] = sdxl_pipeline_utils.get_sdxl_controlnet_preprocessor(ctype)

    except Exception as e:
        print(f"FATAL: Failed to load SDXL pipeline or preprocessors.")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("Check model names, network connection, and available disk space/memory.")
        return # Cannot proceed if models fail to load

    # --- 4. Prepare Prompts and Generate Image ---
    # Apply style prefix/suffix if a valid style is chosen
    style_prompt_prefix = ""
    style_negative_suffix = ""
    if args.style in config.STYLES:
        style_prompt_prefix, style_negative_suffix = config.STYLES[args.style]
        print(f"Applying style: '{args.style}'")
    else:
        print(f"Warning: Style '{args.style}' not found in config. Using no style.")

    final_prompt = style_prompt_prefix + args.query
    # Combine user negative prompt, default negative prefix, and style negative suffix
    final_negative_prompt = f"{args.negative_prompt}, {config.DEFAULT_NEGATIVE_PROMPT_PREFIX}, {style_negative_suffix}".strip(", ")

    # Clean up prompt strings (remove extra commas, spaces)
    final_prompt = ', '.join(filter(None, [s.strip() for s in final_prompt.split(',')]))
    final_negative_prompt = ', '.join(filter(None, [s.strip() for s in final_negative_prompt.split(',')]))

    # Call the generation function
    generated_img_pil, control_imgs_pil_used = generation.generate_sdxl_controlled_image(
        pipe=pipe,
        refiner_pipe=refiner,
        preprocessors=preprocessors,
        prompt=final_prompt,
        negative_prompt=final_negative_prompt,
        conditioning_image_pils=conditioning_pils, # PIL images in correct order
        control_types=control_types_to_use, # Types in correct order
        controlnet_scales=controlnet_scales, # Scales in correct order
        output_path=args.output_path,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        image_resolution=args.resolution,
        refiner_steps_ratio=args.refiner_ratio
    )

    # --- 5. Post-Generation ---
    if generated_img_pil:
        print(f"\nSuccessfully generated image: {args.output_path}")

        # Optional: Save the processed control maps for inspection
        if control_imgs_pil_used:
            base_output_name = os.path.splitext(args.output_path)[0]
            try:
                for i, ctype in enumerate(control_types_to_use):
                    if i < len(control_imgs_pil_used):
                        control_map_path = f"{base_output_name}_control_{ctype}.png"
                        control_imgs_pil_used[i].save(control_map_path)
                        print(f"  Saved {ctype} control map to: {control_map_path}")
            except Exception as e_save_map:
                print(f"Warning: Failed to save one or more control maps: {e_save_map}")
    else:
        print("\nImage generation failed.")

    end_time_main = rag_utils.time.time()
    print(f"\n--- Total Pipeline Execution Time: {end_time_main - start_time_main:.2f} seconds ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image RAG with SDXL, Refiner, and Multi-ControlNet. Retrieves relevant images and uses them to condition SDXL generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )

    # --- RAG Arguments ---
    rag_group = parser.add_argument_group('RAG - Image Retrieval Arguments')
    rag_group.add_argument("query", type=str, help="Text query to search for relevant images and use as base for generation prompt.")
    rag_group.add_argument("--image_folder", type=str, default=config.DEFAULT_IMAGE_FOLDER, help="Folder containing the image knowledge base.")
    rag_group.add_argument("--index_path", type=str, default=config.VECTOR_DB_INDEX_PATH, help="Path to save/load the FAISS index file.")
    rag_group.add_argument("--filenames_path", type=str, default=config.IMAGE_FILENAMES_PATH, help="Path to save/load the image filenames pickle file.")
    rag_group.add_argument("--reindex", action="store_true", help="Force re-indexing of the image folder, ignoring existing index/filenames files.")

    # --- Model Arguments ---
    model_group = parser.add_argument_group('Model Arguments (SDXL)')
    model_group.add_argument("--base_model", type=str, default=config.BASE_SDXL_MODEL, help="Base SDXL model ID from Hugging Face Hub.")
    model_group.add_argument("--refiner_model", type=str, default=config.REFINER_SDXL_MODEL, help="Refiner SDXL model ID from Hugging Face Hub.")
    model_group.add_argument("--vae_model", type=str, default=config.VAE_SDXL_MODEL, help="VAE model ID (SDXL specific) from Hugging Face Hub.")
    # ControlNet models are now linked to control types in config.py and loaded dynamically

    # --- Generation Arguments ---
    gen_group = parser.add_argument_group('Generation Arguments')
    gen_group.add_argument("--output_path", type=str, default=config.DEFAULT_OUTPUT_PATH, help="Path to save the generated image.")
    gen_group.add_argument("--style", type=str, default=config.DEFAULT_STYLE, choices=list(config.STYLES.keys()), help="Apply a predefined style prompt preset.")
    gen_group.add_argument("--negative_prompt", type=str, default="", help="Additional negative prompt text to guide generation away from.")
    gen_group.add_argument("--num_steps", type=int, default=config.DEFAULT_NUM_STEPS, help="Total number of diffusion inference steps (base + refiner).")
    gen_group.add_argument("--guidance_scale", "-cfg", type=float, default=config.DEFAULT_GUIDANCE_SCALE, help="Guidance scale (for classifier-free guidance). Higher values follow prompt more closely.")
    gen_group.add_argument("--depth_scale", type=float, default=config.DEFAULT_DEPTH_SCALE, help="Conditioning scale for the Depth ControlNet (if used).")
    gen_group.add_argument("--canny_scale", type=float, default=config.DEFAULT_CANNY_SCALE, help="Conditioning scale for the Canny ControlNet (if used).")
    # Add args for other controlnet scales if you add more controls, e.g., --pose_scale
    gen_group.add_argument("--refiner_ratio", type=float, default=config.DEFAULT_REFINER_RATIO, help="Fraction of total steps dedicated to the refiner (e.g., 0.2 means refiner runs for last 20%% of steps). Must be > 0 and < 1.")
    gen_group.add_argument("--resolution", type=int, default=config.DEFAULT_RESOLUTION, help="Resolution (height and width) for SDXL generation (e.g., 1024).")
    gen_group.add_argument("--seed", type=int, default=config.DEFAULT_SEED, help="Random seed for reproducibility (optional, omit for random).")

    # --- Performance Arguments ---
    perf_group = parser.add_argument_group('Performance Arguments')
    perf_group.add_argument("--enable_cpu_offload", action="store_true", default=config.DEFAULT_ENABLE_CPU_OFFLOAD, help="Enable model CPU offloading to save VRAM (requires 'accelerate', may be slower).")
    perf_group.add_argument("--enable_attention_slicing", action="store_true", default=config.DEFAULT_ENABLE_ATTENTION_SLICING, help="Enable attention slicing to save VRAM (may be slightly slower).")


    args = parser.parse_args()

    # --- Validate Arguments ---
    if not (0 < args.refiner_ratio < 1):
        print("Error: --refiner_ratio must be between 0 and 1 (exclusive).")
        exit(1) # Use exit(1) to indicate error

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating output directory {output_dir}: {e}")
            # Decide if this is fatal, maybe continue if root dir exists? For now, exit.
            exit(1)

    # --- Run Main Pipeline ---
    main(args)