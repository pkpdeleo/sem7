# --- Enhanced Image RAG with ControlNet Generation ---
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
import cv2 # For Canny edge detection fallback if controlnet_aux not used
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import CannyDetector # Recommended preprocessor
import time

# --- Configuration ---
# Models
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
# Use SD v1.5 base and ControlNet for better compatibility unless high VRAM available
BASE_SD_MODEL = "runwayml/stable-diffusion-v1-5"
# Corresponding ControlNet model for Canny
CONTROLNET_MODEL_CANNY = "lllyasviel/sd-controlnet-canny"
# Add other ControlNet model IDs if needed (e.g., depth, pose)
# CONTROLNET_MODEL_DEPTH = "lllyasviel/sd-controlnet-depth"

# Files
DEFAULT_IMAGE_FOLDER = "image_knowledge_base"
VECTOR_DB_INDEX_PATH = "image_rag_controlnet.faiss"
IMAGE_FILENAMES_PATH = "image_filenames_controlnet.pkl"
DEFAULT_OUTPUT_PATH = "generated_controlled_image.png"

# Parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# --- Core RAG Components (Adapted from ep1.py) ---

# Global variables for loaded models to avoid reloading
clip_model = None
clip_processor = None

def load_clip_models():
    """Loads CLIP model and processor."""
    global clip_model, clip_processor
    if clip_model is None:
        print(f"Loading CLIP model: {CLIP_MODEL_NAME}")
        clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
    if clip_processor is None:
        print(f"Loading CLIP processor: {CLIP_MODEL_NAME}")
        clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    return clip_model, clip_processor

def initialize_vector_db(embedding_dimension):
    """Initializes a FAISS index."""
    return faiss.IndexFlatL2(embedding_dimension)

def add_embeddings_to_index(index, embeddings):
    """Adds embeddings to the FAISS index."""
    index.add(embeddings.astype('float32'))

def save_vector_db_index(index, index_path):
    """Saves the FAISS index to disk."""
    faiss.write_index(index, index_path)
    print(f"Vector index SAVED to: {index_path}")

def load_vector_db_index(index_path):
    """Loads a FAISS index from disk."""
    if os.path.exists(index_path):
        print(f"Loading vector index from: {index_path}")
        try:
            return faiss.read_index(index_path)
        except Exception as e:
            print(f"Error loading FAISS index: {e}. Will re-index.")
            return None
    else:
        print("Vector index file not found.")
        return None

def save_image_filenames(filenames, filenames_path):
    """Saves image filenames to disk using pickle."""
    with open(filenames_path, 'wb') as f:
        pickle.dump(filenames, f)
    print(f"Image filenames SAVED to: {filenames_path}")

def load_image_filenames(filenames_path):
    """Loads image filenames from disk using pickle."""
    if os.path.exists(filenames_path):
        print(f"Loading image filenames from: {filenames_path}")
        try:
            with open(filenames_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading filenames pickle: {e}. Will re-index.")
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
            print(f"Image path does not exist: {image_path}")
            return None
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def get_clip_image_embedding(image_pil, model, processor):
    """Generates CLIP embedding for a PIL image."""
    if image_pil is None:
        return None
    try:
        inputs = processor(images=image_pil, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
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
        inputs = processor(text=[query_text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
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
    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                image_paths.append(os.path.join(root, file))

    if not image_paths:
        print(f"No images found in folder: {image_folder}")
        return None, None # Indicate indexing failure

    embeddings_list = []
    image_filenames = []
    processed_count = 0
    total_images = len(image_paths)

    for i, img_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{total_images}: {img_path} ...", end='\r')
        image_pil = load_image(img_path)
        if image_pil:
            embedding = get_clip_image_embedding(image_pil, model, processor)
            if embedding is not None:
                embeddings_list.append(embedding)
                image_filenames.append(img_path)
                processed_count += 1
            else:
                print(f"\nSkipping {img_path} due to embedding error.")
        else:
             print(f"\nSkipping {img_path} due to loading error.")
        # Optional: Clear CUDA cache periodically if memory issues arise
        # if DEVICE == 'cuda' and (i+1) % 50 == 0:
        #     torch.cuda.empty_cache()


    print(f"\nSuccessfully processed {processed_count}/{total_images} images.")

    if not embeddings_list:
        print("No valid image embeddings were generated. Index not created.")
        return None, None # Indicate indexing failure

    embeddings_matrix = np.concatenate(embeddings_list, axis=0)
    embedding_dimension = embeddings_matrix.shape[1]

    index = initialize_vector_db(embedding_dimension)
    add_embeddings_to_index(index, embeddings_matrix)
    save_vector_db_index(index, index_path)
    save_image_filenames(image_filenames, filenames_path)

    print(f"Indexed {len(image_filenames)} images.")
    return index, image_filenames

def retrieve_relevant_images(query_embedding, index, image_filenames, top_k=5):
    """Retrieves top_k most relevant image filenames based on query embedding."""
    print(f"\n--- Retrieving top {top_k} relevant images ---")
    if index is None or index.ntotal == 0:
        print("Vector database index is empty or not loaded.")
        return []
    if image_filenames is None:
        print("Image filenames not loaded.")
        return []
    if index.ntotal != len(image_filenames):
         print(f"Warning: FAISS index size ({index.ntotal}) does not match number of filenames ({len(image_filenames)}). Results might be incorrect. Consider re-indexing.")
         # You might want to handle this more gracefully, e.g., refuse retrieval or try to align

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
            retrieved_filenames.append(image_filenames[idx])
            retrieved_distances.append(distances[0][i])
        else:
             print(f"Warning: Retrieved index {idx} is out of bounds (max valid: {max_valid_index}). Skipping.")

    print(f"Retrieved {len(retrieved_filenames)} filenames.")
    for fname, dist in zip(retrieved_filenames, retrieved_distances):
        print(f"  - {fname} (Distance: {dist:.4f})")
    print("--- Retrieval complete ---")
    return retrieved_filenames

# --- ControlNet Components ---

def get_controlnet_preprocessor(control_type):
    """Gets the appropriate preprocessor for the control type."""
    print(f"Loading preprocessor for control type: {control_type}")
    if control_type == "canny":
        # Using controlnet_aux is generally easier
        try:
            return CannyDetector()
        except ImportError:
            print("Warning: controlnet_aux not installed. Falling back to basic OpenCV Canny.")
            print("Install with: pip install controlnet_aux")
            # Fallback function if controlnet_aux is unavailable
            def opencv_canny(image_pil, low_threshold=100, high_threshold=200):
                 image_np = np.array(image_pil.convert("L")) # Convert to grayscale
                 canny_map_np = cv2.Canny(image_np, low_threshold, high_threshold)
                 # Convert back to PIL Image, ControlNet often expects 3 channels
                 canny_image_pil = Image.fromarray(canny_map_np).convert("RGB")
                 return canny_image_pil
            return opencv_canny

    elif control_type == "depth":
        # Example for depth (would require installing transformers and potentially specific models)
        # from transformers import pipeline
        # return pipeline('depth-estimation')
        raise NotImplementedError("Depth preprocessor not fully implemented here. Requires 'transformers'.")
    # Add other control types (pose, HED, etc.) here
    else:
        raise ValueError(f"Unsupported control_type: {control_type}")

def load_controlnet_pipeline(base_model_id, controlnet_model_id, control_type):
    """Loads the Stable Diffusion ControlNet pipeline."""
    print(f"Loading ControlNet model: {controlnet_model_id}")
    controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=DTYPE)

    print(f"Loading Base Stable Diffusion model: {base_model_id}")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_id,
        controlnet=controlnet,
        torch_dtype=DTYPE,
        safety_checker=None # Optional: Disable safety checker if needed
    )

    # Optimization: Use a faster scheduler
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # Optimization: Enable memory-efficient attention if available/needed
    # pipe.enable_xformers_memory_efficient_attention() # Requires xformers

    pipe = pipe.to(DEVICE)
    print("ControlNet pipeline loaded.")
    return pipe

def generate_controlled_image(pipe, preprocessor, prompt, conditioning_image_pil, output_path,
                              num_inference_steps=30, guidance_scale=7.5,
                              controlnet_conditioning_scale=1.0, seed=None, image_resolution=512):
    """Generates an image using the ControlNet pipeline."""
    print("\n--- Starting Controlled Image Generation ---")
    print(f"Prompt: {prompt}")
    print(f"Conditioning image size: {conditioning_image_pil.size}")
    print(f"Output path: {output_path}")
    print(f"Steps: {num_inference_steps}, Guidance: {guidance_scale}, ControlNet Scale: {controlnet_conditioning_scale}")

    # Prepare conditioning image using the preprocessor
    print("Preprocessing conditioning image...")
    control_image = preprocessor(conditioning_image_pil)
    print(f"Control image size after preprocessing: {control_image.size}")

    # Ensure control image is RGB PIL
    if not isinstance(control_image, Image.Image):
         # Handle cases where preprocessor returns something else (like depth map tensors)
         # This part might need adjustment depending on the specific preprocessor
         if isinstance(control_image, torch.Tensor):
              # Example conversion logic, may need refinement
              control_image = control_image.squeeze().cpu().numpy()
              # Normalize if needed, then convert to PIL
              # ... (conversion logic specific to the preprocessor output format) ...
              raise NotImplementedError("Handling non-PIL preprocessor output needs specific logic.")
         else:
              raise TypeError(f"Preprocessor returned unexpected type: {type(control_image)}")

    control_image = control_image.convert("RGB").resize((image_resolution, image_resolution))
    print(f"Resized control image to: {control_image.size}")

    # Set seed for reproducibility
    generator = None
    if seed is not None:
        print(f"Using seed: {seed}")
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

    # Run the pipeline
    print("Running ControlNet pipeline...")
    start_time = time.time()
    try:
        # Use torch.autocast for mixed precision if on CUDA
        with torch.autocast(DEVICE) if DEVICE == "cuda" else torch.no_grad():
            output = pipe(
                prompt,
                image=control_image, # The preprocessed control image
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=float(controlnet_conditioning_scale), # Ensure float
                generator=generator,
            ).images[0]
        end_time = time.time()
        print(f"Pipeline execution time: {end_time - start_time:.2f} seconds")

        # Save the generated image
        output.save(output_path)
        print(f"Generated image saved to: {output_path}")
        print("--- Generation Complete ---")
        return output, control_image # Return generated and control images

    except Exception as e:
        print(f"\nError during image generation pipeline: {e}")
        if "out of memory" in str(e).lower() and DEVICE == 'cuda':
             print("CUDA out of memory. Try reducing image resolution or batch size (if applicable), or use a smaller model.")
             torch.cuda.empty_cache()
        # You might want to raise the exception again for debugging
        # raise e
        return None, control_image

# --- Main Execution Logic ---
def main(args):
    # --- 1. Setup & Load/Index Data ---
    # Create image folder if it doesn't exist
    if not os.path.exists(args.image_folder):
        print(f"Image folder '{args.image_folder}' not found. Please create it and add images.")
        # Optionally, create it: os.makedirs(args.image_folder)
        return

    # Load or create index and filenames
    index = None
    image_filenames = None
    if not args.reindex:
        index = load_vector_db_index(args.index_path)
        image_filenames = load_image_filenames(args.filenames_path)

    if index is None or image_filenames is None or args.reindex:
        if args.reindex:
             print("Re-indexing requested.")
        else:
             print("Index or filenames not found/loaded. Starting indexing...")
        index, image_filenames = index_image_knowledge_base(
            args.image_folder, args.index_path, args.filenames_path
        )
        if index is None or image_filenames is None:
            print("Indexing failed. Cannot proceed.")
            return
    else:
        print("Successfully loaded existing index and filenames.")
        # Verify index and filenames match
        if index.ntotal != len(image_filenames):
            print(f"Warning: Index size ({index.ntotal}) and filenames count ({len(image_filenames)}) mismatch. Re-indexing is recommended.")
            # Optionally force re-indexing or exit based on strictness needed
            # print("Forcing re-index due to mismatch.")
            # index, image_filenames = index_image_knowledge_base(...)
            # if index is None ... return

    # --- 2. Load Models ---
    clip_model_instance, clip_processor_instance = load_clip_models() # Ensures CLIP is loaded

    # --- 3. Process Query and Retrieve ---
    print(f"\nProcessing query: '{args.query}'")
    query_embedding = get_clip_text_embedding(args.query, clip_model_instance, clip_processor_instance)
    if query_embedding is None:
        print("Failed to generate query embedding.")
        return

    retrieved_image_paths = retrieve_relevant_images(query_embedding, index, image_filenames, top_k=args.top_k)

    if not retrieved_image_paths:
        print("No relevant images found for the query.")
        return

    # --- 4. Select Conditioning Image ---
    # Using the top retrieved image for conditioning
    conditioning_image_path = retrieved_image_paths[0]
    print(f"\nSelected conditioning image: {conditioning_image_path}")
    conditioning_image_pil = load_image(conditioning_image_path)
    if conditioning_image_pil is None:
        print("Failed to load the selected conditioning image.")
        return

    # --- 5. Prepare for Generation ---
    # Determine ControlNet model based on selected type
    if args.control_type == 'canny':
        controlnet_model_id = CONTROLNET_MODEL_CANNY
    # Add elif for other types (depth, pose, etc.)
    # elif args.control_type == 'depth':
    #     controlnet_model_id = CONTROLNET_MODEL_DEPTH
    else:
        print(f"Error: Control type '{args.control_type}' not configured with a model ID.")
        return

    # Load the ControlNet pipeline (loads base SD + ControlNet)
    try:
        pipe = load_controlnet_pipeline(args.base_model, controlnet_model_id, args.control_type)
        preprocessor = get_controlnet_preprocessor(args.control_type)
    except Exception as e:
        print(f"Failed to load pipeline or preprocessor: {e}")
        return

    # --- 6. Generate Image ---
    generated_img, control_img = generate_controlled_image(
        pipe=pipe,
        preprocessor=preprocessor,
        prompt=args.query, # Use the original query as the generation prompt
        conditioning_image_pil=conditioning_image_pil,
        output_path=args.output_path,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_scale,
        seed=args.seed,
        image_resolution=args.resolution
    )

    # Optional: Save the control image for inspection
    if control_img and generated_img:
         control_img_path = os.path.splitext(args.output_path)[0] + "_control_map.png"
         control_img.save(control_img_path)
         print(f"Control map saved to: {control_img_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image RAG with ControlNet Generation")

    # RAG Arguments
    parser.add_argument("query", type=str, help="Text query to search for relevant images and use as generation prompt.")
    parser.add_argument("--image_folder", type=str, default=DEFAULT_IMAGE_FOLDER, help="Folder containing the image knowledge base.")
    parser.add_argument("--index_path", type=str, default=VECTOR_DB_INDEX_PATH, help="Path to save/load the FAISS index file.")
    parser.add_argument("--filenames_path", type=str, default=IMAGE_FILENAMES_PATH, help="Path to save/load the image filenames pickle file.")
    parser.add_argument("--reindex", action="store_true", help="Force re-indexing of the image folder.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of relevant images to retrieve (top one used for conditioning).")

    # ControlNet Arguments
    parser.add_argument("--control_type", type=str, default="canny", choices=["canny"], help="Type of ControlNet conditioning to use. Add more choices as implemented (e.g., 'depth', 'pose').") # Currently only canny implemented easily
    parser.add_argument("--base_model", type=str, default=BASE_SD_MODEL, help="Base Stable Diffusion model ID.")
    # ControlNet model ID is inferred from control_type

    # Generation Arguments
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH, help="Path to save the generated image.")
    parser.add_argument("--num_steps", type=int, default=30, help="Number of diffusion inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale (for classifier-free guidance).")
    parser.add_argument("--controlnet_scale", type=float, default=1.0, help="Conditioning scale for the ControlNet.")
    parser.add_argument("--resolution", type=int, default=512, help="Resolution (height and width) of the generated image.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (optional).")

    args = parser.parse_args()

    # --- Run Main Pipeline ---
    main(args)