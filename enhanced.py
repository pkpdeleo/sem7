# Combined RAG + SDXL + Multi-ControlNet + Enhancements Pipeline Code

# --- Requirements ---
# (Save as requirements.txt and run `pip install -r requirements.txt`)
# torch>=2.1.0 --index-url https://download.pytorch.org/whl/cu121 # Or cu118, cpu depending on your system
# transformers>=4.35.0
# diffusers>=0.25.0
# accelerate>=0.25.0
# faiss-cpu>=1.7.4 # Or faiss-gpu
# numpy>=1.23.0
# Pillow>=9.0.0
# opencv-python-headless>=4.8.0
# controlnet_aux>=0.0.7
# huggingface_hub>=0.19.0
# requests>=2.28.0
# safetensors>=0.4.0
# scikit-learn>=1.0.0 # For cosine_similarity in MMR

# --- Library Imports ---
import torch
import faiss
import numpy as np
from PIL import Image, ImageOps
import os
import requests
from io import BytesIO
import pickle
import time
import gc
import argparse
import sys
import traceback

# Check if running in an environment where __file__ is defined
try:
    _PROJECT_ROOT_GUESS = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _PROJECT_ROOT_GUESS = os.getcwd() # Fallback to current working directory
    print(f"Warning: __file__ not defined. Using current working directory as project root: {_PROJECT_ROOT_GUESS}")


from transformers import (
    CLIPModel, CLIPProcessor,
    BlipProcessor, BlipForConditionalGeneration # For captioning
)
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLImg2ImgPipeline,
    ControlNetModel,
    AutoencoderKL,
    UniPCMultistepScheduler
)
from controlnet_aux import CannyDetector, MidasDetector
from sklearn.metrics.pairwise import cosine_similarity


# ==============================================================================
# --- CONFIGURATION (from config.py) ---
# ==============================================================================

# --- Models ---
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
BASE_SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
REFINER_SDXL_MODEL = "stabilityai/stable-diffusion-xl-refiner-1.0"
VAE_SDXL_MODEL = "madebyollin/sdxl-vae-fp16-fix"
CONTROLNET_MODEL_DEPTH_SDXL = "diffusers/controlnet-depth-sdxl-1.0"
CONTROLNET_MODEL_CANNY_SDXL = "diffusers/controlnet-canny-sdxl-1.0"
PREPROCESSOR_ANNOTATOR_REPO = "lllyasviel/Annotators"
CAPTIONING_MODEL_ID = "Salesforce/blip-image-captioning-large" # Example captioning model

# --- File Paths ---
# Use the guessed project root directory
DEFAULT_IMAGE_FOLDER = os.path.join(_PROJECT_ROOT_GUESS, "image_knowledge_base_xl")
VECTOR_DB_INDEX_PATH = os.path.join(_PROJECT_ROOT_GUESS, "image_rag_sdxl_controlnet.faiss")
IMAGE_FILENAMES_PATH = os.path.join(_PROJECT_ROOT_GUESS, "image_filenames_sdxl_controlnet.pkl")
IMAGE_EMBEDDINGS_PATH = os.path.join(_PROJECT_ROOT_GUESS, "image_rag_sdxl_controlnet_embeddings.npy") # For MMR
DEFAULT_OUTPUT_PATH = os.path.join(_PROJECT_ROOT_GUESS, "generated_sdxl_realistic.png")

# --- Hardware & Performance ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE_VAE = torch.float16
DTYPE_CONTROL = torch.float16
DTYPE_UNET = torch.float16
DTYPE_REFINER = torch.float16
DTYPE_CAPTION = torch.float16 # Use float16 for caption model if possible
DEFAULT_ENABLE_CPU_OFFLOAD = False
DEFAULT_ENABLE_ATTENTION_SLICING = False

# --- Default Generation Parameters ---
DEFAULT_RESOLUTION = 1024
DEFAULT_NUM_STEPS = 40
DEFAULT_GUIDANCE_SCALE = 7.0
DEFAULT_REFINER_RATIO = 0.2
DEFAULT_DEPTH_SCALE = 0.5
DEFAULT_CANNY_SCALE = 0.5
DEFAULT_SEED = None

# --- RAG Parameters ---
DEFAULT_TOP_K_RAG = 2
DEFAULT_RETRIEVAL_MODE = "topk" # or "mmr"
DEFAULT_MMR_LAMBDA = 0.5
DEFAULT_MMR_TOP_N = 20
DEFAULT_AUGMENT_PROMPT = False

# --- Prompting ---
STYLES = {
    "default": ("", ""),
    "photorealistic": ("photorealistic DSLR photo, photograph, high detail, sharp focus, natural lighting, cinematic composition, ",
                       " painting, drawing, illustration, sketch, cartoon, anime, 3D render, CGI, unrealistic, artificial, blurry, low quality, noisy, text, signature, watermark"),
    "cinematic": ("cinematic film still, dramatic lighting, shallow depth of field, high detail, sharp focus, ",
                  " painting, drawing, illustration, sketch, cartoon, anime, 3D render, CGI, unrealistic, artificial, blurry, low quality, noisy, text, signature, watermark, boring"),
    "anime": ("anime artwork, anime style, key visual, vibrant colors, sharp lines, detailed background, ",
              " photo, photograph, realistic, 3D render, CGI, blurry, low quality, text, signature, watermark"),
    "illustration": ("illustration, detailed illustration, vibrant colors, high fantasy, intricate details, painterly style, ",
                    " photo, photograph, realistic, 3D render, CGI, blurry, low quality, text, signature, watermark"),
}
DEFAULT_STYLE = "photorealistic"
DEFAULT_NEGATIVE_PROMPT_PREFIX = "worst quality, low quality, jpeg artifacts, blurry, noisy, text, signature, watermark, username, artist name, deformed, mutated, disfigured, morbid, mutilated, extra limbs, missing limbs, duplicate"


# ==============================================================================
# --- RAG UTILITIES (from rag_utils.py) ---
# ==============================================================================

# Global cache for CLIP models
_clip_model_cache = None
_clip_processor_cache = None

def load_clip_models():
    """Loads CLIP model and processor, using a cache."""
    global _clip_model_cache, _clip_processor_cache
    if _clip_model_cache is None:
        print(f"Loading CLIP model: {CLIP_MODEL_NAME}")
        try:
            _clip_model_cache = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
            _clip_model_cache.eval()
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            raise
    if _clip_processor_cache is None:
        print(f"Loading CLIP processor: {CLIP_MODEL_NAME}")
        try:
            _clip_processor_cache = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        except Exception as e:
            print(f"Error loading CLIP processor: {e}")
            raise
    return _clip_model_cache, _clip_processor_cache

def load_image(image_path_or_url):
    """Loads an image from path or URL into PIL format."""
    try:
        if str(image_path_or_url).startswith("http://") or str(image_path_or_url).startswith("https://"):
            response = requests.get(image_path_or_url, timeout=15)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
        elif os.path.exists(image_path_or_url):
            img = Image.open(image_path_or_url).convert("RGB")
        else:
            print(f"Error: Image path does not exist: {image_path_or_url}")
            return None
        return img
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image URL {image_path_or_url}: {e}")
        return None
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path_or_url}")
        return None
    except Exception as e:
        print(f"Error loading image {image_path_or_url}: {e}")
        return None

@torch.no_grad()
def get_clip_image_embedding(image_pil):
    """Generates normalized CLIP embedding for a PIL image."""
    if image_pil is None: return None
    model, processor = load_clip_models()
    try:
        inputs = processor(images=image_pil, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model.get_image_features(**inputs)
        embedding = outputs.cpu().numpy()
        norm = np.linalg.norm(embedding, axis=-1, keepdims=True)
        norm[norm == 0] = 1e-10 # Avoid division by zero
        normalized_embedding = embedding / norm
        return normalized_embedding
    except Exception as e:
        print(f"Error generating CLIP image embedding: {e}")
        return None

@torch.no_grad()
def get_clip_image_embedding_batch(image_pils: list):
    """Generates normalized CLIP embeddings for a batch of PIL images."""
    if not image_pils: return []
    model, processor = load_clip_models()
    try:
        valid_images = [img for img in image_pils if img is not None]
        if not valid_images: return [None] * len(image_pils)

        inputs = processor(images=valid_images, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model.get_image_features(**inputs)
        embeddings = outputs.cpu().numpy()

        norm = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        norm[norm == 0] = 1e-10
        normalized_embeddings = embeddings / norm

        result_list = []
        valid_idx = 0
        for img in image_pils:
            if img is not None:
                result_list.append(normalized_embeddings[valid_idx])
                valid_idx += 1
            else:
                result_list.append(None)
        return result_list
    except Exception as e:
        print(f"Error generating batch CLIP image embeddings: {e}")
        return [None] * len(image_pils)

@torch.no_grad()
def get_clip_text_embedding(query_text):
    """Generates normalized CLIP embedding for a text query."""
    model, processor = load_clip_models()
    try:
        inputs = processor(text=[query_text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model.get_text_features(**inputs)
        embedding = outputs.cpu().numpy()
        norm = np.linalg.norm(embedding, axis=-1, keepdims=True)
        norm[norm == 0] = 1e-10
        normalized_embedding = embedding / norm
        return normalized_embedding
    except Exception as e:
        print(f"Error generating CLIP text embedding for '{query_text}': {e}")
        return None

# --- FAISS Indexing Functions ---

def initialize_vector_db(embedding_dimension):
    print(f"Initializing FAISS IndexFlatL2 with dimension {embedding_dimension}")
    return faiss.IndexFlatL2(embedding_dimension)

def add_embeddings_to_index(index, embeddings):
    if embeddings.ndim == 1:
        embeddings = np.expand_dims(embeddings, axis=0)
    print(f"Adding {embeddings.shape[0]} embeddings to FAISS index...")
    index.add(embeddings.astype('float32'))
    print(f"Index size after adding: {index.ntotal}")

def save_vector_db_index(index, index_path):
    print(f"Saving FAISS index to: {index_path}")
    try:
        faiss.write_index(index, index_path)
        print("FAISS index saved successfully.")
    except Exception as e:
        print(f"Error saving FAISS index: {e}")

def load_vector_db_index(index_path):
    if not os.path.exists(index_path):
        print(f"FAISS index file not found at: {index_path}")
        return None
    print(f"Loading FAISS index from: {index_path}")
    try:
        index = faiss.read_index(index_path)
        print(f"FAISS index loaded successfully with {index.ntotal} vectors.")
        return index
    except Exception as e:
        print(f"Error loading FAISS index: {e}. Consider re-indexing.")
        return None

def save_image_filenames(filenames, filenames_path):
    print(f"Saving image filenames list to: {filenames_path}")
    try:
        with open(filenames_path, 'wb') as f:
            pickle.dump(filenames, f)
        print("Image filenames saved successfully.")
    except Exception as e:
        print(f"Error saving image filenames: {e}")

def load_image_filenames(filenames_path):
    if not os.path.exists(filenames_path):
        print(f"Image filenames file not found at: {filenames_path}")
        return None
    print(f"Loading image filenames from: {filenames_path}")
    try:
        with open(filenames_path, 'rb') as f:
            filenames = pickle.load(f)
        print(f"Image filenames loaded successfully with {len(filenames)} entries.")
        return filenames
    except Exception as e:
        print(f"Error loading image filenames pickle: {e}. Consider re-indexing.")
        return None

# --- Indexing Process ---

def index_image_knowledge_base(image_folder, index_path, filenames_path, embeddings_path):
    """
    Indexes images: generates embeddings, creates FAISS index, saves index, filenames, and embeddings.
    """
    print(f"\n--- Starting Image Indexing Process ---")
    print(f"Image Folder: {image_folder}")

    if not os.path.isdir(image_folder):
        print(f"Error: Image folder '{image_folder}' not found.")
        return None, None, None

    image_paths = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
    print("Scanning for images...")
    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith(valid_extensions):
                full_path = os.path.join(root, file)
                image_paths.append(os.path.abspath(full_path))

    if not image_paths:
        print(f"No images found in folder: {image_folder}")
        return None, None, None
    print(f"Found {len(image_paths)} potential image files.")

    embeddings_list = []
    image_filenames_stored = []
    processed_count = 0
    skipped_count = 0
    total_images = len(image_paths)
    start_time = time.time()

    load_clip_models() # Load once

    print("Generating embeddings...")
    batch_size = 32 # Process in batches
    for i in range(0, total_images, batch_size):
        batch_paths = image_paths[i:min(i+batch_size, total_images)]
        print(f"Processing images {i+1}-{min(i+batch_size, total_images)}/{total_images}...", end='\r')
        batch_pils = [load_image(p) for p in batch_paths]
        batch_embeddings = get_clip_image_embedding_batch(batch_pils)

        for j, emb in enumerate(batch_embeddings):
             current_path = batch_paths[j]
             if emb is not None:
                 embeddings_list.append(emb)
                 image_filenames_stored.append(current_path)
                 processed_count += 1
             else:
                 print(f"\nSkipping {os.path.basename(current_path)} - embedding/loading failed.")
                 skipped_count += 1
        # Clear memory
        del batch_pils, batch_embeddings
        if DEVICE == 'cuda': torch.cuda.empty_cache()


    end_time = time.time()
    print(f"\n--- Indexing Summary ---")
    print(f"Successfully processed: {processed_count}/{total_images} images.")
    print(f"Skipped: {skipped_count} images.")
    print(f"Total time: {end_time - start_time:.2f} seconds.")

    if not embeddings_list:
        print("No valid embeddings generated. Index cannot be created.")
        return None, None, None

    embeddings_matrix = np.vstack(embeddings_list).astype('float32') # Ensure float32
    embedding_dimension = embeddings_matrix.shape[1]

    index = initialize_vector_db(embedding_dimension)
    add_embeddings_to_index(index, embeddings_matrix)

    save_vector_db_index(index, index_path)
    save_image_filenames(image_filenames_stored, filenames_path)

    # --- SAVE EMBEDDINGS FOR MMR ---
    try:
        print(f"Saving normalized embeddings matrix to: {embeddings_path}")
        np.save(embeddings_path, embeddings_matrix) # Save the normalized float32 matrix
        print("Embeddings matrix saved successfully.")
    except Exception as e:
        print(f"Error saving embeddings matrix: {e}")
    # --- END SAVE EMBEDDINGS ---

    print("--- Indexing Complete ---")
    return index, image_filenames_stored, embeddings_matrix # Return embeddings too

# --- Retrieval ---

def retrieve_relevant_images(query_embedding, index, image_filenames, top_k=1):
    """Retrieves top_k most relevant image filenames based on query embedding (simple top-k)."""
    print(f"\n--- Retrieving Top {top_k} Relevant Images (Simple Top-K) ---")
    if index is None or image_filenames is None or index.ntotal == 0 or not image_filenames:
        print("Error: Index or filenames empty/not loaded.")
        return []
    if index.ntotal != len(image_filenames):
        print(f"Warning: FAISS index size ({index.ntotal}) != filenames ({len(image_filenames)}). Re-index recommended.")

    try:
        k_search = min(top_k, index.ntotal)
        print(f"Searching index for {k_search} nearest neighbors...")
        distances, indices = index.search(query_embedding.astype('float32'), k_search)
        print(f"FAISS search complete. Indices found: {indices[0]}")
    except Exception as e:
        print(f"Error during FAISS index search: {e}")
        return []

    retrieved_filenames = []
    max_valid_index = len(image_filenames) - 1
    for i, idx in enumerate(indices[0]):
        if 0 <= idx <= max_valid_index:
            retrieved_filenames.append(image_filenames[idx])
        else:
            print(f"Warning: Retrieved index {idx} out of bounds (max: {max_valid_index}). Skipping.")

    print("Retrieved candidates (Top-K):")
    for fname in retrieved_filenames:
        print(f"  - {os.path.basename(fname)}")
    print("--- Top-K Retrieval Complete ---")
    return retrieved_filenames

# --- MMR Retrieval ---
def calculate_embedding_similarity(emb1, emb2):
    """Calculates cosine similarity between two embedding vectors."""
    if emb1 is None or emb2 is None: return 0.0
    emb1 = emb1.reshape(1, -1)
    emb2 = emb2.reshape(1, -1)
    # Assumes embeddings are already normalized, so dot product is cosine similarity
    # return np.dot(emb1.flatten(), emb2.flatten())
    # Using sklearn's cosine_similarity is safer if normalization isn't guaranteed
    return cosine_similarity(emb1, emb2)[0][0]

def retrieve_relevant_images_mmr(query_embedding, index, image_filenames, all_image_embeddings_normalized, top_k=2, top_n=20, lambda_mult=0.5):
    """Retrieves top_k diverse images using Maximal Marginal Relevance (MMR)."""
    print(f"\n--- Retrieving Top {top_k} Diverse Images using MMR (lambda={lambda_mult}) ---")
    # (Error checking as before...)
    if index is None or image_filenames is None or all_image_embeddings_normalized is None or index.ntotal == 0 or len(image_filenames) == 0 or len(all_image_embeddings_normalized) == 0:
         print("Error: Index, filenames, or embeddings are empty/not provided for MMR.")
         return []
    if index.ntotal != len(image_filenames) or index.ntotal != len(all_image_embeddings_normalized):
         print(f"Warning: Mismatch between index ({index.ntotal}), filenames ({len(image_filenames)}), and embeddings count ({len(all_image_embeddings_normalized)}). Re-index recommended.")
         return []


    k_search = min(top_n, index.ntotal)
    print(f"Searching index for initial {k_search} candidates...")
    try:
        distances, indices = index.search(query_embedding.astype('float32'), k_search)
        initial_indices = indices[0]
    except Exception as e:
        print(f"Error during FAISS index search: {e}")
        return []

    if not initial_indices.size: return []

    # Get data for initial candidates, ensuring indices are valid
    candidate_data = []
    for i in initial_indices:
        if 0 <= i < len(all_image_embeddings_normalized) and 0 <= i < len(image_filenames):
             emb = all_image_embeddings_normalized[i]
             fname = image_filenames[i]
             if emb is not None: # Check embedding validity
                 candidate_data.append({'original_index': i, 'filename': fname, 'embedding': emb})
        else:
            print(f"Warning: Initial candidate index {i} out of bounds. Skipping.")

    if not candidate_data:
        print("No valid initial candidates found after filtering.")
        return []

    # Calculate relevance (similarity to query)
    for candidate in candidate_data:
        candidate['relevance'] = calculate_embedding_similarity(query_embedding, candidate['embedding'])

    # MMR Selection Loop
    selected_indices_original = [] # Store original indices
    selected_embeddings = []
    candidate_pool = candidate_data.copy()

    while len(selected_indices_original) < min(top_k, len(candidate_pool)):
        best_score = -np.inf
        best_candidate = None
        candidate_to_remove = None

        for candidate in candidate_pool:
            relevance_score = candidate['relevance']
            diversity_penalty = 0.0
            if selected_embeddings:
                similarities_to_selected = [calculate_embedding_similarity(candidate['embedding'], emb_selected) for emb_selected in selected_embeddings]
                diversity_penalty = np.max(similarities_to_selected) if similarities_to_selected else 0.0

            mmr_score = lambda_mult * relevance_score - (1.0 - lambda_mult) * diversity_penalty

            if mmr_score > best_score:
                best_score = mmr_score
                best_candidate = candidate

        if best_candidate is not None:
            selected_indices_original.append(best_candidate['original_index'])
            selected_embeddings.append(best_candidate['embedding'])
            # Find and remove the best candidate from the pool for the next iteration
            candidate_pool = [c for c in candidate_pool if c['original_index'] != best_candidate['original_index']]
        else:
            break # No more candidates or error

    # Retrieve filenames
    final_filenames = [image_filenames[idx] for idx in selected_indices_original] # Assumes indices are valid

    print(f"Selected {len(final_filenames)} diverse filenames using MMR:")
    for fname in final_filenames:
        print(f"  - {os.path.basename(fname)}")
    print("--- MMR Retrieval Complete ---")
    return final_filenames


def load_all_image_embeddings(embeddings_path):
    """Loads all pre-computed image embeddings from a .npy file."""
    print(f"Loading all image embeddings from: {embeddings_path}")
    if not os.path.exists(embeddings_path):
        print(f"Error: Embeddings file not found at {embeddings_path}.")
        print("Please run indexing first to generate embeddings.")
        return None
    try:
        all_embeddings = np.load(embeddings_path)
        # Ensure float32, normalization should have happened during saving
        all_embeddings = all_embeddings.astype('float32')
        print(f"Loaded {len(all_embeddings)} embeddings.")
        return all_embeddings
    except Exception as e:
        print(f"Error loading embeddings file {embeddings_path}: {e}")
        return None

# ==============================================================================
# --- SDXL PIPELINE UTILITIES (from sdxl_pipeline_utils.py) ---
# ==============================================================================

def get_sdxl_controlnet_preprocessor(control_type):
    """Gets the appropriate preprocessor instance for a given control type."""
    print(f"Loading preprocessor for control type: {control_type}")
    try:
        if control_type == "canny":
            return CannyDetector()
        elif control_type == "depth":
            print(f"  Loading MidasDetector (from {PREPROCESSOR_ANNOTATOR_REPO})")
            detector = MidasDetector.from_pretrained(PREPROCESSOR_ANNOTATOR_REPO)
            if hasattr(detector, 'model') and hasattr(detector.model, 'to'):
                 detector.model.to('cpu') # Keep preprocessor models on CPU initially
            print("  MidasDetector loaded.")
            return detector
        # Add other control types here if needed
        else:
            raise ValueError(f"Unsupported control_type for preprocessor: {control_type}")
    except Exception as e:
        print(f"Error loading preprocessor for {control_type}: {e}")
        raise

def load_sdxl_controlnet_pipeline(base_model_id, refiner_model_id, vae_model_id,
                                controlnet_items: list, # List of tuples: (id, type_name)
                                enable_cpu_offload=False, enable_attention_slicing=False):
    """Loads the Stable Diffusion XL Multi-ControlNet pipeline with VAE and Refiner."""
    print("\n--- Loading SDXL Multi-ControlNet Pipeline ---")
    start_time = time.time()
    pipe = None
    refiner_pipe = None
    controlnets = []

    # 1. Load ControlNets
    print("Loading ControlNet models...")
    for cn_id, cn_type in controlnet_items:
        print(f"  Loading ControlNet ({cn_type}): {cn_id}")
        try:
            cn = ControlNetModel.from_pretrained(cn_id, torch_dtype=DTYPE_CONTROL, use_safetensors=True)
            print(f"    Loaded {cn_id} (safetensors)")
            controlnets.append(cn)
        except EnvironmentError:
            print(f"    Safetensors not found for {cn_id}, trying .bin")
            try:
                cn = ControlNetModel.from_pretrained(cn_id, torch_dtype=DTYPE_CONTROL, use_safetensors=False)
                print(f"    Loaded {cn_id} (.bin)")
                controlnets.append(cn)
            except Exception as e_bin:
                raise RuntimeError(f"Could not load ControlNet model: {cn_id}") from e_bin
        except Exception as e:
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
            raise RuntimeError(f"Could not load VAE model: {vae_model_id}") from e_vae_bin
    except Exception as e_vae:
        raise RuntimeError(f"Could not load VAE model: {vae_model_id}") from e_vae

    # 3. Load Refiner Pipeline
    print(f"Loading Refiner Pipeline: {refiner_model_id}")
    try:
        refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            refiner_model_id, vae=vae, torch_dtype=DTYPE_REFINER, use_safetensors=True,
            variant="fp16" if DTYPE_REFINER == torch.float16 else None, add_watermarker=False,
        )
        print(f"  Loaded Refiner {refiner_model_id} (safetensors)")
    except EnvironmentError:
        print(f"  Safetensors not found for Refiner {refiner_model_id}, trying .bin")
        try:
            refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                refiner_model_id, vae=vae, torch_dtype=DTYPE_REFINER, use_safetensors=False,
                variant="fp16" if DTYPE_REFINER == torch.float16 else None, add_watermarker=False,
            )
            print(f"  Loaded Refiner {refiner_model_id} (.bin)")
        except Exception as e_refiner_bin:
            raise RuntimeError(f"Could not load Refiner model: {refiner_model_id}") from e_refiner_bin
    except Exception as e_refiner:
        raise RuntimeError(f"Could not load Refiner model: {refiner_model_id}") from e_refiner

    # 4. Load Base ControlNet Pipeline
    print(f"Loading Base SDXL ControlNet Pipeline: {base_model_id}")
    try:
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_model_id, controlnet=controlnets, vae=vae, torch_dtype=DTYPE_UNET,
            use_safetensors=True, variant="fp16" if DTYPE_UNET == torch.float16 else None,
            add_watermarker=False,
        )
        print(f"  Loaded Base {base_model_id} (safetensors)")
    except EnvironmentError:
        print(f"  Safetensors not found for Base {base_model_id}, trying .bin")
        try:
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                base_model_id, controlnet=controlnets, vae=vae, torch_dtype=DTYPE_UNET,
                use_safetensors=False, variant="fp16" if DTYPE_UNET == torch.float16 else None,
                add_watermarker=False,
            )
            print(f"  Loaded Base {base_model_id} (.bin)")
        except Exception as e_base_bin:
            raise RuntimeError(f"Could not load Base SDXL model: {base_model_id}") from e_base_bin
    except Exception as e_base:
        raise RuntimeError(f"Could not load Base SDXL model: {base_model_id}") from e_base

    # 5. Apply Optimizations
    print("Applying optimizations...")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if refiner_pipe:
         refiner_pipe.scheduler = UniPCMultistepScheduler.from_config(refiner_pipe.scheduler.config)
    print("  Using UniPCMultistepScheduler.")

    if enable_attention_slicing:
        print("  Enabling Attention Slicing...")
        pipe.enable_attention_slicing()
        if refiner_pipe: refiner_pipe.enable_attention_slicing()

    if enable_cpu_offload and DEVICE == 'cuda':
        print("  Enabling Model CPU Offload (requires 'accelerate')...")
        try:
            if refiner_pipe:
                print("    Offloading Refiner pipeline...")
                refiner_pipe.enable_model_cpu_offload()
                gc.collect(); torch.cuda.empty_cache()
                print("    Refiner offloaded.")
                time.sleep(1)
            print("    Offloading Base pipeline...")
            pipe.enable_model_cpu_offload()
            gc.collect(); torch.cuda.empty_cache()
            print("    Base pipeline offloaded.")
            print("  CPU Offload enabled successfully.")
        except ImportError:
            print("  Warning: 'accelerate' not found. CPU offload unavailable. Moving models to GPU.")
            pipe.to(DEVICE);
            if refiner_pipe: refiner_pipe.to(DEVICE)
        except Exception as e_offload:
            print(f"  Warning: Failed to enable CPU offload: {e_offload}. Moving models to GPU.")
            pipe.to(DEVICE);
            if refiner_pipe: refiner_pipe.to(DEVICE)
    elif DEVICE == 'cuda':
        print(f"  Moving pipeline components to GPU ({DEVICE})...")
        pipe.to(DEVICE)
        if refiner_pipe: refiner_pipe.to(DEVICE)
        print("  Pipeline components moved to GPU.")
    else:
        print("  Running on CPU (expect very slow performance).")

    end_time = time.time()
    print(f"--- SDXL Pipeline Loading Complete (Time: {end_time - start_time:.2f} seconds) ---")
    return pipe, refiner_pipe


# ==============================================================================
# --- IMAGE GENERATION (from generation.py) ---
# ==============================================================================

def preprocess_control_image(image_pil, preprocessor, control_type, target_size, device_for_preprocessor):
    """Applies preprocessing and resizes the control image."""
    print(f"  Preprocessing for {control_type}...")
    preprocessor_needs_move = False
    original_device = 'cpu'
    model_to_move = None

    # Identify the actual model attribute within potentially nested preprocessor objects
    if hasattr(preprocessor, 'model') and hasattr(preprocessor.model, 'to') and hasattr(preprocessor.model, 'parameters'):
        model_to_move = preprocessor.model
    elif control_type == 'depth' and hasattr(preprocessor, 'model') and hasattr(preprocessor.model, 'pretrained') and hasattr(preprocessor.model.pretrained, 'model'): # Specific check for Midas structure
        model_to_move = preprocessor.model.pretrained.model
    # Add checks for other preprocessors if needed (e.g., OpenPose has multiple models)

    if model_to_move is not None:
        try: original_device = next(model_to_move.parameters()).device
        except StopIteration: pass # Model might have no parameters

        if str(original_device) != str(device_for_preprocessor):
             print(f"    Moving {control_type} preprocessor model from {original_device} to {device_for_preprocessor}")
             model_to_move.to(device_for_preprocessor)
             preprocessor_needs_move = True

    # Run preprocessing
    control_image = preprocessor(image_pil) # Add specific args if needed based on detector type

    if isinstance(control_image, np.ndarray): control_image = Image.fromarray(control_image)
    control_image = control_image.convert("RGB")

    print(f"    Raw {control_type} map size: {control_image.size}")
    if control_image.size != target_size:
        print(f"    Resizing {control_type} map from {control_image.size} to {target_size}...")
        control_image = control_image.resize(target_size, Image.Resampling.LANCZOS)
        print(f"    Resized {control_type} map size: {control_image.size}")

    # Move back to CPU if offloading active and it was moved to GPU
    if preprocessor_needs_move and device_for_preprocessor != 'cpu':
        print(f"    Moving {control_type} preprocessor model back to CPU")
        model_to_move.to('cpu')

    gc.collect()
    if DEVICE == 'cuda': torch.cuda.empty_cache()
    return control_image

@torch.no_grad()
def generate_sdxl_controlled_image(
    pipe, refiner_pipe, preprocessors: dict, prompt: str, negative_prompt: str,
    conditioning_image_pils: list, control_types: list, controlnet_scales: list,
    output_path: str, num_inference_steps: int = 50, guidance_scale: float = 7.5,
    seed: int = None, image_resolution: int = 1024, refiner_steps_ratio: float = 0.2
    ):
    """Generates an image using the SDXL ControlNet pipeline and Refiner."""
    print("\n--- Starting SDXL Controlled Image Generation ---")
    # (Print parameters...)
    print(f"Prompt: {prompt[:100]}...") # Print truncated prompt
    print(f"Negative Prompt: {negative_prompt[:100]}...")
    print(f"Output path: {output_path}")
    # ... other params

    if not (0 < refiner_steps_ratio < 1):
        print("Warning: refiner_steps_ratio invalid, using 0.2")
        refiner_steps_ratio = 0.2
    if len(conditioning_image_pils) != len(control_types) or len(control_types) != len(controlnet_scales):
        print("Error: Mismatch lengths: conditioning images, control types, scales.")
        return None, []

    # --- 1. Prepare Conditioning Images ---
    control_images_processed = []
    control_images_saved = {}
    print("Preprocessing conditioning images...")
    target_size = (image_resolution, image_resolution)
    is_offloaded = hasattr(pipe, 'hf_device_map') and pipe.hf_device_map is not None
    device_for_preprocessor = 'cpu' if is_offloaded else DEVICE
    print(f"Device for preprocessors: {device_for_preprocessor}")

    for i, (img_pil, ctype) in enumerate(zip(conditioning_image_pils, control_types)):
        if img_pil is None or ctype not in preprocessors:
            print(f"Error: Invalid image or no preprocessor for {ctype}.")
            return None, []
        try:
            control_image = preprocess_control_image(
                img_pil, preprocessors[ctype], ctype, target_size, device_for_preprocessor
            )
            control_images_processed.append(control_image)
            control_images_saved[ctype] = control_image
        except Exception as e:
            print(f"Error during preprocessing for {ctype}: {e}")
            traceback.print_exc()
            return None, []

    if not all(img.size == target_size for img in control_images_processed):
        print(f"Error: Not all control images resized to {target_size}.")
        return None, list(control_images_saved.values())

    # --- 2. Setup Generator ---
    generator = None
    if seed is not None:
        print(f"Using seed: {seed}")
        generator_device = 'cpu' if is_offloaded or DEVICE == 'cpu' else DEVICE
        generator = torch.Generator(device=generator_device).manual_seed(seed)

    # --- 3. Run Base Pipeline ---
    print("Running SDXL ControlNet pipeline (Base)...")
    start_time_base = time.time()
    image_latents = None
    denoising_end_for_base = 1.0 - refiner_steps_ratio

    try:
        pipe_device = pipe.device if hasattr(pipe, 'device') else DEVICE
        print(f"  Base pipeline expected device: {pipe_device}")
        output = pipe(
            prompt=prompt, negative_prompt=negative_prompt, image=control_images_processed,
            controlnet_conditioning_scale=controlnet_scales, num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale, generator=generator, output_type="latent",
            denoising_end=denoising_end_for_base
        )
        image_latents = output.images
        end_time_base = time.time()
        print(f"Base pipeline completed in {end_time_base - start_time_base:.2f} seconds.")
    except Exception as e_base:
        print(f"\nError during Base pipeline execution: {e_base}")
        traceback.print_exc()
        if "out of memory" in str(e_base).lower() and DEVICE == 'cuda': print("CUDA OOM during base pass. Try --enable_cpu_offload, lower --resolution, or --enable_attention_slicing.")
        torch.cuda.empty_cache(); gc.collect()
        return None, list(control_images_saved.values())

    # --- 4. Run Refiner Pipeline ---
    output_image = None
    if refiner_pipe and refiner_steps_ratio > 0:
        print("Running SDXL Refiner pipeline...")
        start_time_refiner = time.time()
        try:
            refiner_device = refiner_pipe.device if hasattr(refiner_pipe, 'device') else DEVICE
            print(f"  Refiner pipeline expected device: {refiner_device}")
            if image_latents.device != refiner_device:
                print(f"  Moving latents from {image_latents.device} to {refiner_device} for refiner.")
                image_latents = image_latents.to(refiner_device)

            output_image = refiner_pipe(
                prompt=prompt, negative_prompt=negative_prompt, image=image_latents,
                num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                denoising_start=denoising_end_for_base, generator=generator, # Re-use generator
            ).images[0]
            end_time_refiner = time.time()
            print(f"Refiner pipeline completed in {end_time_refiner - start_time_refiner:.2f} seconds.")
        except Exception as e_refiner:
            print(f"\nError during Refiner pipeline execution: {e_refiner}")
            traceback.print_exc()
            if "out of memory" in str(e_refiner).lower() and DEVICE == 'cuda': print("CUDA OOM during refiner pass.")
            torch.cuda.empty_cache(); gc.collect()
            print("Refiner failed. Attempting to decode latents from base pipe as fallback...")
            # Fallback decoding moved to after the 'if refiner_pipe' block
            output_image = None # Ensure fallback happens

    # --- 5. Decode Latents (If Refiner Skipped or Failed) ---
    if output_image is None:
        if refiner_steps_ratio == 0: print("Skipping Refiner. Decoding latents from base pipeline...")
        # Else: Fallback from refiner failure already printed message

        try:
            vae_device = pipe.vae.device if hasattr(pipe, 'vae') else DEVICE
            if image_latents.device != vae_device:
                print(f"  Moving latents from {image_latents.device} to {vae_device} for VAE decoding.")
                image_latents = image_latents.to(vae_device)

            image_latents = image_latents / pipe.vae.config.scaling_factor
            output_image = pipe.vae.decode(image_latents, return_dict=False)[0]
            output_image = pipe.image_processor.postprocess(output_image, output_type="pil")[0]
            print("Latents decoded successfully.")
        except Exception as e_decode:
            print(f"\nError decoding latents: {e_decode}")
            traceback.print_exc()
            return None, list(control_images_saved.values())

    # --- 6. Save Output ---
    try:
        output_image.save(output_path)
        print(f"Generated image saved successfully to: {output_path}")
    except Exception as e_save:
        print(f"Error saving generated image to {output_path}: {e_save}")
        # Return image even if save fails
    print("--- Generation Complete ---")
    return output_image, list(control_images_saved.values())


# ==============================================================================
# --- EVALUATION UTILITY (from evaluate_similarity.py) ---
# ==============================================================================

def calculate_clip_similarity(image_path1: str, image_path2: str):
    """Calculates the cosine similarity between the CLIP embeddings of two images."""
    print(f"Calculating similarity between '{os.path.basename(image_path1)}' and '{os.path.basename(image_path2)}'")
    try:
        load_clip_models() # Ensure models are loaded
    except Exception as e:
        print(f"Fatal: Could not load CLIP model/processor for evaluation. Error: {e}")
        return None

    img1_pil = load_image(image_path1)
    img2_pil = load_image(image_path2)
    if img1_pil is None or img2_pil is None: return None

    print("Generating embeddings...")
    emb1 = get_clip_image_embedding(img1_pil)
    emb2 = get_clip_image_embedding(img2_pil)
    if emb1 is None or emb2 is None: return None

    emb1_flat = emb1.flatten()
    emb2_flat = emb2.flatten()
    # Use sklearn for robustness, handles potential minor normalization issues
    similarity = cosine_similarity(emb1_flat.reshape(1,-1), emb2_flat.reshape(1,-1))[0][0]
    # similarity = np.dot(emb1_flat, emb2_flat) # Assumes perfect normalization

    similarity = np.clip(similarity, -1.0, 1.0)
    print(f"Embeddings generated. Similarity calculated.")
    return float(similarity)

def run_similarity_evaluation(gen_path, src1_path, src2_path):
    """Runs the similarity comparison and prints results."""
    print("\n--- Starting Similarity Evaluation ---")
    similarity_gen_src1 = calculate_clip_similarity(gen_path, src1_path)
    if similarity_gen_src1 is not None:
        print(f"==> CLIP Similarity (Generated vs Source 1): {similarity_gen_src1:.4f}")

    similarity_gen_src2 = calculate_clip_similarity(gen_path, src2_path)
    if similarity_gen_src2 is not None:
        print(f"==> CLIP Similarity (Generated vs Source 2): {similarity_gen_src2:.4f}")

    similarity_src1_src2 = calculate_clip_similarity(src1_path, src2_path)
    if similarity_src1_src2 is not None:
        print(f"==> CLIP Similarity (Source 1 vs Source 2): {similarity_src1_src2:.4f}")
    print("\n--- Evaluation Complete ---")


# ==============================================================================
# --- MAIN EXECUTION LOGIC (from main.py) ---
# ==============================================================================

# --- Captioning Model Globals ---
_caption_processor = None
_caption_model = None

def check_gpu_memory(required_gb=15):
    """Checks available VRAM if CUDA is available."""
    if DEVICE == "cuda":
        try:
            gpu_props = torch.cuda.get_device_properties(0)
            total_mem_gb = gpu_props.total_memory / (1024**3)
            print(f"GPU Name: {gpu_props.name}")
            print(f"Total VRAM: {total_mem_gb:.2f} GB")
            if total_mem_gb < required_gb:
                 print(f"Warning: Detected less than {required_gb}GB VRAM ({total_mem_gb:.2f}GB).")
                 print("         Enable CPU offload (--enable_cpu_offload) if you encounter OOM errors.")
                 return False
            return True
        except Exception as e:
            print(f"Could not get GPU details: {e}")
            return False
    else:
        print("CUDA not available. Running on CPU.")
        return False

def load_captioning_model(model_id=CAPTIONING_MODEL_ID):
    """Loads the image captioning model and processor."""
    global _caption_processor, _caption_model
    if _caption_processor is None:
        print(f"Loading captioning processor: {model_id}")
        _caption_processor = BlipProcessor.from_pretrained(model_id)
    if _caption_model is None:
        print(f"Loading captioning model: {model_id}")
        try:
             # Try loading to default device with float16
             _caption_model = BlipForConditionalGeneration.from_pretrained(
                 model_id, torch_dtype=DTYPE_CAPTION
             ).to(DEVICE)
             _caption_model.eval()
             print(f"  Caption model loaded to {DEVICE} with dtype {DTYPE_CAPTION}")
        except Exception as e:
             print(f"  Warning: Could not load caption model to {DEVICE} with {DTYPE_CAPTION}, trying CPU: {e}")
             try:
                 _caption_model = BlipForConditionalGeneration.from_pretrained(model_id) # Load in full precision on CPU
                 _caption_model.eval()
                 print("  Caption model loaded to CPU.")
             except Exception as e_cpu:
                 print(f"Error: Failed to load caption model even on CPU: {e_cpu}")
                 raise # Re-raise error if captioning cannot be loaded at all

    return _caption_processor, _caption_model

@torch.no_grad()
def generate_caption(image_pil, processor, model):
    """Generates a caption for a single PIL image."""
    if image_pil is None: return " "
    caption = " "
    model_device = next(model.parameters()).device
    try:
        # Force inputs to CPU first for BlipProcessor if model is on CPU
        inputs_device = 'cpu' if model_device == torch.device('cpu') else model_device
        inputs = processor(images=image_pil, return_tensors="pt").to(inputs_device)

        # If model is fp16, convert inputs only if they are on GPU
        if model.dtype == torch.float16 and inputs_device != 'cpu':
             inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)

        # Move inputs to final model device if they were processed on CPU but model is GPU
        if inputs_device != model_device:
             inputs = {k: v.to(model_device) for k, v in inputs.items()}

        generated_ids = model.generate(**inputs, max_new_tokens=50)
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        caption = caption.replace("arafed ", "") # Example cleanup
    except Exception as e:
        print(f"Error during caption generation: {e}")
        caption = " "
    finally:
        del inputs, generated_ids
        if 'cuda' in str(model_device): torch.cuda.empty_cache()
    return caption

def main(args):
    """Main function to run the RAG + SDXL generation pipeline."""
    start_time_main = time.time()
    print("--- Starting Image RAG SDXL Pipeline ---")
    print(f"Using device: {DEVICE}")
    check_gpu_memory()

    # --- 1. Setup Image Knowledge Base (Index/Load) ---
    index_exists = os.path.exists(args.index_path)
    filenames_exist = os.path.exists(args.filenames_path)
    embeddings_exist = os.path.exists(args.embeddings_path) # Check for embeddings file

    rag_index = None
    image_filenames = None
    all_embeddings = None # Needed for MMR

    if args.reindex or not index_exists or not filenames_exist or (args.retrieval_mode == 'mmr' and not embeddings_exist):
        if args.reindex: print("Re-indexing requested.")
        # (Add specific messages why indexing is needed...)

        if not os.path.isdir(args.image_folder):
             print(f"Error: Image folder '{args.image_folder}' not found.")
             return

        # Perform indexing (now also returns embeddings)
        rag_index, image_filenames, all_embeddings = index_image_knowledge_base(
            args.image_folder, args.index_path, args.filenames_path, args.embeddings_path
        )
        if rag_index is None or image_filenames is None:
            print("Indexing failed. Cannot proceed.")
            return
        # We have embeddings now if indexing succeeded
    else:
        print("Found existing index and filename files. Loading...")
        rag_index = load_vector_db_index(args.index_path)
        image_filenames = load_image_filenames(args.filenames_path)
        if args.retrieval_mode == 'mmr':
            all_embeddings = load_all_image_embeddings(args.embeddings_path) # Load pre-saved embeddings

        if rag_index is None or image_filenames is None or (args.retrieval_mode == 'mmr' and all_embeddings is None):
            print("Failed to load existing index, filenames, or embeddings needed. Try re-indexing.")
            return

        if rag_index.ntotal != len(image_filenames) or \
           (all_embeddings is not None and rag_index.ntotal != len(all_embeddings)):
            print(f"Warning: Mismatch between index ({rag_index.ntotal}), filenames ({len(image_filenames)}), and/or embeddings count ({len(all_embeddings) if all_embeddings is not None else 'N/A'}). Re-index recommended.")

    print("--- RAG Setup Complete ---")

    # --- 2. Process Query and Retrieve Relevant Images ---
    print(f"\nProcessing query: '{args.query}'")
    try: load_clip_models()
    except Exception as e: print(f"Fatal: Could not load CLIP models: {e}"); return
    query_embedding = get_clip_text_embedding(args.query)
    if query_embedding is None: print("Failed to generate query embedding."); return

    if args.retrieval_mode == 'mmr':
        if all_embeddings is None: # Should have been loaded or indexing failed
             print("Error: Embeddings required for MMR not available. Exiting.")
             return
        retrieved_image_paths = retrieve_relevant_images_mmr(
            query_embedding, rag_index, image_filenames, all_embeddings,
            top_k=args.top_k, top_n=args.mmr_top_n, lambda_mult=args.mmr_lambda
        )
    else: # Default Top-K
        retrieved_image_paths = retrieve_relevant_images(
            query_embedding, rag_index, image_filenames, top_k=args.top_k
        )

    # --- Handle Retrieval Results ---
    if not retrieved_image_paths or len(retrieved_image_paths) < args.top_k:
         print(f"Warning/Error: Could not retrieve enough images ({len(retrieved_image_paths)}/{args.top_k}).")
         # Fill with first image if at least one retrieved
         if retrieved_image_paths:
              print("Using first retrieved image to fill slots.")
              first_path = retrieved_image_paths[0]
              while len(retrieved_image_paths) < args.top_k: retrieved_image_paths.append(first_path)
         else:
              print("Cannot proceed without retrieved images.")
              return


    # Assign to control types (Assuming top_k=2 for Depth/Canny)
    if args.top_k != 2: print(f"Warning: Code assumes top_k=2 for Depth/Canny assignment.")
    control_image_map = {
        "depth": retrieved_image_paths[0] if len(retrieved_image_paths) > 0 else None,
        "canny": retrieved_image_paths[1] if len(retrieved_image_paths) > 1 else retrieved_image_paths[0] # Fallback to first
    }
    control_types_to_use = ["depth", "canny"]

    print("\nSelected conditioning images for ControlNets:")
    conditioning_pils = []
    control_paths_ordered = []
    for ctype in control_types_to_use:
        path = control_image_map.get(ctype)
        if path is None:
            print(f"Error: No path found for control type {ctype}. Cannot proceed.")
            return
        print(f"  - {ctype.capitalize()} Control Source: {os.path.basename(path)}")
        img = load_image(path)
        if img is None:
            print(f"Error: Failed to load conditioning image for {ctype} from {path}.")
            return
        conditioning_pils.append(img)
        control_paths_ordered.append(path)

    # --- RAG-Informed Prompt Augmentation ---
    augmented_prompt = args.query
    if args.augment_prompt:
        print("\nAugmenting prompt based on retrieved images...")
        try:
            caption_processor, caption_model = load_captioning_model()
            captions = []
            if conditioning_pils:
                 for i, img_pil in enumerate(conditioning_pils[:args.top_k]): # Caption up to top_k images
                     print(f"  Generating caption for source image {i+1}...")
                     caption = generate_caption(img_pil, caption_processor, caption_model)
                     print(f"    Caption {i+1}: {caption}")
                     if caption and caption.strip(): captions.append(caption.strip())
            if captions:
                augmentation_text = ", ".join(captions)
                augmented_prompt = f"{args.query}, visually featuring elements like: {augmentation_text}"
                print(f"  Augmented Prompt: {augmented_prompt[:200]}...") # Truncate long prompts
            else: print("  No captions generated or added.")
        except Exception as e_caption:
            print(f"Warning: Failed to perform prompt augmentation: {e_caption}")
            augmented_prompt = args.query

    # --- 3. Load SDXL Pipeline and Preprocessors ---
    controlnet_items_to_load = []
    controlnet_scales = []
    if "depth" in control_types_to_use:
        controlnet_items_to_load.append((CONTROLNET_MODEL_DEPTH_SDXL, "depth"))
        controlnet_scales.append(args.depth_scale)
    if "canny" in control_types_to_use:
        controlnet_items_to_load.append((CONTROLNET_MODEL_CANNY_SDXL, "canny"))
        controlnet_scales.append(args.canny_scale)

    # Clean up RAG memory before loading heavy models
    del rag_index, image_filenames, query_embedding, all_embeddings, retrieved_image_paths
    gc.collect();
    if DEVICE == 'cuda': torch.cuda.empty_cache()

    pipe = None; refiner = None; preprocessors = {}
    try:
        pipe, refiner = load_sdxl_controlnet_pipeline(
            args.base_model, args.refiner_model, args.vae_model,
            controlnet_items_to_load, args.enable_cpu_offload, args.enable_attention_slicing
        )
        for ctype in control_types_to_use:
            preprocessors[ctype] = get_sdxl_controlnet_preprocessor(ctype)
    except Exception as e:
        print(f"FATAL: Failed to load SDXL pipeline or preprocessors: {e}")
        traceback.print_exc(); return

    # --- 4. Prepare Prompts and Generate Image ---
    style_prompt_prefix, style_negative_suffix = ("", "")
    if args.style in STYLES: style_prompt_prefix, style_negative_suffix = STYLES[args.style]
    else: print(f"Warning: Style '{args.style}' not found. Using no style.")

    final_prompt = style_prompt_prefix + augmented_prompt
    final_negative_prompt = f"{args.negative_prompt}, {DEFAULT_NEGATIVE_PROMPT_PREFIX}, {style_negative_suffix}".strip(", ")
    final_prompt = ', '.join(filter(None, [s.strip() for s in final_prompt.split(',')]))
    final_negative_prompt = ', '.join(filter(None, [s.strip() for s in final_negative_prompt.split(',')]))

    generated_img_pil, control_imgs_pil_used = generate_sdxl_controlled_image(
        pipe=pipe, refiner_pipe=refiner, preprocessors=preprocessors, prompt=final_prompt,
        negative_prompt=final_negative_prompt, conditioning_image_pils=conditioning_pils,
        control_types=control_types_to_use, controlnet_scales=controlnet_scales,
        output_path=args.output_path, num_inference_steps=args.num_steps, guidance_scale=args.guidance_scale,
        seed=args.seed, image_resolution=args.resolution, refiner_steps_ratio=args.refiner_ratio
    )

    # --- 5. Post-Generation ---
    if generated_img_pil:
        print(f"\nSuccessfully generated image: {args.output_path}")
        if control_imgs_pil_used:
            base_output_name = os.path.splitext(args.output_path)[0]
            try:
                for i, ctype in enumerate(control_types_to_use):
                    if i < len(control_imgs_pil_used):
                        control_map_path = f"{base_output_name}_control_{ctype}.png"
                        control_imgs_pil_used[i].save(control_map_path)
                        print(f"  Saved {ctype} control map to: {control_map_path}")
            except Exception as e_save_map: print(f"Warning: Failed to save control maps: {e_save_map}")

        # --- Optional: Run Similarity Evaluation ---
        if args.evaluate_similarity and len(control_paths_ordered) >= 2 :
             run_similarity_evaluation(args.output_path, control_paths_ordered[0], control_paths_ordered[1])

    else:
        print("\nImage generation failed.")

    end_time_main = time.time()
    print(f"\n--- Total Pipeline Execution Time: {end_time_main - start_time_main:.2f} seconds ---")


# ==============================================================================
# --- ARGUMENT PARSING & SCRIPT ENTRY POINT ---
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image RAG + SDXL + Multi-ControlNet (Enhanced Pipeline)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # RAG Arguments
    rag_group = parser.add_argument_group('RAG - Retrieval Arguments')
    rag_group.add_argument("query", type=str, help="Text query.")
    rag_group.add_argument("--image_folder", type=str, default=DEFAULT_IMAGE_FOLDER, help="Image knowledge base folder.")
    rag_group.add_argument("--index_path", type=str, default=VECTOR_DB_INDEX_PATH, help="FAISS index file path.")
    rag_group.add_argument("--filenames_path", type=str, default=IMAGE_FILENAMES_PATH, help="Image filenames pickle file path.")
    rag_group.add_argument("--embeddings_path", type=str, default=IMAGE_EMBEDDINGS_PATH, help="Saved image embeddings numpy file path (for MMR).")
    rag_group.add_argument("--reindex", action="store_true", help="Force re-indexing.")
    rag_group.add_argument("--top_k", type=int, default=DEFAULT_TOP_K_RAG, help="Number of images to retrieve/use.")
    rag_group.add_argument("--retrieval_mode", type=str, default=DEFAULT_RETRIEVAL_MODE, choices=["topk", "mmr"], help="Retrieval strategy.")
    rag_group.add_argument("--mmr_lambda", type=float, default=DEFAULT_MMR_LAMBDA, help="MMR diversity parameter (0-1).")
    rag_group.add_argument("--mmr_top_n", type=int, default=DEFAULT_MMR_TOP_N, help="Initial candidates for MMR.")
    rag_group.add_argument("--augment_prompt", action="store_true", default=DEFAULT_AUGMENT_PROMPT, help="Enable RAG-informed prompt augmentation.")

    # Model Arguments
    model_group = parser.add_argument_group('Model Arguments')
    model_group.add_argument("--base_model", type=str, default=BASE_SDXL_MODEL, help="Base SDXL model ID.")
    model_group.add_argument("--refiner_model", type=str, default=REFINER_SDXL_MODEL, help="Refiner SDXL model ID.")
    model_group.add_argument("--vae_model", type=str, default=VAE_SDXL_MODEL, help="VAE model ID.")
    # ControlNet models defined in config section

    # Generation Arguments
    gen_group = parser.add_argument_group('Generation Arguments')
    gen_group.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH, help="Output image path.")
    gen_group.add_argument("--style", type=str, default=DEFAULT_STYLE, choices=list(STYLES.keys()), help="Style preset.")
    gen_group.add_argument("--negative_prompt", type=str, default="", help="Additional negative prompt text.")
    gen_group.add_argument("--num_steps", type=int, default=DEFAULT_NUM_STEPS, help="Total diffusion steps.")
    gen_group.add_argument("--guidance_scale", "-cfg", type=float, default=DEFAULT_GUIDANCE_SCALE, help="Guidance scale (CFG).")
    gen_group.add_argument("--depth_scale", type=float, default=DEFAULT_DEPTH_SCALE, help="Depth ControlNet scale.")
    gen_group.add_argument("--canny_scale", type=float, default=DEFAULT_CANNY_SCALE, help="Canny ControlNet scale.")
    gen_group.add_argument("--refiner_ratio", type=float, default=DEFAULT_REFINER_RATIO, help="Refiner steps ratio (0 to 1).")
    gen_group.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION, help="Image resolution (square).")
    gen_group.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed (optional).")
    gen_group.add_argument("--evaluate_similarity", action="store_true", help="Run CLIP similarity evaluation after generation.")


    # Performance Arguments
    perf_group = parser.add_argument_group('Performance Arguments')
    perf_group.add_argument("--enable_cpu_offload", action="store_true", default=DEFAULT_ENABLE_CPU_OFFLOAD, help="Enable model CPU offloading (requires accelerate).")
    perf_group.add_argument("--enable_attention_slicing", action="store_true", default=DEFAULT_ENABLE_ATTENTION_SLICING, help="Enable attention slicing.")

    args = parser.parse_args()

    # Validate Arguments
    if not (0 < args.refiner_ratio < 1): print("Error: --refiner_ratio must be between 0 and 1."); exit(1)
    if args.retrieval_mode == 'mmr' and not (0 <= args.mmr_lambda <= 1): print("Error: --mmr_lambda must be between 0 and 1."); exit(1)
    if args.retrieval_mode == 'mmr' and args.mmr_top_n < args.top_k: print("Error: --mmr_top_n must be >= --top_k."); exit(1)

    # Create output directory if needed
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        try: os.makedirs(output_dir); print(f"Created output directory: {output_dir}")
        except OSError as e: print(f"Error creating output directory {output_dir}: {e}"); exit(1)

    # Run Main Pipeline
    main(args)