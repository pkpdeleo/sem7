# rag_utils.py
import torch
from transformers import CLIPModel, CLIPProcessor
import faiss
import numpy as np
from PIL import Image
import os
import requests
from io import BytesIO
import pickle
import time

# Import config settings
from config import CLIP_MODEL_NAME, DEVICE

# Global cache for CLIP models to avoid reloading
_clip_model_cache = None
_clip_processor_cache = None

def load_clip_models():
    """Loads CLIP model and processor, using a cache."""
    global _clip_model_cache, _clip_processor_cache
    if _clip_model_cache is None:
        print(f"Loading CLIP model: {CLIP_MODEL_NAME}")
        try:
            _clip_model_cache = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
            _clip_model_cache.eval() # Set to evaluation mode
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

@torch.no_grad() # Disable gradient calculation for inference
def get_clip_image_embedding(image_pil):
    """Generates normalized CLIP embedding for a PIL image."""
    if image_pil is None: return None
    model, processor = load_clip_models()
    try:
        inputs = processor(images=image_pil, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model.get_image_features(**inputs)
        embedding = outputs.cpu().numpy()
        # Normalize embedding (important for cosine similarity/L2 distance)
        norm = np.linalg.norm(embedding, axis=-1, keepdims=True)
        normalized_embedding = embedding / norm
        return normalized_embedding
    except Exception as e:
        print(f"Error generating CLIP image embedding: {e}")
        return None

@torch.no_grad() # Disable gradient calculation for inference
def get_clip_text_embedding(query_text):
    """Generates normalized CLIP embedding for a text query."""
    model, processor = load_clip_models()
    try:
        inputs = processor(text=[query_text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model.get_text_features(**inputs)
        embedding = outputs.cpu().numpy()
        # Normalize embedding
        norm = np.linalg.norm(embedding, axis=-1, keepdims=True)
        normalized_embedding = embedding / norm
        return normalized_embedding
    except Exception as e:
        print(f"Error generating CLIP text embedding for '{query_text}': {e}")
        return None

# --- FAISS Indexing Functions ---

def initialize_vector_db(embedding_dimension):
    """Initializes a FAISS index."""
    print(f"Initializing FAISS IndexFlatL2 with dimension {embedding_dimension}")
    return faiss.IndexFlatL2(embedding_dimension)

def add_embeddings_to_index(index, embeddings):
    """Adds embeddings to the FAISS index."""
    if embeddings.ndim == 1:
        embeddings = np.expand_dims(embeddings, axis=0)
    print(f"Adding {embeddings.shape[0]} embeddings to FAISS index...")
    index.add(embeddings.astype('float32')) # FAISS expects float32
    print(f"Index size after adding: {index.ntotal}")

def save_vector_db_index(index, index_path):
    """Saves the FAISS index to a file."""
    print(f"Saving FAISS index to: {index_path}")
    try:
        faiss.write_index(index, index_path)
        print("FAISS index saved successfully.")
    except Exception as e:
        print(f"Error saving FAISS index: {e}")

def load_vector_db_index(index_path):
    """Loads a FAISS index from a file."""
    if not os.path.exists(index_path):
        print(f"FAISS index file not found at: {index_path}")
        return None
    print(f"Loading FAISS index from: {index_path}")
    try:
        index = faiss.read_index(index_path)
        print(f"FAISS index loaded successfully with {index.ntotal} vectors.")
        return index
    except Exception as e:
        print(f"Error loading FAISS index: {e}. Please check file integrity or consider re-indexing.")
        return None

def save_image_filenames(filenames, filenames_path):
    """Saves the list of image filenames using pickle."""
    print(f"Saving image filenames list to: {filenames_path}")
    try:
        with open(filenames_path, 'wb') as f:
            pickle.dump(filenames, f)
        print("Image filenames saved successfully.")
    except Exception as e:
        print(f"Error saving image filenames: {e}")

def load_image_filenames(filenames_path):
    """Loads the list of image filenames from a pickle file."""
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
        print(f"Error loading image filenames pickle: {e}. Please check file integrity or consider re-indexing.")
        return None

# --- Indexing Process ---

def index_image_knowledge_base(image_folder, index_path, filenames_path):
    """
    Indexes images in a folder: generates CLIP embeddings, creates FAISS index,
    and saves the index and corresponding filenames. Returns the index and filenames.
    """
    print(f"\n--- Starting Image Indexing Process ---")
    print(f"Image Folder: {image_folder}")

    if not os.path.isdir(image_folder):
        print(f"Error: Image folder '{image_folder}' not found or is not a directory.")
        print("Please create the folder and add images, or specify the correct path.")
        return None, None

    # Find image files recursively
    image_paths = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
    print("Scanning for images...")
    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith(valid_extensions):
                full_path = os.path.join(root, file)
                image_paths.append(os.path.abspath(full_path)) # Store absolute paths

    if not image_paths:
        print(f"No images found in folder: {image_folder}")
        return None, None
    print(f"Found {len(image_paths)} potential image files.")

    embeddings_list = []
    image_filenames_stored = [] # Filenames corresponding to successful embeddings
    processed_count = 0
    skipped_count = 0
    total_images = len(image_paths)
    start_time = time.time()

    # Load CLIP models once before the loop
    load_clip_models()

    print("Generating embeddings...")
    for i, img_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{total_images}: {os.path.basename(img_path)} ...", end='\r')
        image_pil = load_image(img_path)
        if image_pil:
            embedding = get_clip_image_embedding(image_pil)
            if embedding is not None:
                embeddings_list.append(embedding)
                image_filenames_stored.append(img_path) # Store the path if embedding succeeded
                processed_count += 1
            else:
                print(f"\nSkipping {os.path.basename(img_path)} - embedding generation failed.")
                skipped_count += 1
        else:
            print(f"\nSkipping {os.path.basename(img_path)} - image loading failed.")
            skipped_count += 1

        # Optional: Clear CUDA cache periodically for very large datasets
        if DEVICE == 'cuda' and (i + 1) % 100 == 0:
            torch.cuda.empty_cache()

    end_time = time.time()
    print(f"\n--- Indexing Summary ---")
    print(f"Successfully processed: {processed_count}/{total_images} images.")
    print(f"Skipped: {skipped_count} images.")
    print(f"Total time: {end_time - start_time:.2f} seconds.")

    if not embeddings_list:
        print("No valid image embeddings were generated. Index cannot be created.")
        return None, None

    # Concatenate embeddings into a single NumPy matrix
    embeddings_matrix = np.concatenate(embeddings_list, axis=0)
    embedding_dimension = embeddings_matrix.shape[1]

    # Create and populate the FAISS index
    index = initialize_vector_db(embedding_dimension)
    add_embeddings_to_index(index, embeddings_matrix)

    # Save the index and the list of filenames
    save_vector_db_index(index, index_path)
    save_image_filenames(image_filenames_stored, filenames_path)

    print("--- Indexing Complete ---")
    return index, image_filenames_stored


# --- Retrieval ---

def retrieve_relevant_images(query_embedding, index, image_filenames, top_k=1):
    """Retrieves top_k most relevant image filenames based on query embedding."""
    print(f"\n--- Retrieving Top {top_k} Relevant Images ---")
    if index is None:
        print("Error: FAISS index is not loaded.")
        return []
    if image_filenames is None:
        print("Error: Image filenames are not loaded.")
        return []
    if index.ntotal == 0:
        print("FAISS index is empty.")
        return []
    if index.ntotal != len(image_filenames):
        print(f"Warning: FAISS index size ({index.ntotal}) differs from filenames count ({len(image_filenames)}).")
        print("Results might be inconsistent. Re-indexing is recommended.")
        # Proceed cautiously, but warn the user

    try:
        k_search = min(top_k, index.ntotal) # Ensure we don't ask for more results than exist
        print(f"Searching index for {k_search} nearest neighbors...")
        distances, indices = index.search(query_embedding.astype('float32'), k_search)
        print(f"FAISS search complete. Indices found: {indices[0]}")

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
            # This should ideally not happen if the index and filenames are synced
            print(f"Warning: Retrieved index {idx} is out of bounds (max valid: {max_valid_index}). This indicates a potential index/filename mismatch. Skipping.")

    print("Retrieved candidates:")
    for fname, dist in zip(retrieved_filenames, retrieved_distances):
        print(f"  - {os.path.basename(fname)} (Distance: {dist:.4f})") # L2 distance, lower is better
    print("--- Retrieval Complete ---")
    return retrieved_filenames