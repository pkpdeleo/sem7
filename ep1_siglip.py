import torch
from transformers import AutoProcessor, AutoModel
import faiss
import numpy as np
from PIL import Image
import os
import pickle
from tqdm import tqdm
from datetime import datetime

# Configuration
SIGLIP_MODEL_NAME = "google/siglip-base-patch16-384"
VECTOR_DB_PATH = "siglip_image_database.faiss"
FILENAMES_PATH = "siglip_image_filenames.pkl"
IMAGE_FOLDER = r"C:\Projects\Gpt\example_images"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32  # Adjust based on your GPU memory

# Print session information
print(f"\nStarting at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
print(f"User: {os.getlogin()}")
print(f"Device: {DEVICE}")
print(f"Image Folder: {IMAGE_FOLDER}")

def initialize_siglip():
    """Initialize SigLIP model and processor"""
    try:
        print("Initializing SigLIP model and processor...")
        processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)
        model = AutoModel.from_pretrained(SIGLIP_MODEL_NAME).to(DEVICE)
        model.eval()  # Set to evaluation mode
        print("SigLIP initialization successful!")
        return model, processor
    except Exception as e:
        print(f"Error initializing SigLIP: {str(e)}")
        raise

def process_image_batch(image_paths, model, processor):
    """Process a batch of images and return their embeddings"""
    images = []
    valid_paths = []
    
    # Load and preprocess images
    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert('RGB')
            images.append(image)
            valid_paths.append(img_path)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue
    
    if not images:
        return [], []
    
    try:
        # Process images with dummy text input
        inputs = processor(
            images=images,
            text=[""] * len(images),  # Empty text for each image
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Get image embeddings only
            embeddings = outputs.image_embeds.cpu().numpy()
            
        # Normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings, valid_paths
    except Exception as e:
        print(f"Error processing batch: {e}")
        return [], []

def build_image_database():
    """Build and save the image database with SigLIP embeddings"""
    print(f"\nStarting database build at {os.path.abspath(IMAGE_FOLDER)}")
    
    # Initialize SigLIP
    model, processor = initialize_siglip()
    
    # Get all image files
    image_files = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    for file in os.listdir(IMAGE_FOLDER):
        if os.path.splitext(file)[1].lower() in valid_extensions:
            image_files.append(os.path.join(IMAGE_FOLDER, file))
    
    if not image_files:
        print(f"No images found in {IMAGE_FOLDER}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Process images in batches
    all_embeddings = []
    all_valid_paths = []
    
    # Use tqdm for progress bar
    for i in tqdm(range(0, len(image_files), BATCH_SIZE), desc="Processing batches"):
        batch_files = image_files[i:i + BATCH_SIZE]
        embeddings, valid_paths = process_image_batch(batch_files, model, processor)
        
        if len(embeddings) > 0:
            all_embeddings.extend(embeddings)
            all_valid_paths.extend(valid_paths)
        
        # Optional: Print progress
        if (i + BATCH_SIZE) % (BATCH_SIZE * 10) == 0:
            print(f"\nProcessed {i + BATCH_SIZE}/{len(image_files)} images")
    
    if not all_embeddings:
        print("No valid embeddings generated!")
        return
    
    # Convert to numpy array
    embeddings_matrix = np.stack(all_embeddings)
    
    # Create and save FAISS index
    print("\nBuilding FAISS index...")
    dimension = embeddings_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_matrix.astype('float32'))
    
    # Save index and filenames
    print("Saving index and filenames...")
    faiss.write_index(index, VECTOR_DB_PATH)
    with open(FILENAMES_PATH, 'wb') as f:
        pickle.dump(all_valid_paths, f)
    
    print(f"\nDatabase built successfully!")
    print(f"Processed {len(all_valid_paths)} images")
    print(f"Index saved to: {VECTOR_DB_PATH}")
    print(f"Filenames saved to: {FILENAMES_PATH}")

def get_text_embedding(text, model, processor):
    """Generate embedding for a text query"""
    inputs = processor(
        text=[text],
        images=None,  # No images for text-only embedding
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.text_embeds.cpu().numpy()
    
    # Normalize embedding
    embedding = embedding / np.linalg.norm(embedding)
    return embedding

if __name__ == "__main__":
    try:
        build_image_database()
    except Exception as e:
        print(f"\nError during execution: {str(e)}")