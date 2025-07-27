import torch
from transformers import CLIPModel, CLIPProcessor, pipeline
import faiss
import numpy as np
from PIL import Image
import os
import requests
from io import BytesIO
import pickle  # To save and load image filenames

# --- Configuration ---
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
VECTOR_DB_INDEX_PATH = "image_rag_index.faiss"
IMAGE_FILENAMES_PATH = "image_filenames.pkl"  # Path to save/load image filenames
TEXT_GENERATION_MODEL_NAME = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Setup CLIP Model and Processor ---
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

# --- Setup Text Generation Pipeline ---
text_generator = pipeline(
    'text-generation', 
    model=TEXT_GENERATION_MODEL_NAME, 
    device=0 if DEVICE=="cuda" else -1
)

# --- Vector Database Initialization (FAISS) ---
def initialize_vector_db(embedding_dimension):
    """Initializes a FAISS index."""
    index = faiss.IndexFlatL2(embedding_dimension)
    return index

def add_embeddings_to_index(index, embeddings):
    """Adds embeddings to the FAISS index."""
    index.add(embeddings.astype('float32'))

def save_vector_db_index(index, index_path):
    """Saves the FAISS index to disk."""
    faiss.write_index(index, index_path)
    print(f"Vector index SAVED to: {index_path}")

def load_vector_db_index(index_path):
    """Loads a FAISS index from disk."""
    print(f"Loading vector index from path: {index_path}")  # Debugging print
    if os.path.exists(index_path):
        print("Vector index file EXISTS.")  # Debugging print
        return faiss.read_index(index_path)
    else:
        print("Vector index file DOES NOT EXIST.")  # Debugging print
        return None

def save_image_filenames(filenames, filenames_path):
    """Saves image filenames to disk using pickle."""
    with open(filenames_path, 'wb') as f:
        pickle.dump(filenames, f)
    print(f"Image filenames SAVED to: {filenames_path}")  # Added print

def load_image_filenames(filenames_path):
    """Loads image filenames from disk using pickle."""
    print(f"Loading image filenames from path: {filenames_path}")  # Debugging print
    if os.path.exists(filenames_path):
        print("Image filenames file EXISTS.")  # Debugging print
        with open(filenames_path, 'rb') as f:
            return pickle.load(f)
    else:
        print("Image filenames file DOES NOT EXIST.")  # Debugging print
        return None

# --- Image Embedding Function ---
def get_clip_image_embedding(image_path, model, processor):
    """Generates CLIP embedding for a single image from a file path or URL."""
    try:
        if image_path.startswith("http://") or image_path.startswith("https://"):
            response = requests.get(image_path)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image from {image_path}: {e}")
        return None

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    embedding = outputs.cpu().numpy()
    embedding = embedding / np.linalg.norm(embedding, axis=-1, keepdims=True)
    return embedding

# --- Text Query Embedding Function ---
def get_clip_text_embedding(query_text, model, processor):
    """Generates CLIP embedding for a text query."""
    # Explicitly set truncation=True to avoid warnings from the tokenizer
    inputs = processor(text=[query_text], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
    embedding = outputs.cpu().numpy()
    embedding = embedding / np.linalg.norm(embedding, axis=-1, keepdims=True)
    return embedding

# --- Indexing Knowledge Base ---
def index_image_knowledge_base(image_folder, index_path=VECTOR_DB_INDEX_PATH, filenames_path=IMAGE_FILENAMES_PATH):
    """Indexes images in a folder and saves embeddings and filenames."""
    image_paths = []
    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_paths.append(os.path.join(root, file))

    if not image_paths:
        print(f"No images found in folder: {image_folder}")
        return None  # Indicate indexing failure - return None

    embeddings_list = []
    image_filenames = []

    for img_path in image_paths:
        embedding = get_clip_image_embedding(img_path, clip_model, clip_processor)
        if embedding is not None:
            embeddings_list.append(embedding)
            image_filenames.append(img_path)
        else:
            print(f"Skipping indexing {img_path} due to embedding error.")

    if not embeddings_list:
        print("No valid image embeddings were generated. Index not created.")
        return None  # Indicate indexing failure - return None

    embeddings_matrix = np.concatenate(embeddings_list, axis=0)

    index = initialize_vector_db(embeddings_matrix.shape[1])
    add_embeddings_to_index(index, embeddings_matrix)
    save_vector_db_index(index, index_path)
    save_image_filenames(image_filenames, filenames_path)  # SAVE filenames here

    print(f"Indexed {len(image_filenames)} images. Index saved to: {index_path}")
    print(f"Image filenames saved to: {filenames_path}")
    print(f"Number of image filenames returned from indexing: {len(image_filenames)}")
    return image_filenames  # Return filenames

# --- Retrieval Function ---
def retrieve_relevant_images(query_embedding, index, image_filenames, top_k=5):
    """Retrieves top_k most relevant image filenames based on query embedding."""
    print("--- retrieve_relevant_images START ---")
    if index is None:
        print("Vector database index not loaded. Please index images first.")
        print("--- retrieve_relevant_images END (index None) ---")
        return []

    if image_filenames is None:
        print("Image filenames not loaded.")
        print("--- retrieve_relevant_images END (filenames None) ---")
        return []

    print(f"Index is loaded: {index is not None}")
    print(f"Image filenames loaded: {image_filenames is not None}, Length: {len(image_filenames) if image_filenames else 0}")

    try:
        print("Starting FAISS search...")
        D, I = index.search(query_embedding.astype('float32'), top_k)
        print("FAISS search COMPLETED.")
        print(f"Distances (D): {D}")
        print(f"Indices (I): {I}")
    except Exception as e:
        print(f"Error during FAISS index search: {e}")
        print("--- retrieve_relevant_images END (FAISS error) ---")
        return []

    retrieved_filenames = []
    print("Processing retrieved indices...")
    for idx in I[0]:
        print(f"Processing index: {idx}")
        if idx < len(image_filenames):
            retrieved_filenames.append(image_filenames[idx])
            print(f"  Filename added: {image_filenames[idx]}")
        else:
            print(f"Warning: Index out of bounds during retrieval (index:{idx}, filenames len:{len(image_filenames)}).")
            break

    print(f"Retrieved filenames: {retrieved_filenames}")
    print("--- retrieve_relevant_images END (success) ---")
    return retrieved_filenames

# --- Augmented Context Function ---
def create_augmented_context(query_text, retrieved_image_filenames):
    """Creates a text context by combining the query and retrieved image info."""
    context_text = f"User query: {query_text}\nRetrieved images:\n"
    for filename in retrieved_image_filenames:
        context_text += f"- {filename}\n"
    context_text += "\nUsing the above information, generate a detailed and informative response to the user query."
    return context_text

# --- Response Generation Function ---
def generate_text_response(context_text, generator=text_generator):
    """Generates a text response using a language model based on the context."""
    prompt = context_text
    try:
        response = generator(prompt, max_length=200, num_return_sequences=1, stop_sequence="\nResponse:", truncation=True)
        # Debug print the full response
        print("Full text generation response:", response)
        generated_text = response[0]['generated_text']
        if prompt in generated_text:
            generated_text = generated_text.replace(prompt, "").strip()
        return generated_text
    except Exception as e:
        print(f"Error in text generation: {e}")
        return "Sorry, I encountered an error during response generation."

# --- Main Image RAG Pipeline Function ---
def image_rag_pipeline(query, image_folder, vector_index_path=VECTOR_DB_INDEX_PATH, image_filenames_path=IMAGE_FILENAMES_PATH):
    """
    Runs the Image RAG pipeline.
    """
    image_filenames = None
    index = None

    # Load vector index and filenames
    index = load_vector_db_index(vector_index_path)
    image_filenames = load_image_filenames(image_filenames_path)

    if index is None or image_filenames is None:
        print("Vector index or image filenames not found. Starting indexing...")
        image_filenames = index_image_knowledge_base(image_folder, vector_index_path, IMAGE_FILENAMES_PATH)
        if image_filenames is None:
            print("Indexing failed. Cannot proceed.")
            return None, None
        index = load_vector_db_index(vector_index_path)
        if index is None:
            print("Error loading vector index even after indexing. Pipeline cannot continue.")
            return None, None
        image_filenames = load_image_filenames(image_filenames_path)
        if image_filenames is None:
            print("Error loading image filenames even after indexing. Pipeline cannot continue.")
            return None, None
    else:
        print("Vector index and image filenames loaded from disk.")
        image_filenames = load_image_filenames(image_filenames_path)
        if image_filenames is None:
            print("Error loading image filenames from disk. Pipeline cannot continue.")
            return None, None
        print(f"Loaded image filenames: {image_filenames}")

    # 1. Get Query Embedding
    query_embedding = get_clip_text_embedding(query, clip_model, clip_processor)

    # 2. Retrieve Relevant Images
    retrieved_image_filenames = retrieve_relevant_images(query_embedding, index, image_filenames)

    # 3. Create Augmented Context
    context_text = create_augmented_context(query, retrieved_image_filenames)

    # 4. Generate Response
    generated_response_text = generate_text_response(context_text)

    return generated_response_text, retrieved_image_filenames

# --- Example Usage ---
if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")
    image_folder_path = "example_images"
    if not os.path.exists(image_folder_path):
        os.makedirs(image_folder_path)
        print(f"Created example image folder: {image_folder_path}. Please add some images inside.")
        example_image_urls = [
            r"C:\Projects\Gpt\example_images\ILSVRC2012_val_00000003.JPEG"
        ]
        for i, url in enumerate(example_image_urls):
            try:
                response = requests.get(url)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content))
                img.save(os.path.join(image_folder_path, f"example_image_{i+1}.jpg"))
            except Exception as e:
                print(f"Error downloading example image from {url}: {e}")

    query_text = "What is there in the picture"
    response, retrieved_images = image_rag_pipeline(query_text, image_folder_path)

    if response and retrieved_images:
        print(f"\nQuery: {query_text}")
        print(f"Retrieved Images: {retrieved_images}")
        print(f"Generated Response:\n{response}")
    else:
        print("Image RAG pipeline failed to complete indexing or retrieval.")