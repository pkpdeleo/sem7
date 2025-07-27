import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import faiss
import numpy as np
import os

class ImageEmbeddingExtractor:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        Loads a pre-trained CLIP model and associated processor.
        """
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def get_image_embedding(self, image_path: str) -> np.ndarray:
        """
        Computes the embedding for a single image using CLIP.
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            image_embeds = self.model.get_image_features(**inputs)
        image_embeds = image_embeds.cpu().numpy()
        return image_embeds.squeeze()

    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Computes the embedding for a text query using CLIP.
        """
        inputs = self.processor(text=[text], return_tensors="pt")
        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            text_embeds = self.model.get_text_features(**inputs)
        return text_embeds.cpu().numpy().squeeze()


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Builds a simple FAISS index (inner product) from a numpy array of embeddings.
    """
    index = faiss.IndexFlatIP(embeddings.shape[1])  # using IP for cosine similarity with normalized vectors
    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / (norms + 1e-7)
    index.add(normalized_embeddings)
    return index


def main():
    """
    Example usage:
    1) Compute embeddings for a directory of images
    2) Store them in a FAISS index
    3) Demonstrate similarity search for a sample query
    """
    extractor = ImageEmbeddingExtractor()

    # Updated image directory path to user's path
    image_dir = r"C:\Projects\Gpt\example_images"
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith((".jpg",".png",".jpeg"))]

    # Compute embeddings for all images
    all_embeddings = []
    for img_path in image_paths:
        emb = extractor.get_image_embedding(img_path)
        all_embeddings.append(emb)
    all_embeddings = np.array(all_embeddings)

    # Build FAISS index
    index = build_faiss_index(all_embeddings)

    # Sample query
    query = "a photo of a golden retriever running in a park"
    query_embedding = extractor.get_text_embedding(query)
    query_norm = np.linalg.norm(query_embedding, keepdims=True)
    query_embedding = query_embedding / (query_norm + 1e-7)

    # Perform search
    top_k = 3
    distances, indices = index.search(np.array([query_embedding]), top_k)
    for rank, idx in enumerate(indices[0]):
        print(f"Rank {rank+1}, File: {image_paths[idx]}, Distance: {distances[0][rank]}")

if __name__ == "__main__":
    main()