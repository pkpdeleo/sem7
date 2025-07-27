import torch
from transformers import AutoProcessor, AutoModel
# Import from diffusers instead of transformers
from diffusers import StableDiffusionImg2ImgPipeline
import faiss
import numpy as np
from PIL import Image
import os
import pickle
from datetime import datetime
from tqdm import tqdm

class ImageGenerator:
    def __init__(self):
        print("Loading pipeline components...")
        
        # Configuration
        self.SIGLIP_MODEL_NAME = "google/siglip-base-patch16-384"
        self.VECTOR_DB_PATH = "siglip_image_database.faiss"
        self.FILENAMES_PATH = "siglip_image_filenames.pkl"
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {self.DEVICE}")
        print(f"Loading SigLIP model from {self.SIGLIP_MODEL_NAME}")
        
        # Initialize SigLIP components
        self.processor = AutoProcessor.from_pretrained(self.SIGLIP_MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.SIGLIP_MODEL_NAME).to(self.DEVICE)
        self.model.eval()
        
        print("SigLIP model loaded successfully")
        
        # Load the image database
        try:
            print(f"Loading image database from {self.VECTOR_DB_PATH}")
            self.index = faiss.read_index(self.VECTOR_DB_PATH)
            with open(self.FILENAMES_PATH, 'rb') as f:
                self.image_filenames = pickle.load(f)
            print(f"Loaded database with {len(self.image_filenames)} images")
        except Exception as e:
            print(f"Error loading database: {e}")
            raise
        
        # Initialize Stable Diffusion
        try:
            print("\nLoading Stable Diffusion model...")
            self.sd_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                torch_dtype=torch.float16 if self.DEVICE == "cuda" else torch.float32,
                safety_checker=None
            ).to(self.DEVICE)
            print("Stable Diffusion model loaded successfully")
        except Exception as e:
            print(f"Error loading Stable Diffusion: {e}")
            raise
        
        print("\nAll components initialized successfully!")

    def get_text_embedding(self, text_query):
        """Convert text query to SigLIP embedding"""
        try:
            # Process text with dummy image input (required by SigLIP)
            dummy_image = torch.zeros((1, 3, 384, 384)).to(self.DEVICE)
            inputs = self.processor(
                text=[text_query],
                images=dummy_image,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.DEVICE) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.text_embeds.cpu().numpy()
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
            return embedding
        except Exception as e:
            print(f"Error getting text embedding: {e}")
            raise

    def find_similar_images(self, text_query, k=5):
        """Find k most similar images to the text query"""
        print(f"\nFinding similar images for query: '{text_query}'")
        query_embedding = self.get_text_embedding(text_query)
        
        # Search the index
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Get the matching filenames
        similar_images = [self.image_filenames[idx] for idx in indices[0]]
        
        return similar_images, distances[0]

    def generate_image(self, text_prompt, reference_image_path, strength=0.75):
        """Generate new image based on text prompt and reference image"""
        try:
            print(f"Loading reference image: {os.path.basename(reference_image_path)}")
            reference_image = Image.open(reference_image_path).convert('RGB')
            reference_image = reference_image.resize((768, 768))
            
            print(f"Generating new image with prompt: '{text_prompt}'")
            result = self.sd_pipeline(
                prompt=text_prompt,
                image=reference_image,
                strength=strength,
                guidance_scale=7.5,
                num_inference_steps=50
            )
            
            return result.images[0]
        except Exception as e:
            print(f"Error generating image: {e}")
            raise

def main():
    # Create output directory
    os.makedirs("generated_images", exist_ok=True)
    
    try:
        # Initialize generator
        print("\nInitializing image generator...")
        generator = ImageGenerator()
        
        # Get user input
        query = input("\nEnter your text description: ")
        
        # Find similar images
        similar_images, distances = generator.find_similar_images(query, k=1)
        
        if not similar_images:
            print("No similar images found!")
            return
        
        # Get the most similar image
        reference_image = similar_images[0]
        print(f"\nFound similar image: {os.path.basename(reference_image)}")
        print(f"Similarity distance: {distances[0]:.4f}")
        
        # Generate new image
        print("\nGenerating new image...")
        generated_image = generator.generate_image(query, reference_image)
        
        # Save images
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = os.path.join("generated_images", f"image_{timestamp}")
        reference_output = f"{output_base}_reference.png"
        generated_output = f"{output_base}_generated.png"
        
        print("\nSaving images...")
        Image.open(reference_image).save(reference_output)
        generated_image.save(generated_output)
        
        print("\nGeneration complete!")
        print(f"Reference image saved as: {reference_output}")
        print(f"Generated image saved as: {generated_output}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure the image database files exist (run ep1_siglip.py first)")
        print("2. Check you have enough GPU memory (try reducing batch size if needed)")
        print("3. Verify all required packages are installed properly")
        print("4. Make sure your CUDA drivers are up to date")

if __name__ == "__main__":
    main()