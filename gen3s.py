import torch
from transformers import AutoProcessor, AutoModel
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DDIMScheduler
)
import faiss
import numpy as np
from PIL import Image
import os
import pickle
from datetime import datetime
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights
import cv2

class EnhancedImageGenerator:
    def __init__(self):
        print("Loading pipeline components...")
        
        # Configuration
        self.SIGLIP_MODEL_NAME = "google/siglip-base-patch16-384"
        self.VECTOR_DB_PATH = "siglip_image_database.faiss"
        self.FILENAMES_PATH = "siglip_image_filenames.pkl"
        self.EMBEDDINGS_PATH = "siglip_image_embeddings.npy"
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.IMAGE_SIZE = 768
        self.STYLE_LAYERS = ['r11', 'r21', 'r31', 'r41', 'r51']
        self.CONTENT_LAYERS = ['r42']
        self.STYLE_WEIGHT = 1e6
        self.CONTENT_WEIGHT = 1
        
        print(f"Using device: {self.DEVICE}")
        
        # Initialize components
        self._init_siglip()
        self._init_stable_diffusion()
        self._init_style_transfer()
        self._load_database()
        
        print("\nAll components initialized successfully!")

    def _init_siglip(self):
        print(f"Loading SigLIP model from {self.SIGLIP_MODEL_NAME}")
        self.processor = AutoProcessor.from_pretrained(self.SIGLIP_MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.SIGLIP_MODEL_NAME).to(self.DEVICE)
        self.model.eval()

    def _init_stable_diffusion(self):
        print("\nLoading Stable Diffusion components...")
        
        # Load main pipeline
        self.sd_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype=torch.float16 if self.DEVICE == "cuda" else torch.float32,
            safety_checker=None
        ).to(self.DEVICE)
        
        # Load ControlNet
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1p_sd15_depth",
            torch_dtype=torch.float16 if self.DEVICE == "cuda" else torch.float32
        ).to(self.DEVICE)
        
        self.controlnet_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16 if self.DEVICE == "cuda" else torch.float32,
            safety_checker=None
        ).to(self.DEVICE)
        
        # Use DDIM scheduler for better quality
        self.sd_pipeline.scheduler = DDIMScheduler.from_config(self.sd_pipeline.scheduler.config)
        self.controlnet_pipeline.scheduler = DDIMScheduler.from_config(self.controlnet_pipeline.scheduler.config)

    def _init_style_transfer(self):
        print("\nInitializing style transfer components...")
        self.vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(self.DEVICE).eval()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])

    def _load_database(self):
        try:
            print(f"Loading image database from {self.VECTOR_DB_PATH}")
            self.index = faiss.read_index(self.VECTOR_DB_PATH)
            
            with open(self.FILENAMES_PATH, 'rb') as f:
                self.image_filenames = pickle.load(f)
            
            self.stored_embeddings = np.load(self.EMBEDDINGS_PATH)
            print(f"Loaded database with {len(self.image_filenames)} images")
        except Exception as e:
            print(f"Error loading database: {e}")
            raise

    def get_text_embedding(self, text_query):
        """Convert text query to SigLIP embedding"""
        try:
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
            
            embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
            return embedding
        except Exception as e:
            print(f"Error getting text embedding: {e}")
            raise

    def find_similar_images(self, text_query, k=5):
        """Find k most similar images to the text query with diversity"""
        print(f"\nFinding diverse similar images for query: '{text_query}'")
        query_embedding = self.get_text_embedding(text_query)
        
        # Convert to float32 and normalize
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Get more candidates for diversity
        num_candidates = min(k * 3, len(self.image_filenames))
        distances, indices = self.index.search(query_embedding, num_candidates)
        
        # Use only the first row of indices
        indices = indices[0]
        distances = distances[0]
        
        # Select diverse subset
        selected_indices = []
        selected_distances = []
        used_indices = set()
        
        for i in range(k):
            best_distance = float('inf')
            best_idx = None
            best_original_idx = None
            
            for j, (idx, dist) in enumerate(zip(indices, distances)):
                if idx in used_indices:
                    continue
                    
                # Calculate diversity penalty
                diversity_penalty = 0
                for selected_idx in selected_indices:
                    similarity = np.dot(self.stored_embeddings[idx], self.stored_embeddings[selected_idx])
                    diversity_penalty += similarity
                
                # Combine distance with diversity penalty
                adjusted_distance = dist + (diversity_penalty * 0.1 if selected_indices else 0)
                
                if adjusted_distance < best_distance:
                    best_distance = adjusted_distance
                    best_idx = idx
                    best_original_idx = j
            
            if best_idx is None:
                break
                
            selected_indices.append(best_idx)
            selected_distances.append(distances[best_original_idx])
            used_indices.add(best_idx)
        
        # Get the corresponding filenames
        similar_images = [self.image_filenames[idx] for idx in selected_indices]
        
        return similar_images, selected_distances

    def _generate_depth_map(self, image):
        """Generate depth map for ControlNet"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate depth using Sobel operators
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        depth = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize depth map
        depth = ((depth - depth.min()) * (255 / (depth.max() - depth.min()))).astype(np.uint8)
        
        return Image.fromarray(depth)

    def _prepare_image_for_style_transfer(self, image):
        """Prepare image for style transfer"""
        if isinstance(image, Image.Image):
            image = transforms.ToTensor()(image)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        return self.normalize(image).to(self.DEVICE)

    def blend_styles(self, generated_image, style_images, style_weights=None):
        """Blend multiple styles into the generated image"""
        try:
            if style_weights is None:
                style_weights = [1.0 / len(style_images)] * len(style_images)
                
            print("\nBlending styles from reference images...")
            
            # Prepare images
            content_img = self._prepare_image_for_style_transfer(generated_image)
            style_imgs = [self._prepare_image_for_style_transfer(img) for img in style_images]
            
            # Initialize target image
            target = content_img.clone().requires_grad_(True)
            optimizer = torch.optim.LBFGS([target])
            
            # Extract content features
            with torch.no_grad():
                content_features = self.vgg(content_img)
            
            # Extract style features
            style_features = []
            for style_img in style_imgs:
                with torch.no_grad():
                    style_features.append(self.vgg(style_img))
            
            # Style transfer optimization
            num_steps = 1
            style_weight = 1e6
            content_weight = 1
            
            for step in tqdm(range(num_steps), desc="Style transfer progress"):
                def closure():
                    optimizer.zero_grad()
                    
                    # Get current features
                    current_features = self.vgg(target)
                    
                    # Content loss
                    content_loss = F.mse_loss(current_features, content_features)
                    
                    # Style loss
                    style_loss = 0
                    for i, sf in enumerate(style_features):
                        # Calculate Gram matrices
                        current_gram = self._gram_matrix(current_features)
                        style_gram = self._gram_matrix(sf)
                        style_loss += style_weights[i] * F.mse_loss(current_gram, style_gram)
                    
                    # Total loss
                    total_loss = content_weight * content_loss + style_weight * style_loss
                    
                    if total_loss.requires_grad:
                        total_loss.backward()
                    
                    return total_loss
                
                optimizer.step(closure)
            
            # Convert back to image
            with torch.no_grad():
                output = target.squeeze(0).cpu()
                output = torch.clamp(output, 0, 1)
                
            return transforms.ToPILImage()(output)
            
        except Exception as e:
            print(f"Error in style blending: {e}")
            return generated_image  # Return original image if style transfer fails

    def _gram_matrix(self, x):
        """Calculate Gram matrix"""
        b, c, h, w = x.size()
        features = x.view(b * c, h * w)
        gram = torch.mm(features, features.t())
        return gram.div(b * c * h * w)

    def generate_enhanced_image(self, text_prompt, num_references=3, strength=0.75):
        """Generate enhanced image using RAG and style transfer"""
        try:
            # Find similar images
            similar_images, distances = self.find_similar_images(text_prompt, k=num_references)
            
            if not similar_images:
                raise ValueError("No similar images found!")
            
            print("\nGenerating base image...")
            # Generate initial image using the most similar reference
            reference_image = Image.open(similar_images[0]).convert('RGB')
            reference_image = reference_image.resize((self.IMAGE_SIZE, self.IMAGE_SIZE))
            
            # Generate depth map for ControlNet
            depth_image = self._generate_depth_map(reference_image)
            
            # Generate base image
            base_image = self.sd_pipeline(
                prompt=text_prompt,
                image=reference_image,
                strength=strength,
                guidance_scale=7.5,
                num_inference_steps=50
            ).images[0]
            
            # Return both base image and reference images before style transfer
            return base_image, similar_images, distances
            
        except Exception as e:
            print(f"Error generating enhanced image: {e}")
            raise

def main():
    # Create output directory
    os.makedirs("generated_images", exist_ok=True)
    
    try:
        print("\nInitializing enhanced image generator...")
        generator = EnhancedImageGenerator()
        
        while True:
            # Get user input
            query = input("\nEnter your text description (or 'quit' to exit): ")
            
            if query.lower() == 'quit':
                break
                
            num_references = int(input("Enter number of reference images to use (1-5): "))
            num_references = max(1, min(5, num_references))
            
            # Generate enhanced image
            print("\nGenerating enhanced image...")
            base_image, reference_images, distances = generator.generate_enhanced_image(
                query,
                num_references=num_references
            )
            
            # Save images
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_base = os.path.join("generated_images", f"enhanced_{timestamp}")
            
            # Save base image
            base_output = f"{output_base}_base.png"
            base_image.save(base_output)
            print(f"\nBase image saved as: {base_output}")
            
            # Save reference images
            print("\nSaving reference images...")
            for i, ref_image in enumerate(reference_images):
                ref_output = f"{output_base}_reference_{i+1}.png"
                Image.open(ref_image).save(ref_output)
                print(f"Reference image {i+1} saved as: {ref_output}")
            
            # Attempt style transfer
            try:
                print("\nAttempting style transfer...")
                style_images = [Image.open(img).convert('RGB').resize((generator.IMAGE_SIZE, generator.IMAGE_SIZE))
                              for img in reference_images]
                
                # Calculate style weights based on distances
                max_dist = max(distances)
                style_weights = [1 - (d / max_dist) for d in distances]
                style_weights = [w / sum(style_weights) for w in style_weights]
                
                # Blend styles
                final_image = generator.blend_styles(base_image, style_images, style_weights)
                
                # Save final image
                final_output = f"{output_base}_final.png"
                final_image.save(final_output)
                print(f"Final styled image saved as: {final_output}")
            except Exception as e:
                print(f"\nStyle transfer failed: {e}")
                print("Using base image as final output.")
            
            print("\nGeneration complete!")
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Ensure the image database files exist")
        print("2. Check GPU memory (reduce batch size or ▋")
if __name__ == "__main__":
    main()