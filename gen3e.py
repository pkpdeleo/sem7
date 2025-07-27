import argparse
import torch
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from ep1 import (
    clip_model,
    clip_processor,
    VECTOR_DB_INDEX_PATH,
    IMAGE_FILENAMES_PATH,
    get_clip_text_embedding,
    retrieve_relevant_images,
    load_vector_db_index,
    load_image_filenames
)

def load_image(image_path):
    """Load an image and ensure it's in RGB format."""
    try:
        image = Image.open(image_path)
        # Convert to RGB and return as PIL Image
        return image.convert("RGB")
    except Exception as e:
        print(f"Error loading image from {image_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Find a similar image based on text input and generate a new image based on it."
    )
    parser.add_argument("text_prompt", type=str, help="Text description to find and generate similar images")
    parser.add_argument("--output", type=str, default="generated_image.png", 
                        help="Output filename for the generated image")
    parser.add_argument("--strength", type=float, default=0.3, 
                        help="Strength for image transformation (lower values keep more of original)")
    parser.add_argument("--guidance_scale", type=float, default=7.5, 
                        help="Classifier-Free Guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=50, 
                        help="Number of denoising steps")
    args = parser.parse_args()

    # Load the vector index and image filenames
    index = load_vector_db_index(VECTOR_DB_INDEX_PATH)
    image_filenames = load_image_filenames(IMAGE_FILENAMES_PATH)
    
    if index is None or image_filenames is None:
        print("Vector index or image filenames not found. Please run the indexing pipeline first.")
        return

    # Generate text embedding
    text_embedding = get_clip_text_embedding(args.text_prompt, clip_model, clip_processor)
    
    # Find similar image
    retrieved_images = retrieve_relevant_images(text_embedding, index, image_filenames, top_k=1)
    
    if not retrieved_images:
        print("No similar images found.")
        return

    # Load the most similar image
    reference_image_path = retrieved_images[0]
    print(f"Found similar image: {reference_image_path}")
    
    # Load and prepare the reference image
    reference_image = load_image(reference_image_path)
    if reference_image is None:
        print("Failed to load the reference image.")
        return

    # Ensure image is in correct format and size
    reference_image = reference_image.resize((512, 512))
    
    # Debug print
    print(f"Reference image type: {type(reference_image)}")
    print(f"Reference image mode: {reference_image.mode}")
    print(f"Reference image size: {reference_image.size}")

    # Initialize Stable Diffusion pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        variant="fp16" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    # Convert PIL Image to numpy array
    init_image_np = np.array(reference_image)
    
    # Generate new image
    print(f"Generating new image based on text prompt: {args.text_prompt}")
    try:
        with torch.autocast("cuda") if device == "cuda" else torch.no_grad():
            output = pipe(
                prompt=args.text_prompt,
                image=init_image_np,  # Changed from init_image to image
                strength=args.strength,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps
            )

        # Save both reference and generated images
        reference_image.save("reference_image.png")
        output.images[0].save(args.output)
        
        print(f"Reference image saved as: reference_image.png")
        print(f"Generated image saved as: {args.output}")
    
    except Exception as e:
        print(f"Error during image generation: {str(e)}")
        raise  # Re-raise the exception for debugging

if __name__ == "__main__":
    main()