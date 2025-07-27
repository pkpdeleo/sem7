import sys
import os
import argparse
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline
import requests
from io import BytesIO

# Import functions and global variables from your existing ep1.py pipeline.
from ep1 import (
    get_clip_image_embedding,
    get_clip_text_embedding,
    load_vector_db_index,
    load_image_filenames,
    retrieve_relevant_images,
    clip_model,
    clip_processor,
    VECTOR_DB_INDEX_PATH,
    IMAGE_FILENAMES_PATH
)

def load_image(image_path):
    """Load an image from a local path or URL."""
    try:
        if image_path.startswith("http://") or image_path.startswith("https://"):
            response = requests.get(image_path)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image from {image_path}: {e}")
        return None

def format_embedding(embedding, num_features=5):
    """
    Format the first few features of an embedding vector into a human-readable string.
    This is just for demonstration purposes, to emphasize the role of the embedding.
    """
    formatted = ", ".join([f"{x:.3f}" for x in embedding.flatten()[:num_features]])
    return formatted

def main():
    parser = argparse.ArgumentParser(
        description="Generate an image using a user prompt with extra emphasis on CLIP embeddings."
    )
    parser.add_argument("text_prompt", type=str, help="The text prompt for image generation.")
    parser.add_argument("--image", type=str, default=None, help="Optional path or URL to an image to boost embeddings influence.")
    parser.add_argument("--output", type=str, default="generated_image.png", help="Output filename for the generated image.")
    args = parser.parse_args()

    # Load existing FAISS index and image filenames for retrieval.
    index = load_vector_db_index(VECTOR_DB_INDEX_PATH)
    image_filenames = load_image_filenames(IMAGE_FILENAMES_PATH)
    if index is None or image_filenames is None:
        print("Error: Vector index or image filenames not found. Run the indexing pipeline first.")
        sys.exit(1)

    final_prompt = args.text_prompt
    embedding_info = ""
    retrieved_images = []
    
    # Check if an image prompt is provided.
    if args.image:
        print(f"Processing provided image prompt: {args.image}")
        image = load_image(args.image)
        if image is None:
            print("Failed to load the image prompt. Exiting.")
            sys.exit(1)
        # Get CLIP image embedding for the provided image.
        image_embedding = get_clip_image_embedding(args.image, clip_model, clip_processor)
        if image_embedding is None:
            print("Failed to generate embedding for the provided image prompt. Exiting.")
            sys.exit(1)
        # Format embedding information to include in the prompt.
        embedding_info = format_embedding(np.array(image_embedding))
        # Retrieve similar images based on the image embedding.
        retrieved_images = retrieve_relevant_images(image_embedding, index, image_filenames)
    else:
        # Use the text prompt to get a CLIP text embedding.
        print("No image prompt provided. Using text prompt for CLIP text embedding retrieval.")
        text_embedding = get_clip_text_embedding(args.text_prompt, clip_model, clip_processor)
        if text_embedding is None:
            print("Failed to generate CLIP text embedding. Exiting.")
            sys.exit(1)
        embedding_info = format_embedding(np.array(text_embedding))
        # Retrieve similar images based on the text embedding.
        retrieved_images = retrieve_relevant_images(text_embedding, index, image_filenames)

    # Emphasize the embeddings by injecting the embedding details multiple times.
    emphasis_text = (
        f"\nEMBEDDING DETAILS (high importance): {embedding_info}.\n"
        "The above embedding features are central to the image style and content. "
    )
    
    # Incorporate retrieved images (if any) and the emphasized embedding info into the final prompt.
    if retrieved_images:
        retrieved_str = " | ".join(retrieved_images)
        final_prompt += f" [Retrieved similar images: {retrieved_str}]."
        # Add extra embedding emphasis based on similarity.
        emphasis_text += "Retrieved images confirm these embedding characteristics. "
    
    final_prompt = emphasis_text + "\nPrompt: " + final_prompt

    print(f"Final prompt for image generation:\n{final_prompt}")

    # Initialize the Stable Diffusion pipeline for image generation.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = pipe.to(device)

    # Generate the image using the augmented final prompt.
    if device == "cuda":
        context = torch.autocast("cuda")
    else:
        context = torch.no_grad()
    with context:
        generated_image = pipe(final_prompt).images[0]

    # Save the generated image.
    output_path = args.output
    generated_image.save(output_path)
    print(f"Generated image saved to: {output_path}")

if __name__ == "__main__":
    main()