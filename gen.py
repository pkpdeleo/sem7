import sys
import os
import argparse
import torch
import pickle
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline
import requests
from io import BytesIO

# Import functions and global variables from your existing ep1.py
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

def main():
    parser = argparse.ArgumentParser(description="Generate an image using a text prompt and optionally an image prompt based on existing embeddings.")
    parser.add_argument("text_prompt", type=str, help="The text prompt for image generation.")
    parser.add_argument("--image", type=str, default=None, help="Optional path or URL to an image for hybrid (image+text) prompt.")
    parser.add_argument("--output", type=str, default="generated_image.png", help="Output filename for the generated image.")
    args = parser.parse_args()

    # Load existing FAISS index and image filenames (for retrieval)
    index = load_vector_db_index(VECTOR_DB_INDEX_PATH)
    image_filenames = load_image_filenames(IMAGE_FILENAMES_PATH)
    if index is None or image_filenames is None:
        print("Vector index or image filenames not found. Run the indexing pipeline first.")
        sys.exit(1)

    # Initialize the hybrid prompt content with the text prompt provided by user.
    final_prompt = args.text_prompt

    if args.image:
        # If an image prompt is provided, load it and get its embedding.
        print(f"Processing image prompt: {args.image}")
        image = load_image(args.image)
        if image is None:
            print("Failed to load the image prompt. Exiting.")
            sys.exit(1)
        # Get CLIP embedding for the image prompt.
        image_embedding = get_clip_image_embedding(args.image, clip_model, clip_processor)
        if image_embedding is None:
            print("Failed to generate embedding for the image prompt. Exiting.")
            sys.exit(1)
        # Retrieve similar images from the index based on the provided image.
        retrieved_images = retrieve_relevant_images(image_embedding, index, image_filenames)
        if retrieved_images:
            retrieved_str = " ".join(retrieved_images)
            # Append details from retrieved images into the prompt.
            final_prompt += f" Inspired by similar images: {retrieved_str}."
        else:
            print("No similar images retrieved from the index for the image prompt.")

    else:
        # If no image prompt is provided, use the text prompt to generate a CLIP text embedding and then perform retrieval.
        print("No image prompt provided. Using text prompt to retrieve similar images.")
        text_embedding = get_clip_text_embedding(args.text_prompt, clip_model, clip_processor)
        retrieved_images = retrieve_relevant_images(text_embedding, index, image_filenames)
        if retrieved_images:
            retrieved_str = " ".join(retrieved_images)
            final_prompt += f" Inspired by similar images: {retrieved_str}."
        else:
            print("No similar images retrieved using text prompt.")

    print(f"Final prompt for image generation:\n{final_prompt}")

    # Initialize the Stable Diffusion pipeline for image generation.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
    pipe = pipe.to(device)

    # Generate the image using the final prompt.
    with torch.autocast(device) if device=="cuda" else torch.no_grad():
        generated_image = pipe(final_prompt).images[0]

    # Save the generated image.
    output_path = args.output
    generated_image.save(output_path)
    print(f"Generated image saved to: {output_path}")

if __name__ == "__main__":
    main()