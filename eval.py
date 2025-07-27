# evaluate_similarity.py
import numpy as np
import os
import argparse

# Assuming rag_utils.py is in the same directory or Python path
try:
    import rag_utils
except ImportError:
    print("Error: Could not import rag_utils.py.")
    print("Ensure rag_utils.py is in the same directory or your PYTHONPATH.")
    exit(1)

def calculate_clip_similarity(image_path1: str, image_path2: str):
    """
    Calculates the cosine similarity between the CLIP embeddings of two images.

    Args:
        image_path1: Path to the first image.
        image_path2: Path to the second image.

    Returns:
        Cosine similarity score (float) between -1 and 1, or None if an error occurs.
    """
    print(f"Calculating similarity between '{os.path.basename(image_path1)}' and '{os.path.basename(image_path2)}'")

    # Load CLIP model and processor (uses caching from rag_utils)
    try:
        rag_utils.load_clip_models()
        print("CLIP models loaded/retrieved.")
    except Exception as e:
        print(f"Fatal: Could not load CLIP model/processor. Error: {e}")
        return None

    # Load images
    img1_pil = rag_utils.load_image(image_path1)
    img2_pil = rag_utils.load_image(image_path2)

    if img1_pil is None or img2_pil is None:
        print("Error loading one or both images.")
        return None

    # Get embeddings (function handles normalization)
    print("Generating embeddings...")
    emb1 = rag_utils.get_clip_image_embedding(img1_pil)
    emb2 = rag_utils.get_clip_image_embedding(img2_pil)

    if emb1 is None or emb2 is None:
        print("Error generating one or both embeddings.")
        return None

    # Calculate cosine similarity
    # Since embeddings are L2 normalized, dot product equals cosine similarity
    # Ensure embeddings are flattened if they have an extra dimension
    emb1_flat = emb1.flatten()
    emb2_flat = emb2.flatten()
    similarity = np.dot(emb1_flat, emb2_flat)

    # Clamp similarity score to [-1, 1] due to potential float precision issues
    similarity = np.clip(similarity, -1.0, 1.0)

    print(f"Embeddings generated. Similarity calculated.")
    return float(similarity) # Return as standard float

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate CLIP similarity between a generated image and its source images.")
    parser.add_argument("generated_image_path", type=str, help="Path to the generated image.")
    parser.add_argument("source_image1_path", type=str, help="Path to the first source image (e.g., used for Depth).")
    parser.add_argument("source_image2_path", type=str, help="Path to the second source image (e.g., used for Canny).")

    args = parser.parse_args()

    print("\n--- Starting Similarity Evaluation ---")

    # Compare Generated vs Source 1
    print("\nComparing Generated vs. Source 1...")
    similarity_gen_src1 = calculate_clip_similarity(args.generated_image_path, args.source_image1_path)
    if similarity_gen_src1 is not None:
        print(f"==> CLIP Similarity (Generated vs Source 1): {similarity_gen_src1:.4f}")

    # Compare Generated vs Source 2
    print("\nComparing Generated vs. Source 2...")
    similarity_gen_src2 = calculate_clip_similarity(args.generated_image_path, args.source_image2_path)
    if similarity_gen_src2 is not None:
        print(f"==> CLIP Similarity (Generated vs Source 2): {similarity_gen_src2:.4f}")

    # Optional: Compare Source 1 vs Source 2 (to see how different the sources were)
    print("\nComparing Source 1 vs. Source 2...")
    similarity_src1_src2 = calculate_clip_similarity(args.source_image1_path, args.source_image2_path)
    if similarity_src1_src2 is not None:
        print(f"==> CLIP Similarity (Source 1 vs Source 2): {similarity_src1_src2:.4f}")

    print("\n--- Evaluation Complete ---")