# config.py
import torch
import os

# --- Models ---
# Core Models
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
BASE_SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
REFINER_SDXL_MODEL = "stabilityai/stable-diffusion-xl-refiner-1.0"
VAE_SDXL_MODEL = "madebyollin/sdxl-vae-fp16-fix" # Recommended SDXL VAE

# ControlNets (Ensure these are compatible diffusers models for SDXL 1.0)
CONTROLNET_MODEL_DEPTH_SDXL = "diffusers/controlnet-depth-sdxl-1.0"
CONTROLNET_MODEL_CANNY_SDXL = "diffusers/controlnet-canny-sdxl-1.0"
# Add more ControlNet models here if needed, e.g.:
# CONTROLNET_MODEL_POSE_SDXL = "thibaud/controlnet-openpose-sdxl-1.0"

# Preprocessor Models (used by controlnet_aux)
PREPROCESSOR_ANNOTATOR_REPO = "lllyasviel/Annotators"

# --- File Paths ---
# Get the directory where this config file is located
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_IMAGE_FOLDER = os.path.join(_PROJECT_DIR, "image_knowledge_base_xl")
VECTOR_DB_INDEX_PATH = os.path.join(_PROJECT_DIR, "image_rag_sdxl_controlnet.faiss")
IMAGE_FILENAMES_PATH = os.path.join(_PROJECT_DIR, "image_filenames_sdxl_controlnet.pkl")
DEFAULT_OUTPUT_PATH = os.path.join(_PROJECT_DIR, "generated_sdxl_realistic.png")

# --- Hardware & Performance ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# SDXL runs best in float16 on GPU
DTYPE_VAE = torch.float16
DTYPE_CONTROL = torch.float16
DTYPE_UNET = torch.float16 # Base model components
DTYPE_REFINER = torch.float16
# Set to True to use less VRAM at the cost of speed
DEFAULT_ENABLE_CPU_OFFLOAD = False
# Set to True to potentially reduce VRAM usage further, might impact speed/quality slightly
DEFAULT_ENABLE_ATTENTION_SLICING = False # Fooocus often enables this by default

# --- Default Generation Parameters ---
DEFAULT_RESOLUTION = 1024
DEFAULT_NUM_STEPS = 40 # Slightly lower default, adjust as needed
DEFAULT_GUIDANCE_SCALE = 7.0
DEFAULT_REFINER_RATIO = 0.2 # 20% of steps for refiner
DEFAULT_DEPTH_SCALE = 0.5 # Adjusted default scales
DEFAULT_CANNY_SCALE = 0.5
DEFAULT_SEED = None # No fixed seed by default

# --- RAG Parameters ---
DEFAULT_TOP_K_RAG = 2 # Retrieve top 2 images for Depth and Canny

# --- Prompting ---
# Inspired by Fooocus Styles (add more as needed)
STYLES = {
    "default": ("", ""), # No added style
    "photorealistic": ("photorealistic DSLR photo, photograph, high detail, sharp focus, natural lighting, cinematic composition, ",
                       " painting, drawing, illustration, sketch, cartoon, anime, 3D render, CGI, unrealistic, artificial, blurry, low quality, noisy, text, signature, watermark"),
    "cinematic": ("cinematic film still, dramatic lighting, shallow depth of field, high detail, sharp focus, ",
                  " painting, drawing, illustration, sketch, cartoon, anime, 3D render, CGI, unrealistic, artificial, blurry, low quality, noisy, text, signature, watermark, boring"),
    "anime": ("anime artwork, anime style, key visual, vibrant colors, sharp lines, detailed background, ",
              " photo, photograph, realistic, 3D render, CGI, blurry, low quality, text, signature, watermark"),
    "illustration": ("illustration, detailed illustration, vibrant colors, high fantasy, intricate details, painterly style, ",
                    " photo, photograph, realistic, 3D render, CGI, blurry, low quality, text, signature, watermark"),
}
DEFAULT_STYLE = "photorealistic" # Default style to apply

# Add negative words commonly used
DEFAULT_NEGATIVE_PROMPT_PREFIX = "worst quality, low quality, jpeg artifacts, blurry, noisy, text, signature, watermark, username, artist name, deformed, mutated, disfigured, morbid, mutilated, extra limbs, missing limbs, duplicate"