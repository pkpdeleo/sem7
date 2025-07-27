# Install the necessary library if you haven't already:
# pip install graphviz
    
import graphviz
import os

# Ensure the Graphviz executables are in your system's PATH or specify the path.
# For example, on Windows, you might need:
# os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
# Or on macOS/Linux, it's often handled by the installation.

# Create a new directed graph
dot = graphviz.Digraph(comment='Text-to-Image Generation Flow', format='png')
dot.attr(rankdir='TB', splines='ortho', bgcolor='white') # Top-to-bottom layout, orthogonal lines, white background
dot.attr('node', shape='box', style='rounded,filled', fillcolor='gray90', fontname='Helvetica') # Default node style
dot.attr('edge', fontname='Helvetica', fontsize='10')

# --- Define Nodes ---

# Inputs
dot.node('user_input', 'User Input', shape='ellipse', fillcolor='lightblue')
dot.node('img1_path', 'Image 1 Path', shape='box', style='rounded,filled', fillcolor='lightgrey') # Placeholder input
dot.node('img2_path', 'Image 2 Path', shape='box', style='rounded,filled', fillcolor='lightgrey') # Placeholder input
dot.node('img_kb', 'Image Knowledge Base', shape='box', style='rounded,filled', fillcolor='lightgrey')

# RAG Pipeline
dot.node('clip', 'CLIP Text Encoder')
dot.node('embedding', 'Text Embedding', shape='diamond', fillcolor='lightyellow')
dot.node('faiss', 'FAISS Index Search')
dot.node('topk', 'Top K Image Paths', shape='diamond', fillcolor='lightyellow')

# Preprocessing Subgraphs/Clusters
with dot.subgraph(name='cluster_0') as c:
    c.attr(label='IMAGE 1 PREPROCESSING', style='rounded', color='blue')
    c.attr(bgcolor='whitesmoke') # Background for the cluster box
    c.node('load1', 'Load Image 1')
    c.node('midas', 'Midas Depth Preprocessor')
    c.node('resize_depth', 'Resize Depth Map')
    # Edges within cluster 0
    c.edge('load1', 'midas')
    c.edge('midas', 'resize_depth')

with dot.subgraph(name='cluster_1') as c:
    c.attr(label='IMAGE 2 PREPROCESSING', style='rounded', color='blue')
    c.attr(bgcolor='whitesmoke') # Background for the cluster box
    c.node('load2', 'Load Image 2')
    c.node('canny', 'Canny Edge Preprocessor')
    c.node('resize_canny', 'Resize Canny Map')
    # Edges within cluster 1
    c.edge('load2', 'canny')
    c.edge('canny', 'resize_canny')

# Main Generation Pipeline
dot.node('controlnet', 'SDXL ControlNet Pipeline Base')
dot.node('latents', 'Generated Latents', shape='diamond', fillcolor='lightyellow')
dot.node('refiner', 'SDXL Refiner Pipeline')
dot.node('vae', 'Decode Latents via VAE')
dot.node('final_img', 'Final Generated Image', shape='ellipse', fillcolor='lightblue')

# --- Define Edges ---

# RAG Flow
dot.edge('user_input', 'clip')
dot.edge('clip', 'embedding')
dot.edge('embedding', 'faiss')
dot.edge('faiss', 'topk')
dot.edge('img_kb', 'faiss') # Connect KB to FAISS

# TopK to Preprocessing
# Using invisible nodes for cleaner routing from diamond if needed, or direct edge
dot.edge('topk', 'load1', lhead='cluster_0', label=' Path 1') # lhead points to the cluster boundary
dot.edge('topk', 'load2', lhead='cluster_1', label=' Path 2') # lhead points to the cluster boundary

# Preprocessing to ControlNet
dot.edge('resize_depth', 'controlnet', ltail='cluster_0') # ltail points from the cluster boundary
dot.edge('resize_canny', 'controlnet', ltail='cluster_1') # ltail points from the cluster boundary

# Text Prompt routing (Graphviz handles layout, but label the connection)
# Create an edge from user_input towards controlnet, label it. Layout engine will place it.
# For more control, might need specific rank settings or invisible nodes, but keep it simple first.
dot.edge('user_input', 'controlnet', label=' Text Prompt', constraint='false') # Constraint=false helps routing

# Generation Flow
dot.edge('controlnet', 'latents')
dot.edge('latents', 'refiner')
dot.edge('refiner', 'vae')
dot.edge('vae', 'final_img')

# --- Render and Save ---
try:
    output_filename = 'flowchart'
    dot.render(output_filename, view=False) # Saves flowchart.png (and .gv source)
    print(f"Flowchart saved as {output_filename}.png")
except Exception as e:
    print(f"Error rendering graph: {e}")
    print("Please ensure Graphviz is installed and its 'bin' directory is in your system's PATH.")