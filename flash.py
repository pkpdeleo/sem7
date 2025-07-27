import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import timm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing
from transformers import CLIPProcessor, CLIPModel

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10 dataset loading and preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

# 1. CLIP Embedding (using Hugging Face Transformers)
def embed_with_clip(images):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image_batch = []
    for img_tensor in images:
        # Move tensor to CPU before converting to NumPy
        img_tensor_cpu = img_tensor.cpu()
        image_batch.append(Image.fromarray((img_tensor_cpu.numpy() * 255).astype(np.uint8).transpose(1, 2, 0)))

    inputs = processor(images=image_batch, return_tensors="pt").to(device)

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()

# 2. Vision Transformer (ViT) Embedding
def embed_with_vit(images):
    model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)
    model.eval()

    image_batch = []
    for img_tensor in images:
        image_batch.append(img_tensor.unsqueeze(0))

    processed_images = torch.cat(image_batch, dim=0).to(device)
    with torch.no_grad():
        image_features = model.forward_features(processed_images)
        image_features = image_features[:, 0]  # Extract the [CLS] token embedding
    return image_features.cpu().numpy()

# Evaluation Function (Cosine Similarity)
def evaluate_embeddings(train_embeddings, test_embeddings, train_labels, test_labels):
    similarities = cosine_similarity(test_embeddings, train_embeddings)
    top_k_indices = np.argsort(similarities, axis=1)[:, ::-1][:, :1]
    top_k_labels = train_labels[top_k_indices]
    accuracy = np.mean(top_k_labels.flatten() == test_labels)
    return accuracy

if __name__ == '__main__':
    multiprocessing.freeze_support()

    embedding_methods = {"CLIP": embed_with_clip, "ViT": embed_with_vit}

    for method_name, embed_func in embedding_methods.items():
        train_embeddings = []
        train_labels = []
        test_embeddings = []
        test_labels = []

        print(f"Embedding with {method_name}...")

        for images, labels in trainloader:
            images = images.to(device)
            embeddings = embed_func(images)
            train_embeddings.extend(embeddings)
            train_labels.extend(labels.numpy())

        for images, labels in testloader:
            images = images.to(device)
            embeddings = embed_func(images)
            test_embeddings.extend(embeddings)
            test_labels.extend(labels.numpy())

        train_embeddings = np.array(train_embeddings)
        test_embeddings = np.array(test_embeddings)
        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)

        accuracy = evaluate_embeddings(train_embeddings, test_embeddings, train_labels, test_labels)
        print(f"{method_name} Accuracy: {accuracy}")