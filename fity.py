import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

def extract_embeddings(dataloader, model):
    embeddings = []
    labels = []
    with torch.no_grad():
        for images, label in dataloader:
            images = images.to(device)
            outputs = model(images)
            outputs = outputs.view(outputs.size(0), -1)
            embeddings.append(outputs.cpu().numpy())
            labels.append(label.numpy())
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    return embeddings, labels

def find_similar(query_embedding, train_embeddings, top_k=5):
    similarities = cosine_similarity([query_embedding], train_embeddings)
    similar_indices = similarities.argsort()[0][-top_k:][::-1]
    return similar_indices, similarities[0][similar_indices]

def show_images(query_idx, similar_indices, query_dataset, similar_dataset):
    plt.figure(figsize=(15, 5))
    
    # Query Image
    ax = plt.subplot(1, len(similar_indices) + 1, 1)
    image, label = query_dataset[query_idx]
    image = image.permute(1, 2, 0)
    image = image * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
    image = image.numpy()
    plt.imshow(image)
    plt.title("Query Image")
    plt.axis('off')
    
    # Similar Images
    for i, idx in enumerate(similar_indices):
        ax = plt.subplot(1, len(similar_indices) + 1, i + 2)
        image, label = similar_dataset[idx]
        image = image.permute(1, 2, 0)
        image = image * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        image = image.numpy()
        plt.imshow(image)
        plt.title(f"Similar {i + 1}")
        plt.axis('off')
    
    plt.show()

if __name__ == '__main__':
    # 1. Load and preprocess the dataset
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=False, num_workers=2)

    # 2. Load pre-trained ResNet-50 and modify it
    model = models.resnet50(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 3. Extract embeddings
    train_embeddings, train_labels = extract_embeddings(trainloader, model)
    test_embeddings, test_labels = extract_embeddings(testloader, model)

    # Example query
    query_idx = 100  # Change as needed
    query_embedding = test_embeddings[query_idx]
    similar_indices, similarities = find_similar(query_embedding, train_embeddings, top_k=5)
    print("Top 5 similar images indices:", similar_indices)
    print("Similarities:", similarities)
    show_images(query_idx, similar_indices, testset, trainset)  # Pass both datasets