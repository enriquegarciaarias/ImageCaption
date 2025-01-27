from sources.common import logger, logProc, processControl, log_
from sources.processTrain import processTrain
import requests
import zipfile
import os
import torch
import shutil

def featureExtract(device="cuda" if torch.cuda.is_available() else "cpu"):
    image_folder = processControl.env['inputPath']
    output_file = os.path.join(processControl.env['outputPath'], "features.pth")
    from PIL import Image
    from tqdm import tqdm
    import open_clip

    def load_model():
        model_name = "ViT-L/14"  # Vision Transformer, large architecture
        pretrained_dataset = "laion2b_s32b_b82k"  # Pretrained on LAION-2B
        model, preprocess, _ = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained_dataset
        )
        model.eval()  # Set model to evaluation mode
        return model, preprocess

    model, preprocess = load_model()
    model.to(device)
    image_features = {}
    for image_name in tqdm(os.listdir(image_folder), desc="Extracting features"):
        try:
            image_path = os.path.join(image_folder, image_name)
            image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                features = model.encode_image(image).squeeze(0).cpu()  # Move to CPU for storage
            image_features[image_name] = features
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
    torch.save(image_features, output_file)
    print(f"Features saved to {output_file}")
    return output_file



def clusterImages(featuresFile):
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    import numpy as np

    # Load saved features
    image_features = torch.load(featuresFile)

    # Convert feature tensors to a matrix for clustering
    feature_matrix = torch.stack(list(image_features.values())).numpy()

    # Dimensionality reduction (optional)
    pca = PCA(n_components=50)  # Reduce to 50 dimensions
    reduced_features = pca.fit_transform(feature_matrix)

    # Clustering
    num_clusters = 5
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(reduced_features)

    centroids = kmeans.cluster_centers_

    # Assign images to clusters
    clustered_images = {i: [] for i in range(num_clusters)}
    for idx, image_name in enumerate(image_features.keys()):
        clustered_images[cluster_labels[idx]].append(image_name)

    # Print clusters
    for cluster, images in clustered_images.items():
        print(f"Cluster {cluster}: {len(images)} images")

    return clustered_images, centroids

def structureFiles(clustered_images):

    for index, images in clustered_images.items():
        # Create directory name
        dir_name = os.path.join(processControl.env['outputPath'], f"images_{index}")

        # Create the directory if it doesn't exist
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        # Move images to the directory
        for image in images:
            # Assuming images are in the current working directory
            imgSource = os.path.join(processControl.env['inputPath'], image)
            if os.path.exists(imgSource):
                shutil.copy(imgSource, os.path.join(dir_name, image))
            else:
                print(f"Image {image} not found.")

    print("Images have been organized into directories.")

def buildLabels(clustered_images):
    labels = {}
    for cluster_label, images in clustered_images.items():
        for image in images:
            labels[image] = cluster_label
    return labels


def processFeatures():
    featuresFile = featureExtract()
    clusteredImages, centroids = clusterImages(featuresFile)
    structureFiles(clusteredImages)
    imagesLabels = buildLabels(clusteredImages)
    processTrain(featuresFile, imagesLabels)
