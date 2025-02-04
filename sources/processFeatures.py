from sources.common.common import logger, processControl, log_

from sources.dataManager import saveModel
import os
import torch
import shutil
import joblib
import numpy as np

def featureExtract(imageFolder, device="cuda" if torch.cuda.is_available() else "cpu"):
    from PIL import Image
    from tqdm import tqdm
    import open_clip

    def load_model():
        model, preprocess, _ = open_clip.create_model_and_transforms(
            model_name=processControl.process['modelName'],
            pretrained=processControl.process['pretrainedDataset']
        )
        model.eval()  # Set model to evaluation mode
        return model, preprocess

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_model()
    model.to(device)

    image_features = {}
    for image_name in tqdm(os.listdir(imageFolder), desc="Extracting features"):
        image_path = os.path.join(processControl.env['inputPath'], image_name)
        if not os.path.exists(image_path):
            log_("error", logger, f"Image {image_name} not found.")
            continue  # Skip missing images
        try:
            image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                features = model.encode_image(image).squeeze(0).cpu()  # Move to CPU for storage
            image_features[image_name] = features
        except Exception as e:
            log_("exception", logger, f"Error processing {image_name}: {e}")

    return image_features

def OLDfeatureExtract(device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Extracts image features from a folder of images using a pretrained Vision Transformer model.
    The features are computed and stored as a PyTorch tensor on the disk for later retrieval or analysis.

    Arguments:
        device: str
            The device to perform computation on. Defaults to 'cuda' if available, otherwise 'cpu'.

    Returns:
        str
            The path to the file where the extracted features are stored.
    """
    image_folder = processControl.env['inputPath']

    from PIL import Image
    from tqdm import tqdm
    import open_clip

    def load_model():
        model, preprocess, _ = open_clip.create_model_and_transforms(
            model_name=processControl.process['modelName'] ,
            pretrained=processControl.process['pretrainedDataset']
        )
        model.eval()  # Set model to evaluation mode
        return model, preprocess

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_model()
    model.to(device)

    image_features = {}
    for image_name in tqdm(os.listdir(image_folder), desc="Extracting features"):
        image_path = os.path.join(processControl.env['inputPath'], image_name)
        if not os.path.exists(image_path):
            log_("error", logger, f"Image {image_name} not found.")
            continue  # Skip missing images
        try:
            image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                features = model.encode_image(image).squeeze(0).cpu()  # Move to CPU for storage
            image_features[image_name] = features
        except Exception as e:
            log_("exception", logger, f"Error processing {image_name}: {e}")


    return image_features

def extractFeaturesForInference(imageFolder):
    """
    Extract features for new images before making predictions.
    Args:
        image_list: List of new image filenames.
        device: Computation device (cuda or cpu).

    Returns:
        numpy array of extracted image features.
    """
    features = featureExtract(imageFolder)
    image_features = []
    for feature in features.values():
        image_features.append(feature)
    return np.array(image_features)

"""
    from PIL import Image
    import open_clip

    # Load the model and preprocess functions
    model, preprocess, tokenizer = open_clip.create_model_and_transforms(
        model_name=processControl.process['modelName'],
        pretrained=processControl.process['pretrainedDataset']
    )

    model.eval().to(device)

    image_features = []
    for image_name in os.listdir(imageFolder):

        image_path = os.path.join(processControl.env['inputPath'], image_name)
        if not os.path.exists(image_path):
            log_("error", logger, f"Image {image_name} not found.")
            continue  # Skip missing images

        try:
            image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                features = model.encode_image(image).squeeze(0).cpu().numpy()
            image_features.append(features)
        except Exception as e:
            log_("error", logger, f"Error processing {image_name}: {e}")

    return np.array(image_features)
"""

def clusterImages(featuresFile):
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    # Load saved features
    # weights_only=True se puede incluir para evitar warnings dado que solo queremos cargar los pesos del modelo
    image_features = torch.load(featuresFile, weights_only=True)

    # Convert feature tensors to a matrix for clustering
    feature_matrix = torch.stack(list(image_features.values())).numpy()

    # Dimensionality reduction (optional)
    pca = PCA(n_components=processControl.defaults['features'])  # Reduce dimensions
    reduced_features = pca.fit_transform(feature_matrix)
    joblib.dump(pca, os.path.join(processControl.env['models'], "pca_transform.pkl"))
    # Clustering
    num_clusters = len(processControl.defaults['imageClasses'])
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(reduced_features)  # Train clustering
    cluster_labels = kmeans.predict(reduced_features)

    centroids = kmeans.cluster_centers_

    # Assign images to clusters
    clustered_images = {i: [] for i in range(num_clusters)}
    for idx, image_name in enumerate(image_features.keys()):
        clustered_images[cluster_labels[idx]].append(image_name)

    # Print clusters
    for cluster, images in clustered_images.items():
        log_("info", logger, f"Clustering {len(images)} images")

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
                log_("error", logger, f"Image {image} not found.")

    log_("info", logger, f"Images organized into directories.")

def buildLabels(clustered_images):
    labels = {}
    for cluster_label, images in clustered_images.items():
        for image in images:
            labels[image] = cluster_label
    return labels


def optimizeDimensions(image_features):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    feature_matrix = np.array([tensor.numpy() for tensor in image_features.values()])
    image_names = list(image_features.keys())
    # Initialize PCA without specifying n_components to analyze variance
    pca = PCA()
    pca.fit(feature_matrix)

    # Compute cumulative explained variance
    explained_variance = np.cumsum(pca.explained_variance_ratio_)

    # Plot variance to choose an optimal number of components
    plt.plot(range(1, len(explained_variance) + 1), explained_variance)
    plt.xlabel('Number of PCA Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance by PCA Components')
    plt.grid()
    plt.show()

    # Choose n_components dynamically, e.g., for 95% explained variance
    optimal_components = np.argmax(explained_variance >= 0.95) + 1
    log_("info", logger, f"Optimal number of PCA components: {optimal_components}")
    # Apply PCA with the optimal number of components
    pca = PCA(n_components=optimal_components)
    reduced_features = pca.fit_transform(feature_matrix)
    # ✅ Convert reduced feature array back into a dictionary
    reduced_feature_dict = {image_names[i]: reduced_features[i] for i in range(len(image_names))}

    return reduced_feature_dict

def processFeatures():
    imageFeatures = featureExtract(processControl.env['inputPath'])
    imageFeatures2 = optimizeDimensions(imageFeatures)
    featuresFile = saveModel(imageFeatures, "features")
    clusteredImages, centroids = clusterImages(featuresFile)
    structureFiles(clusteredImages)
    imagesLabels = buildLabels(clusteredImages)
    return featuresFile, imagesLabels
