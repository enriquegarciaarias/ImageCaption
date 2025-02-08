from sources.common.common import logger, processControl, log_
from sources.dataManager import saveModel
import os
import torch
import shutil
import joblib
import numpy as np
from PIL import Image
from tqdm import tqdm
import open_clip
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def featureExtract(imageFolder, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Extract features from images in a specified folder using a pre-trained model.

    This function processes the images in the specified folder and extracts their features using a pre-trained model.
    The extracted features are returned as a dictionary where the keys are image names and the values are the feature vectors.

    :param imageFolder: Path to the folder containing images from which features will be extracted.
    :type imageFolder: str
    :param device: The device on which the model will run, either 'cuda' for GPU or 'cpu'. Defaults to 'cuda' if available.
    :type device: str, optional

    :return: A dictionary containing the extracted features for each image, with the image names as keys.
    :rtype: dict
    """
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
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    for image_name in tqdm(os.listdir(imageFolder), desc="Extracting features"):
        if os.path.splitext(image_name)[1].lower() in supported_extensions:
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
    Extract features from images in a specified folder for inference.

    This function extracts features from images stored in a folder, typically used for inference. The extracted features
    are returned as a numpy array, which can then be processed or used for prediction.

    :param imageFolder: Path to the folder containing images for which features are to be extracted.
    :type imageFolder: str

    :return: A numpy array containing the extracted features for each image in the folder.
    :rtype: numpy.ndarray
    """
    features = featureExtract(imageFolder)
    image_features = []
    for feature in features.values():
        image_features.append(feature)
    return np.array(image_features)


def clusterImages(featuresFile):
    """
    Perform clustering on image features to group images based on similarity.

    This function loads precomputed image features from a specified file, applies PCA for dimensionality reduction,
    and then clusters the images using KMeans. The images are grouped into clusters based on their feature vectors.

    :param featuresFile: Path to the file containing saved image features.
    :type featuresFile: str

    :return: A tuple containing:
        - clustered_images (dict): A dictionary mapping cluster labels to lists of image names.
        - centroids (numpy.ndarray): The cluster centers after the KMeans clustering.
    :rtype: tuple
    """
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
    """
    Organize images into directories based on their cluster labels.

    This function takes a dictionary of clustered images, where each key is a cluster label and
    the corresponding value is a list of image names. It creates directories named by cluster labels and
    moves the images into the appropriate directories.

    :param clustered_images: A dictionary mapping cluster labels to lists of image names belonging to those clusters.
    :type clustered_images: dict

    :return: None
    :rtype: None
    """
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
    """
    Build a mapping of image names to their corresponding cluster labels.

    This function takes a dictionary of clustered images, where each key is a cluster label and
    the corresponding value is a list of images in that cluster. It then creates a mapping of image names
    to their respective cluster labels.

    :param clustered_images: A dictionary mapping cluster labels to lists of image names belonging to those clusters.
    :type clustered_images: dict

    :return: A dictionary mapping image names to their corresponding cluster labels.
    :rtype: dict
    """
    labels = {}
    for cluster_label, images in clustered_images.items():
        for image in images:
            labels[image] = cluster_label
    return labels


def optimizeDimensions(image_features):
    """
    Optimize the dimensionality of image features using PCA.

    This function applies Principal Component Analysis (PCA) to reduce the dimensionality
    of the extracted image features while retaining the most important variance. It computes
    the cumulative explained variance and plots it to help select the optimal number of PCA components
    that capture a desired level of variance (e.g., 95%).

    :param image_features: A dictionary mapping image names to their corresponding feature tensors.
    :type image_features: dict

    :return: A dictionary mapping image names to their corresponding reduced feature arrays.
    :rtype: dict

    :note: The number of PCA components is dynamically chosen to capture at least 95% of the explained variance.
    """
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
    """
    Extract, optimize, cluster, and organize image features.

    This function performs the following steps:
    1. Extracts features from images in the specified input directory.
    2. Optimizes the dimensionality of the extracted features using PCA.
    3. Saves the extracted features to a file.
    4. Clusters the images based on their features.
    5. Organizes the images into directories based on their cluster assignments.
    6. Builds a mapping of images to their respective cluster labels.

    :return: A tuple containing:
        - featuresFile (str): The path to the file containing the extracted image features.
        - imagesLabels (dict): A dictionary mapping image names to their corresponding cluster labels.
    :rtype: tuple
    """
    # Step 1: Extract features from images in the input directory
    imageFeatures = featureExtract(processControl.env['inputPath'])

    # Step 2: Optimize the dimensionality of the extracted features using PCA
    imageFeatures2 = optimizeDimensions(imageFeatures)

    # Step 3: Save the extracted features to a file
    featuresFile = saveModel(imageFeatures, "features")

    # Step 4: Cluster the images based on their features
    clusteredImages, centroids = clusterImages(featuresFile)

    # Step 5: Organize the images into directories based on their clusters
    structureFiles(clusteredImages)

    # Step 6: Build a mapping of images to their cluster labels
    imagesLabels = buildLabels(clusteredImages)

    # Return the path to the features file and the image-to-label mapping
    return featuresFile, imagesLabels
