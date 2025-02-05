from sources.common.common import logger, processControl, log_

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import lightgbm as lgb
import joblib
import os
import tqdm
import numpy as np

from sources.processFeatures import extractFeaturesForInference


def processTrainTransformers(features_file, labels):
    """
    Processes feature and label data to train a Transformer-based classification model.

    This function loads feature and label data, converts them into tensors, and prepares a dataset
    compatible with Hugging Face's `Trainer` API. It splits the data into training and testing sets,
    initializes a BERT-based model, and configures training parameters. The model is trained,
    evaluated, and saved to a specified directory. Additionally, the function generates plots
    to visualize validation loss and accuracy per epoch.

    :param features_file: Path to the file containing feature data.
    :type features_file: str
    :param labels: List or array of labels corresponding to the features.
    :type labels: list or numpy.ndarray

    :return: Path to the saved model directory.
    :rtype: str

    :raises Exception: If there is an issue with data loading, model training, or saving.
    """
    from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
    from datasets import Dataset
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    import matplotlib.pyplot as plt

    # Cargar datos
    X, y = loadFeatureslabels(features_file, labels)

    # Convertir X e y a tensores
    X = np.array(X)  # Convertir la lista de arrays a un solo numpy.ndarray
    X = torch.tensor(X)  # Convertir a tensor

    # Asegurar la forma correcta de X si es necesario
    if len(X.shape) == 2:  # Si falta la dimensión de secuencia
        X = X[:, np.newaxis, :]  # Añadir dimensión para secuencia

    y = torch.tensor(y)

    # Crear Dataset de Hugging Face
    dataset = Dataset.from_dict({
        "inputs_embeds": X.tolist(),  # Convertir directamente a listas compatibles
        "labels": y.tolist()
    })

    # División en entrenamiento y prueba
    train_test = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset, test_dataset = train_test["train"], train_test["test"]

    # Modelo preentrenado
    num_labels = len(set(y))  # Determinar dinámicamente el número de clases
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

    # Configuración del entrenamiento
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16
    )

    # Función personalizada de métricas
    def compute_metrics(pred):
        preds = pred.predictions.argmax(-1)
        precision = precision_score(pred.label_ids, preds, average='weighted')
        recall = recall_score(pred.label_ids, preds, average='weighted')
        f1 = f1_score(pred.label_ids, preds, average='weighted')
        accuracy = accuracy_score(pred.label_ids, preds)
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    # Entrenador
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # Entrenar el modelo
    training_output = trainer.train()

    # Evaluar el modelo
    metrics = trainer.evaluate()
    print(metrics)

    # Guardar el modelo
    modelPath = os.path.join(processControl.env['models'], "transformers_model")
    trainer.save_model(modelPath)
    log_("info", logger, f"Model saved to {modelPath}")

    # Graficar precisión y pérdida
    plt.figure(figsize=(12, 5))
    epochs = [log['epoch'] for log in trainer.state.log_history if 'epoch' in log]
    eval_losses = [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log]
    eval_accuracies = [log['eval_accuracy'] for log in trainer.state.log_history if 'eval_accuracy' in log]

    # Asegúrate de que todas las listas tengan la misma longitud
    min_len = min(len(epochs), len(eval_losses), len(eval_accuracies))

    # Recorta las listas para que tengan la misma longitud
    epochs = epochs[:min_len]
    eval_losses = eval_losses[:min_len]
    eval_accuracies = eval_accuracies[:min_len]

    # Gráfica de Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, eval_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss per Epoch')
    plt.legend()

    # Gráfica de Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, eval_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy per Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return modelPath


def predictTransformers(imageFolder, imageList):
    """
    Make predictions using a pre-trained transformer model.

    This function loads a pre-trained transformer model, extracts features from images in the specified folder, applies PCA transformation to these features, and then uses the model to make predictions on the transformed features.

    :param imageFolder: The folder containing the images for which predictions are to be made.
    :type imageFolder: str
    :param imageList: A list of image filenames for which predictions are to be made.
    :type imageList: list

    :return: The predictions made by the transformer model.
    :rtype: numpy.ndarray
    """

    from transformers import AutoModelForSequenceClassification, Trainer
    from datasets import Dataset
    import joblib
    import numpy as np
    import os

    modelPath = os.path.join(processControl.env['models'], "transformers_model")
    pcaPath = os.path.join(processControl.env['models'], "pca_transform.pkl")

    # Load trained transformer model
    model = AutoModelForSequenceClassification.from_pretrained(modelPath)

    X_new = extractFeaturesForInference(imageFolder)

    pca = joblib.load(os.path.join(processControl.env['models'], "pca_transform.pkl"))
    X_new = pca.transform(X_new)
    # Convert to Hugging Face dataset
    new_dataset = Dataset.from_dict({"features": X_new})

    # Load Trainer and make predictions
    trainer = Trainer(model=model)
    predictions = trainer.predict(new_dataset)

    return predictions.predictions

def loadFeatureslabels(features_file, labels):
    """
    Load image features and corresponding labels from a file.

    This function loads precomputed image features from a specified file and matches them with their corresponding labels.

    :param features_file: Path to the file containing saved image features.
    :type features_file: str
    :param labels: A dictionary mapping image names to their corresponding labels.
    :type labels: dict

    :return: A tuple containing two lists:
        - X (list of numpy.ndarray): The list of feature arrays for each image.
        - y (list): The list of labels corresponding to each image.
    :rtype: tuple
    """
    # image_features = torch.load(features_file)
    # weights_only=True se puede incluir para evitar warnings dado que solo queremos cargar los pesos del modelo
    image_features = torch.load(features_file, weights_only=True)
    X, y = [], []
    for image_name, features in image_features.items():
        if image_name in labels:  # Ensure label exists for the image
            X.append(features.numpy())  # Convert tensor to numpy array
            y.append(labels[image_name])
    return X, y

def prepare_data(features_file, labels):
    """
    Prepare training and testing data from precomputed features and labels.

    This function loads image features and corresponding labels from a specified file, splits the data into training and testing sets, and applies oversampling to the training set if needed. SMOTE or RandomOverSampler can be used to balance the class distribution in the training data.

    :param features_file: Path to the file containing the precomputed image features.
    :type features_file: str
    :param labels: A dictionary mapping image names to their corresponding labels.
    :type labels: dict

    :return: A tuple containing the following:
        - X_train (numpy.ndarray): The feature matrix for the training set.
        - X_test (numpy.ndarray): The feature matrix for the test set.
        - y_train (numpy.ndarray): The labels for the training set.
        - y_test (numpy.ndarray): The labels for the test set.
    :rtype: tuple
    """
    X, y = loadFeatureslabels(features_file, labels)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    if processControl.defaults['smoteFeatures']:
        from imblearn.over_sampling import SMOTE, RandomOverSampler
        ros = RandomOverSampler(random_state=42)
        X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
        # El dataset es demasiado pequeño para smote
        #smote = SMOTE(k_neighbors=2, random_state=42)
        #X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        X_train, y_train = X_train_resampled, y_train_resampled
        
    return X_train, X_test, y_train, y_test


def processTrainLightgbm(X_train, X_test, y_train, y_test):
    """
    Train a LightGBM model on preprocessed training data and evaluate its performance.

    This function normalizes the input features using StandardScaler, applies PCA for dimensionality reduction, trains a LightGBM classifier on the transformed features, and evaluates the model on the test set. A classification report is printed to show the model's performance.

    :param X_train: Training feature matrix.
    :type X_train: numpy.ndarray
    :param X_test: Testing feature matrix.
    :type X_test: numpy.ndarray
    :param y_train: Training labels.
    :type y_train: numpy.ndarray
    :param y_test: Testing labels.
    :type y_test: numpy.ndarray

    :return: The trained LightGBM classifier.
    :rtype: lightgbm.LGBMClassifier

    :raises ValueError: If the input feature matrices do not have matching numbers of samples.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report
    import lightgbm as lgb

    # Normalización
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # PCA
    pca = PCA(n_components=processControl.defaults['features'])
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    print("Train shape before LightGBM:", X_train.shape)
    print("Test shape before LightGBM:", X_test.shape)

    # LightGBM con ajuste de hiperparámetros
    clf = lgb.LGBMClassifier(
        num_leaves=31,
        min_data_in_leaf=5,
        learning_rate=0.05,
        n_estimators=100,
        is_unbalance=True,
        random_state=42,
        force_row_wise=True
    )

    # Entrenamiento
    clf.fit(X_train, y_train)

    # Predicción y evaluación
    y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return clf


def predictLightgbm(imageFolder, imageList):
    """
    Predict the labels for a list of images using a pre-trained LightGBM model.

    This function loads a pre-trained LightGBM model and PCA transformation, extracts features for the new images, applies PCA, and makes predictions using the loaded model.

    :param imageFolder: Path to the folder containing the images for which predictions are to be made.
    :type imageFolder: str
    :param imageList: List of image filenames for which predictions are to be generated.
    :type imageList: list

    :return: A dictionary mapping image filenames to their predicted labels.
    :rtype: dict

    :raises FileNotFoundError: If the model file is not found at the specified path.
    :raises ValueError: If no valid image features are extracted from the input images.
    """
    modelPath = os.path.join(processControl.env['models'], "lightgbm_model.pkl")

    if not os.path.exists(modelPath):
        raise FileNotFoundError(f"Model file not found at {modelPath}")

    clf = joblib.load(modelPath)
    log_("info", logger, f"Model loaded from {modelPath}")

    # Extract features for new images
    X_new = extractFeaturesForInference(imageFolder)
    pca = joblib.load(os.path.join(processControl.env['models'], "pca_transform.pkl"))
    X_new = pca.transform(X_new)
    assert X_new.shape[1] == processControl.defaults['features'], f"Expected {processControl.defaults['features']} features, got {X_new.shape[1]}"

    if X_new.shape[0] == 0:
        raise ValueError("No valid image features extracted. Check your input images.")

    # Make predictions
    predictions = clf.predict(X_new)
    return dict(zip(imageList, predictions))  # Return as dictionary (image -> cluster)


def processTrain(featuresFile, imagesLabels):
    """
    Train a model based on the specified model type in processControl.

    This function trains a model using either LightGBM or transformers, depending on the model type specified in `processControl.args.model`. It prepares the training data, trains the model, and returns the trained model.

    :param featuresFile: Path to the file containing the image features.
    :type featuresFile: str
    :param imagesLabels: A dictionary mapping image names to their corresponding labels.
    :type imagesLabels: dict

    :return: The trained model.
    :rtype: model
    """
    model = None

    if processControl.args.model == "lightgbm":
        X_train, X_test, y_train, y_test = prepare_data(featuresFile, imagesLabels)
        model = processTrainLightgbm(X_train, X_test, y_train, y_test)

    elif processControl.args.model == "transformers":
        model = processTrainTransformers(featuresFile, imagesLabels)
    return model

def processApply():
    """
    Applies the selected model (LightGBM or Transformers) to predict image clusters.

    This function extracts features from images located in the specified input path,
    applies the selected model to make predictions, and returns the predicted clusters.

    :return: A dictionary mapping image filenames to their predicted clusters.
    :rtype: dict
    """
    imageFolder = processControl.env['inputPath']
    imageList = os.listdir(imageFolder)
    log_("info", logger, f"Extracting features for {len(imageList)} images with model {processControl.args.model}")
    if processControl.args.model == "lightgbm":
        imagesPredicted = predictLightgbm(imageFolder, imageList)

    if processControl.args.model == "transformers":
        imagesPredicted = predictTransformers(imageFolder, imageList)

    return imagesPredicted
