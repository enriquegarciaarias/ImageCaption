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
    from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
    from datasets import Dataset
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    import torch
    import os
    import numpy as np
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
    Prepares data for training and testing with LightGBM.

    Args:
        features_file (str): Path to the saved features file (features.pth).
        labels (dict): Dictionary mapping image names to their cluster labels.

    Returns:
        tuple: X_train, X_test, y_train, y_test
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


def OLDprocessTrainLightgbm(X_train, X_test, y_train, y_test):
    """
    Train and evaluate a LightGBM classifier.

    Args:
        X_train: Training feature set.
        X_test: Testing feature set.
        y_train: Training labels.
        y_test: Testing labels.
    """
    from sklearn.decomposition import PCA
    from sklearn.metrics import classification_report
    pca = PCA(n_components=processControl.defaults['features'])
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    log_("debug", logger, f"Varianza PCA: {pca.explained_variance_ratio_}")
    print("Train shape before LightGBM:", X_train.shape)
    print("Test shape before LightGBM:", X_test.shape)
    # Create LightGBM classifier
    clf = lgb.LGBMClassifier(random_state=42)

    # Train the classifier
    clf.fit(X_train, y_train)

    # Predict on test data
    y_pred = clf.predict(X_test)

    # Evaluate performance
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return clf

def processTrainLightgbm(X_train, X_test, y_train, y_test):
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


def OLDpredictLightgbm(X_new):
    """
    Load a trained LightGBM model and make predictions on new data.

    Args:
        model_path: Path to the saved model.
        X_new: New data for inference.

    Returns:
        Predictions for the new data.
    """
    modelPath = os.path.join(processControl.env['models'], "lightgbm_model.pkl")
    # Load the trained model
    if not os.path.exists(modelPath):
        raise FileNotFoundError(f"Model file not found at {modelPath}")

    clf = joblib.load(modelPath)
    log_("info", logger, f"Model loaded from {modelPath}")

    # Make predictions on new data
    predictions = clf.predict(X_new)
    return predictions



def predictLightgbm(imageFolder, imageList):
    """
    Extract image features, load trained LightGBM model, and make predictions.

    Args:
        image_list: List of new image filenames.

    Returns:
        Predicted clusters for new images.
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
    model = None

    if processControl.args.model == "lightgbm":
        X_train, X_test, y_train, y_test = prepare_data(featuresFile, imagesLabels)
        model = processTrainLightgbm(X_train, X_test, y_train, y_test)

    elif processControl.args.model == "transformers":
        model = processTrainTransformers(featuresFile, imagesLabels)
    return model

def processApply():
    imageFolder = processControl.env['inputPath']
    imageList = os.listdir(imageFolder)
    log_("info", logger, f"Extracting features for {len(imageList)} images with model {processControl.args.model}")
    if processControl.args.model == "lightgbm":
        imagesPredicted = predictLightgbm(imageFolder, imageList)

    if processControl.args.model == "transformers":
        imagesPredicted = predictTransformers(imageFolder, imageList)

    return imagesPredicted


