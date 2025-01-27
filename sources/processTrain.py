from sources.common import logger, logProc, processControl, log_

import requests
import zipfile
import os
import torch
import shutil

def processTransformers(image_features, labels):
    from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
    from datasets import Dataset

    # Prepare Dataset
    labels = {"image1.jpg": 0, "image2.jpg": 1}
    X = [image_features[image_name].numpy().tolist() for image_name in labels.keys()]
    y = list(labels.values())
    dataset = Dataset.from_dict({"features": X, "labels": y})

    # Split Dataset
    train_test = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset, test_dataset = train_test["train"], train_test["test"]

    # Load Pretrained Model
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # Train
    training_args = TrainingArguments(output_dir="./results", evaluation_strategy="epoch", num_train_epochs=3)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset)
    trainer.train()

    # Evaluate
    metrics = trainer.evaluate()
    print(metrics)


import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import lightgbm as lgb


def prepare_data(features_file, labels):
    """
    Prepares data for training and testing with LightGBM.

    Args:
        features_file (str): Path to the saved features file (features.pth).
        labels (dict): Dictionary mapping image names to their cluster labels.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # Load features from file
    image_features = torch.load(features_file)

    # Prepare X and y
    X, y = [], []
    for image_name, features in image_features.items():
        if image_name in labels:  # Ensure label exists for the image
            X.append(features.numpy())  # Convert tensor to numpy array
            y.append(labels[image_name])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def processLightgbm(X_train, X_test, y_train, y_test):
    """
    Train and evaluate a LightGBM classifier.

    Args:
        X_train: Training feature set.
        X_test: Testing feature set.
        y_train: Training labels.
        y_test: Testing labels.
    """
    # Create LightGBM classifier
    clf = lgb.LGBMClassifier(random_state=42)

    # Train the classifier
    clf.fit(X_train, y_train)

    # Predict on test data
    y_pred = clf.predict(X_test)

    # Evaluate performance
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


def processTrain(featuresFile, imagesLabels):

    labels = {
        "image1.jpg": 0,
        "image2.jpg": 1,
        "image3.jpg": 4,
        # Add more mappings: image name -> cluster/label
    }

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(featuresFile, imagesLabels)

    # Train and evaluate model
    processLightgbm(X_train, X_test, y_train, y_test)

