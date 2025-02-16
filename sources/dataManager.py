from sources.common.common import logger, processControl, log_

import torch
import joblib
import os
import shutil

def saveModel(model, type):
    """
    Save the model to a specified file based on its type.

    This function saves the trained model to a file depending on the specified type. It supports saving LightGBM models
    as `.pkl` files and PyTorch models as `.pth` files. If an error occurs during the saving process, it raises an exception.

    :param model: The trained model to be saved.
    :type model: object
    :param type: The type of the model, which determines the file format to be used.
    :type type: str

    :return: The path where the model was saved.
    :rtype: str

    :raises Exception: If an error occurs during the model saving process.
    """
    try:
        if type == "lightgbm":
            modelPath = os.path.join(processControl.env['models'], "lightgbm_model.pkl")
            joblib.dump(model, modelPath)

        if type == "features":
            modelPath = os.path.join(processControl.env['outputPath'], "features.pth")
            torch.save(model, modelPath)

    except Exception as e:
        raise Exception(f"Couldn't save model: {e}")

    log_("info", logger, f"Model type: {type} saved to {modelPath}")
    return modelPath

def writeFilesCategories(clusteredImages, model):
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
    dirModelPath = os.path.join(processControl.env['outputPath'], model)
    if not os.path.exists(dirModelPath):
        os.makedirs(dirModelPath)
    for index, image_info in enumerate(clusteredImages):

        dirCategory = os.path.join(dirModelPath, f"category_{image_info['category']}")
        if not os.path.exists(dirCategory):
            os.makedirs(dirCategory)
        shutil.copy(image_info['path'], os.path.join(dirCategory, image_info['name']))

    log_("info", logger, f"Images organized into directories.")