from sources.common.common import logger, processControl, log_

import torch
import joblib
import os

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

