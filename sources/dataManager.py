from sources.common.common import logger, processControl, log_

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import lightgbm as lgb
import joblib
import os

def saveModel(model, type):
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

