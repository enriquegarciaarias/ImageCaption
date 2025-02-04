"""
@Purpose: Main script for initializing environment settings and processing corpus data, handling main modes:
          1) Social network corpus processing,
          2) Building models and updating datasets,
          3) Loading and processing APK data.

@Usage: Run `python mainProcess.py` or call via `manageArgs()`.

@Output: Processes specified corpus, updates datasets, and logs process details.
~/projects/common/library/deepmountain
~/projects/a6-corpus
"""

from sources.common.common import processControl, logger, log_
from sources.common.paramsManager import getConfigs
from sources.processFeatures import processFeatures
from sources.processTrain import processTrain, processApply
from sources.dataManager import saveModel


def mainProcess():
    processControl.process['modelName'] = "ViT-L/14"
    processControl.process['pretrainedDataset'] = "laion2b_s32b_b82k"

    if processControl.args.proc == "MODEL":
        featuresFile, imagesLabels = processFeatures()
        model = processTrain(featuresFile, imagesLabels)
        result = saveModel(model, processControl.args.model)

    if processControl.args.proc == "APPLY":
        processApply()
    return True


if __name__ == '__main__':
    log_("info", logger, "********** STARTING Main Image Caption Process **********")
    getConfigs()
    mainProcess()
    log_("info", logger, "********** PROCESS COMPLETED **********")
