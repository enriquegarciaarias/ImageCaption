"""
@Purpose: Main script for initializing environment settings and start procesing the Image Captioning project, handling main modes:
@Usage: Run `python mainProcess.py`.
"""

from sources.common.common import processControl, logger, log_
from sources.common.paramsManager import getConfigs
from sources.processFeatures import processFeatures
from sources.processTrain import processTrain, processApply
from sources.dataManager import saveModel


def mainProcess():
    """
    Main process for handling either model training or applying a trained model to new data.

    This function serves as the entry point for executing the main tasks in the pipeline. Depending on the operation mode specified
    (`MODEL` or `APPLY`), it either extracts features, trains a model, and saves it, or applies an existing trained model to new data.

    The function operates in two modes:
    - **MODEL mode**: Extracts features, trains the model, and saves it.
    - **APPLY mode**: Applies the trained model to new data for inference.

    :return: A boolean indicating the success of the main process.
    :rtype: bool
    """
    # Set model and pretrained dataset
    processControl.process['modelName'] = "ViT-L/14"
    processControl.process['pretrainedDataset'] = "laion2b_s32b_b82k"

    # MODEL mode: Extract features, train the model, and save it
    if processControl.args.proc == "MODEL":
        featuresFile, imagesLabels = processFeatures()
        model = processTrain(featuresFile, imagesLabels)
        result = saveModel(model, processControl.args.model)

    # APPLY mode: Apply the trained model to new data
    if processControl.args.proc == "APPLY":
        if processControl.args.model == "LLM":
            from sources.processLLM import processLLM
            processLLM()
        else:
            processApply()

    return True


if __name__ == '__main__':
    """
    Entry point for starting the main image caption process.

    This block of code is executed when the script is run directly. It logs the start of the process, retrieves configuration settings,
    and then triggers the main process. After the main process completes, it logs the completion of the task.

    The function performs the following steps:
    - Logs the start of the process.
    - Calls `getConfigs()` to retrieve necessary configurations.
    - Executes `mainProcess()` to handle model training or application.
    - Logs the completion of the process.

    :return: None
    """

    log_("info", logger, "********** STARTING Main Image Caption Process **********")
    getConfigs()
    mainProcess()

    log_("info", logger, "********** PROCESS COMPLETED **********")
