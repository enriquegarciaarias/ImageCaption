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

import argparse
import os
from os.path import realpath


from sources.common import logger, logProc, processControl, log_
from sources.utils import configLoader
from sources.paramsManager import getConfigs
from sources.processFeatures import processFeatures



def mainProcess():
    processFeatures()

    return True



if __name__ == '__main__':
    log_("info", logger, "********** STARTING Main Image Caption Process **********")

    getConfigs()


    mainProcess()
    log_("info", logger, "********** PROCESS COMPLETED **********")

