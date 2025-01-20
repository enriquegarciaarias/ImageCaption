"""
@Purpose: Handles project-wide parameters
@Usage: Functions called by the main process
"""

import argparse
import json
import os
import sys
import shutil
from sources.common import processControl
from sources.utils import configLoader, dbTimestamp

# Constants for parameter files
JSON_PARMS = "config.json"

def manageArgs():
    """
    @Desc: Parse command-line arguments to configure the process.
    @Result: Returns parsed arguments as a Namespace object.
    """
    parser = argparse.ArgumentParser(description="Main process for Corpus handling.")
    parser.add_argument('--debug', type=str, help='Debug level: 0 Error, 1 Debug, 2 Info', default="DEBUG")
    parser.add_argument('--proc', type=str, help="Process type: Check, CORPUS, MODEL, APPLY", default="APPLY")
    parser.add_argument('--model', type=str, help="algorithm", default="ByLSTM")
    parser.add_argument('--corpus', type=str, help="Name of corpus", default="NLLP2021")
    return parser.parse_args()


def manageEnv():
    """
    @Desc: Defines environment paths and variables.
    @Result: Returns a dictionary containing environment paths.
    """

    config = configLoader()
    environment = config.get_environment()

    env_data = {
        "timestamp": dbTimestamp(),
        "realPath": environment["realPath"],
        "inputPath": os.path.join(environment["realPath"], environment["inputPath"]),
        "outputPath": os.path.join(environment["realPath"], environment["outputPath"]),
        ".pycache": os.path.join(environment["realPath"], environment[".pycache"]),
        "data": os.path.join(environment["realPath"], environment["data"]),
    }

    os.makedirs(env_data['.pycache'], exist_ok=True)
    os.environ['PYTHONPYCACHEPREFIX'] = env_data['.pycache']
    sys.pycache_prefix = env_data['.pycache']
    return env_data


def getConfigs():
    """
    @Desc: Load environment settings, arguments, and hyperparameters.
    @Result: Stores configurations in processControl variables.
    """

    processControl.env = manageEnv()
    processControl.args = manageArgs()


