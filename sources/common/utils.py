from sources.common.common import logger, processControl, log_
import json

import time
import os
from os.path import isdir
from huggingface_hub import login


def mkdir(dir_path):
    """
    @Desc: Creates directory if it doesn't exist.
    @Usage: Ensures a directory exists before proceeding with file operations.
    """
    if not isdir(dir_path):
        os.makedirs(dir_path)


def dbTimestamp():
    """
    @Desc: Generates a timestamp formatted as "YYYYMMDDHHMMSS".
    @Result: Formatted timestamp string.
    """
    timestamp = int(time.time())
    formatted_timestamp = str(time.strftime("%Y%m%d%H%M%S", time.gmtime(timestamp)))
    return formatted_timestamp

class configLoader:
    """
    @Desc: Loads and provides access to JSON configuration data.
    @Usage: Instantiates with path to config JSON file.
    """
    def __init__(self, config_path='config.json'):
        self.base_path = os.path.realpath(os.getcwd())
        realConfigPath = os.path.join(self.base_path, config_path)
        self.config = self.load_config(realConfigPath)

    def load_config(self, realConfigPath):
        """
        @Desc: Loads JSON configuration file.
        @Result: Returns parsed JSON configuration as a dictionary.
        """
        with open(realConfigPath, 'r') as config_file:
            return json.load(config_file)

    def get_environment(self):
        """
        @Desc: Retrieves MongoDB configuration details.
        @Result: MongoDB configuration data or None if unavailable.
        """
        environment =  self.config.get("environment", None)
        environment["realPath"] = self.base_path
        return environment

    def get_defaults(self):
        """
        @Desc: Retrieves environment settings from the configuration.
        @Result: Environment configuration dictionary.
        """
        return self.config.get("defaults", {})


def buildImageProcess(DirectoryPath=None):
    result = []
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    for image_name in os.listdir(DirectoryPath):
        if os.path.splitext(image_name)[1].lower() in supported_extensions:
            result.append({"path": os.path.join(DirectoryPath, image_name), "name": image_name})
    return result


def huggingface_login():
    try:
        # Add your Hugging Face token here, or retrieve it from environment variables
        token = processControl.defaults['token'] if 'token' in processControl.defaults else ['', '']
        login(token)
        print("Successfully logged in to Hugging Face.")
    except Exception as e:
        print("Error logging into Hugging Face:", str(e))
        raise


