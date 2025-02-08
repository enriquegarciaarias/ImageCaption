import json

import time
import os
from os.path import isdir


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


def convert_docx_to_txt(input_path, output_path=None):
    from docx import Document
    """
    Convierte un archivo DOCX a TXT extrayendo todo el texto.

    :param input_path: Ruta del archivo .docx de entrada.
    :param output_path: Ruta del archivo .txt de salida.
    """

    doc = Document(input_path)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])  # Extrae texto con saltos de línea

    if output_path:
        with open(output_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(text)


    return text