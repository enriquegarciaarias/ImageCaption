from sources.common import logger, logProc, processControl, log_

import requests
import zipfile
import os

def importFlickr():
    # URL of the Flickr Image Caption dataset
    dataset_url = "https://www.kaggle.com/adityajn105/flickr8k"  # Replace with the actual URL


    dataset_path = os.path.join(processControl.env['data'], "flickr_image_caption_dataset.zip")

    # Download the dataset
    response = requests.get(dataset_url, stream=True)
    with open(dataset_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    # Extract the dataset
    with zipfile.ZipFile(dataset_path, "r") as zip_ref:
        zip_ref.extractall("flickr_image_caption_dataset")

    # Clean up the zip file
    os.remove(dataset_path)

    print("Dataset downloaded and extracted successfully.")


"""
Make sure to replace "https://example.com/flickr_image_caption_dataset.zip" with the actual URL of the Flickr Image Caption dataset.
This script assumes the dataset is provided as a zip file.
If the dataset format or location changes, you may need to adjust the script accordingly.
"""

