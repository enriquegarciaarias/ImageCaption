# ImageCaption
Image Captioning stands for a Image Captioning project based on the use of AI.  

# Preparation
The project has been developed and tested in a Unix Ubuntu environment Ubuntu 22.04.5 LTS.
If you're using a Windows machine, you will need to set up a virtualized environment for development. We recommend using Hyper-V to create an Ubuntu virtual machine (VM).

# Set Up In Ubuntu environment
Once you have your Ubuntu environment available you can proceed with a standard Unix installation.
Install system requirements:
```aiignore
sudo apt update && sudo apt upgrade -y
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.10
sudo apt-get install python3.10-dev
sudo apt install python3.10-venv
sudo apt-get install build-essential
sudo apt-get install git
sudo apt install python3-tk
```


# PyCharm IDE
PyCharm is the chosen IDE to debug and code with the project.
Download and install PyCharm in your system. Follow the instructions given at:
https://www.jetbrains.com/help/pycharm/installation-guide.html

# Project clone
Go to your selected working directory where you will clone the project.

```aiignore
git clone https://github.com/enriquegarciaarias/ImageCaption.git
```

Go to the cloned project directory
```aiignore
cd ImageCaption
```


Create a completely clean Python virtual environment (named deepmountain_env):
```aiignore
python3.10 -m venv imagecaption_env
source imagecaption_env/bin/activate
```


⚠️ Warning:
If you change the name of the virtual environment, you must add the new name to the .gitignore file to ensure that the virtual environment directory is not tracked by version control.

The Python version used in this project is 3.10.12 and the pip version is 22.0.2 You can verify this with the commands:
```aiignore
python --version
pip --version
```


# PyCharm configuration
PyCharm is the chosen IDE to debug and code with the project
You first need to configure the Python interpreter in PyCharm:
Navigate to: Python Interpreter -> Add Interpreter -> Virtualenv Environment -> Existing
Select the virtual environment created earlier.

Libraries and dependencies
To install the necessary dependencies, navigate to the root directory of the project (where 'requirements.txt' is stored) and execute:
```aiignore
pip install -r requirements.txt
```


# Use the project

Now you are ready to use the project:
Copy the images you want to process to the directory process/input:


Access the help commands:
```
python3 main.py -h
```
Where the option MODEL stands for extracting features, distribute the images on subdirectories inside the process/output directory and creating the model.
The option APPLY stands for using the model to new images in the process/input directory