# Fire and Smoke Training Configuration

This document provides details about the datasets and parameters used for training the fire and smoke detectiontion model using YOLOv8 model and the Application created for seeing the model in use. 

## Table of Contents
<details>

- [Datasets](#datasets)
- [Running the Model](#running-the-model)
- [Parameters](#parameters)
- [Project Naming Convention](#project-naming-convention)
- [Model Dependencies](#model-dependencies)
- [Application Overview](#application-overview)
- [Application Dependencies](#application-dependencies)
- [Running the  Application ](#running-the-application)
- [Setting Model for Application](#setting-model-for-application)
- [References](#references)

</details>

## Datasets

- **DataSet 1** : [Fire-detection-v3-6](https://universe.roboflow.com/touatimed2/fire-detection-v3-or0i1)
- **DataSet 2** : [Fire-Detection-1](https://universe.roboflow.com/situational-awarnessinnovsense/fire-detection-ypseh)
- **DataSet 3** : [Fire-and-Smoke-Detection-1](https://universe.roboflow.com/adib-ga0ow/fire-and-smoke-detection-jngig)

These datasets are taken from the Ultralytics website. Download code in **.ipynb** file

## Running the Model

```bash
!yolo task=detect mode=train model=yolov8n.pt data=Fire-detection-v3-6/data.yaml epochs=100 freeze=10 imgsz=640 plots=true verbose=True save_period=1 project=D1E100F10
```

## Parameters
<details>

| Parameter     | Value                                       | Description                                                                 |
|---------------|---------------------------------------------|-----------------------------------------------------------------------------|
| `task`        | `detect`                                    | Specifies the type of task, which in this case is object detection.         |
| `mode`        | `train`                                     | Indicates that the mode is set to training.                                 |
| `model`       | `yolov8n.pt`                                | Specifies the pre-trained model to be used, which is YOLOv8n.               |
| `data`        | `Fire-detection-v3-6/data.yaml`             | Path to the dataset configuration YAML file.                                |
| `epochs`      | `100`                                       | Number of training epochs.                                                  |
| `freeze`      | `10`                                        | Number of layers to freeze during training.                                 |
| `imgsz`       | `640`                                       | Size of the input images.                                                   |
| `plots`       | `true`                                      | Enables the generation of plots during training.                            |
| `verbose`     | `True`                                      | Enables verbose output during training for detailed logs.                   |
| `save_period` | `1`                                         | Saves the model weights every epoch.                                        |
| `batch`       | `16`                                        | Number of images in each batch.                                             |
| `project`     | `D1E100F10`                                 | Name of the project directory where the training results will be saved.     |

</details>


## Project Naming Convention
In the Results Folder $D_i E _n$ Indicates the $i$-th Dataset trained for $n$-epochs

And later adding freeze, named the folder $D_i E _n F_m$ where the m denoted the number of layers being 

## Model Dependencies
<details>
Ensure you have the following dependencies installed:

- Python 3.7 or later
- PyTorch
- Ultralytics YOLOv8 package
- Numpy
- OpenCV
- Matplotlib
- Pillow
- PyYAML
- TQDM
- SciPy
- TensorBoard
- Seaborn
- Jupyter
- CUDA (if using GPU acceleration)

### Installing Dependencies

#### **It is recommended to use a virtual environment** 

To create a virtual environment and install the required dependencies, follow these steps


```bash
# Create a virtual environment
python -m venv yolov8-env

# Activate the virtual environment
# On Windows
yolov8-env\Scripts\activate
# On macOS/Linux
source yolov8-env/bin/activate
# Install Python 3.7 or later
python --version
# Upgrade pip
pip install --upgrade pip
```
<span style="color:red;">**If using CUDA for GPU Acceleration you MUST INSTALL CUDA BEFORE THE OTHER DEPENDENCIES**</span>


#### Install CUDA 
#### Follow the installation instructions from : https://developer.nvidia.com/cuda-downloads

#### Rest of the Dependencies for Model
```bash

# Install PyTorch (Choose the right command from https://pytorch.org/get-started/locally/ based on your system and CUDA version)
# Example for CUDA 11.7
pip install torch torchvision torchaudio

# Install YOLOv8 from Ultralytics
pip install ultralytics

# Install other dependencies
pip install numpy opencv-python matplotlib pillow pyyaml tqdm scipy tensorboard seaborn

# Install Jupyter
pip install jupyter
```
</details>

## Application Overview

This project also includes a PyQt-based application that allows users to interact with the fire and smoke detection model. The application provides the following features:
- Upload and process images for fire and smoke detection.
- Upload and process videos for fire and smoke detection.
- Open and process live webcam feed for fire and smoke detection.

## Application Dependencies 

To install the required dependencies for the PyQt application, run:

```bash
pip install pyqt5
pip install opencv-python
```
## Running the  Application

Use this command in terminal

```bash
python app.py
```

## Setting Model for Application
Currently `yoloD1E100.pt` is being used as the model in the app. 

After training, pull the `best.pt` model from your project and replace the `model=YOLO("yoloD1E100.pt")` in 3 parts of the code with the relative path to `your_model.pt`.

## References 

- [YoloV8 Docs](https://docs.ultralytics.com/usage/python/)
- [K-Fold Cross Validation with Ultralytics](https://docs.ultralytics.com/guides/kfold-cross-validation/)
- [YoloV8 Model](https://huggingface.co/Ultralytics/YOLOv8/tree/main)
- [Transfer Learning with Frozen Layers](https://docs.ultralytics.com/yolov5/tutorials/transfer_learning_with_frozen_layers/#project-status)


