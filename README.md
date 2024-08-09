# Headlight Classifier

The headlight classifier is designed to identify whether the headlights in given images are turned "on" or "off". It uses pre-trained models with custom modifications to achieve accurate predictions. The code is implemented using PyTorch and torchvision.

## Table of Contents 
- [Usage](#usage) 
- [Project Structure](#project-structure) 
- [Model Architectures](#model-architectures) 
- [Dataset Preparation](#dataset-preparation) 
- [Training](#training) 
- [Evaluation](#evaluation) 
- [Results](#results) 

## Usage

To run the inference or evaluation, you'll need to have pre-trained models saved in the `checkpoints` directory. You can also train your models using the provided `main.ipynb`.

## Project Structure

- `inference.py`: Script for running inference on a set of images. 
- `models.py`: Contains the definitions of various headlight classifier models. 
- `main.ipynb`: Jupyter notebook for training models and testing them. 
- `dataset/`: Directory containing training, validation, and test datasets. 
- `checkpoints/`: Directory to save and load model checkpoints. 
- `requirements.txt`: Lists all dependencies required for the project.

## Model Architectures

Three different models are implemented in this project: 
- **EfficientNet**: A state-of-the-art CNN architecture known for its performance and efficiency. 
- **MobileNet**: A lightweight model designed for mobile and edge devices. 
- **ResNet**: A deep residual network useful for high-accuracy tasks.


## Dataset Preparation
   Ensure you have the dataset organized as follows:

   ```
   dataset/
   ├── train/
   │   ├── off/
   │   └── on/
   ├── val/
   │   ├── off/
   │   └── on/
   └── test/
       ├── off/
       └── on/
   ```

   Each subfolder (`on`, `off`) should contain the respective images.


## Training

1. Open `main.ipynb` to train the models. This notebook includes data augmentation, model training, and loss visualization. 
2. Set the desired configuration parameters like learning rate, epochs, and transformations. 
3. Execute the training loop to train the models. Model checkpoints will be saved automatically.

## Evaluation

Evaluation is done using the `inference.py` script or within the notebook. The model's accuracy on the validation and test datasets is logged and can be reviewed for performance analysis.

## Results

The results, including validation and test accuracy, are logged in `accuracy_results.md`.