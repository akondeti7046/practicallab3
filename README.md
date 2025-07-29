# Dogs vs Cats - CNN and VGG16 Fine-Tuning

## Overview

This project classifies images of dogs and cats using:
- A custom-built Vanilla CNN model
- A fine-tuned VGG16 pre-trained model

The dataset used is a 5000-image subset of the Kaggle Dogs vs Cats dataset. Images were organized into training, validation, and test directories before model training.

## Folder Structure
archive-3/ # Original dataset
└── train/
└── cat/
└── dog/
data/
└── kaggle_dogs_vs_cats_small/
└── train/
└── cat/
└── dog/
└── validation/
└── cat/
└── dog/
└── test/
└── cat/
└── dog/



dogs_vs_cats_cnn_vgg16.ipynb # Main notebook with code and analysis
vanilla_cnn_best.keras # Best saved CNN model
vgg16_finetuned_best.keras # Best saved fine-tuned VGG16 model
requirements.txt # Python dependencies
README.md # Project description


## How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/akondeti7046/practicallab3.git
   cd practicallab3/practicallab3/


2. Create and activate a virtual environment:
python3 -m venv venv
source venv/bin/activate

3. Install dependencies:
pip install -r requirements.txt

4. Launch Jupyter Notebook:
jupyter notebook

5. Open dogs_vs_cats_cnn_vgg16.ipynb and run all cells.



Prerequisites:
Python 3.9 or above
TensorFlow 2.x
Keras
NumPy
Matplotlib
Seaborn
scikit-learn
Jupyter Notebook





You can install all dependencies using the requirements.txt file.


Results Summary-
Vanilla CNN Model:
Best validation accuracy: ~70.2%
Signs of overfitting after epoch 5
Fine-Tuned VGG16 Model:
Best validation accuracy: ~90%
Improved generalization and lower validation loss
Both models were evaluated using confusion matrix and classification report.



student name- Adhitya Kondeti
Conestoga College – Foundations of Machine Learning Frameworks
