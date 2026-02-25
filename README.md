# Introduction to Deep Learning

A collection of projects covering core deep learning topics, from classical machine learning algorithms to modern neural network architectures.

## Projects

| # | Directory | Topic | Key Concepts |
|---|-----------|-------|--------------|
| 1 | `1.knn/` | K-Nearest Neighbors | KNN classification, distance metrics |
| 2 | `2.Linear_Logistic_Reg/` | Linear & Logistic Regression | Linear models, gradient descent, decision boundaries |
| 3 | `3.cnn/` | Convolutional Neural Networks with PyTorch | PyTorch basics, CNN architectures, Module API, Sequential API, ResNet-10, CIFAR-100 |
| 4 | `4.nn_numpy/` | Neural Networks from Scratch (NumPy) | Linear layers, ReLU, Softmax, Cross-Entropy loss, backpropagation |
| 5 | `5.nn_classification/` | Neural Network Classification | Training & testing on CIFAR-10, hyperparameter tuning |
| 6 | `6.fcn/` | Fully Convolutional Networks | Semantic segmentation, FCN-32s, FCN-8s, transfer learning |

## Setup

```bash
# Create a conda environment (optional)
conda create -n intro2dl python=3.9.5
conda activate intro2dl

# Install dependencies (per project)
pip install -r <project_dir>/requirements.txt
```

Some projects require downloading datasets before running:

```bash
# For projects with get_datasets.py
python get_datasets.py

# For project 6
bash get_dataset.sh
```

## Usage

Each project is self-contained in its own directory with Jupyter notebooks as the main entry point. Open the `.ipynb` files and follow the instructions inside.
