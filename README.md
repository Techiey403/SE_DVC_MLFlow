# DVC + MLflow Machine Learning Project

## Overview

This project demonstrates a basic MLOps workflow using **DVC (Data Version Control)** and **MLflow**.

The project uses the Iris dataset to train a Machine Learning model and demonstrates:

- Dataset versioning using DVC
- Source-code version control using Git
- Experiment tracking using MLflow
- Logging model parameters
- Logging evaluation metrics
- Saving the trained model using MLflow
- Managing different versions of the dataset

---

## Technologies Used

- Python
- Scikit-learn
- Pandas
- Git
- GitHub
- DVC
- MLflow

---

## Project Structure

```text
DVC & MLflow/
│
├── data/
│   ├── iris.csv
│   └── iris.csv.dvc
│
├── .dvc/
├── .dvcignore
├── .gitignore
│
├── create_dataset.py
├── train.py
├── requirements.txt
│
└── README.md
