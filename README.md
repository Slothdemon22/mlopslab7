# House Price Prediction Project

This project implements an end-to-end machine learning workflow for house price prediction using the Pakistan House Price Dataset.

## Project Structure

```
├── src/
│   └── train.py              # Training script
├── data/
│   └── Entities.csv          # Dataset
├── models/                   # Model artifacts (generated)
├── metrics/                  # Training metrics (generated)
├── templates/                # Flask templates
├── static/                   # Static files
├── params.yaml              # Configuration parameters
├── dvc.yaml                 # DVC pipeline definition
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Initialize Git and DVC:
```bash
git init
dvc init
```

3. Add DVC remote (optional):
```bash
dvc remote add -d storage /path/to/storage
```

## Usage

### Training the Model

Run the DVC pipeline:
```bash
dvc repro
```

This will:
- Load and preprocess the data
- Train a RandomForestRegressor model
- Save model artifacts to `models/`
- Generate metrics in `metrics/metrics.json`

### Running the Flask App

```bash
python housepk_app.py
```

The app will be available at `http://localhost:5000`

## Configuration

All training parameters are defined in `params.yaml`:
- Model hyperparameters
- Data preprocessing options
- Feature selection
- Output paths

## DVC Workflow

1. **Data Versioning**: The dataset is tracked with DVC
2. **Pipeline**: Training pipeline defined in `dvc.yaml`
3. **Reproducibility**: All experiments are reproducible using `dvc repro`
4. **Metrics Tracking**: Training metrics are automatically tracked

## Model Performance

The trained model achieves:
- R² Score: ~0.82
- RMSE: ~515,000 PKR
- MAE: ~300,000 PKR
