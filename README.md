# README.md

# Similarity Analysis Pipeline

This repository contains scripts for analyzing and visualizing similarity matrices from generative models. The pipeline includes:

- Loading and validating similarity matrices
- Transforming matrices into per-image summary statistics
- Running non-parametric statistical tests
- Generating distribution plots and heatmaps

## Usage

Run the full pipeline with:

```bash
python run_all.py <matrix_dir> <out_dir>
```

- `<matrix_dir>`: Directory containing similarity matrix files
- `<out_dir>`: Directory where outputs will be saved

## Main Scripts

- `run_all.py`: Entry point for the full analysis pipeline
- `transform_samples.py`: Converts matrices to per-image summaries
- `run_stats.py`: Performs statistical tests
- `visualization.py`: Creates plots and heatmaps

## Requirements

- Python 3.8+
- numpy
- pandas
- scipy
- statsmodels
- matplotlib

Install dependencies with:

```bash
pip install numpy pandas scipy statsmodels matplotlib
```

## File Structure

- `run_all.py`
- `transform_samples.py`
- `run_stats.py`
- `visualization.py`
- `load_validate.py`

---
