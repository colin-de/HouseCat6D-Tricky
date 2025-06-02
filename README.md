# HouseCat6D-Tricky Evaluation

This repository provides tools for evaluating 6D object pose estimation results on the HouseCat6D-Tricky dataset, with a focus on challenging transparent and reflective object categories such as cutlery and glass. The evaluation pipeline includes ground truth and prediction result handling, metric computation (mAP, IoU, pose error), and result logging.

## Features

- **Evaluation Metrics**: Computes 3D IoU and pose-based mAP for object categories, with detailed per-category and overall statistics.

## Directory Structure

```
.
├── evaluate_housecat_tricky.py   # Main evaluation and utility script
├── obj_models_small_size_final/  # Directory for object model .pkl files (e.g., objects.pkl)
├── results/                      # Output directory for evaluation results (.pkl files) we prepare example results pkl file for pose format
├── tricky/                       # directory for tricky test scenes and images
├── train/                        # (Optional) training data directory
├── val/                          # (Optional) validation data directory
└── test_logger.log               # Log file for evaluation runs
```

## Requirements

- Python 3.9
- PyTorch 1.12.1+cu116
- OpenCV
- NumPy
- tqdm
- torchvision

## Data Preparation

1. **Dataset**: Place the HouseCat6D-Tricky dataset under the `tricky/` directory. The expected structure is:
    ```
    tricky/
      test_scene1/
      test_scene2/
      test_scene3/
    ```

2. **Results Directory**: The evaluation will output `.pkl` files into the `results/` directory, organized by scene.

## Usage

### 1. Evaluate Results

Run the main evaluation script:

```bash
python evaluate_housecat_tricky.py --data_dir /path/to/HouseCat6D-Tricky --result_dir results
```

- `--data_dir`: Path to the root of the HouseCat6D-Tricky dataset.
- `--result_dir`: Directory where result `.pkl` files will be saved and/or loaded.

### 2. What Happens

- It processes each test image, optionally augmenting prediction `.pkl` files with ground truth.
- It computes evaluation metrics (IoU, pose error) for each object and scene.
- Results are printed to the console and saved to `test_logger.log`.

### 3. Output

- Per-image and per-category evaluation results are saved as `.pkl` files in the `results/` directory.
- Summary statistics are printed and logged.


## Notes

- Only the "cutlery" and "glass" categories are evaluated by default, as per the challenge focus.
