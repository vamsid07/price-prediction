# Transformer-Based Product Price Prediction from Catalog Text

A deep learning solution for predicting product prices from catalog text descriptions using fine-tuned transformer models with ensemble learning.

## Project Overview

This project implements a regression model that predicts product prices based on textual catalog content. The solution leverages pre-trained sentence transformers and employs advanced techniques including weighted sampling, ensemble learning, and custom loss functions to achieve robust price predictions.

## Key Features

- **Transformer-Based Architecture**: Utilizes `sentence-transformers/all-MiniLM-L6-v2` for text encoding
- **Custom Regression Head**: Multi-layer neural network with dropout regularization
- **SMAPE Optimization**: Implements Symmetric Mean Absolute Percentage Error as the primary loss function
- **Ensemble Learning**: Trains multiple models with different random seeds and combines predictions
- **Weighted Sampling**: Addresses class imbalance across different price ranges
- **Comprehensive Evaluation**: Includes multiple metrics (SMAPE, MAE, RMSE, MAPE) and visualization tools

## Technical Architecture

### Model Components

1. **Text Encoder**: Pre-trained `all-MiniLM-L6-v2` transformer (384-dimensional embeddings)
2. **Regression Head**: 
   - Linear(384 → 256) + ReLU + Dropout(0.2)
   - Linear(256 → 128) + ReLU + Dropout(0.2)
   - Linear(128 → 1)
3. **Loss Function**: Custom SMAPE loss with epsilon smoothing
4. **Optimizer**: AdamW with learning rate 2e-5 and weight decay 0.01
5. **Scheduler**: Linear warmup with 500 warmup steps

### Data Processing Pipeline

- Text cleaning and normalization
- Log transformation of target prices
- Tokenization with max length 128 tokens
- Weighted random sampling for balanced training
- Train-validation split (90-10)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd price-prediction

# Install required packages
pip install pandas numpy matplotlib seaborn
pip install torch transformers scikit-learn
pip install tqdm
```

## Usage

Open and run the Jupyter notebook `price_prediction.ipynb` in sequence. The notebook contains:

1. **Data Loading and Exploration**: Initial data analysis and visualization
2. **Data Preprocessing**: Text cleaning and feature engineering
3. **Model Training**: Single model training with early stopping
4. **Evaluation**: Performance metrics and visualization on validation set
5. **Ensemble Training**: Multiple model training with different seeds
6. **Prediction Generation**: Final predictions on test set

### Key Outputs

The notebook generates:
- `best_model.pt`: Best performing single model
- `ensemble_model_*.pt`: Individual ensemble models
- `test_out.csv`: Single model predictions
- `test_out_ensemble_mean.csv`: Mean ensemble predictions
- `test_out_ensemble_median.csv`: Median ensemble predictions
- `test_out_ensemble_weighted.csv`: Weighted ensemble predictions (recommended)
- `sample_test_comparison.csv`: Detailed evaluation metrics

## Results

### Performance Metrics

- **Primary Metric**: SMAPE (Symmetric Mean Absolute Percentage Error)
- **Secondary Metrics**: MAE, RMSE, MAPE
- **Validation Strategy**: 10% holdout with early stopping

### Model Performance

The ensemble approach provides improved robustness and generalization compared to single models. Weighted ensemble predictions (based on validation SMAPE) typically achieve the best performance.

## Project Structure

```
price-prediction/
├── price_prediction.ipynb       # Main Jupyter notebook
└── README.md                    # Project documentation
```

### Generated Files (after running notebook)

```
├── best_model.pt                # Best single model weights
├── ensemble_model_*.pt          # Individual ensemble model weights
├── test_out.csv                 # Single model predictions
├── test_out_ensemble_*.csv      # Ensemble predictions
└── sample_test_comparison.csv   # Detailed evaluation results
```

## Key Techniques

### 1. Log Transformation
Applies log(price + 1) transformation to handle skewed price distributions and improve model stability.

### 2. Weighted Sampling
Creates price bins and assigns sampling weights inversely proportional to bin frequency, ensuring balanced representation of all price ranges.

### 3. Ensemble Learning
Trains multiple models with different:
- Random seeds for initialization
- Train-validation splits
- Combines predictions using weighted average based on validation performance

### 4. Early Stopping
Monitors validation SMAPE with patience of 3 epochs to prevent overfitting.

### 5. Gradient Clipping
Applies gradient norm clipping (max norm = 1.0) for training stability.

## Dependencies

- Python 3.7+
- PyTorch 1.9+
- Transformers 4.0+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- tqdm

## Future Improvements

- Experiment with larger transformer models (BERT, RoBERTa)
- Implement cross-validation for more robust evaluation
- Add feature engineering (text length, keyword extraction)
- Explore multi-task learning with category prediction
- Implement model distillation for faster inference

## License

This project is available for educational and research purposes.

## Contact

For questions or collaboration opportunities, please reach out through GitHub issues or LinkedIn.

---

**Note**: This project was developed as part of a machine learning competition/assignment focusing on price prediction from textual data.
