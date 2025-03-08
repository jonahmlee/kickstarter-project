# Kickstarter Project Success Prediction

This project aims to predict the success of Kickstarter campaigns using machine learning techniques. 

The project is structured into three main components: data preprocessing, model preparation, and model building.

## Project Structure

```
kickstarter-project/
├── data/
│   ├── kickstarter.xlsx          # Original dataset
│   ├── preprocessed_kickstarter.csv  # Preprocessed data
│   ├── X_train.csv               # Training features
│   ├── X_test.csv                # Testing features
│   ├── y_train.csv               # Training labels
│   └── y_test.csv                # Testing labels
├── models/                       # Saved models and transformers
└── src/
    ├── preprocessing.py          # Data preprocessing
    ├── model_preparation.py      # Feature engineering and preparation
    └── model_building.py         # Model training and evaluation
```

## Methodology

### 1. Data Preprocessing (`preprocessing.py`)
- **Data Loading**: Loads the Kickstarter dataset from Excel
- **Data Cleaning**:
  - Handles missing values in main_category using category mapping
  - Filters for only successful and failed campaigns
  - Removes duplicates
- **Feature Engineering**:
  - Creates region mapping from countries
  - Generates time-based features (project duration, creation to launch)
  - Converts goal amounts to USD using static exchange rates
  - One-hot encodes categorical variables (region, category, weekdays)
  - Log transforms skewed numerical features (usd_goal, creation_to_launch)

### 2. Model Preparation (`model_preparation.py`)
- **Feature Construction**:
  - Separates features into numerical and categorical
  - Combines encoded features with numerical features
- **Data Scaling**:
  - Applies StandardScaler to numerical features
  - Log transforms outliers in numerical features
- **Target Encoding**:
  - Uses LabelEncoder for binary classification (successful/failed)
- **Data Splitting**:
  - Splits data into training (80%) and testing (20%) sets
  - Uses stratified sampling to maintain class distribution
- **Data Persistence**:
  - Saves prepared data and transformers for model building

### 3. Model Building (`model_building.py`)
- **Feature Selection**:
  - Uses LASSO (LassoCV) for feature selection
  - Automatically identifies most important features
  - Reduces dimensionality while maintaining predictive power
    
- **Model Implementation**:
  - Neural Network (MLP):
    - Hidden layer size: 3
    - Learning rate: 0.0001
    - L2 regularization (alpha): 1
    - Max iterations: 500
  - XGBoost:
    - Learning rate: 0.01
    - Number of estimators: 1000
  - Random Forest:
    - Number of estimators: 200
    - Max depth: 10
  - Gradient Boosting:
    - Learning rate: 0.01
    - Number of estimators: 1000
  - Logistic Regression:
    - C (inverse regularization strength): 0.1
    - Max iterations: 1000
  - Decision Tree:
    - Max depth: 10
  - SVM:
    - Kernel: RBF
    - C: 1.0
    - Probability estimates enabled
- **Model Evaluation**:
  - 5-fold cross-validation
  - Metrics:
    - Accuracy
    - ROC-AUC
    - Classification Report (precision, recall, F1-score)
- **Results Visualization**:
  - Bar plot comparing model performances
  - Saves comparison results to CSV
  - Saves trained models for future use

## Key Features

1. **Feature Engineering**:
   - Time-based features
   - Categorical encoding
   - Log transformations for skewed data
   - Regional grouping

2. **Advanced Feature Selection**:
   - LASSO-based feature selection
   - Automatic feature importance ranking
   - Dimensionality reduction

3. **Comprehensive Model Testing**:
   - Multiple model types
   - Cross-validation
   - Multiple evaluation metrics
   - Model comparison visualization

4. **Reproducible Pipeline**:
   - Modular code structure
   - Saved intermediate results
   - Consistent random seeds
   - Clear data flow

## Usage

1. Run preprocessing:
```bash
python src/preprocessing.py
```

2. Run model preparation:
```bash
python src/model_preparation.py
```

3. Run model building:
```bash
python src/model_building.py
```

## Dependencies

- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- joblib
