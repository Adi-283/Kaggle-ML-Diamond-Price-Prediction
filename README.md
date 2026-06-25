# Kaggle-ML-Diamond-Price-Prediction

An end-to-end Machine Learning regression pipeline developed in Python using TensorFlow/Keras to predict diamond market valuations based on physical and structural attributes. 

## Project Overview
This project leverages the classic Kaggle Diamonds dataset to build, evaluate, and optimize deep learning regression models. By analyzing structural characteristics like carat weight, cut quality, color grade, clarity, and physical dimensions, the model uncovers complex non-linear relationships to output highly accurate price valuations.

## Key Features
- **Exploratory Data Analysis (EDA):** Visualized distribution of diamond prices, detected multi-collinearity among dimensional attributes (x, y, z), and analyzed categorical feature impacts (Cut, Color, Clarity) using Seaborn and Matplotlib.
- **Robust Data Preprocessing:** - Implemented one-hot encoding for ordinal and nominal categorical features.
  - Applied feature scaling (StandardScaler/MinMaxScaler) to normalize continuous features for stable neural network convergence.
  - Handled outliers and anomalous data points (e.g., zero-value dimensions).
- **Model Architecture:** Built a multi-layer Deep Neural Network (DNN) using TensorFlow/Keras sequential API featuring Dense layers, Batch Normalization, and Dropout regularizations to prevent overfitting.
- **Hyperparameter Tuning:** Systematically optimized learning rates, batch sizes, layer depths, and activation functions (ReLU, Elu) to minimize loss.
- **Evaluation Metrics:** Monitored performance using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and $R^2$ score to ensure robust predictive consistency across various price brackets.

## Tech Stack
- **Language:** Python
- **Frameworks:** TensorFlow, Keras, Flask
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Pickle
