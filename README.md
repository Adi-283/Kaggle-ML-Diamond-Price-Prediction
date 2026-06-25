# Kaggle-ML-Diamond-Price-Prediction

An end-to-end Machine Learning regression pipeline developed in Python using TensorFlow/Keras to predict diamond market valuations based on physical and structural attributes. 

## Project Overview
This project leverages the classic Kaggle Diamonds dataset to build, evaluate, and optimize deep learning regression models. By analyzing structural characteristics like carat weight, cut quality, color grade, clarity, and physical dimensions, the model uncovers complex non-linear relationships to output highly accurate price valuations.

## Key Features
- **Exploratory Data Analysis (EDA):** Visualized distribution of diamond prices, detected multi-collinearity among dimensional attributes (x, y, z), and analyzed categorical feature impacts using Seaborn and Matplotlib.
- **Robust Data Preprocessing:** Implemented feature scaling and categorical encoding via Scikit-Learn to normalize data for stable neural network convergence.
- **Model Architecture & Tuning:** Built a multi-layer Deep Neural Network (DNN) using TensorFlow/Keras, optimizing layers and learning rates to minimize regression loss.
- **Interactive Front-End GUI:** Developed an intuitive user interface that allows users to input diamond attributes (carat, cut, color, clarity, dimensions) and receive instant, real-time price predictions from the trained model.
- **Evaluation Metrics:** Monitored performance using MAE, RMSE, and $R^2$ score to ensure robust predictive consistency across various price brackets.

## Tech Stack
- **Language:** Python
- **Frameworks:** TensorFlow, Keras, Flask, Heroku
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Pickle, Streamlit
