# STOCK-INDEX-USING-TPA-LSTM-MODEL

-----

# 📈 Stock Index Movement Prediction using LSTM Model

This repository contains a Python project utilizing a **Long Short-Term Memory (LSTM)** neural network to reveal and predict patterns in stock index movements. The solution includes a basic **Graphical User Interface (GUI)** built with Tkinter for execution and visualization, as well as a more detailed Jupyter Notebook (`Source.py` adapted) for step-by-step model development.

## 🚀 Overview

The project focuses on time series forecasting of stock prices, using historical price data (specifically the **Close** price) to train an LSTM model.

  * **Model:** A Sequential LSTM model is employed, which is highly effective for capturing time dependencies in sequence data like stock prices.
  * **Data Preparation:** Data is cleaned, scaled using **MinMaxScaler**, and structured into time steps (sequences of past data points) for the LSTM's input.
  * **Prediction:** The model is used to predict future stock prices, and results are visualized against the actual prices.

## ⚙️ Prerequisites

Before running the project, ensure you have **Python 3.x** installed, along with the following libraries:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras yfinance pillow tkinter
```

Specifically, you'll need:

  * `numpy`
  * `pandas`
  * `matplotlib`
  * `scikit-learn` (for `MinMaxScaler`)
  * `tensorflow` (and `keras`)
  * `yfinance` (used in `Source.py` to fetch AAPL data)
  * `Pillow` (PIL) and `tkinter` (for the GUI)

## 📂 Repository Structure

```
STOCK INDEX USING TPA-LSTM MODEL/
├── main.py             # Tkinter GUI application for prediction (Core file)
├── Source.py           # Detailed step-by-step analysis and model building (Jupyter Notebook adapted)
├── Dataset/            # (Placeholder for your CSV dataset)
├── model/
│   └── keras_model.h5  # Saved trained model (will be created after first run)
└── background.jpg      # GUI background image (requires this file)
```

## 💻 Running the Application (`main.py`)

The `main.py` file provides a user-friendly interface to load data, preprocess it, train/load the model, and make predictions.

1.  **Launch the GUI:**
    ```bash
    python main.py
    ```
2.  **Steps in the GUI:**
      * **Upload Dataset:** Click to load your stock data CSV file (expected to be in the `Dataset` folder). The file should contain columns like `Index Name`, `Date`, `Open`, `High`, `Low`, and `Close`.
      * **Pre Processing:** Cleans the data (dropping irrelevant columns) and calculates the **100-day Moving Average (MA100)**. A plot of the `Close` price and `MA100` is displayed.
      * **Splitting:** Splits the data into training (70%) and testing sets and scales the training data.
      * **L S T M:** Builds and compiles the LSTM model. If a saved model (`keras_model.h5`) exists, it's loaded; otherwise, the model is trained for 50 epochs and saved.
      * **Predict:** Generates predictions for the test set and for the next 60 days, then displays a visualization comparing the **Actual Price** vs. **Predicted Price**.

## 🧠 Model Architecture (as defined in `main.py`)

The model is a Sequential Keras model designed for stock price forecasting:

| Layer | Units | Activation | Function |
| :--- | :--- | :--- | :--- |
| **LSTM** | 50 | N/A | `return_sequences=True`, `input_shape=(X_train.shape[1], 1)` |
| **Dropout** | N/A | N/A | Rate: 0.2 (Regularization) |
| **LSTM** | 50 | N/A | `return_sequences=False` |
| **Dropout** | N/A | N/A | Rate: 0.2 (Regularization) |
| **Dense** | 25 | N/A | Fully connected layer |
| **Dense** | 1 | N/A | Output layer (The predicted price) |

  * **Optimizer:** `adam`
  * **Loss Function:** `mean_squared_error` (Standard for regression tasks)

## 📊 Detailed Analysis (`Source.py`)

The `Source.py` file contains the detailed implementation, including:

  * Fetching **AAPL stock data** using `yfinance`.
  * Calculating and visualizing **100-day** and **200-day Moving Averages**. \* A different, slightly deeper **LSTM architecture** with multiple layers:
      * 4 LSTM layers (50, 60, 80, 120 units) with `relu` activation.
      * Dropout layers (0.2, 0.3, 0.4, 0.5) for regularization.
      * A final Dense layer.
  * Detailed steps for preparing the test data by appending the preceding 100 days of training data to create the input sequences.
  * Evaluation using **Root Mean Squared Error (RMSE)**.
  * Visualization of the **Original Price** vs. **Predicted Price**.

This script serves as a robust foundation and a deeper look into the model development process.
