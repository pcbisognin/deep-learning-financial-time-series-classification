# Deep Learning for Financial Time-Series Classification

This repository explores the use of **deep learning architectures** to classify **buy/sell signals** from financial market time-series data. The project focuses on comparing how different neural network designs capture temporal and structural patterns in stock price data.

Rather than optimizing a single model, the goal is to **analyze and compare architectures** commonly used in time-series modeling, highlighting their strengths and limitations when applied to financial signals.

---

## Problem Framing

Financial markets generate sequential data with strong temporal dependencies, noise, and non-stationarity. Accurately identifying buy/sell signals from historical price information is a challenging classification problem that requires models capable of learning both short-term patterns and long-range dependencies.

In this project, historical stock data is framed as a **supervised classification task**, where the model predicts discrete trading signals based on past observations.

---

## Approach Overview

Each notebook applies a different deep learning architecture to the same general problem:

- Time-series data is prepared using sliding windows
- Models are trained to classify buy/sell signals
- Performance is evaluated and compared across architectures
- Emphasis is placed on architectural behavior rather than hyperparameter tuning

The experiments are implemented in **Python** using common deep learning libraries.

---

## Architectures Explored


### 📘 `VALE_LSTM.ipynb`
**Long Short-Term Memory (LSTM) and 1D CNN hybrid** network applied to VALE stock data.  
Serves as a baseline recurrent architecture to capture long-term temporal dependencies in financial time series.

---

### 📘 `BBA_Conv1D.ipynb`
**1D Convolutional Neural Network (Conv1D)** applied to BBA stock data.  
Focuses on detecting local temporal patterns and short-term price movements using convolutional filters.

---

### 📘 `CSNA_Conv2D.ipynb`
**2D Convolutional Neural Network (Conv2D)** applied to CSNA stock data.  
Treats transformed time-series inputs as 2D representations, enabling spatial-style feature extraction across time and indicators.

---

### 📘 `PETRO_TCN.ipynb`
**Temporal Convolutional Network (TCN)** applied to PETRO stock data.  
Uses dilated causal convolutions to model long-range dependencies while maintaining temporal causality.

---

## Evaluation Strategy

- Models are evaluated using classification metrics appropriate for trading signal prediction
- Emphasis is placed on **comparative performance and learning behavior**
- Results are analyzed qualitatively and quantitatively to understand architectural trade-offs

This repository is intended as an **experimental and exploratory study**, not a production trading system.

---
