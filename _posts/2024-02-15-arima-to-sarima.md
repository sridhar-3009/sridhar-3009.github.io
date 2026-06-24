---
layout: post
title: "From ARIMA to LSTM: Choosing the Right Model for Time-Series Forecasting"
date: 2024-02-15
description: A structured beginner-to-advanced guide to understanding ARIMA and LSTM models, how they work mathematically, and when to use each in real-world forecasting systems.
tags: [Time Series, ARIMA, LSTM, Forecasting, Python, Deep Learning]
categories: mlops
giscus_comments: false
related_posts: false
---

# From ARIMA to LSTM: Choosing the Right Model for Time-Series Forecasting

Time-series forecasting is the task of predicting future values using past observations ordered by time.

Examples include:

- Daily sales forecasting
- Stock price prediction
- Electricity demand forecasting
- Website traffic prediction

Time-series data differs from standard machine learning data because:

1. Order matters
2. Observations are time-dependent
3. Shuffling destroys information

This article builds a clear understanding of:

- How ARIMA works
- How LSTM works
- Mathematical intuition behind both
- When to use each
- Common mistakes in practice

---

# 1. Components of Time-Series Data

Most time-series can be decomposed into four parts:

1. Trend  
   Long-term increase or decrease in values.

2. Seasonality  
   Repeating patterns over fixed intervals (daily, weekly, yearly).

3. Cyclic behavior  
   Long irregular cycles such as economic fluctuations.

4. Noise  
   Random fluctuations that cannot be explained by pattern.

A forecasting model must capture these structures effectively.

---

# 2. ARIMA: Classical Statistical Approach

ARIMA stands for:

AR = AutoRegressive  
I = Integrated  
MA = Moving Average

It is written as ARIMA(p, d, q).

Where:

p = number of autoregressive terms  
d = number of differences applied  
q = number of moving average terms

---

## 2.1 AutoRegressive (AR) Component

The AR component assumes that the current value depends on previous values.

The mathematical form is:

y*t = c + φ₁ y*{t-1} + φ₂ y*{t-2} + ... + φ_p y*{t-p} + ε_t

Where:

- y_t is the value at time t
- c is a constant
- φ are coefficients
- ε_t is white noise

This is a linear relationship between past and present values.

---

## 2.2 Integrated (I) Component

Time-series models require stationarity.

A stationary series has:

- Constant mean
- Constant variance
- No long-term trend

To remove trend, we apply differencing:

First difference:

y'_t = y_t − y_{t-1}

If the series is still non-stationary, we difference again.

The number of times differencing is applied is d.

---

## 2.3 Moving Average (MA) Component

The MA part models dependence on past error terms.

Its form is:

y*t = ε_t + θ₁ ε*{t-1} + θ₂ ε*{t-2} + ... + θ_q ε*{t-q}

This smooths the residual noise.

---

## 2.4 Example in Python

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(data, order=(2, 1, 2))
model_fit = model.fit()

forecast = model_fit.forecast(steps=7)
print(forecast)
```

---

# 3. When ARIMA Works Well

ARIMA performs well when:

- Dataset is small to medium
- Patterns are mostly linear
- Seasonality is simple
- Interpretability is required
- Quick baseline is needed

Limitations:

- Cannot capture strong non-linearity
- Struggles with many input variables
- Limited ability for long-term complex dependencies

---

# 4. LSTM: Deep Learning Approach

LSTM stands for Long Short-Term Memory.

It is a special type of Recurrent Neural Network (RNN).

Unlike ARIMA, LSTM:

- Learns non-linear relationships
- Captures long-term dependencies
- Works with multiple features
- Scales better with large datasets

---

# 5. Why Regular Neural Networks Fail for Time-Series

Standard neural networks treat inputs independently.

Time-series requires sequential memory.

LSTM introduces memory using:

- Cell state
- Forget gate
- Input gate
- Output gate

These gates regulate what information is:

- Stored
- Updated
- Discarded

---

# 6. Mathematical Intuition of LSTM

At time step t:

Forget gate:

f*t = σ(W_f x_t + U_f h*{t-1} + b_f)

Input gate:

i*t = σ(W_i x_t + U_i h*{t-1} + b_i)

Candidate state:

C̃*t = tanh(W_c x_t + U_c h*{t-1} + b_c)

Updated cell state:

C*t = f_t ⊙ C*{t-1} + i_t ⊙ C̃_t

Output gate:

o*t = σ(W_o x_t + U_o h*{t-1} + b_o)

Hidden state:

h_t = o_t ⊙ tanh(C_t)

Where:

- σ is the sigmoid function
- ⊙ denotes element-wise multiplication
- x_t is input
- h_t is hidden state

This structure enables long-term memory retention.

---

# 7. LSTM Implementation in PyTorch

```python
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
```

---

# 8. Preparing Data for LSTM (Sliding Window)

LSTM requires sequence input.

```python
import numpy as np

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)
```

If seq_length = 30:

Use last 30 time steps to predict next value.

---

# 9. ARIMA vs LSTM Comparison

| Feature              | ARIMA   | LSTM            |
| -------------------- | ------- | --------------- |
| Data Size            | Small   | Medium to Large |
| Interpretability     | High    | Low             |
| Non-linearity        | No      | Yes             |
| Multivariate Support | Limited | Strong          |
| Training Speed       | Fast    | Slower          |
| Computational Cost   | Low     | Higher          |
| Long-Term Memory     | Weak    | Strong          |

---

# 10. Practical Model Selection Strategy

In real-world systems:

Step 1: Build ARIMA baseline  
Step 2: Try gradient boosting (XGBoost often strong for tabular)  
Step 3: Train LSTM  
Step 4: Compare using objective metrics

Common metrics:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

Do not assume deep learning is superior without evidence.

---

# 11. Common Beginner Mistakes

1. Random train-test split
2. Ignoring seasonality
3. Data leakage
4. No baseline comparison
5. Not normalizing data for LSTM

Correct chronological split:

```python
train = data[:'2023']
test = data['2024':]
```

Never shuffle time-series data.

---

# 12. Final Perspective

ARIMA is a structured statistical model for linear patterns.

LSTM is a flexible non-linear sequence model.

Neither is universally superior.

The correct choice depends on:

- Data size
- Pattern complexity
- Interpretability requirements
- Infrastructure constraints

Forecasting is not about using complex models.

It is about:

- Clean data
- Proper validation
- Strong baselines
- Monitoring in production

Understanding fundamentals always precedes complexity.
