---
layout: page
title: DemandCast
description: Large-scale demand forecasting pipeline — LSTM with Temporal Attention achieving 22% lower MAPE across 50K+ products
img: assets/img/projects/demandcast.png
importance: 1
category: Machine Learning
github: https://github.com/sridhar-3009
---

End-to-end demand forecasting system for predicting daily SKU-level demand across 50K+ products,
comparing classical and deep learning approaches.

**What I built:**
- Benchmarked ARIMA, Prophet, XGBoost, and LSTM — LSTM with Temporal Attention won with **22% lower MAPE** vs ARIMA baseline
- Real-time inference API with FastAPI + Redis caching for sub-100ms predictions
- Kafka-based streaming pipeline to ingest live sales data
- Model monitoring layer that detects data drift and triggers automated retraining

**Tech stack:** Python · PyTorch · XGBoost · FastAPI · Kafka · Redis · PostgreSQL

| Metric | Value |
|--------|-------|
| MAPE improvement | 22% |
| Products modeled | 50K+ |
| Status | Ongoing |
