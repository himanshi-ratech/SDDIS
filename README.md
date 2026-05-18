# SDDIS - Smart Dairy Demand Intelligence System 
### AI-Powered Seasonal Demand Forecasting for Dairy Products
Predicts demand for 12 dairy products across 5 regions of India using ARIMA, SARIMA, Facebook Prophet & LSTM

## Overview
The SmartDairy Demand Intelligence System (SDDIS) is a full-stack Machine Learning web application that forecasts seasonal demand for 12 dairy products across 5 regions and 19 Indian states using 3 years of historical data (2021–2023).
Built with Django, powered by SARIMA + Prophet + LSTM, and featuring an AI chatbot that answers natural language questions about your dairy data — SDDIS transforms reactive dairy operations into proactive, data-driven planning.

**Raw CSV Data → Preprocessing → EDA → Feature Engineering → ML Models → Django Dashboard → AI Chatbot**

### Why SDDIS?
1. Overproduction → SpoilageAccurate demand forecasting 2 weeks ahead
2. Underproduction → ShortageFestival-aware predictions (Diwali, Holi, etc.)
3. Cold storage wasteRegion-specific demand optimization
4. 12–18% revenue lossEstimated 10–12% margin improvement

## Problem Statement
Dairy businesses in India lack a data-driven, automated forecasting system that accounts for:
i. Seasonal demand patterns (winter vs summer)
ii. Festival spikes (Diwali → +21.9% demand uplift)
iii. Regional consumption differences (North vs South vs East)
iv. Temperature-driven demand changes (r = -0.42 correlation with milk)

SDDIS solves this with 4 ML models, 60 pre-trained SARIMA models (12 products × 5 regions), and a Power BI-style interactive dashboard.

## Features
**Analytics Dashboard**
 - Power BI-style dark-mode UI with 7 interactive sections
 - Year filter (2021 / 2022 / 2023 / All) — updates charts in real time
 - Revenue ↔ Units toggle on trend charts
 - Product × Region demand heatmap

**Live Forecast Engine**
 - Select product + region + forecast horizon → instant SARIMA prediction
 - 60 pre-trained models load in < 1 second
 - Historical + forecast on one chart with 95% confidence intervals
 - Auto-generated business recommendation (increase/maintain/reduce production)

**AI Data Chatbot**
 - Online mode: Anthropic Claude API — real AI, context-aware answers
 - Offline mode: Rule-based system — 15+ categories, real dataset numbers
 - 6 quick-question buttons for instant demo
 - Answers questions like "Which product sells most?", "Festival impact kya hai?"

**EDA Insights**
 - Seasonal decomposition (Trend + Seasonality + Residual)
 - Festival impact analysis (18.3% average uplift)
 - Temperature vs demand correlation
 - Regional & state-level analysis

## ML models and results
1. **Train-Test Split**
Train: 29 months (Jan 2021 – May 2023) — 80%
Test: 7 months (Jun 2023 – Dec 2023) — 20%
Split type: Chronological (no data leakage)

Model comparison between (SARIMA, ARIMA, LSTM (2 layers), Facebook prophet)

2. **Why SARIMA ?**
SARIMA explicitly captures the 12-month seasonal cycle in dairy demand (winter peak, summer trough) through the (P=1, D=1, Q=1, m=12) seasonal component — the most important pattern in this dataset.
Business Interpretation

8.2% MAPE means: if actual demand = 40,000 units, SARIMA predicts between 36,720 and 43,280.
Industry standard buffer = 10% → 8.2% error is excellent 

3. **LSTM Architecture**
LSTM(64 units) → Dropout(0.2) → LSTM(32 units) → Dropout(0.2) → Dense(16, relu) → Dense(1)
Input window: 6 months | Batch: 8 | EarlyStopping(patience=15) | Optimizer: Adam

## Areas to Contribute

1. Adding new ML models (XGBoost, N-BEATS)
2. Improving the AI chatbot response quality
3. Adding more EDA visualizations
4. REST API development
5. Mobile app integration
