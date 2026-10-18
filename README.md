# Rossmann Store Sales Forecasting
Live demo: <your Streamlit URL>
## Business problem
Rossmann operates 1,100+ drugstores. Accurate sales forecasts for the next 6 weeks
help with inventory planning and staff scheduling.
## Results
| Model | RMSPE | MAPE % | MAE |
|---|---|---|---|
| Store-weekday average | ... | ... | ... |
| LightGBM | ... | ... | ... |
(screenshot or GIF of the app)
## Approach
1. Cleaning: closed days removed, missing store data filled
2. EDA: promo lift, weekly and yearly seasonality, store types
3. Features: calendar, promo, competition, lags of 49+ days, store statistics
4. Validation: time-based, last 6 weeks held out, no leakage
5. Models: baselines, LightGBM, XGBoost, Prophet
6. Interpretation: SHAP and error analysis
## Key insights and recommendations
- ...
## Limitations and next steps
- Early stopping used the validation set
- Higher errors for ... (from your error analysis)
- Next: external data (weather, events), hierarchical forecasting
## Run it locally
pip install -r requirements-dev.txt
(download data from Kaggle into data/raw/)
run notebooks 01 to 06 in order, then: streamlit run app/app.py