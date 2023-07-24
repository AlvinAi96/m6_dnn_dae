### 1. Introduction
M competition is a well-known competition focused on time series forecasting, held once a year. This year's M6 is a financial forecasting competition that primarily focuses on predicting stock prices (returns) and risks, providing new perspectives on the Efficient Market Hypothesis (EMH).<br>
Our Team: OPPO XZ Lab<br>
Final Global Rank: 4th<br>

### 2. Implementation
```
cd m6_dnn_dae
```

(1) Data Pulling
```
python ./code/feature/data_crawler.py --meta_path './eod_data/M6_Universe.csv' --api_key '[EOD API key]' --save_path './data/' --year_duration 5
```

(2) Feature Engieering
```
python feature_engineer.py --data_path './data/' --self_data_path './pp_data/'
```

(3) Forecast Modeling
we support three models for forecasting.
```
# LightGBM
python lgb_classifier.py

# DNN
python dnn_classifier.py

# DNN with DAE
python autoencoder_dnn_classifier.py
```

(4) Decision Making
Decision making with ``data_preparation.ipynb`` and ``risk_evaluation.ipynb``.
