### 1. Introduction
M competition is a well-known competition focused on time series forecasting, held once a year. This year's M6 is a financial forecasting competition that primarily focuses on predicting stock prices (returns) and risks, providing new perspectives on the Efficient Market Hypothesis (EMH).<br>
Our Team: OPPO XZ Lab<br>
Final Global Rank: 4th<br>

### 2. Implementation
```
cd m6_dea_dnn
```

Data Pulling
```
python ./code/feature/data_crawler.py --meta_path './eod_data/M6_Universe.csv' --api_key '[EOD API key]' --save_path './data/' --year_duration 5
```

(2) Feature Engieering
```
mkdir pp_data2
python ./code/feature/feature_engineer.py --data_path './data/' --self_data_path './pp_data2/'
```

(3) Forecast Modeling
we support three models for forecasting.
```
# LightGBM
python ./code/lgb_classifier.py

# DNN
python ./code/dnn_classifier.py

# DNN with DAE
python ./code/autoencoder_dnn_classifier.py
```

(4) Decision Making

- Data Preparation: ```data_preparation.ipynb```
- Traditional models:  ```traditional_models.ipynb```
- GA and PSO: ```nonlinear_models.ipynb```
- DE: ```DE_experiments.ipynb```

Experiment result see: 
```
# Traditional models
traditional_models.ipynb

# GA
ga_weights.zip

# PSO
pso_weights.zip

# DE
de_weights_100_100.zip
de_weights_maxiter.zip
de_weights_sizepop.zip
```