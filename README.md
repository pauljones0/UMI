# KDD2025 UMI

## Overview
This repository contains code for learning irrational factors and forecasting in financial markets.

## Getting Started

### Main Scripts
- `main_factorlearning.py`: Implements the irrational factors learning process
- `main_forecasting.py`: Handles the forecasting process

## Data Structure

### Main Data (`args.data_dir`)
The main data file is a DataFrame containing the following columns:
- `StkCode`: Stock identifier
- `Date`: Date of the record
- Feature columns (extracted by qlib):
  - `Fea_feature_CLOSE[59-0]`: Close price features
  - `Fea_feature_OPEN[59-0]`: Open price features
  - `Fea_feature_HIGH[59-0]`: High price features
  - `Fea_feature_LOW[59-0]`: Low price features
  - `Fea_feature_VWAP[59-0]`: Volume Weighted Average Price features
  - `Fea_feature_VOLUME[59-0]`: Volume features
- Labels:
  - `label`: Primary model label
  - `label_2`: Secondary model label (identical to primary)

### Additional Data

#### Factor Learning Process (`extra_data_dir`)
Contains price features for MARKET REPRESENTATION FACTOR:
- Columns: `['Date', 'StkCode', 'close_59', ..., 'close_0']`
- `close_xxx`: Historical price values

#### Forecasting Process (`extra_data_dir`)
Contains extracted STOCK RELATION FACTOR data:
- Columns: `['Date', 'StkCode', 'pred_59', ..., 'pred_0', 'price_59', ..., 'price_0']`
- `pred_xxx`: p_tilde values
- `price_xxx`: p values

## Online Testing
We provide real-time output signals from our methodology's application in the Chinese stock market. These signals are:
- Updated daily at market close
- Available at: [https://github.com/lIcIIl/T0](https://github.com/lIcIIl/T0)

This continuous updating process demonstrates the methodology's:
- Real-world effectiveness
- Practical applicability
- Adaptability to dynamic market conditions




