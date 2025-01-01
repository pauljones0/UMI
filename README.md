
## KDD2025 UMI


### Running

main_factorlearning.py: learning irrational factors.

main_forecasting: forecasting process.
### Data Description
args.data_dir corresponds to our main data file, which is a dataframe with columns ['StkCode', 'Date', 'Fea_feature_CLOSE59', ...'Fea_feature_CLOSE0', 'Fea_feature_OPEN59', ...'Fea_feature_OPEN0', 'Fea_feature_HIGH59', ...'Fea_feature_HIGH0', 'Fea_feature_LOW59', .. 'Fea_feature_LOW1', 'Fea_feature_LOW0', 'Fea_feature_VWAP59', 'Fea_feature_VWAP58', ...'Fea_feature_VWAP0', 'Fea_feature_VOLUME59', ... 'Fea_feature_VOLUME0', 'label', 'label_2']. 'Fea_feature_xxx' refers to the features extracted by qlib. 'label' and 'label_2' are the label of the model (they are the same).

In factor learning process (main_factorlearning), extra_data_dir corresponds to the price feature used in MARKET REPRESENTATION FACTOR, which is a dataframe with columns ['Date', 'StkCode', 'close_59', ...,'close_0']. 'close_xxx' refers to the price value.

In forecasting process (main_forecasting), extra_data_dir corresponds to the extracted in  STOCK RELATION FACTOR,which is a dataframe with columns ['Date', 'StkCode', 'pred_59',...,'pred_0','price_59',...,'price_0']. 'pred_xxx' and 'price_xxx' refers to the p_tilde and p.

### Online Test
To substantiate the effectiveness and practical application of our proposed methodology, we have uploaded the real-time output signals derived from its application in the Chinese stock market as an online test. Notably, these signals are meticulously updated at the close of every trading day, ensuring that the data reflects the most current market conditions and predictions. This continuous updating process not only validates our approach but also highlights its adaptability and responsiveness to dynamic market environments.

The signal of our online test is provided at https://github.com/lIcIIl/T0




