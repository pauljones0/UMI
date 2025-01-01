
import logging
import os
from typing import List
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import pandas as pd
import torch,copy,csv
from sklearn import preprocessing

from utils import df_to_dict,df_to_dict2

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s   %(levelname)s   %(message)s')
logger = logging.getLogger("my_logger")

class Query:
    def __init__(self,
                 data_dir: str,
                 input_len: int = 1,args=None):
        self.args=args
        if(args.extra_data_dir is not None):

            extra_data = pd.read_parquet(args.extra_data_dir)
            extra_data['Date'] = extra_data['Date'].apply(lambda x: int("".join(x.split('-'))))
            extra_data['StkCode'] = extra_data['StkCode']
            self.extra_datad=df_to_dict2(extra_data)
            self.extra_data_tensor={}

        all_df = pd.read_parquet(data_dir)
        all_df['Date'] = all_df['Date'].apply(lambda x: int("".join(x.split('-'))))
        all_df['StkCode'] = all_df['StkCode']






        all_df.set_index("StkCode", inplace=True)


        self.input_len = input_len
        self.factor_data = df_to_dict(all_df)
        dates = list(self.factor_data.keys())
        self.dates =dates
        date_codes = list(range(len(dates)))
        self.day2num = dict(zip(dates, date_codes))
        self.num2day = dict(zip(date_codes, dates))
        self.factor_data = dict(zip(date_codes, self.factor_data.values()))

        self.lendate=len(dates)

        self.date_list = {}
        self.date_list_not_contained={}
        self.__features = list(all_df.columns)[1:-2]
        print('__features',self.__features)
        self.label_name =list(all_df.columns)[-1]


        for date in dates[input_len:]:
            d = self.day2num[date]

            daily_df = self.factor_data[d]
            target_list = set(daily_df.index)
            for date_code in range(d - input_len + 1, d + 1):
                target_list &= set(self.factor_data[date_code].index)




            if(len(target_list)>0):
                self.date_list[d] = sorted(list(target_list))
            logger.info(f"date is {self.num2day[d]}, number of stocks is {len(target_list)}")

        del all_df
        if(args.extra_data_dir is not None):
            del extra_data



        self._stockdf2tensor()



        print('finish')

    def small_exist_date(self,date):
        day_=0
        for d in self.dates:
            if d >day_ and d<=date:
                day_=d
        return day_
    def concat2input(self,
                     date_code: int,
                     stock_list: List[int],
                     scaler=preprocessing.StandardScaler):
        result = []
        myScaler = scaler()
        for k in range(date_code - self.input_len + 1, date_code + 1):
            df = self.factor_data[k].loc[stock_list, self.__features]
            df_val = df.values
            if(self.args.fea_norm==1):
                df_val = myScaler.fit_transform(df_val)
            result.append(df_val)

        result = np.array(result)
        result = np.rollaxis(result, 1)
        y = self.factor_data[date_code].loc[stock_list, self.label_name].values

        return result, y

    def one_step(self,
                 begin: int,
                 stock_list: List[int] = None):
        if begin not in self.date_list:
            logger.error(f"illegal date_code: {begin}, date is {self.num2day[begin]}")
            return None
        if stock_list is None:
            fetch_list = self.date_list[begin]
        else:
            fetch_list = stock_list
        data = self.concat2input(begin, fetch_list)
        if (self.args.extra_data_dir is not None):
            if self.num2day[begin] not in self.extra_datad.keys():
                logger.error(f"illegal extra_datad date_code: {begin}, date is {self.num2day[begin]}")
            df_now=self.extra_datad[self.num2day[begin]]
            if (self.args.fea_qlib == 1):
                if (self.args.extra_price == 1):
                    T_list = [f'close_{T}' for T in range(59, -1, -1)]
                    df = df_now.loc[df_now['StkCode'].isin(fetch_list), T_list].values
                    extra_data1 = df.reshape(df.shape[0], 60, -1)
                    extra_data = np.concatenate([extra_data1, extra_data1], axis=-1)
                else:
                    T_list = [f'pred_{T}' for T in range(59, -1, -1)]
                    df = df_now.loc[df_now['StkCode'].isin(fetch_list), T_list].values
                    extra_data1 = df.reshape(df.shape[0], 60, -1)
                    T_list = [f'price_{T}' for T in range(59, -1, -1)]
                    df = df_now.loc[df_now['StkCode'].isin(fetch_list), T_list].values
                    extra_data2 = df.reshape(df.shape[0], 60, -1)
                    extra_data = np.concatenate([extra_data1, extra_data2], axis=-1)

            print('extra_data.shape', extra_data.shape)

            return data,extra_data
        return data

    def one_step_tensor(self,
                        begin: int):
        if begin not in self.factor_tensor:
            logger.error(f"illegal date_code: {begin}, date is {self.num2day[begin]}")
            return None
        factor_data = self.factor_tensor[begin]
        label_data = self.label_tensor[begin]
        if(self.args.fea_qlib==1):
            factor_data= factor_data.reshape(factor_data.shape[0], 6, -1)
            factor_data =factor_data.permute(0,2,1)
        return factor_data, label_data



    def _stockdf2tensor_d(self,d):
        if (self.args.extra_data_dir is not None):
            data, extra_data = self.one_step(d)
            factor_data, label = data
            self.factor_tensor[d] = torch.from_numpy(factor_data).float()
            self.label_tensor[d] = torch.from_numpy(label).float()
            self.extra_data_tensor[d] = torch.from_numpy(extra_data).float()
        else:
            factor_data, label = self.one_step(d)
            self.factor_tensor[d] = torch.from_numpy(factor_data).float()
            self.label_tensor[d] = torch.from_numpy(label).float()



    def _stockdf2tensor(self):
        self.factor_tensor = {}
        self.label_tensor = {}

        logger.info("Transforming stock dataframe to tensor...")
        start_date, end_date = self.get_date_codes()
        items=list(range(start_date, end_date))
        pool = ThreadPool()
        pool.map(self._stockdf2tensor_d, items)
        pool.close()
        pool.join()

        logger.info("Transform finished")


    def get_date_codes(self):
        date_codes = list(self.date_list.keys())
        return date_codes[0], date_codes[-1] + 1

    def get_features(self):
        return self.__features
    def clean_data(self,list):
        for key in self.factor_tensor.keys():
            if key not in list:
                self.factor_tensor[key]=0
                self.factor_data[key]=0
                self.extra_data_tensor[key]=0


