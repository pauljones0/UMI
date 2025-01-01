import os, gc
import sys
import logging
import traceback
from logging import FileHandler
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import math

from torch import Tensor
from typing import Optional, Any, Union, Callable
from torch.nn import MultiheadAttention
from torch.nn import Linear
from torch.nn import Dropout
from torch.nn import LayerNorm

class att_market(nn.Module):
    def __init__(
        self,
        hidden_dim: int,):
        super().__init__()
        hidden_dim=hidden_dim
        self.hidden_dim=hidden_dim
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)


    def forward(self, x,temperature=1):
        out_K=self.layer1(x)
        out_Q =out_K[:,-1:,:]
        out_QK0=out_K*out_Q/np.sqrt(self.hidden_dim)

        out_QK=torch.sum(out_QK0,dim=2)




        self_attn = F.softmax(out_QK/temperature, dim=1).unsqueeze(-1)
        out=torch.sum(x*self_attn,dim=1)
        out =torch.cat([out,x[:,-1,:]],dim=1)

        return out







class stk_dic:
    def ini_stk_dic(self, query):
        set_stk = set()
        for dayd in query.date_list.keys():
            stock_list = query.date_list[dayd]
            set_stk = set_stk | set(stock_list)
        stk_dic = {}
        order = 2
        for stk in set_stk:
            stk_dic[stk] = order
            order += 1
        self.stk_dic = stk_dic

    def stk_code2id(self, stk_code):
        if stk_code in self.stk_dic.keys():
            return self.stk_dic[stk_code]
        else:
            return 1

    def load_(self, stk_dic):
        self.stk_dic = stk_dic


def batch_stk_gen(query, d_list, stk_id_dic, stk_dic_class, type='market2',days=20,sample_range=None ):
    data_dic = {}
    min_sktk = 5000
    for d in d_list:
        x, y = query.one_step_tensor(d)
        addi_x = query.extra_data_tensor[d]
        stock_list = query.date_list[d]
        assert x.shape[0] == addi_x.shape[0]
        data_dic[d] = [x, y, addi_x, stock_list]
        if (d not in stk_id_dic.keys()):
            id_list = [stk_dic_class.stk_code2id(stki) for stki in stock_list]
            id_ten = torch.tensor(id_list).long()
            stk_id_dic[d] = id_ten
        min_sktk = min(min_sktk, len(stk_id_dic[d]))
    if (type == 'Cointegration'):
        x_list, y_list, addi_x_list, id_list = [], [], [], []
        for d in d_list:
            x, y, addi_x, stock_list = data_dic[d]
            id_ten = stk_id_dic[d]
            rand_idx = torch.randperm(x.shape[0])
            rand_idx = rand_idx[:min_sktk]
            x = x[rand_idx]
            y = y[rand_idx]
            addi_x = addi_x[rand_idx]
            id_ten = id_ten[rand_idx]
            x_list.append(x.unsqueeze(0));y_list.append(y.unsqueeze(0))
            addi_x_list.append(addi_x.unsqueeze(0));id_list.append(id_ten.unsqueeze(0))
        x_out, y_out, addi_x_out, id_out = torch.cat(x_list, dim=0), torch.cat(y_list, dim=0) \
            , torch.cat(addi_x_list, dim=0), torch.cat(id_list, dim=0),


        return x_out, y_out, addi_x_out, id_out

    if (type == 'market2'):
        x1_list, x2_list, y_list, addi_x1_list, addi_x2_list, \
        id1_list, id2_list = [], [], [], [], [], [], []
        label_list = []

        distance_list = []

        d = d_list[0]
        d0 = d
        x, y, addi_x, stock_list = data_dic[d]
        id_ten = stk_id_dic[d]
        rand_idx = torch.randperm(x.shape[0])
        rand_idx1 = rand_idx[:min_sktk // 2]
        x1 = x[rand_idx1]
        y1 = y[rand_idx1]
        addi_x1 = addi_x[rand_idx1]
        id1_ten = id_ten[rand_idx1]
        rand0 = np.random.rand()
        rand_idx2 = rand_idx[min_sktk // 2:min_sktk // 2 * 2]
        x2 = x[rand_idx2].squeeze(1)
        y2 = y[rand_idx2]
        addi_x2 = addi_x[rand_idx2]
        id2_ten = id_ten[rand_idx2]
        label_list.append(1)
        x1_list.append(x1.unsqueeze(0))
        x2_list.append(x2.unsqueeze(0))
        addi_x1_list.append(addi_x1.unsqueeze(0))
        addi_x2_list.append(addi_x2.unsqueeze(0))
        id1_list.append(id1_ten.unsqueeze(0))
        id2_list.append(id2_ten.unsqueeze(0))
        distance_list.append(abs(d - d0) + 1)
        for dd in d_list:
            if dd != d:
                d2 = dd
                distance_list.append(abs(d2 - d0) + 1)
                x_, y_, addi_x_, stock_list_ = data_dic[d2]
                id_ten_ = stk_id_dic[d2]
                not_in_id_ind = torch.isin(id_ten_, id1_ten, assume_unique=True, invert=True)
                idx_ = torch.arange(0, x_.shape[0], step=1).long()
                idx_notin = torch.masked_select(idx_, not_in_id_ind)

                rand_idx_id_ = torch.randperm(idx_notin.shape[0])
                rand_idx_id_ = rand_idx_id_[:min_sktk // 2]
                rand_idx_ = idx_notin[rand_idx_id_]

                x2 = x_[rand_idx_].squeeze(1)
                y2 = y_[rand_idx_]
                addi_x2 = addi_x_[rand_idx_]
                id2_ten = id_ten_[rand_idx_]
                label_list.append(0)
                x1_list.append(x1.unsqueeze(0))
                x2_list.append(x2.unsqueeze(0))
                addi_x1_list.append(addi_x1.unsqueeze(0))
                addi_x2_list.append(addi_x2.unsqueeze(0))
                id1_list.append(id1_ten.unsqueeze(0))
                id2_list.append(id2_ten.unsqueeze(0))

        x1_out, x2_out, addi_x1_out, addi_x2_out, id1_out, id2_out = torch.cat(x1_list, dim=0), \
        torch.cat(x2_list, dim=0), torch.cat(addi_x1_list,dim=0), torch.cat(addi_x2_list, dim=0), \
        torch.cat(id1_list, dim=0), torch.cat(id2_list,dim=0)
        label_ten = torch.tensor(label_list).long()
        distance_ten = torch.tensor(distance_list).long()

        return x1_out, x2_out, label_ten, id1_out, id2_out,addi_x1_out, addi_x2_out,distance_ten


def stk_gen_normal(query, d, stk_id_dic, stk_dic_class,):
    x, y = query.one_step_tensor(d)
    addi_x = query.extra_data_tensor[d]
    stock_list = query.date_list[d]
    if (d not in stk_id_dic.keys()):
        id_list = [stk_dic_class.stk_code2id(stki) for stki in stock_list]
        id_ten = torch.tensor(id_list).long()
        stk_id_dic[d] = id_ten
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    addi_x = addi_x.unsqueeze(0)
    id_out = stk_id_dic[d].unsqueeze(0)
    return x, y,addi_x, id_out


class stk_classification_att1(nn.Module):
    def __init__(self, input_size=10,drop_out=0.3,stk_total=4000,use_stk=0):
        super().__init__()
        self.use_stk=use_stk
        self.dropout = drop_out
        self.stk_total=stk_total
        self.stk_matrix =nn.Parameter(torch.zeros(1, stk_total,2*input_size))
        nn.init.xavier_uniform_(self.stk_matrix.data)
        self.weightlayer=nn.Linear(2*input_size, 1)
        self.layer = att_market(hidden_dim=input_size,)
    def reinitial_stk(self,input_size=10,stk_total=4000):
        self.stk_matrix =nn.Parameter(torch.zeros(1, stk_total,2*input_size))
        nn.init.xavier_uniform_(self.stk_matrix.data)


    def forward(self, x, stk_ten,moreout=0):
        stk_ten = stk_ten.long()
        list_out_=[]
        for b in range(x.shape[0]):
            xi=x[b,:,:,:]
            o_i = self.layer(xi).unsqueeze(0)
            stk_m = torch.index_select(self.stk_matrix, 1, stk_ten[b, :])*self.use_stk
            weight_i =F.relu(self.weightlayer(o_i+stk_m))
            out_i = torch.sum(weight_i * o_i, dim=1) / (
                        torch.sum(weight_i, dim=1) + 1e-8)
            list_out_.append(out_i)
        out = torch.cat(list_out_, dim=0)
        if(moreout==1):
            outstks=(o_i+stk_m).squeeze(0)
            return out,outstks
        return out

class stk_classification_small_2(nn.Module):
    def __init__(self, dim_model2=500 * 5, drop_out: float = 0.0, ):
        super().__init__()
        self.bn1 =torch.nn.BatchNorm1d(dim_model2*2)
        self.dropout = drop_out

        self.return_layer = nn.Sequential(
            nn.Linear(dim_model2*2, 1),
        )

    def forward(self, total_embed1,total_embed2):

        total_embed=torch.cat([total_embed1,total_embed2],dim=1)
        total_embed = self.bn1(total_embed)
        score = self.return_layer(
            F.dropout(total_embed, p=self.dropout, training=self.training))

        return score
class stk_marketpred_2(nn.Module):
    def __init__(self, dim_model2=6, drop_out: float = 0.0, out=3):
        super().__init__()
        self.dropout = drop_out
        self.fc1 = nn.Linear(dim_model2, dim_model2)

        self.return_layer = nn.Sequential(
            nn.Linear(dim_model2, out),
        )

    def forward(self, total_embed):
        total_embed=torch.relu(self.fc1(total_embed))
        score = self.return_layer(
            F.dropout(total_embed, p=self.dropout, training=self.training))

        return score
class stk_pred_small_1(nn.Module):
    def __init__(self, stk_total=4000,drop_out=0.3):
        super().__init__()
        self.dropout = drop_out
        self.stk_matrix=nn.Parameter(torch.ones(1,stk_total, stk_total))
        self.stk_weight = nn.Parameter(torch.ones(1, stk_total, stk_total))
        nn.init.xavier_uniform_(self.stk_weight.data, gain=1)

    def forward(self, x, stk_ten):
        stk_ten=stk_ten.long()
        stk_m_sl=[]
        stk_w_sl = []
        for b in range(stk_ten.shape[0]):
            stk_m=torch.index_select(self.stk_matrix, 1, stk_ten[b,:])
            stk_m = torch.index_select(stk_m, 2, stk_ten[b, :])
            stk_m_sl.append(stk_m)
            stk_w = torch.index_select(self.stk_weight, 1, stk_ten[b, :])
            stk_w = torch.index_select(stk_w, 2, stk_ten[b, :])
            stk_w_sl.append(stk_w)
        stk_m_s=torch.cat(stk_m_sl,dim=0)
        diag = torch.diagonal(stk_m_s, dim1=1, dim2=2)
        a_diag = torch.diag_embed(diag)
        stk_m_s2=stk_m_s-a_diag
        stk_m_s2=F.dropout(stk_m_s2, p=self.dropout, training=self.training)

        stk_w_s = torch.cat(stk_w_sl, dim=0)
        stk_w_s=torch.abs(stk_w_s)
        diagw = torch.diagonal(stk_w_s, dim1=1, dim2=2)
        a_diagw = torch.diag_embed(diagw)
        stk_w_s2 = stk_w_s - a_diagw
        stk_w_s2 = F.dropout(stk_w_s2, p=self.dropout, training=self.training)



        att = F.softmax(stk_w_s2, dim=2)



        stk_m_s2=stk_m_s2*att



        out=torch.matmul(stk_m_s2,x)






        return out
class stk_pred_small_2(nn.Module):
    def __init__(self, stk_total=4000,drop_out=0.3):
        super().__init__()
        self.m1 = stk_pred_small_1(stk_total=stk_total,drop_out=drop_out)
        self.m2 = stk_pred_small_1(stk_total=stk_total, drop_out=drop_out)
        self.rho=nn.Parameter(torch.zeros(stk_total))

    def forward(self, x, stk_ten):
        x=x[:,:,-1,:]

        close_=x[:,:,1:2]
        close_mean = torch.mean(close_, dim=1, keepdim=True)
        close_std = torch.std(close_, dim=1, keepdim=True)
        close_ = (close_ - close_mean) / (close_std + 1e-8)
        close_pred=self.m1(close_,stk_ten)
        return close_, close_pred














