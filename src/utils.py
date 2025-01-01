
import numpy as np
import pandas as pd
import torch
from scipy.fft import idct


def weighted_corrcoef(y1, y2, weight):
    if type(y1) == torch.Tensor:
        y1_weighted_mean, y2_weighted_mean = torch.matmul(weight, y1.T), torch.matmul(
            weight, y2.T
        )
        return torch.matmul(
            weight, (y1 - y1_weighted_mean).multiply(y2 - y2_weighted_mean).T
        ) / (
            torch.sqrt(torch.matmul(weight, ((y1 - y1_weighted_mean) ** 2).T))
            * torch.sqrt(torch.matmul(weight, ((y2 - y2_weighted_mean) ** 2).T))
            + 1e-8
        )
    else:
        y1_weighted_mean, y2_weighted_mean = np.matmul(weight, y1.T), np.matmul(
            weight, y2.T
        )
        return np.matmul(
            weight, (y1 - y1_weighted_mean) * (y2 - y2_weighted_mean).T
        ) / (
            np.sqrt(np.matmul(weight, ((y1 - y1_weighted_mean) ** 2).T))
            * np.sqrt(np.matmul(weight, ((y2 - y2_weighted_mean) ** 2).T))
            + 1e-8
        )


def generate_weight(stock_num, method=None):
    if method is None:
        return torch.ones((stock_num,)) / stock_num
    weight_list = []

    one_decile = stock_num // 10
    if method == "exp_decay":
        for j in range(10):
            if j < 9:
                weight_list += [0.9 ** j] * one_decile
            else:
                weight_list += [0.9 ** j] * (stock_num - one_decile * 9)
    else:
        for j in range(10):
            if j < 9:
                weight_list += [(10 - j) / 10] * one_decile
            else:
                weight_list += [(10 - j) / 10] * (stock_num - one_decile * 9)
    weight = torch.FloatTensor(weight_list)
    weight /= weight.sum()
    assert len(weight) == stock_num
    return weight


def ndcg(golden, current, n = -1):
    log2_table = np.log2(np.arange(2, 5002))

    def dcg_at_n(rel, n):
        rel = np.asfarray(rel)[:n]

        dcg = np.sum(np.divide(np.power(2, rel) - 1,
          log2_table[:rel.shape[0]]))
        return dcg


    k = len(current) if n == -1 else n
    idcg = dcg_at_n(sorted(golden, reverse=True), n = k)
    dcg = dcg_at_n(current, n=k)
    tmp_ndcg = 0 if idcg == 0 else dcg / idcg
    return tmp_ndcg


def idcg(n_rel):
    nums = np.ones(n_rel)
    denoms = np.log2(np.arange(n_rel) + 1 + 1)
    return (nums / denoms).sum()


def lambdaRank_update(args, x, y, scores, device):
    scores = scores.unsqueeze(1)
    _, sorted_idx = scores.sort(dim=0, descending=True)
    n_stocks = scores.size(0)
    n_rel = min(args.lambda_topk, n_stocks-1)
    n_irr = n_stocks - n_rel
    stock_ranks = torch.zeros(n_stocks).to(device)
    stock_ranks[sorted_idx] = 1 + torch.arange(n_stocks).view(-1, 1).to(device).float()
    stock_ranks = stock_ranks.view(-1, 1)
    score_diffs = scores[:n_rel] - scores[n_rel:].view(-1)
    exped = score_diffs.exp()
    N =  1 / idcg(n_rel)
    dcg_diffs = 1 / (1 + stock_ranks[:n_rel]).log2() - (1 / (1 + stock_ranks[n_rel:]).log2()).view(-1)
    lamb_updates = 1 / (1 + exped) * N * dcg_diffs.abs()
    lambs = torch.zeros((n_stocks, 1)).to(device)
    lambs[:n_rel] += lamb_updates.sum(dim=1, keepdim=True)
    lambs[n_rel:] -= lamb_updates.sum(dim=0, keepdim=True).t()
    ndcg_value = ndcg(y.cpu().numpy().reshape(-1), y[sorted_idx].cpu().numpy().reshape(-1), n=args.lambda_topk)
    return -ndcg_value, lambs.squeeze()


def df_to_dict(df: pd.DataFrame):
    daily_dict = {}
    df = df.sort_values(by=['Date', 'StkCode'], ascending=True)
    for date, daily_df in df.groupby('Date'):
        daily_dict[date] = daily_df
    return daily_dict

def df_to_dict2(df: pd.DataFrame):
    daily_dict = {}
    df = df.sort_values(by=['Date', 'StkCode'], ascending=True)
    for date, daily_df in df.groupby('Date'):
        daily_dict[int(date)] = daily_df
    return daily_dict
def adjust_lr(optimizer, epoch, learning_rate):
    lr = learning_rate
    if epoch >= 5:
        lr = learning_rate * 0.5
    if epoch >= 15:
        lr = learning_rate * 0.1
    if epoch >= 75:
        lr = learning_rate * 0.05

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
