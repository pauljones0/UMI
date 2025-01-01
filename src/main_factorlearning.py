import logging
from logging import FileHandler
import random, os
from typing import List
import pickle as pk
import numpy as np
import pandas as pd
import torch
from torch import nn
import time
import copy, math
from data import Query
from loss import mse_loss
import model_pretrain
import joblib

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s   %(levelname)s   %(message)s')
logger = logging.getLogger("my_logger")

def day_return_type(return_ten,return_low,return_high,percent_stock):
    return_ten=return_ten.squeeze(0)
    positive=torch.sum(return_ten>return_high)
    negative=torch.sum(return_ten<return_low)
    if ((positive - negative) / return_ten.shape[0] > percent_stock):
        label_day = 0
    elif((negative-positive)/return_ten.shape[0]>percent_stock):
        label_day = 1
    else:
        label_day= 2
    return label_day
def cal_annualized_return(return_l1, hold_period):
    annualized_return = return_l1 * 252 / hold_period
    return annualized_return

def nclamp(input, min, max):
    return input.clamp(min=min, max=max).detach() + input - input.detach()



def evaluation_pre(model, model_2,model_3, list_eval, query,
                   evalrange, stk_id_dic, stk_dic_class, args, device,
                   day_tuple):
    if (args.pre_type == 'market'):
        return_low, return_high, percent_stock=day_tuple

    model.eval()
    model_2.eval()
    model_3.eval()

    total, score_a = 0, 0
    d_list = []
    if (args.pre_type == 'Cointegration'):
        u_oldlist = []
        for ui in range(args.batch_pre):
            u_old = torch.zeros(len(stk_dic_class.stk_dic.keys()) + 2).to(device)
            u_oldlist.append(u_old)

    for d in evalrange:
        d_list.append(d)
        if (len(d_list) >= args.batch_pre or d == evalrange[-1] and len(d_list) >= 3):
            if (args.pre_type == 'Cointegration'):
                x_out_all, y_out, addi_x_out, id_out_rand = \
                    model_pretrain.batch_stk_gen(query, d_list, stk_id_dic, stk_dic_class,
                                                 type=args.pre_type, )

                price, price_pred = model(addi_x_out.to(device), id_out_rand.to(device))
                loss1 = mse_loss(price, price_pred)
                u = price_pred - price
                u = u.squeeze(2)
                rho = model.rho
                loss2 = 0
                for b in range(id_out_rand.shape[0]):
                    u_oldnow = torch.index_select(u_oldlist[b], 0, id_out_rand.to(device)[b, :])
                    rhonow = torch.index_select(rho, 0, id_out_rand.to(device)[b, :])
                    diff = u[b, :] - u_oldnow * rhonow
                    square_diff = torch.pow(diff, 2)
                    loss2 = loss2 + square_diff.mean()
                loss2 = loss2 / id_out_rand.shape[0]
                loss = loss1 + 0.5 * loss2
                total += x_out_all.size(0)
                score_a += loss.item() * x_out_all.size(0)
                with torch.no_grad():
                    for b in range(id_out_rand.shape[0]):
                        u_old__ = u_oldlist[b].index_put_([id_out_rand.to(device)[b, :]], u[b, :].detach())
                        u_oldlist[b] = u_old__

            elif (args.pre_type == 'market'):
                x_out_all1, x_out_all2, label_ten, id1_out_rand, id2_out_rand, addi_x1_out, addi_x2_out,_ = \
                    model_pretrain.batch_stk_gen \
                        (query, d_list, stk_id_dic, stk_dic_class, type='market2', )
                total_embed1 = model(x_out_all1.to(device), id1_out_rand.to(device))
                total_embed2 = model(x_out_all2.to(device), id2_out_rand.to(device))
                score = model_2(total_embed1, total_embed2)
                label = torch.zeros([1], dtype=torch.long).to(device)
                _, predicted = torch.max(score, 0)
                score_a += predicted.eq(0).sum().item()

                x_out_all, y__, addi_x, id_out = \
                    model_pretrain.stk_gen_normal(query, d_list[0], stk_id_dic, stk_dic_class)
                total_embed = model(x_out_all.to(device), id_out.to(device))
                label_day=day_return_type(y__, return_low, return_high, percent_stock)

                marketpred = model_3(total_embed)
                _, predictedday = torch.max(marketpred, 0)
                score_a += predictedday.eq(label_day).sum().item()
                total += 2





            d_list = []


    score_mean = score_a / total
    print('total', total)
    return score_mean, list_eval





def main(args, state="train", num_args=0, trs=0, tes=0, tee=0, add_str=''
         , lenva=30, lente=30 ,use_stk=0  ):
    save_str = str(trs)
    ran = random.randint(0, 10)
    str_save = str(hash(save_str))
    if (args.add_dir is not None):
        str_save = args.add_dir + '/' + str_save
        if not os.path.exists("./models/" + args.add_dir):
            os.makedirs("./models/" + args.add_dir)
        if not os.path.exists("./t_log/" + args.add_dir):
            os.makedirs("./t_log/" + args.add_dir)

    work_dir = "./models/" + str_save + add_str + "r" + str(ran)
    print('dir', str_save + add_str + "r" + str(ran))
    log_dir = "./t_log/" + str_save + add_str + "r" + str(ran)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)







    log_path = log_dir + "/log.txt"
    fh = FileHandler(log_path, mode="w")
    logger.addHandler(fh)
    options = vars(args)

    logger.info(options)
    query = Query(args.data_dir, input_len=args.input_len, args=args)
    tr_start_date = query.small_exist_date(trs)
    te_start_date = query.small_exist_date(tes)
    te_end_date = query.small_exist_date(tee)
    tr_start_num = query.day2num[tr_start_date]
    te_start_num = query.day2num[te_start_date]
    te_end_num = query.day2num[te_end_date]

    train_range = list(range(tr_start_num, te_start_num-2-lenva-2))
    valid_range = list(range(te_start_num-2-lenva, te_start_num-2))
    test_range = list(range(te_start_num, min(te_start_num+lente,te_end_num)))

    print('before get_features')
    features = query.get_features()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('device', device)







    if args.pre_type == "market":
        print('before load model1')
        stk_dic_class = model_pretrain.stk_dic()
        stk_dic_class.ini_stk_dic(query)
        if args.fea_qlib == 1:
            seq_len = 60
            input_size=6
            dim_model2 = 6*2
        else:
            seq_len = args.input_len
            input_size = len(features)+2
            dim_model2 = len(features) + 2

        if (args.model_path_pre is not None):
            state = torch.load(args.model_path_pre, map_location=device)
            stk_dic_class_load = model_pretrain.stk_dic()
            stk_dic_class_load.load_(state['stk_dic'])
            model = model_pretrain.stk_classification_att1(input_size=input_size, drop_out=args.dropout_g,
            stk_total=len(stk_dic_class_load.stk_dic.keys()) + 2,use_stk=use_stk)
            model.load_state_dict(state['para'], strict=False)
            model.reinitial_stk(input_size=input_size,stk_total=len(stk_dic_class.stk_dic.keys()) + 2)
            model.use_stk = 1
        else:
            model = model_pretrain.stk_classification_att1(input_size=input_size, drop_out=args.dropout_g,
              stk_total=len(stk_dic_class.stk_dic.keys()) + 2,use_stk=use_stk)
        print('before model1 to device')
        model.to(device)
        print('before load model2')
        market_model = model_pretrain.stk_classification_small_2 \
            (dim_model2=dim_model2,drop_out=args.dropout)
        market_model_2 = model_pretrain.stk_marketpred_2 \
            (dim_model2=dim_model2,drop_out=args.dropout)
        print('before model2 to device')
        market_model.to(device)
        market_model_2.to(device)
        print('after model2 to device')
        model_list = [model, market_model,market_model_2]
    if args.pre_type == 'Cointegration':
        stk_dic_class = model_pretrain.stk_dic()
        stk_dic_class.ini_stk_dic(query)

        if (args.pre_type == 'Cointegration'):
            model = model_pretrain.stk_pred_small_2(stk_total=len(stk_dic_class.stk_dic.keys()) + 2,
                                                    drop_out=args.dropout_g)
            if (args.model_path_pre is not None):
                state = torch.load(args.model_path_pre, map_location=device)
                stk_dic_class = model_pretrain.stk_dic()
                stk_dic_class.load_(state['stk_dic'])
                model.load_state_dict(state['para'], strict=False)
            model.to(device)
            model_list = [model, model,model]

    para_list_ = [list(v.parameters()) for k, v in enumerate(model_list)]
    para_list = [x for l in para_list_ for x in l]
    para_list = list(set(para_list))
    if args.use_Adam:
        model_opt = torch.optim.Adam(params=para_list, lr=args.learning_rate)
        logger.info("Using Adam Optimizer")
    else:
        model_opt = torch.optim.SGD(params=para_list, lr=args.learning_rate, momentum=0.9)
        logger.info("Using SGD Optimizer")
    logger.info(f"train period: {query.num2day[train_range[0]]} to {query.num2day[train_range[-1]]},"
                    f" length is {len(train_range)}")
    logger.info(f"valid period: {query.num2day[valid_range[0]]} to {query.num2day[valid_range[-1]]},"
                f" length is {len(valid_range)}")
    logger.info(f"test period: {query.num2day[test_range[0]]} to {query.num2day[test_range[-1]]},"
                f" length is {len(test_range)}")
    best_val_loss = -np.Inf
    stk_id_dic = {}
    val_datalist, test_datalist = [], []
    valid_list = []
    train_list = []
    day_tuple =None
    list_threetypes = [0, 0, 0]
    if (args.pre_type == 'market'):
        return_tenl = []
        for d in range(0, len(train_range)):
            x_out_all, y__, addi_x, id_out = \
                    model_pretrain.stk_gen_normal(query, train_range[d], stk_id_dic,
                                          stk_dic_class)
            return_tenl.append(y__.squeeze(0))
        total_return = torch.cat(return_tenl, 0)
        return_ordered, sorted_indices = torch.sort(total_return, dim=-1)
        percent_return = 0.4
        percent_stock = 0.6
        return_low = return_ordered[int(len(return_ordered) * percent_return)]
        return_high = return_ordered[int(len(return_ordered) * (1 - percent_return))]
        day_tuple=(return_low,return_high,percent_stock)


    for epoch in range(args.epoch):
        epoch_loss = 0.0
        mse_loss__=0.0
        for modeli in model_list:
            modeli.train()
        train_range_other = train_range
        train_range_other = copy.deepcopy(train_range_other)
        if (args.pre_type == 'market'):
            np.random.shuffle(train_range_other)
        else:
            u_oldlist=[]
            for ui in range(args.batch_pre):
                u_old=torch.zeros(len(stk_dic_class.stk_dic.keys()) + 2).to(device)
                u_oldlist.append(u_old)
        d_list = []
        total, score_a = 0, 0
        datat, modelt = [], []
        time0 = time.time()

        for d in range(0, len(train_range_other)):
            d_list.append(train_range_other[d])
            if (len(d_list) >= args.batch_pre):

                time1 = time.time()
                if (args.pre_type == 'Cointegration'):

                    x_out_all, y_out, addi_x_out, id_out_rand = \
                        model_pretrain.batch_stk_gen(query, d_list, stk_id_dic, stk_dic_class,
                                                     type=args.pre_type,)
                    price, price_pred = model(addi_x_out.to(device), id_out_rand.to(device))
                    loss1 = mse_loss(price, price_pred)
                    u=price_pred-price
                    u=u.squeeze(2)
                    rho=nclamp(model.rho,-1,1)
                    loss2=0
                    for b in range(id_out_rand.shape[0]):
                        u_oldnow=torch.index_select(u_oldlist[b], 0, id_out_rand.to(device)[b,:])
                        rhonow = torch.index_select(rho, 0, id_out_rand.to(device)[b, :])
                        diff=u[b, :]-u_oldnow*rhonow
                        square_diff = torch.pow(diff, 2)
                        loss2=loss2+square_diff.mean()
                    loss2=loss2/id_out_rand.shape[0]
                    loss = loss1 + 0.5 * loss2
                    mse_loss__=mse_loss__+loss1.item()
                    with torch.no_grad():
                        for b in range(id_out_rand.shape[0]):
                            u_old__=u_oldlist[b].index_put_([id_out_rand.to(device)[b,:]], u[b, :].detach())
                            u_oldlist[b]=u_old__




                elif (args.pre_type == 'market'):
                    x_out_all1, x_out_all2, label_ten, id1_out_rand, id2_out_rand, addi_x1_out\
                        , addi_x2_out,distance_ten = \
                        model_pretrain.batch_stk_gen \
                            (query, d_list, stk_id_dic, stk_dic_class, type='market2')
                    total_embed1 = model(x_out_all1.to(device), id1_out_rand.to(device))
                    total_embed2 = model(x_out_all2.to(device), id2_out_rand.to(device))
                    score = market_model(total_embed1, total_embed2)
                    score=score.reshape(1,-1)
                    label=torch.zeros([1], dtype=torch.long).to(device)
                    distance_ten=distance_ten.to(device)
                    distance_ten = distance_ten.reshape(1, -1)
                    score = score / distance_ten
                    loss1 = torch.nn.functional.cross_entropy(score, label)
                    _, predicted = torch.max(score, 0)
                    total += 1
                    score_a += predicted.eq(0).sum().item()

                    x_out_all, y__, addi_x, id_out = \
                        model_pretrain.stk_gen_normal(query, d_list[0], stk_id_dic,
                                                      stk_dic_class)
                    total_embed= model(x_out_all.to(device), id_out.to(device))
                    label_day=day_return_type(y__, return_low, return_high, percent_stock)

                    if(label_day==0):
                        list_threetypes[0]+=1
                    elif(label_day==1):
                        list_threetypes[1] += 1
                    else:
                        list_threetypes[2] += 1
                    label_day = torch.ones([1], dtype=torch.long).to(device) * int(label_day)
                    marketpred = market_model_2(total_embed)
                    loss2=torch.nn.functional.cross_entropy(marketpred, label_day)



                    loss = loss1 + loss2






                d_list = []
                model_opt.zero_grad()

                loss.backward()
                epoch_loss += loss.item()
                time2 = time.time()
                datat.append(time1 - time0)
                modelt.append(time2 - time1)
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                model_opt.step()
        print("list_threetypes",list_threetypes)
        print('read_data_t', np.mean(datat))
        print('modelt', np.mean(modelt))
        epoch_loss /= len(train_range_other)
        mse_loss__/=len(train_range_other)
        if (args.pre_type == 'Cointegration'):
            logger.info(
                f"Epoch: {epoch}, mse_loss loss: {round(mse_loss__, 4)},abs rho: {round(torch.mean(torch.abs(model.rho)).item(), 4)}")
        else:
            logger.info(f"Epoch: {epoch}, train loss: {round(epoch_loss, 4)}")
        valid_score, val_datalist = evaluation_pre(model_list[0], model_list[1],  model_list[2],val_datalist, query,
                                                   valid_range, stk_id_dic, stk_dic_class, args, device,day_tuple)
        test_score, test_datalist = evaluation_pre(model_list[0], model_list[1], model_list[2], test_datalist, query,
                                                   test_range, stk_id_dic, stk_dic_class, args, device,day_tuple)
        valid_list.append(valid_score)
        train_list.append(epoch_loss)

        if (args.pre_type == 'Cointegration'):
            score_va = -1 * valid_score
        else:
            score_va = valid_score
        if score_va > best_val_loss:
            if (args.pre_type == 'market'):
                print('len va', sum([x[0].shape[0] for x in val_datalist]),
                  'len te', sum([x[0].shape[0] for x in test_datalist]), )
            state_ = {'para': model_list[0].state_dict(), 'stk_dic': stk_dic_class.stk_dic
                , 'para_2': model_list[1].state_dict()}
            torch.save(state_,
                       os.path.join(work_dir, f"best_modelpre_{args.pre_type}.pkl"))
            logger.info(f"update best model!")
            best_val_loss = score_va
            logger.info(f"best model epoch:{epoch}")
        if (args.pre_type == 'Cointegration'):
            logger.info(
                f"Epoch: {epoch}, va_score : {round(valid_score, 4)},te_score : {round(test_score, 4)},abs rho: {round(torch.mean(torch.abs(model.rho)).item(), 4)}")
        else:
            logger.info(f"{args.pre_type},va_score :{valid_score};te_score :{test_score}")
        logger.info(f"model_path_pre:" + os.path.join(work_dir, f"best_modelpre_{args.pre_type}.pkl"))

    print('valid_list',valid_list)
    print('train_list',train_list)


if __name__ == '__main__':
    from arguments import args







    args.data_dir = f''
    args.extra_data_dir = f''

    trs= 20060110

    args.tes, args.tee = 20180301, 20230301
    use_stk=1


    args.fea_qlib = 1
    if (args.pre_type == 'Cointegration'):

        main(args, trs=trs, tes=args.tes,tee=args.tee,lenva=30,lente=30,use_stk=use_stk)
    if (args.pre_type == 'market'):

        main(args, trs=trs, tes=args.tes,tee=args.tee,lenva=30,lente=30,use_stk=use_stk)


