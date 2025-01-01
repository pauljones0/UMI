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
from loss import ic_loss, mse_loss
from utils import adjust_lr
import model_pretrain,model_seq

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s   %(levelname)s   %(message)s')
logger = logging.getLogger("my_logger")


def evaluation(model: nn.Module,
               query: Query,
               val_range: List[int],
               loss_fn,
               device: torch.device):
    model.eval()
    loss = .0
    with torch.no_grad():
        for d in val_range:
            x, y = query.one_step_tensor(d)
            x = x.to(device)
            y = y.to(device)
            scores = model(x).squeeze(1)
            loss += loss_fn(scores, y, device).item()
    loss /= len(val_range)
    model.train()
    return loss


def cal_annualized_return(return_l1, hold_period):
    annualized_return = return_l1 * 250 / hold_period
    return annualized_return



def evaluation3(model_pre_market,model,query,
                   val_range,stk_id_dic,stk_dic_class,device,args):
    import scipy.stats as stats
    if (args.model_path_pre_market is not None):
        model_pre_market.eval()
    model.eval()
    loss = .0
    ic_list = []
    rank_ics_list = []
    l1_return_list = []
    l10_return_list = []
    with torch.no_grad():
        for d in val_range:
            x, y = query.one_step_tensor(d)
            x_out_all, y__, addi_x, id_out = \
                model_pretrain.stk_gen_normal(query, d, stk_id_dic,
                                              stk_dic_class)
            x = x.to(device)
            y = y.to(device)
            if (args.model_path_pre_mask is not None):
                addi_x = addi_x.squeeze(0)[:, :, 0:1] - addi_x.squeeze(0)[:, :, 1:2]
                x = torch.cat([x, addi_x.to(device), addi_x.to(device)], dim=-1)
            if (args.model_path_pre_market is not None):
                x_market = x_out_all

                total_embed, outstks = model_pre_market(x_market.to(device), id_out.to(device), moreout=1)
                total_embed = total_embed.repeat(x.shape[0], 1)
                x_more = (total_embed, outstks)
            else:
                x_more=None
            scores = model(x,addi_x=x_more).squeeze(1)
            scores_np = scores.cpu().numpy()
            rets = y.cpu().numpy()
            ic_list.append(stats.pearsonr(scores_np, rets)[0])
            rank_ics_list.append(stats.spearmanr(scores_np, rets)[0])
            score_sorted_indices = np.argsort(-scores_np)[:int(0.1 * len(scores_np))]
            l1_return_list.append(np.mean(rets[score_sorted_indices]))
            l10_sorted_indices = np.argsort(-scores_np)[-int(0.1 * len(scores_np)):]
            l10_return_list.append(np.mean(rets[l10_sorted_indices]))
    ic = np.mean(ic_list)
    rank_ic = np.mean(rank_ics_list)
    l1_return = np.mean(l1_return_list)
    l10_return = np.mean(l10_return_list)
    return (ic, rank_ic, l1_return, l10_return)







def main(args, state="train", num_args=0, trs=0, tes=0, tee=0, add_str='',hold_day=1
         , lenva=30, lente=30):
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
    features = query.get_features()

    stk_id_dic = {}
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    stk_dic_class = model_pretrain.stk_dic()
    stk_dic_class.ini_stk_dic(query)
    len_pre_mask_stk,len_pre_market=0,0
    if (args.pre_type == 'market'):
        if args.fea_qlib == 1:
            input_size=6
            dim_model2 = 6*2
        else:
            input_size = len(features)+2
            dim_model2 = len(features) + 2
        model_pre_market = model_pretrain.stk_classification_att1(input_size=input_size,drop_out=args.dropout_g,
                                                stk_total=len(stk_dic_class.stk_dic.keys()) + 2)
        if (args.model_path_pre_market is not None):
            state_market = torch.load(args.model_path_pre_market, map_location=device)
            print('load_2')
            model_pre_market.load_state_dict(state_market['para'], strict=False)
        model_pre_market.to(device)
        len_pre_market = dim_model2


    if (args.model_path_pre_market is None):
        len_pre_market = 0
        model_pre_market=0

    seq_len = args.input_len
    input_size = len(features)
    if (args.fea_qlib == 1):
        seq_len = 60
        input_size = 6
        if (args.model_path_pre_mask is not None):
            input_size = input_size+2
    if args.model == "transformer":
        model = model_seq.Trans(input_size=input_size, num_heads=args.num_heads, dim_model=args.hidden_size,
                                dim_ff=args.dim_ff, seq_len=seq_len, num_layers=args.num_layers,
                                dropout=args.dropout, add_xdim=len_pre_mask_stk + len_pre_market,embeddim=6)


    print('device', device)
    model.to(device)
    if args.use_Adam:
        if (args.model_path_pre_market is None):
            model_pre_params =[]
        else:
            model_pre_params = list(model_pre_market.parameters())
        model_params = list(model.parameters())
        model_opt = torch.optim.Adam([{'params': model_pre_params, 'lr': 1e-7},
                                      {'params': model_params, 'lr': args.learning_rate}])
        logger.info("Using Adam Optimizer")





    else:
        model_opt = torch.optim.SGD(params=model.parameters(), lr=args.learning_rate, momentum=0.9)
        logger.info("Using SGD Optimizer")

    logger.info(f"train period: {query.num2day[train_range[0]]} to {query.num2day[train_range[-1]]},"
                f" length is {len(train_range)}")

    logger.info(f"valid period: {query.num2day[valid_range[0]]} to {query.num2day[valid_range[-1]]},"
                f" length is {len(valid_range)}")
    logger.info(f"test period: {query.num2day[test_range[0]]} to {query.num2day[test_range[-1]]},"
                f" length is {len(test_range)}")

    loss_va = ic_loss
    best_val_loss = -np.Inf
    all_list,all_list_extend=[],[]

    for epoch in range(args.epoch):
        epoch_loss = 0.0
        if (args.model_path_pre_market is not None):
            model_pre_market.train()
        model.train()
        train_range_other = train_range
        train_range_other = copy.deepcopy(train_range_other)

        np.random.shuffle(train_range_other)
        read_data_t, device_t, model_ft, losss_t, step_t, = [], [], [], [], [],

        for d in range(0, len(train_range_other)):
            time0 = time.time()

            time1 = time.time()
            read_data_t.append(time1 - time0)

            x, y = query.one_step_tensor(train_range_other[d])
            x_out_all, y__, addi_x, id_out = \
                model_pretrain.stk_gen_normal(query, train_range_other[d], stk_id_dic,
                                              stk_dic_class)
            x = x.to(device)
            y = y.to(device)

            y = (y - torch.mean(y)) / (torch.std(y) + 1e-8)
            sorted_return_label, sorted_indices = torch.sort(y, dim=-1)
            _, index = torch.sort(sorted_indices, dim=-1)
            y_rank = index.float().to(device)
            y_rank = (y_rank - torch.mean(y_rank)) / (torch.std(y_rank) + 1e-8)



            if (args.model_path_pre_mask is not None):
                addi_x=addi_x.squeeze(0)[:,:,0:1]-addi_x.squeeze(0)[:,:,1:2]
                x=torch.cat([x,addi_x.to(device),addi_x.to(device)],dim=-1)
            if (args.model_path_pre_market is not None):
                x_market = x_out_all

                total_embed ,outstks= model_pre_market(x_market.to(device), id_out.to(device),moreout=1)
                total_embed = total_embed.repeat(x.shape[0], 1)
                x_more=(total_embed,outstks)
            else:
                x_more=None


            time2 = time.time()
            device_t.append(time2 - time1)
            scores = model(x, addi_x=x_more).squeeze(1)
            time3 = time.time()
            model_ft.append(time3 - time2)


            loss=torch.nn.functional.mse_loss(scores, y)+0.1*ic_loss(scores, y_rank, device)
            model_opt.zero_grad()

            loss.backward()
            time4 = time.time()
            losss_t.append(time4 - time3)
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            model_opt.step()

            time5 = time.time()
            step_t.append(time5 - time4)

            epoch_loss += loss.item()

        print('read_data_t', np.mean(read_data_t))
        print('device_t', np.mean(device_t))
        print('model_ft', np.mean(model_ft))
        print('losss_t', np.mean(losss_t))
        print('step_t', np.mean(step_t))
        epoch_loss /= len(train_range_other)
        logger.info(f"Epoch: {epoch}, train loss: {round(epoch_loss, 4)}")


        ic_va__, rank_ic_va, l1_return_va, l10_return_va = evaluation3(model_pre_market, model,
                                                               query,
                                                               valid_range, stk_id_dic, stk_dic_class, device,
                                                               args)
        if (args.valid_return == 1):
            ic_va__ = l1_return_va.item()
        elif (args.valid_return_l10 == 1):
            ic_va__ = -l10_return_va.item()
        else:
            ic_va__ = ic_va__.item()
        logger.info(f"Epoch: {epoch}, return_va: {round(ic_va__, 4)}")

        ic_te, rank_ic_te, l1_return_te, l10_return_te = evaluation3(model_pre_market,
                                                                     model, query,
                                                                     test_range, stk_id_dic, stk_dic_class,
                                                                     device, args, )
        ic_va, rank_ic_va, l1_return_va, l10_return_va = evaluation3(model_pre_market,
                                                                     model, query,
                                                                     valid_range, stk_id_dic, stk_dic_class,
                                                                     device, args, )


        if ic_va__ > best_val_loss:
            if (args.model_path_pre_market is not None):
                state_ = {'para_pre_market': model_pre_market.state_dict(), 'stk_dic': stk_dic_class.stk_dic
                , 'para_2': model.state_dict()}
            else:
                state_ = {'stk_dic': stk_dic_class.stk_dic
                    , 'para_2': model.state_dict()}
            torch.save(state_,
                       os.path.join(work_dir, f"best_model.pkl"))
            logger.info(f"update best model!")
            best_val_loss = ic_va__
        adjust_lr(model_opt, epoch, args.learning_rate)

        annual_te = cal_annualized_return(l1_return_te, hold_period=hold_day)
        annual_va = cal_annualized_return(l1_return_va, hold_period=hold_day)
        annual_te_l10 = cal_annualized_return(l10_return_te, hold_period=hold_day)
        annual_va_l10 = cal_annualized_return(l10_return_va, hold_period=hold_day)
        print('annual_va', round(annual_va, 5), 'annual_te', round(annual_te, 5),
              'annual_va_l10', round(annual_va_l10, 5), 'annual_te_l10', round(annual_te_l10, 5),
              'ic_va', round(ic_va, 5), 'ic_te', round(ic_te, 5),
              'rank_ic_va', round(rank_ic_va, 5), 'rank_ic_te', round(rank_ic_te, 5))
    state = torch.load(os.path.join(work_dir, "best_model.pkl"), map_location=device)
    stk_dic_class = model_pretrain.stk_dic()
    stk_dic_class.load_(state['stk_dic'])
    if (args.model_path_pre_market is not None):
        model_pre_market.load_state_dict(state['para_pre_market'], strict=False)
    model.load_state_dict(state['para_2'], strict=False)

    if (state != 'test'):

        signal_dataframe=[]
        with torch.no_grad():
            for d in test_range:
                if (args.model_path_pre_market is not None):
                    model_pre_market.eval()
                model.eval()
                x, _ = query.one_step_tensor(d)
                x_out_all, y__, addi_x, id_out = \
                    model_pretrain.stk_gen_normal(query, d, stk_id_dic,
                                                  stk_dic_class)
                x = x.to(device)
                if (args.model_path_pre_mask is not None):
                    addi_x = addi_x.squeeze(0)[:, :, 0:1] - addi_x.squeeze(0)[:, :, 1:2]
                    x = torch.cat([x, addi_x.to(device), addi_x.to(device)], dim=-1)
                if (args.model_path_pre_market is not None):
                    x_market = x_out_all

                    total_embed, outstks = model_pre_market(x_market.to(device), id_out.to(device), moreout=1)
                    total_embed = total_embed.repeat(x.shape[0], 1)
                    x_more = (total_embed, outstks)
                else:
                    x_more = None
                scores = model(x, addi_x=x_more)
                scores = scores.squeeze(1).cpu().numpy()
                stock_list = query.date_list[d]
                daily_df = pd.DataFrame(scores, columns=['Value'])
                daily_df["StkCode"] = np.array(stock_list)
                daily_df["Date"] = query.num2day[d]
                original_df = query.factor_data[d]
                daily_df = pd.merge(daily_df, original_df, on=["Date", "StkCode"])
                daily_df["Time"] = 83000
                daily_df["Ind_Name"] = "None"
                daily_df = daily_df.loc[
                           :, ["Date", "StkCode", "Time", "Ind_Name", "Value"]
                           ]
                signal_dataframe.append(daily_df)
            signal_dataframe = pd.concat(signal_dataframe, axis=0)
            logger.info(signal_dataframe)
            signalout_path = "./models/" + str_save + add_str + "r" + str(
                ran) + '/signal_out_test'

            signal_dataframe.to_parquet(signalout_path)
        performance_try = [annual_va, annual_te, annual_va_l10, annual_te_l10]
        performance_try_extend = [annual_va, annual_te, annual_va_l10, annual_te_l10,
                                  [ic_va, rank_ic_va, l1_return_va, ic_te, rank_ic_te, l1_return_te],
                                  signalout_path]
        all_list.append(performance_try)
        all_list_extend.append(performance_try_extend)
        save_list = [train_range, all_list, all_list_extend]
        print('start print all_list')
        print('all_list', all_list)
        print('all_list_extend', all_list_extend)



    return 0
if __name__ == '__main__':
    from arguments import args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)




    trs= 20060110
    args.tes, args.tee = 20180301, 20230301
    args.pre_type = 'market'
    args.model = 'transformer'
    args.data_dir = f''
    args.extra_data_dir = f''
    args.model_path_pre_mask = 1
    args.model_path_pre_market = f''
    args.extra_price = 0


    main(args, trs=trs, tes=args.tes,tee=args.tee, hold_day=1,lenva=30,lente=30)

