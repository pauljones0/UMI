

from datetime import date as Date
from typing import List, Optional
import distutils.util


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--batch_pre', type=int, default=4)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--input_len', type=int, default=1)
parser.add_argument('--model', type=str, default='Transformer')
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--dim_ff', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--dropout_g', type=float, default=0.5)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument("--use_Adam", default=True)
parser.add_argument('--loss', type=str, default='MSEIC')
parser.add_argument('--gpu', type=int, default=4)

parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--tr_start_num', type=int, default=0)
parser.add_argument('--tr_end_num', type=int, default=0)
parser.add_argument('--vas', type=int, default=0)
parser.add_argument('--vae', type=int, default=0)
parser.add_argument('--tes', type=int, default=0)
parser.add_argument('--tee', type=int, default=0)
parser.add_argument('--add_dir', type=str, default='./')

parser.add_argument('--valid_return', type=int, default=1)
parser.add_argument('--valid_return_l10', type=int, default=0)






parser.add_argument('--pre_type', type=str, default=None)
parser.add_argument('--dim_model', type=int, default=64)
parser.add_argument('--model_path_pre', type=str, default=None)
parser.add_argument('--model_path_pre_mask', type=str, default=None)
parser.add_argument('--model_path_pre_market', type=str, default=None)




parser.add_argument('--fea_norm', type=int, default=0)
parser.add_argument('--fea_qlib', type=int, default=1)


parser.add_argument('--extra_data_dir', type=str, default=None)
parser.add_argument('--extra_price', type=int, default=1)




parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--signalout_path', type=str, default=None)
args = parser.parse_args()