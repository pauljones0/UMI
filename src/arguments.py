import argparse

# --- Basic Training Settings ---
parser = argparse.ArgumentParser(description='Model Training Arguments')
parser.add_argument('--epoch', type=int, default=200, help='Number of training epochs.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
parser.add_argument('--batch_pre', type=int, default=4, help='Batch size for pre-training (if applicable).')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer.')
parser.add_argument('--use_Adam', type=bool, default=True, help='Whether to use the Adam optimizer.') # Note: Changed type to bool for clarity, default=True implies boolean
parser.add_argument('--loss', type=str, default='MSEIC', help='Loss function to use.')
parser.add_argument('--gpu', type=int, default=4, help='GPU ID to use (if available).')

# --- Model Architecture ---
parser.add_argument('--model', type=str, default='Transformer', help='Type of model architecture.')
parser.add_argument('--input_len', type=int, default=1, help='Length of the input sequence.')
parser.add_argument('--hidden_size', type=int, default=64, help='Size of hidden layers.')
parser.add_argument('--dim_ff', type=int, default=64, help='Dimension of the feed-forward layer (often same as hidden_size).')
parser.add_argument('--dim_model', type=int, default=64, help='Dimension of the model (e.g., embedding dimension).')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--dropout_g', type=float, default=0.5, help='Graph dropout rate (if applicable).')
parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in the model.')
parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads (for Transformer).')

# --- Data Handling & Paths ---
parser.add_argument('--data_dir', type=str, default='', help='Directory containing the main dataset.')
parser.add_argument('--tr_start_num', type=int, default=0, help='Start index for training data.')
parser.add_argument('--tr_end_num', type=int, default=0, help='End index for training data.')
parser.add_argument('--vas', type=int, default=0, help='Start index for validation data.')
parser.add_argument('--vae', type=int, default=0, help='End index for validation data.')
parser.add_argument('--tes', type=int, default=0, help='Start index for test data.')
parser.add_argument('--tee', type=int, default=0, help='End index for test data.')
parser.add_argument('--add_dir', type=str, default='./', help='Additional directory path.')
parser.add_argument('--extra_data_dir', type=str, default=None, help='Directory for extra data (optional).')
parser.add_argument('--extra_price', type=int, default=1, help='Flag related to extra price data.')
parser.add_argument('--fea_norm', type=int, default=0, help='Flag for feature normalization.')
parser.add_argument('--fea_qlib', type=int, default=1, help='Flag related to Qlib features.')

# --- Pre-training Settings ---
parser.add_argument('--pre_type', type=str, default=None, help='Type of pre-training.')
parser.add_argument('--model_path_pre', type=str, default=None, help='Path to the general pre-trained model.')
parser.add_argument('--model_path_pre_mask', type=str, default=None, help='Path to the pre-trained mask model.')
parser.add_argument('--model_path_pre_market', type=str, default=None, help='Path to the pre-trained market model.')

# --- Validation & Output ---
parser.add_argument('--valid_return', type=int, default=1, help='Flag for validation return calculation.')
parser.add_argument('--valid_return_l10', type=int, default=0, help='Flag for validation return calculation (last 10).')
parser.add_argument('--model_path', type=str, default=None, help='Path to save/load the trained model.')
parser.add_argument('--signalout_path', type=str, default=None, help='Path to save output signals.')

# --- Parse Arguments ---
args = parser.parse_args()

# Example of accessing arguments:
# print(f"Training epochs: {args.epoch}")
# print(f"Model type: {args.model}")