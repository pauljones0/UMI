import logging
import random
import os
import time
import copy
import math
import pickle as pk
from typing import List, Dict, Tuple, Any, Optional
from logging import FileHandler

import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
import scipy.stats as stats

# Assuming these modules exist and are correctly structured
from data import Query
from loss import ic_loss, mse_loss # Assuming ic_loss takes (scores, y_rank, device)
from utils import adjust_lr, calculate_annualized_return # Assuming calculate_annualized_return moved to utils
import model_pretrain # Contains pre-training models and data generation
import model_seq # Contains the main forecasting sequence model (e.g., Trans)

# --- Constants ---
MODEL_DIR = "./models"
LOG_DIR = "./t_log"
BEST_MODEL_FILENAME = "best_model.pkl"
SIGNAL_FILENAME_TEMPLATE = "signal_out_{period}.parquet" # e.g., signal_out_test.parquet
LOG_FILENAME = "log.txt"

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Evaluation Functions ---

def evaluate_simple(
    model: nn.Module,
    data_query: Query,
    eval_date_codes: List[int],
    loss_fn: callable, # e.g., mse_loss
    device: torch.device
) -> float:
    """
    Performs a simple evaluation using a given loss function (e.g., MSE).

    Args:
        model: The forecasting model to evaluate.
        data_query: Initialized Query object for data access.
        eval_date_codes: List of date codes for evaluation.
        loss_fn: The loss function to use (e.g., mse_loss).
        device: PyTorch device.

    Returns:
        Average loss over the evaluation period.
    """
    model.eval()
    total_loss = 0.0
    num_samples = 0
    with torch.no_grad():
        for date_code in eval_date_codes:
            tensor_data = data_query.get_data_step_tensor(date_code)
            if tensor_data is None:
                logger.warning(f"Skipping evaluation for date code {date_code}: No tensor data found.")
                continue

            x, y, _ = tensor_data # Ignore extra data for simple eval
            if x is None or y is None:
                 logger.warning(f"Skipping evaluation for date code {date_code}: Missing features or labels.")
                 continue

            x, y = x.to(device), y.to(device)
            try:
                scores = model(x).squeeze(1) # Assuming model output needs squeezing
                # Ensure loss_fn signature matches. mse_loss might just need (pred, target)
                # loss = loss_fn(scores, y, device).item() # Original signature for ic_loss?
                loss = loss_fn(scores, y).item() # Assuming standard MSE signature
                total_loss += loss
                num_samples += 1
            except Exception as e:
                logger.error(f"Error during simple evaluation for date code {date_code}: {e}", exc_info=True)

    model.train() # Set back to train mode
    if num_samples == 0:
        logger.warning("Simple evaluation completed with zero valid samples.")
        return np.inf # Return infinity if no samples evaluated
    return total_loss / num_samples


def evaluate_forecasting_metrics(
    forecasting_model: nn.Module,
    pre_market_model: Optional[nn.Module], # Optional pre-trained market model
    data_query: Query,
    eval_date_codes: List[int],
    stock_id_map: Dict[str, int],
    stock_info: model_pretrain.stk_dic,
    device: torch.device,
    args: Any # Configuration arguments
) -> Tuple[float, float, float, float]:
    """
    Evaluates the forecasting model using IC, Rank IC, and top/bottom decile returns.

    Optionally incorporates features/embeddings from a pre-trained market model.

    Args:
        forecasting_model: The main sequence model to evaluate.
        pre_market_model: Optional pre-trained market embedding model.
        data_query: Initialized Query object for data access.
        eval_date_codes: List of date codes for evaluation.
        stock_id_map: Dictionary mapping stock codes to integer IDs.
        stock_info: Object holding stock dictionary/embeddings.
        device: PyTorch device.
        args: Configuration arguments (needed for pre-training paths/flags).

    Returns:
        Tuple containing: (Average IC, Average Rank IC, Average Top 10% Return, Average Bottom 10% Return).
    """
    if pre_market_model:
        pre_market_model.eval()
    forecasting_model.eval()

    ic_list, rank_ic_list, top_10p_return_list, bottom_10p_return_list = [], [], [], []

    with torch.no_grad():
        for date_code in eval_date_codes:
            # Get tensor data (features, labels)
            tensor_data = data_query.get_data_step_tensor(date_code)
            if tensor_data is None:
                logger.warning(f"Skipping metric evaluation for date code {date_code}: No tensor data.")
                continue
            x_seq, y_true, _ = tensor_data # Base sequence features, true returns
            if x_seq is None or y_true is None:
                 logger.warning(f"Skipping metric evaluation for date code {date_code}: Missing base data.")
                 continue

            x_seq, y_true = x_seq.to(device), y_true.to(device)
            batch_size = x_seq.shape[0]
            if batch_size == 0: continue # Skip if no stocks for this day

            # --- Prepare Additional Inputs (if using pre-trained models) ---
            x_more_input = None
            # 1. Mask-based features (if model_path_pre_mask is set)
            # This logic seems specific and assumes addi_x contains mask-related data.
            # Needs clarification or refactoring based on actual data structure.
            # Original code:
            # if args.model_path_pre_mask is not None:
            #     # Requires addi_x from stk_gen_normal, which isn't fetched here by default
            #     # Fetching it separately:
            #     _, _, addi_x_mask, _ = model_pretrain.stk_gen_normal(data_query, date_code, stock_id_map, stock_info)
            #     addi_x_processed = addi_x_mask.squeeze(0)[:, :, 0:1] - addi_x_mask.squeeze(0)[:, :, 1:2]
            #     # Concatenate processed mask features
            #     x_seq = torch.cat([x_seq, addi_x_processed.to(device), addi_x_processed.to(device)], dim=-1) # Duplicated?

            # 2. Market-based features (if pre_market_model is provided)
            if pre_market_model:
                # Generate necessary inputs for the pre_market_model
                x_market_day, _, _, id_day = \
                    model_pretrain.stk_gen_normal(data_query, date_code, stock_id_map, stock_info)

                # Get market embedding and stock-specific outputs
                market_embedding, stock_outputs = pre_market_model(
                    x_market_day.to(device), id_day.to(device), moreout=1 # Assuming moreout=1 returns both
                )
                # Repeat market embedding for each stock in the batch
                market_embedding_repeated = market_embedding.repeat(batch_size, 1)
                x_more_input = (market_embedding_repeated, stock_outputs)
            # --- End of Additional Inputs ---

            try:
                # Get model predictions (scores)
                scores = forecasting_model(x_seq, addi_x=x_more_input).squeeze(1)
                scores_np = scores.cpu().numpy()
                returns_np = y_true.cpu().numpy()

                if len(scores_np) < 2: continue # Need at least 2 points for correlation

                # Calculate metrics
                ic_list.append(stats.pearsonr(scores_np, returns_np)[0])
                rank_ic_list.append(stats.spearmanr(scores_np, returns_np)[0])

                # Top/Bottom decile returns
                num_stocks = len(scores_np)
                top_10p_count = max(1, int(0.1 * num_stocks)) # Ensure at least 1 stock
                bottom_10p_count = max(1, int(0.1 * num_stocks))

                sorted_indices = np.argsort(-scores_np) # Descending sort by score
                top_indices = sorted_indices[:top_10p_count]
                bottom_indices = sorted_indices[-bottom_10p_count:]

                top_10p_return_list.append(np.mean(returns_np[top_indices]))
                bottom_10p_return_list.append(np.mean(returns_np[bottom_indices]))

            except Exception as e:
                logger.error(f"Error during metric evaluation for date code {date_code}: {e}", exc_info=True)


    # Calculate average metrics, handle cases with no valid data
    avg_ic = np.nanmean(ic_list) if ic_list else 0.0
    avg_rank_ic = np.nanmean(rank_ic_list) if rank_ic_list else 0.0
    avg_l1_return = np.nanmean(top_10p_return_list) if top_10p_return_list else 0.0
    avg_l10_return = np.nanmean(bottom_10p_return_list) if bottom_10p_return_list else 0.0

    # Set model(s) back to train mode
    if pre_market_model:
        pre_market_model.train()
    forecasting_model.train()

    return avg_ic, avg_rank_ic, avg_l1_return, avg_l10_return


# --- Main Training Function ---

def main(
    args: Any, # Argparse namespace or similar
    train_start_date_int: int,
    eval_start_date_int: int,
    eval_end_date_int: int,
    hold_period_days: int = 1,
    val_period_days: int = 30,
    test_period_days: int = 30,
    run_tag: str = ''
):
    """
    Main function to train and evaluate the forecasting model.

    Args:
        args: Configuration arguments object.
        train_start_date_int: Start date for training (YYYYMMDD format).
        eval_start_date_int: Start date for validation/testing (YYYYMMDD format).
        eval_end_date_int: End date for testing (YYYYMMDD format).
        hold_period_days: Holding period in days for annualized return calculation.
        val_period_days: Number of days for the validation set.
        test_period_days: Number of days for the test set.
        run_tag: Additional string tag for save directories.
    """
    # --- Setup Directories and Logging ---
    run_hash = str(hash(str(train_start_date_int)))
    save_subdir = args.add_dir if args.add_dir else run_hash
    run_id = f"{save_subdir}{run_tag}r{random.randint(0, 99):02d}"

    work_dir = os.path.join(MODEL_DIR, run_id)
    log_dir = os.path.join(LOG_DIR, run_id)
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, LOG_FILENAME)
    # Setup file handler
    for handler in logger.handlers[:]:
        if isinstance(handler, FileHandler): logger.removeHandler(handler)
    fh = FileHandler(log_path, mode="w")
    log_formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(log_formatter)
    logger.addHandler(fh)

    logger.info(f"Starting run: {run_id}")
    logger.info(f"Working directory: {work_dir}")
    logger.info(f"Log directory: {log_dir}")
    logger.info("Arguments: %s", vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Load Data ---
    logger.info("Initializing data query...")
    try:
        data_query = Query(args.data_dir, input_len=args.input_len, args=args)
    except FileNotFoundError as e:
        logger.error(f"Data loading error: {e}")
        return
    except ValueError as e:
         logger.error(f"Data configuration error: {e}")
         return

    # --- Determine Date Ranges ---
    # (Similar logic as in main_factorlearning.py)
    actual_train_start_date = data_query.get_closest_past_date(train_start_date_int)
    actual_eval_start_date = data_query.get_closest_past_date(eval_start_date_int)
    actual_eval_end_date = data_query.get_closest_past_date(eval_end_date_int)

    if not all([actual_train_start_date, actual_eval_start_date, actual_eval_end_date]):
        logger.error("Could not find valid start/end dates in the dataset.")
        return

    try:
        train_start_code = data_query.day_to_code[actual_train_start_date]
        eval_start_code = data_query.day_to_code[actual_eval_start_date]
        eval_end_code = data_query.day_to_code[actual_eval_end_date]
    except KeyError as e:
        logger.error(f"Date code mapping error: {e}")
        return

    val_end_code = eval_start_code - 2
    val_start_code = val_end_code - val_period_days
    train_end_code = val_start_code - 2
    min_valid_code = args.input_len - 1
    train_start_code = max(train_start_code, min_valid_code)
    val_start_code = max(val_start_code, min_valid_code)
    eval_start_code = max(eval_start_code, min_valid_code)

    if train_start_code >= train_end_code or val_start_code >= val_end_code or eval_start_code >= eval_end_code:
         logger.error("Invalid date ranges calculated.")
         return

    train_codes = list(range(train_start_code, train_end_code))
    valid_codes = list(range(val_start_code, val_end_code))
    test_end_code = min(eval_start_code + test_period_days, eval_end_code)
    test_codes = list(range(eval_start_code, test_end_code))

    valid_query_codes = set(data_query.valid_stocks_per_code.keys())
    train_codes = sorted(list(set(train_codes) & valid_query_codes))
    valid_codes = sorted(list(set(valid_codes) & valid_query_codes))
    test_codes = sorted(list(set(test_codes) & valid_query_codes))

    if not train_codes or not valid_codes or not test_codes:
        logger.error("No valid data found for one or more periods (train/valid/test) after filtering.")
        return

    logger.info(f"Train period: {data_query.code_to_day[train_codes[0]]} to {data_query.code_to_day[train_codes[-1]]} ({len(train_codes)} days)")
    logger.info(f"Valid period: {data_query.code_to_day[valid_codes[0]]} to {data_query.code_to_day[valid_codes[-1]]} ({len(valid_codes)} days)")
    logger.info(f"Test period:  {data_query.code_to_day[test_codes[0]]} to {data_query.code_to_day[test_codes[-1]]} ({len(test_codes)} days)")

    # --- Initialize Models ---
    logger.info("Initializing models...")
    stock_info = model_pretrain.stk_dic()
    stock_info.ini_stk_dic(data_query)
    stock_id_map: Dict[str, int] = {} # Is this actually used/needed?

    # Load Pre-trained Market Model (Optional)
    pre_market_model: Optional[nn.Module] = None
    market_embedding_dim = 0
    if args.model_path_pre_market:
        logger.info(f"Loading pre-trained market model from: {args.model_path_pre_market}")
        if not os.path.exists(args.model_path_pre_market):
            logger.error(f"Pre-trained market model file not found: {args.model_path_pre_market}")
            # Decide whether to continue without it or exit
            # return
        else:
            try:
                # Determine input size for pre_market_model based on args
                if args.fea_qlib == 1:
                    pre_market_input_size = 6
                    market_embedding_dim = 6 * 2
                else:
                    pre_market_input_size = len(data_query.get_features()) + 2
                    market_embedding_dim = pre_market_input_size

                pre_market_model = model_pretrain.stk_classification_att1(
                    input_size=pre_market_input_size,
                    drop_out=args.dropout_g, # Assuming dropout_g is for pre-trained
                    stk_total=len(stock_info.stk_dic.keys()) + 2
                    # use_stk flag might be needed here too
                )
                state_market = torch.load(args.model_path_pre_market, map_location=device)
                pre_market_model.load_state_dict(state_market['para'], strict=False)
                pre_market_model.to(device)
                pre_market_model.eval() # Set to eval mode if only used for inference
                logger.info("Pre-trained market model loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading pre-trained market model: {e}", exc_info=True)
                pre_market_model = None # Failed to load, proceed without it
                market_embedding_dim = 0

    # Load Pre-trained Mask Model Info (if path provided) - Affects input size
    # This part is unclear how it modifies input, original code concatenated features.
    # Assuming it adds a fixed number of features if the path is present.
    mask_feature_dim = 0
    if args.model_path_pre_mask:
        logger.info("Pre-trained mask model path provided. Adjusting input features.")
        # Determine how many features the mask model adds. Original code added 2*addi_x.shape[-1]?
        # Let's assume it adds 2 features based on the original concatenation. Needs verification.
        mask_feature_dim = 2 # Placeholder - adjust based on actual mask feature logic
        logger.warning("Mask feature integration logic requires review based on data structure.")


    # Configure Main Forecasting Model
    seq_len = args.input_len
    input_size = len(data_query.get_features())
    if args.fea_qlib == 1:
        seq_len = 60 # Qlib uses fixed 60 day lookback?
        input_size = 6 # Qlib uses 6 base features?

    # Adjust input size based on pre-trained models used
    final_input_size = input_size + mask_feature_dim
    # Note: market embedding is passed via addi_x, not concatenated here

    logger.info(f"Forecasting model input config: input_size={final_input_size}, seq_len={seq_len}")

    if args.model.lower() == "transformer":
        logger.info("Initializing Transformer model.")
        forecasting_model = model_seq.Trans(
            input_size=final_input_size,
            num_heads=args.num_heads,
            dim_model=args.hidden_size,
            dim_ff=args.dim_ff,
            seq_len=seq_len,
            num_layers=args.num_layers,
            dropout=args.dropout,
            add_xdim=market_embedding_dim, # Pass market embedding dim here
            embeddim=6 # Original hardcoded value - what does this represent? Stock type embedding?
        )
    # Add elif for other model types (LSTM, GRU, etc.) if needed
    # elif args.model.lower() == "lstm":
    #     forecasting_model = ...
    else:
        logger.error(f"Unsupported model type: {args.model}")
        return

    forecasting_model.to(device)
    logger.info("Forecasting model initialized.")

    # --- Optimizer Setup ---
    forecasting_params = list(forecasting_model.parameters())
    optimizer_params = [{'params': forecasting_params, 'lr': args.learning_rate}]

    # Optionally add pre-trained model params with lower LR if fine-tuning
    if pre_market_model and args.finetune_pretrain: # Add a new arg 'finetune_pretrain'
        logger.info("Adding pre-trained market model parameters to optimizer with reduced LR.")
        optimizer_params.append({'params': pre_market_model.parameters(), 'lr': args.learning_rate * 0.1}) # Example: 1/10th LR

    if args.use_Adam:
        optimizer = torch.optim.Adam(optimizer_params)
        logger.info("Using Adam Optimizer.")
    else:
        optimizer = torch.optim.SGD(optimizer_params, momentum=0.9) # Note: SGD might need separate LR for param groups
        logger.info("Using SGD Optimizer.")


    # --- Training Loop ---
    best_val_metric = -np.inf # Assuming higher IC/Return is better
    best_epoch = -1
    all_results = [] # To store results per epoch

    logger.info(f"Starting training for {args.epoch} epochs...")
    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        epoch_loss_total = 0.0
        epoch_samples_processed = 0

        # Set models to training mode
        forecasting_model.train()
        if pre_market_model and args.finetune_pretrain:
            pre_market_model.train()
        elif pre_market_model:
             pre_market_model.eval() # Keep pre-trained in eval if not fine-tuning

        # Shuffle training data
        current_train_codes = copy.deepcopy(train_codes)
        random.shuffle(current_train_codes)

        # Timing metrics
        batch_data_times, batch_device_times, batch_model_fwd_times, batch_loss_bwd_times, batch_step_times = [], [], [], [], []

        for date_code in current_train_codes:
            batch_start_time = time.time()

            # --- Get Data ---
            tensor_data = data_query.get_data_step_tensor(date_code)
            if tensor_data is None: continue
            x_seq, y_true, _ = tensor_data
            if x_seq is None or y_true is None: continue
            batch_size = x_seq.shape[0]
            if batch_size == 0: continue

            data_load_time = time.time() - batch_start_time
            device_start_time = time.time()

            x_seq, y_true = x_seq.to(device), y_true.to(device)

            # --- Prepare Labels for Loss ---
            # 1. Standardized true returns (for MSE part)
            y_std = (y_true - torch.mean(y_true)) / (torch.std(y_true) + 1e-8)
            # 2. Ranked true returns (for Rank IC part)
            y_rank = torch.argsort(torch.argsort(y_true)).float() # Convert ranks to float
            y_rank_std = (y_rank - torch.mean(y_rank)) / (torch.std(y_rank) + 1e-8)

            device_time = time.time() - device_start_time
            model_fwd_start_time = time.time()

            # --- Prepare Additional Inputs ---
            x_more_input = None
            # Mask features (Placeholder - requires clarification)
            # if args.model_path_pre_mask:
            #     # Fetch/calculate mask features and concatenate to x_seq
            #     pass # x_seq = torch.cat(...)

            # Market features
            if pre_market_model:
                x_market_day, _, _, id_day = \
                    model_pretrain.stk_gen_normal(data_query, date_code, stock_id_map, stock_info)
                with torch.set_grad_enabled(args.finetune_pretrain): # Enable grads only if fine-tuning
                    market_embedding, stock_outputs = pre_market_model(
                        x_market_day.to(device), id_day.to(device), moreout=1
                    )
                market_embedding_repeated = market_embedding.repeat(batch_size, 1)
                x_more_input = (market_embedding_repeated, stock_outputs)
            # --- End Additional Inputs ---

            # --- Forward Pass ---
            scores = forecasting_model(x_seq, addi_x=x_more_input).squeeze(1)
            model_fwd_time = time.time() - model_fwd_start_time
            loss_bwd_start_time = time.time()

            # --- Loss Calculation ---
            # Combine MSE loss on standardized returns and Rank IC loss on standardized ranks
            loss_mse = torch.nn.functional.mse_loss(scores, y_std)
            loss_rank_ic = ic_loss(scores, y_rank_std, device) # Assumes ic_loss returns negative weighted correlation
            # Adjust weighting factor (0.1) as needed
            loss = loss_mse + 0.1 * loss_rank_ic

            # --- Backward Pass & Optimization ---
            optimizer.zero_grad()
            loss.backward()
            loss_bwd_time = time.time() - loss_bwd_start_time
            step_start_time = time.time()

            nn.utils.clip_grad_norm_(forecasting_params, max_norm=5.0) # Clip only forecasting model params?
            if pre_market_model and args.finetune_pretrain:
                 nn.utils.clip_grad_norm_(pre_market_model.parameters(), max_norm=5.0) # Clip pre-trained if fine-tuning

            optimizer.step()
            step_time = time.time() - step_start_time

            # --- Record Metrics ---
            epoch_loss_total += loss.item() * batch_size
            epoch_samples_processed += batch_size
            batch_data_times.append(data_load_time)
            batch_device_times.append(device_time)
            batch_model_fwd_times.append(model_fwd_time)
            batch_loss_bwd_times.append(loss_bwd_time)
            batch_step_times.append(step_time)

        # --- End of Epoch ---
        epoch_duration = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss_total / epoch_samples_processed if epoch_samples_processed > 0 else 0.0

        logger.info(
            f"Epoch: {epoch:03d} | Duration: {epoch_duration:.2f}s | Avg Loss: {avg_epoch_loss:.4f} | "
            f"Timings (Avg ms): Data={np.mean(batch_data_times)*1000:.1f} "
            f"Device={np.mean(batch_device_times)*1000:.1f} Fwd={np.mean(batch_model_fwd_times)*1000:.1f} "
            f"Bwd={np.mean(batch_loss_bwd_times)*1000:.1f} Step={np.mean(batch_step_times)*1000:.1f}"
        )

        # --- Evaluation ---
        eval_start_time = time.time()
        ic_va, rank_ic_va, l1_return_va, l10_return_va = evaluate_forecasting_metrics(
            forecasting_model, pre_market_model, data_query, valid_codes, stock_id_map, stock_info, device, args
        )
        ic_te, rank_ic_te, l1_return_te, l10_return_te = evaluate_forecasting_metrics(
            forecasting_model, pre_market_model, data_query, test_codes, stock_id_map, stock_info, device, args
        )
        eval_duration = time.time() - eval_start_time

        # --- Log Evaluation Metrics ---
        annual_va = calculate_annualized_return(l1_return_va, hold_period_days)
        annual_te = calculate_annualized_return(l1_return_te, hold_period_days)
        annual_va_l10 = calculate_annualized_return(l10_return_va, hold_period_days)
        annual_te_l10 = calculate_annualized_return(l10_return_te, hold_period_days)

        logger.info(
            f"Epoch: {epoch:03d} | Eval Duration: {eval_duration:.2f}s | "
            f"IC (Va/Te): {ic_va:.4f}/{ic_te:.4f} | RankIC (Va/Te): {rank_ic_va:.4f}/{rank_ic_te:.4f} | "
            f"AnnRet L1 (Va/Te): {annual_va:.4f}/{annual_te:.4f} | AnnRet L10 (Va/Te): {annual_va_l10:.4f}/{annual_te_l10:.4f}"
        )

        # --- Determine Validation Metric for Saving ---
        # Choose metric based on args (default to IC)
        if args.valid_return == 1:
            current_val_metric = l1_return_va
        elif args.valid_return_l10 == 1:
            current_val_metric = -l10_return_va # Higher negative return (less loss) is better
        else:
            current_val_metric = ic_va

        # --- Model Saving ---
        if current_val_metric > best_val_metric:
            best_val_metric = current_val_metric
            best_epoch = epoch
            logger.info(f"*** New best validation metric found ({current_val_metric:.4f}). Saving model... ***")
            save_path = os.path.join(work_dir, BEST_MODEL_FILENAME)
            try:
                state_to_save = {
                    'forecasting_model_state_dict': forecasting_model.state_dict(),
                    'stock_info_dict': stock_info.stk_dic, # Save stock info used
                    'epoch': epoch,
                    'best_val_metric': best_val_metric,
                    'args': vars(args)
                }
                if pre_market_model:
                    # Save pre-market model state only if it was loaded/used
                    state_to_save['pre_market_model_state_dict'] = pre_market_model.state_dict()

                torch.save(state_to_save, save_path)
                logger.info(f"Model saved to: {save_path}")
            except Exception as e:
                logger.error(f"Error saving model: {e}", exc_info=True)
        else:
             logger.info(f"Validation metric did not improve from best: {best_val_metric:.4f} (at epoch {best_epoch})")

        # Adjust Learning Rate
        adjust_lr(optimizer, epoch, args.learning_rate) # Pass the optimizer

        # Store results for potential later analysis
        epoch_results = {
            'epoch': epoch, 'train_loss': avg_epoch_loss,
            'ic_va': ic_va, 'rank_ic_va': rank_ic_va, 'l1_ret_va': l1_return_va, 'l10_ret_va': l10_return_va, 'ann_ret_va': annual_va,
            'ic_te': ic_te, 'rank_ic_te': rank_ic_te, 'l1_ret_te': l1_return_te, 'l10_ret_te': l10_return_te, 'ann_ret_te': annual_te,
            'current_val_metric': current_val_metric
        }
        all_results.append(epoch_results)


    logger.info("--- Training Finished ---")
    logger.info(f"Best Validation Metric: {best_val_metric:.4f} at Epoch: {best_epoch}")
    best_model_path = os.path.join(work_dir, BEST_MODEL_FILENAME)
    logger.info(f"Best model saved at: {best_model_path}")

    # --- Generate Signals using Best Model ---
    logger.info("Loading best model for signal generation...")
    if not os.path.exists(best_model_path):
        logger.error("Best model file not found. Cannot generate signals.")
    else:
        try:
            state = torch.load(best_model_path, map_location=device)
            # Re-initialize models before loading state
            # (Ensure architecture matches saved state)
            # ... [Re-init forecasting_model and pre_market_model if needed] ...
            forecasting_model.load_state_dict(state['forecasting_model_state_dict'])
            if 'pre_market_model_state_dict' in state and pre_market_model:
                pre_market_model.load_state_dict(state['pre_market_model_state_dict'])
            logger.info("Best model loaded.")

            signal_dfs = []
            forecasting_model.eval()
            if pre_market_model: pre_market_model.eval()

            logger.info("Generating signals for the test period...")
            with torch.no_grad():
                for date_code in test_codes:
                    tensor_data = data_query.get_data_step_tensor(date_code)
                    if tensor_data is None: continue
                    x_seq, _, _ = tensor_data
                    if x_seq is None: continue
                    batch_size = x_seq.shape[0]
                    if batch_size == 0: continue

                    x_seq = x_seq.to(device)
                    stock_list = data_query.valid_stocks_per_code.get(date_code) # Get stocks for this day
                    if not stock_list: continue

                    # Prepare additional inputs (same logic as in evaluation)
                    x_more_input = None
                    if pre_market_model:
                        x_market_day, _, _, id_day = model_pretrain.stk_gen_normal(data_query, date_code, stock_id_map, stock_info)
                        market_embedding, stock_outputs = pre_market_model(x_market_day.to(device), id_day.to(device), moreout=1)
                        market_embedding_repeated = market_embedding.repeat(batch_size, 1)
                        x_more_input = (market_embedding_repeated, stock_outputs)

                    # Get scores
                    scores = forecasting_model(x_seq, addi_x=x_more_input).squeeze(1)
                    scores_np = scores.cpu().numpy()

                    # Create DataFrame for the day
                    daily_df = pd.DataFrame({'Value': scores_np, data_query.STOCK_COL: stock_list}) # Use constant
                    daily_df[data_query.DATE_COL] = data_query.code_to_day[date_code] # Use constant
                    # Add other required columns (Time, Ind_Name) - potentially from original data
                    daily_df["Time"] = 83000 # Hardcoded time?
                    daily_df["Ind_Name"] = "None" # Hardcoded industry?

                    # Merge with original factor data if needed (optional)
                    # original_df = data_query.factor_data[date_code].reset_index()
                    # daily_df = pd.merge(daily_df, original_df[[DATE_COL, STOCK_COL, ...]], on=[DATE_COL, STOCK_COL], how='left')

                    signal_dfs.append(daily_df[[data_query.DATE_COL, data_query.STOCK_COL, "Time", "Ind_Name", "Value"]]) # Use constants

            if signal_dfs:
                final_signal_df = pd.concat(signal_dfs, axis=0, ignore_index=True)
                signal_filename = SIGNAL_FILENAME_TEMPLATE.format(period="test")
                signal_save_path = os.path.join(work_dir, signal_filename)
                try:
                    final_signal_df.to_parquet(signal_save_path, index=False)
                    logger.info(f"Signals saved to: {signal_save_path}")
                    logger.info(f"Signal DataFrame head:\n{final_signal_df.head()}")
                except Exception as e:
                    logger.error(f"Error saving signal DataFrame: {e}", exc_info=True)
            else:
                logger.warning("No signals generated for the test period.")

        except Exception as e:
            logger.error(f"Error loading best model or generating signals: {e}", exc_info=True)

    # Save all epoch results (optional)
    results_save_path = os.path.join(log_dir, "all_epoch_results.pkl")
    try:
        with open(results_save_path, 'wb') as f:
            pk.dump(all_results, f)
        logger.info(f"Full epoch results saved to: {results_save_path}")
    except Exception as e:
        logger.error(f"Error saving epoch results: {e}", exc_info=True)


    # Close the file handler
    logger.removeHandler(fh)
    fh.close()

    return best_val_metric # Return the best validation metric achieved


# --- Main Execution Block ---

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Stock Return Forecasting Training")

    # Add arguments based on the original 'arguments.py' and function signature
    # Basic Training Settings
    parser.add_argument('--epoch', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--use_Adam', type=bool, default=True, help='Use Adam optimizer.')

    # Model Architecture (Forecasting Model)
    parser.add_argument('--model', type=str, default="transformer", help='Forecasting model type (e.g., transformer).')
    parser.add_argument('--input_len', type=int, default=10, help='Input sequence length for features.')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size (dim_model) for Transformer.')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads for Transformer.')
    parser.add_argument('--dim_ff', type=int, default=64, help='Dimension of feed-forward layer for Transformer.')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers for Transformer.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for forecasting model.')
    parser.add_argument('--dropout_g', type=float, default=0.5, help='Dropout rate for pre-trained model (if used).') # Name clarification needed

    # Data Handling & Paths
    parser.add_argument('--data_dir', type=str, required=True, help='Directory for main data Parquet file.')
    parser.add_argument('--extra_data_dir', type=str, default=None, help='Directory for extra data (used by Query).')
    parser.add_argument('--add_dir', type=str, default=None, help='Subdirectory for saving models/logs.')
    parser.add_argument('--model_path_pre_market', type=str, default=None, help='Path to load pre-trained market model state.')
    parser.add_argument('--model_path_pre_mask', type=str, default=None, help='Path related to pre-trained mask features (logic needs review).')

    # Feature/Data Flags (used by Query)
    parser.add_argument('--fea_norm', type=int, default=0, help='Flag for feature normalization in Query.')
    parser.add_argument('--fea_qlib', type=int, default=1, help='Flag for Qlib feature processing in Query.')
    parser.add_argument('--extra_price', type=int, default=1, help='Flag for extra price data processing in Query.')

    # Evaluation Metric Choice
    parser.add_argument('--valid_return', type=int, default=0, help='Use L1 (top 10%) return as validation metric if 1.')
    parser.add_argument('--valid_return_l10', type=int, default=0, help='Use -L10 (bottom 10%) return as validation metric if 1.')
    # Add finetune_pretrain flag
    parser.add_argument('--finetune_pretrain', action='store_true', help='Fine-tune the pre-trained market model during training.')


    # Add other necessary arguments...

    parsed_args = parser.parse_args()

    # --- Set Run Parameters ---
    # Should ideally be arguments or from config
    train_start = 20060110
    eval_start = 20180301
    eval_end = 20230301
    hold_days = 1
    validation_days = 30
    test_days = 30

    # Set CUDA device visibility (optional, often managed externally)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # --- Run Main Function ---
    try:
        main(
            args=parsed_args,
            train_start_date_int=train_start,
            eval_start_date_int=eval_start,
            eval_end_date_int=eval_end,
            hold_period_days=hold_days,
            val_period_days=validation_days,
            test_period_days=test_days,
            run_tag=f"_{parsed_args.model}" # Add model type to run tag
        )
    except Exception as e:
        logger.error("An error occurred during execution.", exc_info=True)
        # Ensure log handler is closed even on error
        for handler in logger.handlers[:]:
            if isinstance(handler, FileHandler):
                 handler.close()
                 logger.removeHandler(handler)
