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

# Assuming these modules exist and are correctly structured
from data import Query
from loss import mse_loss
import model_pretrain # Contains model definitions and data generation logic

# --- Constants ---
MODEL_DIR = "./models"
LOG_DIR = "./t_log"
BEST_MODEL_FILENAME_TEMPLATE = "best_modelpre_{pre_type}.pkl"
LOG_FILENAME = "log.txt"

# Pre-training types (use constants for clarity)
PRE_TYPE_MARKET = 'market'
PRE_TYPE_COINTEGRATION = 'Cointegration'

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def determine_market_day_label(
    daily_returns: Tensor,
    return_low_threshold: float,
    return_high_threshold: float,
    significant_stock_percent: float
) -> int:
    """
    Determines a market label (0: Up, 1: Down, 2: Neutral) based on daily returns.

    Args:
        daily_returns: Tensor of returns for stocks on a given day. Shape (n_stocks,).
        return_low_threshold: Lower bound for significant negative return.
        return_high_threshold: Upper bound for significant positive return.
        significant_stock_percent: Percentage threshold of stocks needed to declare
                                   a significant up/down day.

    Returns:
        Integer label: 0 (Up), 1 (Down), or 2 (Neutral).
    """
    daily_returns = daily_returns.squeeze() # Ensure 1D
    n_stocks = daily_returns.shape[0]
    if n_stocks == 0:
        return 2 # Neutral if no stocks

    positive_count = torch.sum(daily_returns > return_high_threshold).item()
    negative_count = torch.sum(daily_returns < return_low_threshold).item()

    if (positive_count - negative_count) / n_stocks > significant_stock_percent:
        return 0 # Up day
    elif (negative_count - positive_count) / n_stocks > significant_stock_percent:
        return 1 # Down day
    else:
        return 2 # Neutral day

def calculate_annualized_return(average_daily_return: float, hold_period_days: int) -> float:
    """Calculates annualized return assuming 252 trading days per year."""
    if hold_period_days <= 0:
        return 0.0
    return average_daily_return * 252 / hold_period_days

def non_differentiable_clamp(input_tensor: Tensor, min_val: float, max_val: float) -> Tensor:
    """
    Clamps a tensor within [min_val, max_val] while allowing gradients to flow
    as if the clamp wasn't there (using detach). Useful for parameters like rho.
    """
    return input_tensor.clamp(min=min_val, max=max_val).detach() + input_tensor - input_tensor.detach()

# --- Evaluation Function ---

def evaluate_pretraining_model(
    model_list: List[nn.Module],
    evaluation_data: List[Any], # Type depends on what evaluation_pre returns
    data_query: Query,
    eval_date_codes: List[int],
    stock_id_map: Dict[str, int], # Assuming maps stock code str to an int ID
    stock_info: model_pretrain.stk_dic, # Custom class holding stock info/embeddings
    args: Any, # Argparse namespace or similar
    device: torch.device,
    market_day_params: Optional[Tuple[float, float, float]] = None
) -> Tuple[float, List[Any]]:
    """
    Evaluates the pre-training models on a given date range.

    Args:
        model_list: List containing the models [main_model, market_model, market_pred_model]
                    or [coint_model, coint_model, coint_model].
        evaluation_data: List to store or accumulate evaluation results (structure depends on usage).
        data_query: Initialized Query object for data access.
        eval_date_codes: List of date codes for evaluation.
        stock_id_map: Dictionary mapping stock codes to integer IDs.
        stock_info: Object holding stock dictionary/embeddings.
        args: Configuration arguments.
        device: PyTorch device.
        market_day_params: Tuple (low_thresh, high_thresh, percent_thresh) needed for 'market' type.

    Returns:
        Tuple: (average_score, updated_evaluation_data_list). Score meaning depends on pre_type.
               For 'Cointegration': Lower is better (MSE-based loss).
               For 'market': Higher is better (Accuracy).
    """
    main_model, model_2, model_3 = model_list
    main_model.eval()
    model_2.eval()
    model_3.eval()

    total_samples_or_loss_weight = 0
    cumulative_score_or_loss = 0.0
    current_batch_dates = []

    # Initialize state for Cointegration (if applicable)
    u_prev_list = []
    if args.pre_type == PRE_TYPE_COINTEGRATION:
        num_stocks_total = len(stock_info.stk_dic.keys()) + 2 # +2 for padding/unknown?
        for _ in range(args.batch_pre):
            u_prev_list.append(torch.zeros(num_stocks_total, device=device))

    with torch.no_grad():
        for date_code in eval_date_codes:
            current_batch_dates.append(date_code)

            # Process batch when full or at the end
            is_last_date = (date_code == eval_date_codes[-1])
            if len(current_batch_dates) >= args.batch_pre or (is_last_date and len(current_batch_dates) >= 1): # Allow smaller last batch

                if args.pre_type == PRE_TYPE_COINTEGRATION:
                    # Generate batch data
                    _, _, addi_x_batch, id_batch = \
                        model_pretrain.batch_stk_gen(data_query, current_batch_dates, stock_id_map, stock_info,
                                                     type=args.pre_type)

                    # Model forward pass
                    price_actual, price_pred = main_model(addi_x_batch.to(device), id_batch.to(device))
                    loss1 = mse_loss(price_actual, price_pred) # MSE part

                    # Cointegration loss part
                    u_current = (price_pred - price_actual).squeeze(2) # Residuals (n_batch, n_stocks_in_batch)
                    rho = main_model.rho # Learned mean-reversion speed (n_total_stocks,)
                    loss2 = 0.0
                    batch_size = id_batch.shape[0]

                    for b_idx in range(batch_size):
                        stock_ids_in_sample = id_batch[b_idx, :].to(device) # IDs for this sample
                        u_prev_for_sample = torch.index_select(u_prev_list[b_idx], 0, stock_ids_in_sample)
                        rho_for_sample = torch.index_select(rho, 0, stock_ids_in_sample)
                        diff = u_current[b_idx, :] - u_prev_for_sample * rho_for_sample
                        loss2 += torch.pow(diff, 2).mean()

                        # Update u_prev for the next step (non-differentiable update)
                        u_prev_list[b_idx].scatter_(0, stock_ids_in_sample, u_current[b_idx, :].detach())

                    loss2 /= batch_size
                    batch_loss = loss1 + 0.5 * loss2 # Combine losses
                    cumulative_score_or_loss += batch_loss.item() * batch_size # Weighted sum of loss
                    total_samples_or_loss_weight += batch_size

                elif args.pre_type == PRE_TYPE_MARKET:
                    if market_day_params is None:
                         raise ValueError("market_day_params must be provided for 'market' evaluation.")
                    return_low_thresh, return_high_thresh, percent_stock_thresh = market_day_params

                    # Generate data for pairwise comparison
                    x1_batch, x2_batch, _, id1_batch, id2_batch, _, _, _ = \
                        model_pretrain.batch_stk_gen(data_query, current_batch_dates, stock_id_map,
                                                     stock_info, type='market2') # Assuming 'market2' generates pairs

                    # Pairwise model prediction
                    embed1 = main_model(x1_batch.to(device), id1_batch.to(device))
                    embed2 = main_model(x2_batch.to(device), id2_batch.to(device))
                    pair_score = model_2(embed1, embed2) # Predicts which is 'better'
                    # Assuming label 0 means pair is correctly ordered/similar
                    _, pair_predicted = torch.max(pair_score, 0) # Check prediction dim if needed
                    cumulative_score_or_loss += pair_predicted.eq(0).sum().item() # Count correct pairs

                    # Generate data for market direction prediction (using first day of batch)
                    first_day_code = current_batch_dates[0]
                    x_day, y_day_returns, _, id_day = \
                        model_pretrain.stk_gen_normal(data_query, first_day_code, stock_id_map, stock_info)

                    # Market direction prediction
                    day_embed = main_model(x_day.to(device), id_day.to(device))
                    market_pred_score = model_3(day_embed)
                    _, market_predicted_label = torch.max(market_pred_score, 0) # Check prediction dim

                    # Determine true market label
                    true_market_label = determine_market_day_label(
                        y_day_returns, return_low_thresh, return_high_thresh, percent_stock_thresh
                    )
                    cumulative_score_or_loss += market_predicted_label.eq(true_market_label).sum().item() # Count correct market days
                    total_samples_or_loss_weight += 2 # One for pair, one for market direction

                # Reset batch
                current_batch_dates = []

    if total_samples_or_loss_weight == 0:
        logger.warning("Evaluation completed with zero samples/weight.")
        return 0.0, evaluation_data # Or handle as error

    average_score = cumulative_score_or_loss / total_samples_or_loss_weight
    # logger.info(f"Evaluation: Total Samples/Weight: {total_samples_or_loss_weight}, Cumulative Score/Loss: {cumulative_score_or_loss:.4f}, Average: {average_score:.4f}")

    # evaluation_data list is not modified in this version, return as is
    return average_score, evaluation_data

# --- Main Training Function ---

def main(
    args: Any, # Argparse namespace or similar
    train_start_date_int: int,
    eval_start_date_int: int,
    eval_end_date_int: int,
    val_period_days: int = 30, # Corresponds to original lenva
    test_period_days: int = 30, # Corresponds to original lente
    use_stock_embedding: int = 0, # Corresponds to original use_stk
    run_tag: str = '' # Corresponds to original add_str
):
    """
    Main function to train and evaluate pre-training models.

    Args:
        args: Configuration arguments object.
        train_start_date_int: Start date for training (YYYYMMDD format).
        eval_start_date_int: Start date for validation/testing (YYYYMMDD format).
        eval_end_date_int: End date for testing (YYYYMMDD format).
        val_period_days: Number of days for the validation set.
        test_period_days: Number of days for the test set.
        use_stock_embedding: Flag to indicate usage of stock embeddings (passed to model).
        run_tag: Additional string tag for save directories.
    """
    # --- Setup Directories and Logging ---
    run_hash = str(hash(str(train_start_date_int))) # Simple hash based on start date
    save_subdir = args.add_dir if args.add_dir else run_hash
    run_id = f"{save_subdir}{run_tag}r{random.randint(0, 99):02d}" # Add padding to random int

    work_dir = os.path.join(MODEL_DIR, run_id)
    log_dir = os.path.join(LOG_DIR, run_id)
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, LOG_FILENAME)
    # Remove existing handlers to avoid duplicate logs if function is called multiple times
    for handler in logger.handlers[:]:
        if isinstance(handler, FileHandler):
            logger.removeHandler(handler)
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
    data_query = Query(args.data_dir, input_len=args.input_len, args=args)

    # --- Determine Date Ranges ---
    # Find actual start dates present in the data
    actual_train_start_date = data_query.get_closest_past_date(train_start_date_int)
    actual_eval_start_date = data_query.get_closest_past_date(eval_start_date_int)
    actual_eval_end_date = data_query.get_closest_past_date(eval_end_date_int)

    if not all([actual_train_start_date, actual_eval_start_date, actual_eval_end_date]):
        logger.error("Could not find valid start/end dates in the dataset.")
        return

    # Convert dates to internal codes
    try:
        train_start_code = data_query.day_to_code[actual_train_start_date]
        eval_start_code = data_query.day_to_code[actual_eval_start_date]
        eval_end_code = data_query.day_to_code[actual_eval_end_date]
    except KeyError as e:
        logger.error(f"Date code mapping error: {e}")
        return

    # Define ranges (ensure valid indices)
    # Validation ends 2 days before evaluation start? Original: te_start_num-2
    val_end_code = eval_start_code - 2
    val_start_code = val_end_code - val_period_days
    # Training ends 2 days before validation start? Original: te_start_num-2-lenva-2
    train_end_code = val_start_code - 2

    # Ensure start codes are valid (>= input_len)
    min_valid_code = args.input_len - 1
    train_start_code = max(train_start_code, min_valid_code)
    val_start_code = max(val_start_code, min_valid_code)
    eval_start_code = max(eval_start_code, min_valid_code)

    if train_start_code >= train_end_code or val_start_code >= val_end_code or eval_start_code >= eval_end_code:
         logger.error("Invalid date ranges calculated. Check input dates and data availability.")
         logger.error(f"Train: {train_start_code} -> {train_end_code}")
         logger.error(f"Valid: {val_start_code} -> {val_end_code}")
         logger.error(f"Test:  {eval_start_code} -> {eval_end_code}")
         return

    train_codes = list(range(train_start_code, train_end_code))
    valid_codes = list(range(val_start_code, val_end_code))
    # Test range capped by available data and specified period length
    test_end_code = min(eval_start_code + test_period_days, eval_end_code)
    test_codes = list(range(eval_start_code, test_end_code))

    # Filter ranges based on actually available valid codes from Query object
    valid_query_codes = set(data_query.valid_stocks_per_code.keys())
    train_codes = sorted(list(set(train_codes) & valid_query_codes))
    valid_codes = sorted(list(set(valid_codes) & valid_query_codes))
    test_codes = sorted(list(set(test_codes) & valid_query_codes))

    if not train_codes or not valid_codes or not test_codes:
        logger.error("No valid data found for one or more periods (train/valid/test) after filtering.")
        logger.error(f"Train codes found: {len(train_codes)}")
        logger.error(f"Valid codes found: {len(valid_codes)}")
        logger.error(f"Test codes found: {len(test_codes)}")
        return

    logger.info(f"Train period: {data_query.code_to_day[train_codes[0]]} to {data_query.code_to_day[train_codes[-1]]} ({len(train_codes)} days)")
    logger.info(f"Valid period: {data_query.code_to_day[valid_codes[0]]} to {data_query.code_to_day[valid_codes[-1]]} ({len(valid_codes)} days)")
    logger.info(f"Test period:  {data_query.code_to_day[test_codes[0]]} to {data_query.code_to_day[test_codes[-1]]} ({len(test_codes)} days)")

    # --- Initialize Models ---
    logger.info("Initializing models...")
    stock_info = model_pretrain.stk_dic()
    stock_info.ini_stk_dic(data_query) # Initialize stock dictionary/embeddings
    num_total_stocks = len(stock_info.stk_dic.keys()) + 2 # +2 for padding/unknown?

    # Determine input size based on features
    features = data_query.get_features()
    if args.fea_qlib == 1:
        # Hardcoded assumptions for qlib features - make configurable if possible
        input_size = 6 # Number of base qlib features?
        model_dim_market = 6 * 2 # Dimension for market models
    else:
        input_size = len(features) # Assuming features are directly used
        # Original added +2 - reason unclear, maybe for extra embeddings?
        input_size += 2
        model_dim_market = input_size

    model_list: List[nn.Module] = []
    main_model: nn.Module
    market_model: Optional[nn.Module] = None
    market_pred_model: Optional[nn.Module] = None

    if args.pre_type == PRE_TYPE_MARKET:
        logger.info("Configuring models for 'market' pre-training.")
        # --- Main Embedding Model ---
        main_model = model_pretrain.stk_classification_att1(
            input_size=input_size,
            drop_out=args.dropout_g,
            stk_total=num_total_stocks,
            use_stk=use_stock_embedding
        )
        if args.model_path_pre:
            logger.info(f"Loading pre-trained state for main model from: {args.model_path_pre}")
            if not os.path.exists(args.model_path_pre):
                 logger.error(f"Pre-trained model file not found: {args.model_path_pre}")
                 return
            state = torch.load(args.model_path_pre, map_location=device)
            # Load stock dictionary from saved state if needed (ensure compatibility)
            # stock_info_loaded = model_pretrain.stk_dic()
            # stock_info_loaded.load_(state['stk_dic'])
            # num_total_stocks_loaded = len(stock_info_loaded.stk_dic.keys()) + 2
            # Reinitialize model if stock dict size changed? Original code did this.
            # main_model = model_pretrain.stk_classification_att1(...) # Reinitialize
            main_model.load_state_dict(state['para'], strict=False)
            # main_model.reinitial_stk(...) # If needed
            # main_model.use_stk = 1 # Force usage?

        # --- Market Pairwise Comparison Model ---
        market_model = model_pretrain.stk_classification_small_2(
            dim_model2=model_dim_market, drop_out=args.dropout
        )
        # --- Market Direction Prediction Model ---
        market_pred_model = model_pretrain.stk_marketpred_2(
            dim_model2=model_dim_market, drop_out=args.dropout
        )
        model_list = [main_model, market_model, market_pred_model]

    elif args.pre_type == PRE_TYPE_COINTEGRATION:
        logger.info("Configuring models for 'Cointegration' pre-training.")
        main_model = model_pretrain.stk_pred_small_2(
            stk_total=num_total_stocks, drop_out=args.dropout_g
        )
        if args.model_path_pre:
            logger.info(f"Loading pre-trained state for cointegration model from: {args.model_path_pre}")
            if not os.path.exists(args.model_path_pre):
                 logger.error(f"Pre-trained model file not found: {args.model_path_pre}")
                 return
            state = torch.load(args.model_path_pre, map_location=device)
            # stock_info.load_(state['stk_dic']) # Load stock info if needed
            main_model.load_state_dict(state['para'], strict=False)
        # Use the same model for all slots in model_list for evaluation function compatibility
        model_list = [main_model, main_model, main_model]
    else:
        logger.error(f"Unsupported pre_type: {args.pre_type}")
        return

    # Move models to device
    for m in model_list:
        m.to(device)
    logger.info("Models initialized and moved to device.")

    # --- Optimizer Setup ---
    # Collect unique parameters from all models in the list
    all_params = set()
    for m in model_list:
        all_params.update(m.parameters())

    if args.use_Adam:
        optimizer = torch.optim.Adam(params=list(all_params), lr=args.learning_rate)
        logger.info("Using Adam Optimizer.")
    else:
        optimizer = torch.optim.SGD(params=list(all_params), lr=args.learning_rate, momentum=0.9)
        logger.info("Using SGD Optimizer.")

    # --- Pre-computation for 'market' type ---
    market_day_params: Optional[Tuple[float, float, float]] = None
    if args.pre_type == PRE_TYPE_MARKET:
        logger.info("Calculating return thresholds for market day labeling...")
        all_train_returns = []
        # This part can be slow if train_codes is large, consider optimizing
        for code in train_codes:
             # Use pre-converted tensors if available and efficient
             tensor_data = data_query.get_data_step_tensor(code)
             if tensor_data:
                 _, labels, _ = tensor_data
                 all_train_returns.append(labels.cpu()) # Move to CPU
             else: # Fallback to numpy generation if tensors not ready/available
                 step_data = data_query.get_data_step(code)
                 if step_data:
                     (_, labels_np), _ = step_data
                     all_train_returns.append(torch.from_numpy(labels_np))

        if not all_train_returns:
             logger.error("No training returns found to calculate thresholds.")
             return

        total_returns_tensor = torch.cat(all_train_returns, dim=0)
        if total_returns_tensor.numel() == 0:
             logger.error("Concatenated training returns tensor is empty.")
             return

        # Define thresholds based on quantiles
        percent_return_quantile = 0.4 # Original value
        percent_stock_threshold = 0.6 # Original value
        try:
            return_low_thresh = torch.quantile(total_returns_tensor, percent_return_quantile).item()
            return_high_thresh = torch.quantile(total_returns_tensor, 1.0 - percent_return_quantile).item()
            market_day_params = (return_low_thresh, return_high_thresh, percent_stock_threshold)
            logger.info(f"Market day params: Low={return_low_thresh:.4f}, High={return_high_thresh:.4f}, Stock%={percent_stock_threshold:.2f}")
        except Exception as e:
            logger.error(f"Error calculating return quantiles: {e}")
            return


    # --- Training Loop ---
    best_val_score = -np.inf if args.pre_type == PRE_TYPE_MARKET else np.inf # Market: higher is better, Coint: lower is better
    best_epoch = -1
    train_losses = []
    valid_scores = []
    stock_id_map: Dict[str, int] = {} # Persists across epochs? Seems unused later.

    logger.info(f"Starting training for {args.epoch} epochs...")
    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        epoch_loss_total = 0.0
        epoch_mse_loss_total = 0.0 # Specific for Cointegration
        epoch_samples_processed = 0
        market_label_counts = [0, 0, 0] # For 'market' type stats

        for model_instance in model_list:
            model_instance.train()

        # Shuffle training dates each epoch for market type
        current_train_codes = copy.deepcopy(train_codes)
        if args.pre_type == PRE_TYPE_MARKET:
            random.shuffle(current_train_codes)

        # Initialize state for Cointegration (reset each epoch)
        u_prev_list_train = []
        if args.pre_type == PRE_TYPE_COINTEGRATION:
            num_stocks_total = len(stock_info.stk_dic.keys()) + 2
            for _ in range(args.batch_pre):
                u_prev_list_train.append(torch.zeros(num_stocks_total, device=device))

        # Process training data in batches
        current_batch_dates = []
        batch_data_times = []
        batch_model_times = []
        batch_start_time = time.time()

        for i, date_code in enumerate(current_train_codes):
            current_batch_dates.append(date_code)

            is_last_batch = (i == len(current_train_codes) - 1)
            if len(current_batch_dates) >= args.batch_pre or (is_last_batch and current_batch_dates):
                data_load_start_time = time.time()
                batch_loss = torch.tensor(0.0, device=device)
                loss1 = torch.tensor(0.0, device=device) # For individual loss components
                loss2 = torch.tensor(0.0, device=device)

                # --- Batch Processing Logic ---
                if args.pre_type == PRE_TYPE_COINTEGRATION:
                    # Generate batch data
                    _, _, addi_x_batch, id_batch = \
                        model_pretrain.batch_stk_gen(data_query, current_batch_dates, stock_id_map, stock_info,
                                                     type=args.pre_type)
                    batch_size = id_batch.shape[0]
                    data_load_time = time.time() - data_load_start_time
                    model_fwd_start_time = time.time()

                    # Model forward pass
                    price_actual, price_pred = main_model(addi_x_batch.to(device), id_batch.to(device))
                    loss1 = mse_loss(price_actual, price_pred) # MSE part

                    # Cointegration loss part
                    u_current = (price_pred - price_actual).squeeze(2)
                    # Apply non-differentiable clamp to rho before using in loss
                    rho = non_differentiable_clamp(main_model.rho, -1.0, 1.0)
                    loss2_accum = 0.0

                    for b_idx in range(batch_size):
                        stock_ids_in_sample = id_batch[b_idx, :].to(device)
                        u_prev_for_sample = torch.index_select(u_prev_list_train[b_idx], 0, stock_ids_in_sample)
                        rho_for_sample = torch.index_select(rho, 0, stock_ids_in_sample)
                        diff = u_current[b_idx, :] - u_prev_for_sample * rho_for_sample
                        loss2_accum += torch.pow(diff, 2).mean()

                        # Update u_prev state (non-differentiable)
                        with torch.no_grad():
                            u_prev_list_train[b_idx].scatter_(0, stock_ids_in_sample, u_current[b_idx, :].detach())

                    loss2 = loss2_accum / batch_size
                    batch_loss = loss1 + 0.5 * loss2 # Combine losses
                    epoch_mse_loss_total += loss1.item() * batch_size # Track MSE part

                elif args.pre_type == PRE_TYPE_MARKET:
                    if market_day_params is None: raise ValueError("market_day_params missing") # Should be set
                    return_low_thresh, return_high_thresh, percent_stock_thresh = market_day_params

                    # Generate data for pairwise comparison
                    x1_batch, x2_batch, _, id1_batch, id2_batch, _, _, distance_batch = \
                        model_pretrain.batch_stk_gen(data_query, current_batch_dates, stock_id_map,
                                                     stock_info, type='market2')
                    batch_size = x1_batch.shape[0] # Assuming batch dim is 0
                    data_load_time = time.time() - data_load_start_time
                    model_fwd_start_time = time.time()

                    # Pairwise model prediction
                    embed1 = main_model(x1_batch.to(device), id1_batch.to(device))
                    embed2 = main_model(x2_batch.to(device), id2_batch.to(device))
                    pair_score = market_model(embed1, embed2) # Shape (batch_size, num_classes?)
                    pair_score = pair_score.reshape(batch_size, -1) # Ensure 2D for cross_entropy

                    # Apply distance weighting (inverse distance?)
                    distance_batch = distance_batch.to(device).reshape(batch_size, -1)
                    # Avoid division by zero, add epsilon or clamp distance
                    weighted_pair_score = pair_score / (distance_batch + 1e-6)

                    # Assuming target label for correct pair is 0
                    pair_target = torch.zeros(batch_size, dtype=torch.long, device=device)
                    loss1 = torch.nn.functional.cross_entropy(weighted_pair_score, pair_target)

                    # Market direction prediction (using first day of batch)
                    first_day_code = current_batch_dates[0]
                    x_day, y_day_returns, _, id_day = \
                        model_pretrain.stk_gen_normal(data_query, first_day_code, stock_id_map, stock_info)

                    day_embed = main_model(x_day.to(device), id_day.to(device))
                    market_pred_score = market_pred_model(day_embed) # Shape (batch_size, num_market_classes)
                    market_pred_score = market_pred_score.reshape(x_day.shape[0], -1) # Ensure 2D

                    # Determine true market label
                    true_market_label = determine_market_day_label(
                        y_day_returns, return_low_thresh, return_high_thresh, percent_stock_thresh
                    )
                    market_label_counts[true_market_label] += 1 # Track label distribution
                    market_target = torch.full((x_day.shape[0],), true_market_label, dtype=torch.long, device=device)
                    loss2 = torch.nn.functional.cross_entropy(market_pred_score, market_target)

                    batch_loss = loss1 + loss2 # Combine losses

                # --- Backpropagation ---
                optimizer.zero_grad()
                batch_loss.backward()
                # Gradient clipping
                nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
                optimizer.step()

                model_fwd_bwd_time = time.time() - model_fwd_start_time
                batch_data_times.append(data_load_time)
                batch_model_times.append(model_fwd_bwd_time)

                epoch_loss_total += batch_loss.item() * batch_size
                epoch_samples_processed += batch_size

                # Reset batch
                current_batch_dates = []
                batch_start_time = time.time() # Reset timer for next batch data load

        # --- End of Epoch ---
        epoch_duration = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss_total / epoch_samples_processed if epoch_samples_processed > 0 else 0.0
        avg_mse_loss = epoch_mse_loss_total / epoch_samples_processed if epoch_samples_processed > 0 else 0.0 # For Coint
        train_losses.append(avg_epoch_loss)

        log_msg = f"Epoch: {epoch:03d} | Duration: {epoch_duration:.2f}s"
        if batch_data_times: log_msg += f" | Avg Data Time: {np.mean(batch_data_times):.3f}s"
        if batch_model_times: log_msg += f" | Avg Model Time: {np.mean(batch_model_times):.3f}s"

        if args.pre_type == PRE_TYPE_COINTEGRATION:
             rho_abs_mean = torch.mean(torch.abs(main_model.rho)).item()
             log_msg += f" | Train Loss: {avg_epoch_loss:.4f} (MSE Part: {avg_mse_loss:.4f}) | Avg Abs Rho: {rho_abs_mean:.4f}"
        else: # Market
             log_msg += f" | Train Loss: {avg_epoch_loss:.4f}"
             log_msg += f" | Market Labels (U/D/N): {market_label_counts[0]}/{market_label_counts[1]}/{market_label_counts[2]}"

        logger.info(log_msg)

        # --- Evaluation ---
        eval_start_time = time.time()
        # Pass empty lists for evaluation data accumulation if needed, otherwise ignored
        valid_score, _ = evaluate_pretraining_model(
            model_list, [], data_query, valid_codes, stock_id_map, stock_info, args, device, market_day_params
        )
        test_score, _ = evaluate_pretraining_model(
            model_list, [], data_query, test_codes, stock_id_map, stock_info, args, device, market_day_params
        )
        eval_duration = time.time() - eval_start_time
        valid_scores.append(valid_score)

        logger.info(f"Epoch: {epoch:03d} | Eval Duration: {eval_duration:.2f}s | Valid Score: {valid_score:.4f} | Test Score: {test_score:.4f}")

        # --- Model Saving ---
        current_score_is_better = False
        if args.pre_type == PRE_TYPE_MARKET and valid_score > best_val_score:
            current_score_is_better = True
        elif args.pre_type == PRE_TYPE_COINTEGRATION and valid_score < best_val_score: # Lower loss is better
            current_score_is_better = True

        if current_score_is_better:
            best_val_score = valid_score
            best_epoch = epoch
            logger.info(f"*** New best validation score found at epoch {epoch}. Saving model... ***")
            save_path = os.path.join(work_dir, BEST_MODEL_FILENAME_TEMPLATE.format(pre_type=args.pre_type))
            try:
                # Prepare state dictionary
                state_to_save = {
                    'para': model_list[0].state_dict(), # Main model parameters
                    'stk_dic': stock_info.stk_dic, # Stock dictionary/embeddings
                    'epoch': epoch,
                    'best_val_score': best_val_score,
                    'args': vars(args) # Save args for reproducibility
                }
                # Add parameters for other models if they exist
                if len(model_list) > 1 and model_list[1] is not None:
                    state_to_save['para_2'] = model_list[1].state_dict()
                if len(model_list) > 2 and model_list[2] is not None:
                    state_to_save['para_3'] = model_list[2].state_dict() # Assuming model_3 corresponds to para_3

                torch.save(state_to_save, save_path)
                logger.info(f"Model saved to: {save_path}")
            except Exception as e:
                logger.error(f"Error saving model: {e}", exc_info=True)
        else:
             logger.info(f"Validation score did not improve from best: {best_val_score:.4f} (at epoch {best_epoch})")


    logger.info("--- Training Finished ---")
    logger.info(f"Best Validation Score: {best_val_score:.4f} at Epoch: {best_epoch}")
    best_model_path = os.path.join(work_dir, BEST_MODEL_FILENAME_TEMPLATE.format(pre_type=args.pre_type))
    logger.info(f"Best model saved at: {best_model_path}")

    # Optional: Print final lists for debugging/analysis
    # print('Final Validation Scores:', valid_scores)
    # print('Final Training Losses:', train_losses)

    # Close the file handler
    logger.removeHandler(fh)
    fh.close()


# --- Main Execution Block ---

if __name__ == '__main__':
    # It's better practice to parse arguments here rather than importing a pre-parsed 'args'
    # Example using argparse:
    import argparse
    parser = argparse.ArgumentParser(description="Factor Learning Pre-training")

    # Add arguments based on the original 'arguments.py' structure
    # Basic Training Settings
    parser.add_argument('--epoch', type=int, default=50, help='Number of training epochs.') # Reduced default for testing
    parser.add_argument('--batch_pre', type=int, default=4, help='Batch size for pre-training.')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--use_Adam', type=bool, default=True, help='Use Adam optimizer.')

    # Model Architecture
    parser.add_argument('--input_len', type=int, default=10, help='Input sequence length.') # Example value
    parser.add_argument('--dropout_g', type=float, default=0.5, help='Dropout for main model.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout for other models.')

    # Data Handling & Paths
    parser.add_argument('--data_dir', type=str, required=True, help='Directory for main data Parquet file.')
    parser.add_argument('--extra_data_dir', type=str, default=None, help='Directory for extra data Parquet file.')
    parser.add_argument('--add_dir', type=str, default=None, help='Subdirectory for saving models/logs.')
    parser.add_argument('--model_path_pre', type=str, default=None, help='Path to load pre-trained model state.')

    # Pre-training Specific
    parser.add_argument('--pre_type', type=str, required=True, choices=[PRE_TYPE_MARKET, PRE_TYPE_COINTEGRATION], help='Pre-training type.')
    parser.add_argument('--fea_norm', type=int, default=0, help='Flag for feature normalization in Query.')
    parser.add_argument('--fea_qlib', type=int, default=1, help='Flag for Qlib feature processing in Query.')
    parser.add_argument('--extra_price', type=int, default=1, help='Flag for extra price data processing in Query.')

    # Add other necessary arguments from the original arguments.py if needed

    parsed_args = parser.parse_args()

    # --- Set Run Parameters ---
    # These should ideally also be arguments or loaded from a config file
    train_start = 20060110
    eval_start = 20180301
    eval_end = 20230301
    use_stock_emb = 1 # Use stock embeddings
    validation_days = 30
    test_days = 30

    # --- Run Main Function ---
    try:
        main(
            args=parsed_args,
            train_start_date_int=train_start,
            eval_start_date_int=eval_start,
            eval_end_date_int=eval_end,
            val_period_days=validation_days,
            test_period_days=test_days,
            use_stock_embedding=use_stock_emb,
            run_tag=f"_{parsed_args.pre_type}" # Add pre_type to run tag
        )
    except Exception as e:
        logger.error("An error occurred during execution.", exc_info=True)
        # Ensure log handler is closed even on error
        for handler in logger.handlers[:]:
            if isinstance(handler, FileHandler):
                 handler.close()
                 logger.removeHandler(handler)
