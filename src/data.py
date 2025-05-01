import logging
import os
from multiprocessing.dummy import Pool as ThreadPool
from typing import List, Dict, Optional, Tuple, Any, Type

import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from torch import Tensor

# Assuming df_to_grouped_dict is the consolidated function from utils.py
from utils import df_to_grouped_dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s   %(levelname)s   %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__) # Use __name__ for logger

# Constants for column names (makes code easier to refactor)
DATE_COL = 'Date'
STOCK_COL = 'StkCode'
# Assuming the last column is label and second to last is another identifier?
# Adjust these based on actual DataFrame structure if needed.
LABEL_COL_INDEX = -1
FEATURES_START_INDEX = 1 # Assuming first column is Date or StkCode before reset_index
FEATURES_END_INDEX = -2

class Query:
    """
    Handles loading, processing, and querying of stock factor data.

    Manages time series data for multiple stocks, potentially merging
    with auxiliary data and providing methods to retrieve data slices
    as NumPy arrays or PyTorch tensors.
    """
    def __init__(self,
                 data_dir: str,
                 input_len: int = 1,
                 args: Optional[Any] = None): # Use Any or a specific Namespace type
        """
        Initializes the Query object.

        Args:
            data_dir: Path to the main Parquet data file.
            input_len: Length of the input sequence (number of past days). Default: 1.
            args: Arguments object (e.g., from argparse) containing configuration
                  like 'extra_data_dir', 'fea_norm', 'fea_qlib', 'extra_price'.
        """
        if args is None:
            raise ValueError("args object must be provided for configuration.")
        self.args = args
        self.input_len = input_len
        self.factor_tensor: Dict[int, Tensor] = {}
        self.label_tensor: Dict[int, Tensor] = {}
        self.extra_data_tensor: Dict[int, Tensor] = {}
        self.extra_data_dict: Optional[Dict[int, pd.DataFrame]] = None
        self.features: List[str] = []
        self.label_name: str = ""

        logger.info(f"Loading main data from: {data_dir}")
        all_df = self._load_and_preprocess_data(data_dir)

        # Load extra data if provided
        if self.args.extra_data_dir:
            logger.info(f"Loading extra data from: {self.args.extra_data_dir}")
            extra_df = self._load_and_preprocess_data(self.args.extra_data_dir, is_extra=True)
            # Group extra data by date (using int keys as in original df_to_dict2)
            self.extra_data_dict = df_to_grouped_dict(
                extra_df, group_col=DATE_COL, sort_cols=[DATE_COL, STOCK_COL], key_type=int
            )
            del extra_df # Free memory

        # Extract features and label name
        # Ensure columns are accessed *after* potential index setting
        temp_df_for_cols = all_df.reset_index() # Use a temporary df if index is set
        self.features = list(temp_df_for_cols.columns)[FEATURES_START_INDEX:FEATURES_END_INDEX]
        self.label_name = list(temp_df_for_cols.columns)[LABEL_COL_INDEX]
        logger.info(f"Features identified: {self.features}")
        logger.info(f"Label column identified: {self.label_name}")
        del temp_df_for_cols

        # Set index *after* extracting column names based on original order
        all_df.set_index(STOCK_COL, inplace=True)

        # Group main data by date (using original df_to_dict logic - string keys initially)
        factor_data_str_keys = df_to_grouped_dict(
            all_df.reset_index(), # Pass df with index as column for grouping
            group_col=DATE_COL,
            sort_cols=[DATE_COL, STOCK_COL],
            key_type=str # Keep original string keys first
        )
        del all_df # Free memory

        # Create date mappings and convert keys to integers
        self.dates: List[int] = sorted([int(d) for d in factor_data_str_keys.keys()])
        date_codes = list(range(len(self.dates)))
        self.day_to_code: Dict[int, int] = dict(zip(self.dates, date_codes))
        self.code_to_day: Dict[int, int] = dict(zip(date_codes, self.dates))

        # Convert factor_data keys to integer codes
        self.factor_data: Dict[int, pd.DataFrame] = {
            self.day_to_code[int(date_str)]: df.set_index(STOCK_COL) # Set index back after grouping
            for date_str, df in factor_data_str_keys.items()
        }
        self.num_dates = len(self.dates)

        # Determine valid stocks per date code based on input_len history
        self.valid_stocks_per_code: Dict[int, List[str]] = {}
        logger.info("Determining valid stocks per date based on input length...")
        for date_int in self.dates:
            current_code = self.day_to_code.get(date_int)
            if current_code is None or current_code < self.input_len -1:
                continue # Skip if not enough history

            # Find intersection of stocks available over the lookback window
            try:
                target_stocks = set(self.factor_data[current_code].index)
                for lookback_offset in range(1, self.input_len):
                     prev_code = current_code - lookback_offset
                     target_stocks &= set(self.factor_data[prev_code].index)

                if target_stocks:
                    self.valid_stocks_per_code[current_code] = sorted(list(target_stocks))
                    # logger.debug(f"Date {date_int} (Code {current_code}): {len(target_stocks)} valid stocks.")
                # else:
                    # logger.debug(f"Date {date_int} (Code {current_code}): No stocks valid for full lookback.")
            except KeyError as e:
                 logger.warning(f"KeyError while processing date code {current_code} (Date: {date_int}): {e}. Skipping.")
                 continue

        logger.info("Finished determining valid stocks.")

        # Convert dataframes to tensors in parallel
        self._stock_df_to_tensor()

        logger.info("Query object initialized successfully.")

    def _load_and_preprocess_data(self, file_path: str, is_extra: bool = False) -> pd.DataFrame:
        """Loads and performs initial preprocessing on a Parquet file."""
        if not os.path.exists(file_path):
             raise FileNotFoundError(f"Data file not found: {file_path}")
        df = pd.read_parquet(file_path)
        # Convert date column to integer format YYYYMMDD
        if DATE_COL not in df.columns:
             raise ValueError(f"'{DATE_COL}' column not found in {file_path}")
        df[DATE_COL] = df[DATE_COL].apply(lambda x: int("".join(str(x).split('-'))))
        # Ensure stock code column exists
        if STOCK_COL not in df.columns:
             raise ValueError(f"'{STOCK_COL}' column not found in {file_path}")
        # Optional: Convert stock code type if needed (e.g., to string)
        # df[STOCK_COL] = df[STOCK_COL].astype(str)
        return df

    def get_closest_past_date(self, target_date: int) -> Optional[int]:
        """Finds the latest date in self.dates that is less than or equal to target_date."""
        closest_date = None
        # Assumes self.dates is sorted
        for date_val in self.dates:
            if date_val <= target_date:
                closest_date = date_val
            else:
                break # Since dates are sorted, no need to check further
        return closest_date

    def _concat_input_data(self,
                          date_code: int,
                          stock_list: List[str],
                          scaler_class: Optional[Type[preprocessing.StandardScaler]] = None # Pass class
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Concatenates input features and labels for a given date code and stock list.

        Args:
            date_code: The target date code.
            stock_list: List of stock codes to fetch data for.
            scaler_class: Optional scaler class (e.g., preprocessing.StandardScaler)
                          to apply normalization per day.

        Returns:
            A tuple containing:
            - NumPy array of features with shape (n_stocks, input_len, n_features).
            - NumPy array of labels with shape (n_stocks,).
        """
        feature_arrays = []
        my_scaler = scaler_class() if scaler_class else None

        for k in range(date_code - self.input_len + 1, date_code + 1):
            try:
                daily_df = self.factor_data[k].loc[stock_list, self.features]
                daily_values = daily_df.values
                if my_scaler and self.args.fea_norm == 1:
                    daily_values = my_scaler.fit_transform(daily_values)
                feature_arrays.append(daily_values)
            except KeyError as e:
                 logger.error(f"KeyError accessing data for code {k}, stocks {stock_list}: {e}")
                 # Handle missing data, e.g., return NaNs or raise error
                 # Returning NaNs of the correct shape:
                 nan_array = np.full((len(stock_list), len(self.features)), np.nan)
                 feature_arrays.append(nan_array)
            except Exception as e:
                 logger.error(f"Unexpected error processing code {k}: {e}")
                 raise # Re-raise other unexpected errors

        # Stack along a new axis (axis=1 for time steps) -> (n_stocks, input_len, n_features)
        features_result = np.stack(feature_arrays, axis=1)

        # Get labels for the target date code
        try:
            labels = self.factor_data[date_code].loc[stock_list, self.label_name].values
        except KeyError as e:
            logger.error(f"KeyError accessing labels for code {date_code}, stocks {stock_list}: {e}")
            labels = np.full(len(stock_list), np.nan) # Return NaNs if labels missing
        except Exception as e:
            logger.error(f"Unexpected error accessing labels for code {date_code}: {e}")
            raise

        return features_result, labels

    def get_data_step(self,
                      date_code: int,
                      stock_list: Optional[List[str]] = None
                      ) -> Optional[Tuple[Tuple[np.ndarray, np.ndarray], Optional[np.ndarray]]]:
        """
        Retrieves feature/label data and optional extra data for a single step (date code).

        Args:
            date_code: The target date code.
            stock_list: Optional list of stock codes. If None, uses all valid stocks
                        for the date code determined during initialization.

        Returns:
            A tuple containing:
            - A tuple of (features_array, labels_array).
            - An optional extra_data_array, or None.
            Returns None if the date_code is invalid.
        """
        if date_code not in self.valid_stocks_per_code:
            target_date = self.code_to_day.get(date_code, "Unknown Date")
            logger.warning(f"Invalid or insufficient history for date_code: {date_code} (Date: {target_date}). No valid stocks.")
            return None

        fetch_list = self.valid_stocks_per_code[date_code] if stock_list is None else stock_list
        if not fetch_list:
             logger.warning(f"No stocks provided or found valid for date_code: {date_code}")
             return None

        # Get main features and labels
        scaler_to_use = preprocessing.StandardScaler if self.args.fea_norm == 1 else None
        main_data = self._concat_input_data(date_code, fetch_list, scaler_to_use)

        # Get extra data if configured
        extra_data_result = None
        if self.extra_data_dict:
            target_date = self.code_to_day[date_code]
            if target_date not in self.extra_data_dict:
                logger.warning(f"No extra data found for date_code: {date_code} (Date: {target_date})")
            else:
                daily_extra_df = self.extra_data_dict[target_date]
                # Filter for the required stocks
                filtered_extra_df = daily_extra_df[daily_extra_df[STOCK_COL].isin(fetch_list)]

                # Ensure the order matches fetch_list (important!)
                filtered_extra_df = filtered_extra_df.set_index(STOCK_COL).loc[fetch_list].reset_index()

                if filtered_extra_df.empty:
                     logger.warning(f"Extra data found for date {target_date}, but not for requested stocks: {fetch_list}")
                else:
                    try:
                        # Assuming fea_qlib=1 means specific column processing
                        if self.args.fea_qlib == 1:
                            # Hardcoded sequence length and column names - consider making configurable
                            SEQ_LEN = 60
                            if self.args.extra_price == 1:
                                cols = [f'close_{T}' for T in range(SEQ_LEN - 1, -1, -1)]
                                df_vals = filtered_extra_df[cols].values
                                # Assuming one feature per timestamp, reshape (n_stocks, seq_len) -> (n_stocks, seq_len, 1)
                                extra_data1 = df_vals.reshape(df_vals.shape[0], SEQ_LEN, -1)
                                # Duplicate features? Original code concatenated same data.
                                extra_data_result = np.concatenate([extra_data1, extra_data1], axis=-1)
                            else:
                                pred_cols = [f'pred_{T}' for T in range(SEQ_LEN - 1, -1, -1)]
                                price_cols = [f'price_{T}' for T in range(SEQ_LEN - 1, -1, -1)]
                                df_pred = filtered_extra_df[pred_cols].values
                                df_price = filtered_extra_df[price_cols].values
                                extra_data1 = df_pred.reshape(df_pred.shape[0], SEQ_LEN, -1)
                                extra_data2 = df_price.reshape(df_price.shape[0], SEQ_LEN, -1)
                                extra_data_result = np.concatenate([extra_data1, extra_data2], axis=-1)

                            # logger.debug(f'Extra data shape for code {date_code}: {extra_data_result.shape}')
                        else:
                             # Handle non-qlib extra data if needed
                             logger.warning("fea_qlib != 1, extra data processing not implemented for this case.")
                             pass # Or extract relevant columns differently

                    except KeyError as e:
                         logger.error(f"Missing expected columns in extra data for date {target_date}: {e}")
                         extra_data_result = None # Or handle differently
                    except Exception as e:
                         logger.error(f"Error processing extra data for date {target_date}: {e}")
                         extra_data_result = None


        return main_data, extra_data_result

    def get_data_step_tensor(self, date_code: int) -> Optional[Tuple[Tensor, Tensor, Optional[Tensor]]]:
        """
        Retrieves pre-converted tensor data for a single step (date code).

        Args:
            date_code: The target date code.

        Returns:
            A tuple containing:
            - factor_tensor: Features tensor.
            - label_tensor: Labels tensor.
            - extra_data_tensor: Optional extra data tensor, or None.
            Returns None if the date_code is invalid or data wasn't pre-converted.
        """
        if date_code not in self.factor_tensor:
            target_date = self.code_to_day.get(date_code, "Unknown Date")
            logger.error(f"Tensor data not found for date_code: {date_code} (Date: {target_date})")
            return None

        factor_data = self.factor_tensor[date_code]
        label_data = self.label_tensor[date_code]
        extra_data = self.extra_data_tensor.get(date_code) # Use .get for optional data

        # Reshape features if fea_qlib is set (assuming specific input format)
        # Hardcoded 6 - should relate to features/input_len? Clarify this logic.
        # Example: if features are (price, vol, ...) and input_len=3, shape might be (N, 3, F)
        # Reshaping to (N, 6, -1) seems specific. Let's assume it's (N, input_len, n_features)
        # and fea_qlib requires (N, n_features, input_len) ?
        if self.args.fea_qlib == 1 and len(factor_data.shape) == 3:
             # Original: reshape(factor_data.shape[0], 6, -1).permute(0,2,1)
             # This implies the middle dimension should be 6? Let's assume it should be input_len
             # And the last dim is n_features. Permute swaps last two dims.
             # factor_data = factor_data.permute(0, 2, 1) # (N, n_features, input_len)
             # The reshape(..., 6, -1) is confusing without context. Keeping original for now.
             try:
                 factor_data = factor_data.reshape(factor_data.shape[0], 6, -1)
                 factor_data = factor_data.permute(0, 2, 1)
             except RuntimeError as e:
                 logger.error(f"Error reshaping factor tensor for fea_qlib=1 at code {date_code}: {e}. Original shape: {self.factor_tensor[date_code].shape}")
                 # Return original shape or handle error
                 factor_data = self.factor_tensor[date_code]


        return factor_data, label_data, extra_data

    def _convert_step_to_tensor(self, date_code: int):
        """Helper function to convert data for one date code to tensors."""
        try:
            step_data = self.get_data_step(date_code)
            if step_data:
                (factor_np, label_np), extra_np = step_data
                # Convert valid data to tensors
                if not np.isnan(factor_np).any():
                     self.factor_tensor[date_code] = torch.from_numpy(factor_np).float()
                else:
                     logger.warning(f"NaNs found in factor data for code {date_code}, skipping tensor conversion.")
                if not np.isnan(label_np).any():
                     self.label_tensor[date_code] = torch.from_numpy(label_np).float()
                else:
                     logger.warning(f"NaNs found in label data for code {date_code}, skipping tensor conversion.")

                if extra_np is not None and not np.isnan(extra_np).any():
                    self.extra_data_tensor[date_code] = torch.from_numpy(extra_np).float()
                elif extra_np is not None:
                     logger.warning(f"NaNs found in extra data for code {date_code}, skipping tensor conversion.")

        except Exception as e:
            logger.error(f"Error converting data to tensor for date_code {date_code}: {e}", exc_info=True)


    def _stock_df_to_tensor(self):
        """Converts loaded DataFrame data to PyTorch tensors using threading."""
        logger.info("Converting DataFrame slices to tensors...")
        # Get date codes for which we have valid stock lists
        valid_date_codes = sorted(list(self.valid_stocks_per_code.keys()))

        if not valid_date_codes:
            logger.warning("No valid date codes found with sufficient history. Tensor conversion skipped.")
            return

        # Use ThreadPool for potentially I/O bound operations within get_data_step
        # Adjust pool size based on system resources if needed
        num_threads = min(os.cpu_count() or 1, 8) # Limit threads
        logger.info(f"Using {num_threads} threads for tensor conversion.")
        with ThreadPool(processes=num_threads) as pool:
            pool.map(self._convert_step_to_tensor, valid_date_codes)

        logger.info(f"Tensor conversion finished. {len(self.factor_tensor)} dates processed.")

    def get_valid_date_range_codes(self) -> Optional[Tuple[int, int]]:
        """Gets the start and end date codes for which data is available and valid."""
        valid_codes = sorted(list(self.valid_stocks_per_code.keys()))
        if not valid_codes:
            return None
        # End date is inclusive, so add 1 for range iteration
        return valid_codes[0], valid_codes[-1] + 1

    def get_features(self) -> List[str]:
        """Returns the list of feature names."""
        return self.features

    def clear_data_for_dates(self, dates_to_remove_codes: List[int]):
        """Removes pre-computed tensor data for specified date codes."""
        logger.warning(f"Clearing tensor data for {len(dates_to_remove_codes)} date codes.")
        cleared_count = 0
        for code in dates_to_remove_codes:
            if self.factor_tensor.pop(code, None) is not None:
                 cleared_count += 1
            self.label_tensor.pop(code, None)
            self.extra_data_tensor.pop(code, None)
            # Note: This does NOT remove from self.factor_data (original DFs)
            # or self.valid_stocks_per_code. It only clears the tensors.
        logger.info(f"Cleared tensor data for {cleared_count} codes.")

    # Original clean_data seemed to intend removing dates *not* in the list.
    # Renaming and clarifying:
    def keep_only_dates(self, dates_to_keep_codes: List[int]):
        """Removes tensor data for all date codes *not* in the provided list."""
        all_codes = set(self.factor_tensor.keys())
        codes_to_remove = list(all_codes - set(dates_to_keep_codes))
        self.clear_data_for_dates(codes_to_remove)
