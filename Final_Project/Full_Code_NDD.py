# ================================================================
# Paper-Style Hierarchical Forecasting for Brand B1 → Items
# - Base forecast at brand level (CNN on y, MLP on exog) — F*
# - Disaggregation NND: 1D CNN (brand history) + per-item MLP over exog
#   producing item shares (via softmax) * brand scalar → item forecasts
# - Windows: w=30, hop=1; Train: 2014-01-01..2017-12-31; Test: 2018-01-01..2018-12-31
# Requirements: tensorflow/keras, scikit-learn, pandas, numpy, matplotlib (optional)
# ================================================================

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# Keras / TensorFlow
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------------
# 0) Assumptions & config
# ---------------------------------------------------------------
# - df: DataFrame with DATE index (daily), item-level QTY_B1_i and PROMO_B1_i columns
# - brand_dict: dict mapping brand -> list of item ids as provided

assert isinstance(df.index, pd.DatetimeIndex), "df.index must be a DatetimeIndex"
df = df.sort_index()

BRAND = "B1"
WINDOW = 30
HOP = 1
TRAIN_START = pd.Timestamp("2014-01-01")
TRAIN_END   = pd.Timestamp("2017-12-31")
TEST_START  = pd.Timestamp("2018-01-01")
TEST_END    = pd.Timestamp("2018-12-31")

# Clamp dates to available data
TRAIN_START = max(TRAIN_START, df.index.min())
TEST_END    = min(TEST_END, df.index.max())

# ---------------------------------------------------------------
# 1) Utilities to extract brand & item series and exogenous features
# ---------------------------------------------------------------

def get_items_for_brand(brand: str) -> List[str]:
    items = brand_dict.get(brand, [])
    if not items:
        raise ValueError(f"No items found for brand {brand} in brand_dict.")
    return items


def get_brand_series(frame: pd.DataFrame, brand: str) -> pd.Series:
    # If a precomputed brand column exists, prefer it
    col = f"Brand_QTY_{brand}"
    if col in frame.columns:
        y = frame[col].astype(float)
    else:
        # Sum item-level quantities
        items = get_items_for_brand(brand)
        qty_cols = [f"QTY_{brand}_{i}" for i in items if f"QTY_{brand}_{i}" in frame.columns]
        if not qty_cols:
            raise ValueError(f"No QTY columns for {brand} present in df.")
        y = frame[qty_cols].sum(axis=1).astype(float)
    return y.asfreq("D").fillna(0.0)


def build_brand_exog(frame: pd.DataFrame, brand: str) -> pd.DataFrame:
    ex = pd.DataFrame(index=frame.index)
    # Promo share = relative number of items under promo for this brand
    items = get_items_for_brand(brand)
    promo_cols = [f"PROMO_{brand}_{i}" for i in items if f"PROMO_{brand}_{i}" in frame.columns]
    if promo_cols:
        ex["promo_share"] = frame[promo_cols].sum(axis=1) / float(len(promo_cols))
    else:
        ex["promo_share"] = 0.0
    # Calendar dummies: DOW & Month (drop_first to avoid dummy trap if intercept exists)
    dow = pd.get_dummies(frame.index.dayofweek, prefix="dow", drop_first=True)
    mon = pd.get_dummies(frame.index.month,     prefix="mon", drop_first=True)
    dow.index = ex.index; mon.index = ex.index
    ex = pd.concat([ex, dow, mon], axis=1).astype(float)
    return ex.asfreq("D").fillna(0.0)


def build_item_exog(frame: pd.DataFrame, brand: str, items: List[str]) -> Dict[str, pd.DataFrame]:
    # Shared calendar dummies
    dow = pd.get_dummies(frame.index.dayofweek, prefix="dow", drop_first=True)
    mon = pd.get_dummies(frame.index.month,     prefix="mon", drop_first=True)
    dow.index = frame.index; mon.index = frame.index
    calendar = pd.concat([dow, mon], axis=1).astype(float)

    exog = {}
    for it in items:
        promo_col = f"PROMO_{brand}_{it}"
        if promo_col in frame.columns:
            promo = frame[[promo_col]].astype(float)
        else:
            promo = pd.DataFrame({promo_col: np.zeros(len(frame), dtype=float)}, index=frame.index)
        exog[it] = pd.concat([promo, calendar], axis=1).asfreq("D").fillna(0.0)
    return exog


def get_items_matrix(frame: pd.DataFrame, brand: str, items: List[str]) -> pd.DataFrame:
    cols = [f"QTY_{brand}_{it}" for it in items if f"QTY_{brand}_{it}" in frame.columns]
    if len(cols) != len(items):
        missing = [it for it in items if f"QTY_{brand}_{it}" not in frame.columns]
        raise ValueError(f"Missing QTY columns for items: {missing}")
    return frame[cols].astype(float).asfreq("D").fillna(0.0)


# ---------------------------------------------------------------
# 2) Window builders
# ---------------------------------------------------------------

def build_brand_sequences(y: pd.Series, X: pd.DataFrame, window: int = 30, hop: int = 1,
                          scaler_y: StandardScaler = None, scaler_X: StandardScaler = None):
    """Return y_seq (N,w,1), X_seq (N,w,nx), y_tgt (N,), idx (N,) — scaled via provided scalers.
    Fits scalers if None, using only the rows whose target index is <= TRAIN_END.
    """
    assert y.index.equals(X.index)
    # Build raw sequences first (unscaled)
    vals_y = y.values.astype(float)
    vals_X = X.values.astype(float)
    n, nx = len(y), X.shape[1]

    y_seq_list, X_seq_list, y_tgt_list, idx_list = [], [], [], []
    t = window
    while t < n:
        y_seq_list.append(vals_y[t-window:t].reshape(window, 1))
        X_seq_list.append(vals_X[t-window:t].reshape(window, nx))
        y_tgt_list.append(vals_y[t])
        idx_list.append(y.index[t])
        t += hop

    y_seq = np.stack(y_seq_list, axis=0)
    X_seq = np.stack(X_seq_list, axis=0)
    y_tgt = np.array(y_tgt_list, dtype=float)
    idx = pd.to_datetime(pd.Index(idx_list))

    # Fit scalers on in-sample targets (up to TRAIN_END)
    if scaler_y is None:
        scaler_y = StandardScaler()
        ins_mask = (idx <= TRAIN_END)
        scaler_y.fit(y_tgt[ins_mask].reshape(-1,1))
    if scaler_X is None:
        scaler_X = StandardScaler()
        # To fit X scaler, take all windows whose target <= TRAIN_END
        ins_mask = (idx <= TRAIN_END)
        X_ins = X_seq[ins_mask].reshape(ins_mask.sum()*window, nx)
        scaler_X.fit(X_ins)

    # Apply scaling
    y_tgt_sc = scaler_y.transform(y_tgt.reshape(-1,1)).ravel()
    X_seq_sc = scaler_X.transform(X_seq.reshape(-1, nx)).reshape(-1, window, nx)
    y_seq_sc = scaler_y.transform(y_seq.reshape(-1,1)).reshape(-1, window, 1)

    return y_seq_sc, X_seq_sc, y_tgt_sc, idx, scaler_y, scaler_X


def build_item_disagg_sequences(
    y_brand: pd.Series,
    Y_items: pd.DataFrame,
    item_exog: Dict[str, pd.DataFrame],
    window: int = 30,
    hop: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Build sequences for disaggregation model.
    Returns:
      - yB_seq:   (N, w, 1)    brand history (UNSCALED)
      - Xitem_seq:(N, I, w, m) per-item exog windows (UNSCALED)
      - yB_t:     (N,)         brand scalar target at t (UNSCALED)
      - Yitem_t:  (N, I)       item targets at t (UNSCALED)
      - idx:      (N,)         target timestamps
    """
    items = [c.split("_")[-1] for c in Y_items.columns]
    n_items = len(items)

    # infer item exog width (all items share schema)
    any_item = items[0]
    m = item_exog[any_item].shape[1]

    vals_yB = y_brand.values.astype(float)
    vals_items = Y_items.values.astype(float)  # shape (T, I)
    n = len(y_brand)

    yB_seq_list, Xitem_seq_list, yB_t_list, Yitem_t_list, idx_list = [], [], [], [], []

    t = window
    while t < n:
        # brand history
        yB_seq_list.append(vals_yB[t-window:t].reshape(window, 1))
        # per-item exog windows
        per_item_windows = []
        for it in items:
            Xi = item_exog[it].values.astype(float)
            per_item_windows.append(Xi[t-window:t].reshape(window, m))
        Xitem_seq_list.append(np.stack(per_item_windows, axis=0))  # (I, w, m)
        # targets
        yB_t_list.append(vals_yB[t])
        Yitem_t_list.append(vals_items[t, :])
        idx_list.append(y_brand.index[t])
        t += hop

    yB_seq   = np.stack(yB_seq_list, axis=0)                      # (N, w, 1)
    Xitem_seq= np.stack(Xitem_seq_list, axis=0)                   # (N, I, w, m)
    yB_t     = np.array(yB_t_list, dtype=float)                   # (N,)
    Yitem_t  = np.stack(Yitem_t_list, axis=0).astype(float)       # (N, I)
    idx      = pd.to_datetime(pd.Index(idx_list))

    return yB_seq, Xitem_seq, yB_t, Yitem_t, idx


# ---------------------------------------------------------------
# 3) Models
# ---------------------------------------------------------------

def build_brand_cnn_mlp(window: int, n_exog: int) -> Model:
    # y branch
    inp_y = layers.Input(shape=(window, 1), name="y_seq")
    a = layers.Conv1D(16, 3, padding="causal", activation="relu")(inp_y)
    a = layers.Conv1D(16, 3, padding="causal", activation="relu")(a)
    a = layers.GlobalAveragePooling1D()(a)
    # X branch
    inp_x = layers.Input(shape=(window, n_exog), name="x_seq")
    b = layers.Flatten()(inp_x)
    b = layers.Dense(64, activation="relu")(b)
    # fuse
    h = layers.Concatenate()([a, b])
    h = layers.Dense(64, activation="relu")(h)
    out = layers.Dense(1, name="yhat_scaled")(h)  # trained in scaled space
    model = Model([inp_y, inp_x], out)
    model.compile(optimizer="adam", loss="mse")
    return model

#    NDD 

def build_disagg_nnd(window: int, n_items: int, n_item_exog: int,
                     brand_latent_dim: int = 16) -> Model:
    """NND that outputs per-item forecasts coherent with the brand scalar.
    Inputs:
      - yB_seq: (w,1) brand history
      - Xitem_seq: (I,w,m) per-item exog sequence
      - yB_scalar: (1,) brand scalar at target time (true for training, forecast at inference)
    Output:
      - Yitem_pred: (I,) per-item forecasts that sum to yB_scalar (softmax shares * scalar)
    """
    # Brand branch
    inp_yB = layers.Input(shape=(window, 1), name="yB_seq")
    a = layers.Conv1D(brand_latent_dim, 3, padding="causal", activation="relu")(inp_yB)
    a = layers.GlobalAveragePooling1D()(a)  # (batch, brand_latent_dim)

    # Per-item exog branch
    inp_Xi = layers.Input(shape=(n_items, window, n_item_exog), name="Xitem_seq")
    # Flatten time+features for each item, keep item axis
    z = layers.Reshape((n_items, window * n_item_exog))(inp_Xi)
    z = layers.TimeDistributed(layers.Dense(64, activation="relu"))(z)
    z = layers.TimeDistributed(layers.Dense(32, activation="relu"))(z)  # (batch, I, 32)

    # Repeat brand latent across items and fuse
    a_rep = layers.RepeatVector(n_items)(a)  # (batch, I, brand_latent_dim)
    u = layers.Concatenate(axis=-1)([z, a_rep])  # (batch, I, 32+latent)
    u = layers.TimeDistributed(layers.Dense(32, activation="relu"))(u)
    logits = layers.TimeDistributed(layers.Dense(1))(u)  # (batch, I, 1)
    logits = layers.Reshape((n_items,))(logits)           # (batch, I)
    shares = layers.Softmax(axis=-1, name="shares")(logits)  # sums to 1 across items

    # Multiply by brand scalar to get coherent item forecasts
    inp_yB_scalar = layers.Input(shape=(1,), name="yB_scalar")  # (batch,1)
    yB_rep = layers.Concatenate()([inp_yB_scalar for _ in range(n_items)])  # (batch, I)
    y_items_pred = layers.Multiply(name="Yitem_pred")([shares, yB_rep])

    model = Model([inp_yB, inp_Xi, inp_yB_scalar], y_items_pred)
    model.compile(optimizer="adam", loss="mse")  # MSE on items in original units
    return model


# ---------------------------------------------------------------
# 4) Prepare data for BRAND (F*) and train/predict
# ---------------------------------------------------------------
items_B1 = get_items_for_brand(BRAND)
y_brand = get_brand_series(df, BRAND)
X_brand  = build_brand_exog(df, BRAND)

# Brand windows (scaled)
yb_seq_sc, xb_seq_sc, yb_tgt_sc, idx_b, y_scaler, Xb_scaler = build_brand_sequences(
    y_brand, X_brand, window=WINDOW, hop=HOP, scaler_y=None, scaler_X=None
)

# Train/val/test masks by target index
mask_train_b = (idx_b >= TRAIN_START) & (idx_b <= TRAIN_END)
mask_test_b  = (idx_b >= TEST_START)  & (idx_b <= TEST_END)

# Simple time-based split inside train for validation (last 10% of train windows)
train_idx = np.where(mask_train_b)[0]
cut = int(np.floor(0.9 * len(train_idx)))
tr_sel, va_sel = train_idx[:cut], train_idx[cut:]

brand_model = build_brand_cnn_mlp(WINDOW, X_brand.shape[1])
_ = brand_model.fit(
    [yb_seq_sc[tr_sel], xb_seq_sc[tr_sel]], yb_tgt_sc[tr_sel],
    validation_data=([yb_seq_sc[va_sel], xb_seq_sc[va_sel]], yb_tgt_sc[va_sel]),
    epochs=20, batch_size=64, verbose=0,
    callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
)

# Predict scaled brand on test windows, then invert scaling to get to original units, unnormalized
yb_pred_sc = brand_model.predict([yb_seq_sc[mask_test_b], xb_seq_sc[mask_test_b]], verbose=0).ravel()
yb_pred    = y_scaler.inverse_transform(yb_pred_sc.reshape(-1,1)).ravel()
idx_b_test = idx_b[mask_test_b]
brand_pred_series = pd.Series(yb_pred, index=idx_b_test, name=f"{BRAND}_pred")

# ---------------------------------------------------------------
# 5) Prepare data for DISAGG (NND) and train/predict
# ---------------------------------------------------------------
Y_items = get_items_matrix(df, BRAND, items_B1)  # columns QTY_B1_i
item_exog = build_item_exog(df, BRAND, items_B1) # PROMO_B1_i + calendar per item

# Build disagg sequences (UNSCALED)
yB_seq, Xitem_seq, yB_scalar, Yitem_t, idx_d = build_item_disagg_sequences(
    y_brand, Y_items, item_exog, window=WINDOW, hop=HOP
)

# Align train/test for disagg using target timestamps
d_train = (idx_d >= TRAIN_START) & (idx_d <= TRAIN_END)
d_test  = (idx_d >= TEST_START)  & (idx_d <= TEST_END)

# Build inputs/targets
X_disagg_train = [yB_seq[d_train], Xitem_seq[d_train], yB_scalar[d_train].reshape(-1,1)]
Y_disagg_train = Yitem_t[d_train]
X_disagg_test  = [yB_seq[d_test],  Xitem_seq[d_test],  None]  # yB_scalar for test will be brand forecast

# For training, avoid windows where brand scalar is zero (optional)
nonzero = Y_disagg_train.sum(axis=1) > 0
X_disagg_train = [x[nonzero] for x in X_disagg_train]
Y_disagg_train = Y_disagg_train[nonzero]

# Build and train disaggregation model
n_items = len(items_B1)
n_item_exog = next(iter(item_exog.values())).shape[1]

disagg_model = build_disagg_nnd(WINDOW, n_items=n_items, n_item_exog=n_item_exog)
_ = disagg_model.fit(
    X_disagg_train, Y_disagg_train,
    validation_split=0.1,
    epochs=20, batch_size=64, verbose=0,
    callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
)

# Build brand scalars for test from brand forecast (ensures coherence)
yB_scalar_test = brand_pred_series.reindex(idx_d[d_test]).values.reshape(-1,1)
X_disagg_test[2] = yB_scalar_test

# Predict per-item test forecasts (already coherent with brand)
Y_items_pred = disagg_model.predict(X_disagg_test, verbose=0)  # shape (N_test, I)
idx_items_test = idx_d[d_test]

# Pack item forecasts into a DataFrame (columns aligned to items)
items_cols = [f"QTY_{BRAND}_{it}" for it in items_B1]
items_pred_df = pd.DataFrame(Y_items_pred, index=idx_items_test, columns=items_cols)


# ---------------------------------------------------------------
# 6) Sanity checks & simple metrics
# ---------------------------------------------------------------
# Coherence check: sum of item preds vs brand pred
sum_items_pred = items_pred_df.sum(axis=1)
coh_max_abs_err = np.max(np.abs(sum_items_pred.values - brand_pred_series.reindex(sum_items_pred.index).values))
print(f"Max abs coherence error (should be ~0): {coh_max_abs_err:.6f}")

# Simple accuracy on items where ground truth exists
Y_items_test = Y_items.reindex(idx_items_test)
rmse_items = mean_squared_error(Y_items_test.values.ravel(), items_pred_df.values.ravel(), squared=False)
mae_items  = mean_absolute_error(Y_items_test.values.ravel(), items_pred_df.values.ravel())
print(f"Items — RMSE: {rmse_items:.3f}, MAE: {mae_items:.3f}")

# Brand accuracy on test
y_brand_test = y_brand.reindex(idx_b_test).values
rmse_brand = mean_squared_error(y_brand_test, brand_pred_series.values, squared=False)
mae_brand  = mean_absolute_error(y_brand_test, brand_pred_series.values)
print(f"Brand {BRAND} — RMSE: {rmse_brand:.3f}, MAE: {mae_brand:.3f}")


import matplotlib.pyplot as plt
from typing import Union

def plot_brand_test_overlay(
    df: pd.DataFrame,
    brand: str,
    brand_pred: pd.Series,                 # forecast as a Date-indexed Series
    test_start: Union[str, pd.Timestamp],
    test_end: Union[str, pd.Timestamp],
    date_col_indexed: bool = True,
    y_label: str = "Units",
    actual_label: str = "Actual",
    forecast_label: str = "Forecast",
    title_suffix: str = "(Test Period)"
) -> tuple[pd.Series, pd.Series]:
    """
    Overlay Actual vs Forecast for a brand on ONE chart, restricted to [test_start, test_end].
    Returns (y_true_test, y_pred_on_true_index).
    """

    # Actual brand series (prefer precomputed Brand_QTY_<brand>, else sum items)
    brand_sum_col = f"Brand_QTY_{brand}"

    # Restrict to test window
    test_start = pd.to_datetime(test_start)
    test_end   = pd.to_datetime(test_end)
    y_true_test = y_brand.loc[test_start:test_end]
    if y_true_test.empty:
        raise ValueError("Actual test window is empty; verify test_start/test_end against your data range.")

    # Forecast must be a Series with a DatetimeIndex
    if not isinstance(brand_pred, pd.Series) or not isinstance(brand_pred.index, pd.DatetimeIndex):
        raise ValueError("brand_pred must be a pandas Series with a DatetimeIndex.")
    y_pred_test = brand_pred.sort_index().loc[test_start:test_end]
    if y_pred_test.empty:
        raise ValueError("Forecast test window is empty; check forecast dates vs test window.")

    # Align forecast to actual dates (for metrics if needed)
    y_pred_on_true_index = y_pred_test.reindex(y_true_test.index)

    # Plot
    plt.figure(figsize=(14, 4))
    plt.plot(y_true_test.index, y_true_test.values, label=f"{actual_label} {brand}", linewidth=2)
    plt.plot(y_pred_test.index, y_pred_test.values, label=f"{forecast_label} {brand}", linewidth=2, linestyle="--")
    plt.title(f"{brand} — {actual_label} vs {forecast_label} {title_suffix}".strip())
    plt.xlabel("Date")
    plt.ylabel(y_label)
    plt.grid(True)
    plt.legend()
    plt.show()

    return y_true_test, y_pred_on_true_index

# ---- Call: plot B1 actual vs forecast in test window ----
_ = plot_brand_test_overlay(
    df=df,
    brand=BRAND,
    brand_pred=brand_pred_series,     # <-- pass the FORECAST series (not the actuals array)
    test_start=TEST_START,
    test_end=TEST_END,
    forecast_label="NDD base forecast"
)