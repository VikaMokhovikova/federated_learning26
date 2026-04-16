"""
functions.py — Federated Learning utilities for FMI temperature prediction.

Project task: each FMI weather station is an FL node.
              We predict tomorrow's daily maximum temperature (tmax_{t+1})
              from today's [tmin_t, tmax_t] using linear regression at each node.

Sections
--------
1. Data loading & preprocessing
2. Graph construction  (System A: geographic k-NN;  System B: data similarity)
3. Local model         (linear regression — MSE loss, gradient, smoothness)
4. FL algorithms       (FedGD, FedRelax — both implement GTVMin)
5. Evaluation & hyperparameter tuning
6. Visualization helpers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.linalg import eigvalsh
from typing import Dict, List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Data loading & preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Load daily_records.csv and return a DataFrame sorted by (station, day)."""
    df = pd.read_csv(path, parse_dates=["day"])
    df = df.sort_values(["station", "day"]).reset_index(drop=True)
    return df


def get_station_meta(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per station with columns: station, lat, lon."""
    return (
        df[["station", "lat", "lon"]]
        .drop_duplicates("station")
        .sort_values("station")
        .reset_index(drop=True)
    )


def build_node_datasets(df: pd.DataFrame, min_points: int = 30) -> Dict[str, Dict]:
    """
    Build per-station supervised datasets for next-day tmax prediction.

    For each station we form consecutive-day pairs:
        x_t  = [1,  tmin_t,  tmax_t]   (shape: (m_i, 3))   — feature vector
        y_t  = tmax_{t+1}               (shape: (m_i,))     — label

    The leading `1` is the bias / intercept term so the weight vector
    w ∈ ℝ³ includes the intercept directly.

    Only pairs where both days are consecutive (no gap) AND neither tmin/tmax
    is NaN are included.  Stations with fewer than `min_points` pairs are
    discarded (too little data to learn anything meaningful).

    Returns
    -------
    dict  keyed by station name, each entry:
        "X"    : np.ndarray (m_i, 3)
        "y"    : np.ndarray (m_i,)
        "days" : list of pd.Timestamp  (the t-th day, label is t+1)
        "lat"  : float
        "lon"  : float
    """
    datasets: Dict[str, Dict] = {}

    for station, grp in df.groupby("station"):
        grp = grp.sort_values("day").dropna(subset=["tmin", "tmax"]).reset_index(drop=True)
        n = len(grp)
        if n < 2:
            continue

        tmin = grp["tmin"].values
        tmax = grp["tmax"].values
        days = grp["day"].tolist()

        # Keep only consecutive-day pairs (gap check)
        valid = []
        for t in range(n - 1):
            delta = (days[t + 1] - days[t]).days
            if delta == 1:
                valid.append(t)

        if len(valid) < min_points:
            continue

        idx = np.array(valid)
        X = np.column_stack([np.ones(len(idx)), tmin[idx], tmax[idx]])
        y = tmax[idx + 1]
        day_list = [days[t] for t in idx]

        datasets[station] = {
            "X":    X.astype(np.float64),
            "y":    y.astype(np.float64),
            "days": day_list,
            "lat":  float(grp["lat"].iloc[0]),
            "lon":  float(grp["lon"].iloc[0]),
        }

    return datasets


def chronological_split(
    node_datasets: Dict[str, Dict],
    train_frac: float = 0.60,
    val_frac:   float = 0.20,
) -> Dict[str, Dict]:
    """
    Add chronological train / validation / test index arrays to each dataset.

    Using a chronological split (not random) is mandatory for time-series data:
    future observations must never leak into the training set.

    Adds in-place: 'train_idx', 'val_idx', 'test_idx' (numpy arrays of int).
    """
    for data in node_datasets.values():
        m = len(data["y"])
        n_train = int(m * train_frac)
        n_val   = int(m * val_frac)
        data["train_idx"] = np.arange(0, n_train)
        data["val_idx"]   = np.arange(n_train, n_train + n_val)
        data["test_idx"]  = np.arange(n_train + n_val, m)
    return node_datasets


def standardize_node_datasets(node_datasets: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Standardise feature columns [tmin, tmax] (columns 1 and 2) using
    training-set mean and standard deviation.  The bias column (index 0)
    is left as 1 and is NOT scaled.

    Standardisation is computed from training data ONLY to prevent leakage.
    The same (mean, std) is then applied to validation and test sets.

    Adds in-place: 'X_std' (standardised feature matrix), 'x_mean', 'x_std'.
    """
    for data in node_datasets.values():
        X  = data["X"]
        tr = data["train_idx"]

        x_mean = X[tr, 1:].mean(axis=0)   # shape (2,)
        x_std  = X[tr, 1:].std(axis=0)    # shape (2,)
        x_std[x_std == 0] = 1.0           # avoid division by zero

        X_std = X.copy()
        X_std[:, 1:] = (X[:, 1:] - x_mean) / x_std

        data["X_std"]  = X_std
        data["x_mean"] = x_mean
        data["x_std"]  = x_std

    return node_datasets


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two (lat, lon) points in kilometres."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def build_distance_matrix(stations: List[Dict]) -> np.ndarray:
    """
    Compute pairwise Haversine distances (km) between all stations.
    Each station dict must have 'lat' and 'lon'.
    Returns symmetric (n, n) array with zeros on the diagonal.
    """
    n = len(stations)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(
                stations[i]["lat"], stations[i]["lon"],
                stations[j]["lat"], stations[j]["lon"],
            )
            D[i, j] = D[j, i] = d
    return D


def build_geo_graph(
    stations: List[Dict],
    k:        int            = 3,
    sigma:    Optional[float] = None,
) -> np.ndarray:
    """
    System A — Geographic k-Nearest-Neighbour graph.

    For each station i, connect to its k geographically closest neighbours.
    The graph is then symmetrised (edge added in BOTH directions).

    Edge weight:  A[i,j] = exp( -d(i,j)² / (2σ²) )

    σ defaults to the median pairwise distance over all station pairs,
    which is a scale-free, data-driven choice.

    Parameters
    ----------
    stations : list of dicts with 'lat', 'lon'
    k        : number of nearest neighbours per node
    sigma    : bandwidth for Gaussian kernel (km); None → median distance

    Returns
    -------
    A : symmetric (n, n) adjacency matrix
    """
    D = build_distance_matrix(stations)
    n = len(stations)

    if sigma is None:
        off_diag = D[np.triu_indices(n, k=1)]
        sigma = float(np.median(off_diag))

    W = np.exp(-(D ** 2) / (2.0 * sigma ** 2))
    np.fill_diagonal(W, 0.0)

    A = np.zeros((n, n))
    for i in range(n):
        dist_i = D[i].copy()
        dist_i[i] = np.inf
        nn_idx = np.argsort(dist_i)[:k]
        A[i, nn_idx] = W[i, nn_idx]

    # Symmetrize: include edge if EITHER i→j or j→i was a k-NN link
    A = np.maximum(A, A.T)
    return A


def build_similarity_graph(
    node_datasets:  Dict[str, Dict],
    station_names:  List[str],
    threshold:      float = 0.80,
) -> np.ndarray:
    """
    System B — Data-similarity graph based on Pearson correlation.

    Two stations i, j are connected if the Pearson correlation of their
    training-period tmax time series is ≥ threshold.  The correlation
    is computed only on SHARED days to handle missing data cleanly.

    Edge weight:  A[i,j] = corr(y_train_i, y_train_j)
    (negative correlations are set to zero — they are rare and physically
     implausible for nearby Finnish stations.)

    Parameters
    ----------
    node_datasets  : dict from build_node_datasets + split/standardise steps
    station_names  : ordered list of node names (fixes row/column ordering)
    threshold      : Pearson correlation threshold ∈ [0, 1]

    Returns
    -------
    A : symmetric (n, n) adjacency matrix
    """
    n = len(station_names)
    A = np.zeros((n, n))

    # Build (day → tmax) series for training period of each station
    series: Dict[str, pd.Series] = {}
    for name in station_names:
        data   = node_datasets[name]
        tr_idx = data["train_idx"]
        # Use raw y (tmax_{t+1}) — equivalent to the tmax time series on training days
        days   = pd.DatetimeIndex([data["days"][i] for i in tr_idx])
        y_vals = data["y"][tr_idx]
        series[name] = pd.Series(y_vals, index=days)

    for i in range(n):
        for j in range(i + 1, n):
            si = series[station_names[i]]
            sj = series[station_names[j]]
            common = si.index.intersection(sj.index)
            if len(common) < 5:
                continue
            corr = float(np.corrcoef(si[common].values, sj[common].values)[0, 1])
            if corr >= threshold:
                A[i, j] = A[j, i] = corr

    return A


def graph_laplacian(A: np.ndarray) -> np.ndarray:
    """
    Weighted Laplacian:  L = D − A,  where D[i,i] = Σ_j A[i,j].

    Key identity used by FedGD:
        (L @ W)[i]  =  Σ_{j ∈ N(i)} A[i,j] (w_i − w_j)
    which equals the gradient of the GTV penalty w.r.t. w_i (divided by 2α).
    """
    degrees = A.sum(axis=1)
    L = np.diag(degrees) - A
    return L


def _is_connected_bfs(A: np.ndarray) -> bool:
    """BFS-based connectivity check (more reliable than eignvalue threshold)."""
    n = A.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    queue = [0]
    while queue:
        node = queue.pop(0)
        for nb in np.where(A[node] > 0)[0]:
            if not visited[nb]:
                visited[nb] = True
                queue.append(nb)
    return bool(visited.all())


def graph_info(A: np.ndarray) -> Dict:
    """
    Return a summary dict of graph properties:
      n_nodes, n_edges, density, min/max/mean weighted degree,
      Fiedler value λ₂, λ_max, is_connected (BFS), n_components.
    """
    n  = A.shape[0]
    L  = graph_laplacian(A)
    ev = np.sort(eigvalsh(L))

    n_edges = int((A > 0).sum()) // 2
    deg     = A.sum(axis=1)

    # Count components via BFS (robust to floating-point eigenvalue noise)
    visited = np.zeros(n, dtype=bool)
    n_comp  = 0
    for start in range(n):
        if visited[start]:
            continue
        n_comp += 1
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop(0)
            for nb in np.where(A[node] > 0)[0]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)

    return {
        "n_nodes":      n,
        "n_edges":      n_edges,
        "density":      round(n_edges / max(n * (n - 1) / 2, 1), 4),
        "degree_min":   float(deg.min()),
        "degree_max":   float(deg.max()),
        "degree_mean":  float(deg.mean()),
        "lambda_2":     float(ev[1]),
        "lambda_max":   float(ev[-1]),
        "n_components": n_comp,
        "is_connected": n_comp == 1,
    }


def largest_connected_component(
    A:             np.ndarray,
    station_names: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract the largest connected component using BFS.

    Returns
    -------
    A_sub    : adjacency matrix restricted to the component
    names_sub: corresponding station names
    """
    n = A.shape[0]
    visited = np.zeros(n, dtype=bool)
    components = []

    for start in range(n):
        if visited[start]:
            continue
        # BFS
        component = []
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop(0)
            component.append(node)
            for nb in np.where(A[node] > 0)[0]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        components.append(component)

    largest = max(components, key=len)
    idx     = np.array(sorted(largest))
    A_sub   = A[np.ix_(idx, idx)]
    names_sub = [station_names[i] for i in idx]
    return A_sub, names_sub


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Local model (linear regression)
# ─────────────────────────────────────────────────────────────────────────────

def mse_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """MSE loss:  (1/m) ‖y − Xw‖²"""
    r = y - X @ w
    return float(r @ r / len(y))


def mse_gradient(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Gradient of MSE loss:  ∇_w L(w) = −(2/m) Xᵀ (y − Xw)

    Derivation:  L(w) = (1/m)‖y−Xw‖²,  ∇L = (2/m) Xᵀ(Xw−y)
    """
    r = y - X @ w
    return -(2.0 / len(y)) * (X.T @ r)


def smoothness_constant(X: np.ndarray) -> float:
    """
    Smoothness (gradient Lipschitz) constant of MSE loss:
        β = (2/m) λ_max(XᵀX)

    The safe gradient-descent step size is  η ≤ 1/β.
    """
    m = X.shape[0]
    XtX = X.T @ X
    lam = eigvalsh(XtX, subset_by_index=[XtX.shape[0] - 1, XtX.shape[0] - 1])
    return 2.0 * float(lam[0]) / m


def compute_safe_stepsize(
    node_datasets:  Dict[str, Dict],
    station_names:  List[str],
    A:              np.ndarray,
    alpha:          float,
) -> float:
    """
    Theoretical safe step size for FedGD:
        η ≤ 1 / max_i (β_i  +  2α · d_i)

    where β_i = smoothness of L_i (from training data)
          d_i = weighted degree of node i in the graph.
    """
    degrees = A.sum(axis=1)
    caps = []
    for idx, name in enumerate(station_names):
        data = node_datasets[name]
        tr   = data["train_idx"]
        beta = smoothness_constant(data["X_std"][tr])
        caps.append(beta + 2.0 * alpha * float(degrees[idx]))
    return 1.0 / max(caps)


def local_baseline(
    node_datasets:  Dict[str, Dict],
    station_names:  List[str],
) -> np.ndarray:
    """
    Closed-form ordinary least-squares solution for each node independently.
    This corresponds to the GTVMin solution with α = 0 (no collaboration).

    Returns W of shape (n, d), where W[i] = (X_iᵀ X_i)⁻¹ X_iᵀ y_i.
    Uses numpy's lstsq for numerical stability (handles near-singular cases).
    """
    n = len(station_names)
    d = node_datasets[station_names[0]]["X_std"].shape[1]
    W = np.zeros((n, d))

    for idx, name in enumerate(station_names):
        data = node_datasets[name]
        tr   = data["train_idx"]
        X, y = data["X_std"][tr], data["y"][tr]
        w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        W[idx] = w

    return W


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — FL algorithms (GTVMin)
# ─────────────────────────────────────────────────────────────────────────────

def _mean_mse(
    W:             np.ndarray,
    node_datasets: Dict[str, Dict],
    station_names: List[str],
    split:         str,
) -> float:
    """Average MSE across all nodes on a given split ('train', 'val', 'test')."""
    return float(np.mean([
        mse_loss(W[i], node_datasets[n]["X_std"][node_datasets[n][f"{split}_idx"]],
                        node_datasets[n]["y"][node_datasets[n][f"{split}_idx"]])
        for i, n in enumerate(station_names)
    ]))


def run_fedgd(
    node_datasets:  Dict[str, Dict],
    station_names:  List[str],
    A:              np.ndarray,
    alpha:          float,
    eta:            float,
    n_iter:         int  = 300,
    verbose:        bool = False,
) -> Dict:
    """
    Federated Gradient Descent (FedGD) — synchronous GTVMin solver.

    Implements the update equation from Lecture 4:

        w_i^{t+1} = w_i^t − η [∇L_i(w_i^t) + 2α (L W^t)_i]

    where L is the graph Laplacian and (LW)_i = Σ_{j∈N(i)} A_ij (w_i − w_j).

    All nodes update SYNCHRONOUSLY using the same iteration index t, which
    means every node reads its neighbours' parameters from round t before
    computing the round-(t+1) update.

    Why FedGD over FedSGD?
        With only ~23 training points per station, the full gradient is cheap
        to compute.  Using mini-batches (SGD) would add variance without saving
        meaningful computation.

    Parameters
    ----------
    node_datasets : output of build_node_datasets + split + standardise
    station_names : ordered list fixing the node↔row correspondence
    A             : (n, n) adjacency matrix
    alpha         : GTVMin regularisation strength (≥ 0)
    eta           : step size (learning rate).  Use compute_safe_stepsize()
                    to obtain a theoretically valid value.
    n_iter        : number of gradient steps
    verbose       : if True, print losses every 50 iterations

    Returns
    -------
    dict with:
        "W"             : final weight matrix, shape (n, d)
        "train_loss"    : list[float], mean MSE on training data per iteration
        "val_loss"      : list[float], mean MSE on validation data per iteration
        "station_names" : same list that was passed in (for bookkeeping)
    """
    n = len(station_names)
    d = node_datasets[station_names[0]]["X_std"].shape[1]
    L = graph_laplacian(A)   # (n, n)

    W = np.zeros((n, d))     # initialise all weights to zero (standard)
    train_losses: List[float] = []
    val_losses:   List[float] = []

    for t in range(n_iter):
        # --- Compute per-node gradients on training data ---
        G = np.zeros((n, d))
        for i, name in enumerate(station_names):
            data = node_datasets[name]
            tr   = data["train_idx"]
            G[i] = mse_gradient(W[i], data["X_std"][tr], data["y"][tr])

        # --- Graph-regularisation (consensus) term: 2α L W ---
        # Vectorised: (L @ W)[i] = Σ_{j∈N(i)} A_ij (w_i − w_j)
        consensus = 2.0 * alpha * (L @ W)

        # --- Synchronous gradient step ---
        W = W - eta * (G + consensus)

        # --- Track losses ---
        train_losses.append(_mean_mse(W, node_datasets, station_names, "train"))
        val_losses.append(_mean_mse(W, node_datasets, station_names, "val"))

        if verbose and (t % 50 == 0 or t == n_iter - 1):
            print(f"  iter {t:4d}  train={train_losses[-1]:.4f}  val={val_losses[-1]:.4f}")

    return {
        "W":             W,
        "train_loss":    train_losses,
        "val_loss":      val_losses,
        "station_names": station_names,
    }


def run_fedrelax(
    node_datasets:  Dict[str, Dict],
    station_names:  List[str],
    A:              np.ndarray,
    alpha:          float,
    n_iter:         int  = 100,
    verbose:        bool = False,
) -> Dict:
    """
    FedRelax — Jacobi-style GTVMin solver (Lecture 4, §4.6).

    Each node solves a LOCAL ridge regression subproblem per round,
    treating neighbours' current weights as fixed targets:

        w_i^{t+1} = argmin_{w}  L_i(w) + α Σ_{j∈N(i)} A_ij ‖w − w_j^t‖²

    For linear regression L_i this has a CLOSED FORM:
        w_i^{t+1} = (XᵀX/m + α·d_i·I)⁻¹ (Xᵀy/m + α Σ_j A_ij w_j^t)

    where d_i = Σ_j A_ij is the weighted degree of node i.

    Convergence: FedRelax with smooth, strongly-convex local losses and α > 0
    is guaranteed to converge (the operator is contractive in ‖·‖_∞).

    Comparison to FedGD
    -------------------
    • FedRelax solves an exact local subproblem each round → fewer rounds to
      converge, but each round is slightly more expensive (matrix inversion).
    • FedGD takes a gradient step → requires careful step-size tuning.
    • For our small problem (d=3) both are practically equivalent in speed.

    Parameters
    ----------
    (same as run_fedgd, except no 'eta' parameter — step size is implicit
     in the closed-form solve)
    """
    n = len(station_names)
    d = node_datasets[station_names[0]]["X_std"].shape[1]

    W = np.zeros((n, d))
    train_losses: List[float] = []
    val_losses:   List[float] = []

    # Pre-compute per-node normal-equation matrices (fixed across iterations)
    node_XtX  = {}  # XᵀX / m
    node_Xty  = {}  # Xᵀy / m
    for i, name in enumerate(station_names):
        data   = node_datasets[name]
        tr     = data["train_idx"]
        X, y   = data["X_std"][tr], data["y"][tr]
        m      = len(y)
        node_XtX[i] = (X.T @ X) / m
        node_Xty[i] = (X.T @ y) / m

    degrees = A.sum(axis=1)   # weighted degree of each node

    for t in range(n_iter):
        W_new = np.zeros((n, d))

        for i, name in enumerate(station_names):
            # Ridge penalty on deviation from neighbours: α·d_i·I
            ridge_coeff = alpha * float(degrees[i])
            A_mat = node_XtX[i] + ridge_coeff * np.eye(d)

            # Pull term: α Σ_j A_ij w_j^t
            pull = np.zeros(d)
            for j in range(n):
                if A[i, j] > 0:
                    pull += alpha * A[i, j] * W[j]

            b_vec = node_Xty[i] + pull
            W_new[i] = np.linalg.solve(A_mat, b_vec)

        W = W_new

        train_losses.append(_mean_mse(W, node_datasets, station_names, "train"))
        val_losses.append(_mean_mse(W, node_datasets, station_names, "val"))

        if verbose and (t % 20 == 0 or t == n_iter - 1):
            print(f"  iter {t:4d}  train={train_losses[-1]:.4f}  val={val_losses[-1]:.4f}")

    return {
        "W":             W,
        "train_loss":    train_losses,
        "val_loss":      val_losses,
        "station_names": station_names,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Evaluation & hyperparameter tuning
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    W:             np.ndarray,
    node_datasets: Dict[str, Dict],
    station_names: List[str],
) -> pd.DataFrame:
    """
    Compute per-station train / val / test MSE.

    Returns a DataFrame with columns:
        station, n_train, n_val, n_test, train_mse, val_mse, test_mse
    """
    rows = []
    for i, name in enumerate(station_names):
        data = node_datasets[name]
        row = {"station": name}
        for split in ("train", "val", "test"):
            idx = data[f"{split}_idx"]
            row[f"n_{split}"]   = len(idx)
            row[f"{split}_mse"] = mse_loss(W[i], data["X_std"][idx], data["y"][idx])
        rows.append(row)
    return pd.DataFrame(rows)


def tune_alpha(
    node_datasets:  Dict[str, Dict],
    station_names:  List[str],
    A:              np.ndarray,
    alpha_grid:     List[float],
    n_iter:         int  = 300,
    algorithm:      str  = "fedgd",
) -> Tuple[float, List[float]]:
    """
    Grid search over the regularisation parameter α using VALIDATION MSE.

    For each α:
      1. Compute the theoretically safe step size η (FedGD only).
      2. Run the chosen algorithm for n_iter iterations.
      3. Record the final validation MSE.

    The best α is the one minimising validation MSE.

    Parameters
    ----------
    algorithm : 'fedgd' or 'fedrelax'

    Returns
    -------
    best_alpha : float
    val_mse_list : list[float]  — one entry per alpha in alpha_grid
    """
    val_mses: List[float] = []

    print(f"Tuning α  (algorithm={algorithm}):")
    for alpha in alpha_grid:
        if algorithm == "fedgd":
            eta    = compute_safe_stepsize(node_datasets, station_names, A, alpha)
            result = run_fedgd(node_datasets, station_names, A, alpha, eta, n_iter)
        elif algorithm == "fedrelax":
            result = run_fedrelax(node_datasets, station_names, A, alpha, n_iter)
        else:
            raise ValueError(f"Unknown algorithm '{algorithm}'")

        val_mse = result["val_loss"][-1]
        val_mses.append(val_mse)
        print(f"  α={alpha:.4g}  →  val MSE={val_mse:.4f}")

    best_idx = int(np.argmin(val_mses))
    print(f"  → best α = {alpha_grid[best_idx]}")
    return alpha_grid[best_idx], val_mses


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_station_map(
    node_datasets:  Dict[str, Dict],
    station_names:  List[str],
    title:          str = "FMI weather stations",
    ax=None,
):
    """Scatter plot of station geographic locations."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 8))
    lats = [node_datasets[n]["lat"] for n in station_names]
    lons = [node_datasets[n]["lon"] for n in station_names]
    ax.scatter(lons, lats, s=15, c="steelblue", alpha=0.8, zorder=3)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_graph(
    node_datasets:  Dict[str, Dict],
    station_names:  List[str],
    A:              np.ndarray,
    title:          str = "FL network",
    ax=None,
):
    """Draw the FL graph: stations as dots, edges as lines (weight → thickness)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 8))

    lats = [node_datasets[n]["lat"] for n in station_names]
    lons = [node_datasets[n]["lon"] for n in station_names]
    n    = len(station_names)

    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] > 0:
                ax.plot(
                    [lons[i], lons[j]], [lats[i], lats[j]],
                    color="gray",
                    linewidth=0.3 + 1.2 * float(A[i, j] / A.max()),
                    alpha=0.5, zorder=2,
                )

    ax.scatter(lons, lats, s=15, c="steelblue", zorder=3)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_loss_curves(result: Dict, title: str = "FedGD loss curves", ax=None):
    """Plot training and validation MSE as a function of iteration."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(result["train_loss"], label="Train MSE", linewidth=1.5)
    ax.plot(result["val_loss"],   label="Val MSE",   linewidth=1.5, linestyle="--")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean MSE (°C²)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_alpha_tuning(
    alpha_grid:  List[float],
    val_mses:    List[float],
    best_alpha:  float,
    title:       str = "α tuning",
    ax=None,
):
    """Validation MSE vs. regularisation α (log-scale x-axis)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.semilogx(alpha_grid, val_mses, "o-", linewidth=2, markersize=8)
    ax.axvline(best_alpha, color="tomato", linestyle="--",
               label=f"best α = {best_alpha}")
    ax.set_xlabel("α (regularisation strength)")
    ax.set_ylabel("Validation MSE (°C²)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_per_station_mse(
    results_dict: Dict[str, pd.DataFrame],
    split:        str = "test_mse",
    figsize:      Tuple = (14, 4),
):
    """
    Bar chart comparing per-station MSE across multiple systems.

    Parameters
    ----------
    results_dict : {"System name": DataFrame from evaluate(), ...}
    split        : column to plot ('train_mse', 'val_mse', 'test_mse')
    """
    fig, ax = plt.subplots(figsize=figsize)
    names   = list(results_dict.keys())

    # Use first system as reference for station ordering
    stations = results_dict[names[0]]["station"].tolist()
    x        = np.arange(len(stations))
    w        = 0.8 / len(names)

    for k, sysname in enumerate(names):
        df   = results_dict[sysname].set_index("station")
        mses = [df.loc[s, split] for s in stations]
        ax.bar(x + k * w, mses, w, label=sysname, alpha=0.8)

    ax.set_xticks(x + w * (len(names) - 1) / 2)
    ax.set_xticklabels(stations, rotation=90, fontsize=6)
    ax.set_ylabel(f"{split} (°C²)")
    ax.set_title(f"Per-station {split} by system")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig, ax


def summary_table(results_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Print a compact mean-MSE comparison table.

    Parameters
    ----------
    results_dict : {"System name": DataFrame from evaluate(), ...}

    Returns a DataFrame with columns: System, Train MSE, Val MSE, Test MSE.
    """
    rows = []
    for name, df in results_dict.items():
        rows.append({
            "System":    name,
            "Train MSE": round(df["train_mse"].mean(), 4),
            "Val MSE":   round(df["val_mse"].mean(),   4),
            "Test MSE":  round(df["test_mse"].mean(),  4),
        })
    return pd.DataFrame(rows).set_index("System")
