"""
Author: Zeyu Wang
Date: October 2026
Description: "analytical ptychogrphay"
"""

import math
import numpy as np
import time
import torch

from .utils import print_and_log, log_to_file, get_subplot_grid, RGB_Complex_Plot, calculate_wavelength, Chi_defocus





# ==========================================================================
# Common helpers
# ==========================================================================

def _resolve_device(device):

    if device is None or device == 'None':
        return torch.device('cpu')

    return torch.device(device)




def _f64(x):

    # MPS has no float64, fall back to CPU for the (small) high-precision reductions.
    if x.device.type == 'mps':
        return x.cpu().double()

    return x.double()




def _prepare_matrix(data_4D, nonneg=False):

    sy, sx, dy, dx = data_4D.shape
    X_np = np.ascontiguousarray(data_4D.reshape(-1, dy * dx), dtype=np.float32)

    # Only copy when there is something to fix, the input array is never modified.
    if not np.isfinite(X_np).all():
        print_and_log('Warning: input data contains NaN/Inf, they are replaced by 0.')
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)

    if nonneg and X_np.min() < 0:
        X_np = np.maximum(X_np, 0)

    return X_np




class BlockedMatrix:

    """
    Row-blocked view of a CPU matrix X (N x M).
    Iterating yields (start, end, X_block_on_device).
    If the whole matrix fits into one block and the device is not the CPU,
    X is moved to the device only once and reused by every pass.
    n_block is the minimum number of blocks, the block size is also capped by max_block_bytes.
    """

    def __init__(self, X_np, n_block, device, max_block_bytes=2**30):

        if not isinstance(n_block, int) or n_block < 1:
            raise ValueError('n_block must be a positive integer.')

        self.X = torch.from_numpy(X_np)
        self.N, self.M = self.X.shape
        self.device = torch.device(device)

        rows_max = max(1, int(max_block_bytes // (4 * self.M)))
        self.block_size = max(1, min((self.N + n_block - 1) // n_block, rows_max))

        self.cached = None
        if self.block_size >= self.N and self.device.type != 'cpu':
            self.cached = self.X.to(self.device)

    def __iter__(self):

        if self.cached is not None:
            yield 0, self.N, self.cached
            return

        for start in range(0, self.N, self.block_size):
            end = min(start + self.block_size, self.N)
            yield start, end, self.X[start:end].to(self.device, non_blocking=True)




def _chunked_norms(X, chunk=1 << 20):

    # L2 norms of fixed-size chunks (short accumulation chains, float64 inside the chunk where the device has it);
    # the caller sums their squares in float64. A single float32 reduction over ~1e8 elements can be off by several percent.
    flat = X.reshape(-1)
    dt = None if X.device.type == 'mps' else torch.float64

    return torch.stack([torch.linalg.vector_norm(flat[i:i + chunk], dtype=dt) for i in range(0, flat.numel(), chunk)])




@torch.no_grad()
def _exact_residual_sq(blocks, Wt, H, rows=1024):

    # ||X - W H||_F^2 computed explicitly (one extra pass), used when the fast formula is unreliable.
    parts = []

    for start, end, Xb in blocks:
        for a in range(0, end - start, rows):
            b = min(a + rows, end - start)
            R = Xb[a:b] - Wt[:, start + a:start + b].T @ H
            parts.append(_chunked_norms(R))

    return (torch.cat(parts).cpu().double() ** 2).sum().item()




@torch.no_grad()
def _block_stats(blocks):

    # Column sums (float64, CPU) and ||X||_F^2 (float64). One pass, no per-block sync.
    sums = []
    norms = []

    for start, end, Xb in blocks:
        sums.append(Xb.sum(dim=0))
        norms.append(_chunked_norms(Xb))

    col_sum = torch.stack(sums).cpu().double().sum(dim=0)
    sum_sq = (torch.cat(norms).cpu().double() ** 2).sum().item()

    return col_sum, sum_sq




def _progress_str(label, i, iter_max, time_i0, start_time):

    time_is = time.perf_counter() - time_i0
    speed = 1 / max(time_is, 1e-12)
    process = (i + 1) / iter_max * 100
    aa = "*" * int(process / 10)
    bb = "." * (10 - int(process / 10))
    dur = int(time.perf_counter() - start_time)
    time_remain = int((iter_max - i - 1) * time_is)
    dur_h, dur_rem = divmod(dur, 3600)
    dur_m, dur_s = divmod(dur_rem, 60)
    time_remain_h, time_remain_rem = divmod(time_remain, 3600)
    time_remain_m, time_remain_s = divmod(time_remain_rem, 60)

    return "{} progressing: {:^3.0f}%[{}->{}] {:.2f}iter/s ({:0>2}:{:0>2}:{:0>2}<{:0>2}:{:0>2}:{:0>2})".format(label, process, aa, bb, speed, dur_h, dur_m, dur_s, time_remain_h, time_remain_m, time_remain_s)




def _mean_abs_corr(F):

    # F: (K, n), every row is one component.
    K = F.shape[0]

    if K < 2:
        return 0.0

    C = torch.nan_to_num(torch.corrcoef(_f64(F)))
    mask = ~torch.eye(K, dtype=torch.bool, device=C.device)

    return C[mask].abs().mean().item()




def _fix_sign(U, V):

    # Deterministic sign convention: the largest-magnitude entry of every V column is positive.
    idx = torch.argmax(V.abs(), dim=0)
    sign = torch.sign(V[idx, torch.arange(V.shape[1], device=V.device)])
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)

    return U * sign[None, :], V * sign[None, :]


def _check_output(W, H, label):

    # Report NaN/Inf and all-zero components instead of failing silently later in the plots.
    if not (np.isfinite(W).all() and np.isfinite(H).all()):
        print_and_log(f'Warning: {label} output contains NaN/Inf.')

    dead = [k for k in range(W.shape[0]) if not W[k].any() or not H[k].any()]

    if len(dead) > 0:
        print_and_log(f'Warning: {label} components {dead} are entirely zero.')






# ==========================================================================
# Randomized PCA core, shared by PCA / ICA / NNDSVD
# ==========================================================================

@torch.no_grad()
def _randomized_pca(blocks, K, n_oversamples, n_power_iter, dev, random_state, center=True, stats=None):

    """
    Randomized truncated SVD of Xc (centered if center=True) without ever forming Xc.
    Subspace iteration ends on the right side (Xc ~ Xc Z Z^T), so no extra pass is
    needed for B = Q^T Xc or for the scores.

    Number of passes over X: 2 * n_power_iter + 1  (+1 if stats is not given).

    Returns U (N x K, orthonormal columns), S (K), V (M x K), mean (M), ||Xc||_F^2.
    """

    N, M = blocks.N, blocks.M
    L = min(K + n_oversamples, N, M)

    if stats is None:
        stats = _block_stats(blocks)

    col_sum, sum_sq = stats

    if center:
        mean64 = col_sum / N
        mean = mean64.to(torch.float32).to(dev)
        total_sq = max(sum_sq - N * mean64.square().sum().item(), 0.0)
    else:
        mean = torch.zeros(M, dtype=torch.float32, device=dev)
        total_sq = sum_sq

    # Random start generated on the CPU so that the result does not depend on the device.
    g = torch.Generator().manual_seed(random_state)
    Z = torch.randn(M, L, generator=g, dtype=torch.float32).to(dev)
    Z, _ = torch.linalg.qr(Z, mode='reduced')

    for i in range(n_power_iter + 1):

        # Y = Xc @ Z = X @ Z - 1 (mean @ Z)
        mZ = mean @ Z
        Y = torch.empty((N, L), dtype=torch.float32, device=dev)

        for start, end, Xb in blocks:
            Y[start:end] = Xb @ Z - mZ

        Q, R = torch.linalg.qr(Y, mode='reduced')

        if i < n_power_iter:
            # Z = Xc.T @ Q = X.T @ Q - mean (1^T Q)
            Zn = torch.zeros((M, L), dtype=torch.float32, device=dev)

            for start, end, Xb in blocks:
                Zn += Xb.T @ Q[start:end]

            Zn -= mean[:, None] * Q.sum(dim=0)[None, :]
            Z, _ = torch.linalg.qr(Zn, mode='reduced')

    # Xc ~ Y Z^T = Q R Z^T = (Q Ur) S (Z Vr)^T
    Ur, S_all, Vrh = torch.linalg.svd(R, full_matrices=False)

    U = Q @ Ur[:, :K]
    S = S_all[:K]
    V = Z @ Vrh[:K].T

    return U, S, V, mean, total_sq





# ==========================================================================
# NMF
# ==========================================================================

@torch.no_grad()
def nndsvd_init(X, rank, seed=0, device=None, stats=None):

    # X can be a CPU torch tensor (N x M) or a BlockedMatrix.
    if isinstance(X, BlockedMatrix):
        blocks = X
        dev = blocks.device
    else:
        dev = _resolve_device(device)
        X_np = np.ascontiguousarray(torch.as_tensor(X).cpu().numpy(), dtype=np.float32)
        blocks = BlockedMatrix(X_np, 1, dev)

    N, M = blocks.N, blocks.M

    if stats is None:
        stats = _block_stats(blocks)

    avg = stats[0].sum().item() / (N * M)

    # Leading singular triplets of X itself (no centering).
    U, S, V, _, _ = _randomized_pca(blocks, rank, 10, 2, dev, seed, center=False, stats=stats)

    # Positive / negative parts of every singular vector pair.
    u_pos = U.clamp(min=0)
    u_neg = (-U).clamp(min=0)
    v_pos = V.clamp(min=0)
    v_neg = (-V).clamp(min=0)

    n_up = torch.linalg.vector_norm(u_pos, dim=0)
    n_un = torch.linalg.vector_norm(u_neg, dim=0)
    n_vp = torch.linalg.vector_norm(v_pos, dim=0)
    n_vn = torch.linalg.vector_norm(v_neg, dim=0)

    # Choose the positive or negative parts according to the larger product of their norms.
    use_pos = (n_up * n_vp) >= (n_un * n_vn)

    u_sel = torch.where(use_pos[None, :], u_pos, u_neg)
    v_sel = torch.where(use_pos[None, :], v_pos, v_neg)
    norm_u = torch.where(use_pos, n_up, n_un)
    norm_v = torch.where(use_pos, n_vp, n_vn)

    valid = ((norm_u > 1e-12) & (norm_v > 1e-12)).to(torch.float32)
    scale = torch.sqrt(S * norm_u * norm_v)

    W = u_sel / norm_u.clamp_min(1e-12)[None, :] * (scale * valid)[None, :]
    Hm = v_sel / norm_v.clamp_min(1e-12)[None, :] * (scale * valid)[None, :]

    # First component: absolute leading singular vectors.
    W[:, 0] = torch.sqrt(S[0]) * U[:, 0].abs()
    Hm[:, 0] = torch.sqrt(S[0]) * V[:, 0].abs()

    H = Hm.T.contiguous()

    # Small positive initialization for zero entries.
    eps = max(avg * 1e-3, 1e-6)
    W[W == 0] = eps
    H[H == 0] = eps

    return W, H




@torch.no_grad()
def _hals_nmf_torch(blocks, n_components, iter_max, tol, n_inner, alpha_W, alpha_H, l1_ratio, stats, init=None, label='NMF', random_state=0):

    """
    Fast HALS NMF on the device. Wt is stored as (K, N) so that every row update is contiguous.
    n_inner > 1 gives accelerated HALS: several coordinate sweeps are done on the small
    Gram matrices per pass over X, which is almost free compared with a pass over X.

    init = (Wt (K, N), H (K, M)) continues from given factors, otherwise NNDSVDa.
    Returns Wt (K, N), H (K, M), err_list.
    """

    dev = blocks.device
    N, M = blocks.N, blocks.M
    K = n_components
    eps = 1e-12

    col_sum, norm_X_sq = stats
    avg = col_sum.sum().item() / (N * M)
    init_eps = max(avg * 1e-3, 1e-6)

    if init is None:
        W0, H = nndsvd_init(blocks, K, seed=random_state, stats=stats)
        Wt = W0.T.contiguous()
    else:
        Wt = init[0].to(dev, torch.float32).contiguous().clone()
        H = init[1].to(dev, torch.float32).contiguous().clone()

    reg_W_l1 = alpha_W * M * l1_ratio
    reg_W_l2 = alpha_W * M * (1 - l1_ratio)
    reg_H_l1 = alpha_H * N * l1_ratio
    reg_H_l2 = alpha_H * N * (1 - l1_ratio)

    Q = torch.empty((K, M), dtype=torch.float32, device=dev)
    Pt = torch.empty((K, N), dtype=torch.float32, device=dev)

    err_list = []
    previous_err = None
    last_line = None
    n_reinit = 0
    n_fallback = 0

    start_time = time.perf_counter()
    print_and_log('')
    print_and_log(f'####################### {label} Process ######################')
    print("\r{} progressing: {:^3.0f}%[{}->{}] ?iter/s ({:0>2}:{:0>2}:{:0>2}<??:??:??)".format(label, 0, "*" * 0, "." * 10, 0, 0, 0), end="")

    for i in range(iter_max):

        time_i0 = time.perf_counter()

        # ----------------------------------------------------
        # Update H
        # Q = W.T @ X, A = W.T @ W
        # ----------------------------------------------------
        Q.zero_()

        for start, end, Xb in blocks:
            Q += Wt[:, start:end] @ Xb

        A = Wt @ Wt.T

        for _ in range(n_inner):
            for k in range(K):
                denom = (A[k, k] + reg_H_l2).clamp_min(eps)
                H[k] = torch.clamp(H[k] + (Q[k] - A[k] @ H - reg_H_l1 - reg_H_l2 * H[k]) / denom, min=0)

        # ----------------------------------------------------
        # Update W
        # Pt = H @ X.T, B = H @ H.T
        # ----------------------------------------------------
        for start, end, Xb in blocks:
            Pt[:, start:end] = H @ Xb.T

        B = H @ H.T

        for _ in range(n_inner):
            for k in range(K):
                denom = (B[k, k] + reg_W_l2).clamp_min(eps)
                Wt[k] = torch.clamp(Wt[k] + (Pt[k] - B[k] @ Wt - reg_W_l1 - reg_W_l2 * Wt[k]) / denom, min=0)

        # ----------------------------------------------------
        # Fast error calculation: reuse Pt = H @ X.T (it only depends on H)
        # ||X - WH||^2 = ||X||^2 - 2 <Pt,Wt> + <Wt Wt^T, H H^T>
        # The three terms nearly cancel, so they are evaluated in float64.
        # ----------------------------------------------------
        A_final = Wt @ Wt.T

        cross_term = (_f64(Pt) * _f64(Wt)).sum()
        model_norm_sq = (_f64(A_final) * _f64(B)).sum()
        cross_term, model_norm_sq = torch.stack([cross_term, model_norm_sq]).cpu().tolist()

        residual_sq = norm_X_sq - 2.0 * cross_term + model_norm_sq

        # A residual that is non-finite or ~0 means the three terms cancelled completely: do not trust it.
        if (not math.isfinite(residual_sq)) or residual_sq <= 1e-9 * norm_X_sq:
            exact_sq = _exact_residual_sq(blocks, Wt, H)

            if n_fallback < 3:
                print_and_log(f'{label}: fast error estimate unreliable at iteration {i}: formula = {residual_sq:.6e}, explicit = {exact_sq:.6e}, ||X||^2 = {norm_X_sq:.6e}, <P,W> = {cross_term:.6e}, ||WH||^2 = {model_norm_sq:.6e}')

            n_fallback += 1
            residual_sq = exact_sq

        residual_sq = max(residual_sq, 0.0)

        err = math.sqrt(residual_sq / max(norm_X_sq, eps))
        err_list.append(err)

        # ----------------------------------------------------
        # A collapsed (all zero) component can never recover by itself: restart it.
        # ----------------------------------------------------
        dead = (H.sum(dim=1) <= 0) | (Wt.sum(dim=1) <= 0)

        if dead.any():
            idx = dead.nonzero().flatten()
            H[idx] = init_eps * (1 + torch.rand((idx.numel(), M), device=dev))
            Wt[idx] = init_eps * (1 + torch.rand((idx.numel(), N), device=dev))
            n_reinit += int(idx.numel())

        # Convergence check
        if not math.isfinite(err):
            print('')
            print_and_log(f'{label}: error is not finite at iteration {i}, stopping.')
            break

        if err <= 0.0:
            print('')
            print_and_log(f'{label}: exact fit (error = 0), stopping.')
            break

        if previous_err is not None:
            relative_change = abs(previous_err - err) / max(previous_err, eps)
            if relative_change < tol:
                print('')
                print('error relative change < tol, converged')
                break

        previous_err = err

        last_line = _progress_str(label, i, iter_max, time_i0, start_time)
        print("\r" + last_line, end="")

    print('')

    if last_line is not None:
        log_to_file(last_line)

    if n_reinit > 0:
        print_and_log(f'{label}: {n_reinit} collapsed component(s) were re-initialized.')

    gpu_time = time.perf_counter() - start_time
    print_and_log(f'{label} process finished in {gpu_time} s')

    return Wt, H, err_list




def _log_nmf_params(dev, n_components, iter_max, tol, n_block, alpha_W, alpha_H, l1_ratio, n_inner):

    print_and_log('')
    print_and_log('##################### NMF Parameters ######################')
    print_and_log(f"Device Using: {dev}")
    print_and_log(f'NMF Components Number: {n_components}')
    print_and_log(f'Number of Iterations: {iter_max}')
    print_and_log(f'NMF Convergence Tolerance: {tol}')
    print_and_log(f'NMF Block Number: {n_block}')
    print_and_log(f'NMF alpha_W: {alpha_W}')
    print_and_log(f'NMF alpha_H: {alpha_H}')
    print_and_log(f'NMF L1 Ratio: {l1_ratio}')
    print_and_log(f'NMF Inner Sweeps per Pass: {n_inner}')




@torch.no_grad()
def fast_hals_NMF(data_4D, n_components, iter_max=500, tol=1e-7, n_block=1, alpha_W=0.0, alpha_H=0.0, l1_ratio=0.0, device=None, n_inner=3, random_state=0):

    dev = _resolve_device(device)

    _log_nmf_params(dev, n_components, iter_max, tol, n_block, alpha_W, alpha_H, l1_ratio, n_inner)

    sy, sx, dy, dx = data_4D.shape

    # Nonnegative input is required by NMF.
    X_np = _prepare_matrix(data_4D, nonneg=True)

    N, M = X_np.shape
    K = n_components

    if not isinstance(K, int) or not 1 <= K <= min(N, M):
        raise ValueError(f'n_components must be between 1 and {min(N, M)}.')

    if not isinstance(n_inner, int) or n_inner < 1:
        raise ValueError('n_inner must be a positive integer.')

    blocks = BlockedMatrix(X_np, n_block, dev)
    stats = _block_stats(blocks)

    Wt, H, err_list = _hals_nmf_torch(blocks, K, iter_max, tol, n_inner, alpha_W, alpha_H, l1_ratio, stats, random_state=random_state)

    W = Wt.cpu().numpy().reshape(K, sy, sx)
    H = H.cpu().numpy().reshape(K, dy, dx)

    _check_output(W, H, 'NMF')

    return W, H, err_list





# ==========================================================================
# PCA
# ==========================================================================

# Perform PCA using randomized SVD.
@torch.no_grad()
def PCA(data_4D, n_components, n_block=1, device=None, n_oversamples=10, n_power_iter=2, random_state=0):

    dev = _resolve_device(device)

    print_and_log('')
    print_and_log('##################### PCA Parameters ######################')
    print_and_log(f'Device Using: {dev}')
    print_and_log(f'PCA Components Number: {n_components}')
    print_and_log(f'PCA Block Number: {n_block}')
    print_and_log(f'PCA Oversamples: {n_oversamples}')
    print_and_log(f'PCA Power Iterations: {n_power_iter}')

    start_time = time.perf_counter()

    # --------------------------------------------------
    # 0. Prepare data
    # --------------------------------------------------
    sy, sx, dy, dx = data_4D.shape
    X_np = _prepare_matrix(data_4D)

    N, M = X_np.shape
    K = n_components

    if not isinstance(K, int) or not 1 <= K <= min(N - 1, M):
        raise ValueError(f'n_components must be between 1 and {min(N - 1, M)}.')

    if not isinstance(n_oversamples, int) or n_oversamples < 0:
        raise ValueError('n_oversamples must be a non-negative integer.')

    if not isinstance(n_power_iter, int) or n_power_iter < 0:
        raise ValueError('n_power_iter must be a non-negative integer.')

    blocks = BlockedMatrix(X_np, n_block, dev)

    # --------------------------------------------------
    # 1. Randomized SVD of the centered data
    # Xc ~ U S V^T,  H = V^T,  W = Xc @ V = U S
    # --------------------------------------------------
    U, S, V, mean, total_sq = _randomized_pca(blocks, K, n_oversamples, n_power_iter, dev, random_state, center=True)

    U, V = _fix_sign(U, V)

    # Eigenvalues of covariance matrix C = Xc.T @ Xc / N
    eigvals = S.square() / N
    Lambda = torch.diag(eigvals)

    explained_ratio = S.square().sum().item() / max(total_sq, 1e-12)

    # --------------------------------------------------
    # 2. Reshape and return
    # --------------------------------------------------
    W = (U * S[None, :]).T.reshape(K, sy, sx).cpu().numpy()
    H = V.T.reshape(K, dy, dx).cpu().numpy()
    Lambda = Lambda.cpu().numpy()

    print_and_log(f'Explained Variance Ratio: {explained_ratio:.6f}')
    print_and_log(f'PCA process finished in {time.perf_counter() - start_time:.2f} s')

    _check_output(W, H, 'PCA')

    return W, H, Lambda





# ==========================================================================
# ICA
# ==========================================================================

# Perform FastICA with randomized PCA whitening.
@torch.no_grad()
def fast_ICA(data_4D, n_components, iter_max=500, tol=1e-7, n_block=1, device=None, n_oversamples=10, n_power_iter=2, random_state=0):

    dev = _resolve_device(device)

    print_and_log('')
    print_and_log('###################### ICA Parameters ######################')
    print_and_log(f'Device Using: {dev}')
    print_and_log(f'ICA Components Number: {n_components}')
    print_and_log(f'Number of Iterations: {iter_max}')
    print_and_log(f'ICA Convergence Tolerance: {tol}')
    print_and_log(f'ICA Block Number: {n_block}')
    print_and_log(f'PCA Oversamples: {n_oversamples}')
    print_and_log(f'PCA Power Iterations: {n_power_iter}')

    start_time = time.perf_counter()

    # --------------------------------------------------
    # 0. Prepare data
    # --------------------------------------------------
    sy, sx, dy, dx = data_4D.shape
    X_np = _prepare_matrix(data_4D)

    N, M = X_np.shape
    K = n_components

    if not isinstance(K, int) or not 1 <= K <= min(N - 1, M):
        raise ValueError(f'n_components must be between 1 and {min(N - 1, M)}.')

    if not isinstance(n_oversamples, int) or n_oversamples < 0:
        raise ValueError('n_oversamples must be a non-negative integer.')

    if not isinstance(n_power_iter, int) or n_power_iter < 0:
        raise ValueError('n_power_iter must be a non-negative integer.')

    if iter_max < 1 or tol <= 0:
        raise ValueError('iter_max must be positive and tol must be positive.')

    blocks = BlockedMatrix(X_np, n_block, dev)

    # --------------------------------------------------
    # 1. Randomized PCA whitening
    # Xc ~ U S V^T,  whitened data Z = U sqrt(N),  dewhitening = V S / sqrt(N)
    # --------------------------------------------------
    U, S, V, mean, total_sq = _randomized_pca(blocks, K, n_oversamples, n_power_iter, dev, random_state, center=True)

    eigvals = S.square() / N

    if eigvals.min().item() < 1e-6 * eigvals.max().item():
        print_and_log('Warning: some retained PCA components have tiny variance, whitening will amplify noise. Consider fewer ICA components.')

    eigvals = eigvals.clamp_min(1e-12)
    dewhitening = V * torch.sqrt(eigvals)[None, :]

    # The fixed-point iteration is tiny (N x K), use float64 whenever the device has it.
    work_dtype = torch.float64 if dev.type in ('cpu', 'cuda') else torch.float32
    Z = (U * math.sqrt(N)).to(work_dtype)

    explained_ratio = S.square().sum().item() / max(total_sq, 1e-12)

    print_and_log(f'PCA Explained Variance Ratio: {explained_ratio:.6f}')
    print_and_log(f'PCA Whitening Finished in {time.perf_counter() - start_time:.2f} s')
    print_and_log('')

    # --------------------------------------------------
    # 2. Symmetric FastICA fixed-point iteration
    # --------------------------------------------------
    g = torch.Generator().manual_seed(random_state)
    Q_ica, _ = torch.linalg.qr(torch.randn(K, K, generator=g, dtype=work_dtype), mode='reduced')
    Q_ica = Q_ica.to(dev)

    converged = False
    iteration_count = 0
    lim = float('inf')

    for iteration in range(iter_max):

        Y_ica = Z @ Q_ica.T

        # g(y) = tanh(y), g'(y) = 1 - tanh(y)^2
        G = torch.tanh(Y_ica)
        Gp = 1.0 - G.square()

        Q_new = (G.T @ Z) / N - Gp.mean(dim=0)[:, None] * Q_ica

        # Symmetric decorrelation: Q_new <- (Q_new Q_new.T)^(-1/2) Q_new
        # K x K SVD is faster on the CPU.
        U_ica, _, Vh_ica = torch.linalg.svd(Q_new.cpu(), full_matrices=False)
        Q_new = (U_ica @ Vh_ica).to(dev)

        lim = torch.max(torch.abs(torch.abs(torch.diagonal(Q_new @ Q_ica.T)) - 1)).item()

        Q_ica = Q_new
        iteration_count = iteration + 1

        if lim < tol:
            converged = True
            break

    if converged:
        print_and_log(f'FastICA converged at iteration {iteration_count} / {iter_max}.')
        print_and_log(f'lim = {lim:.6e} < tol = {tol}.')
    else:
        print_and_log(f'Warning: FastICA did not converge within {iter_max} iterations.')
        print_and_log(f'lim = {lim:.6e} > tol = {tol}.')

    # --------------------------------------------------
    # 3. Spatial sources and detector-space patterns
    # --------------------------------------------------
    Y_ica = Z @ Q_ica.T
    H = Q_ica @ dewhitening.to(work_dtype).T

    # Sign: positive skewness. Order: by the strength of the detector pattern.
    skew = (Y_ica ** 3).mean(dim=0)
    sign = torch.where(skew >= 0, torch.ones_like(skew), -torch.ones_like(skew))

    Y_ica = Y_ica * sign[None, :]
    H = H * sign[:, None]

    order = torch.argsort(torch.linalg.vector_norm(H, dim=1), descending=True)

    W32 = Y_ica[:, order].to(torch.float32)
    H32 = H[order].to(torch.float32)

    print_and_log('')
    print_and_log(f'ICA Total Process Finished in {time.perf_counter() - start_time:.2f} s')

    W = W32.T.reshape(K, sy, sx).cpu().numpy()
    H = H32.reshape(K, dy, dx).cpu().numpy()

    _check_output(W, H, 'ICA')

    return W, H





# ==========================================================================
# Nonnegative ICA
# ==========================================================================

@torch.no_grad()
def nonnegative_ica_rotation(W0, H0, n_components=None, beta=1.0, neg_tol=0.0, iter_max=1000, tol=1e-8, step_init=1.0, device='cpu'):

    """
    Nonnegative ICA by rotating a nonnegative factorization X ~ W0 @ H0
    (W0: N x K >= 0, H0: K x M >= 0), Plumbley-style.

    1. Whiten W0 with its centered covariance but keep the mean:
           W0 ~ z @ L,   z = W0 @ Linv,   Cov(z) = I
    2. Nonnegative, independent (hence uncorrelated) unit-variance sources satisfy S = z @ Q
       with Q orthogonal. The mean offset is automatically contained in z.
    3. The same Q rotates the detector side: H = Q^T @ (L @ H0), because S @ H = z @ L @ H0.
    4. Q minimizes the energy of the negative parts of both factors:
           J(Q) = ||min(zQ + tau, 0)||^2 / ||z||^2 + beta ||min(Q^T L H0 + tau, 0)||^2 / ||L H0||^2
       (both norms are invariant under rotation), on the orthogonal group by
       Riemannian gradient descent with an adaptive step.

    n_components = R < K additionally projects on the R leading directions of W0.
    Only R x R matrices are optimized, the cost is negligible compared with the NMF.

    Returns S (N x R), H (R x M) as float32 CPU tensors, and an info dict.
    """

    dev = _resolve_device(device)

    if dev.type == 'mps':
        dev = torch.device('cpu')

    dt = torch.float64
    W0 = torch.as_tensor(W0).detach().cpu().to(dt).to(dev)
    H0 = torch.as_tensor(H0).detach().cpu().to(dt).to(dev)

    N, K = W0.shape
    R = K if n_components is None else int(min(max(n_components, 1), K))

    # ---- whitening without removing the mean -----------------------------
    Wc = W0 - W0.mean(dim=0, keepdim=True)
    C = Wc.T @ Wc / N
    ev, E = torch.linalg.eigh(C)
    ev = ev.flip(0)
    E = E.flip(1)
    ev = ev.clamp_min(1e-12 * max(ev[0].item(), 1e-30))

    E_R = E[:, :R]
    sq = ev[:R].sqrt()

    Linv = E_R / sq[None, :]                # (K, R)
    L = (E_R * sq[None, :]).T               # (R, K)

    if R == K:
        # Symmetric (ZCA) whitening keeps z closest to W0, Q = I is then the NMF solution itself.
        Linv = Linv @ E_R.T
        L = E_R @ L

    z = W0 @ Linv                           # (N, R)
    Hh = L @ H0                             # (R, M)

    nz = z.square().sum().clamp_min(1e-30)
    nh = Hh.square().sum().clamp_min(1e-30)
    tau_z = neg_tol * (nz / z.numel()).sqrt()
    tau_h = neg_tol * (nh / Hh.numel()).sqrt()

    def cost(Q):
        Y = z @ Q
        Hq = Q.T @ Hh
        return (Y + tau_z).clamp(max=0).square().sum() / nz + beta * (Hq + tau_h).clamp(max=0).square().sum() / nh

    def grad(Q):
        Y = z @ Q
        Hq = Q.T @ Hh
        return 2 * z.T @ (Y + tau_z).clamp(max=0) / nz + beta * 2 * Hh @ (Hq + tau_h).clamp(max=0).T / nh

    Q = torch.eye(R, dtype=dt, device=dev)
    f = cost(Q)
    step = step_init
    n_iter = 0

    for it in range(iter_max):

        G = grad(Q)
        Om = G @ Q.T - Q @ G.T              # skew-symmetric Riemannian gradient

        improved = False

        while step > 1e-12:
            Q_new = torch.linalg.matrix_exp(-step * Om) @ Q
            f_new = cost(Q_new)

            if f_new < f:
                improved = True
                break

            step *= 0.5

        if not improved:
            break

        rel = ((f - f_new) / f.clamp_min(1e-30)).item()

        Q, f = Q_new, f_new
        step *= 1.5
        n_iter = it + 1

        if rel < tol:
            break

    S = (z @ Q).to(torch.float32).cpu()
    H = (Q.T @ Hh).to(torch.float32).cpu()

    info = {'cost': f.item(), 'iters': n_iter, 'min_S': S.min().item(), 'min_H': H.min().item(), 'neg_frac_S': (S < 0).float().mean().item(), 'neg_frac_H': (H < 0).float().mean().item()}

    print_and_log(f'Nonnegative ICA rotation: iterations = {n_iter}, cost = {info["cost"]:.6e}')
    print_and_log(f'Negative entries: S {info["neg_frac_S"]:.4%} (min {info["min_S"]:.3e}), H {info["neg_frac_H"]:.4%} (min {info["min_H"]:.3e})')

    return S, H, info




@torch.no_grad()
def fast_hals_NMF_independent(data_4D, n_components, iter_max=500, tol=1e-7, n_block=1, alpha_W=0.0, alpha_H=0.0, l1_ratio=0.0, device=None, independent_target='none', independent_components=None, independent_iter_max=1000, independent_lr=1.0, independent_beta=1.0, independent_neg_tol=0.0, independent_polish_iter=20, independent_tol=1e-8, n_inner=3, random_state=0):

    """
    HALS NMF followed by a nonnegative-ICA rotation.
    independent_target: 'none' (plain NMF), 'W' (spatial maps are the independent sources),
                        'H' (detector patterns are the independent sources).
    independent_polish_iter: HALS iterations after the rotation to clean up residual negative values (0 = only clamp).
    """

    if independent_target == 'both':
        raise ValueError("independent_target 'both' is no longer supported: one rotation can make only one side uncorrelated. Use 'W' or 'H'.")

    if independent_target not in ('none', 'W', 'H'):
        raise ValueError("independent_target must be 'none', 'W', or 'H'.")

    dev = _resolve_device(device)

    _log_nmf_params(dev, n_components, iter_max, tol, n_block, alpha_W, alpha_H, l1_ratio, n_inner)

    sy, sx, dy, dx = data_4D.shape

    X_np = _prepare_matrix(data_4D, nonneg=True)

    N, M = X_np.shape
    K = n_components

    if not isinstance(K, int) or not 1 <= K <= min(N, M):
        raise ValueError(f'n_components must be between 1 and {min(N, M)}.')

    blocks = BlockedMatrix(X_np, n_block, dev)
    stats = _block_stats(blocks)

    # ----------------------------------------------------
    # Stage 1: NMF
    # ----------------------------------------------------
    Wt, H, err_list = _hals_nmf_torch(blocks, K, iter_max, tol, n_inner, alpha_W, alpha_H, l1_ratio, stats, random_state=random_state)

    if independent_target != 'none':

        R = K if independent_components is None else independent_components

        if not isinstance(R, int) or not 1 <= R <= K:
            raise ValueError(f'independent_components must be between 1 and {K}.')

        print_and_log('')
        print_and_log('################ Nonnegative Independent Factorization ################')
        print_and_log(f'Independent Target: {independent_target}')
        print_and_log(f'Independent Components: {R}')
        print_and_log(f'Rotation Max Iterations: {independent_iter_max}')
        print_and_log(f'Rotation Initial Step: {independent_lr}')
        print_and_log(f'Other-side Nonnegativity Weight: {independent_beta}')
        print_and_log(f'Negativity Tolerance: {independent_neg_tol}')
        print_and_log(f'Polish Iterations: {independent_polish_iter}')

        rot_kwargs = dict(n_components=R, beta=independent_beta, neg_tol=independent_neg_tol, iter_max=independent_iter_max, tol=independent_tol, step_init=independent_lr, device=dev)

        # ----------------------------------------------------
        # Stage 2: rotate the NMF solution
        # ----------------------------------------------------
        if independent_target == 'W':
            print_and_log(f'Mean |correlation| of W before rotation: {_mean_abs_corr(Wt):.4e}')

            # X ~ W0 @ H0 with W0 = Wt.T (N, K)
            S, H_rot, _ = nonnegative_ica_rotation(Wt.T, H, **rot_kwargs)

            Wt = S.T.contiguous().to(dev)
            H = H_rot.to(dev)

        else:
            print_and_log(f'Mean |correlation| of H before rotation: {_mean_abs_corr(H):.4e}')

            # X.T ~ H.T @ Wt : the detector patterns play the role of the sources
            S, W_rot, _ = nonnegative_ica_rotation(H.T, Wt, **rot_kwargs)

            H = S.T.contiguous().to(dev)
            Wt = W_rot.to(dev)

        Wt = Wt.clamp(min=0)
        H = H.clamp(min=0)

        # ----------------------------------------------------
        # Stage 3: polish, keeps both factors strictly nonnegative
        # ----------------------------------------------------
        if independent_polish_iter > 0:
            Wt, H, err_polish = _hals_nmf_torch(blocks, R, independent_polish_iter, tol, n_inner, alpha_W, alpha_H, l1_ratio, stats, init=(Wt, H), label='Polish', random_state=random_state)
            err_list = err_list + err_polish

        if independent_target == 'W':
            print_and_log(f'Mean |correlation| of W after rotation: {_mean_abs_corr(Wt):.4e}')
        else:
            print_and_log(f'Mean |correlation| of H after rotation: {_mean_abs_corr(H):.4e}')

    K_out = Wt.shape[0]

    W_out = np.ascontiguousarray(Wt.cpu().numpy().reshape(K_out, sy, sx), dtype=np.float32)
    H_out = np.ascontiguousarray(H.cpu().numpy().reshape(K_out, dy, dx), dtype=np.float32)

    print_and_log('')
    print_and_log(f'Output Components: {K_out}')
    print_and_log(f'W Output Shape: {W_out.shape}')
    print_and_log(f'H Output Shape: {H_out.shape}')

    _check_output(W_out, H_out, 'INMF')

    return W_out, H_out, err_list
