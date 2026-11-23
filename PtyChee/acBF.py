"""
Author: Zeyu Wang
Date: November 2026
Description: "tilt corrected BF" / "aberration corrected BF"

    acBF : Shift every BF virtual image according to the aberration function and sum
           them (i.e. tcBF). The CTF correction is selected by `ctf_correction`:

        None   : no CTF correction, plain tcBF
        'pre'  : before summation, correct the residual PCTF phase of the virtual
                 image of every detector pixel separately (acBF). Reference:
                 Ma et al., "Parallax Depth Sectioning and 3D Reconstruction in 4D-STEM",
                 Eqs. (5), (7), (24), (F5). Only the phase is corrected, the
                 amplitude is untouched, so noise is not amplified.
        'post' : after summation, apply one global transfer-function filter to the
                 summed image (ctf_method = 'wiener' / 'phase_flip')

    'pre' and 'post' are mutually exclusive (using both would correct twice).

    Which aberrations are used is selected by `aberrations`:

        None   : defocus only, the value is the `defocus` argument (fast closed-form path)
        'fit'  : fit the aberrations from the parallax (the shift of every virtual image
                 as a function of the detector pixel), starting from `defocus`
        dict   : coefficients returned by `fit_aberrations` (fit once, reuse many times)

    `defocus_override` replaces the defocus term of the fitted aberrations by a chosen
    value (e.g. another depth, Delta f - z in the paper) while all other fitted
    aberrations are still corrected.

Aberration model (shift = gradient of the aberration function, paper Eq. 24):
    chi(theta) / k0 = sum_k a_k * P_k(theta),   image displacement dr = grad chi / k0
    The registration shift applied to a virtual image is -dr. The polynomials follow the
    standard (Krivanek) series so that |C12|, |C21|, |C23|, C30, |C32|, |C34| are the
    usual coefficients (in Angstrom):
        defocus : P = -(x^2+y^2)/2        (a = Delta f = -C10; > 0: crossover after sample)
        C12a/b  : P = (x^2-y^2)/2,  x*y                          two-fold astigmatism
        C21a/b  : P = x*r^2/3, y*r^2/3                           axial coma
        C23a/b  : P = (x^3-3xy^2)/3, (3x^2y-y^3)/3               three-fold astigmatism
        C30     : P = r^4/4                                      spherical aberration
        C32a/b  : P = (x^4-y^4)/4, x*y*r^2/2                     star aberration
        C34a/b  : P = (x^4-6x^2y^2+y^4)/4, x^3y-xy^3             four-fold astigmatism
    (x, y) is the detector angle in rad, expressed in the scan coordinate frame.

Units:
    alpha        : rad (not mrad)
    scan_step    : Angstrom / pixel
    defocus      : Angstrom
    Voltage      : kV
    aperture_radius / center_dx / center_dy : detector pixels
"""

import numpy as np
import time
from .utils import print_and_log, log_to_file


# =====================================================================
#                           Common utilities
# =====================================================================
def _electron_wavelength(voltage_kV):
    """Relativistic electron wavelength in Angstrom."""
    V = voltage_kV * 1e3
    return 12.2643 / np.sqrt(V * (1 + 0.97845e-6 * V))


def _detector_to_scan_frame(xi, yj, theta_r, scan_flip):
    """
    Rotate / flip the detector coordinate (xi, yj) (relative to the BF center, in
    detector pixels) into the scan coordinate frame. The image shift is a function of it;
    the aperture tests of the per-pixel CTF correction use the same frame.
    Works on scalars and arrays. Returns (tx, ty); the modulus is preserved.
    """
    v = (xi + 1j * yj) * np.exp(-1j * theta_r)
    tx, ty = v.real, v.imag
    if scan_flip:
        tx = -tx
    return tx, ty


def _bf_mask(dy, dx, center_dy, center_dx, aperture_radius, inner_radius, outer_radius):
    """Detector pixels that take part in the summation (annular ROI)."""
    jj_g, ii_g = np.meshgrid(np.arange(dy), np.arange(dx), indexing='ij')
    rr_g = np.sqrt((jj_g + 1 - center_dy) ** 2 + (ii_g + 1 - center_dx) ** 2)
    return (inner_radius * aperture_radius < rr_g) & (rr_g < outer_radius * aperture_radius)


def _progress_line(name, counter, total, start_time, time_i0):
    time_is = max(time.time() - time_i0, 1e-6)
    speed = 1 / time_is
    process = min(counter / total * 100, 100)
    n = int(process / 10)
    aa, bb = "*" * n, "." * (10 - n)
    dur = int(time.time() - start_time)
    remain = int(max(total - counter, 0) * time_is)
    dh, rem = divmod(dur, 3600)
    dm, ds = divmod(rem, 60)
    rh, rem = divmod(remain, 3600)
    rm, rs = divmod(rem, 60)
    return "{} progressing: {:^3.0f}%[{}->{}] {:.2f}iter/s ({:0>2}:{:0>2}:{:0>2}<{:0>2}:{:0>2}:{:0>2})".format(
        name, process, aa, bb, speed, dh, dm, ds, rh, rm, rs)


def _normalize_ctf_mode(ctf_correction):
    if ctf_correction in (None, False, 'none', 'None'):
        return None
    if ctf_correction in ('pre', 'post'):
        return ctf_correction
    raise ValueError("ctf_correction must be None, 'pre' or 'post'")


def _make_grid(Ny, Nx, scan_step, wavelength, alpha, R):
    """Frequency grids of an (Ny, Nx) image, plus lambda*q expressed in detector pixels."""
    fxx = np.fft.fftfreq(Nx)                 # cycles/pixel, used for the shift
    fyy = np.fft.fftfreq(Ny)
    qx = np.fft.fftfreq(Nx, d=scan_step)     # 1/Angstrom
    qy = np.fft.fftfreq(Ny, d=scan_step)
    KX = (wavelength * qx * R / alpha)[None, :]   # lambda*q in detector pixels
    KY = (wavelength * qy * R / alpha)[:, None]
    return dict(fxx=fxx, fyy=fyy, qx=qx, qy=qy, KX=KX, KY=KY)


# =====================================================================
#                        Aberration function model
# =====================================================================
# name: (degree of chi, P(x, y), grad P(x, y))   chi / k0 = sum a_k P_k, a_k in Angstrom
_TERMS = {
    'defocus': (2, lambda x, y: -(x * x + y * y) / 2,                   lambda x, y: (-x, -y)),
    'C12a':    (2, lambda x, y: (x * x - y * y) / 2,                    lambda x, y: (x, -y)),
    'C12b':    (2, lambda x, y: x * y,                                  lambda x, y: (y, x)),
    'C21a':    (3, lambda x, y: x * (x * x + y * y) / 3,                lambda x, y: ((3 * x * x + y * y) / 3, 2 * x * y / 3)),
    'C21b':    (3, lambda x, y: y * (x * x + y * y) / 3,                lambda x, y: (2 * x * y / 3, (x * x + 3 * y * y) / 3)),
    'C23a':    (3, lambda x, y: (x ** 3 - 3 * x * y * y) / 3,           lambda x, y: (x * x - y * y, -2 * x * y)),
    'C23b':    (3, lambda x, y: (3 * x * x * y - y ** 3) / 3,           lambda x, y: (2 * x * y, x * x - y * y)),
    'C30':     (4, lambda x, y: (x * x + y * y) ** 2 / 4,               lambda x, y: (x * (x * x + y * y), y * (x * x + y * y))),
    'C32a':    (4, lambda x, y: (x ** 4 - y ** 4) / 4,                  lambda x, y: (x ** 3, -y ** 3)),
    'C32b':    (4, lambda x, y: x * y * (x * x + y * y) / 2,            lambda x, y: ((3 * x * x * y + y ** 3) / 2, (x ** 3 + 3 * x * y * y) / 2)),
    'C34a':    (4, lambda x, y: (x ** 4 - 6 * x * x * y * y + y ** 4) / 4, lambda x, y: (x ** 3 - 3 * x * y * y, -3 * x * x * y + y ** 3)),
    'C34b':    (4, lambda x, y: x ** 3 * y - x * y ** 3,                lambda x, y: (3 * x * x * y - y ** 3, x ** 3 - 3 * x * y * y)),
}

# fit order 1: shift ~ theta (defocus, 2-fold astigmatism), 2: + coma, 3-fold astigmatism,
# 3: + spherical aberration, star, 4-fold astigmatism
_ORDER_NAMES = {
    1: ['defocus', 'C12a', 'C12b'],
    2: ['C21a', 'C21b', 'C23a', 'C23b'],
    3: ['C30', 'C32a', 'C32b', 'C34a', 'C34b'],
}


def _names_up_to_order(order):
    if order not in (1, 2, 3):
        raise ValueError('fit_order must be 1, 2 or 3')
    names = []
    for o in range(1, order + 1):
        names += _ORDER_NAMES[o]
    return names


def _active_terms(coeffs):
    """[(a, P, gradP)] of the non-zero coefficients."""
    out = []
    for name, a in coeffs.items():
        if name not in _TERMS:
            raise ValueError(f"unknown aberration name '{name}', allowed: {list(_TERMS)}")
        if a != 0.0:
            out.append((float(a), _TERMS[name][1], _TERMS[name][2]))
    return out


def _chi_over_k0(x, y, act):
    """chi/k0 (Angstrom) at the angle (x, y) (rad); x, y may broadcast against each other."""
    out = 0.0
    for a, P, _ in act:
        out = out + a * P(x, y)
    return out


def _model_shift(tx, ty, coeffs, alpha, R, scan_step):
    """
    Registration shift (scan pixels) of the virtual image at the detector pixel (tx, ty)
    (scan frame, detector pixels): s = -grad(chi)/k0 / scan_step.
    """
    thx, thy = alpha / R * np.asarray(tx, float), alpha / R * np.asarray(ty, float)
    gx, gy = 0.0 * thx, 0.0 * thy
    for a, _, G in _active_terms(coeffs):
        px, py = G(thx, thy)
        gx = gx + a * px
        gy = gy + a * py
    return -gx / scan_step, -gy / scan_step


def aberration_summary(coeffs):
    """Convention-independent magnitudes (Angstrom) of a coefficient dict."""
    g = lambda k: float(coeffs.get(k, 0.0))
    return {
        'defocus (df = -C10)': g('defocus'),
        '|C12|': float(np.hypot(g('C12a'), g('C12b'))),
        '|C21|': float(np.hypot(g('C21a'), g('C21b'))),
        '|C23|': float(np.hypot(g('C23a'), g('C23b'))),
        'C30': g('C30'),
        '|C32|': float(np.hypot(g('C32a'), g('C32b'))),
        '|C34|': float(np.hypot(g('C34a'), g('C34b'))),
    }


def _log_coeffs(coeffs, title='Aberrations'):
    print_and_log(f'{title}:')
    for k, v in aberration_summary(coeffs).items():
        if k.startswith('defocus') or v != 0.0:
            print_and_log(f'    {k}: {v:.6g} A')


def aberration_function_map(coeffs, Voltage, alpha, n=129):
    """
    Aberration function chi (rad) on an n x n grid of detector angles in [-alpha, alpha]^2,
    NaN outside the aperture. chi = k0 * sum_k a_k P_k, the constant and the tilt are zero.
    """
    k0 = 2 * np.pi / _electron_wavelength(Voltage)
    th = np.linspace(-alpha, alpha, n)
    X, Y = np.meshgrid(th, th)
    chi = k0 * _chi_over_k0(X, Y, _active_terms(coeffs)) * np.ones_like(X)
    chi[np.hypot(X, Y) > alpha] = np.nan
    return chi


# =====================================================================
#                 Aberration fit from the parallax (shift map)
# =====================================================================
def _tukey(n, a=0.5):
    if n < 2:
        return np.ones(n)
    x = np.linspace(0, 1, n)
    w = np.ones(n)
    m1 = x < a / 2
    w[m1] = 0.5 * (1 + np.cos(2 * np.pi / a * (x[m1] - a / 2)))
    m2 = x > 1 - a / 2
    w[m2] = 0.5 * (1 + np.cos(2 * np.pi / a * (x[m2] - 1 + a / 2)))
    return w


def _parabolic_peak(c, py, px):
    """Sub-pixel offset of a correlation peak (parabolic interpolation, periodic indexing)."""
    ny, nx = c.shape
    c0 = c[py, px]
    ym, yp = c[(py - 1) % ny, px], c[(py + 1) % ny, px]
    xm, xp = c[py, (px - 1) % nx], c[py, (px + 1) % nx]
    dy_den, dx_den = ym - 2 * c0 + yp, xm - 2 * c0 + xp
    dy_ = 0.5 * (ym - yp) / dy_den if dy_den < 0 else 0.0
    dx_ = 0.5 * (xm - xp) / dx_den if dx_den < 0 else 0.0
    return float(np.clip(dy_, -1, 1)), float(np.clip(dx_, -1, 1))


def _design_matrix(tx, ty, names, alpha, R, scan_step):
    """Columns: registration shift (x rows then y rows) per unit coefficient of every term."""
    thx, thy = alpha / R * tx, alpha / R * ty
    cols = []
    for name in names:
        gx, gy = _TERMS[name][2](thx, thy)
        cols.append(np.concatenate([-gx / scan_step, -gy / scan_step]) * np.ones(2 * len(tx)))
    return np.stack(cols, axis=1)


def _weighted_fit(G, y, w, n_rows_half):
    """Weighted least squares of y = G a + offset (x and y offsets are free), 2-pass outlier rejection."""
    N = n_rows_half
    off = np.zeros((2 * N, 2))
    off[:N, 0] = 1.0
    off[N:, 1] = 1.0
    M = np.hstack([G, off])
    ww = np.concatenate([w, w])
    for _ in range(2):
        sw = np.sqrt(ww)
        A = M * sw[:, None]
        cn = np.linalg.norm(A, axis=0)
        cn[cn == 0] = 1.0
        sol = np.linalg.lstsq(A / cn, y * sw, rcond=None)[0] / cn
        res = y - M @ sol
        sig = 1.4826 * np.median(np.abs(res[ww > 0])) + 1e-9
        ww = ww * (np.abs(res) < 4 * sig)
    return sol[:G.shape[1]], sol[G.shape[1]:], res


def _search_limit(ac, lag_r, cap, min_lag=2.0, thr=0.3):
    """
    Largest safe search radius of the correlation peak. For a periodic object the correlation has
    equally high peaks one lattice period away, a search range beyond about half a period would
    lock onto the wrong one. The nearest strong secondary peak of the autocorrelation `ac`
    (lag distance L) limits the search radius to 0.45 * L.
    """
    ac0 = ac[0, 0]
    if ac0 <= 0:
        return cap
    is_max = np.ones(ac.shape, dtype=bool)
    for dy_ in (-1, 0, 1):
        for dx_ in (-1, 0, 1):
            if dy_ != 0 or dx_ != 0:
                is_max &= ac >= np.roll(ac, (dy_, dx_), axis=(0, 1))
    peaks = is_max & (lag_r >= min_lag) & (lag_r <= cap) & (ac > thr * ac0)
    if not peaks.any():
        return cap
    return min(cap, max(0.45 * float(lag_r[peaks].min()), 1.0))


def _pair_fit(Gd, rhs, w):
    """Weighted least squares of Gd a = rhs, column scaled, 2-pass outlier rejection."""
    ww = w.copy()
    for _ in range(2):
        sw = np.sqrt(ww)
        A = Gd * sw[:, None]
        cn = np.linalg.norm(A, axis=0)
        cn[cn == 0] = 1.0
        sol = np.linalg.lstsq(A / cn, rhs * sw, rcond=None)[0] / cn
        res = rhs - Gd @ sol
        sig = 1.4826 * np.median(np.abs(res[ww > 0])) + 1e-9
        ww = ww * (np.abs(res) < 4 * sig)
    return sol, res


def _neighbor_init(F0, jj, ii, tx, ty, names, alpha, R, scan_step, win, Wb, wr, lag_y, lag_x, lag_r,
                   max_nb_shift, ny, nx, dy, dx):
    """
    Initial aberrations from the relative shifts of NEIGHBORING detector pixels.
    The virtual images of two neighboring pixels differ very little (contrast and shift), so their
    relative shift is small and unambiguous even when the shift of an image relative to a far away
    reference is larger than the lattice period of a crystal. The model is fitted to these
    differences, which is a phase unwrapping by continuity. No initial defocus is needed.
    Returns (coefficient dict, mean correlation, rms residual of the fit in pixels).
    """
    N = len(jj)
    FW = np.empty_like(F0)
    for n in range(N):
        FW[n] = np.fft.rfft2(np.fft.irfft2(F0[n], s=(ny, nx)) * win)

    # lattice guard from the autocorrelation of the image nearest to the BF center
    n_c = int(np.argmin(tx ** 2 + ty ** 2))
    ac = np.fft.irfft2(np.abs(FW[n_c]) ** 2 * Wb, s=(ny, nx))
    search = _search_limit(ac, lag_r, max_nb_shift)
    lagmask = lag_r <= search

    idx_map = -np.ones((dy, dx), dtype=int)
    idx_map[jj, ii] = np.arange(N)
    n_list, m_list, d_list, rho_list = [], [], [], []
    for n in range(N):
        for oj, oi in ((0, 1), (1, 0), (1, 1), (1, -1)):
            j2, i2 = jj[n] + oj, ii[n] + oi
            if 0 <= j2 < dy and 0 <= i2 < dx and idx_map[j2, i2] >= 0:
                m = idx_map[j2, i2]
                X = FW[m] * np.conj(FW[n]) * Wb
                c = np.fft.irfft2(X, s=(ny, nx)) * (ny * nx)
                e_m = np.sum(np.abs(FW[m]) ** 2 * Wb * wr)
                e_n = np.sum(np.abs(FW[n]) ** 2 * Wb * wr)
                py, px = np.unravel_index(np.argmax(np.where(lagmask, c, -np.inf)), c.shape)
                sub_y, sub_x = _parabolic_peak(c, py, px)
                n_list.append(n)
                m_list.append(m)
                d_list.append((lag_x[px] + sub_x, lag_y[py] + sub_y))
                rho_list.append(c[py, px] / np.sqrt(e_m * e_n + 1e-30))
    n_arr, m_arr = np.array(n_list), np.array(m_list)
    d = np.array(d_list)
    rho = np.array(rho_list)

    # s_m - s_n = -d (image m is displaced by +d relative to image n, the registration shift is minus the displacement)
    G = _design_matrix(tx, ty, names, alpha, R, scan_step)
    Gd = np.vstack([G[m_arr] - G[n_arr], G[N + m_arr] - G[N + n_arr]])
    rhs = -np.concatenate([d[:, 0], d[:, 1]])
    w = np.concatenate([np.clip(rho, 0, None) ** 2] * 2)
    sol, res = _pair_fit(Gd, rhs, w)
    return dict(zip(names, [float(v) for v in sol])), float(rho.mean()), float(np.sqrt(np.mean(res ** 2))), search


def fit_aberrations(data_4D,
                    Voltage,
                    alpha,
                    scan_step,
                    scan_rotation_angle,
                    scan_flip,
                    aperture_radius,
                    center_dx,
                    center_dy,
                    defocus,
                    fit_order = 3,
                    fit_radius = 0.9,
                    n_stage = 1,
                    n_iter = 12,
                    qmax = 0.5,
                    max_scan = 256,
                    max_shift = 15,
                    phase_correct = True,
                    n_plain = 1,
                    relax = 2.0,
                    neighbor_init = True,
                    initial_coeffs = None,
                    return_info = False,
):
    """
    Fit the aberrations from the parallax of the 4D-STEM data.

    The virtual image of every detector pixel is displaced by grad(chi)/k0. The displacement
    of each image relative to the (leave-one-out) sum of all the others is measured by
    cross-correlation, then a low-order aberration polynomial is fitted to the measured
    shift as a function of the detector pixel. The registration, the measurement and the fit
    are iterated, starting from the aberration function with only `defocus` = initial guess.

    For a periodic object (crystal) the correlation has equally high peaks one lattice period away, and
    a model that is wrong by more than half a period locks onto the wrong one: the measured shift map
    then looks patchy with jumps of the lattice period and the fit is meaningless. Three things avoid it:
    the start model is estimated from the relative shifts of neighboring detector pixels (`neighbor_init`,
    small and unambiguous, no good initial defocus needed), the search range is limited to 0.45 x the
    lattice period, and the phase-corrected alignment makes the true aberrations a fixed point.

    fit_order  : 1 -> defocus + 2-fold astigmatism
                 2 -> + axial coma + 3-fold astigmatism
                 3 -> + spherical aberration + star + 4-fold astigmatism (default)
    fit_radius : only BF pixels with r < fit_radius * aperture_radius are used (edge pixels are weak)
    n_stage    : optional coarse-to-fine schedule (default 1 = off): the radius grows from 0.35 to 1 times
                 fit_radius and the order from 1 to fit_order over the stages, each stage starts from the
                 previous model. It did not help in the tests with `neighbor_init`, only try it if the fit fails
    n_iter     : number of measure / fit iterations per stage
    qmax       : spatial band used for the correlation, as a fraction of Nyquist (noise suppression)
    max_scan   : the central max_scan x max_scan part of the scan is used (speed / memory)
    max_shift  : upper limit of the search range of the correlation peak (pixels) around the current model;
               it is reduced automatically to 0.45 x the lattice period of a periodic object
    phase_correct : after the first `n_plain` iterations, the images are aligned AND phase-corrected
                 with the current aberration model before they are correlated. Without it the
                 correlation is biased: near the BF edge a virtual image is dominated by a single
                 sideband whose phase ramp displaces it (and for large q the higher-order terms of
                 chi do too). With the correction the true aberration function is a fixed point.
    n_plain    : number of initial iterations done with the plain shift alignment (coarse model first)
    relax      : over-relaxation of the shift update. The measured residual shift of partly misaligned
                 images underestimates the true error, so 2 converges about twice as fast as 1
    neighbor_init : start from an estimate made of the relative shifts of neighboring detector pixels
                 (unambiguous for crystals, independent of the initial defocus, see `_neighbor_init`);
                 `defocus` is then only used if this estimate is unreliable
    initial_coeffs : start from this coefficient dict (refine a previous fit), the neighbor estimate is skipped
    return_info: also return a dict for plotting (see aberration_fit_plot): the BF pixels used (jj, ii, tx, ty),
                 measured and fitted registration shifts (pixels, offset removed), correlation rho, the history
                 of the iterations, the fitted coefficients and the aberration function chi (rad) with and
                 without the defocus term

    Returns a dict of coefficients (Angstrom), see the module docstring. The scan rotation and
    flip must be right: with a wrong rotation the shift field is not a gradient field and the
    fit residual stays large (reported in the log).
    """
    if defocus is None and initial_coeffs is None:
        raise ValueError('defocus (initial guess) must be given to fit the aberrations')
    if n_iter < 1 or n_stage < 1:
        raise ValueError('n_iter and n_stage must be at least 1')
    names = _names_up_to_order(fit_order)
    sy, sx, dy, dx = data_4D.shape
    ny, nx = min(sy, max_scan), min(sx, max_scan)
    y0, x0 = (sy - ny) // 2, (sx - nx) // 2
    theta_r = np.pi * scan_rotation_angle / 180
    R = aperture_radius

    jj_g, ii_g = np.meshgrid(np.arange(dy), np.arange(dx), indexing='ij')
    yj_g, xi_g = jj_g + 1 - center_dy, ii_g + 1 - center_dx
    sel = np.hypot(yj_g, xi_g) < fit_radius * R
    jj, ii = jj_g[sel], ii_g[sel]
    tx, ty = _detector_to_scan_frame(xi_g[sel].astype(float), yj_g[sel].astype(float), theta_r, scan_flip)
    rr = np.hypot(yj_g[sel], xi_g[sel]) / R
    N_all = len(jj)
    if N_all < len(names) + 3:
        raise ValueError(f'only {N_all} detector pixels, too few to fit {len(names)} aberration terms')

    # coarse-to-fine schedule: the radius and the order grow stage by stage
    stage_radius = fit_radius * (np.linspace(0.35, 1.0, n_stage) if n_stage > 1 else np.array([1.0]))
    stage_order = np.minimum(fit_order, 1 + np.arange(n_stage))
    stage_order[-1] = fit_order

    wavelength = _electron_wavelength(Voltage)
    k0 = 2 * np.pi / wavelength
    sc = alpha / R
    R2 = R * R
    gfit = _make_grid(ny, nx, scan_step, wavelength, alpha, R)

    print_and_log('')
    print_and_log('################### Aberration fit (parallax) ###################')
    print_and_log(f'fit_order = {fit_order} ({len(names)} terms), {N_all} BF pixels, scan crop {ny} x {nx}, '
                  f'initial defocus = {defocus} A, {n_stage} stage(s) x {n_iter} iterations, '
                  f'phase correction after {n_plain} plain iteration(s): {phase_correct}')

    # spectra of the mean-removed virtual images (cropped scan)
    nfx = nx // 2 + 1
    F0_all = np.empty((N_all, ny, nfx), dtype=np.complex64)
    for n in range(N_all):
        img = data_4D[y0:y0 + ny, x0:x0 + nx, jj[n], ii[n]].astype(np.float64)
        F0_all[n] = np.fft.rfft2(img - img.mean())

    fy = np.fft.fftfreq(ny)[:, None]
    fx = np.fft.rfftfreq(nx)[None, :]
    fr = np.hypot(fy, fx)
    f_hi, f_lo = qmax * 0.5, 2.0 / min(ny, nx)
    Wb = np.exp(-(fr / f_hi) ** 4) * (1 - np.exp(-(fr / f_lo) ** 2))   # band-pass weight
    wr = np.full((1, nfx), 2.0)                                         # half spectrum -> full spectrum sums
    wr[0, 0] = 1.0
    if nx % 2 == 0:
        wr[0, -1] = 1.0
    win = np.outer(_tukey(ny), _tukey(nx))
    lag_y = np.fft.fftfreq(ny) * ny
    lag_x = np.fft.fftfreq(nx) * nx
    lag_r = np.hypot(lag_y[:, None], lag_x[None, :])

    coeffs = {k: 0.0 for k in names}
    coeffs['defocus'] = float(defocus) if defocus is not None else 0.0
    if initial_coeffs is not None:
        coeffs.update({k: float(v) for k, v in initial_coeffs.items()})
    history = dict(iteration=[], stage=[], rms_shift_update=[], rms_fit_residual=[], mean_rho=[], defocus=[])
    counter = 0
    use_corr = False

    if neighbor_init and initial_coeffs is None:
        init, rho_nb, res_nb, search_nb = _neighbor_init(F0_all, jj, ii, tx, ty, names, alpha, R, scan_step, win, Wb, wr,
                                                         lag_y, lag_x, lag_r, min(max_shift, 4.0), ny, nx, dy, dx)
        print_and_log(f'neighbor estimate: mean correlation of neighboring pixels = {rho_nb:.3f}, '
                      f'rms fit residual = {res_nb:.3f} px, search range {search_nb:.1f} px, defocus = {init["defocus"]:.4g} A')
        if rho_nb > 0.3:
            coeffs.update(init)
        else:
            print_and_log('    neighbor estimate unreliable, starting from the given defocus')

    for st in range(n_stage):
        idx = np.nonzero(rr < stage_radius[st] - 1e-9 if st < n_stage - 1 else np.ones(N_all, dtype=bool))[0]
        names_s = _names_up_to_order(int(stage_order[st]))
        if len(idx) < len(names_s) + 3:
            print_and_log(f'stage {st + 1}/{n_stage}: only {len(idx)} pixels, skipped')
            continue
        N = len(idx)
        F0 = F0_all[idx]
        tx_s, ty_s = tx[idx], ty[idx]
        print_and_log(f'stage {st + 1}/{n_stage}: r < {stage_radius[st]:.2f} x aperture, {N} pixels, '
                      f'order {int(stage_order[st])} ({len(names_s)} terms)')
        for it in range(n_iter):
            s_mx, s_my = _model_shift(tx_s, ty_s, coeffs, alpha, R, scan_step)
            act = _active_terms(coeffs)

            def measure(corrected):
                """Align (and phase-correct) with the current model, then measure every image against the others."""
                FA = np.empty_like(F0)
                for n in range(N):
                    if corrected:
                        z = _pctf_bracket(tx_s[n], ty_s[n], gfit, act, k0, sc, R2)[:, :nfx]
                        az = np.abs(z)
                        kern = np.zeros_like(z)
                        np.divide(np.conj(z), az, out=kern, where=az > 1e-8)   # = shift kernel * exp(-i arg PCTF_res)
                    else:
                        kern = np.exp(-2j * np.pi * (fy * s_my[n] + fx * s_mx[n]))
                    A = np.fft.irfft2(F0[n] * kern, s=(ny, nx)) * win
                    FA[n] = np.fft.rfft2(A)
                S = FA.sum(axis=0)

                # search range: for a periodic object stay below half the lattice period
                ac = np.fft.irfft2(np.abs(S) ** 2 * Wb, s=(ny, nx))
                search = _search_limit(ac, lag_r, max_shift)
                lagmask = lag_r <= search

                d = np.zeros((N, 2))      # (x, y) displacement of each aligned image w.r.t. the leave-one-out reference
                rho = np.zeros(N)
                for n in range(N):
                    ref_F = S - FA[n]
                    X = FA[n] * np.conj(ref_F) * Wb
                    c = np.fft.irfft2(X, s=(ny, nx)) * (ny * nx)
                    e_a = np.sum(np.abs(FA[n]) ** 2 * Wb * wr)
                    e_r = np.sum(np.abs(ref_F) ** 2 * Wb * wr)
                    py, px = np.unravel_index(np.argmax(np.where(lagmask, c, -np.inf)), c.shape)
                    sub_y, sub_x = _parabolic_peak(c, py, px)
                    d[n, 0] = lag_x[px] + sub_x
                    d[n, 1] = lag_y[py] + sub_y
                    rho[n] = c[py, px] / np.sqrt(e_a * e_r + 1e-30)
                return d, rho, search

            # phase-corrected alignment is only better when the model is close to the truth. While it is not
            # used yet, it is tried in every iteration and adopted as soon as it correlates better than the plain
            # alignment; once adopted it is kept (the true aberration function is its fixed point)
            if not phase_correct or (st == 0 and it < n_plain):
                use_corr = False
                d, rho, search = measure(False)
            elif use_corr:
                d, rho, search = measure(True)
            else:
                d_p, rho_p, search_p = measure(False)
                d_c, rho_c, search_c = measure(True)
                use_corr = rho_c.mean() >= rho_p.mean()
                d, rho, search = (d_c, rho_c, search_c) if use_corr else (d_p, rho_p, search_p)
                print_and_log(f'    mean correlation: plain alignment {rho_p.mean():.3f}, phase-corrected {rho_c.mean():.3f}, '
                              f'using {"phase-corrected" if use_corr else "plain"} alignment')
            if it == 0 and search < max_shift:
                print_and_log(f'    periodic object detected, search range limited to {search:.1f} px')

            # total registration shift = current model shift + correction (-d: A(r) ~ ref(r - d))
            s_tot = np.concatenate([s_mx - relax * d[:, 0], s_my - relax * d[:, 1]])
            w = np.clip(rho, 0, None) ** 2
            G = _design_matrix(tx_s, ty_s, names_s, alpha, R, scan_step)
            sol, offset, res = _weighted_fit(G, s_tot, w, N)
            coeffs.update(dict(zip(names_s, [float(v) for v in sol])))

            counter += 1
            rms_d = float(np.sqrt(np.mean(d ** 2)))
            rms_res = float(np.sqrt(np.mean(res ** 2)))
            print_and_log(f'    iteration {it + 1}/{n_iter}: rms residual shift before update = {rms_d:.3f} px, '
                          f'rms fit residual = {rms_res:.3f} px, mean correlation = {rho.mean():.3f}, '
                          f'defocus = {coeffs["defocus"]:.4g} A')
            history['iteration'].append(counter)
            history['stage'].append(st + 1)
            history['rms_shift_update'].append(rms_d)
            history['rms_fit_residual'].append(rms_res)
            history['mean_rho'].append(float(rho.mean()))
            history['defocus'].append(coeffs['defocus'])
            if rms_d < 0.02 and not (st == 0 and it < n_plain):
                break

    _log_coeffs(coeffs, 'Fitted aberrations')
    if float(np.mean(rho)) < 0.15:
        print_and_log('WARNING: very low correlation between the virtual images, the fit is probably unreliable '
                      '(too noisy / wrong scan rotation or flip / wrong initial defocus). Try a lower fit_order, '
                      'a smaller qmax, or check the scan rotation.')
    if not return_info:
        return coeffs

    # measured and fitted registration shifts of the last iteration (the global offset is not an aberration)
    fitted = G @ sol
    info = dict(jj=jj[idx], ii=ii[idx], dy=dy, dx=dx, tx=tx_s, ty=ty_s, alpha=alpha, scan_step=scan_step,
                shift_measured=np.stack([s_tot[:N] - offset[0], s_tot[N:] - offset[1]], 1),
                shift_fit=np.stack([fitted[:N], fitted[N:]], 1),
                rho=rho, rms_fit_residual=rms_res,
                history={k: np.asarray(v) for k, v in history.items()},
                coeffs=coeffs,
                chi=aberration_function_map(coeffs, Voltage, alpha),
                chi_wo_defocus=aberration_function_map({**coeffs, 'defocus': 0.0}, Voltage, alpha))
    return coeffs, info


def _resolve_coefficients(aberrations, defocus, defocus_override, fit_options,
                          data_4D, Voltage, alpha, scan_step, scan_rotation_angle, scan_flip,
                          aperture_radius, center_dx, center_dy):
    if aberrations is None:
        if defocus is None and defocus_override is None:
            raise ValueError('defocus must be given when aberrations is None')
        coeffs = {'defocus': float(defocus if defocus is not None else 0.0)}
    elif isinstance(aberrations, str):
        if aberrations != 'fit':
            raise ValueError("aberrations must be None, 'fit' or a dict")
        if defocus is None:
            raise ValueError("defocus (initial guess) must be given when aberrations='fit'")
        coeffs = fit_aberrations(data_4D, Voltage, alpha, scan_step, scan_rotation_angle, scan_flip,
                                 aperture_radius, center_dx, center_dy, defocus, **(fit_options or {}))
    elif isinstance(aberrations, dict):
        coeffs = {k: float(v) for k, v in aberrations.items()}
        coeffs.setdefault('defocus', float(defocus) if defocus is not None else 0.0)
    else:
        raise ValueError("aberrations must be None, 'fit' or a dict")
    if defocus_override is not None:
        coeffs['defocus'] = float(defocus_override)
    _active_terms(coeffs)   # validates the names
    return coeffs


# =====================================================================
#          'post': global CTF correction after summation (defocus only)
# =====================================================================
def _lens_area(r1, r2, d):
    """Overlap area of two circles with radii r1, r2 and center distance d (array)."""
    d = np.asarray(d, dtype=float)
    out = np.zeros_like(d)
    if r1 <= 0 or r2 <= 0:
        return out
    full = d <= abs(r1 - r2)
    out[full] = np.pi * min(r1, r2) ** 2
    part = (d > abs(r1 - r2)) & (d < r1 + r2)
    dp = d[part]
    c1 = np.clip((dp**2 + r1**2 - r2**2) / (2 * dp * r1), -1, 1)
    c2 = np.clip((dp**2 + r2**2 - r1**2) / (2 * dp * r2), -1, 1)
    root = np.sqrt(np.clip((-dp + r1 + r2) * (dp + r1 - r2) * (dp - r1 + r2) * (dp + r1 + r2), 0, None))
    out[part] = r1**2 * np.arccos(c1) + r2**2 * np.arccos(c2) - 0.5 * root
    return out


def _post_sum_ctf_correct(img, scan_step, defocus, wavelength, alpha, aperture_radius,
                          inner_radius, outer_radius, method='wiener', eps=0.05):
    """
    Global filter applied to the summed tcBF image (no per-pixel correction), defocus only.
    Transfer function after alignment and summation: CTF(q) = W(q) * sin(pi*lambda*df*q^2),
    where W(q) is the normalized overlap area between the selected detector annulus and
    the aperture disk displaced by lambda*q.
        wiener     : H = C / (C^2 + eps)
        phase_flip : H = sign(sin chi)  (where W > 0)
    """
    sy, sx = img.shape
    fx = np.fft.fftfreq(sx, d=scan_step)
    fy = np.fft.fftfreq(sy, d=scan_step)
    q2 = fy[:, None] ** 2 + fx[None, :] ** 2
    q = np.sqrt(q2)

    chi = np.pi * wavelength * defocus * q2
    s = np.sin(chi)

    d_pix = wavelength * q * aperture_radius / alpha
    R = aperture_radius
    W = (_lens_area(outer_radius * R, R, d_pix) - _lens_area(inner_radius * R, R, d_pix)) \
        / (np.pi * (outer_radius**2 - inner_radius**2) * R**2)

    if method == 'wiener':
        C = W * s
        H = C / (C**2 + eps)
    elif method == 'phase_flip':
        H = np.where(W > 0, np.sign(s), 0.0)
    else:
        raise ValueError("ctf_method must be 'wiener' or 'phase_flip'")
    H[0, 0] = 1.0  # keep the DC component

    mean = img.mean()
    return np.real(np.fft.ifft2(np.fft.fft2(img - mean) * H)) + mean


def _post_sum_filter_general(img, C, method='wiener', eps=0.05):
    """
    Global filter for general aberrations. C(q) is the (complex) transfer function of the
    summed image, accumulated numerically over the summed pixels.
        wiener     : H = conj(C) / (|C|^2 + eps)
        phase_flip : H = conj(C) / |C|   (phase only; reduces to sign(sin chi) for defocus only)
    """
    aC = np.abs(C)
    if method == 'wiener':
        H = np.conj(C) / (aC ** 2 + eps)
    elif method == 'phase_flip':
        H = np.where(aC > 1e-8, np.conj(C) / np.maximum(aC, 1e-30), 0.0)
    else:
        raise ValueError("ctf_method must be 'wiener' or 'phase_flip'")
    H = H.astype(complex)
    H[0, 0] = 1.0  # keep the DC component
    mean = img.mean()
    return np.real(np.fft.ifft2(np.fft.fft2(img - mean) * H)) + mean


# =====================================================================
#              General per-pixel PCTF (arbitrary aberrations)
# =====================================================================
def _pctf_bracket(tx, ty, grid, act, k0, sc, R2):
    """
    i * [ A(Th - lam q) e^{-i[chi(Th) - chi(Th - lam q)]} - A(Th + lam q) e^{+i[chi(Th) - chi(Th + lam q)]} ]
    (paper Eq. 5 without the real prefactor A(Th)/(2*Omega0)), on the frequency grid `grid`.
    (tx, ty): detector pixel in the scan frame; sc converts detector pixels to rad.
    This contains the first-order image-shift phase; multiply by the shift kernel to get PCTF_res.
    """
    KX, KY = grid['KX'], grid['KY']
    Am = ((ty - KY) ** 2 + (tx - KX) ** 2) < R2
    Ap = ((ty + KY) ** 2 + (tx + KX) ** 2) < R2
    chi0 = k0 * _chi_over_k0(sc * tx, sc * ty, act)
    chim = k0 * _chi_over_k0(sc * (tx - KX), sc * (ty - KY), act)
    chip = k0 * _chi_over_k0(sc * (tx + KX), sc * (ty + KY), act)
    return 1j * (Am * np.exp(-1j * (chi0 - chim)) - Ap * np.exp(1j * (chi0 - chip)))


def _shift_kernel(grid, x_shift, y_shift):
    return np.exp(-2j * np.pi * grid['fyy'] * y_shift)[:, None] * np.exp(-2j * np.pi * grid['fxx'] * x_shift)[None, :]


# =====================================================================
#                                acBF
# =====================================================================
def acBF(data_4D,
        Voltage,
        alpha,
        scan_step,
        scan_rotation_angle,
        scan_flip,
        aperture_radius,
        center_dx,
        center_dy,
        defocus = None,
        inner_radius = 0,
        outer_radius = 1,
        BF_padding = 30,
        ctf_correction = 'pre',
        ctf_method = 'wiener',
        ctf_eps = 0.05,
        contrast_reverse = False,
        aberrations = None,
        defocus_override = None,
        fit_options = None,
):
    """
    ctf_correction : None  -> shift alignment + summation only (tcBF)
                     'pre' -> per-pixel phase correction before summation (acBF)
                     'post'-> global filter after summation, set by ctf_method / ctf_eps

    aberrations      : None  -> defocus only, value = `defocus`
                       'fit' -> fit the aberrations from the shift map (`fit_aberrations`),
                                `defocus` is the initial guess, `fit_options` is a dict of
                                the keyword arguments of `fit_aberrations`
                       dict  -> coefficients from `fit_aberrations` (fit once, reuse)
    defocus_override : replace the defocus term (fitted or given) by this value, all the other
                       aberrations are kept. E.g. the effective defocus of another depth.

    In 'pre' mode, for the virtual image D(r, Theta) of every BF pixel Theta:
        D~ <- D~ * exp(-i*2*pi*q.shift) * exp(-i*arg[PCTF_res(q, Theta)])
        PCTF_res = exp(+i*2*pi*q.dr) * PCTF,  dr = grad(chi)/k0 = -shift
        PCTF ~ i*A(Theta)*[ A(Theta-lam*q) e^{-i[chi(Theta)-chi(Theta-lam*q)]}
                            - A(Theta+lam*q) e^{+i[chi(Theta)-chi(Theta+lam*q)]} ]
    where A is the *full* aperture (radius aperture_radius), independent of the
    inner/outer annular ROI; inner/outer only decide which pixels are summed.
    The correction factor is set to 0 where PCTF_res = 0 (e.g. q = 0, or sin(phi) = 0
    in the double-sideband region). In this mode the mean of every virtual image is
    removed before padding, so the output has no DC component.

    Padding: in 'pre' mode the mean is removed first and zeros are padded; in the
    other modes the padding is filled with the mean of the virtual image. Both avoid
    the dark borders that zero padding would cause when the shifted images are summed.
    """
    mode = _normalize_ctf_mode(ctf_correction)

    print_and_log('')
    print_and_log(f'#################### acBF Parameters #####################')
    print_and_log(f'vSTEM shift rotation angle: {scan_rotation_angle} degrees')
    print_and_log(f'vSTEM shift flip x: {scan_flip}')
    print_and_log(f'inner_radiuss: {inner_radius} Alpha')
    print_and_log(f'outer_radius: {outer_radius} Alpha')
    print_and_log(f'Voltage: {Voltage} kV')

    coeffs = _resolve_coefficients(aberrations, defocus, defocus_override, fit_options,
                                   data_4D, Voltage, alpha, scan_step, scan_rotation_angle, scan_flip,
                                   aperture_radius, center_dx, center_dy)
    df_eff = coeffs.get('defocus', 0.0)
    act = _active_terms(coeffs)
    only_defocus = all(v == 0.0 for k, v in coeffs.items() if k != 'defocus')
    _log_coeffs(coeffs, 'Aberrations used' + (' (defocus overridden)' if defocus_override is not None else ''))
    if mode is None:
        print_and_log('CTF correction: None (plain tcBF)')
    elif mode == 'pre':
        print_and_log('CTF correction: pre-sum, per-pixel phase correction (acBF)')
    else:
        print_and_log(f'CTF correction: post-sum, {ctf_method} (eps={ctf_eps})')

    sy, sx, dy, dx = data_4D.shape
    theta_r = np.pi * scan_rotation_angle / 180
    data_4D_p = np.transpose(data_4D, (2, 3, 0, 1))

    wavelength = _electron_wavelength(Voltage)
    k0 = 2 * np.pi / wavelength
    if mode is not None:
        print_and_log(f'wavelength = {wavelength:.5f} A')

    # The phase of the Fourier shift must be normalized by the padded size
    Ny, Nx = sy + BF_padding * 2, sx + BF_padding * 2
    R = aperture_radius
    R2 = R * R
    sc = alpha / R                      # detector pixel -> rad
    gp = _make_grid(Ny, Nx, scan_step, wavelength, alpha, R)

    if mode == 'pre' and only_defocus:
        # closed form (defocus only): the shift is removed analytically
        phi = np.pi * wavelength * df_eff * (gp['qy'][:, None] ** 2 + gp['qx'][None, :] ** 2)
        e_m = np.exp(-1j * phi)
        e_p = np.exp(1j * phi)
    if mode == 'pre':
        corr = np.zeros((Ny, Nx), dtype=complex)
    if mode == 'post' and not only_defocus:
        gi = _make_grid(sy, sx, scan_step, wavelength, alpha, R)
        C_acc = np.zeros((sy, sx), dtype=complex)

    # pixels that take part in the summation and their registration shifts
    mask = _bf_mask(dy, dx, center_dy, center_dx, R, inner_radius, outer_radius)
    jj_idx, ii_idx = np.nonzero(mask)
    total = max(len(jj_idx), 1)
    tx_all, ty_all = _detector_to_scan_frame(ii_idx + 1 - center_dx, jj_idx + 1 - center_dy, theta_r, scan_flip)
    if only_defocus:
        shift_const = alpha * df_eff / (R * scan_step)   # image shift (scan pixels) = shift_const * detector pixel coordinate
        shx_all, shy_all = tx_all * shift_const, ty_all * shift_const
    else:
        shx_all, shy_all = _model_shift(tx_all, ty_all, coeffs, alpha, R, scan_step)
    max_sh = float(np.max(np.hypot(shx_all, shy_all))) if len(shx_all) else 0.0
    print_and_log(f'max image shift = {max_sh:.2f} pixels (BF_padding = {BF_padding})')
    if max_sh > BF_padding:
        print_and_log('WARNING: the largest image shift exceeds BF_padding, increase BF_padding')

    start_time = time.time()
    print_and_log('')
    print_and_log(f'##################### acBF Process #######################')
    print("\racBF progressing: {:^3.0f}%[{}->{}] ?iter/s ({:0>2}:{:0>2}:{:0>2}<??:??:??)".format(0, "*" * 0, "." * 10, 0, 0, 0), end="")
    Aligned_BF = np.zeros((sy, sx))
    counter = 0
    line = ''
    for idx in range(len(jj_idx)):
        jj, ii = jj_idx[idx], ii_idx[idx]
        time_i0 = time.time()
        tx, ty = tx_all[idx], ty_all[idx]       # Theta in the scan frame (shift and aperture tests share it)
        x_shift, y_shift = shx_all[idx], shy_all[idx]

        BF_ji = data_4D_p[jj][ii]
        BF_mean = BF_ji.mean()
        if mode == 'pre':
            # The PCTF does not transfer q = 0: remove the mean, then pad with zeros
            BF_ji = BF_ji - BF_mean
            pad_value = 0.0
        else:
            # Pad with the mean so the shifted-in border is not darker than the image
            pad_value = BF_mean
        BF_ji_pad = np.pad(BF_ji, ((BF_padding, BF_padding), (BF_padding, BF_padding)),
                           'constant', constant_values=pad_value)

        kernel = _shift_kernel(gp, x_shift, y_shift)

        if mode == 'pre':
            if only_defocus:
                # Aperture functions A(Theta - lam*q), A(Theta + lam*q)
                Am = ((ty - gp['KY']) ** 2 + (tx - gp['KX']) ** 2) < R2
                Ap = ((ty + gp['KY']) ** 2 + (tx + gp['KX']) ** 2) < R2
                # Residual PCTF (shift already removed) and its phase correction factor exp(-i arg z) = conj(z) / |z|
                z = 1j * (Am * e_m - Ap * e_p)
                absz = np.abs(z)
                corr[:] = 0
                np.divide(np.conj(z), absz, out=corr, where=absz > 1e-8)
            else:
                # General aberrations: PCTF_res = shift kernel * full PCTF bracket
                z = kernel * _pctf_bracket(tx, ty, gp, act, k0, sc, R2)
                absz = np.abs(z)
                corr[:] = 0
                np.divide(np.conj(z), absz, out=corr, where=absz > 1e-8)
            kernel = kernel * corr
        elif mode == 'post' and not only_defocus:
            # accumulate the transfer function of the summed image: C = mean over pixels of (i/2)*kernel*bracket
            C_acc += 0.5 * _shift_kernel(gi, x_shift, y_shift) * _pctf_bracket(tx, ty, gi, act, k0, sc, R2)

        BF_ji_pad_moved = np.fft.ifft2(np.fft.fft2(BF_ji_pad) * kernel)
        Aligned_BF += np.real(BF_ji_pad_moved[BF_padding:BF_padding + sy, BF_padding:BF_padding + sx])

        counter += 1
        line = _progress_line('acBF', counter, total, start_time, time_i0)
        print("\r" + line, end="")
    print('')
    if line:
        log_to_file(line)

    print_and_log(f'acBF process finished in {time.time() - start_time} s')

    # ---------------- Global CTF correction after summation ----------------
    if mode == 'post':
        if only_defocus:
            if df_eff != 0:
                Aligned_BF = _post_sum_ctf_correct(Aligned_BF, scan_step, df_eff, wavelength, alpha,
                                                   R, inner_radius, outer_radius,
                                                   method=ctf_method, eps=ctf_eps)
            else:
                print_and_log('defocus = 0, skip post-sum CTF correction')
        else:
            Aligned_BF = _post_sum_filter_general(Aligned_BF, C_acc / total, method=ctf_method, eps=ctf_eps)

    if contrast_reverse:
        Aligned_BF *= -1

    return Aligned_BF
