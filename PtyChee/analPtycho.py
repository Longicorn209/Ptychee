"""
Author: Zeyu Wang
Date: November 2026
Description: "analytical ptychogrphay"
"""

import numpy as np
import time
import torch
from torch.fft import fft2, ifft2, fftshift, ifftshift
from scipy import fft as sfft
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from .utils import print_and_log, log_to_file, Chi_defocus, annular_mask, r_map



def p_fft(data_4D, clamp=False, chunk_bytes=2**28):
    # FFT along (sy, sx) -> (p_y, p_x), centred by fftshift (complex64).
    # The detector dimensions are processed in chunks (low peak memory), negative values can be clamped chunk by chunk.
    time_i0 = time.time()

    sy, sx, dy, dx = data_4D.shape
    data_fft = np.empty((sy, sx, dy, dx), dtype=np.complex64)
    step = max(1, int(chunk_bytes // (sy*sx*dx*8)))
    for i in range(0, dy, step):
        block = np.array(data_4D[:, :, i:i+step], dtype=np.float32)
        if clamp:
            np.maximum(block, 0, out=block)
        data_fft[:, :, i:i+step] = sfft.fftshift(sfft.fft2(block, axes=(0, 1), workers=-1), axes=(0, 1))

    time_ie = time.time()-time_i0
    print_and_log(f'fft along p finished in {time_ie} s')

    return data_fft



def rP_ifft(data_4D):
    time_i0 = time.time()

    data_4D = torch.from_numpy(data_4D)#.to(device)
    data_4D = ifft2((ifftshift(data_4D,dim=(2,3))),dim=(2,3))

    time_ie = time.time()-time_i0
    print_and_log(f'ifft along rPrime finished in {time_ie} s')

    return np.array(data_4D.cpu())



def r_fft(data_4D):
    time_i0 = time.time()

    data_4D = torch.from_numpy(data_4D)#.to(device)
    data_4D = fftshift((fft2(data_4D,dim=(2,3))),dim=(2,3))

    time_ie = time.time()-time_i0
    print_and_log(f'fft along r finished in {time_ie} s')

    return np.array(data_4D.cpu())



def calculate_pPrime_calibration(scan_step, data_shape):
    sy, sx, dy, dx = data_shape

    p_pixel_size = scan_step
    p_size_sy = p_pixel_size*sy
    p_size_sx = p_pixel_size*sx
    pPrime_pixel_size_dy = 1/p_size_sy
    pPrime_pixel_size_dx = 1/p_size_sx

    return pPrime_pixel_size_dy, pPrime_pixel_size_dx



def ApeFunc(yc, xc, dy, dx, aperture_radius, reciprocalSpace_pixel_size, Voltage, defocus):
    mask = annular_mask(dy, dx, 0, aperture_radius, yc, xc)
    Chi, *_ = Chi_defocus(dy, dx, reciprocalSpace_pixel_size, Voltage, defocus, yc, xc)

    return mask * np.exp(-1j*Chi)



def _scan_shifts(sy, sx, center_sy, center_sx, pPrime_pixel_size_dy, pPrime_pixel_size_dx, reciprocalSpace_pixel_size, theta_r, scan_flip):
    # detector shifts (ry, rx), in detector pixels, of every p' = (sj, si); shape (sy, sx).
    # Same rotation / flip convention as the original per-pixel loops, computed once for all p'.
    py = (np.arange(sy) - center_sy)*pPrime_pixel_size_dy/reciprocalSpace_pixel_size
    px = (np.arange(sx) - center_sx)*pPrime_pixel_size_dx/reciprocalSpace_pixel_size
    ry0, rx0 = np.meshgrid(py, px, indexing='ij')
    rx = np.cos(theta_r)*rx0 + np.sin(theta_r)*ry0
    ry = -np.sin(theta_r)*rx0 + np.cos(theta_r)*ry0
    if scan_flip:
        rx = -rx

    return ry, rx



def _progress_line(name, done, total, start_time):
    # done / total are counted in work units of equal cost, so the bar advances uniformly
    elapsed = time.time()-start_time
    speed = done/max(elapsed, 1e-9)
    process = (done/total*100)
    aa = "*" * int(process/10)
    bb = "." * (10-int(process/10))
    dur = int(elapsed)
    time_remain = int((total-done)/max(speed, 1e-9))
    dur_h = dur//3600
    dur_m = (dur-dur_h*3600)//60
    dur_s = dur-dur_h*3600-dur_m*60
    time_remain_h = time_remain//3600
    time_remain_m = (time_remain-time_remain_h*3600)//60
    time_remain_s = time_remain-time_remain_h*3600-time_remain_m*60

    return "{} progressing: {:^3.0f}%[{}->{}] {:.2f}iter/s ({:0>2}:{:0>2}:{:0>2}<{:0>2}:{:0>2}:{:0>2})".format(name,process,aa,bb,speed,dur_h,dur_m,dur_s,time_remain_h,time_remain_m,time_remain_s)



def _row_chunk(n_row, elements_per_row, budget=2e7, min_updates=50):
    # number of p' rows handled at once: limited by memory and small enough for a smooth progress bar
    return max(1, min(int(budget // elements_per_row), int(np.ceil(n_row/min_updates))))



def _make_aperture_stack(dy, dx, center_dy, center_dx, aperture_radius, reciprocalSpace_pixel_size, Voltage, defocus, test_centers):
    # Returns f(yc, xc) -> (n, dy, dx) complex64 stack of ApeFunc(yc[i], xc[i]).
    # Fast path: top-hat * exp(-i*coeff*r^2) (coeff calibrated from Chi_defocus), evaluated only inside the aperture.
    # It is compared with ApeFunc on test_centers (centre, random and the most distant shifts of this scan) and
    # ApeFunc itself is used if they do not agree.
    def slow(yc, xc):
        return np.stack([ApeFunc(y, x, dy, dx, aperture_radius, reciprocalSpace_pixel_size, Voltage, defocus) for y, x in zip(yc, xc)]).astype(np.complex64)

    try:
        Chi0, *_ = Chi_defocus(dy, dx, reciprocalSpace_pixel_size, Voltage, defocus, center_dy, center_dx)
        coeff = float(np.asarray(Chi0)[0, 0])/((0-center_dy)**2 + (0-center_dx)**2)
    except Exception:
        print_and_log('aperture function: fast path unavailable, using ApeFunc')
        return slow

    Y = np.arange(dy, dtype=np.float64)[None, :, None]
    X = np.arange(dx, dtype=np.float64)[None, None, :]
    R2 = float(aperture_radius)**2

    def fast(yc, xc):
        r2 = (Y - np.asarray(yc, dtype=np.float64)[:, None, None])**2 + (X - np.asarray(xc, dtype=np.float64)[:, None, None])**2
        inside = r2 < R2
        out = np.zeros(r2.shape, dtype=np.complex64)
        phase = (coeff*r2[inside]).astype(np.float32)
        out[inside] = np.cos(phase) - 1j*np.sin(phase)
        return out

    yt, xt = np.asarray(test_centers[0], dtype=np.float64), np.asarray(test_centers[1], dtype=np.float64)
    mismatch = np.mean(np.abs(slow(yt, xt) - fast(yt, xt)) > 1e-3)
    if mismatch > 2e-4:
        print_and_log(f'aperture function: fast path differs from ApeFunc on {mismatch:.2%} of pixels, using ApeFunc')
        return slow

    return fast



def _WDD_finalize(obj_WDD_f, center_sy, center_sx, legacy_output):
    # obj_WDD_f: (sy, sx) r-FFT value of obj_WDD at the single used frequency, for one defocus.
    # legacy_output = True : the last step (D00, flip) is exactly the original one, so the result equals the original code.
    # legacy_output = False: D00 = sqrt(|.|) and the flip is done around Q = 0 (np.roll for even sizes); obj_fft is then
    #                        consistent with objFunc and the image is not shifted by one pixel (the result differs from the
    #                        original by that one-pixel shift of objFunc / one-pixel shift of obj_fft).
    sy, sx = obj_WDD_f.shape

    if legacy_output:
        D00 = np.sqrt(obj_WDD_f[center_sy][center_sx])
        obj_fft_minus_rho = np.conj(obj_WDD_f)/D00

        obj_fft = np.flip(obj_fft_minus_rho)
        objFunc = np.flip(np.fft.ifft2(np.fft.ifftshift(obj_fft_minus_rho)))
    else:
        # D00 = |O(0)|^2 is real and positive in theory; the modulus avoids a spurious global phase from noise
        D00 = np.sqrt(np.abs(obj_WDD_f[center_sy][center_sx]))
        if D00 == 0:
            D00 = 1.0
        obj_fft_minus_rho = np.conj(obj_WDD_f)/D00

        # -Q of index j is 2*(N//2) - j; np.flip gives N-1-j, equal only for odd N, so even N needs a roll by one pixel
        obj_fft = np.flip(obj_fft_minus_rho)
        if sy%2 == 0:
            obj_fft = np.roll(obj_fft, 1, axis=0)
        if sx%2 == 0:
            obj_fft = np.roll(obj_fft, 1, axis=1)
        objFunc = np.fft.ifft2(np.fft.ifftshift(obj_fft))

    return obj_fft, objFunc



def WDD_engine(data_4D, 
               pPrime_pixel_size_dy, 
               pPrime_pixel_size_dx, 
               center_dy, 
               center_dx, 
               scan_rotation_angle, 
               scan_flip, 
               aperture_radius, 
               reciprocalSpace_pixel_size, 
               Voltage, 
               defocus,
               epsilon_WDD = 0.1,
               legacy_output = True,
               ):
    # defocus: scalar -> obj_fft, objFunc of shape (sy, sx), the result equals the single-defocus code.
    #          list / array -> defocus section series, obj_fft, objFunc of shape (n_defocus, sy, sx).
    # In the series case the data-dependent steps (FFT along p, ifft of the detector rows) are done only once,
    # only the defocus-dependent kernel is evaluated for every defocus.
    # legacy_output: see _WDD_finalize.
    print_and_log('')
    print_and_log('##################### WDD Parameters ######################')

    sy, sx, dy, dx = data_4D.shape
    np.maximum(data_4D, 0, out=data_4D)                     # same as data_4D[data_4D<0] = 0 (in place, as in the original)

    defocus_series = np.ndim(defocus) > 0
    defocus_list = np.atleast_1d(np.asarray(defocus, dtype=np.float64)).ravel()
    n_defocus = len(defocus_list)

    if not scan_flip:
        scan_rotation_angle *= -1
    theta_r = np.pi*scan_rotation_angle/180

    print_and_log(f'Trotter Pattern rotation angle: {scan_rotation_angle} degrees')
    print_and_log(f'Trotter Pattern flip x: {scan_flip}')
    print_and_log(f'WDD epsilon: {epsilon_WDD}')
    if defocus_series:
        print_and_log(f'defocus sections: {n_defocus}, from {defocus_list[0]:g} Å to {defocus_list[-1]:g} Å')

    center_sy = sy//2
    center_sx = sx//2

    print_and_log('')
    print_and_log('####################### WDD Process #######################')

    data_4D_fft01 = p_fft(data_4D)
    del data_4D

    ry, rx = _scan_shifts(sy, sx, center_sy, center_sx, pPrime_pixel_size_dy, pPrime_pixel_size_dx, reciprocalSpace_pixel_size, theta_r, scan_flip)

    Q2 = ry**2 + rx**2
    pick = np.unique(np.concatenate([np.argsort(Q2.ravel())[-3:], np.arange(0, Q2.size, max(1, Q2.size//5))]))
    test_centers = (np.concatenate([[center_dy, center_dy - 0.37*aperture_radius], center_dy - ry.ravel()[pick]]),
                    np.concatenate([[center_dx, center_dx + 0.52*aperture_radius], center_dx - rx.ravel()[pick]]))

    # defocus-dependent quantities, one entry per defocus
    ApeFunc0_list = []
    epsilon_list = []
    aperture_stack_list = []
    for defocus_i in defocus_list:
        ApeFunc0 = ApeFunc(center_dy, center_dx, dy, dx, aperture_radius, reciprocalSpace_pixel_size, Voltage, defocus_i)
        ApeFunc_WDD_0 = np.fft.ifft2(np.fft.ifftshift(ApeFunc0*np.conj(ApeFunc0)))
        epsilon_list.append(float(np.sum(np.abs(ApeFunc_WDD_0)**2)*epsilon_WDD))
        ApeFunc0_list.append(ApeFunc0.astype(np.complex64))
        aperture_stack_list.append(_make_aperture_stack(dy, dx, center_dy, center_dx, aperture_radius, reciprocalSpace_pixel_size, Voltage, defocus_i, test_centers))

    # Only the (int(center_dy), int(center_dx)) element of the r-FFT of obj_WDD is used afterwards, i.e. a single
    # frequency: it is evaluated directly for each p' row, so ApeFunc_WDD / obj_WDD / obj_WDD_fft23 are never stored.
    fy = int(center_dy) - dy//2
    fx = int(center_dx) - dx//2
    W_f = np.exp(-2j*np.pi*(fy*np.arange(dy)[:, None]/dy + fx*np.arange(dx)[None, :]/dx)).astype(np.complex64)
    obj_WDD_f = np.zeros((n_defocus, sy, sx), dtype=np.complex64)

    rows = _row_chunk(sy, sx*dy*dx)
    total_work = sy*n_defocus
    start_time = time.time()
    print("\rWDD progressing: {:^3.0f}%[{}->{}] ?iter/s ({:0>2}:{:0>2}:{:0>2}<??:??:??)".format(0,"*"*0,"."*10,0,0,0),end = "")
    for s0 in range(0, sy, rows):
        s1 = min(sy, s0+rows)
        c = s1 - s0

        D_rows = sfft.ifft2(sfft.ifftshift(data_4D_fft01[s0:s1], axes=(2, 3)), axes=(2, 3), workers=-1)       # shared by all defocus

        for fi in range(n_defocus):
            A = aperture_stack_list[fi]((center_dy - ry[s0:s1]).ravel(), (center_dx - rx[s0:s1]).ravel()).reshape(c, sx, dy, dx)
            np.conj(A, out=A)
            A *= ApeFunc0_list[fi]
            K = sfft.ifft2(sfft.ifftshift(A, axes=(2, 3)), axes=(2, 3), workers=-1)
            del A

            denominator = K.real**2
            denominator += K.imag**2
            denominator += epsilon_list[fi]
            np.conj(K, out=K)
            K *= D_rows                                      # conj(K)*D/(|K|^2+epsilon), evaluated at the single frequency
            K *= W_f
            K /= denominator
            obj_WDD_f[fi, s0:s1] = K.sum(axis=(2, 3))

            print("\r"+_progress_line('WDD', s0*n_defocus + (fi+1)*c, total_work, start_time),end = "")
    print('')
    log_to_file(_progress_line('WDD', total_work, total_work, start_time))

    total_time = time.time() - start_time
    print_and_log(f'WDD process finished in {total_time} s')


    obj_fft_list = []
    objFunc_list = []
    for fi in range(n_defocus):
        obj_fft_i, objFunc_i = _WDD_finalize(obj_WDD_f[fi], center_sy, center_sx, legacy_output)
        obj_fft_list.append(obj_fft_i)
        objFunc_list.append(objFunc_i)

    if not defocus_series:
        return obj_fft_list[0], objFunc_list[0]

    return np.stack(obj_fft_list), np.stack(objFunc_list)



def SSB_engine(data_4D, 
               pPrime_pixel_size_dy, 
               pPrime_pixel_size_dx, 
               center_dy, 
               center_dx, 
               scan_rotation_angle, 
               scan_flip, 
               aperture_radius, 
               reciprocalSpace_pixel_size, 
               ):
    print_and_log('')
    print_and_log('###################### SSB Parameters #####################')

    sy, sx, dy, dx = data_4D.shape
    np.maximum(data_4D, 0, out=data_4D)                     # same as data_4D[data_4D<0] = 0 (in place, as in the original)

    if not scan_flip:
        scan_rotation_angle *= -1
    theta_r = np.pi*scan_rotation_angle/180

    print_and_log(f'Trotter Pattern rotation angle: {scan_rotation_angle} degrees')
    print_and_log(f'Trotter Pattern flip x: {scan_flip}')

    center_sy = sy//2
    center_sx = sx//2

    print_and_log('')
    print_and_log('####################### SSB Process #######################')

    # Only the bright-field disk contributes (the masks lie inside BFdisk), so the detector is cropped to its
    # bounding box before the FFT along p. The q = 0 value is the sum over the whole detector, i.e. the total intensity.
    total_intensity = 0.0
    for sj in range(sy):
        total_intensity += float(data_4D[sj].sum())

    R = float(aperture_radius)
    y0 = max(0, int(np.floor(center_dy - R)))
    y1 = min(dy, int(np.ceil(center_dy + R)) + 1)
    x0 = max(0, int(np.floor(center_dx - R)))
    x1 = min(dx, int(np.ceil(center_dx + R)) + 1)
    h, w = y1-y0, x1-x0

    BFdisk = np.asarray(annular_mask(dy, dx,  0, aperture_radius, center_dy, center_dx))[y0:y1, x0:x1].astype(np.float32)
    obj_fft = np.zeros((sy, sx), dtype=np.complex64)

    data_4D_fft01 = p_fft(data_4D[:, :, y0:y1, x0:x1])
    del data_4D

    ry, rx = _scan_shifts(sy, sx, center_sy, center_sx, pPrime_pixel_size_dy, pPrime_pixel_size_dx, reciprocalSpace_pixel_size, theta_r, scan_flip)

    # disk membership via squared distances (separable in y and x), checked against r_map on the test shifts
    # (centre, random and the most distant shifts of this scan); r_map itself is used if they do not agree
    Y = np.arange(y0, y1, dtype=np.float64)[:, None]
    X = np.arange(x0, x1, dtype=np.float64)[None, :]
    R2 = R**2

    def inside_fast(yc, xc):                                # yc, xc: (c, sx) -> (c, sx, h, w) bool
        return ((Y - yc[:, :, None, None])**2 + (X - xc[:, :, None, None])**2) < R2

    def inside_slow(yc, xc):
        return np.stack([np.stack([r_map(dy, dx, y, x)[y0:y1, x0:x1] < aperture_radius for y, x in zip(yrow, xrow)]) for yrow, xrow in zip(yc, xc)])

    Q2 = ry**2 + rx**2
    pick = np.unique(np.concatenate([np.argsort(Q2.ravel())[-3:], np.arange(0, Q2.size, max(1, Q2.size//5))]))
    yt = np.concatenate([[center_dy - 0.37*R, center_dy + 0.61*R], center_dy - ry.ravel()[pick], center_dy + ry.ravel()[pick]])[None, :]
    xt = np.concatenate([[center_dx + 0.52*R, center_dx - 0.23*R], center_dx - rx.ravel()[pick], center_dx + rx.ravel()[pick]])[None, :]
    inside = inside_fast
    if np.mean(inside_fast(yt, xt) != inside_slow(yt, xt)) > 2e-4:
        print_and_log('SSB: r_map convention differs from squared-distance test, using r_map')
        inside = inside_slow

    rows = _row_chunk(sy, sx*h*w, budget=1e7)
    start_time = time.time()
    print("\rSSB progressing: {:^3.0f}%[{}->{}] ?iter/s ({:0>2}:{:0>2}:{:0>2}<??:??:??)".format(0,"*"*0,"."*10,0,0,0),end = "")
    for s0 in range(0, sy, rows):
        s1 = min(sy, s0+rows)

        # mask_p - mask_n = BFdisk * (in(center-r) - in(center+r))
        mask_used = inside(center_dy-ry[s0:s1], center_dx-rx[s0:s1]).astype(np.float32)
        mask_used -= inside(center_dy+ry[s0:s1], center_dx+rx[s0:s1])
        mask_used *= BFdisk
        obj_fft[s0:s1] = (data_4D_fft01[s0:s1]*mask_used).sum(axis=(2, 3))

        print("\r"+_progress_line('SSB', s1, sy, start_time),end = "")
    print('')
    log_to_file(_progress_line('SSB', sy, sy, start_time))

    obj_fft[center_sy][center_sx] = total_intensity

    total_time = time.time() - start_time
    print_and_log(f'SSB process finished in {total_time} s')

    obj_fft *= 1j
    objFunc = np.fft.ifft2(np.fft.ifftshift(obj_fft))

    return obj_fft, objFunc



def LINV3D_engine(data_4D, 
                  pPrime_pixel_size_dy, 
                  pPrime_pixel_size_dx, 
                  center_dy, 
                  center_dx, 
                  scan_rotation_angle, 
                  scan_flip, 
                  aperture_radius, 
                  reciprocalSpace_pixel_size, 
                  Voltage, 
                  defocus,
                  thickness,
                  n_slice = None,
                  ctf_mode = 'thick',
                  noise_normalization = True,
                  rcond = 1e-6,
                  qz_margin = 0,
                  ):
    # 3D-structure linear inversion (3D LINV) STEM, Ooe et al., arXiv:2610.15303
    # Thick weak phase object model, G(k, Q_p) = sum_l beta_hat(k, Q_p, q_z,l) * Phi_hat_l(Q_p), solved separately
    # for every Q_p for the q_z within the information limit.
    # ctf_mode 'thin':  every slice is a delta plane at z_j (Eq. 7 of the paper)
    # ctf_mode 'thick': every slice is a slab of thickness dz centred at z_j, the CTF is averaged over the slab
    #                   (each term of beta gets a sinc(omega*dz) factor; this is the default of the public code)
    # noise_normalization: the columns of the 5D-CTF matrix are divided by the noise-transfer weight before the truncated
    #                   solution (as in the public code), so rcond acts on the noise-normalized system.
    # rcond: relative singular value cutoff of the truncated least-squares solution (solved through the Hermitian normal
    #                   equations, eigenvalue cutoff rcond^2; the Q_p are processed in batches).
    print_and_log('')
    print_and_log('#################### 3D LINV Parameters ###################')

    sy, sx, dy, dx = data_4D.shape

    if ctf_mode not in ('thin', 'thick'):
        raise ValueError("ctf_mode must be 'thin' or 'thick'")
    thick = ctf_mode == 'thick'

    if not scan_flip:
        scan_rotation_angle *= -1
    theta_r = np.pi*scan_rotation_angle/180

    print_and_log(f'Trotter Pattern rotation angle: {scan_rotation_angle} degrees')
    print_and_log(f'Trotter Pattern flip x: {scan_flip}')

    center_sy = sy//2
    center_sx = sx//2

    # lens and depth sampling
    Voltage_V = Voltage*1e3
    wavelength = 12.2643/np.sqrt(Voltage_V*(1 + 0.97845e-6*Voltage_V))        # Å
    k0 = aperture_radius*reciprocalSpace_pixel_size                           # 1/Å
    if n_slice is None:
        n_slice = max(1, int(np.ceil(thickness*wavelength*k0**2)))             # Nyquist: N_z = [t' * lambda * k0^2]
    Nz = int(n_slice)
    dz = thickness/(Nz-1) if Nz > 1 else thickness
    z_list = defocus + dz*np.arange(Nz)                                       # z_j, Å
    q_list = np.fft.fftshift(np.fft.fftfreq(Nz, d=dz))                        # q_z,l, 1/Å
    dq = 1/(Nz*dz)
    E = np.exp(2j*np.pi*np.outer(z_list, q_list))                             # E[j, l] = exp(2 pi i q_l z_j)
    E_T_i = 1j*E.T                                                            # i * E^T, (l, j)

    # defocus term of the aberration function in detector pixel units: chi(k; z) = chi_coeff * z * |k_px|^2
    # (calibrated from Chi_defocus so that the sign convention of defocus is the same as in the other methods)
    Chi1, *_ = Chi_defocus(dy, dx, reciprocalSpace_pixel_size, Voltage, 1.0, center_dy, center_dx)
    chi_coeff = float(np.asarray(Chi1, dtype=np.float64)[0, 0]/((0-center_dy)**2 + (0-center_dx)**2))
    chi_coeff_theory = np.pi*wavelength*reciprocalSpace_pixel_size**2
    if abs(abs(chi_coeff)/chi_coeff_theory - 1) > 0.02:
        print_and_log(f'WARNING: Chi_defocus defocus coefficient ({chi_coeff:.4e}) differs from pi*lambda*dk^2 ({chi_coeff_theory:.4e})')

    print_and_log(f'Voltage: {Voltage} kV, wavelength: {wavelength:.5f} Å')
    print_and_log(f'aperture cutoff k0: {k0:.5f} 1/Å, q_z information limit: {wavelength*k0**2/2:.5f} 1/Å')
    print_and_log(f'Nyquist slice thickness: {1/(wavelength*k0**2):.2f} Å')
    print_and_log(f'slice number: {Nz}, slice thickness: {dz:.2f} Å')
    print_and_log(f'z range: {z_list[0]:.2f} Å to {z_list[-1]:.2f} Å')
    print_and_log(f'CTF mode: {ctf_mode}')
    print_and_log(f'noise normalization: {noise_normalization}')
    print_and_log(f'SVD cutoff rcond: {rcond}')
    print_and_log(f'q_z margin: {qz_margin} q_z steps')

    print_and_log('')
    print_and_log('###################### 3D LINV Process ####################')

    # detector cropped to the bounding box of the bright-field disk
    R = float(aperture_radius)
    R2 = R**2
    y0 = max(0, int(np.floor(center_dy - R)))
    y1 = min(dy, int(np.ceil(center_dy + R)) + 1)
    x0 = max(0, int(np.floor(center_dx - R)))
    x1 = min(dx, int(np.ceil(center_dx + R)) + 1)
    h, w = y1-y0, x1-x0
    N = sy*sx

    BFdisk = np.asarray(annular_mask(dy, dx, 0, aperture_radius, center_dy, center_dx))[y0:y1, x0:x1] > 0
    bf_idx = np.flatnonzero(BFdisk.ravel())
    M0 = len(bf_idx)
    ky = np.broadcast_to((np.arange(y0, y1) - center_dy)[:, None], (h, w)).ravel()[bf_idx]
    kx = np.broadcast_to((np.arange(x0, x1) - center_dx)[None, :], (h, w)).ravel()[bf_idx]

    # G(k, Q_p) = FFT_{R_p -> Q_p} [I(k, R_p)]
    G = p_fft(data_4D[:, :, y0:y1, x0:x1], clamp=True)
    del data_4D
    G = G.reshape(sy, sx, h*w)

    # intensity scale: |A(k)|^2 = 1 inside the bright-field disk, so the mean intensity there is the scale of I
    I_scale = float(np.mean(G[center_sy, center_sx, bf_idx].real)/N)
    if I_scale <= 0:
        raise ValueError('mean bright-field intensity is not positive')
    print_and_log(f'mean bright-field intensity per pixel: {I_scale}')

    ry, rx = _scan_shifts(sy, sx, center_sy, center_sx, pPrime_pixel_size_dy, pPrime_pixel_size_dx, reciprocalSpace_pixel_size, theta_r, scan_flip)
    Q2_map = ry**2 + rx**2
    valid = (Q2_map > 0) & (Q2_map <= 4*R2)                                   # 0 < |Q_p| <= 2 k0
    qj, qi = np.nonzero(valid)                                                # raster order
    qy_v, qx_v, Q2_v = ry[qj, qi], rx[qj, qi], Q2_map[qj, qi]
    Q_mag_v = np.sqrt(Q2_v)*reciprocalSpace_pixel_size                        # |Q_p|, 1/Å

    # q_z within the information limit, Eq. (14); with noise normalization only q_z where the noise weight is defined
    abs_q = np.abs(q_list)[None, :]
    qz_max_v = wavelength*(k0*Q_mag_v - Q_mag_v**2/2)
    sel_v = abs_q < (qz_max_v + qz_margin*dq)[:, None]
    if noise_normalization:
        # noise transfer weight, Eq. (24) inverted: w = [k0^2 - (|Q|/2 + |q_z|/(lambda |Q|))^2]^(1/4) / sqrt(lambda |Q|)
        bracket = k0**2 - (Q_mag_v[:, None]/2 + abs_q/(wavelength*Q_mag_v[:, None]))**2
        sel_v &= bracket > 0
        w_v = np.sqrt(np.sqrt(np.maximum(bracket, 0)))/np.sqrt(wavelength*Q_mag_v[:, None])
        if thick:
            w_v = w_v*np.abs(np.sinc(dz*q_list))[None, :]
        w_max = float(w_v[sel_v].max()) if sel_v.any() else 1.0
        col_scale = np.where(sel_v, 1/np.where(sel_v, w_v, 1), 0)
    else:
        col_scale = sel_v.astype(np.float64)

    # The phase is real, so G(k, -Q) = conj(G(k, Q)) and beta(k, -Q) = conj(beta(k, Q)), which gives exactly
    # Phi_hat_l(-Q) = conj(Phi_hat_{-l}(Q)): only one of Q and -Q has to be solved, the other one is filled in afterwards.
    # The Q to be solved are collected first, so that the progress is counted in equal work units.
    l_partner = 2*(Nz//2) - np.arange(Nz)                                     # index of -q_l in q_list
    l_has_partner = (l_partner >= 0) & (l_partner < Nz)
    pj, pi = 2*center_sy - qj, 2*center_sx - qi                               # index of -Q
    in_range = (pj >= 0) & (pj < sy) & (pi >= 0) & (pi < sx)
    pj, pi = np.clip(pj, 0, sy-1), np.clip(pi, 0, sx-1)
    fill = in_range & valid[pj, pi] & (pj*sx + pi < qj*sx + qi) & ~(sel_v & ~l_has_partner[None, :]).any(axis=1)
    solve_idx = np.flatnonzero(~fill)
    n_solve = len(solve_idx)

    Phi_hat = np.zeros((Nz, sy, sx), dtype=np.complex64)
    batch = max(1, int(1.5e6 // (M0*Nz)))
    start_time = time.time()
    print("\r3D LINV progressing: {:^3.0f}%[{}->{}] ?iter/s ({:0>2}:{:0>2}:{:0>2}<??:??:??)".format(0,"*"*0,"."*10,0,0,0),end = "")
    for b0 in range(0, n_solve, batch):
        ids = solve_idx[b0:b0+batch]
        gj, gi = qj[ids], qi[ids]
        qy, qx, Q2 = qy_v[ids][:, None], qx_v[ids][:, None], Q2_v[ids][:, None]

        # beta(k, Q; z_j) = i A(k) [A(k-Q) exp(i(chi(k)-chi(k-Q))) - A(k+Q) exp(i(chi(k+Q)-chi(k)))], arranged as (nb, Nz, M0).
        # z_j is equally spaced, so exp(i a z_j) is a geometric sequence in j (two exponentials per pixel instead of Nz)
        in_m = ((ky[None] - qy)**2 + (kx[None] - qx)**2) < R2                     # A(k - Q)
        in_p = ((ky[None] + qy)**2 + (kx[None] + qx)**2) < R2                     # A(k + Q)
        kQ = ky[None]*qy + kx[None]*qx
        a_m = chi_coeff*(2*kQ - Q2)                                               # d[chi(k) - chi(k-Q)]/dz
        a_p = chi_coeff*(2*kQ + Q2)                                               # d[chi(k+Q) - chi(k)]/dz
        w_m = in_m.astype(np.float64)
        w_p = in_p.astype(np.float64)
        if thick:
            w_m *= np.sinc(a_m*dz/(2*np.pi))                                      # slab average of exp(i a z) over dz
            w_p *= np.sinc(a_p*dz/(2*np.pi))
        beta_T = np.empty((len(ids), Nz, M0), dtype=np.complex128)
        cur_m, step_m = w_m*np.exp(1j*a_m*z_list[0]), np.exp(1j*a_m*dz)
        cur_p, step_p = w_p*np.exp(1j*a_p*z_list[0]), np.exp(1j*a_p*dz)
        for j in range(Nz):
            beta_T[:, j, :] = cur_m - cur_p                                       # beta / i
            cur_m = cur_m*step_m
            cur_p = cur_p*step_p

        B_T = E_T_i @ beta_T                                                      # (5D-CTF matrix)^T, Eq. (17): sum_j beta_j E[j, l], including the factor i
        del beta_T
        g = G[gj, gi][:, bf_idx].astype(np.complex128)/(N*I_scale)                # measured vectors, Eq. (16)

        # normal equations B^H B x = B^H g with the column scaling S (noise weight / q_z selection) applied to the small
        # matrices, truncated pseudo-inverse of the Hermitian matrix (cutoff rcond^2 relative)
        S = col_scale[ids]
        B_H = np.conj(B_T)
        Gram = (B_H @ B_T.transpose(0, 2, 1))*S[:, :, None]*S[:, None, :]
        h = (B_H @ g[:, :, None])*S[:, :, None]
        lam, V = np.linalg.eigh(Gram)                                             # ascending
        inv_lam = np.where(lam > rcond**2*lam[:, -1:], 1/np.where(lam > 0, lam, 1), 0)
        x = (V @ (inv_lam[:, :, None]*(np.conj(V).transpose(0, 2, 1) @ h)))[:, :, 0]
        Phi_hat[:, gj, gi] = x.T                                                  # Eq. (15)

        print("\r"+_progress_line('3D LINV', b0+len(ids), n_solve, start_time),end = "")
    print('')
    log_to_file(_progress_line('3D LINV', n_solve, n_solve, start_time))

    # fill the conjugate partners  Phi_hat_l(-Q) = conj(Phi_hat_{-l}(Q))
    fj, fi_ = qj[fill], qi[fill]
    pj_f, pi_f = pj[fill], pi[fill]
    sel_f = sel_v[fill]
    for l in range(Nz):
        if l_has_partner[l]:
            m = sel_f[:, l]
            Phi_hat[l, fj[m], fi_[m]] = np.conj(Phi_hat[l_partner[l], pj_f[m], pi_f[m]])

    total_time = time.time() - start_time
    print_and_log(f'3D LINV process finished in {total_time} s')

    if noise_normalization:
        # Phi_hat now holds w * Phi_hat (noise-normalized); only a global scale is left: divide by the largest weight
        Phi_hat /= w_max

    # Phi_j(Q_p) = sum_l Phi_hat_l(Q_p) exp(2 pi i q_l z_j), then 2D inverse FFT over Q_p -> phase of every slice (rad)
    Phi_Q = np.einsum('jl,lyx->jyx', E, Phi_hat)
    objFunc = N*sfft.ifft2(sfft.ifftshift(Phi_Q, axes=(1, 2)), axes=(1, 2), workers=-1)

    return Phi_hat, objFunc, z_list
