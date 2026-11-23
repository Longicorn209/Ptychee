"""
Author: Zeyu Wang
Date: November 2026
Description: "iterative ptychogrphay"
"""

import numpy as np
import time

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from skimage.restoration import unwrap_phase

import torch
from torch.fft import fft2, ifft2, fftn, ifftn, fftshift, ifftshift, fftfreq
import torch.nn.functional as F



from .utils import print_and_log, log_to_file, get_subplot_grid, RGB_Complex_Plot, calculate_wavelength, Chi_defocus





def calculate_iterPtycho_move(reciprocalSpace_pixel_size, scan_step, data_shape):
    
    dx = data_shape[3]

    realSpace_pixel_size = 1/reciprocalSpace_pixel_size/dx
    ptycho_move = scan_step/realSpace_pixel_size

    print_and_log(f'real space pixel size: {realSpace_pixel_size} Å/pixel')
    print_and_log(f'ptycho move: {ptycho_move} pixels')

    return ptycho_move



def initialize_iterPtycho_patch_position(data_shape, scan_rotation_angle, scan_flip, ptycho_move, Obj_pad):

    sy, sx, *_ = data_shape

    positions = np.zeros((sy*sx,2))
    positions[:, 0] = np.repeat(np.arange(sy), sx)
    positions[:, 1] = np.tile(np.arange(sx), sy)

    if scan_flip:  
        scan_rotation_angle *= -1
    theta_r = np.pi*scan_rotation_angle/180
    #R_cw_array = np.array([[np.cos(theta_r), -np.sin(theta_r)], [np.sin(theta_r), np.cos(theta_r)]])
    R_ccw_array = np.array([[np.cos(theta_r), np.sin(theta_r)], [-np.sin(theta_r), np.cos(theta_r)]])
    positions_spin = np.dot(R_ccw_array, positions.T)

    positions_spin[0] -= np.min(positions_spin[0])
    positions_spin[1] -= np.min(positions_spin[1])

    posset = np.zeros((sy*sx,5))
    posset[:, 0] = positions[:, 0]
    posset[:, 1] = positions[:, 1]
    posset[:, 2] = positions_spin[0]*ptycho_move + Obj_pad
    posset[:, 3] = positions_spin[1]*ptycho_move + Obj_pad
    posset[:, 4] = np.arange(sy*sx)

    if scan_flip:
        posset[:, 3] = np.max(posset[:, 3]) - posset[:, 3] + Obj_pad

    return posset



def initialize_iterPtycho_probe(paCBED, reciprocalSpace_pixel_size, Voltage, defocus, show_probe0=False):

    dy, dx = paCBED.shape

    chi, k_theta = Chi_defocus(dy, dx, reciprocalSpace_pixel_size, Voltage, defocus)

    Ronchi_norm = (paCBED - np.amin(paCBED)) / np.ptp(paCBED)
    BFdisk = np.ones(Ronchi_norm.shape) * (Ronchi_norm > 0.5)
    # ApeFunc = BFdisk * np.exp(-1j*chi) * np.mean(np.sqrt(paCBED))

    radius = np.sqrt(np.sum(BFdisk) / np.pi)
    qy, qx = np.ogrid[:dy, :dx]
    Aperture = ((qy - (dy - 1) / 2)**2 + (qx - (dx - 1) / 2)**2 <= radius**2).astype(np.float32)
    ApeFunc = Aperture * np.exp(-1j * chi) / np.mean(Aperture) * np.mean(np.sqrt(paCBED))

    # ApeFunc = Aperture * np.exp(-1j * chi) / np.sum(Aperture) * np.sqrt(np.sum(paCBED))

    # ApeFunc = np.sqrt(paCBED)*np.exp(-1j*chi) #/ np.mean(np.sqrt(paCBED))

    
    probe0 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ApeFunc)))

    if show_probe0:

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        im1 = axes[0].imshow(np.abs(probe0), cmap='gray')
        axes[0].set_title('Amplitude Prb0')
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        divider1 = make_axes_locatable(axes[0])
        cax1 = divider1.append_axes("right", size="5%", pad=0.2)
        fig.colorbar(im1, cax=cax1)

        im2 = axes[1].imshow(np.angle(probe0), cmap='jet')
        axes[1].set_title('Phase Prb0')
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        divider2 = make_axes_locatable(axes[1])
        cax2 = divider2.append_axes("right", size="5%", pad=0.2)
        fig.colorbar(im2, cax=cax2)

        plt.tight_layout()
        plt.show()

    return probe0, k_theta



def initialize_iterPtycho_probe_mixstates(paCBED, reciprocalSpace_pixel_size, Voltage, defocus, n_state, gene_mode='Hermite', show_probe_mixstates=False):
    print_and_log('')
    print_and_log(f'############### Probe Mixstates Parameters ###############')

    probe0, k_theta = initialize_iterPtycho_probe(paCBED, reciprocalSpace_pixel_size, Voltage, defocus)

    dy, dx = probe0.shape

    intensity = np.abs(probe0)**2
    total_energy = intensity.sum()

    xx = (np.arange(dx)-dx/2)
    yy = (np.arange(dy)-dy/2)

    xx_grid, yy_grid = np.meshgrid(xx, yy)
    rr = np.sqrt(xx_grid**2 + yy_grid**2)

    sort_idx = np.argsort(rr.flatten())
    cum_energy = np.cumsum(intensity.flatten()[sort_idx])
    w = rr.flatten()[sort_idx][cum_energy >= 0.9*total_energy][0]

    h_list = [[0,0],[0,1],[1,0],[2,0],[1,1],
            [0,2],[0,3],[1,2],[2,1],[3,0],
            [4,0],[3,1],[2,2],[1,3],[0,4],
            [1,4],[2,3],[3,2],[4,1],[4,2],
            [3,3],[2,4],[3,4],[4,3],[4,4]]
    
    proben = np.zeros((n_state, dy, dx), dtype=np.complex64)

    if gene_mode == 'linear':
        for i in range(n_state):
            proben[i] = (1-i/n_state)*probe0

    elif gene_mode == 'Hermite':
        X = xx/w
        Y = yy/w

        def hermiteH(n, x):
            if n == 0:
                return np.ones(len(x))
            elif n == 1:
                return 2*x
            else:
                return 2*x*hermiteH(n-1, x) - 2*(n-1)*hermiteH(n-2, x)

        for i in range(n_state):
            m, n = h_list[i]
            Hx, Hy = np.meshgrid(hermiteH(m, X)*np.exp(-X**2/2), hermiteH(n, Y)*np.exp(-Y**2/2))
            proben[i] = Hx*Hy*probe0/(m+1)**2/(n+1)**2

        intensities = np.sum(np.abs(proben)**2, axis=(-2, -1))
        intensities_order = np.argsort(intensities)[::-1]
        proben = proben[intensities_order]

    elif gene_mode == 'Laguerre':
        X = xx/w
        Y = yy/w
        X_grid, Y_grid = np.meshgrid(X, Y)
        
        R = np.sqrt(X_grid**2 + Y_grid**2)
        theta = np.arctan2(Y_grid, X_grid)

        def laguerre(n, k, x):
            if n == 0:
                return np.ones_like(x)
            elif n == 1:
                return (k + 1 - x)
            else:
                return ( (2*(n-1) + k + 1 - x) * laguerre(n-1, k, x) 
                    - (n-1 + k) * laguerre(n-2, k, x)) / n 
            
        for i in range(n_state):
            p, l = h_list[i]
            L = laguerre(p, l, 2*R**2)
            radial = (R**l)*L*np.exp(-R**2)
            angular = np.cos(l*theta)
            proben[i] = radial*angular*probe0 / ((p+1)**2*(l+1)**2)

            intensities = np.sum(np.abs(proben)**2, axis=(-2, -1))
            intensities_order = np.argsort(intensities)[::-1]
            proben = proben[intensities_order]


    if show_probe_mixstates == True:

        probe_shown_rows, probe_shown_cols = get_subplot_grid(n_state)

        energy_n = np.sum(np.abs(proben)**2, axis=(1,2))
        for i in range(n_state):
            print('  energy of '+str(i+1)+'th probe mode after: '+str(energy_n[i]))
            energy_pp = energy_n[i]/np.sum(energy_n)*100
            plt.subplot(probe_shown_rows, probe_shown_cols, i+1)
            RGB_Complex_Plot(proben[i])
            plt.title('energy proportion: {:^3.0f}%'.format(energy_pp))
        plt.show()

        for i in range(n_state):
            energy_pp = energy_n[i]/np.sum(energy_n)*100
            plt.subplot(probe_shown_rows, probe_shown_cols, i+1)
            plt.imshow(np.abs(proben[i]), cmap='gray')
            plt.title('Amplitude of '+str(i+1)+'th probe\nenergy proportion: {:^3.0f}%'.format(energy_pp))
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()
        plt.show()

        for i in range(n_state):
            energy_pp = energy_n[i]/np.sum(energy_n)*100
            plt.subplot(probe_shown_rows, probe_shown_cols, i+1)
            plt.imshow(np.angle(proben[i]), cmap='jet')
            plt.title('Phase of '+str(i+1)+'th probe\nenergy proportion: {:^3.0f}%'.format(energy_pp))
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()
        plt.show()

    print_and_log(f'Probe States: {n_state}')
    print_and_log(f'Probe Generation Mode: {gene_mode}')
    print_and_log(f'Probe Radius: {w} pixels')

    return proben, k_theta



def inialize_iterPtycho_object_multislice(data_shape, posset, n_slice, Obj_pad):
    *_ , dy, dx = data_shape

    print_and_log('')
    print_and_log(f'############# Object Multi-slice Parameters ##############')
    print_and_log(f'Object Slices: {n_slice}')

    syptp = np.max(posset[:, 2]) - np.min(posset[:, 2])
    sxptp = np.max(posset[:, 3]) - np.min(posset[:, 3])

    objectn = np.ones((n_slice, int(syptp+1)+dy+2*Obj_pad, int(sxptp+1)+dx+2*Obj_pad))
    print_and_log(f"recovered objFunc shape: {objectn.shape}")
    return objectn



def probe_orthogonalization(proben, n_state, device):
    # compute upper half of P* @ P
    pairwise_dot_product = torch.zeros((n_state, n_state), dtype=torch.complex64, device=device)

    for i in range(n_state):
        for j in range(i,n_state):
            pairwise_dot_product[i, j] = (proben[i].conj() * proben[j]).sum()
    # compute eigenvectors (effectively cheaper way of computing V* from SVD)
    _, evecs = torch.linalg.eigh(pairwise_dot_product, UPLO="U")
    proben = torch.tensordot(evecs.T, proben, dims=1)
    return proben



def build_pupil_phase_basis(dy, dx, NA, r_mask, device):
    y, x = torch.meshgrid(
        torch.linspace(-dy//2, dy//2-1, dy, device=device),
        torch.linspace(-dx//2, dx//2-1, dx, device=device),
        indexing='ij')

    rho_pixels = torch.sqrt(x**2 + y**2)
    mask = rho_pixels <= r_mask
    rho = rho_pixels * NA / r_mask
    theta = torch.atan2(y, x)

    basis = []

    basis.append(torch.ones_like(rho))  # Piston

    basis.append( rho * torch.cos(theta))  # Tilt X
    basis.append( rho * torch.sin(theta))  # Tilt Y

    basis.append(rho**2 * torch.cos(2*theta))      # Astigmatism 0°
    basis.append(0.5 * rho**2)                     # Defocus 1/2
    basis.append(rho**2 * torch.sin(2*theta))      # Astigmatism 45°

    basis.append((1/3) * rho**3 * torch.sin(3*theta)) # Trefoil X
    basis.append((1/3) * rho**3 * torch.sin(theta))   # Coma X
    basis.append((1/3) * rho**3 * torch.cos(theta))   # Coma Y
    basis.append((1/3) * rho**3 * torch.cos(3*theta)) # Trefoil Y

    basis.append((1/4) * rho**4 * torch.sin(4*theta))  # Quadrafoil 0°
    basis.append((1/4) * rho**4 * torch.sin(2*theta))  # Secondary Astig 45°
    basis.append((1/4) * rho**4)                        # Primary Spherical
    basis.append((1/4) * rho**4 * torch.cos(2*theta))  # Secondary Astig 0°
    basis.append((1/4) * rho**4 * torch.cos(4*theta))  # Quadrafoil 45°

    basis.append((1/5) * rho**5 * torch.sin(5*theta))  # Pentafoil 45°
    basis.append((1/5) * rho**5 * torch.sin(3*theta))  # Secondary Trefoil Y
    basis.append((1/5) * rho**5 * torch.sin(theta))    # Secondary Coma Y
    basis.append((1/5) * rho**5 * torch.cos(theta))    # Secondary Coma X
    basis.append((1/5) * rho**5 * torch.cos(3*theta))  # Secondary Trefoil X
    basis.append((1/5) * rho**5 * torch.cos(5*theta))  # Pentafoil 0°

    basis = torch.stack(basis, dim=0)
    return basis, mask



def fit_probe_with_aberration(probe, wavelength, basis, mask, device):

    pupil = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(probe)))
    amp = torch.abs(pupil)
    phase = torch.angle(pupil)

    phase_unwrap = torch.tensor(
        unwrap_phase((phase*mask).cpu().numpy()),
        dtype=torch.float32, device=device)


    B = basis.view(basis.shape[0], -1).T
    y = phase_unwrap.flatten()
    M = mask.flatten()
    B_mask = B[M]
    y_mask = y[M]

    coeff_phase = -torch.linalg.lstsq(B_mask, y_mask).solution  # rad

    phi_fit = torch.sum(coeff_phase[:, None, None] * basis, dim=0)
    phi_fit[~mask] = 0.0
    pupil_fit = amp * torch.exp(-1j * phi_fit)
    # pupil_fit[~mask] = 0.0
    probe_fit = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(pupil_fit)))


    aberrations_m = (wavelength / (2 * torch.pi)) * coeff_phase

    aberration_names = [
        "Piston","Tilt X","Tilt Y",
        "Astigmatism 0°","Defocus","Astigmatism 45°",
        "Trefoil X","Coma X","Coma Y","Trefoil Y",
        "Quadrafoil 0°","Secondary Astig 45°","Primary Spherical",
        "Secondary Astig 0°","Quadrafoil 45°",
        "Pentafoil 45°","Secondary Trefoil Y","Secondary Coma Y",
        "Secondary Coma X","Secondary Trefoil X","Pentafoil 0°"
    ]

    aberration_symbols = [
        "C0","T_x", "T_y",
        "A1_0", "C1", "A1_45",
        "B2_X", "C3_1X", "C3_1Y", "B2_Y",
        "A3_0", "A3_45s", "C3", "A3_45", "A4_0",
        "A5_45", "B5_Y", "C5_1Y", "C5_1X", "B5_X", "A6_0"
    ]   

    return {
        "probe_fit": probe_fit,
        "phi_fit": phi_fit,
        "coeff_phase_rad": coeff_phase,          # rad
        "coeff_wavefront_m": aberrations_m,     # 米
        "mask": mask,
        "aberration_names": aberration_names,
        "aberration_symbols": aberration_symbols,
    }


def apply_probe_support_mask(proben_GPU, mask):

    proben_ape_GPU = fftshift(fft2(ifftshift(proben_GPU, dim=(1, 2)), dim=(1, 2)), dim=(1, 2))
    proben_ape_GPU *= mask

    return fftshift(ifft2(ifftshift(proben_ape_GPU, dim=(1, 2)), dim=(1, 2)), dim=(1, 2))


def scan_position_clustering(posset_GPU, n_block, device, plot_cluster=False, mode='BKD'):

    if mode == 'Morton':        
        # Morton order spatial cluster
        pos_y = posset_GPU[:, 2].to(torch.int32)
        pos_x = posset_GPU[:, 3].to(torch.int32)
        max_coord = max(pos_y.max().item(), pos_x.max().item())
        n_bits = max(1, int(max_coord).bit_length())
        morton_code = torch.zeros_like(pos_x)
        for bit in range(n_bits):
            morton_code |= ((pos_x >> bit) & 1) << (2 * bit)
            morton_code |= ((pos_y >> bit) & 1) << (2 * bit + 1)
        position_order = torch.argsort(morton_code)
        posset_sorted = posset_GPU[position_order]
        spatial_blocks = list(torch.tensor_split(posset_sorted, n_block, dim=0))

    if mode == 'kmean':
        # K-mean clustering for scan positions
        time_0 = time.time()
        print_and_log('')
        print('\rK-mean clustering for scan positions started', end="")
        pos_xy = posset_GPU[:, 2:4].to(torch.float32)
        pos_min = pos_xy.min(dim=0).values
        pos_max = pos_xy.max(dim=0).values
        pos_scale = (pos_max - pos_min).clamp_min(1e-8)
        pos_xy_scaled = (pos_xy - pos_min) / pos_scale
        n_block_y = max(1, int(np.sqrt(n_block)))
        n_block_x = int(np.ceil(n_block / n_block_y))
        while n_block_y * n_block_x < n_block:
            n_block_y += 1
        y_centers = torch.linspace(0.5 / n_block_y, 1 - 0.5 / n_block_y, n_block_y, device=device)
        x_centers = torch.linspace(0.5 / n_block_x, 1 - 0.5 / n_block_x, n_block_x, device=device)
        cy, cx = torch.meshgrid(y_centers, x_centers, indexing='ij')
        centroids = torch.stack((cy.flatten(), cx.flatten()), dim=1)[:n_block]
        for _ in range(50):
            dist = torch.cdist(pos_xy_scaled, centroids)
            labels = torch.argmin(dist, dim=1)
            counts = torch.bincount(labels, minlength=n_block)
            empty = torch.where(counts == 0)[0]
            if empty.numel() > 0:
                largest = torch.argsort(counts, descending=True)
                for empty_id, largest_id in zip(empty.tolist(), largest.tolist()):
                    if counts[largest_id] > 1:
                        candidate = torch.where(labels == largest_id)[0]
                        farthest = candidate[torch.argmax(dist[candidate, largest_id])]
                        centroids[empty_id] = pos_xy_scaled[farthest]
                        labels[farthest] = empty_id
                        counts[largest_id] -= 1
                        counts[empty_id] += 1
            new_centroids = torch.zeros_like(centroids)
            new_centroids.index_add_(0, labels, pos_xy_scaled)
            new_centroids = new_centroids / counts.clamp_min(1).to(torch.float32).unsqueeze(1)
            if torch.allclose(centroids, new_centroids, atol=1e-5):
                break
            centroids = new_centroids
        spatial_blocks = [posset_GPU[labels == k] for k in range(n_block) if torch.any(labels == k)]
        print_and_log(f'\rK-mean clustering for scan positions finished in {time.time()-time_0:.2f} s')

    if mode == 'BKD':
        #Balanced KD partition 
        pos_xy = posset_GPU[:, 2:4].to(torch.float32) 
        N_pos = pos_xy.shape[0] 
        n_block = min(n_block, N_pos) 
        spatial_blocks = [] 
        def split_block(indices, n_parts): 
            n_points = indices.numel() 
            if n_parts == 1: 
                spatial_blocks.append(posset_GPU[indices]) 
                return 
            pos = pos_xy[indices] 
            ranges = pos.max(dim=0).values - pos.min(dim=0).values 
            axis = torch.argmax(ranges).item() 
            order = torch.argsort(pos[:, axis]) 
            indices_sorted = indices[order] 
            n_left_parts = n_parts // 2 
            n_right_parts = n_parts - n_left_parts 
            n_left = (n_points * n_left_parts) // n_parts 
            n_left = max(1, min(n_points - 1, n_left)) 
            n_right = n_points - n_left 
            split_block(indices_sorted[:n_left], n_left_parts) 
            split_block(indices_sorted[n_left:], n_right_parts) 
        indices = torch.arange(N_pos, device=posset_GPU.device) 
        split_block(indices, n_block)


    if plot_cluster:
        # plot scan positions clustering
        print('real n_block: '+str(len(spatial_blocks)))
        plt.figure(figsize=(8, 8))
        for cluster_id, block in enumerate(spatial_blocks):
            block_cpu = block[:, 2:4].cpu().numpy()
            plt.scatter(block_cpu[:, 1], block_cpu[:, 0], s=3, label=f'Cluster {cluster_id}')
        plt.gca().invert_yaxis()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Spatial Clustering: {n_block} Blocks')
        plt.tight_layout()
        plt.show()

    return spatial_blocks



def l2_fft_constraint(object_GPU, n_slice, l2_lambdaTikhonov):
    sy, sx = object_GPU.shape[-2:]
    ky = torch.fft.fftfreq(sy, d=1.0, device=object_GPU.device)
    kx = torch.fft.fftfreq(sx, d=1.0, device=object_GPU.device)
    ky, kx = torch.meshgrid(ky, kx, indexing='ij')
    k2 = 4*torch.sin(torch.pi*ky)**2 + 4*torch.sin(torch.pi*kx)**2

    for sn in range(n_slice):
        objSlice = object_GPU[sn]
        objfft = fft2(objSlice)
        objfft_abs = objfft.abs()

        objfft_abs = objfft_abs / (1 + l2_lambdaTikhonov*k2)

        objfft = objfft_abs*torch.exp(1j*objfft.angle())
        object_GPU[sn] = ifft2(objfft)

    return object_GPU



def l1_fft_constraint(object_GPU, n_slice, l1_softThreshold):
    for sn in range(n_slice):
        objSlice = object_GPU[sn]
        objfft = fft2(objSlice)
        objfft_abs = objfft.abs()

        objfft_abs_shift = torch.fft.fftshift(objfft_abs)
        objfft_abs_shift[0,0] = 0

        objfft_abs -= l1_softThreshold*objfft_abs_shift.mean()
        objfft_abs[objfft_abs<=0] = 0

        objfft = objfft_abs*torch.exp(1j*objfft.angle())
        object_GPU[sn] = ifft2(objfft)

    return object_GPU



def l0_fft_constraint(object_GPU, n_slice, l0_hardThreshold):
    for sn in range(n_slice):
        objSlice = object_GPU[sn]
        objfft = fft2(objSlice)
        objfft_abs = objfft.abs()

        hardThreshold = l0_hardThreshold*objfft_abs.mean()
        objfft_abs[objfft_abs<=hardThreshold] = 0

        objfft = objfft_abs*torch.exp(1j*objfft.angle())
        object_GPU[sn] = ifft2(objfft)

    return object_GPU



def build_kz_filter(Oy, Ox, n_slice, kz_beta, device):
    """kz regularization filter W = 1 - 2/pi * arctan2((kz_beta*qz)^2, qx^2 + qy^2), qz in cycles/slice, qx,qy in cycles/pixel.

    With physical frequencies kz = qz/dz and kr = q/dr this equals 1 - 2/pi*arctan((kz_beta*dz/dr)^2 * kz^2/kr^2),
    so for a fixed kz_beta the physical strength grows with the slice thickness dz. The engines therefore pass
    kz_beta = kz_regularization * slice_thickness / thickness_current when the thickness is optimised,
    i.e. kz_regularization keeps its meaning at the initial slice thickness.
    """
    qy = fftfreq(Oy, device=device)
    qx = fftfreq(Ox, device=device)
    qz = fftfreq(n_slice, device=device)
    qza, qya, qxa = torch.meshgrid(qz, qy, qx, indexing='ij')
    qz2 = (qza * kz_beta) ** 2
    qr2 = qxa ** 2 + qya ** 2
    return 1 - 2 / torch.pi * torch.arctan2(qz2, qr2)


def kz_constraint(object_GPU, Wz):
    return ifftn(fftn(object_GPU)*Wz)


def build_zblur_kernel(zblur_std, slice_thickness, device, dtype):
    """Gaussian blur kernel along z for the CURRENT slice thickness (both zblur_std and slice_thickness in Angstrom)."""
    kernel_size = 2 * int(np.ceil(5 * zblur_std / slice_thickness)) + 1
    z = (torch.arange(kernel_size, device=device) - kernel_size // 2) * slice_thickness
    z_kernel = torch.exp(-0.5 * (z / zblur_std)**2)
    z_kernel /= z_kernel.sum()
    return z_kernel.reshape(1, 1, kernel_size).to(dtype=dtype)


def zblur_constraint(object_GPU, z_kernel):
    n_slice = object_GPU.shape[0]
    kernel_size = z_kernel.shape[-1]
    pad = kernel_size // 2

    obj = object_GPU.permute(1, 2, 0).reshape(-1, 1, n_slice)
    obj = F.pad(obj, (pad, pad), mode='replicate')
    obj = F.conv1d(obj, z_kernel)

    object_GPU = obj.reshape(object_GPU.shape[1], object_GPU.shape[2], n_slice).permute(2, 0, 1)

    return object_GPU


def rh_constraint(object_GPU, n_slice, rh_hardThreshold=0):
    for sn in range(n_slice):
        objSlice = object_GPU[sn]
        objSlice_phase = objSlice.angle()
        objSlice_abs = objSlice.abs()
                
        objSlice_phase[objSlice_phase<rh_hardThreshold] = 0
        object_GPU[sn] = objSlice_abs*torch.exp(1j*objSlice_phase)

    return object_GPU



def POA_constraint(object_GPU):
    return torch.exp(1j*object_GPU.angle())



def FFT_phase0_offset(object_GPU, n_slice, FFT_phase_offset=0):
    for sn in range(n_slice):
        objSlice = object_GPU[sn]
        objfft = torch.fft.fft2(objSlice)
        objfft_phase = torch.angle(objfft)

        objfft_phase[0][0] += FFT_phase_offset

        objfft = torch.abs(objfft)*torch.exp(1j*objfft_phase)
        object_GPU[sn] = torch.fft.ifft2(objfft)

    return object_GPU



def build_adaptive_propagator(k_theta, theta_x, theta_y, slice_thickness, wavelength, dy, dx, device):
    qy = torch.fft.fftshift(torch.fft.fftfreq(dy, device=device))
    qx = torch.fft.fftshift(torch.fft.fftfreq(dx, device=device))
    qya, qxa = torch.meshgrid(qy, qx, indexing='ij')
    q_norm = torch.sqrt(qxa**2 + qya**2).clamp_min(1e-12)
    kx_theta = k_theta * qxa / q_norm
    ky_theta = k_theta * qya / q_norm
    phase = -torch.pi * slice_thickness / wavelength * (k_theta**2 - 2*kx_theta[None] * torch.tan(theta_x[:, None, None]) - 2*ky_theta[None] * torch.tan(theta_y[:, None, None]))
    return torch.fft.fftshift(torch.exp(1j * phase), dim=(-2, -1))


def adaptive_propagator_k_grids(k_theta, dy, dx, device):
    """kx, ky, k^2 defined exactly as in build_adaptive_propagator, but returned in FFT (un-shifted) order."""
    qy = torch.fft.fftshift(torch.fft.fftfreq(dy, device=device))
    qx = torch.fft.fftshift(torch.fft.fftfreq(dx, device=device))
    qya, qxa = torch.meshgrid(qy, qx, indexing='ij')
    q_norm = torch.sqrt(qxa**2 + qya**2).clamp_min(1e-12)
    kx_theta = torch.fft.fftshift(k_theta * qxa / q_norm, dim=(-2, -1))
    ky_theta = torch.fft.fftshift(k_theta * qya / q_norm, dim=(-2, -1))
    k_square = torch.fft.fftshift(k_theta**2, dim=(-2, -1))
    return kx_theta, ky_theta, k_square


def gaussian_smooth_maps(maps, sigma):
    """Separable Gaussian smoothing of a (C, sy, sx) stack, replicate padding. sigma in scan pixels, 0 = off."""
    if sigma <= 0:
        return maps
    rad = max(1, int(np.ceil(3.0 * sigma)))
    x = torch.arange(-rad, rad + 1, device=maps.device, dtype=maps.dtype)
    kernel = torch.exp(-0.5 * (x / sigma)**2)
    kernel = kernel / kernel.sum()
    n_c = maps.shape[0]
    m = maps.unsqueeze(0)
    m = F.pad(m, (rad, rad, 0, 0), mode='replicate')
    m = F.conv2d(m, kernel.view(1, 1, 1, -1).repeat(n_c, 1, 1, 1), groups=n_c)
    m = F.pad(m, (0, 0, rad, rad), mode='replicate')
    m = F.conv2d(m, kernel.view(1, 1, -1, 1).repeat(n_c, 1, 1, 1), groups=n_c)
    return m[0]


def amplitude_tangent_jacobians(Prb_ms, object_patches, psi_fft, amplitudes, calculated_amplitudes, propagators_block, detector_mask, i_dphase, D_init, e_A=1e-6):
    """Forward-mode (tangent) Jacobians dA/dp of the calculated amplitude, for several parameters p_j at once.

    Multislice model: Prb_{n+1} = F^-1[ P * F[Prb_n * O_n] ],  Psi = F[Prb_{N-1} * O_{N-1}],  A = sqrt(sum_states |Psi|^2).
    For every parameter j a tangent D_n = dPrb_n/dp_j is propagated slice by slice:
        D_{n+1} = F^-1[ P * F[D_n * O_n]  +  i*(dphase/dp_j) * P * F[Prb_n * O_n] ]
        dPsi    = F[ D_{N-1} * O_{N-1} ]
        dA/dp_j = sum_states Re( conj(Psi) * dPsi ) / A
    i_dphase[j]: i*d(phase of P)/dp_j (B,dy,dx) for propagator parameters (tilt, thickness), None otherwise.
    D_init[j]  : D_0 = dPrb_0/dp_j (S,B,dy,dx) for probe-side parameters (scan position), None otherwise.
    Returns residual r = (A_meas - A_calc) * mask and the list of J_j = dA/dp_j * mask, each (B,dy,dx).
    """
    with torch.no_grad():
        n_slice = object_patches.shape[0]
        n_par = len(i_dphase)
        mask = detector_mask.to(calculated_amplitudes.dtype)
        residual = (amplitudes - calculated_amplitudes) * mask

        P = propagators_block.unsqueeze(0)
        D = list(D_init)
        need_b = any(v is not None for v in i_dphase)
        for sr in range(n_slice - 1):
            patch_sr = object_patches[sr].unsqueeze(0)
            if need_b:
                b_fft = fft2(Prb_ms[sr] * patch_sr, dim=(2, 3)) * P        # = F[Prb_{sr+1}]
            for j in range(n_par):
                X = 0
                if i_dphase[j] is not None:
                    X = X + i_dphase[j].unsqueeze(0) * b_fft
                if D[j] is not None:
                    X = X + P * fft2(D[j] * patch_sr, dim=(2, 3))
                D[j] = ifft2(X, dim=(2, 3))

        amp_floor = e_A * calculated_amplitudes.amax(dim=(-2, -1), keepdim=True)
        A_safe = torch.maximum(calculated_amplitudes, amp_floor).clamp_min(1e-30)
        patch_last = object_patches[-1].unsqueeze(0)
        J = []
        for j in range(n_par):
            dpsi_fft = fftshift(fft2(D[j] * patch_last, dim=(2, 3)), dim=(2, 3))
            J.append((psi_fft.conj() * dpsi_fft).real.sum(dim=0) / A_safe * mask)
        return residual, J


def solve_gn_2x2(g_a, g_b, H_aa, H_ab, H_bb, s, e_damp, step_max):
    """Damped 2x2 Gauss-Newton solve, element-wise over positions: step = s * (H + mu*I)^-1 g, |step| <= step_max.

    mu = e_damp * mean(trace(H)/2) is relative to the average information, so it does not depend on the data scale.
    """
    mu = e_damp * 0.5 * (H_aa + H_bb).mean()
    a = H_aa + mu
    d = H_bb + mu
    det = (a * d - H_ab**2).clamp_min(1e-6 * mu**2)
    step_a = s * (d * g_a - H_ab * g_b) / det
    step_b = s * (-H_ab * g_a + a * g_b) / det
    return step_a.clamp(-step_max, step_max), step_b.clamp(-step_max, step_max)


def accumulate_adaptive_propagator(tilt_g, tilt_H, tilt_c, thick_sum, theta_x_map, theta_y_map, thickness_param, pos_ids, Prb_ms, object_patches, psi_fft, amplitudes, calculated_amplitudes, propagators_block, k_theta, wavelength, detector_mask, dy, dx, device, fit_tilt=True, fit_thickness=True, e_A=1e-6):
    """Accumulate Gauss-Newton statistics of the tilt / thickness update (nothing is updated here).

    Per block it stores
      tilt_g[pos]  = [<r,J_x>, <r,J_y>]                 (N_pos, 2)
      tilt_H[pos]  = [<J_x,J_x>, <J_x,J_y>, <J_y,J_y>]  (N_pos, 3)
      tilt_c[pos]  = [<J_x,J_z>, <J_y,J_z>]             (N_pos, 2)  tilt-thickness cross terms, only if both are fitted
      thick_sum   += [<r,J_z>, <J_z,J_z>, <r,r>]        (3,) float64, summed over all blocks of the iteration
    with r = A_meas - A_calc and <.,.> the sum over detector pixels.
    d(phase)/d(parameter) with phase = -pi*dz/lam * (k^2 - 2*kx*tan(tx) - 2*ky*tan(ty)):
      d/dtx = 2*pi*dz/lam * kx / cos^2(tx),  d/dty = 2*pi*dz/lam * ky / cos^2(ty),  d/ddz = -pi/lam * (k^2 - 2*kx*tan(tx) - 2*ky*tan(ty))
    """
    with torch.no_grad():
        n_slice = object_patches.shape[0]
        if n_slice < 2:
            return
        pos_ids = pos_ids.long()
        theta_x = theta_x_map.reshape(-1)[pos_ids]
        theta_y = theta_y_map.reshape(-1)[pos_ids]

        kx_theta, ky_theta, k_square = adaptive_propagator_k_grids(k_theta, dy, dx, device)
        tan_x = torch.tan(theta_x)[:, None, None]
        tan_y = torch.tan(theta_y)[:, None, None]
        cos_x = torch.cos(theta_x)[:, None, None]
        cos_y = torch.cos(theta_y)[:, None, None]
        pref = torch.pi * thickness_param / wavelength

        i_dphase = []   # order: [tx, ty] (if fit_tilt), then [dz] (if fit_thickness)
        if fit_tilt:
            i_dphase.append(1j * 2 * pref * kx_theta[None] / cos_x**2)
            i_dphase.append(1j * 2 * pref * ky_theta[None] / cos_y**2)
        if fit_thickness:
            i_dphase.append(-1j * torch.pi / wavelength * (k_square[None] - 2 * kx_theta[None] * tan_x - 2 * ky_theta[None] * tan_y))

        residual, J = amplitude_tangent_jacobians(Prb_ms, object_patches, psi_fft, amplitudes, calculated_amplitudes, propagators_block, detector_mask, i_dphase, [None] * len(i_dphase), e_A)

        thick_sum[2] += (residual * residual).sum().double()
        if fit_tilt:
            tilt_g[pos_ids, 0] = (residual * J[0]).sum(dim=(-2, -1))
            tilt_g[pos_ids, 1] = (residual * J[1]).sum(dim=(-2, -1))
            tilt_H[pos_ids, 0] = (J[0] * J[0]).sum(dim=(-2, -1))
            tilt_H[pos_ids, 1] = (J[0] * J[1]).sum(dim=(-2, -1))
            tilt_H[pos_ids, 2] = (J[1] * J[1]).sum(dim=(-2, -1))
        if fit_thickness:
            thick_sum[0] += (residual * J[-1]).sum().double()
            thick_sum[1] += (J[-1] * J[-1]).sum().double()
        if fit_tilt and fit_thickness:
            tilt_c[pos_ids, 0] = (J[0] * J[-1]).sum(dim=(-2, -1))
            tilt_c[pos_ids, 1] = (J[1] * J[-1]).sum(dim=(-2, -1))


def update_adaptive_propagator_gn(theta_x_map, theta_y_map, thickness_param, tilt_g, tilt_H, tilt_c, thick_sum, s_apc_theta=0.3, s_apc_thickness=0.3, e_theta=0.1, smooth_sigma=1.0, theta_step_max=1e-3, theta_abs_max=0.05, thickness_step_max_rel=0.05, thickness_min=1e-3, zero_mean_step=False):
    """Gauss-Newton update of the tilt maps and the slice thickness, called once per iteration after all blocks.

    thickness: step = s_apc_thickness * sum<r,Jz> / sum<Jz,Jz>, limited to thickness_step_max_rel per iteration.
    tilt: normal equations g, H are Gaussian-smoothed over the scan (smooth_sigma pixels), then solved per position (solve_gn_2x2).

    Returns the relative Newton decrement (convergence monitor): the decrease of the data misfit <r,r>/2 that the FULL update
    proposed here (s = 1, no step limit) is predicted to give, divided by <r,r>/2:
        2 * (g.d - d.H.d / 2) / <r,r>,   d = (d_thickness, d_tilt per position)
    g and H are the joint data statistics (per-position tilt blocks, thickness, and their cross terms tilt_c);
    d_tilt comes from the smoothed normal equations. It is dimensionless, <= 1, and goes to ~0 when the parameters have nothing left to gain.
    """
    with torch.no_grad():
        sy, sx = theta_x_map.shape
        lin = 0.0
        quad = 0.0
        full_dz = 0.0

        if s_apc_thickness > 0 and thick_sum[1].item() > 0:
            full_dz = (thick_sum[0] / thick_sum[1]).item()
            lin += thick_sum[0].item() * full_dz
            quad += thick_sum[1].item() * full_dz**2
            step_thick = s_apc_thickness * full_dz
            lim = thickness_step_max_rel * thickness_param.item()
            step_thick = max(-lim, min(lim, step_thick))
            thickness_param.add_(step_thick).clamp_(min=thickness_min)

        if s_apc_theta > 0 and tilt_H.abs().sum().item() > 0:
            maps = torch.cat([tilt_g, tilt_H], dim=1).t().reshape(5, sy, sx)
            maps = gaussian_smooth_maps(maps, smooth_sigma)
            full_tx, full_ty = solve_gn_2x2(maps[0], maps[1], maps[2], maps[3], maps[4], 1.0, e_theta, float('inf'))
            full_tx, full_ty = full_tx.reshape(-1), full_ty.reshape(-1)
            lin += (tilt_g[:, 0] * full_tx + tilt_g[:, 1] * full_ty).sum().item()
            quad += (tilt_H[:, 0] * full_tx**2 + 2 * tilt_H[:, 1] * full_tx * full_ty + tilt_H[:, 2] * full_ty**2).sum().item()
            quad += 2 * full_dz * (tilt_c[:, 0] * full_tx + tilt_c[:, 1] * full_ty).sum().item()

            step_tx, step_ty = solve_gn_2x2(maps[0], maps[1], maps[2], maps[3], maps[4], s_apc_theta, e_theta, theta_step_max)
            if zero_mean_step:
                step_tx = step_tx - step_tx.mean()
                step_ty = step_ty - step_ty.mean()
            theta_x_map.add_(step_tx).clamp_(-theta_abs_max, theta_abs_max)
            theta_y_map.add_(step_ty).clamp_(-theta_abs_max, theta_abs_max)

        newton_decrement = 0.0
        if thick_sum[2].item() > 0:
            newton_decrement = 2 * (lin - 0.5 * quad) / thick_sum[2].item()
        return newton_decrement


def subpixel_ramp(pos_y, pos_x, dy, dx, device):
    """Fourier phase ramp exp(-2*pi*i*(qy*fy + qx*fx)) (B,dy,dx) for the fractional part (fy,fx) of the scan positions.

    The object patch is cut at floor(position); the remaining sub-pixel offset is applied to the probe,
    P(k) -> P(k - f), because psi(k) = P(k - p) O(k) is the same wave in the object frame (intensities are identical).
    """
    fy = (pos_y - torch.floor(pos_y)).to(torch.float32)
    fx = (pos_x - torch.floor(pos_x)).to(torch.float32)
    qy = fftfreq(dy, device=device)[None, :, None]
    qx = fftfreq(dx, device=device)[None, None, :]
    return torch.exp(-2j * torch.pi * (qy * fy[:, None, None] + qx * fx[:, None, None]))


def subpixel_shift_probe(probe, ramp, b_size):
    """probe (S,dy,dx) -> per-position sub-pixel shifted probe (S,B,dy,dx). ramp=None: no shift (plain view)."""
    if ramp is None:
        return probe.unsqueeze(1).expand(-1, b_size, -1, -1)
    return ifft2(fft2(probe, dim=(1, 2)).unsqueeze(1) * ramp.unsqueeze(0), dim=(2, 3))


def subpixel_unshift_sum(g, ramp):
    """Adjoint of subpixel_shift_probe followed by the sum over positions: (S,B,dy,dx) -> (S,dy,dx). ramp=None: plain sum."""
    if ramp is None:
        return g.sum(dim=1)
    return ifft2((fft2(g, dim=(2, 3)) * ramp.conj().unsqueeze(0)).sum(dim=1), dim=(1, 2))


def accumulate_position(pos_g, pos_H, pos_ids, Prb_ms, object_patches, psi_fft, amplitudes, calculated_amplitudes, propagators_block, detector_mask, dy, dx, device, e_A=1e-6):
    """Accumulate Gauss-Newton statistics of the scan-position update (nothing is updated here).

    The tangent starts at the sub-pixel shifted probe, D_0 = dPrb_0/d(position) = F^-1[ -2*pi*i*q * F[Prb_0] ],
    and is propagated through ALL slices (amplitude_tangent_jacobians), so every state and every slice is included.
    pos_g[pos] = [<r,J_y>, <r,J_x>] (N_pos, 2),  pos_H[pos] = [<J_y,J_y>, <J_y,J_x>, <J_x,J_x>] (N_pos, 3).
    Requires the sub-pixel shifted probe, i.e. Prb_ms[0] must have been built with subpixel_shift_probe and a ramp.
    """
    with torch.no_grad():
        pos_ids = pos_ids.long()
        qy = fftfreq(dy, device=device)[None, None, :, None]
        qx = fftfreq(dx, device=device)[None, None, None, :]
        probe_fft = fft2(Prb_ms[0], dim=(2, 3))
        D_init = [ifft2(-2j * torch.pi * qy * probe_fft, dim=(2, 3)), ifft2(-2j * torch.pi * qx * probe_fft, dim=(2, 3))]

        residual, J = amplitude_tangent_jacobians(Prb_ms, object_patches, psi_fft, amplitudes, calculated_amplitudes, propagators_block, detector_mask, [None, None], D_init, e_A)

        pos_g[pos_ids, 0] = (residual * J[0]).sum(dim=(-2, -1))
        pos_g[pos_ids, 1] = (residual * J[1]).sum(dim=(-2, -1))
        pos_H[pos_ids, 0] = (J[0] * J[0]).sum(dim=(-2, -1))
        pos_H[pos_ids, 1] = (J[0] * J[1]).sum(dim=(-2, -1))
        pos_H[pos_ids, 2] = (J[1] * J[1]).sum(dim=(-2, -1))


def update_scan_positions_gn(posset_blocks, pos_g, pos_H, Oy, Ox, dy, dx, s_position_correction, e_pos=0.1, pos_step_max=0.3, remove_mean=True):
    """Gauss-Newton update of the scan positions, called once per iteration after all blocks (in-place on posset_blocks).

    step = s_position_correction * (H + mu*I)^-1 g per position (solve_gn_2x2), |step| <= pos_step_max pixels.
    remove_mean: a common shift of all positions cannot be observed (the object follows it), so the mean step is removed.
    Returns the sum over positions of |step| (pixels).
    """
    with torch.no_grad():
        if pos_H.abs().sum().item() == 0:
            return 0.0
        step_y, step_x = solve_gn_2x2(pos_g[:, 0], pos_g[:, 1], pos_H[:, 0], pos_H[:, 1], pos_H[:, 2], s_position_correction, e_pos, pos_step_max)
        if remove_mean:
            step_y = step_y - step_y.mean()
            step_x = step_x - step_x.mean()
        for pos_block in posset_blocks:
            ids = pos_block[:, 4].long()
            pos_block[:, 2] += step_y[ids]
            pos_block[:, 3] += step_x[ids]
            pos_block[:, 2].clamp_(0, Oy - 1 - dy)
            pos_block[:, 3].clamp_(0, Ox - 1 - dx)
        return torch.sqrt(step_y**2 + step_x**2).sum()


@torch.no_grad()
def LSQML_engine(iter_max, s_O, s_P,
                data_4D, Voltage, alpha, aperture_radius, 
                posset, proben0, objectn, slice_thickness, k_theta,
                device=None,
                n_block=None, 
                position_clustering=False, 
                plot_cluster = False,
                noise_model='amplitude',
                detector_mask=None,
                e_f=1e-9, 
                e_g=1e-1, 
                e_LSQ=5e-1,
                probe_orthog_constr=False, 
                sorting_probe=True, 
                probe_fit=False,
                probe_support_mask=False,
                object_all_states=False,     # True: object update uses all probe states (incoherent sum) instead of state 0 only
                s_apc_theta=0,               # step of the adaptive propagator tilt update, 0 = off
                s_apc_thickness=0,           # step of the adaptive propagator thickness update, 0 = off
                apc_start_iteration=None,
                s_position_correction=0,
                pc_start_iteration=None,
                pc_interval = 5,             # after pc_start_iteration the scan positions are updated once every pc_interval iterations
                subpixel_position=True,     # True: cut object patches at floor(position) and apply the sub-pixel part to the probe
                POA=False,
                l2_fft_lambdaTikhonov=0, 
                l1_fft_softThreshold=0, 
                l0_fft_hardThreshold=0, 
                kz_regularization=0, 
                zblur_std=0,
                rh_positive_phase=False, 
                FFT_phase_offset=0,
                ):
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    sy, sx, dy, dx = data_4D.shape
    n_state = proben0.shape[0]
    n_slice = objectn.shape[0]
    N_pos = sy * sx

    if n_block is None:
        n_block = sy

    n_block = min(n_block, N_pos)

    adaptive_propagator = s_apc_theta > 0 or s_apc_thickness > 0

    position_correction = s_position_correction > 0

    # if s_position_correction > 0:
    #     subpixel_position = True

    if pc_start_iteration is None:
        pc_start_iteration = 0.25 * iter_max
    if apc_start_iteration is None:
        apc_start_iteration = 0.25 * iter_max

    pc_start_iteration = int(np.ceil(pc_start_iteration))
         
    if detector_mask is None:
        detector_mask = np.ones((dy, dx), dtype=np.float32)

    print_and_log(f'Initialized Slice Thickness: {slice_thickness} Å')
    print_and_log('')
    print_and_log(f'############### LSQML Iteration Parameters ###############')
    print_and_log(f"Device Using: {device}")
    print_and_log(f'Number of Iterations: {iter_max}')
    print_and_log(f'Step for Updating Object: {s_O}')
    print_and_log(f'Step for Updating Probe: {s_P}')
    print_and_log(f'Noise Model: {noise_model}')
    print_and_log(f'Epsilon for Recipro Optim: {e_f}')
    print_and_log(f'Epsilon for Real Optim: {e_g}')
    print_and_log(f'Epsilon for LSQ: {e_LSQ}')
    # print_and_log(f'Object Update from All Probe States: {object_all_states}')
    print_and_log(f'')
    print_and_log(f'Probe Constraints:')
    print_and_log(f'    Number of Position Blocks: {n_block}')
    print_and_log(f'    Average Block Size: {N_pos / n_block:.2f} positions')
    print_and_log(f'    Spatial Position Clustering: {position_clustering}')
    print_and_log(f'    Probe Orthog Constraint: {probe_orthog_constr}')
    print_and_log(f'    Probe Support Mask: {probe_support_mask}')
    print_and_log(f'    Sorting Probe by Energy: {sorting_probe}')
    print_and_log(f'')
    print_and_log(f'Object Constraints:')
    print_and_log(f'    Phase Object Approximation (POA): {POA}')
    print_and_log(f'    Object Positive Phase: {rh_positive_phase}')
    print_and_log(f'    Object FFT LambdaTikhonov (L2): {l2_fft_lambdaTikhonov}')
    print_and_log(f'    Object FFT SoftThreshold (L1): {l1_fft_softThreshold}')
    print_and_log(f'    Object FFT HardThreshold (L0): {l0_fft_hardThreshold}')
    print_and_log(f'    Object kz Regularization: {kz_regularization}')
    print_and_log(f'    Object zblur Std: {zblur_std} Å')
    print_and_log(f'    Object FFT phase offset: {FFT_phase_offset}')
    if adaptive_propagator:
        print_and_log(f'')
        print_and_log(f'Adaptive Propagator Activited:')
        print_and_log(f'    Step for Adaptive Propagator Tilt: {s_apc_theta}')
        print_and_log(f'    Step for Adaptive Propagator Thickness: {s_apc_thickness}')
        print_and_log(f'    Start for Adaptive Propagator: {apc_start_iteration}th iteration')
    if position_correction:
        print_and_log(f'')
        print_and_log(f'Scan Position Correction Activited:')
        print_and_log(f'    Step for Position Correction: {s_position_correction}')
        print_and_log(f'    Start for Position Correction: {pc_start_iteration}th iteration')
        print_and_log(f'    Interval for Position Correction: every {pc_interval} iterations')
        print_and_log(f'    Sub-pixel Probe Shift: {subpixel_position}')
    

    
    wavelength = calculate_wavelength(Voltage)
    propagators = np.fft.fftshift(np.exp(-1j*2*np.pi/wavelength*0.5*slice_thickness*k_theta**2))

    data_4D[data_4D < 0] = 0
    data_4D_sqrt_GPU = torch.from_numpy(np.reshape(np.sqrt(data_4D, dtype=np.float32), (N_pos, dy, dx))).to(device)
    proben_GPU = torch.from_numpy(proben0.astype(np.complex64)).to(device)
    object_GPU = torch.from_numpy(objectn.astype(np.complex64)).to(device)
    posset_GPU = torch.from_numpy(posset.astype(np.float32)).to(device)
    propagators_GPU = torch.from_numpy(propagators.astype(np.complex64)).to(device)

    if adaptive_propagator:
        theta_x_map = torch.zeros((sy, sx), dtype=torch.float32, device=device)
        theta_y_map = torch.zeros((sy, sx), dtype=torch.float32, device=device)
        thickness_param = torch.tensor(float(slice_thickness), dtype=torch.float32, device=device)
        k_theta_GPU = torch.as_tensor(k_theta, dtype=torch.float32, device=device)
        thickness_history = torch.zeros(iter_max, dtype=torch.float32, device=device)
        tilt_std_history = torch.zeros(iter_max, dtype=torch.float32, device=device)
        newton_decrement_history = torch.zeros(iter_max, dtype=torch.float32, device=device)
        prop_tilt_g = torch.zeros((sy * sx, 2), dtype=torch.float32, device=device)
        prop_tilt_H = torch.zeros((sy * sx, 3), dtype=torch.float32, device=device)
        prop_tilt_c = torch.zeros((sy * sx, 2), dtype=torch.float32, device=device)
        prop_thick_sum = torch.zeros(3, dtype=torch.float32, device=device)

    _, Oy, Ox = object_GPU.shape
    y_ind = torch.arange(dy, device=device, dtype=torch.int32)
    x_ind = torch.arange(dx, device=device, dtype=torch.int32)

    err = torch.zeros(iter_max, device=device)
    Prb_LSQ_step = torch.zeros((iter_max, n_state), device=device)
    Obj_LSQ_step = torch.zeros((iter_max, n_slice), device=device)
    pc_shift = torch.zeros(iter_max, device=device)

    pos_g = torch.zeros((N_pos, 2), dtype=torch.float32, device=device)
    pos_H = torch.zeros((N_pos, 3), dtype=torch.float32, device=device)

    if kz_regularization > 0:
        Wz = build_kz_filter(Oy, Ox, n_slice, kz_regularization, device)

    if zblur_std > 0:
        z_kernel = build_zblur_kernel(zblur_std, slice_thickness, device, object_GPU.dtype)

    detector_mask_GPU = torch.from_numpy(detector_mask).to(device=device, dtype=torch.float32)

    if probe_fit or probe_support_mask:
        basis, apeture_mask = build_pupil_phase_basis(dy, dx, alpha, aperture_radius, device)

    if position_clustering:
        spatial_blocks = scan_position_clustering(posset_GPU, n_block, device=device, plot_cluster=plot_cluster)


    start_time = time.time()
    print_and_log('')
    print_and_log(f'###################### LSQML Process #####################')
    print("\rLSQML progressing: {:^3.0f}%[{}->{}] ?iter/s ({:0>2}:{:0>2}:{:0>2}<??:??:??)".format(0, "*" * 0, "." * 10, 0, 0, 0), end="")

    for i in range(iter_max):
        time_i0 = time.time()
        err_u = 0
        err_d = 0

        fit_tilt = s_apc_theta > 0 and i >= apc_start_iteration
        fit_thickness = s_apc_thickness > 0 and i >= apc_start_iteration
        pc_active = s_position_correction > 0 and i >= pc_start_iteration and (i - pc_start_iteration) % pc_interval == 0

        if adaptive_propagator:
            prop_tilt_g.zero_()
            prop_tilt_H.zero_()
            prop_tilt_c.zero_()
            prop_thick_sum.zero_()

        if pc_active:
            pos_g.zero_()
            pos_H.zero_()


        if position_clustering:
            block_order = torch.randperm(n_block).tolist()
            posset_blocks = [spatial_blocks[j] for j in block_order]
        else:
            idx = torch.randperm(posset_GPU.shape[0], device=device)
            posset_shuffled = posset_GPU[idx]
            posset_blocks = list(torch.tensor_split(posset_shuffled, n_block, dim=0))

        if probe_orthog_constr:
            proben_GPU = probe_orthogonalization(proben_GPU, n_state, device=device)

        if sorting_probe:
            intensities = proben_GPU.abs().pow(2).sum(dim=(-2, -1))
            intensities_order = torch.argsort(intensities, descending=True)
            proben_GPU = proben_GPU[intensities_order]

        if i == 0 or (i + 1) % 10 == 0:
            gObj_d = torch.zeros((Oy, Ox), dtype=torch.complex64, device=device)

            for pos_block in posset_blocks:
                b_size = pos_block.shape[0]
                ind_y = (pos_block[:, 2][:, None, None] + y_ind[None, :, None]).to(dtype=torch.int32)
                ind_x = (pos_block[:, 3][:, None, None] + x_ind[None, None, :]).to(dtype=torch.int32)
                indice = ind_y * Ox + ind_x
                if object_all_states:
                    proben_i = proben_GPU.abs().pow(2).sum(dim=0).unsqueeze(0).expand(b_size, dy, dx).to(torch.complex64)
                else:
                    proben_i = proben_GPU[0].abs().pow(2).unsqueeze(0).expand(b_size, dy, dx).to(torch.complex64)
                gObj_d = gObj_d.ravel().index_add_(0, indice.ravel(), proben_i.ravel()).reshape((Oy, Ox))

        for pos_block in posset_blocks:
            b_size = pos_block.shape[0]
            if b_size == 0:
                continue
            poss_1D = pos_block[:, 4].to(dtype=torch.int32)
            ind_y = (pos_block[:, 2][:, None, None] + y_ind[None, :, None]).to(dtype=torch.int32)
            ind_x = (pos_block[:, 3][:, None, None] + x_ind[None, None, :]).to(dtype=torch.int32)
            indice = ind_y * Ox + ind_x
            Illuminated_patches = object_GPU[:, ind_y, ind_x]
            amplitudes_patches = data_4D_sqrt_GPU[poss_1D]

            amplitudes_patches *= detector_mask_GPU

            if adaptive_propagator:
                theta_x_now = theta_x_map.reshape(-1)[poss_1D.long()]
                theta_y_now = theta_y_map.reshape(-1)[poss_1D.long()]
                propagators_block = build_adaptive_propagator(k_theta_GPU, theta_x_now, theta_y_now, thickness_param, wavelength, dy, dx, device)
            else:
                propagators_block = propagators_GPU.unsqueeze(0).expand(b_size, dy, dx)

            ramp = subpixel_ramp(pos_block[:, 2], pos_block[:, 3], dy, dx, device) if subpixel_position else None
            Prb_ms = [subpixel_shift_probe(proben_GPU, ramp, b_size)]

            for s in range(n_slice):
                psi = Prb_ms[s] * Illuminated_patches[s].unsqueeze(0)

                if s != n_slice - 1:
                    psifft = fft2(psi, dim=(2, 3))
                    psifft *= propagators_block[None, :, :, :]
                    Prb_ms.append(ifft2(psifft, dim=(2, 3)))

            psi_fft = fftshift(fft2(psi, dim=(2, 3)), dim=(2, 3))
            a_psi_fft = psi_fft.abs().pow(2).sum(dim=0).sqrt()

            if fit_tilt or fit_thickness:
                accumulate_adaptive_propagator(
                    prop_tilt_g, prop_tilt_H, prop_tilt_c, prop_thick_sum,
                    theta_x_map, theta_y_map, thickness_param, poss_1D.long(),
                    Prb_ms, Illuminated_patches, psi_fft, amplitudes_patches,
                    a_psi_fft, propagators_block, k_theta_GPU, wavelength, detector_mask_GPU, dy, dx, device,
                    fit_tilt=fit_tilt, fit_thickness=fit_thickness)

            if pc_active:
                accumulate_position(
                    pos_g, pos_H, poss_1D.long(), Prb_ms, Illuminated_patches, psi_fft, amplitudes_patches,
                    a_psi_fft, propagators_block, detector_mask_GPU, dy, dx, device)


            if noise_model == 'amplitude':
                chi_fft_R = amplitudes_patches / (a_psi_fft + e_f) - 1

            elif noise_model == 'anscombe':
                I_m = amplitudes_patches.pow(2)
                I_c = a_psi_fft.pow(2)
                chi_fft_R = (I_m + 0.375).sqrt() / ((I_c + 0.375).sqrt() + e_f) - 1

            elif noise_model == 'poisson':
                I_m = amplitudes_patches.pow(2)
                I_c = a_psi_fft.pow(2)
                chi_fft_R = 0.5 * (I_m / (I_c + e_f) - 1)
                chi_fft_R = torch.minimum(chi_fft_R, amplitudes_patches / (a_psi_fft + e_f))

            elif noise_model == 'gaussian':
                I_m = amplitudes_patches.pow(2)
                I_c = a_psi_fft.pow(2)
                I_ref = I_c.amax(dim=(-2, -1), keepdim=True)
                chi_fft_R = (I_m - I_c) / (2 * I_ref + e_f)


            chi_fft_R *= detector_mask_GPU

            chi_fft = psi_fft * chi_fft_R.unsqueeze(0).to(torch.complex64)
            chi = ifft2(ifftshift(chi_fft, dim=(2, 3)), dim=(2, 3))


            for sr in range(n_slice - 1, -1, -1):
                if sr != n_slice - 1:
                    chi_fft = fft2(chi, dim=(2, 3))
                    chi_fft /= propagators_block[None, :, :, :]
                    chi = ifft2(chi_fft, dim=(2, 3))

                if object_all_states:
                    gObj_i = (chi * Prb_ms[sr].conj()).sum(dim=0)
                else:
                    gObj_i = chi[0] * Prb_ms[sr][0].conj()
                gObj_u = torch.zeros((Oy, Ox), dtype=torch.complex64, device=device)
                gObj_u = gObj_u.ravel().index_add_(0, indice.ravel(), gObj_i.ravel()).reshape((Oy, Ox))
                gObj = gObj_u / (gObj_d**2 + (e_g * gObj_d.abs().max())**2).sqrt()
                #gObj = gObj_u / (gObj_d  + e_g * gObj_d.abs().max())
                #gObj = gObj_u / (gObj_d_i**2 + (e_g * gObj_d_i.abs().max())**2).sqrt()

                gObj_patches = gObj[ind_y, ind_x]
                if object_all_states:
                    gOP = gObj_patches.unsqueeze(0) * Prb_ms[sr]
                    Mr_0 = (gOP.conj() * chi).real.sum(dim=(0, 2, 3))
                    Ml_00 = gOP.abs().pow(2).sum(dim=(0, 2, 3))
                else:
                    gOP = gObj_patches * Prb_ms[sr].sum(dim=0)
                    Mr_0 = (gOP.conj() * chi.sum(dim=0)).real.sum(dim=(1, 2))
                    Ml_00 = gOP.abs().pow(2).sum(dim=(1, 2))
                Ml_00 += e_LSQ * Ml_00.mean()
                a_O = (Mr_0 / Ml_00 * s_O).mean() / n_slice
                Obj_LSQ_step[i][sr] += a_O / n_block
                object_GPU[sr] += a_O * gObj

                gPrb_i = chi * Illuminated_patches[sr].unsqueeze(0).conj()

                if sr == 0:
                    gPrb_u = subpixel_unshift_sum(gPrb_i, ramp)
                    gPrb_d = Illuminated_patches[sr].abs().pow(2).sum(dim=0).unsqueeze(0)
                    #gPrb = gPrb_u / (gPrb_d**2 + (e_g * gPrb_d.max())**2).sqrt()
                    #gPrb = gPrb_u / (gPrb_d + e_g * gPrb_d.max())
                    gPrb = gPrb_u / gPrb_d

                    gPO = subpixel_shift_probe(gPrb, ramp, b_size) * Illuminated_patches[sr].unsqueeze(0)
                    Mr_1 = (gPO.conj() * chi).real.sum(dim=(2, 3))
                    Ml_11 = gPO.abs().pow(2).sum(dim=(2, 3))
                    Ml_11 += e_LSQ * Ml_11.mean()
                    a_P = (Mr_1 / Ml_11).mean(dim=1) * s_P
                    Prb_LSQ_step[i] += a_P / n_block
                    proben_GPU += a_P[:, None, None] * gPrb
                chi = gPrb_i

            I_m = amplitudes_patches.pow(2)
            I_c = psi_fft.abs().pow(2).sum(dim=0)
            A_c = I_c.sqrt()
            m = detector_mask_GPU

            if noise_model == 'amplitude':
                err_u += ((amplitudes_patches - A_c) * m).pow(2).sum()
                err_d += (amplitudes_patches * m).pow(2).sum()
            elif noise_model == 'anscombe':
                err_u += (((I_m + 0.375).sqrt() - (I_c + 0.375).sqrt()) * m).pow(2).sum()
                err_d += ((I_m + 0.375) * m).sum()
            elif noise_model == 'gaussian':
                err_u += ((I_m - I_c) * m).pow(2).sum()
                err_d += (I_m * m).pow(2).sum()
            elif noise_model == 'poisson':
                Ic = I_c.clamp_min(1e-12)
                dev = 2 * (Ic - I_m + torch.where(I_m > 0, I_m * (I_m.clamp_min(1e-12) / Ic).log(), torch.zeros_like(I_m)))
                err_u += (dev * m).sum()
                err_d += m.sum() * Ic.shape[0] 

        if torch.isnan(err_u / err_d):
            print('')
            print('Interrupted')
            plt.figure(figsize=(6,4))
            plt.semilogy(range(len(err[:i].cpu())), err[:i].cpu(), color='red')
            plt.xlabel("Iteration")
            plt.ylabel("Error")
            plt.title("Convergence Curve at Interruption", color='red')
            plt.tight_layout()
            plt.show()

            raise ValueError("err is NaN. Try using different values of s_O and s_P")
        
        err[i] = err_u / err_d
        if pc_active:
            pc_shift[i] = update_scan_positions_gn(posset_blocks, pos_g, pos_H, Oy, Ox, dy, dx, s_position_correction)

        posset_GPU = torch.cat(posset_blocks, dim=0)

        if probe_fit:
            for ns in range(n_state):
                proben_GPU[ns] = fit_probe_with_aberration(proben_GPU[ns], wavelength, basis, apeture_mask, device=device)["probe_fit"]

        if probe_support_mask and i > iter_max/4:
            proben_GPU = apply_probe_support_mask(proben_GPU, apeture_mask)

        if l2_fft_lambdaTikhonov > 0:
            object_GPU = l2_fft_constraint(object_GPU, n_slice, l2_fft_lambdaTikhonov)

        if l1_fft_softThreshold > 0:
            object_GPU = l1_fft_constraint(object_GPU, n_slice, l1_fft_softThreshold)

        if l0_fft_hardThreshold > 0:
            object_GPU = l0_fft_constraint(object_GPU, n_slice, l0_fft_hardThreshold)

        if kz_regularization > 0:
            if adaptive_propagator:
                Wz = build_kz_filter(Oy, Ox, n_slice, kz_regularization * slice_thickness / thickness_param.item(), device)
            object_GPU = kz_constraint(object_GPU, Wz)

        if zblur_std > 0:
            if adaptive_propagator:
                z_kernel = build_zblur_kernel(zblur_std, thickness_param.item(), device, object_GPU.dtype)
            object_GPU = zblur_constraint(object_GPU, z_kernel)

        if FFT_phase_offset > 0 and i > iter_max * 2 / 3:
            object_GPU = FFT_phase0_offset(object_GPU, n_slice, FFT_phase_offset)

        if rh_positive_phase:
            object_GPU = rh_constraint(object_GPU, n_slice, rh_hardThreshold=0)

        if POA:
            object_GPU = POA_constraint(object_GPU)

        if adaptive_propagator:
            with torch.no_grad():

                newton_decrement_history[i] = update_adaptive_propagator_gn(
                    theta_x_map, theta_y_map, thickness_param, prop_tilt_g, prop_tilt_H, prop_tilt_c, prop_thick_sum,
                    s_apc_theta=s_apc_theta if fit_tilt else 0, s_apc_thickness=s_apc_thickness if fit_thickness else 0)

                thickness_history[i] = thickness_param
                tilt_std_history[i] = torch.sqrt(theta_x_map.var(unbiased=False) + theta_y_map.var(unbiased=False))

        time_is = time.time() - time_i0
        speed = 1 / max(time_is, 1e-12)
        process = (i + 1) / iter_max * 100
        aa = "*" * int(process / 10)
        bb = "." * (10 - int(process / 10))
        dur = int(time.time() - start_time)
        time_remain = int((iter_max - i - 1) * time_is)
        dur_h = dur // 3600
        dur_m = (dur - dur_h * 3600) // 60
        dur_s = dur - dur_h * 3600 - dur_m * 60
        time_remain_h = time_remain // 3600
        time_remain_m = (time_remain - time_remain_h * 3600) // 60
        time_remain_s = time_remain - time_remain_h * 3600 - time_remain_m * 60

        print("\rLSQML progressing: {:^3.0f}%[{}->{}] {:.2f}iter/s ({:0>2}:{:0>2}:{:0>2}<{:0>2}:{:0>2}:{:0>2})".format(process, aa, bb, speed, dur_h, dur_m, dur_s, time_remain_h, time_remain_m, time_remain_s), end="")

    print('')
    gpu_time = time.time() - start_time


    msLSQML_Obj = np.array(object_GPU.cpu())
    msLSQML_Prb = np.array(proben_GPU.cpu())
    # Prb_EW = np.array(psi[0][0].cpu())
    Prb_EW = np.array(psi.sum(axis=0).mean(axis=0).cpu())
    msLSQML_err = np.array(err.cpu())
    Obj_LSQ_step = np.array(Obj_LSQ_step.cpu())
    Prb_LSQ_step = np.array(Prb_LSQ_step.cpu())
    pc_mean_shift = np.array((pc_shift/sy/sx).cpu())
    cbed_comparison = np.stack([amplitudes_patches[0].cpu(), a_psi_fft[0].cpu(), amplitudes_patches[-1].cpu(), a_psi_fft[-1].cpu()])
    posset = np.array(posset_GPU.cpu())

    log_to_file("LSQML progressing: {:^3.0f}%[{}->{}] {:.2f}iter/s ({:0>2}:{:0>2}:{:0>2}<{:0>2}:{:0>2}:{:0>2})".format(process, aa, bb, speed, dur_h, dur_m, dur_s, time_remain_h, time_remain_m, time_remain_s))
    print_and_log(f'LSQML process finished in {gpu_time} s')

    if probe_fit:
        print_and_log(f'')
        print_and_log(f'################# probe fitting results ##################')
        probe_fit_results = fit_probe_with_aberration(proben_GPU[0], wavelength, basis, apeture_mask, device=device)
        print_and_log("    Fitted Aberration:")
        for i_aberr, (symbol, name, coeff_m) in enumerate(zip(probe_fit_results["aberration_symbols"],probe_fit_results["aberration_names"],probe_fit_results["coeff_wavefront_m"])):
            print_and_log(f"    {(i_aberr):02d}. {symbol:8s} | {name:28s} = {(coeff_m).item():+.3e} A")

    app_parameters = None
    if adaptive_propagator:
        app_parameters = {
            'theta_x': theta_x_map.cpu().numpy(),
            'theta_y': theta_y_map.cpu().numpy(),
            'slice_thickness': float(thickness_param.cpu()),
            'thickness_history': thickness_history.cpu().numpy(),
            'tilt_std_history': tilt_std_history.cpu().numpy() * 1000,
            'newton_decrement_history': newton_decrement_history.cpu().numpy(),
            's_apc_theta': s_apc_theta,
            's_apc_thickness': s_apc_thickness,
            'apc_start_iteration': apc_start_iteration,
        }
        if s_apc_thickness > 0:
            print_and_log(f'Optimized Slice Thickness: {app_parameters["slice_thickness"]:.6f} Å')
        if s_apc_theta > 0:
            print_and_log(f'Final Tilt Spatial Std: {app_parameters["tilt_std_history"][-1]:.6f} mrad')
            print_and_log(f'Final Tilt x Spatial Mean: {app_parameters["theta_x"].mean():.6f} mrad')
            print_and_log(f'Final Tilt y Spatial Mean: {app_parameters["theta_y"].mean():.6f} mrad')
        # print_and_log(f'Final Newton Decrement: {app_parameters["newton_decrement_history"][-1]:.3e}')

    return msLSQML_Obj, msLSQML_Prb, Prb_EW, msLSQML_err, Obj_LSQ_step, Prb_LSQ_step, pc_mean_shift, posset, cbed_comparison, app_parameters




@torch.no_grad()
def ePIE_engine(iter_max, s_O, s_P,
                data_4D, Voltage, alpha, aperture_radius,
                posset, proben0, objectn, slice_thickness, k_theta,
                device=None,
                n_block=None,
                position_clustering=False,
                plot_cluster=False,
                noise_model='amplitude',
                detector_mask=None,
                e_f=1e-9,
                alpha_O=0,
                alpha_P=0,
                probe_orthog_constr=False,
                sorting_probe=True,
                probe_fit=False,
                probe_support_mask=False,
                object_all_states=False,
                s_apc_theta=0,
                s_apc_thickness=0,
                apc_start_iteration=None,
                s_position_correction=0,
                pc_start_iteration=None,
                pc_interval=5,     
                subpixel_position=True,
                POA=False,
                l2_fft_lambdaTikhonov=0,
                l1_fft_softThreshold=0,
                l0_fft_hardThreshold=0,
                kz_regularization=0,
                zblur_std=0,
                rh_positive_phase=False,
                FFT_phase_offset=0,
                ):

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    
    sy, sx, dy, dx = data_4D.shape
    n_state = proben0.shape[0]
    n_slice = objectn.shape[0]
    N_pos = sy * sx

    if n_block is None:
        n_block = sy
    n_block = min(n_block, N_pos)

    adaptive_propagator = s_apc_theta > 0 or s_apc_thickness > 0

    position_correction = s_position_correction > 0

    # if s_position_correction > 0:
    #     subpixel_position = True

    if pc_start_iteration is None:
        pc_start_iteration = 0.25 * iter_max
    if apc_start_iteration is None:
        apc_start_iteration = 0.25 * iter_max

    pc_start_iteration = int(np.ceil(pc_start_iteration))

    if detector_mask is None:
        detector_mask = np.ones((dy, dx), dtype=np.float32)

    print_and_log(f'Initialized Slice Thickness: {slice_thickness} Å')
    print_and_log('')
    print_and_log(f'############### LSQML Iteration Parameters ###############')
    print_and_log(f"Device Using: {device}")
    print_and_log(f'Number of Iterations: {iter_max}')
    print_and_log(f'Step for Updating Object: {s_O}')
    print_and_log(f'Step for Updating Probe: {s_P}')
    print_and_log(f'Noise Model: {noise_model}')
    print_and_log(f'Epsilon for Recipro Optim: {e_f}')
    print_and_log(f'rPIE-alpha for Object: {alpha_O }')
    print_and_log(f'rPIE-alpha for Probe: {alpha_P}')
    # print_and_log(f'Object Update from All Probe States: {object_all_states}')
    print_and_log(f'')
    print_and_log(f'Probe Constraints:')
    print_and_log(f'    Number of Position Blocks: {n_block}')
    print_and_log(f'    Average Block Size: {N_pos / n_block:.2f} positions')
    print_and_log(f'    Spatial Position Clustering: {position_clustering}')
    print_and_log(f'    Probe Orthog Constraint: {probe_orthog_constr}')
    print_and_log(f'    Probe Support Mask: {probe_support_mask}')
    print_and_log(f'    Sorting Probe by Energy: {sorting_probe}')
    print_and_log(f'')
    print_and_log(f'Object Constraints:')
    print_and_log(f'    Phase Object Approximation (POA): {POA}')
    print_and_log(f'    Object Positive Phase: {rh_positive_phase}')
    print_and_log(f'    Object FFT LambdaTikhonov (L2): {l2_fft_lambdaTikhonov}')
    print_and_log(f'    Object FFT SoftThreshold (L1): {l1_fft_softThreshold}')
    print_and_log(f'    Object FFT HardThreshold (L0): {l0_fft_hardThreshold}')
    print_and_log(f'    Object kz Regularization: {kz_regularization}')
    print_and_log(f'    Object zblur Std: {zblur_std} Å')
    print_and_log(f'    Object FFT phase offset: {FFT_phase_offset}')
    if adaptive_propagator:
        print_and_log(f'')
        print_and_log(f'Adaptive Propagator Activited:')
        print_and_log(f'    Step for Adaptive Propagator Tilt: {s_apc_theta}')
        print_and_log(f'    Step for Adaptive Propagator Thickness: {s_apc_thickness}')
        print_and_log(f'    Start for Adaptive Propagator: {apc_start_iteration}th iteration')
    if position_correction:
        print_and_log(f'')
        print_and_log(f'Scan Position Correction Activited:')
        print_and_log(f'    Step for Position Correction: {s_position_correction}')
        print_and_log(f'    Start for Position Correction: {pc_start_iteration}th iteration')
        print_and_log(f'    Interval for Position Correction: every {pc_interval} iterations')
        print_and_log(f'    Sub-pixel Probe Shift: {subpixel_position}')

    wavelength = calculate_wavelength(Voltage)
    propagators = np.fft.fftshift(np.exp(-1j*2*np.pi/wavelength*0.5*slice_thickness*k_theta**2))

    data_4D[data_4D < 0] = 0
    data_4D_sqrt_GPU = torch.from_numpy(np.reshape(np.sqrt(data_4D, dtype=np.float32), (N_pos, dy, dx))).to(device)
    proben_GPU = torch.from_numpy(proben0.astype(np.complex64)).to(device)
    object_GPU = torch.from_numpy(objectn.astype(np.complex64)).to(device)
    posset_GPU = torch.from_numpy(posset.astype(np.float32)).to(device)
    propagators_GPU = torch.from_numpy(propagators.astype(np.complex64)).to(device)

    if adaptive_propagator:
        theta_x_map = torch.zeros((sy, sx), dtype=torch.float32, device=device)
        theta_y_map = torch.zeros((sy, sx), dtype=torch.float32, device=device)
        thickness_param = torch.tensor(float(slice_thickness), dtype=torch.float32, device=device)
        k_theta_GPU = torch.as_tensor(k_theta, dtype=torch.float32, device=device)
        thickness_history = torch.zeros(iter_max, dtype=torch.float32, device=device)
        tilt_std_history = torch.zeros(iter_max, dtype=torch.float32, device=device)
        newton_decrement_history = torch.zeros(iter_max, dtype=torch.float32, device=device)
        prop_tilt_g = torch.zeros((sy * sx, 2), dtype=torch.float32, device=device)
        prop_tilt_H = torch.zeros((sy * sx, 3), dtype=torch.float32, device=device)
        prop_tilt_c = torch.zeros((sy * sx, 2), dtype=torch.float32, device=device)
        prop_thick_sum = torch.zeros(3, dtype=torch.float32, device=device)

    _, Oy, Ox = object_GPU.shape
    y_ind = torch.arange(dy, device=device, dtype=torch.int32)
    x_ind = torch.arange(dx, device=device, dtype=torch.int32)

    err = torch.zeros(iter_max, device=device)
    pc_shift = torch.zeros(iter_max, device=device)

    pos_g = torch.zeros((N_pos, 2), dtype=torch.float32, device=device)
    pos_H = torch.zeros((N_pos, 3), dtype=torch.float32, device=device)

    if kz_regularization > 0:
        Wz = build_kz_filter(Oy, Ox, n_slice, kz_regularization, device)

    if zblur_std > 0:
        z_kernel = build_zblur_kernel(zblur_std, slice_thickness, device, object_GPU.dtype)

    detector_mask_GPU = torch.from_numpy(detector_mask).to(device=device, dtype=torch.float32)

    if probe_fit or probe_support_mask:
        basis, apeture_mask = build_pupil_phase_basis(dy, dx, alpha, aperture_radius, device)

    if position_clustering:
        spatial_blocks = scan_position_clustering(posset_GPU, n_block, device=device, plot_cluster=plot_cluster)


    start_time = time.time()
    print_and_log('')
    print_and_log('###################### ePIE Process ######################')
    print('\rePIE progressing: {:^3.0f}%[{}->{}] ?iter/s ({:0>2}:{:0>2}:{:0>2}<??:??:??)'.format(0, '*' * 0, '.' * 10, 0, 0, 0), end='')


    for i in range(iter_max):
        time_i0 = time.time()
        err_u = 0
        err_d = 0

        fit_tilt = s_apc_theta > 0 and i >= apc_start_iteration
        fit_thickness = s_apc_thickness > 0 and i >= apc_start_iteration
        pc_active = s_position_correction > 0 and i >= pc_start_iteration and (i - pc_start_iteration) % pc_interval == 0

        if adaptive_propagator:
            prop_tilt_g.zero_()
            prop_tilt_H.zero_()
            prop_tilt_c.zero_()
            prop_thick_sum.zero_()

        if pc_active:
            pos_g.zero_()
            pos_H.zero_()

        if position_clustering:
            block_order = torch.randperm(n_block).tolist()
            posset_blocks = [spatial_blocks[j] for j in block_order]
        else:
            idx = torch.randperm(posset_GPU.shape[0], device=device)
            posset_shuffled = posset_GPU[idx]
            posset_blocks = list(torch.tensor_split(posset_shuffled, n_block, dim=0))
        if probe_orthog_constr:
            proben_GPU = probe_orthogonalization(proben_GPU, n_state, device=device)
        if sorting_probe:
            intensities = proben_GPU.abs().pow(2).sum(dim=(-2, -1))
            intensities_order = torch.argsort(intensities, descending=True)
            proben_GPU = proben_GPU[intensities_order]

        for pos_block in posset_blocks:
            b_size = pos_block.shape[0]
            if b_size == 0:
                continue
            poss_1D = pos_block[:, 4].to(dtype=torch.int32)
            ind_y = (pos_block[:, 2][:, None, None] + y_ind[None, :, None]).to(dtype=torch.int32)
            ind_x = (pos_block[:, 3][:, None, None] + x_ind[None, None, :]).to(dtype=torch.int32)
            indice = ind_y * Ox + ind_x
            Illuminated_patches = object_GPU[:, ind_y, ind_x]
            amplitudes_patches = data_4D_sqrt_GPU[poss_1D]
            amplitudes_patches *= detector_mask_GPU

            if adaptive_propagator:
                theta_x_now = theta_x_map.reshape(-1)[poss_1D.long()]
                theta_y_now = theta_y_map.reshape(-1)[poss_1D.long()]
                propagators_block = build_adaptive_propagator(k_theta_GPU, theta_x_now, theta_y_now, thickness_param, wavelength, dy, dx, device)
            else:
                propagators_block = propagators_GPU.unsqueeze(0).expand(b_size, dy, dx)

            ramp = subpixel_ramp(pos_block[:, 2], pos_block[:, 3], dy, dx, device) if subpixel_position else None
            Prb_ms = [subpixel_shift_probe(proben_GPU, ramp, b_size)]
            for s in range(n_slice):
                psi = Prb_ms[s] * Illuminated_patches[s].unsqueeze(0)
                if s != n_slice - 1:
                    psifft = fft2(psi, dim=(2, 3))
                    psifft *= propagators_block[None, :, :, :]
                    Prb_ms.append(ifft2(psifft, dim=(2, 3)))

            psi_fft = fftshift(fft2(psi, dim=(2, 3)), dim=(2, 3))
            a_psi_fft = psi_fft.abs().pow(2).sum(dim=0).sqrt()
            if fit_tilt or fit_thickness:
                accumulate_adaptive_propagator(
                    prop_tilt_g, prop_tilt_H, prop_tilt_c, prop_thick_sum,
                    theta_x_map, theta_y_map, thickness_param, poss_1D.long(),
                    Prb_ms, Illuminated_patches, psi_fft, amplitudes_patches,
                    a_psi_fft, propagators_block, k_theta_GPU, wavelength, detector_mask_GPU, dy, dx, device,
                    fit_tilt=fit_tilt, fit_thickness=fit_thickness)

            if pc_active:
                accumulate_position(
                    pos_g, pos_H, poss_1D.long(), Prb_ms, Illuminated_patches, psi_fft, amplitudes_patches,
                    a_psi_fft, propagators_block, detector_mask_GPU, dy, dx, device)


            if noise_model == 'amplitude':
                chi_fft_R = amplitudes_patches / (a_psi_fft + e_f) - 1

            elif noise_model == 'anscombe':
                I_m = amplitudes_patches.pow(2)
                I_c = a_psi_fft.pow(2)
                chi_fft_R = (I_m + 0.375).sqrt() / ((I_c + 0.375).sqrt() + e_f) - 1

            elif noise_model == 'poisson':
                I_m = amplitudes_patches.pow(2)
                I_c = a_psi_fft.pow(2)
                chi_fft_R = 0.5 * (I_m / (I_c + e_f) - 1)
                chi_fft_R = torch.minimum(chi_fft_R, amplitudes_patches / (a_psi_fft + e_f))

            elif noise_model == 'gaussian':
                I_m = amplitudes_patches.pow(2)
                I_c = a_psi_fft.pow(2)
                I_ref = I_c.amax(dim=(-2, -1), keepdim=True)
                chi_fft_R = (I_m - I_c) / (2 * I_ref + e_f)

            chi_fft_R *= detector_mask_GPU

            chi_fft = psi_fft * chi_fft_R.unsqueeze(0).to(torch.complex64)
            chi = ifft2(ifftshift(chi_fft, dim=(2, 3)), dim=(2, 3))


            for sr in range(n_slice - 1, -1, -1):
                if sr != n_slice - 1:
                    chi_fft = fft2(chi, dim=(2, 3))
                    chi_fft /= propagators_block[None, :, :, :]
                    chi = ifft2(chi_fft, dim=(2, 3))
                if object_all_states:
                    P_abs2 = Prb_ms[sr].abs().pow(2).sum(dim=0)
                else:
                    P_abs2 = Prb_ms[sr][0].abs().pow(2)
                P_max = P_abs2.amax(dim=(-2, -1), keepdim=True)
                P_norm = alpha_O * P_abs2 + (1 - alpha_O) * P_max
                if object_all_states:
                    dObj_i = (chi * Prb_ms[sr].conj()).sum(dim=0) / P_norm
                else:
                    dObj_i = chi[0] * Prb_ms[sr][0].conj() / P_norm
                dObj = torch.zeros((Oy, Ox), dtype=torch.complex64, device=device)
                dObj = dObj.ravel().index_add_(0, indice.ravel(), dObj_i.ravel()).reshape((Oy, Ox))
                object_GPU[sr] += s_O * dObj / b_size
                gPrb_i = chi * Illuminated_patches[sr].unsqueeze(0).conj()
                if sr == 0:
                    O_abs2 = Illuminated_patches[sr].abs().pow(2)
                    O_max = O_abs2.amax(dim=(-2, -1), keepdim=True)
                    O_norm = alpha_P * O_abs2 + (1 - alpha_P) * O_max
                    dPrb_i = gPrb_i / O_norm.unsqueeze(0)
                    dPrb = subpixel_unshift_sum(dPrb_i, ramp)
                    proben_GPU += s_P * dPrb / b_size
                chi = gPrb_i

            I_m = amplitudes_patches.pow(2)
            I_c = psi_fft.abs().pow(2).sum(dim=0)
            A_c = I_c.sqrt()
            m = detector_mask_GPU

            if noise_model == 'amplitude':
                err_u += ((amplitudes_patches - A_c) * m).pow(2).sum()
                err_d += (amplitudes_patches * m).pow(2).sum()
            elif noise_model == 'anscombe':
                err_u += (((I_m + 0.375).sqrt() - (I_c + 0.375).sqrt()) * m).pow(2).sum()
                err_d += ((I_m + 0.375) * m).sum()
            elif noise_model == 'gaussian':
                err_u += ((I_m - I_c) * m).pow(2).sum()
                err_d += (I_m * m).pow(2).sum()
            elif noise_model == 'poisson':
                Ic = I_c.clamp_min(1e-12)
                dev = 2 * (Ic - I_m + torch.where(I_m > 0, I_m * (I_m.clamp_min(1e-12) / Ic).log(), torch.zeros_like(I_m)))
                err_u += (dev * m).sum()
                err_d += m.sum() * Ic.shape[0] 

        if torch.isnan(err_u / err_d):
            print('')
            print('Interrupted')
            plt.figure(figsize=(6, 4))
            plt.semilogy(range(len(err[:i].cpu())), err[:i].cpu(), color='red')
            plt.xlabel('Iteration')
            plt.ylabel('Error')
            plt.title('Convergence Curve at Interruption', color='red')
            plt.tight_layout()
            plt.show()
            raise ValueError('err is NaN. Try using different values of s_O and s_P')

        err[i] = err_u / err_d
        if pc_active:
            pc_shift[i] = update_scan_positions_gn(posset_blocks, pos_g, pos_H, Oy, Ox, dy, dx, s_position_correction)

        posset_GPU = torch.cat(posset_blocks, dim=0)

        if probe_fit:
            for ns in range(n_state):
                proben_GPU[ns] = fit_probe_with_aberration(proben_GPU[ns], wavelength, basis, apeture_mask, device=device)['probe_fit']
        if probe_support_mask and i > iter_max/4:
            proben_GPU = apply_probe_support_mask(proben_GPU, apeture_mask)
        if l2_fft_lambdaTikhonov > 0:
            object_GPU = l2_fft_constraint(object_GPU, n_slice, l2_fft_lambdaTikhonov)
        if l1_fft_softThreshold > 0:
            object_GPU = l1_fft_constraint(object_GPU, n_slice, l1_fft_softThreshold)
        if l0_fft_hardThreshold > 0:
            object_GPU = l0_fft_constraint(object_GPU, n_slice, l0_fft_hardThreshold)
        if kz_regularization > 0:
            if adaptive_propagator:
                Wz = build_kz_filter(Oy, Ox, n_slice, kz_regularization * slice_thickness / thickness_param.item(), device)
            object_GPU = kz_constraint(object_GPU, Wz)
        if zblur_std > 0:
            if adaptive_propagator:
                z_kernel = build_zblur_kernel(zblur_std, thickness_param.item(), device, object_GPU.dtype)
            object_GPU = zblur_constraint(object_GPU, z_kernel)
        if FFT_phase_offset > 0 and i > iter_max * 2 / 3:
            object_GPU = FFT_phase0_offset(object_GPU, n_slice, FFT_phase_offset)
        if rh_positive_phase:
            object_GPU = rh_constraint(object_GPU, n_slice, rh_hardThreshold=0)
        if POA:
            object_GPU = POA_constraint(object_GPU)

        if adaptive_propagator:
            with torch.no_grad():
                newton_decrement_history[i] = update_adaptive_propagator_gn(
                    theta_x_map, theta_y_map, thickness_param, prop_tilt_g, prop_tilt_H, prop_tilt_c, prop_thick_sum,
                    s_apc_theta=s_apc_theta if fit_tilt else 0, s_apc_thickness=s_apc_thickness if fit_thickness else 0)

                thickness_history[i] = thickness_param
                tilt_std_history[i] = torch.sqrt(theta_x_map.var(unbiased=False) + theta_y_map.var(unbiased=False))

        time_is = time.time() - time_i0
        speed = 1 / max(time_is, 1e-12)
        process = (i + 1) / iter_max * 100
        aa = "*" * int(process / 10)
        bb = "." * (10 - int(process / 10))
        dur = int(time.time() - start_time)
        time_remain = int((iter_max - i - 1) * time_is)
        dur_h = dur // 3600
        dur_m = (dur - dur_h * 3600) // 60
        dur_s = dur - dur_h * 3600 - dur_m * 60
        time_remain_h = time_remain // 3600
        time_remain_m = (time_remain - time_remain_h * 3600) // 60
        time_remain_s = time_remain - time_remain_h * 3600 - time_remain_m * 60
        print('\rePIE progressing: {:^3.0f}%[{}->{}] {:.2f}iter/s ({:0>2}:{:0>2}:{:0>2}<{:0>2}:{:0>2}:{:0>2})'.format(process, aa, bb, speed, dur_h, dur_m, dur_s, time_remain_h, time_remain_m, time_remain_s), end='')

    print('')
    gpu_time = time.time() - start_time
    ePIE_Obj = np.array(object_GPU.cpu())
    ePIE_Prb = np.array(proben_GPU.cpu())
    # Prb_EW = np.array(psi[0][0].cpu())
    Prb_EW = np.array(psi.sum(axis=0).mean(axis=0).cpu())
    ePIE_err = np.array(err.cpu())
    pc_mean_shift = np.array((pc_shift / sy / sx).cpu())
    cbed_comparison = np.stack([amplitudes_patches[0].cpu(), a_psi_fft[0].cpu(), amplitudes_patches[-1].cpu(), a_psi_fft[-1].cpu()])
    posset = np.array(posset_GPU.cpu())

    log_to_file("ePIE progressing: {:^3.0f}%[{}->{}] {:.2f}iter/s ({:0>2}:{:0>2}:{:0>2}<{:0>2}:{:0>2}:{:0>2})".format(process, aa, bb, speed, dur_h, dur_m, dur_s, time_remain_h, time_remain_m, time_remain_s))
    print_and_log(f'ePIE process finished in {gpu_time} s')

    if probe_fit:
        print_and_log('')
        print_and_log('################# probe fitting results ##################')
        probe_fit_results = fit_probe_with_aberration(proben_GPU[0], wavelength, basis, apeture_mask, device=device)
        print_and_log('    Fitted Aberration:')
        for i_aberr, (symbol, name, coeff_m) in enumerate(zip(probe_fit_results['aberration_symbols'], probe_fit_results['aberration_names'], probe_fit_results['coeff_wavefront_m'])):
            print_and_log(f'    {i_aberr:02d}. {symbol:8s} | {name:28s} = {coeff_m.item():+.3e} A')

    app_parameters = None
    if adaptive_propagator:
        app_parameters = {
            'theta_x': theta_x_map.cpu().numpy(),
            'theta_y': theta_y_map.cpu().numpy(),
            'slice_thickness': float(thickness_param.cpu()),
            'thickness_history': thickness_history.cpu().numpy(),
            'tilt_std_history': tilt_std_history.cpu().numpy() * 1000,
            'newton_decrement_history': newton_decrement_history.cpu().numpy(),
            's_apc_theta': s_apc_theta,
            's_apc_thickness': s_apc_thickness,
            'apc_start_iteration': apc_start_iteration,
        }
        if s_apc_thickness > 0:
            print_and_log(f'Optimized Slice Thickness: {app_parameters["slice_thickness"]:.6f} Å')
        if s_apc_theta > 0:
            print_and_log(f'Final Tilt Spatial Std: {app_parameters["tilt_std_history"][-1]:.6f} mrad')
            print_and_log(f'Final Tilt x Spatial Mean: {app_parameters["theta_x"].mean():.6f} mrad')
            print_and_log(f'Final Tilt y Spatial Mean: {app_parameters["theta_y"].mean():.6f} mrad')
        # print_and_log(f'Final Newton Decrement: {app_parameters["newton_decrement_history"][-1]:.3e}')

    return ePIE_Obj, ePIE_Prb, Prb_EW, ePIE_err, pc_mean_shift, posset, cbed_comparison, app_parameters

