import os.path
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import numpy as np
import h5py
import logging
import yaml
from typing import NamedTuple
import nibabel as nib
import glob
from medutils.mri import ifft2c, rss, fft2c
from scipy.ndimage import binary_erosion
from mr_utils.motion_transforms import *


# Set NumPy to ignore division errors
np.seterr(divide='ignore', invalid='ignore')


def generate_masks(configuration, k_shape):
    """
    Generate masks based on configuration list.

    Parameters
    ----------
    configuration : list
        First entry defines which mask to generate. Further entries
        define parameters for this type of mask.
    k_shape : tuple
        Shape of k-space to generate mask for.

    Returns
    -------
    mask : np.ndarray
    """

    nc, ne, npe, nfe = k_shape
    keep_central_point = False

    if (configuration[0] == "RandomCenterPixel"
            or configuration[0] == "VarDensBlocksCenterPixel"):
        # set the four central points to 1
        keep_central_point = True

    if configuration[0] == "Random" or configuration[0] == "RandomCenterPixel":
        # configuration: type, fraction of 0 lines, size of fully sampled center
        mask = np.ones(npe)
        total_zeros = int(npe * configuration[1])
        outside_indices = np.concatenate((
            np.arange(0, npe // 2 - configuration[2] // 2),
            np.arange(npe // 2 + configuration[2] // 2, npe)
        ))
        zero_indices = np.random.choice(
            outside_indices, total_zeros, replace=False
        )
        mask[zero_indices] = 0
    elif configuration[0] == "Gaussian":
        # configuration: type, fraction of 0 lines, size of fully sampled center
        mask = np.zeros(npe)
        indices = np.random.normal(
            npe // 2, npe / 3, npe // (1 - configuration[1])
        ).astype(int)
        indices = indices[indices < 92]
        indices = indices[indices >= 0]
        mask[indices] = 1
        mask[npe // 2 - configuration[2] // 2:
             npe // 2 + configuration[2] // 2] = 1
    elif (configuration[0] == "VarDensBlocks"
          or configuration[0] == "VarDensBlocksCenterPixel"):
        # configuration: type, fraction of 0 lines, max width of blocks
        std_scale = 5
        mask = np.ones(npe)
        nr_var0lines = int(npe * configuration[1])
        # variable density via sampling from gaussian:
        count = 0
        while count < nr_var0lines:
            indx = int(np.round(np.random.normal(
                loc=npe // 2, scale=(npe - 1) / std_scale
            )))
            if indx < 46:
                indx += 46
            elif indx >= 46:
                indx -= 46
            width = np.random.randint(0, configuration[2])
            if 0 <= indx < npe - width and mask[indx] == 1:
                for w in range(0, width + 1):
                    if mask[indx+w] == 1:
                        mask[indx + w] = 0
                        count += 1
                        if count == nr_var0lines:
                            break
    elif configuration[0] == "Continuous":
        # configuration: type, exclusion rate
        mask = np.random.beta(a=1-configuration[1], b=configuration[1],
                              size=npe)
    elif configuration[0] == "Cartesian":
        mask = np.zeros(npe)
        acceleration_rate, center_fraction = configuration[1], configuration[2]

        center_size = int(round(npe * center_fraction))
        if center_size % 2 == 1: # Symmetry
            center_size += 1        

        num_sampled_lines_outside_center = max(0, (npe // acceleration_rate) - center_size)

        # Define the region outside the fully sampled center
        outside_center_indices = np.concatenate((
            np.arange(0, npe // 2 - center_size // 2),
            np.arange(npe // 2 + center_size // 2, npe)
        ))

        num_to_sample = min(num_sampled_lines_outside_center, len(outside_center_indices))
        
        sampled_indices_outside_center = np.random.choice(
            outside_center_indices, num_to_sample, replace=False
        )
        mask[sampled_indices_outside_center] = 1

        # Add the fully sampled center
        mask[npe // 2 - center_size // 2:
             npe // 2 + center_size // 2] = 1
    else:
        print("T2star_loader::ERROR: This random mask is not implemented.: "
              "{}".format(configuration))


    # Reshape to (npe, 1) and repeat `nfe` times along axis 1
    final_2d_mask = mask.reshape(npe, 1).repeat(nfe, axis=1)

    # Now, `final_2d_mask` has shape (npe, nfe).
    # Expand this to (nc, ne, npe, nfe) by repeating.
    final_mask = final_2d_mask.reshape(1, 1, npe, nfe) \
                              .repeat(nc, axis=0) \
                              .repeat(ne, axis=1)

    if keep_central_point:
        npe_center = npe // 2
        nfe_center = nfe // 2
        final_mask[:, :, npe_center-1:npe_center+1, nfe_center-1:nfe_center+1] = 1

    return final_mask


# def process_raw_data(hf_file):
#     """Load raw data from h5 file and process to proper complex data."""

#     print("The top level keys in the h5 file are: ", list(hf_file.keys()))
#     print("kspace shape:", hf_file['kspace'].shape)
#     print("sens_maps shape:", hf_file['sens_maps'].shape)



#     raw_data = hf_file['kspace'][:, :, 0, 0, :, 0]
#     sens_maps = hf_file['sens_maps'][:, :, 0, 0, :, 0]

#     if (isinstance(raw_data, np.ndarray) and
#             raw_data.dtype == [('real', '<f4'), ('imag', '<f4')]):
#         return (raw_data.view(np.complex64).astype(np.complex64),
#                 sens_maps.view(np.complex64).astype(np.complex64))
#     else:
#         print('Error in load_raw_mat: Unexpected data format: ',
#               raw_data.dtype)


def process_raw_data(hf_file):
    """Load raw data from h5 file and process to proper complex data."""

    # Directly load data
    raw_data = hf_file['kspace'][()]
    sens_maps = hf_file['sens_maps'][()]

    return raw_data, sens_maps


def get_yshift(hf_file):
    """Get the y_shift to be applied on reconstructed raw images."""

    tmp = hf_file['mrecon_header']['Parameter']['YRange'][()]
    if len(np.unique(tmp[0])) > 1 or len(np.unique(tmp[1])) > 1:
        print('Error: different y shifts for different echoes!')
    return -int((tmp[0, 0] + tmp[1, 0]) / 2)


def pad_sensitivity_maps(sens_maps, kspace_shape):
    """Pad coil sensitivity maps to have same shape as images."""

    pad_width = ((0, 0), (0, 0), (0, 0), (0, 0),
                 (int((kspace_shape[-1] - sens_maps.shape[-1]) / 2),
                  int((kspace_shape[-1] - sens_maps.shape[-1]) / 2))
                 )
    sens_maps = np.pad(sens_maps, pad_width, mode='constant')
    return np.nan_to_num(sens_maps / rss(sens_maps, 2)[:, None])


def remove_readout_oversampling(data, nr_lines):
    """Remove readout oversampling."""

    return data[..., nr_lines:-nr_lines]


def compute_coil_combined_reconstructions(kspace, sens_maps,
                                          y_shift, remove_oversampling=True):
    """Compute coil combined reconstructions."""

    coil_imgs = ifft2c(kspace)
    coil_imgs = np.roll(coil_imgs, shift=y_shift, axis=-2)
    img_cc = np.sum(coil_imgs * np.conj(sens_maps), axis=2)
    if remove_oversampling:
        img_cc = remove_readout_oversampling(img_cc,
                                             int(img_cc.shape[-1] / 4))
    return img_cc


def load_mask_from_nii(filename, binary=True):
    """Load mask from nii file."""

    mask = nib.load(filename).get_fdata()[10:-10][::-1, ::-1, :]
    if binary:
        mask = np.where(mask < 0.5, 0, 1)

    return mask


def replace_string(string, replacements):
    """Replace strings in a string."""

    for key, value in replacements.items():
        string = string.replace(key, value)
    return string


def load_brainmask_nii(filename, no_CSF=False):
    """Load brainmask in correct orientation and threshold it."""

    filename_bm = replace_string(
        filename,
        {"/raw_data/": "/brain_masks/", ".mat": "_bm.nii", ".h5": "_bm.nii", "t2s_gre_fr.hf" : "seg_brain_reg-to-t2s.nii"}
    )

    brain_mask = load_mask_from_nii(filename_bm)
    if no_CSF:
        filename_bm_noCSF = os.readlink(filename_bm).replace("_CSF", "")
        brain_mask_noCSF = load_mask_from_nii(filename_bm_noCSF)
        return brain_mask, brain_mask_noCSF
    else:
        return brain_mask


def load_segmentations_nii(filename, load_aal3=False):
    """
        Load segmentations in correct orientation and threshold them.

        Parameters
        ----------
        filename : str
            Path to folder containing the raw h5 files.
        load_aal3 : bool
            Whether to load the AAL3 segmentation.

        Returns
        ----------
        brain_mask : np.ndarray
            Corresponding brainmask (including CSF).
        gray_matter : np.ndarray
            Corresponding gray matter segmentation.
        white_matter : np.ndarray
            Corresponding white matter segmentation.
        aal3 : np.ndarray
            Corresponding AAL3 segmentation.
    """

    filename_bm = os.readlink(replace_string(
        filename,
        {"/raw_data/": "/brain_masks/", ".mat": "_bm.nii",
         ".h5": "_bm.nii"}
    ))

    filename_gm = glob.glob(os.path.dirname(filename_bm)
                            +"/rc1sub**.nii")[0]
    filename_wm = glob.glob(os.path.dirname(filename_bm)
                            + "/rc2sub**.nii")[0]
    if load_aal3:
        filename_aal3 = os.path.dirname(filename_bm) + "/y_AAL3v1.nii"
        return (load_mask_from_nii(filename_bm),
                load_mask_from_nii(filename_gm),
                load_mask_from_nii(filename_wm),
                load_mask_from_nii(filename_aal3, binary=False))
    else:
        return (load_mask_from_nii(filename_bm),
                load_mask_from_nii(filename_gm),
                load_mask_from_nii(filename_wm))


def apply_undersampling_mask(coil_imgs, sens_maps, mask_configuration=None,
                             mask=None, add_mean_fs=False):
    """Apply undersampling mask to kspace data."""

    kspace = fft2c(coil_imgs)

    if mask is None:
        mask = generate_masks(mask_configuration, kspace.shape)

    kspace_zf = kspace * mask
    coil_imgs_zf = ifft2c(kspace_zf)

    if add_mean_fs:
        npe = coil_imgs.shape[2]
        if npe % 2 == 0:
            condition = mask[:, :, npe//2-1:npe//2+1].sum() == 0
        else:
            condition = mask[:, :, npe//2].sum() == 0
        if condition:
            coil_imgs_zf = (coil_imgs_zf
                            + np.mean(coil_imgs, axis=(2,3), keepdims=True)
                            )

    img_cc_zf = np.sum(coil_imgs_zf * np.conj(sens_maps), axis=1)
    return img_cc_zf, mask


def normalize_images_max(ref_img, images_to_be_normalized):
    """Normalize images by maximum absolute value of reference image."""

    norm = np.nanmax(abs(ref_img)) + 1e-9
    return [img / norm for img in images_to_be_normalized]


def normalize_images_percentile(ref_img, images_to_be_normalized, percentile=99.9):
    """Normalize images by percentile of absolute values of reference image."""

    norm = np.percentile(abs(ref_img), percentile) + 1e-9
    return [img / norm for img in images_to_be_normalized]


def normalize_images(ref_img, images_to_be_normalized,
                     normalization_method="abs_image"):
    """Normalize images according to specified normalization method."""

    if normalization_method == "abs_image":
        return normalize_images_max(ref_img, images_to_be_normalized)
    elif normalization_method == "percentile_image":
        return normalize_images_percentile(ref_img, images_to_be_normalized)
    else:
        return images_to_be_normalized


def equalize_coil_dimensions(data):
    """Equalize coil dimension (axis=1) of input data to 32."""

    data_32 = np.zeros((data.shape[0], 32, data.shape[2], data.shape[3]),
                       dtype=data.dtype)
    data_32[:, :data.shape[1]] = data
    return data_32


def generate_and_apply_masks(img_cc_fs, sens_maps, mask_configuration=None,
                             var_acc_rate="None", add_mean_fs=False):
    """Generate and apply masks to input data according to mask_configuration
    and var_acc_rate."""

    if var_acc_rate != "None":
        if isinstance(var_acc_rate, (int, float)):
            # A single specific acceleration rate is provided
            acc_rate = var_acc_rate
        elif isinstance(var_acc_rate, (list, tuple)):
            if len(var_acc_rate) == 1 or mask_configuration[0] == "Cartesian":
                acc_rate = var_acc_rate[0]
            elif len(var_acc_rate) == 2 and mask_configuration[0] != "Cartesian":
                # A range [min, max] for uniform sampling (original behavior)
                acc_rate = np.random.uniform(var_acc_rate[0], var_acc_rate[1])

        # Update the mask_configuration with the determined acceleration rate
        if mask_configuration is None:
            # Create a default if None, assuming first element is mask type
            # Assuming "Cartesian" type and default 10% center lines if not specified
            mask_configuration = ["Cartesian", acc_rate, 0.1] 
        else:
            # Create a mutable copy if mask_configuration is a tuple/immutable
            if isinstance(mask_configuration, tuple):
                mask_configuration = list(mask_configuration)
            # Ensure it has at least 2 elements for the acc_rate
            if len(mask_configuration) < 2:
                raise ValueError("mask_configuration must have at least 2 elements for mask type and acceleration rate.")
            mask_configuration[1] = acc_rate

    # If var_acc_rate is "None", we must rely on mask_configuration already having an acc_rate
    if mask_configuration is None:
        raise ValueError("mask_configuration cannot be None if var_acc_rate is 'None'.")
    # Ensure mask_configuration is a list for mutability if it was a tuple initially
    if isinstance(mask_configuration, tuple):
        mask_configuration = list(mask_configuration)

    # transform to k-space:
    coil_imgs_batch = img_cc_fs[:, :, None] * sens_maps

    masks_batch, img_cc_zfs_batch = [], []
    for coil_imgs, smap in zip(coil_imgs_batch, sens_maps):
        img_cc_zf, mask = apply_undersampling_mask(
            coil_imgs, smap, mask_configuration=mask_configuration,
            add_mean_fs=add_mean_fs
        )
        masks_batch.append(mask)
        img_cc_zfs_batch.append(img_cc_zf)

    return (torch.as_tensor(np.array(masks_batch), dtype=torch.complex64),
            torch.as_tensor(np.array(img_cc_zfs_batch), dtype=torch.complex64))


def load_hr_qr_moco_data(filename, simulated_data=False):
    """Load HR/QR-MoCo images (exported from Matlab mqBOLD pipeline)."""


    filename_bm = replace_string(
        filename,
        {"/raw_data/": "/brain_masks/", ".mat": "_bm.nii", ".h5": "_bm.nii"}
    )
    filename_hrqrmoco = os.readlink(filename_bm).replace(
        "T1w_coreg/rcBrMsk_CSF.nii", "T2S/RDmoco/T2sw_RDMoCo.mat"
    )
    if simulated_data:
        subject = os.path.basename(filename).split("_nr")[0]
        filename_hrqrmoco = filename_hrqrmoco.replace(
            "bids_from_raw", "bids_from_raw_simulated"
        ).replace(subject.split("-sim")[0], subject).replace(
            "task-still", "task-move"
        )

    with h5py.File(filename_hrqrmoco, "r") as r:
        img_hrqrmoco = r["out"]["Data"][:]
        if isinstance(img_hrqrmoco, np.ndarray) and img_hrqrmoco.dtype == [
            ('real', '<f8'), ('imag', '<f8')]:
            img_hrqrmoco = img_hrqrmoco.view('complex').astype(np.complex64)

    return np.swapaxes(img_hrqrmoco, 2, 3)[:, :, 10:-10][:, :, ::-1, ::-1]


def get_mask_from_motion_data(simulated_data, motion_data, scan_order_path,
                              data_shape):
    """Get mask from motion data for the simulation experiment."""

    motion_parameters = np.array([motion_data[key]
                                  for key in ['t_z', 't_y', 't_x',
                                              'r_z', 'r_y', 'r_x']]).T

    scan_order = type('', (), {})()
    (scan_order.acquisition_times, scan_order.repetitions,
     scan_order.echoes, scan_order.slices,
     scan_order.pe_lines) = np.loadtxt(scan_order_path, unpack=True)

    magnitude_displacement, _ = calculate_average_displacement(
        motion_parameters, tuple(simulated_data["pixel_spacing"]), data_shape,
        radius=64
    )
    mask_simulation, _ = displacement_to_mask(
        magnitude_displacement, simulated_data["motion_threshold"], data_shape,
        scan_order, motion_data["time"]
    )
    # median over echoes for reduced mask:
    mask_simulation = np.median(mask_simulation, axis=1).astype(int)

    return mask_simulation


def load_h5_data(filename, normalize="abs_image", normalize_volume=False):
    """Load data from h5 file."""

    with h5py.File(filename, "r") as f:
        kspace, sens_maps = process_raw_data(f)
        y_shift = get_yshift(f)
        sens_maps = pad_sensitivity_maps(sens_maps, kspace.shape)

    img_cc_fs = compute_coil_combined_reconstructions(
        kspace, sens_maps, y_shift
    )
    if normalize_volume:
        img_cc_fs = normalize_images(img_cc_fs, [img_cc_fs],
                                     normalization_method=normalize)[0]
    sens_maps = remove_readout_oversampling(sens_maps,
                                            int(kspace.shape[-1] / 4))

    return sens_maps, img_cc_fs


def load_h5_data_motion(filename_move, filename_gt, normalize="abs_image",
                        normalize_volume=False, simulated_data=False):
    """Load data from h5 file."""

    # Motion-corrupted data:
    if not simulated_data:
        with h5py.File(filename_move, "r") as f:
            kspace, sens_maps = process_raw_data(f)
            y_shift = get_yshift(f)
            sens_maps = pad_sensitivity_maps(sens_maps, kspace.shape)

        img_cc_fs = compute_coil_combined_reconstructions(
            kspace, sens_maps, y_shift
        )
        sens_maps = remove_readout_oversampling(sens_maps,
                                                int(kspace.shape[-1] / 4))
        mask_simulation = None
    else:
        with h5py.File(filename_move, "r") as f:
            kspace = f["RawData"]["data"][:]
            sens_maps = f["RawData"]["sens_maps"][:]
            motion_data = {}
            for key in f["MotionSimulation"]["MotionData"].keys():
                motion_data[key] = f["MotionSimulation"]["MotionData"][key][:]
            scan_order_path = f["MotionSimulation"]["ScanOrder"][()].decode("utf-8")
            img_data_shape = f["RawData"]["data"][:, :, 0].shape

        img_cc_fs = compute_coil_combined_reconstructions(
            kspace, sens_maps, y_shift=0, remove_oversampling=False
        )
        mask_simulation = get_mask_from_motion_data(
            simulated_data, motion_data, scan_order_path, img_data_shape
        )

    # Motion-free ground truth data:
    with h5py.File(filename_gt, "r") as g:
        kspace_gt, sens_maps_gt = process_raw_data(g)
        y_shift = get_yshift(g)
        sens_maps_gt = pad_sensitivity_maps(sens_maps_gt, kspace_gt.shape)

    img_cc_fs_gt = compute_coil_combined_reconstructions(
        kspace_gt, sens_maps_gt, y_shift
    )

    # HR/QR-MoCo data:
    img_hrqrmoco = load_hr_qr_moco_data(filename_move, simulated_data)

    if normalize_volume:
        img_cc_fs_gt = normalize_images(img_cc_fs_gt, [img_cc_fs_gt],
                                        normalization_method=normalize)[0]
        img_cc_fs = normalize_images(img_cc_fs, [img_cc_fs],
                                    normalization_method=normalize)[0]
        if not simulated_data:
            img_hrqrmoco = normalize_images(img_hrqrmoco, [img_hrqrmoco],
                                            normalization_method=normalize)[0]

    return sens_maps, img_cc_fs, img_cc_fs_gt, img_hrqrmoco, mask_simulation


class RawT2starDataset(Dataset):
    """
    Dataset for loading undersampled, raw T2* data in the reconstruction task.

    Parameters
    ----------
    path : str
        Path to folder containing the relevant h5 files.
    only_bm_slices : bool
        Whether slices with percentage of brainmask voxels < bm_thr*100% should
        be excluded or not.
    bm_thr : float
        Threshold for including / excluding slices based on percentage of
        brainmask voxels.
    normalize : str
        Whether to normalize the data with maximum absolute value of image
        ('abs_image').
    select_echo : bool or int
        Whether to select a specific echo.
    random_mask : bool or list
        How to generate the random mask. First entry defines which mask to
        generate. Further entries define parameters for this type of mask. The
        default is None, corresponding to generating a VarDensBlocks mask with
        parameters 2 and 3.
    overfit_one_sample : bool
        Whether only one sample should be loaded to test overfitting.
    load_whole_set : bool
        Whether to load the entire dataset during initialization, which is
        quite memory-demanding.
    """

    def __init__(self, path, only_bm_slices=False, bm_thr=0.1,
                 normalize="abs_image", normalize_volume=False,
                 select_echo=False, random_mask=None, var_acc_rate="None",
                 overfit_one_sample=False, load_whole_set=True,
                 select_one_scan=False):
        super().__init__()

        self.path = path
        self.raw_samples = []
        self.only_bm_slices = only_bm_slices
        self.bm_thr = bm_thr
        self.normalize = normalize
        self.normalize_volume = normalize_volume
        self.overfit_one_sample = overfit_one_sample
        self.select_echo = select_echo
        if not random_mask:
            self.random_mask = ["VarDensBlocks", 2, 3]
        else:
            self.random_mask = random_mask
        self.var_acc_rate = var_acc_rate
        self.load_whole_set = load_whole_set
        self.select_one_scan = select_one_scan

        files = glob.glob(self.path + "raw_data/**")

        if self.select_one_scan is not False:
            files = [f for f in files if self.select_one_scan in f]
            if len(files) == 0:
                logging.info("[RawT2starDataset::ERROR: no files match the "
                             "selected scan: {}".format(self.select_one_scan))
        for filename in sorted(files):
            slices_ind = self._get_slice_indices(filename)
            new_samples = []

            if not load_whole_set:
                for dataslice in slices_ind:
                    new_samples.append((filename, dataslice))
            else:
                (sens_maps, img_cc_fs) = load_h5_data(
                    filename, normalize=self.normalize,
                    normalize_volume=self.normalize_volume
                )
                _, bm_ = load_brainmask_nii(filename, no_CSF=True)
                for dataslice in slices_ind:
                    img = img_cc_fs[dataslice]
                    if not self.normalize_volume:
                        img = normalize_images(img, [img],
                                               normalization_method=self.normalize)[0]
                    brain_mask = binary_erosion(
                        binary_erosion(bm_[:, :, dataslice]))

                    new_samples.append(T2StarRawDataSample(
                        filename, dataslice, sens_maps[dataslice],
                        img, brain_mask
                    ))
            self.raw_samples += new_samples

    def _get_slice_indices(self, filename):
        """Get indices of slices based on specified conditions."""

        if self.overfit_one_sample:
            slices_ind = [self.overfit_one_sample]
        elif self.only_bm_slices:
            brain_mask = load_brainmask_nii(filename)
            bm_summed = np.sum(brain_mask, axis=(0, 1))
            slices_ind = np.where(bm_summed
                                  / (brain_mask.shape[0] * brain_mask.shape[1])
                                  > self.bm_thr)[0]
        else:
            with h5py.File(filename, "r") as hf:
                slices_ind = np.arange(hf['out']['Data'].shape[0])
        return slices_ind

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, idx):
        """
        Retrieve an item from the dataset at the specified index.

        Parameters
        ----------
        idx : int
            Index of the item to retrieve.

        Returns
        -------
        tuple
            A tuple containing the following elements:
            - img_cc_zf : torch.Tensor
                Zero-filled image.
            - img_cc_fs : torch.Tensor
                Fully sampled image.
            - mask : torch.Tensor
                Mask for undersampling in the Fourier domain, with equalized
                coil dimension.
            - sens_maps : torch.Tensor
                Coil sensitivity maps, with equalized coil dimension.
            - brain_mask : torch.Tensor
                Corresponding brain mask.
            - filename : str
                Path to the file containing the data.
            - dataslice : int
                Index of the slice within the dataset.

        Notes
        -----
        If 'load_whole_set' is False, the method loads data from the file
        specified by the 'idx' index, processes it, and returns relevant
        tensors. If 'load_whole_set' is True, the method directly retrieves
        pre-loaded data from the 'raw_samples' list (which requires more
        memory).
        """

        if not self.load_whole_set:
            filename, dataslice = self.raw_samples[idx]
            sens_maps, img_cc_fs = load_h5_data(
                filename, normalize=self.normalize,
                normalize_volume=self.normalize_volume
            )
            img_cc_fs = img_cc_fs[dataslice]
            sens_maps = sens_maps[dataslice]
            if not self.normalize_volume:
                img_cc_fs = normalize_images(img_cc_fs, [img_cc_fs],
                                       normalization_method=self.normalize)[0]
            _, bm_ = load_brainmask_nii(filename, no_CSF=True)
            brain_mask = binary_erosion(
                binary_erosion(bm_[:, :, dataslice]))
        else:
            (filename, dataslice, sens_maps, img_cc_fs,
             brain_mask) = self.raw_samples[idx]

        if self.select_echo:
            img_cc_fs = np.array([img_cc_fs[self.select_echo]])

        sens_maps = equalize_coil_dimensions(sens_maps)

        return (torch.as_tensor(img_cc_fs, dtype=torch.complex64),
                torch.as_tensor(sens_maps, dtype=torch.complex64),
                torch.as_tensor(brain_mask),
                str(filename),
                dataslice,
                self.random_mask,
                self.var_acc_rate
                )


class RawT2starLoader(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()

        self.batch_size = args.get('batch_size', 8)
        self.only_brainmask_slices = args.get('only_brainmask_slices', False)
        self.bm_thr = args.get('bm_thr', 0.1)
        self.normalize = args.get('normalize', 'abs_image')
        self.normalize_volume = args.get('normalize_volume', False)
        self.overfit_one_sample = args.get('overfit_one_sample', False)
        self.select_echo = args.get('select_echo', False)
        self.random_mask = args.get('random_mask', ["VarDensBlocks", 2, 3])
        self.var_acc_rate = args.get('var_acc_rate', None)
        self.load_whole_set = args.get('load_whole_set', True)
        self.drop_last = False if self.overfit_one_sample else True
        self.num_workers = args.get('num_workers', 1)
        self.data_dir = args.get('data_dir', None)

        assert type(self.data_dir) is dict, ("DefaultDataset::init(): "
                                             "data_dir variable should be "
                                             "a dictionary")

    def train_dataloader(self):
        """Loads a batch of training data."""

        trainset = RawT2starDataset(self.data_dir['train'],
                                    only_bm_slices=self.only_brainmask_slices,
                                    bm_thr=self.bm_thr,
                                    normalize=self.normalize,
                                    normalize_volume=self.normalize_volume,
                                    select_echo=self.select_echo,
                                    random_mask=self.random_mask,
                                    var_acc_rate=self.var_acc_rate,
                                    overfit_one_sample=self.overfit_one_sample,
                                    load_whole_set=self.load_whole_set)
        logging.info(f"Size of the train dataset: {trainset.__len__()}.")

        dataloader = DataLoader(
            trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=self.drop_last,
            pin_memory=True
        )
        return dataloader

    def val_dataloader(self):
        """Loads a batch of validation data."""

        valset = RawT2starDataset(self.data_dir['val'],
                                  only_bm_slices=self.only_brainmask_slices,
                                  bm_thr=self.bm_thr,
                                  normalize=self.normalize,
                                  normalize_volume=self.normalize_volume,
                                  select_echo=self.select_echo,
                                  random_mask=self.random_mask,
                                  var_acc_rate=self.var_acc_rate,
                                  overfit_one_sample=self.overfit_one_sample,
                                  load_whole_set=self.load_whole_set)
        logging.info(f"Size of the validation dataset: {valset.__len__()}.")

        dataloader = DataLoader(
            valset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=self.drop_last,
            pin_memory=True
        )
        return dataloader

    def test_dataloader(self):
        """Loads a batch of testing data.

        Note: the test dataset is not shuffled.
        """
        testset = RawT2starDataset(self.data_dir['test'],
                                   only_bm_slices=self.only_brainmask_slices,
                                   bm_thr=self.bm_thr,
                                   normalize=self.normalize,
                                   normalize_volume=self.normalize_volume,
                                   select_echo=self.select_echo,
                                   random_mask=self.random_mask,
                                   var_acc_rate=self.var_acc_rate,
                                   overfit_one_sample=self.overfit_one_sample,
                                   load_whole_set=self.load_whole_set)
        logging.info(f"Size of the test dataset: {testset.__len__()}.")

        dataloader = DataLoader(
            testset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader


class RawMotionT2starDataset(Dataset):
    """
    Dataset for loading undersampled, raw T2* data (motion-corrupted and
    motion-free) in the motion correction task

    Parameters
    ----------
    path : str
        Path to folder containing the relevant h5 files.
    only_bm_slices : bool
        Whether slices with percentage of brainmask voxels < bm_thr*100% should
        be excluded or not.
    bm_thr : float
        Threshold for including / excluding slices based on percentage of
        brainmask voxels.
    normalize : str
        Whether to normalize the data with maximum absolute value of image
        ('abs_image').
    select_echo : bool or int
        Whether to select a specific echo.
    random_mask : bool or list
        How to generate the random mask. First entry defines which mask to
        generate. Further entries define parameters for this type of mask. The
        default is None, corresponding to generating a VarDensBlocks mask with
        parameters 2 and 3.
    overfit_one_sample : bool
        Whether only one sample should be loaded to test overfitting.
    load_whole_set : bool
        Whether to load the entire dataset during initialization, which is
        quite memory-demanding.
    select_one_scan : str
        Select a specific scan based on the name.
    min_slice_nr : int or dict
        Minimum slice number to be included in the dataset. If a dictionary is
        provided, the key should match the selected scan name.
    simulated_data : bool
        Whether the data is simulated or not.
    """

    def __init__(self, path, only_bm_slices=False, bm_thr=0.1,
                 normalize="abs_image", normalize_volume=False,
                 select_echo=False, random_mask=None, var_acc_rate="None",
                 overfit_one_sample=False, load_whole_set=True,
                 select_one_scan=False, min_slice_nr=None,
                 simulated_data=False, max_nr_slices=False,
                 slice_ranges_file=None, range_slice_nrs=None):
        super().__init__()

        self.path = path
        self.raw_samples = []
        self.only_bm_slices = only_bm_slices
        self.bm_thr = bm_thr
        self.normalize = normalize
        self.normalize_volume = normalize_volume
        self.overfit_one_sample = overfit_one_sample
        self.select_echo = select_echo
        self.random_mask = random_mask
        if not random_mask:
            self.random_mask = ["VarDensBlocks", 2, 3]
        self.var_acc_rate = var_acc_rate
        self.load_whole_set = load_whole_set
        self.select_one_scan = select_one_scan
        self.min_slice_nr = min_slice_nr
        self.simulated_data = simulated_data
        self.max_nr_slices = max_nr_slices
        if slice_ranges_file is not None:
            logging.info(f"Loading slice ranges from file: {slice_ranges_file}")
            with open(slice_ranges_file, "r") as f:
                with open(slice_ranges_file, 'r') as file:
                    self.range_slice_nrs = yaml.safe_load(file)["range_slice_nrs"]
        else:
            self.range_slice_nrs = range_slice_nrs

        if self.range_slice_nrs is not None and (self.min_slice_nr is not None
                                                 or self.max_nr_slices):
            print("ERROR: range_slice_nrs and min_slice_nr/max_nr_slices "
                  "cannot be used together.")
            raise ValueError

        files_move = sorted(glob.glob(self.path+"raw_data/**_move**"))
        files_gt = sorted(glob.glob(self.path+"raw_data/**sg_fV4**"))
        if len(files_move) != len(files_gt):
            logging.info("[RawMotionT2starDataset::ERROR: number of "
                         "motion-free  and motion-corrupted files in {} "
                         "does not match.".format(self.path))

        if self.select_one_scan is not False:
            files_move = [f for f in files_move if self.select_one_scan in f]
            files_gt = [f for f in files_gt if self.select_one_scan in f]

            if len(files_move) == 0:
                logging.info("[RawMotionT2starDataset::ERROR: no files match"
                             " the selected scan: {}".format(
                    self.select_one_scan)
                )
            if len(files_move) > 1:
                logging.info("[RawMotionT2starDataset::ERROR: more than one "
                             "file matches the selected scan: {}".format(
                    self.select_one_scan)
                )

        for filename_move, filename_gt in zip(files_move, files_gt):
            slices_ind = self.get_slice_indices(filename_move)

            new_samples = []

            if not load_whole_set:
                for dataslice in slices_ind:
                    new_samples.append((filename_move, filename_gt, dataslice))

            else:
                (sens_maps, img_cc_fs, img_cc_fs_gt, img_hrqrmoco,
                 mask_simulation) = load_h5_data_motion(
                    filename_move, filename_gt, normalize=self.normalize,
                    normalize_volume=self.normalize_volume,
                    simulated_data=self.simulated_data
                )
                brain_mask_, brain_mask_nocsf_ = load_brainmask_nii(
                    filename_move,
                    no_CSF=True)
                for dataslice in slices_ind:
                    brain_mask = brain_mask_[:, :, dataslice]
                    brain_mask_nocsf = brain_mask_nocsf_[:, :, dataslice]
                    cc_fs_gt = img_cc_fs_gt[dataslice]
                    if not self.simulated_data:
                        hrqrmoco = img_hrqrmoco[:, dataslice]
                    else:
                        hrqrmoco = None
                    cc_fs = img_cc_fs[dataslice]

                    if not self.normalize_volume:
                        cc_fs_gt = normalize_images(
                            cc_fs_gt, [cc_fs_gt],
                            normalization_method=self.normalize
                        )[0]
                        cc_fs = normalize_images(
                            cc_fs, [cc_fs],
                            normalization_method=self.normalize
                        )[0]
                        if not simulated_data:
                            hrqrmoco = normalize_images(
                                hrqrmoco, [hrqrmoco],
                                normalization_method=self.normalize
                            )[0]

                    mask_sim = (mask_simulation[dataslice]
                                if mask_simulation is not None else None)

                    new_samples.append(T2StarRawMotionDataSample(
                        filename_move, dataslice, sens_maps[dataslice],
                        cc_fs, mask_sim, brain_mask,
                        brain_mask_nocsf, cc_fs_gt, hrqrmoco)
                    )
            self.raw_samples += new_samples

    def get_slice_indices(self, filename):
        """Get indices of slices based on specified conditions."""

        if self.overfit_one_sample:
            slices_ind = np.array([self.overfit_one_sample])
        elif self.only_bm_slices:
            brain_mask = load_brainmask_nii(filename)
            bm_summed = np.sum(brain_mask, axis=(0, 1))
            slices_ind = np.where(bm_summed
                                  / (brain_mask.shape[0] * brain_mask.shape[1])
                                  > self.bm_thr)[0]
        else:
            with h5py.File(filename, "r") as hf:
                slices_ind = np.arange(hf['out']['Data'].shape[0])

        if self.range_slice_nrs is not None:
            if self.select_one_scan in self.range_slice_nrs.keys():
                indices = self.range_slice_nrs[self.select_one_scan]
            else:
                indices = self.range_slice_nrs["default"]
            slices_ind = slices_ind[slices_ind >= indices[0]]
            slices_ind = slices_ind[slices_ind <= indices[1]]
        elif self.min_slice_nr is not None:
            if isinstance(self.min_slice_nr, dict):
                if self.select_one_scan in self.min_slice_nr.keys():
                    ind = self.min_slice_nr[self.select_one_scan]
                else:
                    ind = self.min_slice_nr["default"]
                slices_ind = slices_ind[slices_ind >= ind]
            else:
                print("ERROR: expecting a dictionary for min_slice_nr.")
        if self.max_nr_slices is not False:
            slices_ind = slices_ind[:self.max_nr_slices]
        return slices_ind

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, idx):
        """
        Retrieve an item from the dataset at the specified index.

        Parameters
        ----------
        idx : int
            Index of the item to retrieve.

        Returns
        -------
        tuple
            A tuple containing the following elements:
            - img_cc_fs : torch.Tensor
                Fully sampled image (with motion).
            - sens_maps : torch.Tensor
                Coil sensitivity maps, with equalized coil dimension.
            - img_cc_fs_gt : torch.Tensor
                Fully sampled ground truth image (without motion).
            - img_hrqrmoco : torch.Tensor
                HR/QR-MoCo image (from mqBOLD Matlab pipeline).
            - brain_mask : torch.Tensor
                Corresponding brain mask.
            - brain_mask_nocsf : torch.Tensor
                Corresponding brain mask without CSF.
            - filename : str
                Path to the file containing the data.
            - dataslice : int
                Index of the slice within the dataset.
            - random_mask : list
                List defining the configuration of the random mask to be
                generated.
            - simulated_data : bool
                Whether the data is simulated or not.

        Notes
        -----
        If 'load_whole_set' is False, the method loads data from the file
        specified by the 'idx' index, processes it, and returns relevant
        tensors. If 'load_whole_set' is True, the method directly retrieves
        pre-loaded data from the 'raw_samples' list (which requires more
        memory).
        """

        if not self.load_whole_set:
            filename_move, filename_gt, dataslice = self.raw_samples[idx]
            (sens_maps, img_cc_fs, img_cc_fs_gt, img_hrqrmoco,
             mask_simulation) = load_h5_data_motion(
                filename_move, filename_gt, normalize=self.normalize,
                normalize_volume=self.normalize_volume,
                simulated_data=self.simulated_data
            )
            brain_mask, brain_mask_nocsf = load_brainmask_nii(
                filename_move,
                no_CSF=True)
            brain_mask = brain_mask[:, :, dataslice]
            brain_mask_nocsf = brain_mask_nocsf[:, :, dataslice]

            img_cc_fs_gt = img_cc_fs_gt[dataslice]
            if not self.simulated_data:
                img_hrqrmoco = img_hrqrmoco[:, dataslice]
            img_cc_fs = img_cc_fs[dataslice]

            if not self.normalize_volume:
                img_cc_fs_gt = normalize_images(
                    img_cc_fs_gt, [img_cc_fs_gt],
                    normalization_method=self.normalize
                )[0]
                img_cc_fs = normalize_images(
                    img_cc_fs, [img_cc_fs],
                    normalization_method=self.normalize
                )[0]
                if not self.simulated_data:
                    img_hrqrmoco = normalize_images(
                        img_hrqrmoco, [img_hrqrmoco],
                        normalization_method=self.normalize
                    )[0]
        else:
            (filename_move, dataslice, sens_maps, img_cc_fs, mask_simulation,
             brain_mask, brain_mask_nocsf, img_cc_fs_gt,
             img_hrqrmoco) = self.raw_samples[idx]

        if self.select_echo is not False:
            img_cc_fs = np.array([img_cc_fs[self.select_echo]])
            img_cc_fs_gt = np.array([img_cc_fs_gt[self.select_echo]])
            img_hrqrmoco = (np.array([img_hrqrmoco[self.select_echo]])
                            if img_hrqrmoco is not None else None)

        sens_maps = equalize_coil_dimensions(sens_maps)
        simulated_data_bool = True if self.simulated_data else False

        return (
            torch.as_tensor(img_cc_fs, dtype=torch.complex64),
            torch.as_tensor(sens_maps, dtype=torch.complex64),
            torch.as_tensor(img_cc_fs_gt, dtype=torch.complex64),
            (torch.as_tensor(img_hrqrmoco.copy(), dtype=torch.complex64)
             if img_hrqrmoco is not None else False),
            (torch.as_tensor(mask_simulation)
             if mask_simulation is not None else False),
            torch.as_tensor(brain_mask),
            torch.as_tensor(brain_mask_nocsf),
            str(filename_move),
            dataslice,
            (self.random_mask if self.random_mask is not None else False),
            (self.var_acc_rate if self.var_acc_rate is not None else False),
            simulated_data_bool
        )


class RawMotionT2starLoader(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()

        self.batch_size = args.get('batch_size', 8)
        self.only_brainmask_slices = args.get('only_brainmask_slices', False)
        self.bm_thr = args.get('bm_thr', 0.1)
        self.normalize = args.get('normalize', 'abs_image')
        self.normalize_volume = args.get('normalize_volume', False)
        self.overfit_one_sample = args.get('overfit_one_sample', False)
        self.select_echo = args.get('select_echo', False)
        self.random_mask = args.get('random_mask', ["VarDensBlocks", 2, 3])
        self.var_acc_rate = args.get('var_acc_rate', None)
        self.load_whole_set = args.get('load_whole_set', True)
        self.drop_last = args.get('drop_last', False)
        self.num_workers = args.get('num_workers', 1)
        self.data_dir = args.get('data_dir', None)
        self.select_one_scan = args.get('select_one_scan', False)
        self.min_slice_nr = args.get('min_slice_nr', None)
        self.simulated_data = args.get('simulated_data', False)
        self.max_nr_slices = args.get('max_nr_slices', False)
        self.slice_ranges_file = args.get('slice_ranges_file', None)
        self.range_slice_nrs = args.get('range_slice_nrs', None)

        assert type(self.data_dir) is dict, ("DefaultDataset::init(): "
                                             "data_dir variable should be "
                                             "a dictionary")

    def train_dataloader(self):
        """Loads a batch of training data."""

        trainset = RawMotionT2starDataset(
            self.data_dir['train'],
            only_bm_slices=self.only_brainmask_slices,
            bm_thr=self.bm_thr,
            normalize=self.normalize,
            normalize_volume=self.normalize_volume,
            select_echo=self.select_echo,
            random_mask=self.random_mask,
            var_acc_rate=self.var_acc_rate,
            overfit_one_sample=self.overfit_one_sample,
            load_whole_set=self.load_whole_set,
            select_one_scan=self.select_one_scan,
            min_slice_nr=self.min_slice_nr,
            simulated_data=self.simulated_data,
            max_nr_slices=self.max_nr_slices,
            range_slice_nrs = self.range_slice_nrs,
            slice_ranges_file=self.slice_ranges_file
        )
        logging.info(f"Size of the train dataset: {trainset.__len__()}.")

        dataloader = DataLoader(
            trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=self.drop_last,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        """Loads a batch of validation data."""

        if 'val' in self.data_dir:
           valset = RawMotionT2starDataset(
                self.data_dir['val'],
                only_bm_slices=self.only_brainmask_slices,
                bm_thr=self.bm_thr,
                normalize=self.normalize,
                normalize_volume=self.normalize_volume,
                select_echo=self.select_echo,
                random_mask=self.random_mask,
                var_acc_rate=self.var_acc_rate,
                overfit_one_sample=self.overfit_one_sample,
                load_whole_set=self.load_whole_set,
                select_one_scan=self.select_one_scan,
                min_slice_nr=self.min_slice_nr,
                simulated_data=self.simulated_data,
                max_nr_slices=self.max_nr_slices,
                range_slice_nrs=self.range_slice_nrs,
                slice_ranges_file=self.slice_ranges_file
            )
           logging.info(f"Size of the validation dataset: {valset.__len__()}.")

           dataloader = DataLoader(
                valset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                drop_last=self.drop_last,
                pin_memory=True,
            )
           return dataloader
        else:
            logging.info("No validation dataset provided.")
            return None

    def test_dataloader(self):
        """Loads a batch of testing data.

        Note: the test dataset is not shuffled.
        """

        if 'val' or 'test' in self.data_dir:
            testset = RawMotionT2starDataset(
                self.data_dir['test'],
                only_bm_slices=self.only_brainmask_slices,
                bm_thr=self.bm_thr,
                normalize=self.normalize,
                normalize_volume=self.normalize_volume,
                select_echo=self.select_echo,
                random_mask=self.random_mask,
                var_acc_rate=self.var_acc_rate,
                overfit_one_sample=self.overfit_one_sample,
                load_whole_set=self.load_whole_set,
                select_one_scan=self.select_one_scan,
                min_slice_nr=self.min_slice_nr,
                simulated_data=self.simulated_data,
                max_nr_slices=self.max_nr_slices,
                range_slice_nrs=self.range_slice_nrs,
                slice_ranges_file=self.slice_ranges_file
            )
            logging.info(f"Size of the test dataset: {testset.__len__()}.")

            dataloader = DataLoader(
                testset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
            )
            return dataloader
        else:
            logging.info("No test dataset provided.")
            return None


class RawMotionBootstrapSamples:
    def __init__(self, nr_bootstrap_samples, random_mask, var_acc_rate="None"):
        super().__init__()

        self.nr_bootstrap_samples = nr_bootstrap_samples
        self.random_mask = [random_mask[0][0]] + [r[0].numpy()
                                                  for r in random_mask[1:]]
        self.var_acc_rate = var_acc_rate

    def apply_random_masks(self, img_cc_fs, sens_maps):
        """Apply randomly generated masks to fully sampled images."""

        # transform to k-space:
        coil_imgs = img_cc_fs[:, :, None] * sens_maps
        kspace_batch = fft2c(coil_imgs)

        # create bootstrap samples:
        masks_batch, img_cc_zfs_batch = [], []
        for kspace, smap in zip(kspace_batch, sens_maps):
            masks, img_cc_zfs = [], []
            for i in range(self.nr_bootstrap_samples):
                if self.var_acc_rate != "None":
                    # Generate a new random acceleration rate for each batch
                    acc_rate = np.random.uniform(self.var_acc_rate[0],
                                                 self.var_acc_rate[1])
                    self.random_mask[1] = acc_rate

                img_cc_zf, mask = apply_undersampling_mask(
                    kspace, smap, mask_configuration=self.random_mask
                )
                masks.append(mask)
                img_cc_zfs.append(img_cc_zf)

            masks_batch.append(masks)
            img_cc_zfs_batch.append(img_cc_zfs)

        return (torch.as_tensor(np.array(masks_batch), dtype=torch.complex64),
                torch.as_tensor(np.array(img_cc_zfs_batch),
                                dtype=torch.complex64))

    def apply_aggr_masks(self, img_cc_fs, masks, sens_maps):
        """Apply aggregated masks to fully sampled images."""

        # transform to k-space:
        coil_imgs = img_cc_fs[:, :, None] * sens_maps
        kspace_batch = fft2c(coil_imgs)

        img_cc_zfs_batch = []
        for kspace, mask, smap in zip(kspace_batch, masks, sens_maps):
            img_cc_zf, _ = apply_undersampling_mask(kspace, smap, mask=mask)
            img_cc_zfs_batch.append(img_cc_zf)

        return torch.as_tensor(np.array(img_cc_zfs_batch),
                               dtype=torch.complex64)


class T2StarMetaSample(NamedTuple):
    """Generate named tuples consisting of filename and slice index."""

    fname: str
    slice_ind: int


class T2StarRawDataSample(NamedTuple):
    """
    Generate named tuples consisting of filename, slice index and
    loaded raw data.
    """

    fname: str
    slice_ind: int
    sens_maps: np.ndarray
    img_cc_fs: np.ndarray
    brain_mask: np.ndarray


class T2StarRawMotionDataSample(NamedTuple):
    """
    Generate named tuples consisting of filename, slice index and
    loaded raw data.
    """

    fname: str
    slice_ind: int
    sens_maps: np.ndarray
    img_cc_fs: np.ndarray
    mask_simulation: np.ndarray
    brainmask: np.ndarray
    brainmask_noCSF: np.ndarray
    img_cc_fs_gt: np.ndarray
    img_rdmoco: np.ndarray


class T2StarMotionMetaSample(NamedTuple):
    """Generate named tuples consisting of filenames and slice index."""

    fname_move: str
    fname_gt: str
    slice_ind: int


def fft2c(x, shape=None, dim=(-2, -1)):
    """Centered Fourier transform."""

    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=dim),
                                       axes=dim, norm='ortho', s=shape),
                           axes=dim)
