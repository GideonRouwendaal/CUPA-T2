"""
Build the structured-uncertainty inputs (Cholesky factors and low-rank factors)
from the stored MC samples produced by generate_t2star_maps.py.

Expects, under <data_path>:
    gt/                    gt_s0/                raw_mc_predictions/    brain_masks/
Writes, under <data_path>:
    cholesky_means/        cholesky_matrices/
    lowrank_means/rank_R/  lowrank_U_matrices/rank_R/

Example (acceleration rate is configurable):
    python generate_sample_matrices.py --acc_rate 2 --split test --ranks 6 12

--data_root defaults to <CUPA_ROOT>/data/T2_param_data/With_Unc, resolved
relative to this file's location.
"""

import argparse
import numpy as np
import os
from pathlib import Path

from tqdm import tqdm

# ── Path anchors ──────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parents[1]
CUPA_ROOT = REPO_ROOT.parent
DEFAULT_DATA_ROOT = CUPA_ROOT / "data" / "T2_param_data" / "With_Unc" 
# ──────────────────────────────────────────────────────────────────────────────


# Cholesky decomposition function
def compute_cholesky_all_voxels_vectorized(samples_real, samples_imag, brain_mask, gt_t2, gt_s0):
    """
    samples_real, samples_imag: shape (n_samples=100, n_TEs=12, height, width)
    brain_mask: shape (height, width)
    
    Returns:
    - mean_vectors: (height, width, 24) - concatenated real+imag means
    - cholesky_matrices: (height, width, 24, 24) - Cholesky decomposition per voxel
    """
    n_samples, n_TEs, height, width = samples_real.shape
    
    n_TEs = n_TEs * 2  # 24 features (real + imag)

    real_perm = samples_real.transpose(2, 3, 0, 1)  # (height, width, n_samples, 12)
    imag_perm = samples_imag.transpose(2, 3, 0, 1)  # (height, width, n_samples, 12)

    data = np.concatenate([real_perm, imag_perm], axis=3) # (height, width, n_samples, 24)

    # Initialize outputs
    mean_vectors = np.zeros((height, width, n_TEs))
    cholesky_matrices = np.zeros((height, width, n_TEs, n_TEs))

    # 2. Masking: Extract only brain voxels to a flat array.
    # A voxel counts only where the brain mask and both ground-truth maps are positive.
    valid_mask_indices = np.where((brain_mask > 0) & (gt_t2 > 0) & (gt_s0 > 0))
    
    # Extract data: (N_voxels, n_samples, 24)
    voxel_data = data[valid_mask_indices] 
    
    if voxel_data.shape[0] == 0:
        return mean_vectors, cholesky_matrices


    # 3. Compute Means (Vectorized)
    # Shape: (N_voxels, 24)
    means = np.mean(voxel_data, axis=1)
    
    # 4. Compute Covariance (Vectorized via Matrix Multiplication)
    # Center the data: X - mean
    # Shape: (N_voxels, n_samples, 24)
    centered_data = voxel_data - means[:, np.newaxis, :]
    
    # Covariance = (X.T @ X) / (N - 1)
    # We use matmul on the last two dimensions. 
    # Transpose the stack to (N_voxels, 24, n_samples) for the first operand
    # Result shape: (N_voxels, 24, 24)
    covariances = np.matmul(centered_data.transpose(0, 2, 1), centered_data)
    covariances /= (n_samples - 1)
    
    # 5. Regularization (Vectorized)
    # Add epsilon to the diagonal of every matrix to ensure positive definiteness
    eps = 1e-10
    eye_reg = np.eye(n_TEs) * eps
    covariances += eye_reg[np.newaxis, :, :]


    # 6. Compute Cholesky (Vectorized)
    # np.linalg.cholesky broadcasts over the first dimension!
    try:
        # This runs in C-speed for the whole batch
        L_stack = np.linalg.cholesky(covariances)
    except np.linalg.LinAlgError:
        print("Warning: Batched Cholesky failed (some voxels non-PD). Falling back to loop for robust handling.")
        # Fallback: Iterate only if the batch operation fails
        L_stack = np.zeros_like(covariances)
        for k in range(len(covariances)):
            try:
                L_stack[k] = np.linalg.cholesky(covariances[k])
            except np.linalg.LinAlgError:
                # Eigenvalue fallback (Note: Result is NOT lower triangular)
                eigvals, eigvecs = np.linalg.eigh(covariances[k])
                eigvals = np.maximum(eigvals, 1e-10)
                # If you specifically need lower triangular, you'd need QR decomp here,
                # otherwise this matrix square root is valid for sampling.
                L_stack[k] = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    # 7. Map back to original image shape
    mean_vectors[valid_mask_indices] = means
    cholesky_matrices[valid_mask_indices] = L_stack
    
    return mean_vectors, cholesky_matrices


# Low rank matrices
def compute_lowrank_decomposition_all_voxels(samples_real, samples_imag, brain_mask, gt_t2, gt_s0, rank):
    """
    Vectorized computation of low-rank approximation.
    
    Parameters:
    -----------
    samples_real, samples_imag: (n_samples, n_TEs, height, width)
    brain_mask: (height, width)
    rank: int - desired rank
    
    Returns:
    --------
    mean_vectors: (height, width, 24)
    U_matrices: (height, width, 24, rank)
    """
    n_samples, n_TEs, height, width = samples_real.shape
    n_features = n_TEs * 2
    
    # 1. Prepare Data: (Height, Width, Samples, Features)
    real_perm = samples_real.transpose(2, 3, 0, 1)
    imag_perm = samples_imag.transpose(2, 3, 0, 1)
    data = np.concatenate([real_perm, imag_perm], axis=3)
    
    # 2. Masking
    mask_indices = np.where((brain_mask > 0) & (gt_t2 > 0) & (gt_s0 > 0))
    voxel_data = data[mask_indices] # Shape: (N_voxels, n_samples, 24)
    
    n_voxels = voxel_data.shape[0]
    if n_voxels == 0:
        return np.zeros((height, width, n_features)), np.zeros((height, width, n_features, rank))

    print(f"Vectorized rank-{rank} decomp for {n_voxels} voxels...")

    # 3. Compute Means
    means = np.mean(voxel_data, axis=1) # (N_voxels, 24)
    
    # 4. Compute Covariance (Batched Matrix Multiplication)
    # Formula: (X - mu).T @ (X - mu) / (N - 1)
    centered_data = voxel_data - means[:, np.newaxis, :]
    
    # Transpose centered_data for the first term of matmul: (N, 24, Samples)
    # Result: (N, 24, 24)
    covariances = np.matmul(centered_data.transpose(0, 2, 1), centered_data)
    covariances /= (n_samples - 1)
    
    # 5. Batched Eigendecomposition
    # np.linalg.eigh is for symmetric matrices (faster/more stable than eig)
    # It returns eigenvalues in ASCENDING order.
    # eigvals: (N, 24), eigvecs: (N, 24, 24)
    eigvals, eigvecs = np.linalg.eigh(covariances)
    
    # 6. Sort and Truncate (Vectorized)
    # Since eigh is ascending, the largest eigenvalues are at the end.
    # We take the last 'rank' elements.
    
    # Slice: Take last 'rank' columns
    top_eigvals = eigvals[:, -rank:]         # Shape: (N, rank) (smallest -> largest)
    top_eigvecs = eigvecs[:, :, -rank:]      # Shape: (N, 24, rank)
    
    # Reverse to make them descending (Largest -> Smallest), matching standard PCA conventions
    top_eigvals = np.flip(top_eigvals, axis=1)
    top_eigvecs = np.flip(top_eigvecs, axis=2)
    
    # 7. Compute U_matrices (Scaled Eigenvectors)
    # Logic: U_scaled = eigenvectors @ diag(sqrt(eigenvalues))
    # We enforce non-negative eigenvalues just like the loop version
    sqrt_eigvals = np.sqrt(np.maximum(top_eigvals, 1e-10))
    
    # Broadcasting: Multiply (N, 24, rank) by (N, 1, rank)
    # This is equivalent to matrix multiplication with a diagonal matrix
    U_computed = top_eigvecs * sqrt_eigvals[:, np.newaxis, :]
    
    # 8. Map back to original shape
    mean_vectors = np.zeros((height, width, n_features))
    U_matrices = np.zeros((height, width, n_features, rank))
    
    mean_vectors[mask_indices] = means
    U_matrices[mask_indices] = U_computed
    
    return mean_vectors, U_matrices


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", type=str, default=str(DEFAULT_DATA_ROOT),
                   help="Experiment root holding the acc_rate_<R>/<split>/ folders.")
    p.add_argument("--acc_rate", type=str, default="2",
                   help="Acceleration rate R (selects acc_rate_<R>/ under --data_root).")
    p.add_argument("--split", type=str, default="test",
                   choices=["train", "val", "test"],
                   help="Which split to process.")
    p.add_argument("--data_path", type=str, default=None,
                   help="Full override for the split folder; bypasses --data_root/--acc_rate/--split.")
    p.add_argument("--ranks", type=int, nargs="+", default=[6, 12],
                   help="Low-rank factorisation ranks to generate (paper used 12 at R=2, 6 at R=3 and R=4).")
    p.add_argument("--skip_lowrank", action="store_true",
                   help="Only build the Cholesky factors.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.data_path is not None:
        data_path = args.data_path
    else:
        data_path = os.path.join(args.data_root, f"acc_rate_{args.acc_rate}", args.split)
    data_path = data_path.rstrip("/") + "/"

    gt_t2_path = data_path + "gt/"
    gt_s0_path = data_path + "gt_s0/"
    raw_mc_path = data_path + "raw_mc_predictions/"
    brain_mask_path = data_path + "brain_masks/"

    for name, path in [("gt", gt_t2_path), ("gt_s0", gt_s0_path),
                       ("raw_mc_predictions", raw_mc_path), ("brain_masks", brain_mask_path)]:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Missing required input folder '{name}': {path}")

    subjects = sorted(os.listdir(gt_t2_path))
    ranks = [] if args.skip_lowrank else args.ranks
    print(f"Data path: {data_path}\nSubjects: {len(subjects)}   ranks: {ranks}")

    for subject in tqdm(subjects, desc="subjects"):
        slices = sorted(os.listdir(os.path.join(gt_t2_path, subject)))
        for slice_num in slices:
            gt_t2 = np.load(os.path.join(gt_t2_path, subject, slice_num))
            gt_s0 = np.load(os.path.join(gt_s0_path, subject, slice_num))
            raw_mc_preds = np.load(os.path.join(raw_mc_path, subject, slice_num))
            brain_mask = np.load(os.path.join(brain_mask_path, subject, slice_num))
            samples_real, samples_imag = raw_mc_preds.real, raw_mc_preds.imag

            cholesky_mean, cholesky_matrices = compute_cholesky_all_voxels_vectorized(
                samples_real, samples_imag, brain_mask, gt_t2, gt_s0)
            output_dir = f"{data_path}cholesky_means/{subject}/"
            os.makedirs(output_dir, exist_ok=True)
            np.save(f"{output_dir}{slice_num}", cholesky_mean)
            output_dir = f"{data_path}cholesky_matrices/{subject}/"
            os.makedirs(output_dir, exist_ok=True)
            np.save(f"{output_dir}{slice_num}", cholesky_matrices)

            for rank in ranks:
                mean_vecs, U_mats = compute_lowrank_decomposition_all_voxels(
                    samples_real, samples_imag, brain_mask, gt_t2, gt_s0, rank)
                output_dir = f"{data_path}lowrank_means/rank_{rank}/{subject}/"
                os.makedirs(output_dir, exist_ok=True)
                np.save(f"{output_dir}{slice_num}", mean_vecs)
                output_dir = f"{data_path}lowrank_U_matrices/rank_{rank}/{subject}/"
                os.makedirs(output_dir, exist_ok=True)
                np.save(f"{output_dir}{slice_num}", U_mats)


if __name__ == "__main__":
    main()