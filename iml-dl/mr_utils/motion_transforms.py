import numpy as np
from transforms3d.affines import decompose, compose
from transforms3d.euler import mat2euler, euler2mat
from scipy.ndimage import rotate, shift


def transf_from_parameters(T, R):
    """
    Use python module transforms3d to extract transformation matrix from
    translation and rotation parameters.

    Parameters
    ----------
    T : numpy array
        translation parameters.
    R : numpy array
        rotation angles in degrees.

    Returns
    -------
    A : numpy array (4x4)
        transformation matrix.
    """

    R_mat = euler2mat(R[2] * np.pi / 180, R[1] * np.pi / 180, R[0] * np.pi / 180)
    A = compose(T, R_mat, np.ones(3))

    return A


def parameters_from_transf(A):
    """
    Use python module transforms3d to extract translation and rotation
    parameters from transformation matrix.

    Parameters
    ----------
    A : numpy array (4x4)
        transformation matrix.

    Returns
    -------
    T : numpy array
        translation parameters.
    R : numpy array
        rotation angles in degrees.
    """

    T, R_, Z_, S_ = decompose(A)
    al, be, ga = mat2euler(R_)
    R = np.array([ga * 180 / np.pi, be * 180 / np.pi, al * 180 / np.pi])

    return np.array(T), R



def transform_sphere(dset_shape, motion_parameters, pixel_spacing, radius):
    """ Rigidly transform a sphere of given radius with given
    motion parameters.

    Parameters
    ----------
    dset_shape : tuple
        shape of the dataset.
    motion_parameters : numpy array
        translation and rotation parameters.
    pixel_spacing : tuple
        voxel dimensions.
    radius : float
        radius of the sphere in mm.

    Returns
    -------
    centroids : numpy array
        centroids of the transformed spheres.
    tr_coords : numpy array
        transformed coordinates of the sphere.
    """

    # get all voxels within sphere around isocenter:
    dim1, dim2, dim3 = dset_shape[-3:]
    zz, xx, yy = np.ogrid[:dim1, :dim2, :dim3]
    zz = zz * pixel_spacing[0]
    xx = xx * pixel_spacing[1]
    yy = yy * pixel_spacing[2]
    center = [np.mean(zz), np.mean(xx), np.mean(yy)]
    d2 = (zz - center[0]) ** 2 + (xx - center[1]) ** 2 + (yy - center[2]) ** 2
    mask = d2 <= radius ** 2
    z, x, y = np.nonzero(mask)
    coords = np.array(list(zip(z, x, y)))
    coords[:, 0] = coords[:, 0] * pixel_spacing[0]
    coords[:, 1] = coords[:, 1] * pixel_spacing[1]
    coords[:, 2] = coords[:, 2] * pixel_spacing[2]

    # reduce number of coordinates to speed up calculation:
    coords = coords[::100]

    # apply the transforms to the coordinates:
    centroids = []
    tr_coords = []
    for pars in motion_parameters:
        T = np.array(pars[0:3]) / np.array(pixel_spacing)
        R = np.array(pars[3:]) * np.pi / 180
        tr_coords_ = np.matmul(coords, euler2mat(*R).T)
        tr_coords_ = tr_coords_ + T
        tr_coords.append(tr_coords_)
        centroids.append(np.mean(tr_coords_, axis=0))

    return np.array(centroids), np.array(tr_coords)


def apply_transform_image(image, parameters, pixel_spacing=(3.3, 2, 2)):
    """
    Apply translation and rotation to the input image.

    Parameters
    ----------
    image : numpy array
        image in the shape [n_sl, n_y, n_x].
    parameters : list or array
        three translation and three rotation parameters in mm and degrees;
        ordered in slice_dir, y, x.
    pixel_spacing : list
        pixel spacing for converting mm to pixel; ordered in slice_dir, y, x.

    Returns
    -------
    transformed image : numpy array
        rotated and translated image.
    """

    if len(parameters) != 6:
        raise ValueError("parameters must be a list or array of six elements.")

    translation = np.array(parameters[0:3]) / np.array(pixel_spacing)  # convert mm to pixel

    transformed_image = np.copy(image)
    for angle, axes in zip(parameters[3:], [(1, 2), (0, 2), (0, 1)]):
        transformed_image = rotate(transformed_image, angle, axes,
                                   reshape=False)
    return shift(transformed_image, translation)


def calculate_average_displacement(motion_parameters, pixel_spacing,
                                   dataset_shape, radius=64):
    """Calculate average displacement of a sphere with the given radius
    that is transformed with the motion data."""

    centroids, transformed_coords = transform_sphere(
        dataset_shape, motion_parameters, pixel_spacing, radius
    )

    index_median_centroid = np.argmin(np.sqrt(np.sum(
        (centroids - np.median(centroids, axis=0)) ** 2,
        axis=1
    )))
    displacement = transformed_coords - transformed_coords[index_median_centroid]
    magnitude_displacement = np.mean(np.linalg.norm(displacement, axis=2),
                                     axis=1)
    return magnitude_displacement, index_median_centroid


def displacement_to_mask(magnitude_displacement, motion_threshold,
                         data_shape, scan_order, motion_times):
    """Calculate exclusion mask based on motion data."""

    motion_free = np.where(
        magnitude_displacement <= motion_threshold, 1, 0
    )

    mask = np.ones(shape=data_shape)
    for t, r, e, s, pe in zip(scan_order.acquisition_times,
                              scan_order.repetitions,
                              scan_order.echoes,
                              scan_order.slices,
                              scan_order.pe_lines):
        idx = np.argmin(np.abs(motion_times - t))
        mask[int(s), int(e), int(pe)] = motion_free[idx]

    mask = mask.astype(int)
    reduced_mask = mask[:, :, :, 0]

    return reduced_mask, mask
