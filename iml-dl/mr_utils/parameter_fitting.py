import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import scipy.ndimage as ndimage


class T2StarFit:
    """ Computes the T2star maps with least_squares_fit.

    Parameters
    ----------
    dim : int, optional
        The number of dimensions of the input image. Default is 3.
    te1 : int, optional
        The first echo time in ms. Default is 5.
    d_te : int, optional
        The echo time difference in ms. Default is 5.
    exclude_last_echoes : int, optional
        The number of last echoes to exclude. Default is 0.
    perform_bgf_correction : bool, optional
        Whether to perform background field correction. Default is False.
    mode : str, optional
        The mode in which the fitting and background field correction is
        supposed to run, i.e. if gradients need to be calculated ("backprop")
        or not ("inference", default).
    """

    def __init__(self, dim=3, te1=5, d_te=5,
                 exclude_last_echoes=0, perform_bgf_correction=False,
                 mode="inference", return_susc_grad=False):
        super(T2StarFit, self).__init__()
        self.te1 = te1
        self.d_te = d_te
        self.dim = dim
        self.exclude_last_echoes = exclude_last_echoes
        self.perform_bgf_correction = perform_bgf_correction
        self.mode = mode
        self.return_susc_grad = return_susc_grad

    def _least_squares_fit(self, data):
        """
        Fit a linear model to the input data.

        Use torch.linalg.lstsq solve AX-B for
            X with shape (n, k) = (2, 1),
        given
            B with shape (m, k) = (12, 1),
            A with shape (m, n) = (12, 2).
        """

        echo_times = torch.arange(self.te1, data.shape[-1] * self.te1 + 1,
                                  self.d_te, dtype=data.dtype,
                                  device=data.device, requires_grad=True)
        if self.dim == 3:
            echo_times = echo_times.unsqueeze(0).unsqueeze(0).repeat(
                data.shape[0], data.shape[1], 1
            )
        if self.dim == 4:
            echo_times = echo_times.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(
                data.shape[0], data.shape[1], data.shape[2], 1
            )

        self.B = data.unsqueeze(-1)
        self.A = torch.cat((echo_times.unsqueeze(-1),
                            torch.ones_like(echo_times).unsqueeze(-1)),
                           dim=-1)
        return torch.linalg.lstsq(self.A, self.B).solution

    @staticmethod
    def _calc_t2star_a0(solution):
        """
        Calculate T2* relaxation times from the least squares fit solution.

        Signal model:   img(t) = A_0 * exp(-t/T2*)
        Linearized:     log(img(t)) = log(A_0) - t/T2*
                        B = X[0] * t +X[1] = AX
        with    X = [- 1/T2*, log(A_0),],
                B = log(img(t)),
                A = [t, 1].
        """

        return - 1 / solution[..., 0, 0], torch.exp(solution[..., 1, 0])

    def __call__(self, img, mask=None):

        if len(img.shape) != self.dim:
            print("This image has different dimensions than expected: ",
                  len(img.shape), self.dim)

        if self.dim == 3:
            img = img.permute(1, 2, 0)
        if self.dim == 4:
            img = img.permute(0, 2, 3, 1)

        if self.exclude_last_echoes > 0:
            img = img[..., :-self.exclude_last_echoes]

        if self.perform_bgf_correction:
            if self.return_susc_grad:
                bgf_correction = BackgroundFieldCorrection(img, mode=self.mode,
                                                           return_susc_grad=True)
                (img, echo_mask, corr_factor, susc_grad_x,
                 susc_grad_y, susc_grad_z) = bgf_correction()
            else:
                bgf_correction = BackgroundFieldCorrection(img, mode=self.mode)
                img, echo_mask, corr_factor = bgf_correction()

            img = torch.log(img + 1e-9)
            img = torch.where(torch.isnan(img), torch.zeros_like(img), img)
            t2stars, a0s = torch.zeros_like(img), torch.zeros_like(img)
            for i in range(4, img.shape[-1]+1):
                t2stars[..., i-1], a0s[..., i-1] = self._calc_t2star_a0(
                    self._least_squares_fit(img[..., :i])
                )

            indices = torch.sum(echo_mask, dim=-1).unsqueeze(-1) - 1
            indices = torch.clamp(indices, min=0, max=img.shape[-1]-1)
            t2star = t2stars.gather(-1, indices.to(torch.int64)).squeeze(-1)
            a0 = a0s.gather(-1, indices.to(torch.int64)).squeeze(-1)

        else:
            img = torch.log(torch.abs(img) + 1e-9)
            t2star, a0 = self._calc_t2star_a0(self._least_squares_fit(img))
            corr_factor, echo_mask = None, None

        if self.mode == "backprop":
            return t2star, a0, corr_factor, echo_mask
        elif self.mode == "regression":
            # t2star = torch.clamp(t2star, min=0, max=200)
            # a0 = torch.clamp(a0, min=0, max=2.5)
            if mask is not None:
                t2star = t2star * mask.to(t2star.device)
                a0 = a0 * mask.to(a0.device)
            if self.return_susc_grad:
                return t2star, a0, susc_grad_x, susc_grad_y, susc_grad_z
            else:
                return t2star, a0
        else:
            # t2star = torch.clamp(t2star, min=0, max=200)
            if mask is not None:
                t2star = t2star * mask.to(t2star.device)
            if self.return_susc_grad:
                return t2star, susc_grad_x, susc_grad_y, susc_grad_z
            else:
                return t2star


class CalcSusceptibilityGradient:
    """ Calculate the susceptibility gradient from the input data.

    Parameters
    ----------
    complex_img : torch.Tensor
        The input image (complex data), dimensions: z, y, x, echoes.
    mask : torch.Tensor, optional
        The corresponding brain mask. Default is None. In this case, a mask
        is estimated from the input image.
    d_te : int, optional
        The echo time difference in ms. Default is 5.
    voxel_sizes : list, optional
        The voxel sizes in mm. Default is [2, 2, 3.3], where a slice gap of
        0.3 mm is included in the z-dimension.
    kernel_size : int, optional
        The size of the smoothing kernel for the susceptibility gradient map.
        Default is 5.
    mode : str, optional
        The mode in which the background field correction is supposed to
        run, i.e. if gradients need to be calculated or not. Default is
        "backprop", which allows for gradients to be calculated. If used in
        inference mode, set to "inference".
    """

    def __init__(self, complex_img, mask=None, d_te=5, voxel_sizes=None,
                 kernel_size=5, mode="backprop"):
        super(CalcSusceptibilityGradient, self).__init__()

        self.complex_img = complex_img
        self.voxel_sizes = voxel_sizes
        self.d_te = d_te
        if voxel_sizes is None:
            self.voxel_sizes = [3.3, 2, 2]
        else:
            self.voxel_sizes = voxel_sizes
        self.kernel_size = kernel_size
        self.mode = mode
        if mask is not None:
            self.mask = mask
            if mask.shape != complex_img.shape[:-1]:
                raise ValueError("[CalcSusceptibilityGradient::init] ERROR: "
                                 "The mask has different dimensions than the "
                                 "input image.")
        else:
            self.mask = self._get_background_mask()

    def __call__(self):
        """ Calculate the susceptibility gradient from the input data.

        The susceptibility gradient is calculated from the B0 difference of
        adjacent pixels (calculated from the phase difference of the first and
        the second echo):

        susc_grad = delta_B0 / voxel_size      [microT/m],
        with B0 = (phase_echo2 - phase_echo1) / (gamma * d_te).

        Using gyro-magnetic ratio of water protons:
            gamma / 2pi = 42.5764 MHz/T,
            gamma = 2pi * 42.5764 MHz/T = 2pi * 42.5764 * 1e-3 1/(ms microT).

        The susceptibility gradient is smoothed  with a normalised kernel
        (kernel size defined in the constructor).
        """

        # instead of subtracting phases to calculate "B0", calculate quotient
        # of complex images (and later take the angle of the result);
        # not only calculate phase difference of first and second echo, but
        # (weighted) average over all echoes:
        quotient = (torch.roll(self.complex_img, shifts=-1, dims=-1)
                    / (self.complex_img + 1e-9))[..., :-1]
        weight = torch.abs(torch.roll(self.complex_img, shifts=-1, dims=-1))[..., :-1]
        weight = weight / (torch.sum(weight, dim=-1, keepdim=True) + 1e-9)
        quotient = torch.sum(quotient * weight, dim=-1)

        # calc gradient maps for z, y amd x direction
        susc_grad_z = self._calc_gradient(quotient, 0)
        susc_grad_y = self._calc_gradient(quotient, 1)
        susc_grad_x = self._calc_gradient(quotient, 2)

        # multiply with constants:
        constant = 2 * torch.pi * 42.5764 * 1e-3
        susc_grad_z *= 1 / constant / self.voxel_sizes[0] / 1e-3 / self.d_te
        susc_grad_y *= 1 / constant / self.voxel_sizes[1] / 1e-3 / self.d_te
        susc_grad_x *= 1 / constant / self.voxel_sizes[2] / 1e-3 / self.d_te

        # threshold the susceptibility gradient maps:
        susc_grad_z = torch.clamp(susc_grad_z, min=0, max=400)
        susc_grad_y = torch.clamp(susc_grad_y, min=0, max=400)
        susc_grad_x = torch.clamp(susc_grad_x, min=0, max=400)

        # combine smoothing and masking:
        susc_grad_z = self._convolve_and_normalize_with_mask(susc_grad_z)
        susc_grad_y = self._convolve_and_normalize_with_mask(susc_grad_y)
        susc_grad_x = self._convolve_and_normalize_with_mask(susc_grad_x)

        return susc_grad_z, susc_grad_y, susc_grad_x

    def _get_background_mask(self):
        """ Estimate a background mask from the input image.

        The mask is calculated by thresholding the magnitude image (first echo)
        with the mean intensity and in inference mode applying
        scipy.ndimage.binary_fill_holes.
        """

        mask = (abs(self.complex_img[..., 0])
                > 0.2 * torch.mean(abs(self.complex_img[..., 0])))

        if self.mode == "inference":
            mask = ndimage.binary_closing(
                ndimage.binary_fill_holes(mask.detach().cpu().numpy()),
                structure=np.ones((1, 10, 10))
            )
            mask = torch.from_numpy(
                mask.astype(np.float32)
            ).to(self.complex_img.device)

        return mask

    @staticmethod
    def _calc_gradient(data, dim):
        """ Calculate the gradient of the input data along the specified
        dimension. """

        # pad data to calculate gradient at the edges
        ind_1 = torch.tensor([1])
        ind_2 = torch.tensor([data.shape[dim]-2])
        element_1 = torch.index_select(data, dim,
                                       torch.tensor([ind_1]).to(data.device))
        element_2 = torch.index_select(data, dim,
                                       torch.tensor([ind_2]).to(data.device))
        data_padded = torch.cat((element_1, data, element_2), dim=dim)

        # roll the data to calculate difference to both neighbours
        data_shift_p = torch.roll(data_padded, shifts=1, dims=dim)
        data_shift_m = torch.roll(data_padded, shifts=-1, dims=dim)

        # calculate the angle of the quotient of the shifted data
        grad_p = data_padded / (data_shift_p + 1e-9)
        grad_m = data_shift_m / (data_padded + 1e-9)

        indices = torch.arange(1, grad_m.shape[dim]-1,
                               dtype=torch.int64).to(grad_m.device)
        return torch.abs(torch.angle(
            0.5 * (torch.index_select(grad_p, dim, indices)
                   + torch.index_select(grad_m, dim, indices))
        ))

    def _convolve_and_normalize_with_mask(self, data):
        """ Convolve the masked input data with a kernel and normalize it. """

        kernel = torch.ones((1, 1, self.kernel_size, self.kernel_size),
                            device=data.device)
        kernel /= kernel.sum()
        padding = (self.kernel_size - 1) // 2
        numerator = F.conv2d((data * self.mask).unsqueeze(1),
                             kernel, padding=padding).squeeze(1)
        denominator = F.conv2d(self.mask.unsqueeze(1).float(),
                               kernel, padding=padding).squeeze(1)
        return abs(numerator / (1e-9 + denominator))


class BackgroundFieldCorrection:
    """ Performs a background field correction assuming a sinc-gaussian RF
    pulse shape.

    References
    ----------
    [1] Baudrexel, Simon et al. (2009): Rapid single-scan T 2*-mapping using
    exponential excitation pulses and image-based correction for linear
    background gradients. In Magnetic Resonance in Medicine 62 (1).
    [2] Hirsch, N. M.; Preibisch, C. (2013): T2* Mapping with Background
    Gradient Correction Using Different Excitation Pulse Shapes. In American
    Journal of Neuroradiology 34 (6).

    Parameters
    ----------
    complex_img : torch.Tensor
        The input image (complex data), dimensions: z, y, x, echoes.
    mask : torch.Tensor, optional
        The corresponding brain mask. Default is None. In this case, a mask
        is estimated from the input image.
    te1 : int, optional
        The first echo time in ms. Default is 5.
    d_te : int, optional
        The echo time difference in ms. Default is 5.
    voxel_sizes : list, optional
        The voxel sizes in mm. Default is [2, 2, 3.3], where a slice gap of
        0.3 mm is included in the z-dimension.
    kernel_size : int, optional
        The size of the smoothing kernel for the susceptibility gradient map.
        Default is 5.
    mode : str, optional
        The mode in which the the background field correction is supposed to
        run, i.e. if gradients need to be calculated or not. Default is
        "backprop", which allows for gradients to be calculated. If used in
        inference mode, set to "inference".
    """

    def __init__(self, complex_img, mask=None, te1=5, d_te=5, voxel_sizes=None,
                 kernel_size=5, mode="backprop", return_susc_grad=False):
        super(BackgroundFieldCorrection, self).__init__()

        self.complex_img = complex_img
        if self.complex_img.shape[0] < 3:
            raise ValueError("[BackgroundFieldCorrection::init] ERROR: The "
                             "input image has less than 3 slices.")
        self.te1 = te1
        self.d_te = d_te
        if voxel_sizes is None:
            self.voxel_sizes = [3.3, 2, 2]
        else:
            self.voxel_sizes = voxel_sizes
        self.kernel_size = kernel_size
        self.mode = mode
        self.calc_susc_grad = CalcSusceptibilityGradient(
            complex_img, mask=mask, d_te=d_te, voxel_sizes=voxel_sizes,
            kernel_size=kernel_size, mode=mode
        )
        self.return_susc_grad = return_susc_grad

    def __call__(self):
        """ Perform the background field correction.

        Returns
        -------
        img_corr : torch.Tensor
            The corrected image.
        echo_mask : torch.Tensor
            The mask indicating which echoes are to be excluded from fitting.
        corr_factor : torch.Tensor
            The correction factor.
        susc_grad_x : torch.Tensor
            The susceptibility gradient in x-direction (optional).
        susc_grad_y : torch.Tensor
            The susceptibility gradient in y-direction (optional).
        susc_grad_z : torch.Tensor
            The susceptibility gradient in z-direction (optional).
        """

        # calc susceptibility gradient
        self.susc_grad_z, self.susc_grad_y, self.susc_grad_x = self.calc_susc_grad()

        # apply through-plane correction:
        img_corr, echo_mask_z, corr_factor = self._apply_through_plane_correction()

        # in-plane correction:
        echo_mask_xy = self._perform_in_plane_correction()

        if not self.return_susc_grad:
            return img_corr, echo_mask_z * echo_mask_xy, corr_factor
        else:
            return (img_corr, echo_mask_z * echo_mask_xy, corr_factor,
                    self.susc_grad_x, self.susc_grad_y, self.susc_grad_z)

    @staticmethod
    def _sinc_gauss_pulse_shape(t, pulse_duration):
        """ Sinc gaussian pulse shape """

        sinc = torch.sinc(1.5 * torch.pi * t / (2.7282 * pulse_duration * 0.5))
        gauss = 2.25 * torch.exp(
            -1 * (t / (2.7282 * pulse_duration * 0.5 * 1.67)) ** 2
        )

        return sinc * gauss

    def _apply_through_plane_correction(self):
        """ Apply through-plane background field correction

        The correction factor is calculated from the time profile of the
        excitation pulse, using the susceptibility gradient in z-direction
        and the slice selection gradient strength g_s:
           S(t_e) = S_0 * A(susc_grad_z / g_s * t_e) * exp(-t_e/T2*)

        The correction factor F(t_e) is consequently:
           F(t_e) = 1 / A(susc_grad_z / g_s * t_e) + eps,

        With the time profile of the Sinc-Gaussian pulse A(t), defined in the
        staticmethod sinc_gauss_pulse_shape, and the following constants:
            g_s = 11289 microT/m
            p = 2.11 ms.

        The correction factor is thresholded at -5 and 5.
        """

        tes = torch.arange(self.te1, self.complex_img.shape[-1] * self.te1 + 1,
                           self.d_te, device=self.complex_img.device)
        arguments = (self.susc_grad_z.unsqueeze(3)
                     * tes.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                     / 11289 )

        correction_factor = 1 / (self._sinc_gauss_pulse_shape(arguments, 2.11)
                                 + 1e-9)

        correction_factor = torch.where(
            (correction_factor < -5) | (correction_factor > 5),
            torch.tensor(0.0, device=correction_factor.device),
            correction_factor
        )

        corr_img = abs(self.complex_img) * correction_factor

        echo_mask = torch.sigmoid(corr_img * 1e6)
        echo_mask = torch.where(echo_mask == 0.5,
                                torch.tensor(0.0, device=echo_mask.device),
                                echo_mask)
        # ensure that mask is zero after the first zero entry for remaining echoes
        echo_mask = echo_mask.cumprod(dim=-1)

        return corr_img, echo_mask, correction_factor

    def _perform_in_plane_correction(self):
        """ Perform in-plane background field correction.

        The maximum echo is calculated from the x- and y-susceptibility
        gradients:

        TE_max = 0.7 * pi / (gamma * abs(susc_grad) * voxel_size),

        gamma = 2pi * 42.5764 MHz/T = 2pi * 42.5764 * 1e-3 1/(ms microT),
        susc_grad: maximum of the x- and y-susceptibility gradient maps.
        """

        max_grad = torch.max(torch.abs(self.susc_grad_x),
                             torch.abs(self.susc_grad_y))
        te_max = 0.7 / 42.5764 / 2e-3 / (max_grad+1e-9) / self.voxel_sizes[1] / 1e-3
        tes = torch.arange(
            self.te1, self.te1*self.complex_img.shape[-1]+1, self.d_te,
            device=self.susc_grad_x.device, dtype=torch.float32,
            requires_grad=True
        ).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        echo_mask = te_max.unsqueeze(-1) - tes
        echo_mask = torch.sigmoid(echo_mask * 1e6)

        return echo_mask
