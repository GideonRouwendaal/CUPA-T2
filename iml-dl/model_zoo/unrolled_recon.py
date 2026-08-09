import torch
import logging
import merlinth
from merlinth.layers import ComplexConv2d, ComplexConv3d
from merlinth.layers.complex_act import cReLU
from merlinth.layers.module import ComplexModule


def complex2real(z, channel_last=False, separate_echo_dim=False):
    stack_dim = -1 if channel_last else 1
    if separate_echo_dim:
        return torch.stack([torch.real(z), torch.imag(z)],
                           dim=stack_dim)
    else:
        return torch.cat([torch.real(z), torch.imag(z)], stack_dim)


def real2complex(z, channel_last=False, separate_echo_dim=False):
    stack_dim = -1 if channel_last else 1
    (real, imag) = torch.chunk(z, 2, axis=stack_dim)
    if separate_echo_dim:
        return torch.complex(real, imag).squeeze(1)
    else:
        return torch.complex(real, imag)


class ComplexUnrolledNetwork(ComplexModule):
    """ Unrolled network for iterative reconstruction

    Input to the network are zero-filled, coil-combined images, corresponding
    undersampling masks and coil sensitivtiy maps. Output is a reconstructed
    coil-combined image.

    """
    def __init__(self,
                 nr_iterations=10,
                 dc_method="GD",
                 denoiser_method="ComplexCNN",
                 weight_sharing=True,
                 partial_weight_sharing=False,
                 select_echo=False,
                 nr_filters=64,
                 kernel_size=3,
                 nr_layers=5,
                 activation="relu",
                 fc_echo=False,
                 nr_echoes=12,
                 conv_echo=False,
                 echo_kernel_size=3,
                 **kwargs):
        super(ComplexUnrolledNetwork, self).__init__()

        self.nr_iterations = nr_iterations
        self.dc_method = dc_method
        self.T = 1 if weight_sharing else nr_iterations
        input_dim = 12 if select_echo is False else 1
        self.partial_weight_sharing = partial_weight_sharing
        self.conv_echo = conv_echo
        self.fc_echo = fc_echo
        self.separate_echo_dim = (True if (self.conv_echo or self.fc_echo)
                                  else False)
        if self.conv_echo:
            input_dim = 1
            logging.info("[ComplexUnrolledNetwork::init]: Alternating  "
                         "spatial and echo convolution with kernels of size: "
                         "1x{}x{} and {}x1x1, respectively.".format(
                kernel_size, kernel_size, echo_kernel_size)
            )
        elif self.fc_echo:
            input_dim = 1
            logging.info("[ComplexUnrolledNetwork::init]: Alternating  "
                         "spatial convolution (with kernels of size 1x{}x{}) "
                         "and fully connected layers across echo "
                         "dimension.".format(kernel_size, kernel_size))

        # create layers
        if denoiser_method == "Real2chCNN":
            if not partial_weight_sharing:
                self.denoiser = torch.nn.ModuleList([Real2chCNN(
                    dim='2D',
                    input_dim=input_dim * 2,
                    filters=nr_filters,
                    kernel_size=kernel_size,
                    num_layer=nr_layers,
                    activation=activation,
                    use_bias=True,
                    normalization=None,
                    fc_echo=self.fc_echo,
                    nr_echoes=nr_echoes,
                    conv_echo=self.conv_echo,
                    echo_kernel_size=echo_kernel_size,
                    dropout=0,
                    **kwargs
                ) for _ in range(self.T)])
            else:
                # share the first nr_layers-1 layers between iterations
                self.shared_denoiser = torch.nn.ModuleList([Real2chCNN(
                    dim='2D',
                    input_dim=input_dim * 2,
                    output_dim=nr_filters,
                    filters=nr_filters,
                    kernel_size=kernel_size,
                    num_layer=nr_layers - 1,
                    activation=activation,
                    use_bias=True,
                    normalization=None,
                    last_activation=True,
                    fc_echo=self.fc_echo,
                    nr_echoes=nr_echoes,
                    conv_echo=self.conv_echo,
                    echo_kernel_size=echo_kernel_size,
                    dropout=0,
                    **kwargs
                )])
                self.individual_denoiser = torch.nn.ModuleList([Real2chCNN(
                    dim='2D',
                    input_dim=nr_filters,
                    output_dim=input_dim * 2,
                    filters=nr_filters,
                    kernel_size=kernel_size,
                    num_layer=1,
                    activation=activation,
                    use_bias=True,
                    normalization=None,
                    fc_echo=self.fc_echo,
                    nr_echoes=nr_echoes,
                    conv_echo=self.conv_echo,
                    echo_kernel_size=echo_kernel_size,
                    dropout=0,
                    **kwargs
                ) for _ in range(self.T)])
        elif denoiser_method == "UncReal2chCNN":
            if not partial_weight_sharing:
                self.denoiser = torch.nn.ModuleList([Real2chCNN(
                    dim='2D',
                    input_dim=input_dim * 2,
                    filters=nr_filters,
                    kernel_size=kernel_size,
                    num_layer=nr_layers,
                    activation=activation,
                    use_bias=True,
                    normalization=None,
                    fc_echo=self.fc_echo,
                    nr_echoes=nr_echoes,
                    conv_echo=self.conv_echo,
                    echo_kernel_size=echo_kernel_size,
                    dropout=0.3,
                    **kwargs
                ) for _ in range(self.T)])
            else:
                # share the first nr_layers-1 layers between iterations
                self.shared_denoiser = torch.nn.ModuleList([Real2chCNN(
                    dim='2D',
                    input_dim=input_dim * 2,
                    output_dim=nr_filters,
                    filters=nr_filters,
                    kernel_size=kernel_size,
                    num_layer=nr_layers - 1,
                    activation=activation,
                    use_bias=True,
                    normalization=None,
                    last_activation=True,
                    fc_echo=self.fc_echo,
                    nr_echoes=nr_echoes,
                    conv_echo=self.conv_echo,
                    echo_kernel_size=echo_kernel_size,
                    dropout=0.3,
                    **kwargs
                )])
                self.individual_denoiser = torch.nn.ModuleList([Real2chCNN(
                    dim='2D',
                    input_dim=nr_filters,
                    output_dim=input_dim * 2,
                    filters=nr_filters,
                    kernel_size=kernel_size,
                    num_layer=1,
                    activation=activation,
                    use_bias=True,
                    normalization=None,
                    fc_echo=self.fc_echo,
                    nr_echoes=nr_echoes,
                    conv_echo=self.conv_echo,
                    echo_kernel_size=echo_kernel_size,
                    dropout=0.3,
                    **kwargs
                ) for _ in range(self.T)])
        else:
            print("This denoiser method is not implemented yet.")

        A = merlinth.layers.mri.MulticoilForwardOp(center=True,
                                                   channel_dim_defined=False)
        AH = merlinth.layers.mri.MulticoilAdjointOp(center=True,
                                                    channel_dim_defined=False)
        if self.dc_method == "GD":
            self.DC = torch.nn.ModuleList([
                merlinth.layers.data_consistency.DCGD(A, AH, weight_init=1e-4)
                for _ in range(self.T)
            ])
        elif self.dc_method == "PM":
            self.DC = torch.nn.ModuleList([
                merlinth.layers.data_consistency.DCPM(A, AH, weight_init=1e-2)
                for _ in range(self.T)
            ])
        elif self.dc_method == "None":
            self.DC = []
        else:
            print("This DC Method is not implemented.")

        self.apply(self.weight_init)

    def weight_init(self, module):
        if isinstance(module, torch.nn.Conv2d) or isinstance(module,
                                                             torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=1)
            if module.bias is not None:
                module.bias.data.fill_(0)

    def forward(self, img, y, mask, smaps):
        x = img
        for i in range(self.nr_iterations):
            ii = i % self.T
            if not self.partial_weight_sharing:
                x = x + real2complex(
                    self.denoiser[ii](
                        complex2real(x, separate_echo_dim=self.separate_echo_dim)
                    ), separate_echo_dim=self.separate_echo_dim
                )
            else:
                x = x + real2complex(
                    self.individual_denoiser[ii](
                        self.shared_denoiser[0](
                            complex2real(x, separate_echo_dim=self.separate_echo_dim)
                        )
                    ), separate_echo_dim=self.separate_echo_dim
                )
            if self.dc_method != "None":
                x = self.DC[ii]([x, y, mask, smaps])
        return x


class Real2chCNN(ComplexModule):
    """adapted from merlinth.models.cnn.Real2chCNN"""

    def __init__(
            self,
            dim="2D",
            input_dim=1,
            output_dim=False,
            filters=64,
            kernel_size=3,
            num_layer=5,
            activation="relu",
            use_bias=True,
            normalization=None,
            last_activation=False,
            fc_echo=False,
            nr_echoes=12,
            conv_echo=False,
            echo_kernel_size=3,
            dropout=0.3,
            **kwargs,
    ):
        super().__init__()
        # get correct conv operator
        if conv_echo or fc_echo:
            if dim == "3D":
                raise RuntimeError("Convolutions over echoes for dim=3D not "
                                   "implemented.")
            else:
                conv_layer = torch.nn.Conv3d
                kernel_size = (1, kernel_size, kernel_size)
                padding = (0, kernel_size[1] // 2, kernel_size[2] // 2)
                if conv_echo:
                    echo_kernel_size = (echo_kernel_size, 1, 1)
                    echo_padding = (echo_kernel_size[0] // 2, 0, 0)
        else:
            padding = kernel_size // 2
            if dim == "2D":
                conv_layer = torch.nn.Conv2d
            elif dim == "3D":
                conv_layer = torch.nn.Conv3d
            else:
                raise RuntimeError(f"Convolutions for dim={dim} not implemented!")

        if activation == "relu":
            act_layer = torch.nn.ReLU

        if output_dim is False:
            output_dim = input_dim

        self.dropout_rate = dropout
        self.nr_layers = num_layer

        # create layers
        self.ops = []

        # Assuming we have five layers
        dropout_start_idx = (self.nr_layers - 3) // 2
        dropout_end_idx = dropout_start_idx + 3

        if num_layer == 1:
            self.ops.append(
                conv_layer(
                    input_dim,
                    output_dim,
                    kernel_size,
                    padding=padding,
                    bias=use_bias,
                    **kwargs,
                )
            )
            if last_activation:
                if normalization is not None:
                    self.ops.append(normalization())
                self.ops.append(act_layer(inplace=True))

            if conv_echo:
                self.ops.append(
                    conv_layer(
                        in_channels=output_dim,
                        out_channels=output_dim,
                        kernel_size=echo_kernel_size,
                        padding=echo_padding,
                        bias=use_bias
                    )
                )
            elif fc_echo:
                self.ops.append(EchoFullyConnectedLayer(nr_echoes,
                                                        bias=use_bias))
            if conv_echo or fc_echo:
                if last_activation:
                    if normalization is not None:
                        self.ops.append(normalization())
                    self.ops.append(act_layer(inplace=True))

        else:
            # First layer
            self.ops.append(
                conv_layer(
                    input_dim,
                    filters,
                    kernel_size,
                    padding=padding,
                    bias=use_bias,
                    **kwargs,
                )
            )
            if normalization is not None:
                self.ops.append(normalization())
            self.ops.append(act_layer(inplace=True))
            
            if conv_echo:
                self.ops.append(
                    conv_layer(
                        in_channels=filters,
                        out_channels=filters,
                        kernel_size=echo_kernel_size,
                        padding=echo_padding,
                        bias=use_bias
                    )
                )
            elif fc_echo:
                self.ops.append(EchoFullyConnectedLayer(nr_echoes,
                                                        bias=use_bias))
            if conv_echo or fc_echo:
                if normalization is not None:
                    self.ops.append(normalization())

                self.ops.append(act_layer(inplace=True))

            for _ in range(num_layer - 2):
                self.ops.append(
                    conv_layer(
                        filters,
                        filters,
                        kernel_size,
                        padding=padding,
                        bias=use_bias,
                        **kwargs,
                    )
                )
                if normalization is not None:
                    self.ops.append(normalization())
                self.ops.append(act_layer(inplace=True))
                if self.dropout_rate > 0:
                    self.ops.append(torch.nn.Dropout2d(p=self.dropout_rate))

                if conv_echo:
                    self.ops.append(
                        conv_layer(
                            in_channels=filters,
                            out_channels=filters,
                            kernel_size=echo_kernel_size,
                            padding=echo_padding,
                            bias=use_bias
                        )
                    )
                elif fc_echo:
                    self.ops.append(EchoFullyConnectedLayer(nr_echoes,
                                                            bias=use_bias))
                if conv_echo or fc_echo:
                    if normalization is not None:
                        self.ops.append(normalization())
                    self.ops.append(act_layer(inplace=True))
                    if self.dropout_rate > 0:
                        self.ops.append(torch.nn.Dropout2d(p=self.dropout_rate))

            self.ops.append(
                conv_layer(
                    filters,
                    output_dim,
                    kernel_size,
                    bias=False,
                    padding=padding,
                    **kwargs,
                )
            )
            if last_activation:
                if normalization is not None:
                    self.ops.append(normalization())
                self.ops.append(act_layer(inplace=True))
            if conv_echo:
                self.ops.append(
                    conv_layer(
                        in_channels=output_dim,
                        out_channels=output_dim,
                        kernel_size=echo_kernel_size,
                        padding=echo_padding,
                        bias=use_bias
                    )
                )
            elif fc_echo:
                self.ops.append(EchoFullyConnectedLayer(nr_echoes,
                                                        bias=use_bias))
            if conv_echo or fc_echo:
                if last_activation:
                    if normalization is not None:
                        self.ops.append(normalization())
                    self.ops.append(act_layer(inplace=True))

        self.ops = torch.nn.Sequential(*self.ops)
        self.apply(self.weight_initializer)

    def weight_initializer(self, module):
        if isinstance(module, torch.nn.Conv2d) or isinstance(module,
                                                             torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=1)
            if module.bias is not None:
                module.bias.data.fill_(0)

    def forward(self, x):
        x = self.ops(x)
        return x


class EchoFullyConnectedLayer(torch.nn.Module):
    def __init__(self, nr_echoes, bias=True):
        super().__init__()
        self.fc = torch.nn.Linear(nr_echoes, nr_echoes, bias=bias)

    def forward(self, x):
        """Forward pass through the fully connected layer for the echo
        dimension.

        Note: Expected input shape: [batch_size, channel, echo, height, width].
        It reshapes the input to [batch_size*channel*height*width, echo] and
        applies the fully connected layer. The output is then reshaped back to
        the input shape.
        """

        batch_size, channel, echo, height, width = x.shape
        x = x.permute(0, 1, 3, 4, 2)
        x = x.reshape(-1, echo)
        x = self.fc(x)
        x = x.reshape(batch_size, channel, height, width, -1)
        x = x.permute(0, 1, 4, 2, 3)
        return x


class MerlinthComplexCNN(ComplexModule):
    """
    This is a copy of merlinth.models.cnn.ComplexCNN since the module could not
    be loaded due to problems with incomplete optox installation.
    """

    def __init__(
        self,
        dim="2D",
        input_dim=1,
        filters=64,
        kernel_size=3,
        num_layer=5,
        activation="relu",
        use_bias=True,
        normalization=None,
        weight_std=False,
        **kwargs,
    ):
        super().__init__()
        # get correct conv operator
        if dim == "2D":
            conv_layer = ComplexConv2d
        elif dim == "3D":
            conv_layer = ComplexConv3d
        else:
            raise RuntimeError(f"Convlutions for dim={dim} not implemented!")

        if activation == "relu":
            act_layer = cReLU

        padding = kernel_size // 2
        # create layers
        self.ops = []
        self.ops.append(
            conv_layer(
                input_dim,
                filters,
                kernel_size,
                padding=padding,
                bias=use_bias,
                weight_std=weight_std,
                **kwargs,
            )
        )
        if normalization is not None:
            self.ops.append(normalization())

        self.ops.append(act_layer())

        for _ in range(num_layer - 2):
            self.ops.append(
                conv_layer(
                    filters,
                    filters,
                    kernel_size,
                    padding=padding,
                    bias=use_bias,
                    **kwargs,
                )
            )
            if normalization is not None:
                self.ops.append(normalization())
            self.ops.append(act_layer())

        self.ops.append(
            conv_layer(
                filters,
                input_dim,
                kernel_size,
                bias=False,
                padding=padding,
                **kwargs,
            )
        )
        self.ops = torch.nn.Sequential(*self.ops)
        self.apply(self.weight_initializer)

    def weight_initializer(self, module):
        if isinstance(module, torch.nn.Conv2d) or isinstance(module,
                                                             torch.nn.Linear):
            # equivalent to tf.layers.xavier_initalization()
            torch.nn.init.xavier_uniform_(module.weight, gain=1)
            if module.bias is not None:
                module.bias.data.fill_(0)

    def forward(self, x):
        x = self.ops(x)
        return x
