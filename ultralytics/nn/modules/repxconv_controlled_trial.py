import torch
from torch import nn

from ultralytics.nn.modules.conv import autopad


class RepXConvCCAB(nn.Module):
    '''RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''

    default_act = nn.SiLU()

    # default silu
    def __init__(self, dim, kernel_size_dense=5, kernel_size_tiny=3,
                 stride=1, padding=None, dilation=1, padding_mode='zeros', act=True, deploy=False):
        super(RepXConvCCAB, self).__init__()
        """ Initialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy

        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size_dense,
                                         stride=stride,
                                         padding=autopad(kernel_size_dense, padding), dilation=dilation, groups=dim,
                                         bias=True,
                                         padding_mode=padding_mode)
        else:

            self.rbr_dense = nn.Conv2d(dim, dim, kernel_size_dense, stride, autopad(kernel_size_dense, padding),
                                       groups=dim,
                                       bias=False)
            # self.rbr_dense_bn = nn.BatchNorm2d(num_features=dim)

            self.rbr_3x3 = nn.Conv2d(dim, dim, kernel_size_tiny, stride, autopad(kernel_size_tiny, padding),
                                     groups=dim,
                                     bias=False)
            # self.rbr_3x3_bn = nn.BatchNorm2d(num_features=dim)
            self.bn_main = nn.BatchNorm2d(num_features=dim)

    def forward(self, inputs):
        # return self.act(self.bn_main(self.rbr_dense_bn(self.rbr_dense(inputs)) + self.rbr_3x3_bn(self.rbr_3x3(inputs))))  # 1
        return self.act(self.bn_main(self.rbr_dense(inputs) + self.rbr_3x3(inputs)))  # 2
        # return self.act(self.rbr_dense_bn(self.rbr_3x3_bn(self.rbr_dense(inputs)) + self.rbr_3x3(inputs))) # 3

    def forward_fuse(self, inputs):
        return self.act(self.rbr_reparam(inputs))  # 1
        # return self.act(self.rbr_reparam(inputs))  # 2
        # return self.act(self.bn_main(self.rbr_reparam(inputs))) # 3

    def get_equivalent_kernel_bias(self):

        # # 1
        # kernel3x3_5x5, bia3x3_5x5 = self._fuse_bn_tensor(self.rbr_3x3.weight, *self._get_bn_params(self.rbr_3x3_bn))
        #
        #
        # return self._fuse_3x3_and_5x5_kernel(self._pad_3x3_to_5x5_tensor(kernel3x3_5x5),
        #                                      bia3x3_5x5,
        #                                      *self._fuse_bn_tensor(self.rbr_dense.weight,
        #                                                            *self._get_bn_params(self.rbr_dense_bn)))

        # 2
        kernel5x5_main = self._pad_3x3_to_5x5_tensor(self.rbr_3x3.weight) + self.rbr_dense.weight

        return self._fuse_bn_tensor(kernel5x5_main, *self._get_bn_params(self.bn_main))

    def _pad_3x3_to_5x5_tensor(self, kernel3x3):
        if kernel3x3 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel3x3, [1, 1, 1, 1])

    def _get_bn_params(self, branch):
        assert isinstance(branch, nn.BatchNorm2d)
        running_mean = branch.running_mean
        running_var = branch.running_var
        gamma = branch.weight
        beta = branch.bias
        eps = branch.eps
        return running_mean, running_var, gamma, beta, eps

    def _fuse_3x3_and_5x5_kernel(self, k3x3, b3x3, k5x5, b5x5):
        return k3x3 + k5x5, b3x3 + b5x5

    def _fuse_bn_tensor(self, kernel, running_mean, running_var, gamma, beta, eps):
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_half(self):
        if not hasattr(self, 'rbr_reparam'):
            self.switch_to_deploy()
        self.kernel_float32 = self.rbr_reparam.weight
        self.rbr_reparam.weight.data = self.rbr_reparam.weight.half()

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            if hasattr(self, 'kernel_float32'):
                self.rbr_reparam.weight.data = self.kernel_float32
                self.__delattr__('kernel_float32')
            return
        kernel, bias = self.get_equivalent_kernel_bias()

        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.in_channels,
                                     out_channels=self.rbr_dense.out_channels,
                                     kernel_size=self.rbr_dense.kernel_size, stride=self.rbr_dense.stride,
                                     padding=self.rbr_dense.padding, dilation=self.rbr_dense.dilation,
                                     groups=self.rbr_dense.groups, bias=True)

        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_3x3')
        # if hasattr(self, 'rbr_3x3_bn'):
        #     self.__delattr__('rbr_3x3_bn')
        #
        # if hasattr(self, 'rbr_dense_bn'):
        #     self.__delattr__('rbr_dense_bn')

        if hasattr(self, 'bn_main'):
            self.__delattr__('bn_main')
        self.deploy = True
    # Represented convolution
    # https://arxiv.org/abs/2101.03697


class RepXConvCBCBA(nn.Module):
    '''RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''

    default_act = nn.SiLU()

    # default silu
    def __init__(self, dim, kernel_size_dense=5, kernel_size_tiny=3,
                 stride=1, padding=None, dilation=1, padding_mode='zeros', act=True, deploy=False):
        super(RepXConvCBCBA, self).__init__()
        """ Initialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy

        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size_dense,
                                         stride=stride,
                                         padding=autopad(kernel_size_dense, padding), dilation=dilation, groups=dim,
                                         bias=True,
                                         padding_mode=padding_mode)
        else:

            self.rbr_dense = nn.Conv2d(dim, dim, kernel_size_dense, stride, autopad(kernel_size_dense, padding),
                                       groups=dim,
                                       bias=False)
            self.rbr_dense_bn = nn.BatchNorm2d(num_features=dim)

            self.rbr_3x3 = nn.Conv2d(dim, dim, kernel_size_tiny, stride, autopad(kernel_size_tiny, padding),
                                     groups=dim,
                                     bias=False)
            self.rbr_3x3_bn = nn.BatchNorm2d(num_features=dim)
            # self.bn_main = nn.BatchNorm2d(num_features=dim)

    def forward(self, inputs):
        # return self.act(self.bn_main(self.rbr_dense_bn(self.rbr_dense(inputs)) + self.rbr_3x3_bn(self.rbr_3x3(inputs))))  # 1
        # return self.act(self.rbr_dense_bn(self.rbr_dense(inputs) + self.rbr_3x3(inputs)))  # 2
        return self.act(self.rbr_dense_bn(self.rbr_dense(inputs)) + self.rbr_3x3_bn(self.rbr_3x3(inputs)))  # 3

    def forward_fuse(self, inputs):
        return self.act(self.rbr_reparam(inputs))  # 1
        # return self.act(self.rbr_reparam(inputs))  # 2
        # return self.act(self.bn_main(self.rbr_reparam(inputs))) # 3

    def get_equivalent_kernel_bias(self):

        # 1
        kernel3x3_5x5, bia3x3_5x5 = self._fuse_bn_tensor( self.rbr_3x3.weight,
                                                         *self._get_bn_params(self.rbr_3x3_bn))

        kernel5x5, bia5x5 = self._fuse_bn_tensor(self.rbr_dense.weight,
                                                 *self._get_bn_params(self.rbr_dense_bn))

        return self._fuse_3x3_and_5x5_kernel(self._pad_3x3_to_5x5_tensor(kernel3x3_5x5),
                                             bia3x3_5x5,
                                             kernel5x5,
                                             bia5x5)

        # # 2
        # kernel5x5_main = self._fuse_3x3_and_5x5_kernel(self._pad_3x3_to_5x5_tensor(self.rbr_3x3.weight),
        #                                                self.rbr_dense.weight)

        # return self._fuse_bn_tensor(kernel5x5_main, *self._get_bn_params(self.bn_main))

    def _pad_3x3_to_5x5_tensor(self, kernel3x3):
        if kernel3x3 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel3x3, [1, 1, 1, 1])

    def _get_bn_params(self, branch):
        assert isinstance(branch, nn.BatchNorm2d)
        running_mean = branch.running_mean
        running_var = branch.running_var
        gamma = branch.weight
        beta = branch.bias
        eps = branch.eps
        return running_mean, running_var, gamma, beta, eps

    def _fuse_3x3_and_5x5_kernel(self, k3x3, b3x3, k5x5, b5x5):
        return k3x3 + k5x5, b3x3 + b5x5

    def _fuse_bn_tensor(self, kernel, running_mean, running_var, gamma, beta, eps):
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_half(self):
        if not hasattr(self, 'rbr_reparam'):
            self.switch_to_deploy()
        self.kernel_float32 = self.rbr_reparam.weight
        self.rbr_reparam.weight.data = self.rbr_reparam.weight.half()

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            if hasattr(self, 'kernel_float32'):
                self.rbr_reparam.weight.data = self.kernel_float32
                self.__delattr__('kernel_float32')
            return
        kernel, bias = self.get_equivalent_kernel_bias()

        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.in_channels,
                                     out_channels=self.rbr_dense.out_channels,
                                     kernel_size=self.rbr_dense.kernel_size, stride=self.rbr_dense.stride,
                                     padding=self.rbr_dense.padding, dilation=self.rbr_dense.dilation,
                                     groups=self.rbr_dense.groups, bias=True)

        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_3x3')
        if hasattr(self, 'rbr_3x3_bn'):
            self.__delattr__('rbr_3x3_bn')

        if hasattr(self, 'rbr_dense_bn'):
            self.__delattr__('rbr_dense_bn')

        # if hasattr(self, 'bn_main'):
        #     self.__delattr__('bn_main')
        self.deploy = True
    # Represented convolution
    # https://arxiv.org/abs/2101.03697
