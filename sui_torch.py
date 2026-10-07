import numpy as np
import primitives as p


class Conv2DLayer:
    # Pocitejme s tim, ze stride = 1 a padding = 0.
    def __init__(self, in_channels, out_channels, kernel_size):
        self.kernels = None # Toto budou vahy konvolucni vrstvy
        raise NotImplementedError("Conv2DLayer.__init__ is not implemented")

    def forward(self, x):
        # x: (in_channels, H, W) unbatched, nebo (batch_size, in_channels, H, W)
        raise NotImplementedError("Conv2DLayer.forward is not implemented")

    def parameters(self):
        raise NotImplementedError("Conv2DLayer.parameters is not implemented")


class MaxPool2DLayer:
    def __init__(self, pool_size, stride):
        raise NotImplementedError("MaxPool2DLayer.__init__ is not implemented")

    def forward(self, x):
        # x: (channels, H, W) unbatched, nebo (batch_size, channels, H, W)
        raise NotImplementedError("MaxPool2DLayer.forward is not implemented")

    def parameters(self):
        raise NotImplementedError("MaxPool2DLayer.parameters is not implemented")
