import unittest
from numpy.testing import assert_allclose
import numpy as np
import sui_torch as st
import primitives as p


class TestConv2DLayer(unittest.TestCase):
    def test_forward_output_shape_batched(self):
        batch_size = 2
        in_channels = 3
        out_channels = 4
        image_size = 10
        kernel_size = 3

        conv = st.Conv2DLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        input_tensor = p.Tensor(np.random.randn(batch_size, in_channels, image_size, image_size))

        output = conv.forward(input_tensor)

        expected_length = image_size - kernel_size + 1
        expected_shape = (batch_size, out_channels, expected_length, expected_length)
        self.assertEqual(output.value.shape, expected_shape)

    def test_forward_output_shape_unbatched(self):
        in_channels = 3
        out_channels = 4
        image_size = 10
        kernel_size = 3

        conv = st.Conv2DLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        input_tensor = p.Tensor(np.random.randn(in_channels, image_size, image_size))

        output = conv.forward(input_tensor)

        expected_length = image_size - kernel_size + 1
        expected_shape = (out_channels, expected_length, expected_length)
        self.assertEqual(output.value.shape, expected_shape)

    def test_forward_simple_convolution(self):
        conv = st.Conv2DLayer(in_channels=1, out_channels=1, kernel_size=2)
        conv.kernels[0].value = np.array([[[1.0, 1.0], [1.0, 1.0]]])

        input_tensor = p.Tensor(np.array([[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]]))
        output = conv.forward(input_tensor)

        expected = np.array([[[12.0, 16.0], [24.0, 28.0]]])
        assert_allclose(output.value, expected, rtol=1e-5)

    def test_backward_gradient_flow(self):
        conv = st.Conv2DLayer(in_channels=2, out_channels=3, kernel_size=3)
        input_tensor = p.Tensor(np.random.randn(2, 6, 6))

        output = conv.forward(input_tensor)
        grad = np.ones_like(output.value)
        output.backward(grad)

        self.assertTrue(np.any(input_tensor.grad != 0))
        self.assertTrue(np.any(conv.kernels[0].grad != 0))
        self.assertTrue(np.any(conv.bias.grad != 0))


class TestMaxPool2DLayer(unittest.TestCase):
    def test_forward_output_shape_batched(self):
        batch_size = 2
        channels = 3
        image_size = 10
        pool_size = 2
        stride = 2

        pool = st.MaxPool2DLayer(pool_size=pool_size, stride=stride)
        input_tensor = p.Tensor(np.random.randn(batch_size, channels, image_size, image_size))

        output = pool.forward(input_tensor)

        expected_length = (image_size - pool_size) // stride + 1
        expected_shape = (batch_size, channels, expected_length, expected_length)
        self.assertEqual(output.value.shape, expected_shape)

    def test_forward_output_shape_unbatched(self):
        channels = 3
        image_size = 10
        pool_size = 2
        stride = 2

        pool = st.MaxPool2DLayer(pool_size=pool_size, stride=stride)
        input_tensor = p.Tensor(np.random.randn(channels, image_size, image_size))

        output = pool.forward(input_tensor)

        expected_length = (image_size - pool_size) // stride + 1
        expected_shape = (channels, expected_length, expected_length)
        self.assertEqual(output.value.shape, expected_shape)

    def test_forward_simple_pooling(self):
        pool = st.MaxPool2DLayer(pool_size=2, stride=2)

        input_tensor = p.Tensor(np.array([[
            [1.0, 5.0, 2.0, 8.0],
            [3.0, 7.0, 4.0, 6.0],
            [9.0, 2.0, 1.0, 3.0],
            [5.0, 4.0, 6.0, 0.0],
        ]]))

        output = pool.forward(input_tensor)

        expected = np.array([[[7.0, 8.0], [9.0, 6.0]]])
        assert_allclose(output.value, expected, rtol=1e-5)

    def test_backward_gradient_flow(self):
        pool = st.MaxPool2DLayer(pool_size=2, stride=2)
        input_tensor = p.Tensor(np.random.randn(2, 6, 6))

        output = pool.forward(input_tensor)
        grad = np.ones_like(output.value)
        output.backward(grad)

        self.assertTrue(np.any(input_tensor.grad != 0))


if __name__ == '__main__':
    unittest.main()
