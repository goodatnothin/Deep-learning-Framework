import numpy as np


class Initializer:
	def __init__(self):
		self.layer_shapes=None 
		self.index=None

	def set_layer_shapes(self, layer_shapes):
        self.layer_shapes = layer_shapes

    def set_layer_index(self, index):
        self.index = index

    def get_io_shape(self):
        return self.layer_shapes[self.index]

    def get(self):
        return self.get(1)[0]

    def get(self, *shape):
        raise NotImplementedError



class He(Initializer):
    def get(self, *shape):
        io = self.get_io_shape()
        input_neurons = np.prod(io[0])
        return np.random.randn(*shape) * np.sqrt(2 / input_neurons)

class Normal(Initializer):
    def __init__(self, mean=0, std=1):
        super().__init__()
        self.mean = mean
        self.std = std

    def get(self, *shape):
        return np.random.normal(self.mean, self.std, shape)

class Xavier(Initializer):
    def get(self, *shape):
        io = self.get_io_shape()
        input_neurons = np.prod(io[0])
        return np.random.randn(*shape) * np.sqrt(1 / input_neurons)

