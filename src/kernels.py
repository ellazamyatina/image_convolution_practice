import numpy as np


class Kernel:
    def __init__(self, matrix):
        if isinstance(matrix, list):
            matrix = np.array(matrix, dtype=np.float32)
        self.matrix = matrix
        self.shape = matrix.shape

    def __call__(self):
        return self.matrix

    def __repr__(self):
        return f"Kernel{self.shape}"

    @classmethod
    def blur(cls):
        return cls(
            [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]]
        )

    @classmethod
    def sharpen(cls):
        return cls([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    @classmethod
    def gaussian(cls):
        return cls(
            [
                [1 / 16, 2 / 16, 1 / 16],
                [2 / 16, 4 / 16, 2 / 16],
                [1 / 16, 2 / 16, 1 / 16],
            ]
        )
