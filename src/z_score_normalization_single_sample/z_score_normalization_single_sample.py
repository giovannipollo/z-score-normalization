import numpy as np


class ZScoreNormalizationSingleSample:
    def __init__(self, len_data: int):
        """
        Initialize the ZScoreNormalizationSingleSample class.

        Args:
            len_data: length of the input data

        Returns:
            None
        """
        self.mean = 0
        self.std = 0
        self.len_data = len_data
        self.input_buffer = np.zeros(len_data)

    def __compute_mean(self, data: float):
        """
        Compute the mean of the input data.

        Args:
            data (float): single number

        Returns:
            mean of the input data
        """
        self.mean = 0
        for i in range(len(data)):
            self.mean += data[i]
        self.mean /= len(data)

    def __compute_std(self, data: list):
        """
        Compute the standard deviation of the input data.

        Args:
            data: list of numbers

        Returns:
            standard deviation of the input data
        """
        self.std = 0
        for i in range(len(data)):
            self.std += (data[i] - self.mean) ** 2
        self.std = (self.std / len(data)) ** 0.5

    def apply_z_score_normalization(self, data: float) -> list:
        """
        Apply z-score normalization to the input data.

        Args:
            data: single number

        Returns:
            list of numbers after applying z-score normalization
        """
        # Update the input buffer
        for i in range(1, len(self.input_buffer)):
            self.input_buffer[self.len_data - i] = self.input_buffer[
                self.len_data - 1 - i
            ]
        self.input_buffer[0] = data

        self.__compute_mean(data=self.input_buffer)
        self.__compute_std(data=self.input_buffer)

        normalized_data = np.zeros(len(self.input_buffer))

        for i in range(len(self.input_buffer)):
            normalized_data[i] = (
                self.input_buffer[len(self.input_buffer) - 1 - i] - self.mean
            ) / self.std

        return normalized_data
