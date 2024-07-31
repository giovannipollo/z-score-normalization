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
        self.mean = None
        self.std = None
        self.input_buffer = np.zeros(len_data)

    def __compute_mean(self, data: list):
        """
        Compute the mean of the input data.

        Args:
            data: list of numbers

        Returns:
            mean of the input data
        """
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
        self.__compute_mean(data=self.input_buffer)
        self.__compute_std(data=self.input_buffer)

        normalized_data = np.zeros(len(self.input_buffer))

        # Normalize the input data
        for i in range(len(self.input_buffer)):
            normalized_data[i] = (self.input_buffer[i] - self.mean) / self.std
        
        # Update the input buffer
        for i in range(1, len(self.input_buffer)):
            self.input_buffer[i] = self.input_buffer[i - 1]
        self.input_buffer[0] = data

        return normalized_data
    
