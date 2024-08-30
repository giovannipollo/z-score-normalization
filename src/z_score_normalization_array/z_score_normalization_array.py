class ZScoreNormalizationArray:
    def __init__(self):
        self.mean = 0
        self.std = 0

    def __compute_mean(self, data: list):
        """
        Compute the mean of the input data.

        Args:
            data: list of numbers

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


    def apply_z_score_normalization(self, data: list) -> list:
        """
        Apply z-score normalization to the input data.

        Args:
            data: list of numbers

        Returns:
            list of numbers after applying z-score normalization
        """
        self.__compute_mean(data=data)
        self.__compute_std(data=data)

        print(self.mean)
        print(self.std)
        normalized_data = []
        for i in range(len(data)):
            normalized_data.append((data[i] - self.mean) / self.std)

        return normalized_data
    
