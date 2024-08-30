import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from z_score_normalization_array.z_score_normalization_array import ZScoreNormalizationArray
from z_score_normalization_single_sample.z_score_normalization_single_sample import ZScoreNormalizationSingleSample
import random
from scipy import stats
import numpy as np

def test_apply_z_score_normalization_array():
    z_score_normalization_array = ZScoreNormalizationArray()

    # Seed the random number generator
    random.seed(0)

    # Generate some random data
    data = [random.randint(0, 100) for i in range(10)]

    # Apply z-score normalization
    normalized_data = z_score_normalization_array.apply_z_score_normalization(data=data)

    # Compare the results with scipy
    normalized_data_scipy = stats.zscore(data)

    # Compute the mse
    mse = np.mean((np.array(normalized_data) - np.array(normalized_data_scipy)) ** 2)

    # Check if the mse is less than 1e-10
    assert mse < 1e-10

def test_apply_z_score_normalization_single_sample():
    z_score_normalization_single_sample = ZScoreNormalizationSingleSample(len_data=10)

    # Seed the random number generator
    random.seed(0)
    # Generate some random data
    data = [random.randint(0, 100) for i in range(10)]

    print(data)
    # Apply z-score normalization
    for i in range(len(data)):
        normalized_data = z_score_normalization_single_sample.apply_z_score_normalization(data=data[i])

    # Compare the results with scipy
    normalized_data_scipy = stats.zscore(data)

    # Compute the mse
    mse = np.mean((np.array(normalized_data) - np.array(normalized_data_scipy)) ** 2)

    # Check if the mse is less than 1e-10
    assert mse < 1e-10

if __name__ == '__main__':
    pytest.main()
    # test_apply_z_score_normalization_array()
    # test_apply_z_score_normalization_single_sample()