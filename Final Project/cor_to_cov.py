""" Correlation to Covariance

    This module converts a correlation matrix to a covariance matrix.

"""
import numpy as np

def convert(correlations: np.ndarray, stdev: np.ndarray) -> np.array:
    """
    Takes the correlation matrix and standard deviations an uses them to produce
    the covariance matrix.

    Args:
        correlations: A numpy array containing the correlations between assets.
        stdev: A numpy array containing the standard deviations of each asset.

    Returns:
        np_covariance: A numpy array of containing the covariance between each
        asset.

    Raises:
        TypeError: If the correlations matrix is not a numpy array.
        TypeError: If the standard deviations matrix is not a numpy array.
        ValueError: If the correlations matrix is not square.
        ValueError: If the length of the correlations and standard matrices are
        not the same.
    """

    covariance = []
    stdev_index = 0
    for row in correlations:
        stdev_index2 = 0
        cov_row = []
        for correlation in row:
            if correlation == 1:
                cov_row.append(round(stdev[stdev_index]**2, 7))
            else:
                calc = correlation * stdev[stdev_index] * stdev[stdev_index2]
                cov_row.append(round(calc, 7))
            stdev_index2 += 1
        covariance.append(cov_row)
        stdev_index += 1
    np_covariance = np.array(covariance)
    return np_covariance

def validate_input(correlations: np.ndarray, stdev: np.ndarray) -> None:
    """
    Validates the correlation and standard deviation parameters.

    Args:
        correlations: A numpy array containing the correlations between assets.
        stdev: A numpy array containing the standard deviations of each asset.

    Returns:
        None

    Raises:
        TypeError: If the correlations matrix is not a numpy array.
        TypeError: If the standard deviations matrix is not a numpy array.
        ValueError: If the correlations matrix is not square.
        ValueError: If the length of the correlations and standard matrices are
        not the same.
    """
    if not isinstance(correlations, np.ndarray):
        raise TypeError('Correlation matrix must be a numpy array')
    if not isinstance(stdev, np.ndarray):
        raise TypeError('Standard deviations matrix must be a numpy array')
    if correlations.shape[0] != correlations.shape[1]:
        raise ValueError('Correlation matrix must be square')
    if correlations.shape[0] != stdev.shape[0]:
        raise ValueError('The correlation and standard deviation matrices must'\
            ' be the same length')
