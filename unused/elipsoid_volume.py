import numpy as np
from scipy.stats import chi2
from scipy.special import gamma


def elipsoid_volume(cov_matrix, confidence_level):
    """
    Calculates the volume of a confidence ellipsoid for a given covariance matrix.

    Parameters:
    cov_matrix (numpy.ndarray): The covariance matrix (n x n).
    confidence_level (float): The desired confidence level (e.g., 0.95).

    Returns:
    float: The volume of the ellipsoid.
    """
    n = cov_matrix.shape[0]  # Number of dimensions

    # Get the eigenvalues of the covariance matrix
    eigenvalues, _ = np.linalg.eigh(cov_matrix)

    # Chi-square critical value for the desired confidence level
    chi2_val = chi2.ppf(confidence_level, df=n)

    # Lengths of the semi-axes
    semi_axes = np.sqrt(eigenvalues * chi2_val)

    # Scaling factor for the volume in n dimensions
    volume_factor = (np.pi ** (n / 2)) / gamma(n / 2 + 1)

    # Calculate the volume of the ellipsoid
    volume = volume_factor * np.prod(semi_axes)

    return volume


# Example usage:
cov_matrix = np.array([[4, 2, 0],
                       [2, 3, 1],
                       [0, 1, 2]])

confidence_level = 0.95
volume = elipsoid_volume(cov_matrix, confidence_level)
print(f"Volume of the confidence ellipsoid: {volume}")
