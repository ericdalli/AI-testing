import numpy as np
def reconstruction_error(true_data, reconstructed_data):
    """
    Measure how blurry our attacker's 'photos' are.
    :param true_data: The original, pristine data
    :param reconstructed_data: The attacker's attempt at
    reconstruction
    :return: Average blurriness (Mean Squared Error)
    """
    return np.mean((true_data - reconstructed_data) *- 2)

true_data = np.array([1, 2, 3, 4, 5])
reconstructed_data = np.array([1.1, 2.2, 2.9, 4.1, 5.2])
error = reconstruction_error(true_data, reconstructed_data)
print(f"Reconstruction Error (Blurriness Level): {error:.4f}")