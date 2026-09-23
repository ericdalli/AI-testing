import numpy as np
def check_differential_privacy(prob_with, prob_without, epsilon):
    """
    Check if the privacy budget is within bounds.
    :param prob_with: Probability of output with individual's
    data
    :param prob_without: Probability of output without
    individual's data
    :param epsilon: Privacy budget
    :return: True if ε-differentially private
    """
    ratio = prob_with / prob_without
    print(f"Ratio is {ratio}")
    return ratio <= np.exp(epsilon)


prob_with_alice = 0.6
prob_without_alice = 0.4
epsilon = 0.4
is_private = check_differential_privacy(
    prob_with_alice,
    prob_without_alice,
    epsilon
)
print(f"Is the algorithm ε-differentially private? {is_private}")

import matplotlib.pyplot as plt

eps = np.linspace(-2, 3, 200)
plt.plot(eps, np.exp(eps), label="np.exp(epsilon)")
plt.plot(eps, 1 + eps, "--", label="1 + epsilon")
plt.xlabel("epsilon"); plt.legend(); plt.grid(True)
plt.show()