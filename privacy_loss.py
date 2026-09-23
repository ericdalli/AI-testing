import numpy as np
def privacy_loss(prob_with, prob_without):
    """
    Calculate the privacy loss.
    :param prob_with: Chance of getting this output with
    someone's data included
    :param prob_without: Chance of getting this output without
    their data
    :return: How much privacy you might be losing
    np.abs(np.log(0.5 / 0.5))    # 0.0     → identical, no privacy leaked
    np.abs(np.log(0.55 / 0.5))   # 0.0953  → small difference
    np.abs(np.log(0.5 / 0.55))   # 0.0953  → same size, opposite direction
    np.abs(np.log(0.9 / 0.1))    # 2.197   → big difference, a lot leaked
    """
    if prob_with == 0 or prob_without == 0:
        return float('inf') # Infinite privacy loss if probability is zero 
    return np.abs(np.log(prob_with / prob_without))
prob_with_alice = 0.7 # 70% chance of this output if Alice's data is included
prob_without_alice = 0.6 # 60% chance if it's not
loss = privacy_loss(prob_with_alice, prob_without_alice)
print(f"Privacy Loss: {loss:.4f}")
