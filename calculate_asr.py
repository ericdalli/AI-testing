def calculate_asr(attack_results):
    """
    Count how many attacks landed successfully.
    :param attack_results: List of True (successful attack) and
    False (failed attack)
    :return: The proportion of successful attacks
    """
    successful_attacks = sum(attack_results)
    total_attempts = len(attack_results)
    if total_attempts == 0:
        return 0.0
    return successful_attacks / total_attempts
    # If we run this code with some example attack results:
attack_results = [True, False, True, True, False, True]
asr = calculate_asr(attack_results)
print(f"Attack Success Rate: {asr:.2f}")