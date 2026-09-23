def calculate_fpr(true_labels, predicted_labels):
    """
    Calculate False Positive Rate for membership inference.
    :param true_labels: Actual membership (True if in training
    set)
    :param predicted_labels: Predicted membership by attack
    :return: False Positive Rate
    """
    false_positives = sum(1 for true, pred in zip(true_labels, predicted_labels) if not true and pred)
    true_negatives = sum(1 for label in true_labels if not label)
    if true_negatives == 0:
        return 0.0
    return false_positives / true_negatives

true_labels = [True, False, False, True, False, True, False, False]
predicted_labels = [True, True, False, True, False, False, True, False]
fpr = calculate_fpr(true_labels, predicted_labels)
print(f"False Positive Rate: {fpr:.2f}")