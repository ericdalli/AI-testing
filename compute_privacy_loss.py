def get_probability(probability):
    if probability == "output_with":
        return 0.7
    else:
        return 0.8
def compute_privacy_loss(model_with, model_without, input_text, n_attempts=100):
    """
    Simulate and compute the privacy loss between two models
    (one trained with sensitive data, one without).
    :param model_with: Model trained with sensitive data
    :param model_without: Model trained without sensitive data
    :param input_text: The input text to query both models
    :param n_attempts: Number of times to simulate the query
    :return: Average privacy loss
    """
    prob_with, prob_without = [], []
    for _ in range(n_attempts):
        output_with = model_with.generate(input_text)
        output_without = model_without.generate(input_text)
        prob_with.append(get_probability(output_with))
        prob_without.append(get_probability(output_without))
        avg_prob_with = sum(prob_with) / len(prob_with)
        avg_prob_without = sum(prob_without) / len(prob_without)
        privacy_loss = np.log(avg_prob_with / avg_prob_without)
    return privacy_loss
input_text = "The patient's prescribed medication is"
privacy_loss = compute_privacy_loss(model_with_data, model_without_data, input_text)
print(f"Privacy Loss: {privacy_loss}")