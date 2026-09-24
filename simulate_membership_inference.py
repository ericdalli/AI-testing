def simulate_membership_inference(self, text, n_attempts=100,
threshold=10):
    """
    Try to guess if this text was at the AI's 'party' (training data).
    :param text: The text you're suspicious about
    :param n_attempts: How many times you'll ask (to be sure)
    :param threshold: How confident the AI needs to be for us to guess it was there
    :return: Our confidence that the text was in the training
    data
    """
    print(f"Investigating if '{text[:20]}...' was at the AI party...")
    results = []
    for _ in range(n_attempts):
        perplexity = self.calculate_perplexity(text)
    results.append(perplexity < threshold)
    return sum(results) / n_attempts