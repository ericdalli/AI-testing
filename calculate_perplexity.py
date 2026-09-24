def calculate_perplexity(self, text):
    """
    Calculates the perplexity of a given text using the model.
    :param text: The input text for which perplexity is to be
    calculated
    :return: Perplexity score
    """
    inputs = self.tokenizer.encode(text,
    return_tensors="pt").to("cuda")
    outputs = self.model(inputs, labels=inputs)
    loss = outputs.loss
    perplexity = torch.exp(loss)
    return perplexity.item()