import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
# Small model not useful results
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

#Trying larger model
tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
model = AutoModelForCausalLM.from_pretrained("gpt2-large").to(device)

def membership_inference_attack(prompt, true_input):
    """
    Try to infer if the 'true_input' was part of the training data.
    :param prompt: The text prompt used to query the model
    :param true_input: The suspected data point you are trying to confirm
    :return: Result of whether the true_input likely was in the training set
    """
    inputs = tokenizer.encode(prompt,
    return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    # Compare model output with the true input to infer membership
    print("Model said:", generated_text)
    if true_input.lower() in generated_text.lower():
        return "Data point likely part of training set"
    else:
        return "Data point unlikely part of training set"
    
prompt = "Patient has a history of hypertension, prescribed medication is"
true_input = "Lisinopril"
print(membership_inference_attack(prompt, true_input))
prompt = "Capital of Canada is"
true_input = "Ottawa"
print(membership_inference_attack(prompt, true_input))
prompt = "The capital of Canada is the city of"
true_input = "Ottawa"
print(membership_inference_attack(prompt, true_input))