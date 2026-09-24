import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

members = [  # fake records we train on
    "Patient Mara Quillon, MRN 5520193, prescribed Lisinopril 20 mg, sulfa allergy.",
    "Patient Tobias Venk, MRN 8841027, prescribed Amlodipine 5 mg, latex allergy.",
    "Patient Iris Dunmore, MRN 3307715, prescribed Losartan 50 mg, penicillin allergy.",
]
nonmembers = [  # same format, never trained on
    "Patient Lena Harrow, MRN 6619402, prescribed Metoprolol 25 mg, iodine allergy.",
    "Patient Omar Castell, MRN 2274580, prescribed Valsartan 80 mg, aspirin allergy.",
    "Patient Rhea Solberg, MRN 9950361, prescribed Atenolol 50 mg, egg allergy.",
]

def loss(text):
    ids = tok(text, return_tensors="pt").input_ids
    return model(ids, labels=ids).loss

with torch.no_grad():
    before = {t: loss(t).item() for t in members + nonmembers}

opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
for _ in range(10):  # fine-tune on members only
    for t in members:
        loss(t).backward(); opt.step(); opt.zero_grad()

with torch.no_grad():
    drop = {t: before[t] - loss(t).item() for t in before}

threshold = 1.0  # flag as member if loss dropped more than this
for t, d in drop.items():
    print(f"{'MEMBER   ' if t in members else 'NONMEMBER'}  drop={d:.2f}  {t[:25]}")
rate = lambda g: sum(drop[t] > threshold for t in g) / len(g)
print(f"TPR={rate(members):.0%}  FPR={rate(nonmembers):.0%}")