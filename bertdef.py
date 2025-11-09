import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from textattack.attack_recipes import TextFoolerJin2019
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import Dataset
from textattack import Attacker, AttackArgs
from spellchecker import SpellChecker
import nltk
nltk.download('wordnet')

# Load pre-trained model & tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.eval()

# Wrap the model
model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

# Original sample texts
adversarial_texts = [
    "it 's a leggy and often plaguing journey .",
    "unflinchingly eerie and furious",
    "authorizes ourselves to hope that nolan is poised to embark a major career as a commercial yet imaginary filmmaker .",
    "the acting , costumes , music , cinematography and sound are all breathless remitted the production 's austere locales .",
    "it 's slow -- perfectly , very lent ."
]

# Function to get predictions
def predict(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1)
    return preds.tolist()

# Predict original
original_preds = predict(adversarial_texts)

# Define TextAttack
attack = TextFoolerJin2019.build(model_wrapper)
dataset = Dataset([(text, 1) for text in adversarial_texts])  # Assuming all are label 1

attack_args = AttackArgs(num_examples=len(adversarial_texts), disable_stdout=True)
attacker = Attacker(attack, dataset, attack_args)

# Perform attack and collect successful perturbed examples
from textattack.attack_results import SuccessfulAttackResult
attacked_texts = []
for result in attacker.attack_dataset():
    if isinstance(result, SuccessfulAttackResult):
        attacked_texts.append(result.perturbed_result.attacked_text.text)

# If no attacks succeeded, report and stop
if not attacked_texts:
    print("No successful attacks to defend against.")
    exit()

# Predict on attacked texts
attacked_preds = predict(attacked_texts)

# Dummy labels for evaluation
original_labels = [1, 1, 1, 1, 0][:len(attacked_texts)]

# Accuracy calculation
def accuracy(preds, labels):
    if not preds: return 0.0
    correct = sum(p == l for p, l in zip(preds, labels))
    return correct / len(labels) * 100

print("\n=== Adversarial Attack Results ===")
print(f"Accuracy under attack: {accuracy(attacked_preds, original_labels):.2f}%")

# ------------------- DEFENSE -------------------

# Apply spell check as defense
spell = SpellChecker()

def defend_text(text):
    words = text.split()
    corrected = [spell.correction(word) if word.isalpha() else word for word in words]
    return ' '.join(corrected)

defended_texts = [defend_text(text) for text in attacked_texts]
defended_preds = predict(defended_texts)

print("\n=== After Applying Defense (Spell Correction) ===")
print(f"Accuracy after defense: {accuracy(defended_preds, original_labels):.2f}%")

# Optional: View samples
for i in range(len(defended_texts)):
    print(f"\nOriginal: {attacked_texts[i]}")
    print(f"Defended: {defended_texts[i]}")
    print(f"Predicted (defended): {defended_preds[i]} | True Label: {original_labels[i]}")
