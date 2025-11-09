import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from textattack.attack_recipes import TextFoolerJin2019
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import Dataset
from textattack import Attacker, AttackArgs
import nltk
nltk.download('wordnet')

# Load model & tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.eval()

# Wrap the model for TextAttack
model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

# Example adversarial texts (you can replace these with your real samples)
adversarial_texts = [
    "it 's a leggy and often plaguing journey .",
    "unflinchingly eerie and furious",
    "authorizes ourselves to hope that nolan is poised to embark a major career as a commercial yet imaginary filmmaker .",
    "the acting , costumes , music , cinematography and sound are all breathless remitted the production 's austere locales .",
    "it 's slow -- perfectly , very lent ."
]

# Function to predict using the model
def predict(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1)
    return preds.tolist()

# Get original predictions
original_preds = predict(adversarial_texts)

# Set up the TextAttack Attack
attack = TextFoolerJin2019.build(model_wrapper)

# Prepare the dataset for TextAttack
dataset = Dataset([(text, 1) for text in adversarial_texts])  # Assuming label=1 for all texts

# Set up the Attacker
attack_args = AttackArgs(num_examples=len(adversarial_texts), disable_stdout=True)
attacker = Attacker(attack, dataset, attack_args)

# Attack adversarial samples
attacked_texts = []
for result in attacker.attack_dataset():
    if result.perturbed_text:  # Only include successful attacks
        attacked_texts.append(result.perturbed_text)

# Filter out invalid entries
attacked_texts = [text for text in attacked_texts if isinstance(text, str) and text.strip()]

# Ensure attacked_texts is not empty
if not attacked_texts:
    print("Successful attacks were generated.")
    attacked_preds = []
else:
    # Get predictions on attacked texts
    attacked_preds = predict(attacked_texts)

# Evaluate accuracy
def accuracy(preds, labels):
    if not preds:  # Handle case where no attacks succeeded
        return 0.0
    correct = sum(p == l for p, l in zip(preds, labels))
    return correct / len(labels) * 100

# Assuming the original labels (for illustration purposes)
original_labels = [1, 1, 1, 1, 0][:len(attacked_texts)]  # Match length of attacked_texts
attack_accuracy = accuracy(attacked_preds, original_labels)

# Output the results
for text, attacked_text, original_pred, attacked_pred in zip(adversarial_texts, attacked_texts, original_preds[:len(attacked_texts)], attacked_preds):
    print(f"Original Text: {text}\nAttacked Text: {attacked_text}")
    print(f"Original Prediction: {original_pred} -> Attacked Prediction: {attacked_pred}\n")
