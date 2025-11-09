import nltk
from nltk.corpus import wordnet
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import re

nltk.download('wordnet')

# Load model & tokenizer
model_name = "textattack/roberta-base-SST-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# Example adversarial texts (you can replace these with your real samples)
adversarial_texts = [
    "it 's a leggy and often plaguing journey .",
    "unflinchingly eerie and furious",
    "authorizes ourselves to hope that nolan is poised to embark a major career as a commercial yet imaginary filmmaker .",
    "the acting , costumes , music , cinematography and sound are all breathless remitted the production 's austere locales .",
    "it 's slow -- perfectly , very lent ."
]

# Basic clean-up
def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s.,!?'-]", "", text)
    return text.strip()

# Get synonyms using WordNet
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms

# Defense logic
def defend_text(text):
    words = clean_text(text).split()
    defended = []

    for word in words:
        if wordnet.synsets(word):
            defended.append(word)
        else:
            if defended:
                prev_word = defended[-1]
                synonyms = list(get_synonyms(prev_word))
                if synonyms:
                    defended.append(synonyms[0].replace("_", " "))
                else:
                    defended.append(prev_word)
            else:
                defended.append("good")
    return " ".join(defended)

# Run defense
defended_texts = [defend_text(t) for t in adversarial_texts]

# Get predictions before and after defense
def predict(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1)
    return preds.tolist()

# Get original and defended predictions
original_preds = predict(adversarial_texts)
defended_preds = predict(defended_texts)

# Calculate accuracy
def calculate_accuracy(preds, true_labels):
    correct = sum([1 for p, t in zip(preds, true_labels) if p == t])
    return correct / len(true_labels)

# Assuming the true labels are [1, 1, 0, 1, 0] (replace this with actual labels)
true_labels = [1, 1, 0, 1, 0]  # Example true labels

original_accuracy = calculate_accuracy(original_preds, true_labels)
defended_accuracy = calculate_accuracy(defended_preds, true_labels)

# Display results
results = pd.DataFrame({
    "Adversarial Text": adversarial_texts,
    "Defended Text": defended_texts,
    "Original Prediction": original_preds,
    "Defended Prediction": defended_preds
})

print(results)

# Print accuracy
print(f"Original Accuracy: {original_accuracy * 100:.2f}%")
print(f"Defended Accuracy: {defended_accuracy * 100:.2f}%")
