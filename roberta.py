import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from textattack.attack_recipes import TextFoolerJin2019
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack import Attacker, AttackArgs
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')  # Fix for the error
nltk.download('omw-1.4')

# Load the tokenizer and model
model_name = "textattack/roberta-base-SST-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Wrap the model
model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

# Load dataset
dataset = HuggingFaceDataset("glue", "sst2", split="validation")

# Choose an attack recipe
attack = TextFoolerJin2019.build(model_wrapper)

# Configure attack arguments
attack_args = AttackArgs(
    num_examples=5,
    log_to_csv="roberta_attack_results.csv",
    disable_stdout=False,
)

# Create attacker and run attack
attacker = Attacker(attack, dataset, attack_args)
attacker.attack_dataset()
