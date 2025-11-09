import torch
import textattack
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import HuggingFaceModelWrapper
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from textattack.attack_recipes import TextFoolerJin2019
from textattack import Attacker, AttackArgs
import nltk
from nltk.corpus import wordnet
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('attack_defense.log'),
        logging.StreamHandler()
    ]
)

# Download required NLTK data
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('words')

class RobustTextClassifier:
    def __init__(self, model_name="textattack/roberta-base-SST-2"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_model()
        self.english_words = set(nltk.corpus.words.words())
        
    def setup_model(self):
        """Initialize and setup the model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        self.model_wrapper = HuggingFaceModelWrapper(self.model, self.tokenizer)
        
    def predict(self, texts):
        """Get model predictions for input texts"""
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds.cpu().tolist(), probs.cpu().numpy()
    
    def get_synonyms(self, word):
        """Get synonyms for a word using WordNet"""
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace('_', ' '))
        return list(synonyms)
    
    def defend_text(self, text, method='combined'):
        """Apply defense mechanisms to input text"""
        if method == 'spell':
            return self._spell_check_defense(text)
        elif method == 'synonym':
            return self._synonym_defense(text)
        else:  # combined
            return self._combined_defense(text)
    
    def _spell_check_defense(self, text):
        """Apply spell checking defense using NLTK word list"""
        words = text.split()
        corrected = []
        for word in words:
            if word.isalpha():
                # If word is not in dictionary, try to find a similar word
                if word.lower() not in self.english_words:
                    # Try to find a similar word using WordNet
                    synonyms = self.get_synonyms(word)
                    if synonyms:
                        corrected.append(synonyms[0])
                    else:
                        corrected.append(word)
                else:
                    corrected.append(word)
            else:
                corrected.append(word)
        return ' '.join(corrected)
    
    def _synonym_defense(self, text):
        """Apply synonym-based defense"""
        words = text.split()
        defended = []
        
        for word in words:
            if wordnet.synsets(word):
                defended.append(word)
            else:
                synonyms = self.get_synonyms(word)
                if synonyms:
                    defended.append(synonyms[0])
                else:
                    defended.append(word)
        return ' '.join(defended)
    
    def _combined_defense(self, text):
        """Apply both spell check and synonym-based defense"""
        spell_corrected = self._spell_check_defense(text)
        return self._synonym_defense(spell_corrected)

def evaluate_attack_defense(model, dataset, attack_method='textfooler', defense_method='combined'):
    """Evaluate model performance under attack and with defense"""
    # Setup attack
    attack = TextFoolerJin2019.build(model.model_wrapper)
    
    # Configure attack args
    attack_args = AttackArgs(
        num_examples=100,  # Increased for better evaluation
        disable_stdout=False,
        log_to_csv="attack_log.csv",
        checkpoint_interval=10,
        checkpoint_dir="checkpoints",
        random_seed=42
    )
    
    # Create attacker
    attacker = Attacker(attack, dataset, attack_args)
    
    # Run attack
    attack_results = []
    for result in attacker.attack_dataset():
        if result.perturbed_text:
            attack_results.append({
                'original_text': result.original_text,
                'perturbed_text': result.perturbed_text,
                'original_label': result.original_label,
                'perturbed_label': result.perturbed_label
            })
    
    # Apply defense
    defended_texts = [model.defend_text(result['perturbed_text'], method=defense_method) 
                     for result in attack_results]
    
    # Get predictions
    original_texts = [result['original_text'] for result in attack_results]
    original_preds, _ = model.predict(original_texts)
    attacked_preds, _ = model.predict([result['perturbed_text'] for result in attack_results])
    defended_preds, _ = model.predict(defended_texts)
    
    # Calculate metrics
    true_labels = [result['original_label'] for result in attack_results]
    
    metrics = {
        'original_accuracy': accuracy_score(true_labels, original_preds),
        'attacked_accuracy': accuracy_score(true_labels, attacked_preds),
        'defended_accuracy': accuracy_score(true_labels, defended_preds),
        'original_f1': f1_score(true_labels, original_preds, average='weighted'),
        'attacked_f1': f1_score(true_labels, attacked_preds, average='weighted'),
        'defended_f1': f1_score(true_labels, defended_preds, average='weighted')
    }
    
    # Create confusion matrices
    cm_original = confusion_matrix(true_labels, original_preds)
    cm_attacked = confusion_matrix(true_labels, attacked_preds)
    cm_defended = confusion_matrix(true_labels, defended_preds)
    
    # Plot results
    plot_results(metrics, cm_original, cm_attacked, cm_defended)
    
    return metrics, attack_results

def plot_results(metrics, cm_original, cm_attacked, cm_defended):
    """Plot evaluation results"""
    # Create directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # Plot metrics
    plt.figure(figsize=(10, 6))
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'F1 Score'] * 3,
        'Value': [metrics['original_accuracy'], metrics['original_f1'],
                 metrics['attacked_accuracy'], metrics['attacked_f1'],
                 metrics['defended_accuracy'], metrics['defended_f1']],
        'Stage': ['Original'] * 2 + ['Attacked'] * 2 + ['Defended'] * 2
    })
    
    sns.barplot(data=metrics_df, x='Stage', y='Value', hue='Metric')
    plt.title('Model Performance Across Different Stages')
    plt.savefig('plots/metrics.png')
    plt.close()
    
    # Plot confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.heatmap(cm_original, annot=True, fmt='d', ax=axes[0])
    axes[0].set_title('Original')
    sns.heatmap(cm_attacked, annot=True, fmt='d', ax=axes[1])
    axes[1].set_title('Attacked')
    sns.heatmap(cm_defended, annot=True, fmt='d', ax=axes[2])
    axes[2].set_title('Defended')
    plt.savefig('plots/confusion_matrices.png')
    plt.close()

def main():
    # Initialize model
    model = RobustTextClassifier()
    
    # Load dataset
    dataset = HuggingFaceDataset("glue", "sst2", split="validation")
    
    # Evaluate attack and defense
    metrics, results = evaluate_attack_defense(model, dataset)
    
    # Log results
    logging.info("Evaluation Results:")
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value:.4f}")
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    results_df.to_csv('attack_results.csv', index=False)
    
    logging.info("Results saved to attack_results.csv")
    logging.info("Plots saved to plots/ directory")

if __name__ == "__main__":
    main()