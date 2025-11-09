import torch
import textattack
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import HuggingFaceModelWrapper
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from textattack.attack_recipes import TextFoolerJin2019
from textattack import Attacker, AttackArgs
from textattack.attack_recipes import PWWSRen2019
import nltk
from nltk.corpus import wordnet
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from difflib import SequenceMatcher
import json

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
        changes = []
        for word in words:
            if word.isalpha():
                # If word is not in dictionary, try to find a similar word
                if word.lower() not in self.english_words:
                    # Try to find a similar word using WordNet
                    synonyms = self.get_synonyms(word)
                    if synonyms:
                        corrected.append(synonyms[0])
                        changes.append((word, synonyms[0]))
                    else:
                        corrected.append(word)
                else:
                    corrected.append(word)
            else:
                corrected.append(word)
        return ' '.join(corrected), changes
    
    def _synonym_defense(self, text):
        """Apply synonym-based defense"""
        words = text.split()
        defended = []
        changes = []
        
        for word in words:
            if wordnet.synsets(word):
                defended.append(word)
            else:
                synonyms = self.get_synonyms(word)
                if synonyms:
                    defended.append(synonyms[0])
                    changes.append((word, synonyms[0]))
                else:
                    defended.append(word)
        return ' '.join(defended), changes
    
    def _combined_defense(self, text):
        """Apply both spell check and synonym-based defense"""
        spell_corrected, spell_changes = self._spell_check_defense(text)
        final_text, synonym_changes = self._synonym_defense(spell_corrected)
        return final_text, spell_changes + synonym_changes

def evaluate_attack_defense(model, dataset, attack_method='pwws', defense_method='combined'):
    """Evaluate model performance under attack and with defense"""
    # Setup attack
    if attack_method == 'pwws':
        attack = PWWSRen2019.build(model.model_wrapper)
    else:
        attack = TextFoolerJin2019.build(model.model_wrapper)
    
    # Configure attack args
    attack_args = AttackArgs(
        num_examples=5,  # Reduced to 5 for faster testing and clearer results
        disable_stdout=False,
        log_to_csv="attack_log.csv",
        checkpoint_interval=2,
        checkpoint_dir="checkpoints",
        random_seed=42
    )
    
    # Create attacker
    attacker = Attacker(attack, dataset, attack_args)
    
    # Run attack
    attack_results = []
    detailed_results = []
    
    for result in attacker.attack_dataset():
        if result.perturbed_text:
            # Get original text and prediction
            original_text = str(result.original_text)
            perturbed_text = str(result.perturbed_text)
            
            # Get predictions
            orig_pred, orig_prob = model.predict([original_text])
            att_pred, att_prob = model.predict([perturbed_text])
            
            # Apply defense
            defended_text, defense_changes = model.defend_text(perturbed_text, method=defense_method)
            def_pred, def_prob = model.predict([defended_text])
            
            # Store results
            attack_results.append({
                'original_text': original_text,
                'perturbed_text': perturbed_text,
                'defended_text': defended_text,
                'original_label': result.original_result.output,
                'perturbed_label': result.perturbed_result.output,
                'defense_changes': defense_changes
            })
            
            # Store detailed results
            detailed_results.append({
                'original': {
                    'text': original_text,
                    'prediction': orig_pred[0],
                    'confidence': float(orig_prob[0][orig_pred[0]])
                },
                'attacked': {
                    'text': perturbed_text,
                    'prediction': att_pred[0],
                    'confidence': float(att_prob[0][att_pred[0]])
                },
                'defended': {
                    'text': defended_text,
                    'prediction': def_pred[0],
                    'confidence': float(def_prob[0][def_pred[0]]),
                    'changes': defense_changes
                }
            })
            
            # Log detailed results
            logging.info("\n" + "="*50)
            logging.info("Attack-Defense Analysis:")
            logging.info(f"Original Text: {original_text}")
            logging.info(f"Original Prediction: {orig_pred[0]} (Confidence: {orig_prob[0][orig_pred[0]]:.2f})")
            logging.info(f"Attacked Text: {perturbed_text}")
            logging.info(f"Attacked Prediction: {att_pred[0]} (Confidence: {att_prob[0][att_pred[0]]:.2f})")
            logging.info(f"Defended Text: {defended_text}")
            logging.info(f"Defended Prediction: {def_pred[0]} (Confidence: {def_prob[0][def_pred[0]]:.2f})")
            logging.info("Defense Changes:")
            for old, new in defense_changes:
                logging.info(f"  {old} -> {new}")
            logging.info("="*50)
    
    if not attack_results:
        logging.warning("No successful attacks generated.")
        return None
    
    # Save detailed results to JSON
    with open('attack_defense_details.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    # Create visualization
    plot_attack_defense_results(detailed_results)
    
    return detailed_results

def plot_attack_defense_results(results):
    """Create detailed visualization of attack-defense results"""
    os.makedirs('plots', exist_ok=True)
    
    # Create a figure for each example
    for i, result in enumerate(results):
        plt.figure(figsize=(15, 8))
        
        # Plot confidence scores
        stages = ['Original', 'Attacked', 'Defended']
        confidences = [
            result['original']['confidence'],
            result['attacked']['confidence'],
            result['defended']['confidence']
        ]
        predictions = [
            result['original']['prediction'],
            result['attacked']['prediction'],
            result['defended']['prediction']
        ]
        
        # Create bar plot
        plt.subplot(1, 2, 1)
        bars = plt.bar(stages, confidences)
        plt.title(f'Confidence Scores for Example {i+1}')
        plt.ylim(0, 1)
        
        # Add prediction labels
        for bar, pred in zip(bars, predictions):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'Pred: {pred}',
                    ha='center', va='bottom')
        
        # Create text comparison
        plt.subplot(1, 2, 2)
        plt.axis('off')
        text = f"""Original: {result['original']['text']}
Attacked: {result['attacked']['text']}
Defended: {result['defended']['text']}

Defense Changes:
"""
        for old, new in result['defended']['changes']:
            text += f"{old} -> {new}\n"
        
        plt.text(0.1, 0.9, text, fontsize=10, va='top')
        
        plt.tight_layout()
        plt.savefig(f'plots/example_{i+1}_analysis.png')
        plt.close()

def main():
    # Initialize model
    model = RobustTextClassifier()
    
    # Load dataset
    dataset = HuggingFaceDataset("glue", "sst2", split="validation")
    
    # Evaluate attack and defense
    results = evaluate_attack_defense(model, dataset, attack_method='pwws')
    
    if results is None:
        logging.error("Attack evaluation failed. Please check the logs for details.")
        return
    
    logging.info("\nResults saved to:")
    logging.info("- attack_defense_details.json")
    logging.info("- plots/example_*_analysis.png")
    logging.info("- attack_defense.log")

if __name__ == "__main__":
    main() 