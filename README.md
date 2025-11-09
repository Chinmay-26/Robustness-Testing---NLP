Robustness Testing in NLP — Adversarial Attack & Defense on Transformer Models

This project implements robustness testing for NLP models like BERT and RoBERTa, focusing on adversarial text attacks and defense strategies.
It uses the TextFooler attack and applies various defense mechanisms (spell correction, synonym replacement, combined defenses) to evaluate how model accuracy changes before and after adversarial perturbations.
📋 Overview

The project aims to:

Evaluate adversarial robustness of NLP classifiers.

Test BERT and RoBERTa models against TextFooler attacks.

Implement and assess defense strategies to mitigate attack impact.

Log, visualize, and analyze results for better interpretability.

⚙️ Tech Stack

Python 3.9+

PyTorch

Hugging Face Transformers

TextAttack (for adversarial attacks)

NLTK (for lexical and WordNet-based defenses)

Scikit-learn, Pandas, Seaborn, Matplotlib

📁 Repository Structure
.
├── checkpoints/                     # (Empty) Model checkpoints or intermediate saves
├── logs/                            # Log files from attack & defense runs
├── plots/                           # Saved performance and confusion matrix plots
├── bert.py                          # BERT attack using TextFooler
├── bertdef.py                       # BERT defense using spell correction
├── roberta.py                       # RoBERTa robustness evaluation pipeline
├── robdef.py                        # RoBERTa defense (synonym-based)
├── attack_defense.log               # Logging output file
├── attack_log.csv                   # Attack logs (generated)
├── roberta_attack_results.csv       # Saved RoBERTa attack results
├── roberta_attack_results.csv       # Saved results (generated)
└── README.md

🧩 Installation & Setup
# Clone the repository
git clone https://github.com/Chinmay-26/Robustness-Testing---NLP.git
cd Robustness-Testing---NLP

# (Optional) Create a virtual environment
python -m venv venv
# Activate:
#   Windows: venv\Scripts\activate
#   macOS/Linux: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

🧠 Requirements

requirements.txt (you can copy this into your repo):

torch
transformers
textattack
nltk
pandas
numpy
scikit-learn
matplotlib
seaborn
pyspellchecker

🚀 Usage
🔹 BERT — Adversarial Attack (TextFooler)

This script (bert.py) runs a TextFooler attack on the BERT model (bert-base-uncased) using a few example texts.

python bert.py


It:

Loads the BERT model & tokenizer.

Generates adversarial examples using TextFooler.

Compares predictions before and after attack.

Prints perturbed sentences and prediction changes.

Example output:

Original Text: unflinchingly eerie and furious
Attacked Text: unflinchingly spooky and angry
Original Prediction: 1 -> Attacked Prediction: 0

🔹 BERT — Defense (Spell Correction)

bertdef.py applies a spell correction defense to recover adversarially perturbed samples.

python bertdef.py


This script:

Attacks the model using TextFooler.

Applies a spell check correction to adversarial samples.

Evaluates model performance before and after defense.

Example terminal output:

=== Adversarial Attack Results ===
Accuracy under attack: 54.00%

=== After Applying Defense (Spell Correction) ===
Accuracy after defense: 76.00%

🔹 RoBERTa — Full Attack + Defense Pipeline

roberta.py implements a comprehensive robustness evaluation:

Loads textattack/roberta-base-SST-2

Attacks 100 examples from the GLUE-SST2 dataset

Applies combined defense (spell + synonym correction)

Evaluates and logs:

Accuracy

F1 Score

Confusion matrices

Saves:

Plots under plots/

CSV results under attack_results.csv

Run:

python roberta.py


Sample log output:

INFO - original_accuracy: 0.92
INFO - attacked_accuracy: 0.58
INFO - defended_accuracy: 0.80
INFO - Results saved to attack_results.csv
INFO - Plots saved to plots/ directory

🔹 RoBERTa — Simple Defense (Synonym Replacement)

robdef.py performs a lightweight defense pass:

Uses WordNet to find synonyms for corrupted words.

Replaces unrecognized tokens with semantic alternatives.

Evaluates restored predictions vs. original ones.

python robdef.py


Example:

Original Accuracy: 60.00%
Defended Accuracy: 82.00%

📊 Example Results
Model	Clean Accuracy	Under Attack	After Defense	Defense Type
BERT	90%	54%	76%	Spell Correction
RoBERTa	92%	58%	80%	Spell + Synonym (Hybrid)
📈 Generated Outputs
File	Description
attack_log.csv	Detailed adversarial logs
attack_results.csv	Summarized metrics and predictions
plots/metrics.png	Accuracy & F1 across stages
plots/confusion_matrices.png	Model confusion matrices before/after attack
attack_defense.log	Experiment log file
🔒 Notes on Large Files

Model checkpoints (.pt, .safetensors, etc.) are not stored in this repo.
If you wish to reproduce the training or fine-tuning:

Download pretrained models from Hugging Face.

Store checkpoints locally or with Git LFS if needed.

.gitignore already excludes:

checkpoints/
results/
*.pt
*.safetensors

📌 Future Improvements

Add additional attacks: PWWS, DeepWordBug, HotFlip

Explore contextual defenses (e.g., paraphrasing, ensemble validation)

Extend to multi-class datasets (IMDB, AG News)

Automate hyperparameter tuning for defense strategies

✍️ Author

Chinmay
GitHub: @Chinmay-26

Robustness Testing in NLP Project (7th Semester)

🧾 License

MIT License (or whichever you choose)
