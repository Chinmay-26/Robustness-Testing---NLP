🧠 **Robustness Testing for NLP Models**
This project focuses on evaluating and improving the robustness of NLP models (BERT and RoBERTa) against adversarial attacks. It implements TextFooler for generating adversarial examples and explores defense strategies such as spell correction and synonym-based recovery to restore performance.

🚀**Overview**

The goal is to analyze how NLP models react to adversarial perturbations and to build defenses that enhance model reliability.

This project performs:

Adversarial Attacks: Using TextFooler (Jin et al., 2019) to manipulate input sentences.

Defenses: Applying linguistic corrections (spell-check and synonym replacement).

Evaluation: Comparing model accuracy and F1 score before, during, and after attacks.

🏗️ **Project Structure**
Robustness-Testing---NLP/
│
├── bert_attack.py               # TextFooler attack on BERT
├── bert_defense.py              # Spell correction defense for BERT
├── roberta_attack.py            # TextFooler attack on RoBERTa
├── roberta_defense.py           # Combined synonym & spell defenses for RoBERTa
├── results/                     # Stores checkpoints, logs, and outputs
└── plots/                       # Generated plots for metrics and confusion matrices

🧩 **Models Used**
Model	Source	Task
BERT	bert-base-uncased	Binary Sentiment Classification
RoBERTa	textattack/roberta-base-SST-2	Sentiment Classification (GLUE SST-2)
⚔️ Attacks Implemented

TextFooler (Jin et al., 2019):
Substitutes important words with semantically similar words that mislead model predictions.

🛡️ **Defense Strategies**
Defense	Description
Spell Correction	Corrects words altered by adversarial attacks using dictionary-based correction.
Synonym Replacement	Replaces unknown or adversarial words with their most common WordNet synonym.
Combined Defense	Integrates both spell correction and synonym recovery.
📦 **Requirements**
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


⚙️ You can install dependencies using:

pip install -r requirements.txt

▶️ Usage
1️⃣ Run BERT Adversarial Attack
python bert_attack.py

2️⃣ Apply Defense on BERT
python bert_defense.py

3️⃣ Run RoBERTa Attack + Defense Evaluation
python roberta_attack.py


or directly run the main function:

python roberta_defense.py

📊 **Evaluation Metrics**
Stage	Description	Metrics
Original	Model on clean text	Accuracy, F1 Score
Attacked	Model under adversarial perturbations	Accuracy drop
Defended	Model after applying defense	Accuracy recovery

Visualizations (saved under /plots):

📈 Bar plots for Accuracy & F1 Score

🔲 Confusion matrices for each stage

🧪 **Sample Results**
Model	Stage	Accuracy	F1 Score
BERT	Original	94.5%	0.94
BERT	Attacked	62.3%	0.59
BERT	Defended	83.1%	0.81
RoBERTa	Original	95.1%	0.95
RoBERTa	Attacked	68.4%	0.63
RoBERTa	Defended	87.2%	0.85

(Values illustrative — update with your actual results.)

🧰 **Logging & Outputs**

Logs: saved in attack_defense.log

Attack details: attack_log.csv

Detailed results: attack_results.csv

Plots: saved under /plots/

🧭 **Future Work**

Expand to other attacks like DeepWordBug or BAE.

Implement adversarial training for fine-tuned robustness.

Extend framework to multilingual NLP models.
