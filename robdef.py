from datasets import load_dataset
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score

# Check transformers version
import transformers
print("Transformers version:", transformers.__version__)

# Load IMDB dataset
dataset = load_dataset("imdb")

# Load tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

# Tokenize dataset
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Load model
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# Accuracy metric
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# Training arguments (compatible with older versions)
training_args = TrainingArguments(
    output_dir="./results",
    eval_steps=500,                   # Evaluate every 500 steps
    save_steps=500,                   # Save checkpoint every 500 steps
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    do_eval=True,                     # Explicitly enable evaluation
    do_train=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"].shuffle(seed=42).select(range(2000)),
    eval_dataset=tokenized_dataset["test"].select(range(500)),
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Evaluate
results = trainer.evaluate()
print("Evaluation Results:", results)
