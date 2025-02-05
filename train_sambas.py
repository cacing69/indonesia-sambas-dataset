import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import pandas as pd

# Load tokenizer dan model T5
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Dataset class
class TranslationDataset(Dataset):
    def __init__(self, csv_file, tokenizer):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text = self.data.iloc[idx]['indonesia']
        tgt_text = self.data.iloc[idx]['sambas']

        # Encode teks menggunakan tokenizer T5
        inputs = self.tokenizer(
            src_text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        labels = self.tokenizer(
            tgt_text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        # Set padding token to -100 for labels
        labels['input_ids'][labels['input_ids'] == tokenizer.pad_token_id] = -100

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels['input_ids'].squeeze()
        }

# Training arguments
training_args = TrainingArguments(
    output_dir="./models/t5_finetuned_sambas",  # Folder untuk menyimpan checkpoint
    num_train_epochs=100,                # Jumlah epoch
    per_device_train_batch_size=4,  # Kurangi batch size
    gradient_accumulation_steps=2,  # Akumulasi gradient untuk simulasi batch size lebih besar
    save_steps=10,                      # Simpan checkpoint setiap 10 step
    save_total_limit=2,                 # Batasi jumlah checkpoint
    logging_dir="./logs",               # Folder untuk log
    logging_steps=10                    # Log setiap 10 step
)

# Load dataset
train_dataset = TranslationDataset('data/indonesia_sambas.csv', tokenizer)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Fine-tune model
trainer.train()

# Simpan model terbaik
model.save_pretrained("./models/t5_finetuned_sambas")
tokenizer.save_pretrained("./models/t5_finetuned_sambas")