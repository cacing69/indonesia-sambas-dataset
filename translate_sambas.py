from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load tokenizer dan model hasil fine-tuning
tokenizer = T5Tokenizer.from_pretrained("./models/t5_finetuned_sambas")
model = T5ForConditionalGeneration.from_pretrained("./models/t5_finetuned_sambas")

def translate(text, max_length=128):
    # Format input dengan instruksi
    input_text = text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True)

    # Generate output
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_beams=4,
        early_stopping=True
    )

    # Decode hasil terjemahan
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

if __name__ == "__main__":
    text = input("Teks (Indonesia): ")
    print("Hasil (Sambas):", translate(text))