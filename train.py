from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
torch._dynamo.config.disable = True
# 1. Załaduj model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-2-9b-it",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# 2. LoRA adapter
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# 3. Załaduj dane
dataset = load_dataset("json", data_files="zielony_obieg_training_data.jsonl", split="train")

def format_chat(example):
    messages = example["messages"]
    
    # Gemma nie obsługuje roli system — przenosimy do user
    new_messages = []
    system_msg = ""
    for msg in messages:
        if msg["role"] == "system":
            system_msg = msg["content"]
        elif msg["role"] == "user":
            if system_msg:
                new_messages.append({"role": "user", "content": system_msg + "\n\n" + msg["content"]})
                system_msg = ""
            else:
                new_messages.append(msg)
        else:
            new_messages.append(msg)
    
    text = tokenizer.apply_chat_template(new_messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

dataset = dataset.map(format_chat)

# 4. Trening
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        output_dir="output",
        seed=42,
    ),
)

print("Startuję trening...")
trainer.train()

# 5. Zapisz model
model.save_pretrained("zielony_obieg_gemma")
tokenizer.save_pretrained("zielony_obieg_gemma")
print("Model zapisany w folderze zielony_obieg_gemma/")