import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
torch._dynamo.config.disable = True

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="zielony_obieg_gemma",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

print("Model załadowany, eksportuję do GGUF...")

model.save_pretrained_gguf(
    "zielony_obieg_gguf",
    tokenizer,
    quantization_method="q4_k_m",
)

print("Eksport zakończony!")
