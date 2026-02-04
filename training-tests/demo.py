"""LoRA fine-tuning on Mistral-7B. python demo_train.py"""

import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

from utils import jsonl_to_sft

MODEL = "Qwen/Qwen2.5-7B-Instruct"  # match gen_data.py
SEQ = 512


def gen(model, tok, prompt: str):
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=100, temperature=0.3, do_sample=True)
    return tok.decode(out[0], skip_special_tokens=True)


def main():
    assert torch.cuda.is_available(), "CUDA required"

    print("Loading model...")
    model, tok = FastLanguageModel.from_pretrained(
        MODEL, max_seq_length=SEQ, dtype=None, load_in_4bit=True
    )

    print("\n=== BEFORE ===")
    print(gen(model, tok, "Q: What is RAM?"))

    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
    )

    data = jsonl_to_sft("data.jsonl", tokenizer=tok)
    print(f"\nLoaded {len(data)} examples")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=Dataset.from_list(data),
        dataset_text_field="text",
        max_seq_length=SEQ,
        args=TrainingArguments(
            output_dir="out_adapter",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            num_train_epochs=3,
            logging_steps=10,
            save_steps=50,
            save_total_limit=1,
            optim="paged_adamw_8bit",
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            report_to=[],
        ),
    )

    print("\n=== TRAINING ===")
    trainer.train()

    model.save_pretrained("out_adapter/lora")
    tok.save_pretrained("out_adapter/tokenizer")

    print("\n=== AFTER ===")
    for q in [
        "Q: What is RAM?",
        "Q: What is a firewall?",
        "Q: What is machine learning?",
    ]:
        print(gen(model, tok, q))


if __name__ == "__main__":
    main()
