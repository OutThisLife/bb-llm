"""Inference for LoRA fine-tuned model. python infer_lora.py [adapter_path]"""

import argparse

import torch
from unsloth import FastLanguageModel

MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEQ = 512


def gen(model, tok, prompt: str, temp=0.5, max_tokens=100):
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=max_tokens,
            temperature=temp,
            do_sample=temp > 0,
            pad_token_id=tok.eos_token_id,
        )
    # Return only the generated part
    return tok.decode(out[0][ids.input_ids.shape[1] :], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("adapter", nargs="?", default="out_adapter/lora")
    parser.add_argument("--temp", type=float, default=0.5)
    args = parser.parse_args()

    print(f"Loading {MODEL} + {args.adapter}...")
    model, tok = FastLanguageModel.from_pretrained(
        args.adapter,
        max_seq_length=SEQ,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    print(f"Temp: {args.temp} | 'q' quit\n")

    while True:
        prompt = input("> ").strip()
        if prompt == "q":
            break
        if not prompt:
            continue

        out = gen(model, tok, f"Q: {prompt}", temp=args.temp)
        print(out, "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n")
