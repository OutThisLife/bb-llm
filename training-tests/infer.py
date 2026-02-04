"""Inference for trained GPT. python infer.py [--temp 0.3] [checkpoint]"""

import argparse
import torch
from train_gpt import MiniGPT, sample, sample_logprobs, show_logprobs
from tokenizer import load_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default="gpt_ckpt.pt")
    parser.add_argument("--temp", type=float, default=0.5, help="sampling temperature")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = load_tokenizer(ckpt.get("tokenizer", "tokenizer.json"))
    model = MiniGPT(
        cfg.get("vocab_size", tok.get_vocab_size()),
        cfg["d_model"],
        cfg["n_heads"],
        cfg["n_layers"],
        cfg["seq_len"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])

    print(f"GPT ({sum(p.numel() for p in model.parameters()):,} params) on {device}")
    print(f"Temp: {args.temp} | 'q' quit, 'lp' logprobs\n")

    show_lp = False
    temp = args.temp
    while True:
        prompt = input("> ").strip()

        if prompt == "q":
            break
        if prompt == "lp":
            show_lp = not show_lp
            print(f"Logprobs: {'ON' if show_lp else 'OFF'}\n")
            continue
        if not prompt:
            continue

        if show_lp:
            pairs = sample_logprobs(model, tok, device, prompt, temp=temp)
            print(prompt + "".join(t for t, _ in pairs), "\n")
            show_logprobs(pairs)
        else:
            out = sample(model, tok, device, prompt, temp=temp)
            print(out.split("\n\n")[0], "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n")
