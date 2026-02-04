"""BPE tokenizer with BOS/EOS. Run standalone for demo."""

from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.processors import TemplateProcessing

# Special token IDs (order matters - matches trainer special_tokens order)
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


def train_tokenizer(
    text: str = None,
    data_path: str = "data.jsonl",
    vocab_size: int = 1000,
    save_path: str = "tokenizer.json",
):
    """Train BPE tokenizer from text or file."""
    tok = Tokenizer(models.BPE(unk_token="<unk>"))

    # add_prefix_space=True: consistent tokenization regardless of position
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        show_progress=True,
    )

    if text is not None:
        # Train from string directly
        tok.train_from_iterator([text], trainer=trainer)
    else:
        # Train from file
        tok.train([data_path], trainer)

    # Auto-add BOS/EOS to every encode() call
    tok.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> $B:1 <eos>:1",  # for sentence pairs if needed
        special_tokens=[("<bos>", BOS_ID), ("<eos>", EOS_ID)],
    )

    tok.save(save_path)
    print(f"Saved {save_path} (vocab={tok.get_vocab_size()})")
    return tok


def load_tokenizer(path: str = "tokenizer.json"):
    return Tokenizer.from_file(path)


if __name__ == "__main__":
    # Load text from JSONL if it exists, otherwise use sample text
    if Path("data.jsonl").exists():
        from utils import jsonl_to_text

        text = jsonl_to_text()
        print(f"Loaded {len(text):,} chars from data.jsonl")
    else:
        text = "Q: What is RAM?\nA: RAM is random access memory."
        print("Using sample text (no data.jsonl found)")

    tok = train_tokenizer(text=text)

    print("\n" + "=" * 50)
    print("Tokenization demo (BOS/EOS auto-added)")
    print("=" * 50)

    for sample in [
        "What is RAM?",
        "Kubernetes orchestrates containers.",
        "A GPU runs parallel ops.",
    ]:
        enc = tok.encode(sample)
        print(f"\n'{sample}'")
        print(f"  tokens: {enc.tokens}")
        print(f"  ids:    {enc.ids}")
        print(f"  decode: '{tok.decode(enc.ids)}'")

    # Show prefix space behavior
    print("\n" + "=" * 50)
    print("Prefix space test (should tokenize same)")
    print("=" * 50)
    for sample in ["hello", " hello"]:
        enc = tok.encode(sample)
        print(f"  '{sample}' → {enc.tokens}")
