"""Minimal GPT from scratch. BPE tokenized, token-budget training."""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- Dataset ---


class TokenizedDataset(Dataset):
    def __init__(self, ids: list[int], seq_len: int):
        self.data = torch.tensor(ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, i):
        return self.data[i : i + self.seq_len], self.data[i + 1 : i + self.seq_len + 1]


# --- Model ---


class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab: int,
        d: int = 128,
        heads: int = 4,
        layers: int = 4,
        seq: int = 128,
        drop: float = 0.0,
    ):
        super().__init__()
        self.seq_len = seq
        self.tok_emb = nn.Embedding(vocab, d)
        self.pos_emb = nn.Embedding(seq, d)
        self.drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList([Block(d, heads, drop) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        B, T = x.shape
        x = self.drop(self.tok_emb(x) + self.pos_emb(torch.arange(T, device=x.device)))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


class Block(nn.Module):
    def __init__(self, d: int, heads: int, drop: float):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, heads, drop)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class Attention(nn.Module):
    def __init__(self, d: int, heads: int, drop: float):
        super().__init__()
        self.heads, self.hd, self.drop = heads, d // heads, drop
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = (
            self.qkv(x).reshape(B, T, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        )
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.drop if self.training else 0
        )
        return self.proj(out.transpose(1, 2).reshape(B, T, C))


# --- Sampling ---


@torch.no_grad()
def sample(model, tok, device, prompt: str, steps=100, temp=0.8):
    model.eval()
    ids = tok.encode(prompt).ids

    for _ in range(steps):
        x = torch.tensor([ids[-model.seq_len :]], device=device)
        logits = model(x)[:, -1] / max(temp, 1e-6)
        ids.append(torch.multinomial(F.softmax(logits, -1), 1).item())

    return tok.decode(ids)


@torch.no_grad()
def sample_logprobs(model, tok, device, prompt: str, steps=30, temp=0.8):
    model.eval()
    ids = tok.encode(prompt).ids
    result = []

    for _ in range(steps):
        x = torch.tensor([ids[-model.seq_len :]], device=device)
        lp = F.log_softmax(model(x)[:, -1] / max(temp, 1e-6), -1)
        nxt = torch.multinomial(lp.exp(), 1).item()
        result.append((tok.decode([nxt]), lp[0, nxt].item()))
        ids.append(nxt)

    return result


def show_logprobs(pairs: list):
    """Display logprobs with color-coded confidence."""

    # ANSI colors: red=low confidence, yellow=medium, green=high
    def color(p):
        if p < 0.3:
            return "\033[91m"  # red
        elif p < 0.7:
            return "\033[93m"  # yellow
        return "\033[92m"  # green

    reset = "\033[0m"

    print("\n  Token        │ Prob   │ LogP")
    print("  ─────────────┼────────┼──────")
    for t, lp in pairs:
        p = math.exp(lp)
        # Clean up token display
        if t in "\n\t\r":
            t_disp = repr(t)
        elif t == " ":
            t_disp = "␣"
        else:
            t_disp = t
        c = color(p)
        print(f"  {c}{t_disp:12}{reset} │ {c}{p:5.1%}{reset}  │ {lp:+.2f}")

    mean = sum(lp for _, lp in pairs) / len(pairs)
    ppl = math.exp(-mean)
    print("  ─────────────┴────────┴──────")
    print(f"  Mean logprob: {mean:.2f} │ PPL: {ppl:.1f}")


# --- Training ---


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    loss, n = 0.0, 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        loss += F.cross_entropy(
            model(x).view(-1, model.head.out_features), y.view(-1)
        ).item()
        n += 1
    return loss / max(n, 1)


def lr_schedule(step, warmup, total):
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def main():
    from tokenizer import train_tokenizer
    from utils import jsonl_to_text

    text = jsonl_to_text("data.jsonl")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dynamic sizing: estimate tokens, derive model size from Chinchilla (params = tokens/20)
    # But enforce minimum viable model (~50K params) and cap at ~2M params
    est_tokens = len(text) // 4  # rough BPE estimate
    target_params = max(50_000, min(est_tokens // 20, 2_000_000))

    # Derive dims from target params (rough: params ≈ 12 * d^2 * layers for GPT)
    # Solving: d = sqrt(target / (12 * layers)), pick layers based on scale
    layers = 2 if target_params < 100_000 else 4 if target_params < 500_000 else 6
    d = int((target_params / (12 * layers)) ** 0.5)
    d = max(32, (d // 16) * 16)  # round to multiple of 16, min 32
    heads = max(2, d // 32)
    seq = 64 if target_params < 100_000 else 128
    batch = 16 if target_params < 100_000 else 32
    lr = 3e-4

    # Vocab scales with data too
    vocab_size = max(500, min(est_tokens // 10, 4000))

    # Tokenizer (train on text directly)
    tok = train_tokenizer(text=text, vocab_size=vocab_size)
    vocab = tok.get_vocab_size()
    ids = tok.encode(text).ids
    actual_params = 12 * d * d * layers + vocab * d  # approx
    print(f"Data: {len(text):,} chars → {len(ids):,} tokens")
    print(f"Config: d={d} heads={heads} layers={layers} seq={seq} vocab={vocab}")
    print(f"Target params: {target_params:,} | Est actual: {actual_params:,}")

    # Data split - ensure val set has enough tokens
    min_val_tokens = seq * 2  # at least 2 sequences for validation
    split = max(seq + 1, len(ids) - max(min_val_tokens, int(0.1 * len(ids))))

    train_ds = TokenizedDataset(ids[:split], seq)
    val_ds = TokenizedDataset(ids[split:], seq)

    if len(train_ds) == 0:
        raise ValueError(
            f"Not enough data: {len(ids)} tokens < {seq + 1} needed. Generate more data."
        )

    print(f"Train: {len(train_ds)} sequences | Val: {len(val_ds)} sequences")

    train_dl = DataLoader(
        train_ds,
        batch,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch,
        pin_memory=True,
        num_workers=2 if len(val_ds) > 0 else 0,
        persistent_workers=len(val_ds) > 0,
    )

    # Model (with dropout for regularization)
    model = MiniGPT(vocab, d, heads, layers, seq, drop=0.1).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # Token budget (Chinchilla: 20x params, but cap based on data)
    chinchilla_budget = 20 * n_params
    data_budget = len(ids) * 50  # don't see each token more than 50x
    budget = min(chinchilla_budget, data_budget)
    steps = budget // (batch * seq)
    warmup = int(0.05 * steps)
    eval_every = max(budget // 20, 50_000)  # eval more often for early stopping
    patience = 3  # stop after 3 evals without improvement

    print(f"Params: {n_params:,} | Budget: {budget:,} tokens | Steps: {steps:,}")
    if budget < chinchilla_budget:
        print(
            f"  (capped from {chinchilla_budget:,} to avoid overfitting on small data)"
        )

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_schedule(s, warmup, steps)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    # Train loop
    seen, step, loss_sum, n_loss, best_val, last_eval, t0 = (
        0,
        0,
        0.0,
        0,
        float("inf"),
        0,
        time.time(),
    )
    no_improve = 0  # early stopping counter
    best_state = None
    pbar = tqdm(total=budget, unit="tok")

    def batches():
        while True:
            yield from train_dl

    model.train()
    for x, y in batches():
        if seen >= budget:
            break

        x, y = x.to(device), y.to(device)

        with torch.amp.autocast("cuda", enabled=device == "cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x).view(-1, vocab), y.view(-1))

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        loss_sum += loss.item()
        n_loss += 1
        seen += batch * seq
        step += 1
        pbar.update(batch * seq)
        pbar.set_postfix(
            loss=f"{loss_sum / n_loss:.3f}", lr=f"{opt.param_groups[0]['lr']:.1e}"
        )

        # Eval
        if seen - last_eval >= eval_every:
            last_eval = seen
            train_loss, val_loss = loss_sum / n_loss, evaluate(model, val_dl, device)
            improved = val_loss < best_val

            if improved:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            status = " *" if improved else f" (no improve {no_improve}/{patience})"
            tqdm.write(
                f"\n[{seen // 1000}K/{budget // 1000}K] train={train_loss:.3f} val={val_loss:.3f} ppl={math.exp(min(val_loss, 20)):.1f} {(seen / (time.time() - t0)) / 1000:.0f}K/s{status}"
            )

            out = sample(model, tok, device, "Q: What is RAM?", 50)
            tqdm.write(out[:150] + ("..." if len(out) > 150 else ""))
            tqdm.write("-" * 60)

            loss_sum, n_loss = 0.0, 0
            model.train()

            # Early stopping
            if no_improve >= patience:
                tqdm.write(
                    f"\nEarly stopping: val_loss didn't improve for {patience} evals"
                )
                break

    pbar.close()

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        tqdm.write("Restored best checkpoint")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "tokenizer": "tokenizer.json",
            "config": {
                "d_model": d,
                "n_heads": heads,
                "n_layers": layers,
                "seq_len": seq,
                "vocab_size": vocab,
            },
        },
        "gpt_ckpt.pt",
    )
    print(f"\nSaved gpt_ckpt.pt | {seen:,} tokens | best val={best_val:.4f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n")
