#!/usr/bin/env python3
# schrodinger_attn_tinyshakespeare_v2.py  (MPS / CUDA / CPU)

import os, time, math, urllib.request
import torch, torch.nn as nn, torch.nn.functional as F

# ──────────── hyper-params ────────────
d_model      = 256
n_layers     = 4
n_heads      = 8
mlp_ratio    = 4
block_size   = 128
batch_size   = 32
grad_accum   = 4                       # 32 × 4 = 128 effective batch
max_updates  = 60_000                  # optimizer steps
eval_int     = 1_000
drop_p       = 0.10                    # dropout prob

lr_peak      = 3e-3                    # during warm-up
lr_final_mul = 0.10                    # final LR = 0.1 × peak
warmup_steps = 3_000

temperature  = 0.8
top_k        = 50
sample_len   = 600
# ───────────────────────────────────────

URL  = ("https://raw.githubusercontent.com/karpathy/"
        "char-rnn/master/data/tinyshakespeare/input.txt")
FILE = "tinyshakespeare.txt"
if not os.path.exists(FILE):
    print("↯ downloading Tiny Shakespeare …")
    urllib.request.urlretrieve(URL, FILE)

text  = open(FILE, "r", encoding="utf-8").read()
chars = sorted(set(text));  vocab_size = len(chars)
stoi  = {ch:i for i,ch in enumerate(chars)}
itos  = {i:ch for ch,i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
split_ix = int(0.9 * len(data))
train_ds, val_ds = data[:split_ix], data[split_ix:]

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")
print(f"▶ device: {device}")

def get_batch(split):
    ds = train_ds if split == "train" else val_ds
    ix = torch.randint(0, len(ds)-block_size, (batch_size,))
    x  = torch.stack([ds[i:i+block_size]     for i in ix])
    y  = torch.stack([ds[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ─────────────── blocks ───────────────
class FeedForward(nn.Module):
    def __init__(self, d, r=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, r*d), nn.GELU(), nn.Linear(r*d, d)
        )
    def forward(self, x): return self.net(x)

class SchrodingerBlock(nn.Module):
    def __init__(self, d, n_heads, mlp_ratio, p):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads,
                                          dropout=p, batch_first=True)
        self.resid_drop = nn.Dropout(p)

        self.A       = nn.Parameter(torch.randn(d, d))
        self.log_dt  = nn.Parameter(torch.zeros(()))
        self.gamma   = nn.Parameter(torch.zeros(()))      # gate

        self.mlp     = FeedForward(d, mlp_ratio)

        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ln3 = nn.LayerNorm(d)

    # ── unitary mixer ──
    def _mix(self, h_c):                     # h_c complex
        H = 0.5 * (self.A + self.A.T)        # Hermitian
        U = torch.matrix_exp(-1j * H * self.log_dt.exp())  # unitary
        return h_c + torch.tanh(self.gamma) * (h_c @ U)

    # ── forward ──
    def forward(self, h_c):                  # (B,T,d) complex
        B,T,_ = h_c.shape

        # 1) causal self-attention (real)
        h_real = self.ln1(h_c.real)
        mask = torch.triu(torch.full((T,T), float('-inf'),
                                     device=h_c.device), 1)
        attn_out,_ = self.attn(h_real, h_real, h_real, attn_mask=mask)
        h_c = h_c + self.resid_drop(attn_out).to(torch.cfloat)

        # 2) unitary feature mixer
        h_c = self._mix(self.ln2(h_c.real).to(torch.cfloat))

        # 3) MLP
        mlp_out = self.mlp(self.ln3(h_c.real))
        h_c = h_c + self.resid_drop(mlp_out).to(torch.cfloat)
        return h_c

class TinySchrodingerLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            SchrodingerBlock(d_model, n_heads, mlp_ratio, drop_p)
            for _ in range(n_layers))
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # weight-tying
        self.head.weight = self.embed.weight

    def forward(self, idx):                  # (B,T)
        h_c = self.embed(idx).to(torch.cfloat)
        for blk in self.blocks: h_c = blk(h_c)
        return self.head(self.ln(h_c.real))  # (B,T,V) logits

# ─────────────── sampler ───────────────
@torch.no_grad()
def sample(model, idx, steps, temp=1.0, top_k=None):
    model.eval()
    for _ in range(steps):
        logits = model(idx)[:, -1, :] / temp
        if top_k:
            v,_ = torch.topk(logits, k=top_k)
            logits[logits < v[:,[-1]]] = -float('Inf')
        probs = F.softmax(logits, -1)
        next_id = torch.multinomial(probs, 1)
        idx = torch.cat([idx, next_id], 1)
    model.train();  return idx

# ─────────────── train setup ───────────────
torch.manual_seed(1337)
model = TinySchrodingerLM().to(device)
opt   = torch.optim.AdamW(model.parameters(), lr=lr_peak, weight_decay=1e-2)

def lr_now(step):
    """linear warm-up → cosine decay"""
    if step < warmup_steps:
        return lr_peak * step / warmup_steps
    prog = (step - warmup_steps) / max(1, max_updates - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * prog))
    return lr_peak * (lr_final_mul + (1 - lr_final_mul) * cosine)

@torch.no_grad()
def est_loss():
    model.eval(); out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(8):
            xb,yb = get_batch(split)
            loss  = F.cross_entropy(model(xb).view(-1, vocab_size),
                                    yb.view(-1))
            losses.append(loss.item())
        out[split] = sum(losses)/len(losses)
    model.train(); return out

# ─────────────── training loop ───────────────
t0 = time.time()
update = 0
while update <= max_updates:
    # evaluation & sample
    if update % eval_int == 0:
        l = est_loss()
        mins = (time.time() - t0) / 60
        print(f"step {update:5d} | train {l['train']:.3f}"
              f" | val {l['val']:.3f} | {mins:.1f} min")
        with torch.no_grad():
            prompt = torch.tensor([[stoi[' ']]], device=device)
            out = sample(model, prompt, 200, temp=temperature, top_k=top_k)
            print("»", decode(out[0].tolist()), "\n")

    # --------- gradient accumulation ---------
    opt.zero_grad(set_to_none=True)
    for _ in range(grad_accum):
        xb, yb = get_batch("train")
        loss = F.cross_entropy(model(xb).view(-1, vocab_size),
                               yb.view(-1)) / grad_accum
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    opt.step()

    # --------- LR scheduler ---------
    for g in opt.param_groups:
        g['lr'] = lr_now(update+1)

    update += 1

# ─────────────── generate ───────────────
prompt = torch.tensor([[stoi[' ']]], device=device)
gen = sample(model, prompt, sample_len,
             temp=temperature, top_k=top_k)[0].tolist()
print("\n" + decode(gen))
