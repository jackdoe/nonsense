#!/usr/bin/env python3
# schrodinger_attn_tinyshakespeare.py  (MPS-safe)

import os, time, urllib.request
import torch, torch.nn as nn, torch.nn.functional as F

# ─────────────── hyper-params ───────────────
d_model     = 128
n_layers    = 4
n_heads     = 4
mlp_ratio   = 4
block_size  = 64
batch_size  = 32
max_iters   = 20_000
eval_int    = 1_000
lr          = 3e-3
temperature = 0.8
top_k       = 50
sample_len  = 600
# ─────────────────────────────────────────────

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
train_ds, val_ds = torch.split(
    data, [int(0.9*len(data)), len(data)-int(0.9*len(data))])

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
        self.fc1 = nn.Linear(d, r*d)
        self.fc2 = nn.Linear(r*d, d)
        self.act = nn.GELU()
    def forward(self, x):          # x real
        return self.fc2(self.act(self.fc1(x)))

class SchrodingerBlock(nn.Module):
    def __init__(self, d, n_heads, mlp_ratio):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.A      = nn.Parameter(torch.randn(d, d))
        self.log_dt = nn.Parameter(torch.zeros(()))
        self.mlp    = FeedForward(d, mlp_ratio)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ln3 = nn.LayerNorm(d)

    def _unitary(self, h_c):       # h_c complex
        H = 0.5 * (self.A + self.A.T)
        U = torch.matrix_exp(-1j * H * self.log_dt.exp())
        return h_c + (h_c @ U)

    def forward(self, h_c):                                # h_c complex  (B,T,d)
        B,T,_ = h_c.shape

        # 1) causal self-attention on REAL part
        h_real = self.ln1(h_c.real)                        # (B,T,d) real
        mask = torch.triu(torch.full((T, T), float("-inf"),
                                     device=h_c.device), 1)
        attn_out,_ = self.attn(h_real, h_real, h_real,
                               attn_mask=mask)
        h_c = h_c + attn_out.to(torch.cfloat)              # back to complex

        # 2) unitary feature mixer (LayerNorm on real channel!)
        h_c = self._unitary(self.ln2(h_c.real).to(torch.cfloat))

        # 3) small MLP (again only real, then cast)
        h_c = h_c + self.mlp(self.ln3(h_c.real)).to(torch.cfloat)
        return h_c

class TinySchrodingerLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            SchrodingerBlock(d_model, n_heads, mlp_ratio)
            for _ in range(n_layers))
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
    def forward(self, idx):
        h_c = self.embed(idx).to(torch.cfloat)
        for blk in self.blocks:
            h_c = blk(h_c)
        return self.head(self.ln(h_c.real))     # logits (B,T,V)

# ─────────────── sampler ───────────────
@torch.no_grad()
def sample(model, idx, steps, temp=1.0, top_k=None):
    model.eval()
    for _ in range(steps):
        logits = model(idx)[:, -1, :] / temp
        if top_k:
            v,_ = torch.topk(logits, k=top_k)
            logits[logits < v[:,[-1]]] = -float("Inf")
        probs = F.softmax(logits, -1)
        next_id = torch.multinomial(probs, 1)
        idx = torch.cat([idx, next_id], 1)
    model.train();  return idx

# ─────────────── train ───────────────
torch.manual_seed(0)
model = TinySchrodingerLM().to(device)
opt   = torch.optim.AdamW(model.parameters(), lr=lr)

def est_loss():
    model.eval(); out={}
    with torch.no_grad():
        for split in ("train","val"):
            losses=[]
            for _ in range(8):
                xb,yb = get_batch(split)
                loss  = F.cross_entropy(model(xb).view(-1,vocab_size),
                                        yb.view(-1))
                losses.append(loss.item())
            out[split]=sum(losses)/len(losses)
    model.train(); return out

# ───────── training loop ─────────
t0 = time.time()
for step in range(max_iters + 1):
    if step % eval_int == 0:
        l = est_loss()
        mins = (time.time() - t0) / 60
        print(f"step {step:5d} | train {l['train']:.3f}"
              f" | val {l['val']:.3f} | {mins:.1f} min")

        # ── NEW: quick sample after every eval_int steps ──
        with torch.no_grad():
            prompt = torch.tensor([[stoi[" "]]], device=device)
            out = sample(model, prompt, 200,          # 200 new chars
                         temp=temperature, top_k=top_k)
            print("»", decode(out[0].tolist()), "\n")
        # ────────────────────────────────────────────────

    xb, yb = get_batch("train")
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(xb).view(-1, vocab_size), yb.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

# ─────────────── generate ───────────────
prompt = torch.tensor([[stoi[" "]]], device=device)
gen = sample(model, prompt, sample_len,
             temp=temperature, top_k=top_k)[0].tolist()
print("\n"+decode(gen))
