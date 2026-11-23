#pre-training stage of a LLM
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#hyperparameters
batch_size = 64 #32    #how many independent sequences of block we process in parallel
block_size = 256 #8     #max context length for prediction
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4 #1e-3
eval_iters = 200
n_embd = 384 #32
num_heads = 6 #4    as 364/6 = 64 (the block size)
n_block_layer = 6
dropout = 0.2

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text)))     # to get all characters there are!
vocab_size = len(chars)
# a simple mapping from characters to integers[0-64]
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s : [stoi[c] for c in s]
decode = lambda l : ''.join([itos[i] for i in l])

# encode the dataset
data = torch.tensor(encode(text))
data = data.to(device)
# spliting into train and val sets
n = int(0.9 * len(data))
train_set = data[:n]
val_set = data[n:]


def get_batch(split):
    data = train_set if split == 'train' else val_set
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i+1 : i + block_size+1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            losses[k] = loss
        out[split] = losses.mean()
    model.train()
    return out


# single self-attention head of a decoder block of transformer
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)    #(B, T, C)
        k = self.key(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5                      #(B, T, T) 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) 
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v      #(B, T, T) * (B, T, C) -> (B, T, C)
        return out

#multi head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # projection is the linear transformation of the cat and we return the projection as we are adding it to the x (residual connection)
        out = self.proj(out)
        out = self.dropout(out)
        return out


#feed-forward block
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),   # in paper d_model is 512 and d_ff is 2048 so 4x of n_embd
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),   
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, num_heads, n_embd):
        super().__init__()
        head_size = n_embd // num_heads
        self.sa_heads = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # different from original paper used now a days
        x = x + self.sa_heads(self.ln1(x))  #skip connections!
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) #embedding table
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)
        #self.sa_head = Head(n_embd)
        #self.sa_heads = MultiHeadAttention(4, n_embd // 4)    # 4 heads of 8 dim self-attention (4 of 8d vectors concats to give out our n_emb size!)
        #self.ffwd = FeedForward(n_embd)
        #made a Block out of it to run multiple blocks of attention
        self.blocks = nn.Sequential( #single loop for n_layer hyperparameter!
            *[Block(num_heads, n_embd) for _ in range(n_block_layer)]
            # Block(4, n_embd),
            # Block(4, n_embd),
            # Block(4, n_embd),
        )
        self.ln_f = nn.LayerNorm(n_embd)             # right before it goes into the lm_head linear layer or you can say after ffwd
        self.lm_head = nn.Linear(n_embd, vocab_size) # linear layer


    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  #(B, T, C)
        pos_emb = self.positional_embedding_table(torch.arange(T, device=device))  #(T, C)
        x = tok_emb + pos_emb    #(B, T, C)
        #x = self.sa_head(x)      #single self-attention head
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)   #(B, C, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_token):
        for _ in range(max_new_token):
            # crop idx to the last block size token as we are using pos_emb we can never have more than block size comming in 
            # or else pos_emb will run out of scope as it only have emb upto block_size
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BigramLanguageModel()
model = model.to(device)
opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for i in range(max_iters):
    if i % 500 == 0:
        losses = estimate_loss()
        print(
            f"step {i}: "
            f"train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}"
        )
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    opt.zero_grad()
    loss.backward()
    opt.step()

idx = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(idx, 500)[0].tolist()))


# we only used decoder part as thats our need the original paper was a translation paper so it has a encoder and a decoder, they are allowed to talk to each other as 
# much as they want  key and value comming from encoder to the decoder!



# step 0: train loss 4.3457, val loss 4.3334
# step 500: train loss 1.8951, val loss 2.0024
# step 1000: train loss 1.5447, val loss 1.7340
# step 1500: train loss 1.4010, val loss 1.6049
# step 2000: train loss 1.3155, val loss 1.5479
# step 2500: train loss 1.2592, val loss 1.5196
# step 3000: train loss 1.2173, val loss 1.5040
# step 3500: train loss 1.1730, val loss 1.4902
# step 4000: train loss 1.1343, val loss 1.4781
# step 4500: train loss 1.0997, val loss 1.4803


# CAPULET:
# A general o' the more is again, but that one
# That I know you'll drunk: the king's name.

# KING BRAY:
# How may, men?

# NGercedge, you cannot pardon wretch his chert;
# Your merelel, joy take up them and impeach me too.
# Murder'd; you guess, and hire merry and sin.

# NORTHUMBENDOND:
# My life command, my lord, for my father
# For theirs bent you misforwhen you out a garder.

# NORSTUS:
# But load of that, both silking,
# That words may yet take the Duke of Norfolk?--
# The crown talk of Phepets, my lordship

# ~/nanoGPT main* 41m 6s