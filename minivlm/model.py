import math
import torch
import torch.nn as nn
import torch.nn.functional as F

TOKENS = [
    '<pad>', '<bos>', '<sep>', '<eos>',
    'what', 'color', 'is', 'the', 'object', '?', 'shape', 'where',
    'describe', 'image', '.', 'a', 'red', 'blue', 'green', 'yellow',
    'circle', 'square', 'triangle', 'on', 'left', 'center', 'right',
]
ITOS = TOKENS
STOI = {s: i for i, s in enumerate(TOKENS)}
COLORS = ('red', 'blue', 'green', 'yellow')
SHAPES = ('circle', 'square', 'triangle')
POSITIONS = ('left', 'center', 'right')
QUESTION_TOKENS = {
    'color': 'what color is the object ?',
    'shape': 'what shape is the object ?',
    'position': 'where is the object ?',
    'describe': 'describe the image .',
}


def prompt_tokens(task):
    return [STOI['<bos>']] + [STOI[t] for t in QUESTION_TOKENS[task].split()] + [STOI['<sep>']]


def answer_tokens(task, color, shape, position):
    if task == 'color':
        words = [color, '.']
    elif task == 'shape':
        words = [shape, '.']
    elif task == 'position':
        words = [position, '.']
    else:
        words = ['a', color, shape, 'on', 'the', position, '.']
    return [STOI[t] for t in words] + [STOI['<eos>']]


class TinyBlock(nn.Module):
    """Single pre-norm masked multi-head self-attention transformer block."""
    def __init__(self, d_model=64, heads=4, expansion=2):
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.attn_out = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * expansion)
        self.fc2 = nn.Linear(d_model * expansion, d_model)

    def forward(self, x):
        b, t, d = x.shape
        normalized = self.ln1(x)
        q, k, v = self.qkv(normalized).chunk(3, dim=-1)
        h = self.heads
        dh = d // h
        q = q.view(b, t, h, dh).transpose(1, 2)
        k = k.view(b, t, h, dh).transpose(1, 2)
        v = v.view(b, t, h, dh).transpose(1, 2)
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
        logits = logits.masked_fill(mask, -1e4)
        attn = torch.softmax(logits, dim=-1)
        y = (attn @ v).transpose(1, 2).contiguous().view(b, t, d)
        x = x + self.attn_out(y)
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate='tanh'))
        return x


class MiniVLM(nn.Module):
    def __init__(self, d_model=64, heads=4, max_tokens=24):
        super().__init__()
        self.d_model = d_model
        self.max_tokens = max_tokens
        self.vision = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.visual_projector = nn.Linear(32 * 4 * 4, d_model)
        self.embedding = nn.Embedding(len(TOKENS), d_model)
        self.position_embedding = nn.Embedding(max_tokens + 1, d_model)
        self.block = TinyBlock(d_model, heads)
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, len(TOKENS))

    def forward(self, images, input_ids):
        bs, length = input_ids.shape
        if length > self.max_tokens:
            raise ValueError('Prompt exceeds maximum token length')
        features = self.vision(images).flatten(1)
        image_emb = self.visual_projector(features).unsqueeze(1)
        token_emb = self.embedding(input_ids)
        x = torch.cat((image_emb, token_emb), dim=1)
        pos = self.position_embedding(torch.arange(length + 1, device=x.device))
        x = x + pos.unsqueeze(0)
        x = self.block(x)
        logits = self.lm_head(self.final_norm(x[:, 1:]))
        return logits

    @torch.no_grad()
    def generate(self, image, task='describe', max_new_tokens=10):
        self.eval()
        seq = prompt_tokens(task)
        for _ in range(max_new_tokens):
            x = torch.tensor(seq, dtype=torch.long, device=image.device).unsqueeze(0)
            logits = self(image.unsqueeze(0), x)
            nxt = int(torch.argmax(logits[0, -1]))
            if nxt == STOI['<eos>']:
                break
            seq.append(nxt)
        ans = [ITOS[n] for n in seq[len(prompt_tokens(task)):]]
        return ' '.join(ans).replace(' .', '.').replace(' ?', '?')
