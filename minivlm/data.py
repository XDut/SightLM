import random
from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
import torch
from torch.utils.data import Dataset
from .model import COLORS, SHAPES, POSITIONS, STOI, prompt_tokens, answer_tokens

RGB = {
    'red': (217, 56, 60), 'blue': (48, 105, 219),
    'green': (37, 161, 104), 'yellow': (231, 176, 35),
}
TASKS = ('color', 'shape', 'position', 'describe')


def draw_example(color, shape, position, seed=0, size=32):
    """Procedurally generate a labeled image. No external datasets or pretrained weights."""
    rng = random.Random(seed)
    bg = rng.randint(239, 255)
    im = Image.new('RGB', (size, size), (bg, bg, bg))
    draw = ImageDraw.Draw(im)
    centers = {'left': 9, 'center': 16, 'right': 23}
    cx = centers[position] + rng.choice((-1, 0, 0, 1))
    cy = 16 + rng.randint(-3, 3)
    half = rng.choice((5, 6, 7))
    rgb = tuple(max(0, min(255, c + rng.randint(-12, 12))) for c in RGB[color])
    if shape == 'circle':
        draw.ellipse((cx-half, cy-half, cx+half, cy+half), fill=rgb)
    elif shape == 'square':
        draw.rectangle((cx-half, cy-half, cx+half, cy+half), fill=rgb)
    else:
        draw.polygon([(cx,cy-half-1),(cx-half-1,cy+half),(cx+half+1,cy+half)], fill=rgb)
    return im


def image_to_tensor(img):
    arr = np.asarray(img.convert('RGB').resize((32, 32)), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2,0,1).contiguous()


class ShapeQADataset(Dataset):
    def __init__(self, num_images=2000, seed=10000, max_length=24):
        self.max_length = max_length
        rng = random.Random(seed)
        self.records=[]
        self.images=[]
        for i in range(num_images):
            color = rng.choice(COLORS)
            shape = rng.choice(SHAPES)
            position = rng.choice(POSITIONS)
            image = draw_example(color, shape, position, seed=seed + i*13)
            self.images.append(image_to_tensor(image))
            self.records.append((color, shape, position))
        self.images = torch.stack(self.images)

    def __len__(self):
        return len(self.images) * len(TASKS)

    def __getitem__(self, idx):
        imageidx, qidx = divmod(idx, len(TASKS))
        color,shape,pos = self.records[imageidx]
        task = TASKS[qidx]
        seq = prompt_tokens(task) + answer_tokens(task, color, shape, pos)
        inp = seq[:-1]
        target = seq[1:]
        prompt_len = len(prompt_tokens(task))
        labels = [-100] * (prompt_len-1) + target[prompt_len-1:]
        assert len(labels)==len(inp)
        return (self.images[imageidx],
                torch.tensor(inp + [STOI['<pad>']]*(self.max_length-len(inp)), dtype=torch.long),
                torch.tensor(labels + [-100]*(self.max_length-len(labels)),dtype=torch.long))
