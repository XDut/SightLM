"""Run autoregressive evaluation on a held-out synthetic image set."""
import json
import time
from collections import defaultdict
import torch
from minivlm.model import MiniVLM, COLORS, SHAPES, POSITIONS
from minivlm.data import ShapeQADataset, TASKS

ckpt=torch.load('artifacts/model.pt',weights_only=False,map_location='cpu')
model=MiniVLM(**ckpt['config']);model.load_state_dict(ckpt['model']);model.eval()
torch.set_num_threads(3)
dataset=ShapeQADataset(80,seed=90031)
correct=defaultdict(int);total=defaultdict(int);failures=[]
start=time.monotonic()
for i,image in enumerate(dataset.images):
  color,shape,pos=dataset.records[i]
  for task in TASKS:
    pred=model.generate(image,task)
    desired={
      'color':color+'.',
      'shape':shape+'.',
      'position':pos+'.',
      'describe':f'a {color} {shape} on the {pos}.',
    }[task]
    correct[task]+=(pred==desired)
    total[task]+=1
    if pred!=desired and len(failures)<10:failures.append({'expected':desired,'got':pred,'task':task})
report={'greedy_exact_match':{task:correct[task]/total[task] for task in TASKS},'correct':dict(correct),'total':dict(total),'failures':failures,'examples':len(dataset.images),'elapsed_s':round(time.monotonic()-start,2)}
open('artifacts/evaluation.json','w').write(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
