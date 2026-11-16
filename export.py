"""Export custom float32 weights for dependency-free browser CPU / WebGPU inference."""
import json
from pathlib import Path
import torch
import numpy as np
from minivlm.model import MiniVLM,TOKENS,QUESTION_TOKENS

ckpt=torch.load('artifacts/model.pt',map_location='cpu',weights_only=False)
model=MiniVLM(**ckpt['config']); model.load_state_dict(ckpt['model']); model.eval()
weights={}; offset=0; arrays=[]
for key,param in model.state_dict().items():
    data=param.detach().numpy().astype('<f4').ravel()
    weights[key]={'offset':offset,'length':len(data),'shape':list(param.shape)}
    arrays.append(data)
    offset+=len(data)
root=Path('web/assets');root.mkdir(parents=True,exist_ok=True)
np.concatenate(arrays).tofile(root/'model.bin')
config={**ckpt['config'],'tokens':TOKENS,'questions':QUESTION_TOKENS,'weights':weights,'parameters':offset,'validation_token_accuracy':ckpt.get('val_token_accuracy'),'training_checkpoint_epoch':ckpt.get('epoch')}
(root/'model.json').write_text(json.dumps(config,indent=2))
print(f'Exported {offset:,} parameters ({offset*4/1024:.1f} KB), checkpoint epoch {ckpt.get("epoch")}')
