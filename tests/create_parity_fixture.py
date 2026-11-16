import json
import torch
from minivlm.model import MiniVLM, prompt_tokens
from minivlm.data import draw_example,image_to_tensor
ckpt=torch.load('artifacts/model.pt',weights_only=False,map_location='cpu')
model=MiniVLM(**ckpt['config']);model.load_state_dict(ckpt['model']);model.eval()
image=image_to_tensor(draw_example('red','triangle','left',seed=345))
tokens=prompt_tokens('describe')
with torch.no_grad():
    conv=model.vision(image.unsqueeze(0)).flatten(1)
    image_embed=model.visual_projector(conv)
    logits=model(image.unsqueeze(0),torch.tensor(tokens).unsqueeze(0))
fixture={'image':image.flatten().tolist(),'tokens':tokens,'image_embedding':image_embed[0].tolist(),'last_logits':logits[0,-1].tolist(),'expected_text':model.generate(image,task='describe')}
open('tests/parity_fixture.json','w').write(json.dumps(fixture))
print('Fixture expected:',fixture['expected_text'])
