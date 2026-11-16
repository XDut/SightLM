import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from minivlm.model import MiniVLM, TOKENS
from minivlm.data import ShapeQADataset


def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum, steps=0,0,0.0,0
    with torch.no_grad():
        for image, ids, target in loader:
            image,ids,target = image.to(device),ids.to(device),target.to(device)
            logits = model(image,ids)
            valid = target != -100
            correct += ((logits.argmax(-1) == target) & valid).sum().item()
            total += valid.sum().item()
            loss_sum += F.cross_entropy(logits.reshape(-1,logits.size(-1)),target.reshape(-1),ignore_index=-100).item()
            steps += 1
    return {'token_accuracy':correct/max(total,1), 'loss':loss_sum/max(steps,1)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=17)
    parser.add_argument('--train-images',type=int,default=1600)
    parser.add_argument('--val-images',type=int,default=200)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--lr',type=float,default=0.0015)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--output',default='artifacts/model.pt')
    args=parser.parse_args()
    torch.set_num_threads(min(4,torch.get_num_threads()))
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    device = 'cuda' if (args.device == 'auto' and torch.cuda.is_available()) else ('cpu' if args.device=='auto' else args.device)
    trainset=ShapeQADataset(args.train_images,seed=10000)
    valset=ShapeQADataset(args.val_images,seed=78000)
    trainloader=DataLoader(trainset,batch_size=args.batch_size,shuffle=True,num_workers=0)
    valloader=DataLoader(valset,batch_size=args.batch_size,num_workers=0)
    model=MiniVLM().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.003)
    param_count=sum(p.numel() for p in model.parameters())
    print('device:',device,'parameters:',param_count,'train pairs:',len(trainset),'val pairs:',len(valset),flush=True)
    best=0
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    history=[]
    for epoch in range(args.epochs):
        start=time.monotonic()
        model.train()
        running=0
        for image,ids,target in trainloader:
            image,ids,target=image.to(device),ids.to(device),target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits=model(image,ids)
            loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),target.reshape(-1),ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            running+=loss.item()
        val=evaluate(model,valloader,device)
        history.append({'epoch':epoch+1,'train_loss':running/len(trainloader),**val})
        print(f'epoch {epoch+1:02d}: train_loss={running/len(trainloader):.4f} val_loss={val["loss"]:.4f} val_token_acc={val["token_accuracy"]:.3f} elapsed={time.monotonic()-start:.1f}s',flush=True)
        if val['token_accuracy'] > best:
            best=val['token_accuracy']
            torch.save({'model':model.state_dict(),'config':{'d_model':64,'heads':4,'max_tokens':24},'tokens':TOKENS, 'epoch':epoch+1, 'val_token_accuracy':best}, args.output)
    report={'parameters':param_count,'training_images':args.train_images,'validation_images':args.val_images,'train_pairs':len(trainset),'validation_pairs':len(valset),'best_validation_token_accuracy':best,'history':history,'training_device':device}
    Path('artifacts/training_metrics.json').write_text(json.dumps(report,indent=2))
    print('Saved model to',args.output, 'best validation token accuracy:',best,flush=True)

if __name__=='__main__':main()
