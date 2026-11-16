"""Check trained model against actual browser Canvas rendering across all 36 combinations."""
import base64,json
from pathlib import Path
from playwright.sync_api import sync_playwright
root=Path('web')
params={'source':(root/'model.js').read_text(),
'config':json.loads((root/'assets/model.json').read_text()),
'weights':base64.b64encode((root/'assets/model.bin').read_bytes()).decode('ascii')}
with sync_playwright() as p:
 b=p.chromium.launch(headless=True,executable_path='/usr/lib/chromium/chromium',args=['--no-sandbox'])
 page=b.new_page();page.set_content('<html><body>test</body></html>')
 result=page.evaluate('''async ({source,config,weights})=>{
  const url=URL.createObjectURL(new Blob([source],{type:'text/javascript'}));
  const {TinyVLM}=await import(url);
  const raw=atob(weights);const buf=new ArrayBuffer(raw.length);const dest=new Uint8Array(buf);
  for(let k=0;k<dest.length;k++)dest[k]=raw.charCodeAt(k);
  const model=new TinyVLM(config,new Float32Array(buf),'cpu');
  const canvas=document.createElement('canvas');canvas.width=32;canvas.height=32;
  const ctx=canvas.getContext('2d',{willReadFrequently:true});
  const colors={red:'#d9383c',blue:'#3069db',green:'#25a168',yellow:'#e7b023'};
  const shapes=['circle','square','triangle'],positions=['left','center','right'];
  const acc={color:0,shape:0,position:0,describe:0};const failures=[];
  for(const [color,rgb] of Object.entries(colors))for(const shape of shapes)for(const position of positions){
   ctx.fillStyle='#f8f8f8';ctx.fillRect(0,0,32,32);
   let x={left:9,center:16,right:23}[position],y=16,r=6;ctx.fillStyle=rgb;
   if(shape==='circle'){ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fill();}
   else if(shape==='square'){ctx.fillRect(x-r,y-r,r*2+1,r*2+1);}
   else{ctx.beginPath();ctx.moveTo(x,y-r-1);ctx.lineTo(x-r-1,y+r);ctx.lineTo(x+r+1,y+r);ctx.closePath();ctx.fill();}
   const image=TinyVLM.imageDataFromImageData(ctx.getImageData(0,0,32,32));
   for(const task of Object.keys(acc)){
     const result=await model.generate(image,task);
     const wanted= task==='color'?color+'.':task==='shape'?shape+'.':task==='position'?position+'.':`a ${color} ${shape} on the ${position}.`;
     if(result===wanted)acc[task]++;
     else failures.push({color,shape,position,task,result,wanted});
   }
  }
  return {acc,failures,totalPerTask:36};
}''',params)
 print(json.dumps(result,indent=2))
 Path('artifacts/canvas_generalization.json').write_text(json.dumps(result,indent=2))
 b.close()
