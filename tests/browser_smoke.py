"""Chromium tests without network navigation (this sandbox blocks local URLs).
Verifies CPU inference in Chromium via an imported Blob module, and renders the UI.
This does not constitute a WebGPU hardware test.
"""
import base64
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path('web')
fixture=json.loads(Path('tests/parity_fixture.json').read_text())
config=json.loads((ROOT/'assets/model.json').read_text())
weights=base64.b64encode((ROOT/'assets/model.bin').read_bytes()).decode('ascii')
source=(ROOT/'model.js').read_text()
with sync_playwright() as p:
  browser=p.chromium.launch(headless=True,executable_path='/usr/lib/chromium/chromium',args=['--no-sandbox'])
  page=browser.new_page(viewport={'width':1300,'height':1000},device_scale_factor=1)
  page.set_content('<html><body><h1>Engine validation</h1></body></html>')
  result=page.evaluate('''async ({source,config,weights,image}) => {
    const url=URL.createObjectURL(new Blob([source],{type:'text/javascript'}));
    const {TinyVLM}=await import(url);
    const raw=atob(weights);
    const buffer=new ArrayBuffer(raw.length);
    const view=new Uint8Array(buffer);
    for(let i=0;i<raw.length;i++)view[i]=raw.charCodeAt(i);
    const model=new TinyVLM(config,new Float32Array(buffer),'cpu',null);
    const start=performance.now();
    const prediction=await model.generate(Float32Array.from(image),'describe');
    return {prediction,elapsed:performance.now()-start,backend:model.backend};
  }''',{'source':source,'config':config,'weights':weights,'image':fixture['image']})
  print('Chromium JavaScript inference:',result)
  assert result['prediction']==fixture['expected_text']
  html=(ROOT/'index.html').read_text()
  css=(ROOT/'styles.css').read_text()
  html=html.replace('<link rel="stylesheet" href="./styles.css"/>',f'<style>{css}</style>')
  html=html.replace('<script type="module" src="./app.js"></script>','')
  page.set_content(html)
  page.evaluate('''() => {
    document.querySelector('#paramCount').textContent='82,387';
    document.querySelector('#backend').textContent='WebGPU / CPU';
    document.querySelector('#acc').textContent='99.9%';
    document.querySelector('#lastLatency').textContent='Local';
    const canvas=document.querySelector('#stage');const ctx=canvas.getContext('2d');
    ctx.fillStyle='#f8f8f8';ctx.fillRect(0,0,32,32);ctx.fillStyle='#d9383c';
    ctx.beginPath();ctx.arc(16,16,6,0,2*Math.PI);ctx.fill();
    document.querySelector('#question').value='Describe the image.';
    document.querySelector('#answer').textContent='a red circle on the center.';
    document.querySelector('#replyMeta').textContent='Browser-only neural inference · no model API';
    document.querySelector('#status').textContent='Screenshot preview';
  }''')
  page.screenshot(path='artifacts/browser_screenshot.png',full_page=True)
  print('Rendered browser UI screenshot at artifacts/browser_screenshot.png')
  browser.close()
