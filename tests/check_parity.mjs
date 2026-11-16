import fs from 'node:fs';
import assert from 'node:assert/strict';
import { TinyVLM } from '../web/model.js';
const config=JSON.parse(fs.readFileSync('web/assets/model.json','utf8'));
const data=fs.readFileSync('web/assets/model.bin');
const weights=new Float32Array(data.buffer,data.byteOffset,data.byteLength/4);
const model=new TinyVLM(config,weights,'cpu',null);
const fixture=JSON.parse(fs.readFileSync('tests/parity_fixture.json','utf8'));
function difference(a,b){
  let max=0,mean=0;
  for(let i=0;i<a.length;i++) {const v=Math.abs(a[i]-b[i]);if(v>max)max=v;mean+=v;}
  return {max,mean:mean/a.length};
}
const emb=await model.encode(Float32Array.from(fixture.image));
const logits=await model.forward(fixture.tokens,emb);
const d1=difference(emb,fixture.image_embedding);
const d2=difference(logits,fixture.last_logits);
console.log('Visual embedding PyTorch↔JavaScript',d1);
console.log('Last-token logits PyTorch↔JavaScript',d2);
assert.ok(d1.max<0.005,'Vision encoder output parity is unacceptable');
assert.ok(d2.max<0.02,'Full decoder output parity is unacceptable');
const prediction=await model.generate(Float32Array.from(fixture.image),'describe');
console.log('JS prediction:',prediction,'PyTorch prediction:',fixture.expected_text);
assert.equal(prediction,fixture.expected_text);
console.log('PASS: CPU inference faithfully reproduces PyTorch checkpoint.');
