import { TinyVLM } from './model.js';

const $ = sel => document.querySelector(sel);
const stage=$('#stage');
const context=stage.getContext('2d',{willReadFrequently:true});
const state={model:null,image: null,running:false,gpuPreferred:true};
const RGB={red:'#d9383c',blue:'#3069db',green:'#25a168',yellow:'#e7b023'};
const tasks={
  'What color is the object?':'color',
  'What shape is the object?':'shape',
  'Where is the object?':'position',
  'Describe the image.':'describe',
};

function setStatus(text,type='neutral'){$('#status').textContent=text;$('#status').dataset.type=type;}
function setImage(){
  state.image=TinyVLM.imageDataFromImageData(context.getImageData(0,0,32,32));
  $('#imageStatus').textContent='Image ready · 32 × 32 model input';
  $('#askButton').disabled=!state.model;
}
function renderDemo(){
  // Sample controls are removed; each click produces a different random sample.
  const colors=Object.keys(RGB), shapes=['circle','square','triangle'], positions=['left','center','right'];
  let color, shape, position, key;
  do {
    color=colors[Math.floor(Math.random()*colors.length)];
    shape=shapes[Math.floor(Math.random()*shapes.length)];
    position=positions[Math.floor(Math.random()*positions.length)];
    key=`${color}:${shape}:${position}`;
  } while (key===state.lastSample);
  state.lastSample=key;
  context.fillStyle='#f8f8f8';context.fillRect(0,0,32,32);
  const x={left:9,center:16,right:23}[position],y=16,r=6;
  context.fillStyle=RGB[color];
  if(shape==='circle'){context.beginPath();context.arc(x,y,r,0,Math.PI*2);context.fill();}
  else if(shape==='square'){context.fillRect(x-r,y-r,r*2+1,r*2+1);}
  else {context.beginPath();context.moveTo(x,y-r-1);context.lineTo(x-r-1,y+r);context.lineTo(x+r+1,y+r);context.closePath();context.fill();}
  $('#stageWrap').dataset.uploaded='false';
  setImage();
}
function setTask(task){
  const [question]=Object.entries(tasks).find(([_,t])=>t===task);
  $('#question').value=question;
  document.querySelectorAll('[data-task]').forEach(btn=>btn.classList.toggle('active',btn.dataset.task===task));
}
function getQuestionTask(q){
  const text=q.trim().toLowerCase();
  if(/\bcolou?r\b/.test(text))return 'color';
  if(/\bshape\b|\bcircle\b|\bsquare\b|\btriangle\b/.test(text))return 'shape';
  if(/\bwhere\b|\bposition\b|\blocation\b/.test(text))return 'position';
  if(/\bdescribe\b|\bwhat.*(image|picture)\b|\bwhat.*see\b|\bcaption\b/.test(text))return 'describe';
  return null;
}
function setResponse(text,meta){$('#answer').textContent=text;$('#replyMeta').textContent=meta;}

async function boot(){
  setStatus('Loading trained model…');
  try{
    state.model=await TinyVLM.load('./assets/','webgpu');
    const m=state.model;
    const mode=m.backend==='webgpu'?'WebGPU':'CPU / JavaScript';
    $('#backend').textContent=mode;
    $('#paramCount').textContent=m.config.parameters.toLocaleString();
    $('#acc').textContent=`${(m.config.validation_token_accuracy*100).toFixed(1)}%`;
    setStatus(`Model loaded · ${mode}`,'good');
    $('#askButton').disabled=false;
    $('#reloadButton').disabled=false;
  }catch(e){
    console.error(e);setStatus('Model failed to load','error');
    setResponse('Could not load local model assets. Start the static server and refresh.','Check console for details.');
  }
}

async function answer(){
  if(state.running||!state.model||!state.image)return;
  const task=getQuestionTask($('#question').value);
  if(!task){
    setResponse('This miniature model supports only four kinds of question: color, shape, position, and description.','Limited domain: one colored geometric shape on a light background.');
    return;
  }
  state.running=true;
  $('#askButton').disabled=true;
  setResponse('Thinking…','Performing neural inference locally in your browser');
  await new Promise(resolve=>setTimeout(resolve,40));
  try{
    const start=performance.now();
    const result=await state.model.generate(state.image,task,partial=>{$('#answer').textContent=partial+'▍';});
    const elapsed=performance.now()-start;
    setResponse(result||'(empty response)',`${elapsed.toFixed(0)} ms · ${state.model.backend.toUpperCase()} · ${state.model.lastTimings.outputTokens} generated tokens · no cloud API`);
    $('#lastLatency').textContent=`${elapsed.toFixed(0)} ms`;
  }catch(e){
    console.error(e);setResponse('Inference error: '+e.message,'See developer console.');
  }finally{
    state.running=false;$('#askButton').disabled=false;
  }
}

$('#drawButton').addEventListener('click',renderDemo);
$('#upload').addEventListener('change',async e=>{
  const file=e.target.files?.[0];if(!file)return;
  const bitmap=await createImageBitmap(file);
  context.fillStyle='white';context.fillRect(0,0,32,32);
  const ratio=Math.min(32/bitmap.width,32/bitmap.height);
  const w=bitmap.width*ratio,h=bitmap.height*ratio;
  context.drawImage(bitmap,(32-w)/2,(32-h)/2,w,h);
  bitmap.close();
  $('#stageWrap').dataset.uploaded='true';
  setImage();
  $('#imageStatus').textContent='Custom image · out-of-domain photos may fail';
});
$('#askButton').addEventListener('click',answer);
$('#question').addEventListener('keydown',e=>{if(e.key==='Enter')answer();});
document.querySelectorAll('[data-task]').forEach(btn=>btn.addEventListener('click',()=>setTask(btn.dataset.task)));
$('#reloadButton').addEventListener('click',async()=>{
  const desired=state.model.backend==='webgpu'?'cpu':'webgpu';
  $('#reloadButton').disabled=true;setStatus('Switching inference backend…');
  try{
    state.model=await TinyVLM.load('./assets/',desired);
    $('#backend').textContent=state.model.backend==='webgpu'?'WebGPU':'CPU / JavaScript';
    setStatus('Backend selected · '+state.model.backend,'good');
  }catch(e){setStatus(e.message,'error');}
  $('#reloadButton').disabled=false;
});
renderDemo();setTask('describe');boot();

// Optional offline support. Static assets are cached after their first successful load.
if ('serviceWorker' in navigator && location.protocol.startsWith('http')) {
  window.addEventListener('load', () => navigator.serviceWorker.register('./sw.js').catch(err => console.warn('Offline cache unavailable:', err)));
}
