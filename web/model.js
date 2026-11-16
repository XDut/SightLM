/**
 * TinyVLM browser runtime. No WebLLM, ONNX Runtime, libraries, Python servers or APIs.
 * Supports pure JavaScript CPU inference and optional custom WebGPU compute shaders.
 * Model architecture is identical to minivlm/model.py.
 */
const GELU_FACTOR = Math.sqrt(2 / Math.PI);
const EPS = 1e-5;

export class TinyVLM {
  constructor(config, weights, backend = 'cpu', device = null) {
    this.config = config;
    this.floatWeights = weights;
    this.backend = backend;
    this.device = device;
    this.layers = {};
    this.gpu = null;
    this.params = config.weights;
    this.tokens = config.tokens;
    this.stoi = Object.fromEntries(config.tokens.map((token, i) => [token, i]));
    this.lastTimings = {};
  }

  static async load(baseUrl = './assets/', preferredBackend = 'webgpu') {
    const [meta, data] = await Promise.all([
      fetch(new URL('model.json', new URL(baseUrl, import.meta.url))).then(r => { if (!r.ok) throw Error(`Cannot load model.json: ${r.status}`); return r.json(); }),
      fetch(new URL('model.bin', new URL(baseUrl, import.meta.url))).then(r => { if (!r.ok) throw Error(`Cannot load model.bin: ${r.status}`); return r.arrayBuffer(); }),
    ]);
    let backend = 'cpu', device = null;
    if (preferredBackend === 'webgpu' && typeof navigator !== 'undefined' && navigator.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        device = await adapter?.requestDevice();
        if (device) backend = 'webgpu';
      } catch (e) { console.warn('WebGPU unavailable; falling back to CPU', e); }
    }
    const model = new TinyVLM(meta, new Float32Array(data), backend, device);
    if (device) {
      try { model.gpu = new GPUOps(device, model); }
      catch (e) { console.warn('WebGPU setup failed; CPU fallback', e); model.backend = 'cpu'; }
    }
    return model;
  }

  weight(key) {
    let result = this.layers[key];
    if (!result) {
      const spec = this.params[key];
      if (!spec) throw Error(`Unknown tensor: ${key}`);
      result = this.floatWeights.subarray(spec.offset, spec.offset + spec.length);
      this.layers[key] = result;
    }
    return result;
  }

  layerNorm(x, rows, width, prefix) {
    const gamma = this.weight(prefix + '.weight');
    const beta = this.weight(prefix + '.bias');
    const out = new Float32Array(x.length);
    for (let r=0; r<rows; r++) {
      let sum=0, sq=0;
      for (let k=0;k<width;k++) sum += x[r*width+k];
      const avg = sum/width;
      for (let k=0;k<width;k++) { const d=x[r*width+k]-avg; sq+=d*d; }
      const inv = 1/Math.sqrt(sq/width+EPS);
      for (let k=0;k<width;k++) out[r*width+k] = ((x[r*width+k]-avg)*inv)*gamma[k]+beta[k];
    }
    return out;
  }

  async dense(x, n, prefix) {
    const shape = this.params[prefix+'.weight'].shape;
    if (this.backend === 'webgpu' && this.gpu) {
      try { return await this.gpu.dense(x, n, prefix, shape); }
      catch(e) { console.warn('WebGPU operation failed; CPU fallback:',e); this.backend='cpu'; }
    }
    const [outDim, inDim] = shape;
    const w = this.weight(prefix+'.weight'), b = this.weight(prefix+'.bias');
    const out = new Float32Array(n * outDim);
    for (let row=0;row<n;row++) {
      const base=row*inDim;
      for (let o=0;o<outDim;o++) {
        let acc=b[o];const wb=o*inDim;
        for(let k=0;k<inDim;k++) acc+=x[base+k]*w[wb+k];
        out[row*outDim+o]=acc;
      }
    }
    return out;
  }

  async conv(x, height, width, prefix) {
    const shape=this.params[prefix+'.weight'].shape;
    if (this.backend === 'webgpu' && this.gpu) {
      try { return await this.gpu.conv(x,height,width,prefix,shape); }
      catch(e){console.warn('WebGPU convolution failed; CPU fallback:',e);this.backend='cpu';}
    }
    const [outChannels,inChannels,kh,kw] = shape;
    const outH=Math.floor((height+2-3)/2)+1;
    const outW=Math.floor((width+2-3)/2)+1;
    const w=this.weight(prefix+'.weight'),bias=this.weight(prefix+'.bias');
    const out=new Float32Array(outChannels*outH*outW);
    for(let oc=0;oc<outChannels;oc++){
      for(let oy=0;oy<outH;oy++){
        for(let ox=0;ox<outW;ox++){
          let total=bias[oc];
          for(let ic=0;ic<inChannels;ic++){
            for(let ky=0;ky<kh;ky++){
              const iy=oy*2+ky-1;if(iy<0||iy>=height)continue;
              for(let kx=0;kx<kw;kx++){
                const ix=ox*2+kx-1;if(ix<0||ix>=width)continue;
                total+=x[(ic*height+iy)*width+ix]*w[((oc*inChannels+ic)*kh+ky)*kw+kx];
              }
            }
          }
          out[(oc*outH+oy)*outW+ox]=Math.max(0,total);
        }
      }
    }
    return {tensor:out,height:outH,width:outW};
  }

  async encode(imageTensor) {
    let x=imageTensor,h=32,w=32;
    for(const prefix of ['vision.0','vision.2','vision.4']) {
      const layer=await this.conv(x,h,w,prefix);
      x=layer.tensor; h=layer.height; w=layer.width;
    }
    return this.dense(x,1,'visual_projector');
  }

  async forward(tokens, imageEmbedding) {
    const D=this.config.d_model;
    const T=tokens.length+1;
    if(T>this.config.max_tokens+1)throw Error('Sequence exceeds the model context length');
    const embed=this.weight('embedding.weight');
    const pos=this.weight('position_embedding.weight');
    const x=new Float32Array(T*D);
    for(let i=0;i<D;i++)x[i]=imageEmbedding[i]+pos[i];
    for(let t=1;t<T;t++) {
      const id=tokens[t-1];
      for(let i=0;i<D;i++)x[t*D+i]=embed[id*D+i]+pos[t*D+i];
    }
    const ln1=this.layerNorm(x,T,D,'block.ln1');
    const qkv=await this.dense(ln1,T,'block.qkv');
    const H=this.config.heads;
    const Dh=D/H;
    const attention=new Float32Array(T*D);
    const attnScores=new Float32Array(T);
    for(let t=0;t<T;t++){
      for(let head=0;head<H;head++){
        let max=-Infinity;
        for(let prev=0;prev<=t;prev++){
          let dot=0;
          for(let k=0;k<Dh;k++){
            const index=head*Dh+k;
            dot+=qkv[t*3*D+index]*qkv[prev*3*D+D+index];
          }
          const val=dot/Math.sqrt(Dh);
          attnScores[prev]=val;if(val>max)max=val;
        }
        let denominator=0;
        for(let prev=0;prev<=t;prev++){
          attnScores[prev]=Math.exp(attnScores[prev]-max);
          denominator+=attnScores[prev];
        }
        for(let k=0;k<Dh;k++){
          const index=head*Dh+k;
          let total=0;
          for(let prev=0;prev<=t;prev++){
            total+=(attnScores[prev]/denominator)*qkv[prev*3*D+2*D+index];
          }
          attention[t*D+index]=total;
        }
      }
    }
    const projected=await this.dense(attention,T,'block.attn_out');
    for(let i=0;i<x.length;i++)x[i]+=projected[i];
    const ln2=this.layerNorm(x,T,D,'block.ln2');
    const hidden=await this.dense(ln2,T,'block.fc1');
    for(let i=0;i<hidden.length;i++){
      const v=hidden[i];
      hidden[i]=0.5*v*(1+Math.tanh(GELU_FACTOR*(v+0.044715*v*v*v)));
    }
    const feedforward=await this.dense(hidden,T,'block.fc2');
    for(let i=0;i<x.length;i++)x[i]+=feedforward[i];
    const normalized=this.layerNorm(x,T,D,'final_norm');
    const last=normalized.subarray((T-1)*D,T*D);
    return this.dense(last,1,'lm_head');
  }

  async generate(imageTensor, task='describe', onToken=null, maxTokens=10) {
    if(!this.config.questions[task])throw Error(`Unknown question: ${task}`);
    const prompt=['<bos>',...this.config.questions[task].split(' '),'<sep>'];
    const inputIds=prompt.map(s=>this.stoi[s]);
    const t0=performance.now();
    const visual=await this.encode(imageTensor);
    const imageMs=performance.now()-t0;
    const words=[];
    for(let i=0;i<maxTokens;i++){
      const logits=await this.forward(inputIds,visual);
      let idx=0,max=-Infinity;
      for(let k=0;k<logits.length;k++)if(logits[k]>max){max=logits[k];idx=k;}
      if(idx===this.stoi['<eos>'])break;
      words.push(this.tokens[idx]);inputIds.push(idx);
      if(onToken)onToken(this.stringify(words),words.slice());
      if(inputIds.length>=this.config.max_tokens)break;
    }
    const totalMs=performance.now()-t0;
    this.lastTimings={imageMs,generateMs:totalMs-imageMs,totalMs,outputTokens:words.length};
    return this.stringify(words);
  }

  stringify(words){return words.join(' ').replace(/ \./g,'.').replace(/ \?/g,'?');}

  static imageDataFromImageData(imageData) {
    const {data,width,height}=imageData;
    if(width!==32||height!==32)throw Error('Preprocessing expects 32×32 RGB input');
    const out=new Float32Array(3*32*32);
    for(let i=0;i<1024;i++){
      out[i]=data[i*4]/255;
      out[1024+i]=data[i*4+1]/255;
      out[2048+i]=data[i*4+2]/255;
    }
    return out;
  }
}

class GPUOps {
  constructor(device, model){
    this.device=device;this.model=model;
    this.weightBuffers=new Map();
    this.densePipeline=this.makePipeline(`
      struct Params {n:u32,k:u32,m:u32,pad:u32};
      @group(0) @binding(0) var<storage,read> x:array<f32>;
      @group(0) @binding(1) var<storage,read> w:array<f32>;
      @group(0) @binding(2) var<storage,read> bias:array<f32>;
      @group(0) @binding(3) var<storage,read_write> y:array<f32>;
      @group(0) @binding(4) var<uniform> p:Params;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) tid:vec3<u32>){
        let id=tid.x;if(id>=p.n*p.m){return;}
        let r=id/p.m;let c=id%p.m;
        var sum=bias[c];
        for(var j:u32=0u;j<p.k;j=j+1u){sum=sum+x[r*p.k+j]*w[c*p.k+j];}
        y[id]=sum;
      }
    `);
    this.convPipeline=this.makePipeline(`
      struct Params {h:u32,w:u32,ic:u32,oc:u32,oh:u32,ow:u32,pad0:u32,pad1:u32};
      @group(0) @binding(0) var<storage,read> x:array<f32>;
      @group(0) @binding(1) var<storage,read> weights:array<f32>;
      @group(0) @binding(2) var<storage,read> bias:array<f32>;
      @group(0) @binding(3) var<storage,read_write> y:array<f32>;
      @group(0) @binding(4) var<uniform> p:Params;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) tid:vec3<u32>){
        let id=tid.x;if(id>=p.oc*p.oh*p.ow){return;}
        let c=id/(p.oh*p.ow);let oy=(id/p.ow)%p.oh;let ox=id%p.ow;
        var sum=bias[c];
        for(var ic:u32=0u;ic<p.ic;ic=ic+1u){
          for(var ky:u32=0u;ky<3u;ky=ky+1u){
            let iy=i32(oy)*2+i32(ky)-1;
            if(iy<0||iy>=i32(p.h)){continue;}
            for(var kx:u32=0u;kx<3u;kx=kx+1u){
              let ix=i32(ox)*2+i32(kx)-1;
              if(ix<0||ix>=i32(p.w)){continue;}
              let index=(ic*p.h+u32(iy))*p.w+u32(ix);
              let wi=((c*p.ic+ic)*3u+ky)*3u+kx;
              sum=sum+x[index]*weights[wi];
            }
          }
        }
        y[id]=max(0.0,sum);
      }
    `);
  }
  makePipeline(wgsl){
    const shader=this.device.createShaderModule({code:wgsl});
    return this.device.createComputePipeline({layout:'auto',compute:{module:shader,entryPoint:'main'}});
  }
  bufferFrom(data,usage){
    const gpuBuf=this.device.createBuffer({size:Math.max(data.byteLength,4),usage:usage|GPUBufferUsage.COPY_DST});
    this.device.queue.writeBuffer(gpuBuf,0,data.buffer,data.byteOffset,data.byteLength);
    return gpuBuf;
  }
  getStaticWeights(name){
    if(this.weightBuffers.has(name))return this.weightBuffers.get(name);
    const w=this.bufferFrom(this.model.weight(name+'.weight'),GPUBufferUsage.STORAGE);
    const b=this.bufferFrom(this.model.weight(name+'.bias'),GPUBufferUsage.STORAGE);
    const result={w,b};this.weightBuffers.set(name,result);return result;
  }
  async execute(pipeline,input,prefix,paramArray,outputs){
    const usage=GPUBufferUsage.STORAGE;
    const inputBuffer=this.bufferFrom(input,usage);
    const {w,b}=this.getStaticWeights(prefix);
    const paramBuffer=this.bufferFrom(new Uint32Array(paramArray),GPUBufferUsage.UNIFORM);
    const outBuffer=this.device.createBuffer({size:outputs*4,usage:usage|GPUBufferUsage.COPY_SRC});
    const readBuffer=this.device.createBuffer({size:outputs*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    const entries=[inputBuffer,w,b,outBuffer,paramBuffer].map((buffer,binding)=>({binding,resource:{buffer}}));
    const bind=this.device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries});
    const encoder=this.device.createCommandEncoder();
    const pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,bind);pass.dispatchWorkgroups(Math.ceil(outputs/64));pass.end();
    encoder.copyBufferToBuffer(outBuffer,0,readBuffer,0,outputs*4);
    this.device.queue.submit([encoder.finish()]);
    await readBuffer.mapAsync(GPUMapMode.READ);
    const out=new Float32Array(readBuffer.getMappedRange().slice(0));readBuffer.unmap();
    inputBuffer.destroy();paramBuffer.destroy();outBuffer.destroy();readBuffer.destroy();
    return out;
  }
  async dense(x,n,prefix,shape){
    const [outDim,inDim]=shape;
    return this.execute(this.densePipeline,x,prefix,[n,inDim,outDim,0],n*outDim);
  }
  async conv(x,height,width,prefix,shape){
    const [outChannels,inChannels]=shape;
    const outH=Math.floor((height+2-3)/2)+1, outW=Math.floor((width+2-3)/2)+1;
    const out=await this.execute(this.convPipeline,x,prefix,[height,width,inChannels,outChannels,outH,outW,0,0],outChannels*outH*outW);
    return {tensor:out,height:outH,width:outW};
  }
}
