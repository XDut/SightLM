# SightLM - Lightweight, Browser-Native VLM trained from scratch

**A real 82,387-parameter vision-language model**: three CNN layers + learned visual projector + a one-block, four-head causal Transformer. Every model weight was learned from random initialization in PyTorch. The generated neural network runs entirely in the browser through a **custom JavaScript inference implementation**, with **handwritten WebGPU / WGSL convolution and dense kernels**, plus a tested pure-JavaScript CPU fallback.

This is an educational, domain-constrained MiniVLM, not a general-purpose photograph-understanding model. It recognizes **one colored circle, triangle or square** placed at **left / center / right**, and answers four question types: **color, shape, location, description**. Images outside that training distribution may produce incorrect predictions. No pretrained VLM, pretrained vision backbone, model API, WebLLM, ONNX, or server-side ML inference is used.

## Running the app (no training necessary)

Click on the netlify link for the demo.

## Architecture

```text
Image 32x32 RGB → CNN 3→16→24→32, 3x3 stride=2, ReLU
                → Flatten(32 x 4 x 4 = 512)
                → Learned Linear Projection (512 → 64)
                → Image Prefix Token ───────────┐
                                                │
Question → Tokenize → Embedding (27 × 64) ──────┤
                                                ↓
                           Positional Embeddings (25 × 64)
                                                ↓
                           Pre-norm Causal Self-Attention
                           4 heads, width 64
                                                ↓
                           MLP GELU (64 → 128 → 64)
                                                ↓
                           LayerNorm + token logits
                                                ↓
                           Greedy autoregressive text output
```

**Training stack:** Python, PyTorch, NumPy, Pillow. Optimizer: AdamW, cross-entropy on answer tokens, gradient clipping, supervised visual question answering.

**Runtime stack:** HTML, CSS, native JS ES modules, Canvas, WebGPU (WGSL), Float32 binary weights, Service Worker for optional offline caching, CPU fallback. No npm, WASM dependency, GPU library, ONNX, WebLLM or external API.

In the WebGPU backend, the project's own WGSL compute shaders perform all three convolutions and six dense projections; JavaScript currently performs activation, layer normalization, masked attention, tokenization and generation control. This design avoids a third-party runtime but uses CPU/GPU readbacks, which can make GPU execution slower than CPU execution for a model this small.

## Training from scratch

```bash
pip install -r requirements.txt
python train.py --epochs 16 --train-images 1800 --val-images 250 --batch-size 128
python export.py
python evaluate.py
```

No dataset download or prior checkpoint is needed. The `draw_example` generator creates randomized RGB shapes and supervised question/answer pairs. The training and validation generators use separate random seeds. An independent held-out test generator is used for final exact-match assessment.

Training does **not** claim open-domain visual reasoning. The 27-token vocabulary, single-shape task, and fixed question set were chosen to make a complete from-scratch training-to-browser workflow achievable on CPU.

### Supplied artifacts

- `artifacts/model.pt` — complete PyTorch checkpoint, saved at the best validation epoch.
- `web/assets/model.bin` — lightweight contiguous float32 weights for browser inference.
- `web/assets/model.json` — tensor offsets, shapes, tokenizer, architecture metadata.
- `artifacts/training_metrics.json` — per-epoch train/validation loss and validation token accuracy.
- `artifacts/evaluation.json` — fully autoregressive independent test results.
- `artifacts/canvas_generalization.json` — model answers on all 36 browser Canvas visual combinations.
- `artifacts/browser_screenshot.png` — screenshot of the interface.

### Measured results

| Test | Result | Interpretation |
| --- | ---: | --- |
| Model parameters | 82,387 | ~322 KiB float32 weights |
| Best validation token accuracy | 99.906% | Teacher-forced target tokens on 250 held-out images, 1,000 Q&A pairs |
| Held-out autoregressive color accuracy | 79/80 (98.75%) | Exact answer match, independent synthetic examples |
| Held-out autoregressive shape accuracy | 79/80 (98.75%) | Exact answer match, independent synthetic examples |
| Held-out autoregressive position accuracy | 80/80 (100%) | Exact answer match, independent synthetic examples |
| Held-out autoregressive description accuracy | 79/80 (98.75%) | Exact answer match, independent synthetic examples |
| Browser Canvas shape combinations | 144/144 (100%) | Four Q&A tasks for all 36 color/shape/position combinations |
| PyTorch ↔ JS inference parity | Passed | Max logit deviation < 0.000001 in tested fixture |
| Chromium JS inference | Passed | Visual prompt generated in a real browser context |
| WebGPU GPU hardware inference | **Not verified in this sandbox** | Hardware/secure-origin execution must be checked on your device |

**Numbers are for the synthetic 32×32 shape domain only**, not general VQA accuracy. Validation token accuracy differs from independent greedy autoregressive exact-match accuracy. For open-world CV/DSAI research, evaluate on a real dataset and compare against meaningful baselines.

## Tests

```bash
python evaluate.py
PYTHONPATH=. python tests/create_parity_fixture.py
node tests/check_parity.mjs
python tests/browser_smoke.py                # optional: requires Playwright + Chromium
python tests/check_canvas_generalization.py # optional: requires Playwright + Chromium
```

`tests/browser_smoke.py` tests the model's JavaScript CPU backend inside Chromium and produces a browser screenshot; this sandbox blocks navigation even to localhost, so it injects the script directly rather than testing HTTP loading. GPU execution is implemented but hasn't been independently hardware-tested here. The default browser app automatically falls back to CPU if WebGPU isn't available or a shader operation fails.

## Extending this into a portfolio-grade VLM

1. **Add a ViT encoder** and run CNN-vs-ViT comparisons with matched parameters and training budget.
2. Train on a larger image-captioning dataset (e.g., Flickr8k, subject to its dataset terms) and expand vocabulary, image size, and language decoder depth.
3. Add variable-length questions, more objects, attribute relations, and spatial reasoning.
4. Implement on-GPU layer normalization and attention, avoiding per-layer readbacks; benchmark GPU latency vs JS CPU.
5. Add quantized weights, KV caching and streaming generation for larger models.
6. Include ablations, confidence calibration, error analysis and evaluate on held-out distribution shifts.

## File structure

```text
minivlm-browser/
├── launch.py                   # local static hosting + browser opener
├── launch_windows.pyw          # Windows no-console launcher
├── start_windows.bat           # double-click helper
├── requirements.txt           # training deps only
├── train.py                    # full scratch training
├── evaluate.py                 # autoregressive independent test
├── export.py                   # custom binary model export
├── minivlm/
│   ├── model.py                # CNN + causal Transformer
│   └── data.py                 # procedural images, Q&A labels
├── web/
│   ├── index.html              # browser UI
│   ├── styles.css              # responsive layout
│   ├── app.js                  # UI + image processing + generation
│   ├── model.js                # CPU and original WebGPU inference code
│   ├── sw.js                   # static asset offline caching
│   └── assets/model.{json,bin}# serialized trained weights
├── tests/                      # Python↔JS, browser and visual tests
└── artifacts/                  # checkpoints, results, screenshot
```

### License

Source code is released under the MIT license in `LICENSE`. No third-party pretrained model weights are bundled. For any future datasets or pretrained components, review their separate license terms.

### Interface update

The Color, Shape and Position dropdowns have been removed. Use **Generate random sample** for a new synthetic image or **Upload image** for your own PNG/JPG. All four question prompts and browser inference remain available.
