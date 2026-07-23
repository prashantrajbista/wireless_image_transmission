"""Bake an eval sample dump (.npz from eval.py --dump-samples) into a single
self-contained interactive HTML. No PyTorch in the browser: reconstructions are
precomputed per SNR; the channel constellation gets fresh JS noise per slider.

  python scripts/bake_html.py --npz results/adjscc_r16_samples.npz --out viz/interactive.html
"""
import argparse
import base64
import io
import json
import os
import numpy as np
from PIL import Image


def img_to_b64(arr, scale=5):
    """arr: (3,H,W) float [0,1] -> upscaled base64 PNG data URI."""
    a = (np.clip(arr, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)
    im = Image.fromarray(a).resize((a.shape[1] * scale, a.shape[0] * scale),
                                   Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return float(10 * np.log10(1.0 / max(mse, 1e-12)))


def build_data(npz_path, n_points=200):
    d = np.load(npz_path)
    inputs = d["inputs"]              # (N,3,32,32)
    outputs = d["outputs"]           # (S,N,3,32,32)
    symbols = d["symbols"]           # (S,N,k_real)
    snr_list = d["snr_list"].tolist()  # (S,)
    N = inputs.shape[0]
    half = symbols.shape[2] // 2

    images = []
    for i in range(N):
        outs, psnrs, tx = [], [], []
        for s in range(len(snr_list)):
            outs.append(img_to_b64(outputs[s, i]))
            psnrs.append(round(psnr(inputs[i], outputs[s, i]), 2))
            sym = symbols[s, i]
            idx = np.linspace(0, half - 1, min(n_points, half)).astype(int)
            pts = np.stack([sym[:half][idx], sym[half:][idx]], axis=1)
            tx.append([[round(float(re), 3), round(float(im), 3)] for re, im in pts])
        images.append({"input": img_to_b64(inputs[i]), "outputs": outs,
                       "psnr": psnrs, "tx": tx})
    return {"snr_list": snr_list, "images": images}


HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ADJSCC — Interactive Wireless Image Transmission</title>
<style>
:root{--bg:#0f1117;--card:#1a1d27;--fg:#e6e8ef;--mut:#9aa1b4;--acc:#5b8cff;--line:#2a2e3c}
@media(prefers-color-scheme:light){:root{--bg:#f6f7fb;--card:#fff;--fg:#1a1d27;--mut:#5a6178;--acc:#2f6bff;--line:#e2e5ee}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:26px;margin:0 0 4px}.sub{color:var(--mut);margin:0 0 24px}
.sub a{color:var(--acc)}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;
padding:20px;margin:16px 0}
.pipe{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0 4px}
.stage{flex:1;min-width:120px;text-align:center;background:var(--bg);
border:1px solid var(--line);border-radius:10px;padding:10px 6px;font-size:13px}
.stage b{display:block;color:var(--acc)}
.panel{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:640px){.panel{grid-template-columns:1fr}}
img.rec{width:100%;image-rendering:pixelated;border-radius:8px;border:1px solid var(--line)}
.imglabel{font-size:12px;color:var(--mut);margin:4px 0 0;text-align:center}
canvas{width:100%;background:var(--bg);border:1px solid var(--line);border-radius:8px}
.controls{margin:18px 0 6px}
input[type=range]{width:100%;accent-color:var(--acc)}
.snrval{font-weight:700;color:var(--acc)}
.thumbs{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}
.thumbs img{width:44px;height:44px;image-rendering:pixelated;border-radius:6px;
border:2px solid transparent;cursor:pointer}
.thumbs img.sel{border-color:var(--acc)}
.big{font-size:22px;font-weight:700}
details{border-top:1px solid var(--line);padding:10px 0}details:first-of-type{border-top:0}
summary{cursor:pointer;font-weight:600}summary::marker{color:var(--acc)}
details p{color:var(--mut);margin:8px 0 0}
code{background:var(--bg);padding:1px 5px;border-radius:4px;font-size:13px}
.note{color:var(--mut);font-size:12px;margin-top:8px}
</style></head><body><div class="wrap">
<h1>ADJSCC — Wireless Image Transmission</h1>
<p class="sub">Interactive reproduction of
<a href="https://arxiv.org/abs/2012.00533">arXiv:2012.00533</a> — one attention
model transmits an image over a noisy channel and adapts to any SNR. Drag the slider.</p>

<div class="card">
  <div class="pipe">
    <div class="stage"><b>1 Encoder</b>CNN + AF<br>image&rarr;symbols</div>
    <div class="stage"><b>2 Power norm</b>unit avg power</div>
    <div class="stage"><b>3 Channel</b>AWGN adds noise</div>
    <div class="stage"><b>4 Decoder</b>CNN + AF<br>symbols&rarr;image</div>
    <div class="stage"><b>5 AF module</b>SNR gates features</div>
  </div>
</div>

<div class="card">
  <div class="controls">
    Channel SNR: <span class="snrval" id="snrTxt"></span> dB &nbsp;
    <span style="float:right">PSNR: <span class="big" id="psnr"></span> dB</span>
    <input type="range" id="snr" min="0" max="0" step="1">
  </div>
  <div class="panel">
    <div>
      <img class="rec" id="inImg" alt="input">
      <p class="imglabel">Stage 0 &mdash; original image sent</p>
    </div>
    <div>
      <img class="rec" id="outImg" alt="reconstruction">
      <p class="imglabel">Stage 4 &mdash; reconstruction at this SNR</p>
    </div>
  </div>
  <canvas id="const" height="260"></canvas>
  <p class="imglabel">Stage 3 &mdash; received constellation (each dot = one complex channel symbol + live noise)</p>
  <div class="thumbs" id="thumbs"></div>
  <p class="note" id="note"></p>
</div>

<div class="card">
  <h3 style="margin-top:0">What each stage does</h3>
  <details open><summary>1 &middot; Encoder (CNN)</summary><p>A convolutional net maps the
  32&times;32 image to <code>C</code> feature channels at 8&times;8 = <code>k</code> complex
  channel symbols. Bandwidth ratio <code>R = k/n = C/96</code>; the paper's main
  setting is <code>R=1/6</code> (<code>C=16</code>). Fewer symbols = more
  compression = lower quality.</p></details>
  <details><summary>2 &middot; Power normalization</summary><p>Symbols are scaled so
  average transmit power per symbol is 1 &mdash; the hardware power budget. Fixed, no
  learnable part.</p></details>
  <details><summary>3 &middot; AWGN channel</summary><p>The channel adds Gaussian noise:
  <code>y = x + n</code>, with <code>SNR(dB)=10&middot;log&#8321;&#8320;(1/&sigma;&sup2;)</code>. Low SNR = fat
  noise cloud: watch the constellation dots scatter as you drag left.</p></details>
  <details><summary>4 &middot; Decoder (CNN)</summary><p>A mirror-image transpose-conv
  net rebuilds the image from the noisy symbols. Trained end to end with the
  encoder on MSE &mdash; no separate JPEG or error-correcting code.</p></details>
  <details><summary>5 &middot; Attention Feature (AF) module &mdash; the paper's contribution</summary>
  <p>Each AF module pools the feature map, concatenates the current SNR, and
  produces a per-channel gate (0&ndash;1) that rescales features. This lets <b>one</b>
  model adapt across the whole SNR range instead of training a separate model per
  SNR &mdash; and it degrades gracefully under channel mismatch.</p></details>
</div>
<script>
const DATA = __DATA__;
let img = 0;
const snr = document.getElementById('snr');
snr.max = DATA.snr_list.length - 1;
snr.value = Math.floor(DATA.snr_list.length/2);
const cv = document.getElementById('const'), ctx = cv.getContext('2d');
function randn(){let u=0,v=0;while(!u)u=Math.random();while(!v)v=Math.random();
  return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}
function draw(){
  const si = +snr.value, s = DATA.snr_list[si], im = DATA.images[img];
  document.getElementById('snrTxt').textContent = s;
  document.getElementById('psnr').textContent = im.psnr[si].toFixed(2);
  document.getElementById('inImg').src = im.input;
  document.getElementById('outImg').src = im.outputs[si];
  const sigma = Math.sqrt(1/Math.pow(10, s/10))/Math.SQRT2;
  const w = cv.width = cv.clientWidth, h = cv.height, cx=w/2, cy=h/2, sc=Math.min(w,h)/7;
  ctx.clearRect(0,0,w,h);
  ctx.strokeStyle='rgba(128,128,128,.3)';ctx.beginPath();
  ctx.moveTo(0,cy);ctx.lineTo(w,cy);ctx.moveTo(cx,0);ctx.lineTo(cx,h);ctx.stroke();
  ctx.fillStyle=getComputedStyle(document.documentElement).getPropertyValue('--acc');
  for(const [re,ie] of im.tx[si]){
    const x=cx+(re+sigma*randn())*sc, y=cy-(ie+sigma*randn())*sc;
    ctx.globalAlpha=.5;ctx.beginPath();ctx.arc(x,y,2.5,0,7);ctx.fill();
  }
  ctx.globalAlpha=1;
}
function thumbs(){
  const t = document.getElementById('thumbs'); t.innerHTML='';
  DATA.images.forEach((im,i)=>{const e=document.createElement('img');
    e.src=im.input;e.className=i===img?'sel':'';e.onclick=()=>{img=i;thumbs();draw();};
    t.appendChild(e);});
}
snr.addEventListener('input',draw);
document.getElementById('note').textContent =
  DATA.images.length+' test images · '+DATA.snr_list.length+' precomputed SNR levels · reconstructions baked, channel noise live in JS';
thumbs();draw();
</script>
</div></body></html>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="results/adjscc_r16_samples.npz")
    p.add_argument("--out", default="viz/interactive.html")
    args = p.parse_args()
    data = build_data(args.npz)
    html = HTML.replace("__DATA__", json.dumps(data))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)
    kb = os.path.getsize(args.out) / 1024
    print(f"wrote {args.out} ({kb:.0f} KB, {len(data['images'])} imgs x "
          f"{len(data['snr_list'])} SNR)")


if __name__ == "__main__":
    main()
