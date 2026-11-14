"""The viewer page: one self-contained HTML file that plays recorded scenes in 3D.

Contract: `build_page(entries, out_path)` takes (Scene, Recording, happened)
triples and writes one HTML file starting with <title> then <style>, no
<html>/<head>/<body> tags (a host wraps it). The only external request is
three.js r128 from cdnjs. Frame data is embedded as compact JSON: positions
and joint energies as integers in thousandths (positions of the frame-0 RMS
radius, energies of the frame-0 median joint energy), so three decimals of
precision. Positions keep the frame-0 scale and the camera never re-fits, so
a shrink of all of space shows as the drawing shrinking; the status line gives
each frame's far-background mean connection length relative to frame 0. Every second step is embedded, plus every step where the joint
count changed and the last step, to keep the page small. Each scene shows
its "watch for" sentence, the "what happened" sentence and its seed and
switched-on rule strengths, the drift marked as the balancing calibration.
Returns out_path.
"""
from __future__ import annotations

import json
import os
from dataclasses import fields

import numpy as np

from physics.rules import RuleSettings
from physics.viewer.frame import Frame
from physics.viewer.recording import Recording
from physics.viewer.scene import Scene

FRAME_STRIDE = 2
THREE_URL = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"


def build_page(entries: list[tuple[Scene, Recording, str]], out_path: str) -> str:
    """Write the viewer page for the given scenes and return its path."""
    data = [_scene_data(scene, recording, happened) for scene, recording, happened in entries]
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page = (f"<title>Quantum-relativity viewer</title>\n<style>{_STYLE}</style>\n{_BODY}\n"
            f'<script src="{THREE_URL}"></script>\n'
            f'<script id="scenes" type="application/json">{payload}</script>\n'
            f"<script>{_SCRIPT}</script>\n")
    folder = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(folder, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(page)
    return out_path


def _scene_data(scene: Scene, recording: Recording, happened: str) -> dict:
    first = recording.frames[0]
    radius = float(np.sqrt(np.mean(np.sum(first.positions ** 2, axis=1))))
    unit = float(np.median(first.energy))
    kept = _kept_frames(recording.frames)
    pooled = np.concatenate([np.log10(np.maximum(frame.energy / unit, 1e-9)) for frame in kept])
    ruler = first.background_length
    return {"name": scene.name, "title": scene.title, "watch": scene.watch_for,
            "happened": happened, "seed": scene.seed, "rules": _switched_on(scene.settings),
            "charge": "" if np.any(scene.network.charge) else " No joint carries charge, so repulsion has nothing to act on.",
            "steps": recording.frames[-1].step,
            "low": round(float(np.percentile(pooled, 2)), 3), "high": round(float(np.percentile(pooled, 99.5)), 3),
            "frames": [_frame_data(frame, radius, unit, ruler) for frame in kept]}


def _kept_frames(frames: tuple[Frame, ...]) -> list[Frame]:
    return [frame for index, frame in enumerate(frames)
            if frame.step % FRAME_STRIDE == 0 or index == len(frames) - 1
            or frame.joint_count != frames[index - 1].joint_count]


def _frame_data(frame: Frame, radius: float, unit: float, ruler: float) -> dict:
    size = frame.background_length / ruler
    return {"s": frame.step, "n": frame.joint_count, "g": round(size, 4) if np.isfinite(size) else None,
            "p": np.rint(frame.positions.ravel() / radius * 1000).astype(int).tolist(),
            "e": np.rint(frame.energy / unit * 1000).astype(np.int64).tolist(),
            "b": frame.bright.ravel().astype(int).tolist(), "m": list(frame.marked)}


def _switched_on(settings: RuleSettings) -> str:
    defaults = RuleSettings()
    changed = [f"{field.name} {_number(getattr(settings, field.name))}" for field in fields(settings)
               if getattr(settings, field.name) != getattr(defaults, field.name)
               or field.name == "energy_length"]
    return ", ".join(changed)


def _number(value: float) -> str:
    return f"{value:.4g}"


_STYLE = """
:root{--bg:#f7f6f2;--panel:#ffffff;--ink:#1c1b19;--muted:#66635c;--line:#dedad0;--accent:#2d5bd3;
--view:#eceae3;--mark:#111111;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121211;--panel:#1c1c1a;
--ink:#ecebe6;--muted:#a3a097;--line:#33322e;--accent:#8fb0ff;--view:#0b0b0a;--mark:#ffffff;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#121211;--panel:#1c1c1a;--ink:#ecebe6;--muted:#a3a097;--line:#33322e;
--accent:#8fb0ff;--view:#0b0b0a;--mark:#ffffff;color-scheme:dark}
body{background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:60rem;margin:0 auto;padding-inline:16px;padding-block:1.25rem 2.5rem}
h1{font-size:1.35rem;margin:0 0 .25rem}
.lede{color:var(--muted);margin:0 0 1rem}
.picker{display:flex;flex-wrap:wrap;gap:.5rem;margin-bottom:.75rem}
.picker button,.controls button,.controls select{font:inherit;color:var(--ink);background:var(--panel);
border:1px solid var(--line);border-radius:.5rem;padding:.35rem .8rem;cursor:pointer}
.picker button[aria-pressed="true"]{border-color:var(--accent);color:var(--accent);font-weight:600}
.view{position:relative;background:var(--view);border:1px solid var(--line);border-radius:.75rem;
overflow:hidden;aspect-ratio:4/3;max-height:72vh;width:100%;touch-action:none;cursor:grab}
.view canvas{display:block;width:100%;height:100%}
.legend{position:absolute;left:.75rem;bottom:.6rem;font-size:.75rem;color:var(--muted);
display:flex;align-items:center;gap:.4rem;pointer-events:none}
.legend i{display:inline-block;width:6rem;height:.5rem;border-radius:.25rem;
background:linear-gradient(90deg,#2f4fb5,#2fa7b8,#f0c44c,#e0462a)}
.controls{display:flex;flex-wrap:wrap;align-items:center;gap:.6rem;margin:.75rem 0}
.controls input[type=range]{flex:1 1 12rem;min-width:0;accent-color:var(--accent)}
.status{font-variant-numeric:tabular-nums;color:var(--muted);font-size:.9rem}
.notes{background:var(--panel);border:1px solid var(--line);border-radius:.75rem;padding:.9rem 1rem}
.notes p{margin:.2rem 0 .6rem}
.notes .label{font-size:.75rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0}
.notes .small{font-size:.8rem;color:var(--muted);margin-bottom:0;overflow-wrap:anywhere}
.warn{color:#b3261e}
"""

_BODY = """<div class="wrap">
<h1>Quantum-relativity viewer</h1>
<p class="lede">Joints coloured by the energy they hold, brightest connections as faint lines,
followed joints in white or black. Positions are a drawing of network distances only; the network has none.
The camera stays put, so if all of space shrinks or grows the whole drawing does too.</p>
<div class="picker" id="picker"></div>
<div class="view" id="view"><div class="legend">low <i></i> high joint energy</div></div>
<div class="controls">
<button id="play" type="button">Pause</button>
<input id="slider" type="range" min="0" value="0" aria-label="Frame">
<select id="speed" aria-label="Speed"><option value="0.5">0.5x</option><option value="1" selected>1x</option>
<option value="2">2x</option><option value="4">4x</option></select>
<span class="status" id="status"></span>
</div>
<div class="notes">
<p class="label">What to watch for</p><p id="watch"></p>
<p class="label">What happened</p><p id="happened"></p>
<p class="small" id="rules"></p>
</div>
</div>"""

_SCRIPT = r"""
(function(){
const scenes = JSON.parse(document.getElementById('scenes').textContent);
const view = document.getElementById('view'), slider = document.getElementById('slider');
const playButton = document.getElementById('play'), speed = document.getElementById('speed');
const status = document.getElementById('status');
if (typeof THREE === 'undefined') { view.textContent = 'three.js did not load, so the 3D view cannot draw.'; return; }
const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
view.prepend(renderer.domElement);
const world = new THREE.Scene(), camera = new THREE.PerspectiveCamera(45, 4 / 3, 0.01, 100);
const group = new THREE.Group(); world.add(group);
let distance = 4.2; group.rotation.set(0.35, 0.6, 0);
const stops = [[0.18,0.31,0.71],[0.18,0.65,0.72],[0.94,0.77,0.30],[0.88,0.27,0.16]];
function colour(level, out, at) {
  const scaled = Math.min(Math.max(level, 0), 1) * (stops.length - 1);
  const low = Math.min(Math.floor(scaled), stops.length - 2), part = scaled - low;
  for (let axis = 0; axis < 3; axis++) out[at + axis] = stops[low][axis] + (stops[low + 1][axis] - stops[low][axis]) * part;
}
const most = Math.max(...scenes.flatMap(scene => scene.frames.map(frame => frame.n)));
const pointGeometry = new THREE.BufferGeometry();
const pointPositions = new Float32Array(most * 3), pointColours = new Float32Array(most * 3);
pointGeometry.setAttribute('position', new THREE.BufferAttribute(pointPositions, 3));
pointGeometry.setAttribute('color', new THREE.BufferAttribute(pointColours, 3));
const points = new THREE.Points(pointGeometry, new THREE.PointsMaterial({size: 0.03, vertexColors: true}));
const lineMost = Math.max(...scenes.flatMap(scene => scene.frames.map(frame => frame.b.length)));
const lineGeometry = new THREE.BufferGeometry(), linePositions = new Float32Array(lineMost * 3);
lineGeometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
const lineMaterial = new THREE.LineBasicMaterial({color: 0xe0462a, transparent: true, opacity: 0.22});
const lines = new THREE.LineSegments(lineGeometry, lineMaterial);
const markGeometry = new THREE.BufferGeometry(), markPositions = new Float32Array(30);
markGeometry.setAttribute('position', new THREE.BufferAttribute(markPositions, 3));
const markMaterial = new THREE.PointsMaterial({size: 0.12, transparent: true, opacity: 0.85});
const marks = new THREE.Points(markGeometry, markMaterial);
group.add(points, lines, marks);
function themeColours() {
  const style = getComputedStyle(document.documentElement);
  renderer.setClearColor(new THREE.Color(style.getPropertyValue('--view').trim() || '#000'));
  markMaterial.color = new THREE.Color(style.getPropertyValue('--mark').trim() || '#fff');
  draw();
}
let current = 0, frameIndex = 0, playing = true, clock = 0, last = performance.now();
function show(index) {
  const scene = scenes[current], frame = scene.frames[index]; frameIndex = index;
  const span = Math.max(scene.high - scene.low, 1e-6);
  for (let joint = 0; joint < frame.n; joint++) {
    for (let axis = 0; axis < 3; axis++) pointPositions[joint * 3 + axis] = frame.p[joint * 3 + axis] / 1000;
    colour((Math.log10(Math.max(frame.e[joint], 1e-6) / 1000) - scene.low) / span, pointColours, joint * 3);
  }
  pointGeometry.setDrawRange(0, frame.n);
  pointGeometry.attributes.position.needsUpdate = true; pointGeometry.attributes.color.needsUpdate = true;
  for (let end = 0; end < frame.b.length; end++)
    for (let axis = 0; axis < 3; axis++) linePositions[end * 3 + axis] = frame.p[frame.b[end] * 3 + axis] / 1000;
  lineGeometry.setDrawRange(0, frame.b.length); lineGeometry.attributes.position.needsUpdate = true;
  frame.m.slice(0, 10).forEach((joint, slot) => {
    for (let axis = 0; axis < 3; axis++) markPositions[slot * 3 + axis] = frame.p[joint * 3 + axis] / 1000;
  });
  markGeometry.setDrawRange(0, Math.min(frame.m.length, 10)); markGeometry.attributes.position.needsUpdate = true;
  slider.value = index;
  const space = frame.g === null ? 'far space collapsed' : 'far space ' + (frame.g * 100).toFixed(1) + '% of start size';
  status.textContent = 'step ' + frame.s + ' of ' + scene.steps + ', ' + frame.n + ' joints, ' + space;
  draw();
}
function pick(index) {
  current = index; const scene = scenes[index];
  document.querySelectorAll('#picker button').forEach((button, at) => button.setAttribute('aria-pressed', at === index));
  document.getElementById('watch').textContent = scene.watch;
  document.getElementById('happened').textContent = scene.happened;
  document.getElementById('rules').textContent = 'Seed ' + scene.seed + '. All rules on: ' + scene.rules +
    '. The drift is calibrated so even space keeps its size.' + scene.charge;
  slider.max = scene.frames.length - 1; clock = 0; show(0);
}
scenes.forEach((scene, index) => {
  const button = document.createElement('button'); button.type = 'button';
  button.textContent = scene.title; button.onclick = () => pick(index);
  document.getElementById('picker').append(button);
});
function draw() {
  const width = view.clientWidth, height = view.clientHeight;
  if (renderer.domElement.width !== Math.round(width * renderer.getPixelRatio())) renderer.setSize(width, height, false);
  camera.aspect = width / Math.max(height, 1); camera.updateProjectionMatrix();
  camera.position.set(0, 0, distance); camera.lookAt(0, 0, 0);
  renderer.render(world, camera);
}
function tick(now) {
  const elapsed = Math.min(now - last, 100) / 1000; last = now;
  const frames = scenes[current].frames;
  if (playing) {
    clock += elapsed * 8 * Number(speed.value);
    if (clock >= 1) { clock = 0; show(frameIndex + 1 < frames.length ? frameIndex + 1 : 0); }
  }
  requestAnimationFrame(tick);
}
playButton.onclick = () => { playing = !playing; playButton.textContent = playing ? 'Pause' : 'Play'; };
slider.oninput = () => { playing = false; playButton.textContent = 'Play'; show(Number(slider.value)); };
const pointers = new Map(); let pinch = 0;
view.addEventListener('pointerdown', event => { view.setPointerCapture(event.pointerId); pointers.set(event.pointerId, [event.clientX, event.clientY]); });
view.addEventListener('pointermove', event => {
  if (!pointers.has(event.pointerId)) return;
  const [oldX, oldY] = pointers.get(event.pointerId); pointers.set(event.pointerId, [event.clientX, event.clientY]);
  if (pointers.size === 1) {
    group.rotation.y += (event.clientX - oldX) * 0.008; group.rotation.x += (event.clientY - oldY) * 0.008;
  } else if (pointers.size === 2) {
    const [first, second] = [...pointers.values()];
    const spread = Math.hypot(first[0] - second[0], first[1] - second[1]);
    if (pinch) distance = Math.min(Math.max(distance * pinch / spread, 0.3), 20);
    pinch = spread;
  }
  draw();
});
['pointerup', 'pointercancel'].forEach(name => view.addEventListener(name, event => { pointers.delete(event.pointerId); pinch = 0; }));
view.addEventListener('wheel', event => {
  event.preventDefault(); distance = Math.min(Math.max(distance * Math.exp(event.deltaY * 0.001), 0.3), 20); draw();
}, {passive: false});
window.addEventListener('resize', draw);
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', themeColours);
new MutationObserver(themeColours).observe(document.documentElement, {attributes: true, attributeFilter: ['data-theme']});
pick(0); themeColours(); requestAnimationFrame(tick);
})();
"""
