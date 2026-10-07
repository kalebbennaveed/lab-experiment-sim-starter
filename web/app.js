"use strict";

const $ = (id) => document.getElementById(id);
const scene = $("scene"), chart = $("error-chart");
const ctx = scene.getContext("2d"), chartCtx = chart.getContext("2d");
const mint = "#78dbc3", orange = "#eea369";
let result = null, resultRevision = "", webRevision = "", playbackTime = 0;
let playing = true, camera = { azimuth: -0.58, elevation: 0.42, zoom: 1, view: "perspective" };
let previousFrame = performance.now(), dragging = null;
let selectedMode = "lab", selectedPattern = "";

function metric(id, value) {
  const node = $(id);
  node.replaceChildren(document.createTextNode(value.toFixed(3)), Object.assign(document.createElement("small"), {textContent: "m"}));
}

function installResult(data) {
  result = data;
  playbackTime = 0;
  const {config, metrics: m} = data;
  $("lab-name").textContent = config.lab.name;
  $("mass").textContent = `${config.vehicle.mass.toFixed(3)} kg`;
  $("rate").textContent = `${Math.round(1 / config.simulation.dt)} Hz`;
  $("volume").textContent = config.lab.enabled ? `${config.lab.z_min}–${config.lab.z_max} m height` : "Unbounded";
  $("mode-caption").textContent = config.lab.enabled ? "01 / LAB TRACKING" : "02 / FREE-SPACE TRACKING";
  $("scene-type").textContent = config.lab.enabled ? "/ 3D flight volume" : "/ unbounded 3D trajectory";
  $("bounds-legend").hidden = !config.lab.enabled;
  $("duration").textContent = `${config.simulation.duration.toFixed(1)} s`;
  metric("rms", m.rms_error);
  metric("max-error", m.max_error);
  metric("final-error", m.final_error);
  $("clearance").textContent = config.lab.enabled ? `${m.minimum_clearance.toFixed(3)} m` : "Not checked";
  $("saturation").textContent = `${(100 * m.saturation_fraction).toFixed(1)}%`;
  const outside = m.boundary_violations > 0 || m.reference_violations > 0;
  document.querySelector(".bounds-panel").classList.toggle("warning", outside);
  $("bounds-status").textContent = !config.lab.enabled ? "Free space · no bounds" : outside ? "Boundary crossing detected" : "Inside the flight volume";
  $("bounds-detail").textContent = !config.lab.enabled ? "Place the trajectory anywhere. The view follows the path; the world grid is only a visual guide." : outside
    ? `${m.boundary_violations} vehicle and ${m.reference_violations} reference integration samples cross the clearance boundary.`
    : `Vehicle and reference stay inside with a ${config.lab.vehicle_radius.toFixed(2)} m vehicle radius.`;
  $("run-info").textContent = `${m.steps.toLocaleString()} steps · computed in ${m.compute_seconds.toFixed(2)} s · ${new Date().toLocaleTimeString()}`;
}

async function poll() {
  try {
    const response = await fetch("/api/status", {cache: "no-store"});
    if (!response.ok) throw new Error(`Server returned ${response.status}`);
    const status = await response.json();
    if (webRevision && webRevision !== status.web_revision) {
      window.location.reload();
      return;
    }
    webRevision = status.web_revision;
    $("example").textContent = status.example;
    $("experiment-controls").hidden = status.mode === "custom";
    selectedMode = status.mode;
    selectedPattern = status.pattern || "";
    $("pattern-control").hidden = selectedMode !== "generic";
    $("pattern").value = selectedPattern;
    document.querySelectorAll("[data-mode]").forEach(button => {const active = button.dataset.mode === selectedMode;button.classList.toggle("active",active);button.setAttribute("aria-pressed",String(active));});
    $("status").textContent = {running: "Recomputing experiment", ready: "Experiment up to date", error: "Latest edit needs a fix"}[status.state] || status.state;
    $("status-dot").className = `dot ${status.state === "running" ? "working" : status.state === "error" ? "failed" : ""}`;
    $("error-panel").hidden = !status.error;
    $("error-message").textContent = status.error || "";
    if (status.result_revision && resultRevision !== status.result_revision) {
      const runResponse = await fetch("/api/result", {cache: "no-store"});
      if (!runResponse.ok) throw new Error("Could not load the simulation result");
      installResult(await runResponse.json());
      resultRevision = status.result_revision;
    }
  } catch (error) {
    $("status").textContent = "Waiting for the local server";
    $("status-dot").className = "dot working";
  } finally {
    window.setTimeout(poll, 800);
  }
}

function canvasSize(canvas, context) {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (canvas.width !== Math.round(rect.width * dpr) || canvas.height !== Math.round(rect.height * dpr)) {
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
  }
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  return {w: rect.width, h: rect.height};
}

function sceneGeometry() {
  if (result.config.lab.enabled) return result.config.lab;
  const points = [...result.history.reference, ...result.history.position];
  const minima = [0,1,2].map(i => Math.min(...points.map(p => p[i])));
  const maxima = [0,1,2].map(i => Math.max(...points.map(p => p[i])));
  const pad = .7;
  return {floor:[[minima[0]-pad,minima[1]-pad],[maxima[0]+pad,minima[1]-pad],[maxima[0]+pad,maxima[1]+pad],[minima[0]-pad,maxima[1]+pad]],z_min:minima[2]-.6,z_max:Math.max(maxima[2]+.6,minima[2]+2)};
}

function projector(w, h) {
  const lab = sceneGeometry();
  const xs = lab.floor.map(p => p[0]), ys = lab.floor.map(p => p[1]);
  const center = [(Math.min(...xs) + Math.max(...xs)) / 2, (Math.min(...ys) + Math.max(...ys)) / 2, (lab.z_min + lab.z_max) / 2];
  const c = Math.cos(camera.azimuth), s = Math.sin(camera.azimuth);
  const se = Math.sin(camera.elevation), ce = Math.cos(camera.elevation);
  const raw = (p) => {
    const [x, y, z] = p.map((value, i) => value - center[i]);
    return [c*x - s*y, se*(s*x + c*y) - ce*z];
  };
  const corners = lab.floor.flatMap(p => [raw([...p, lab.z_min]), raw([...p, lab.z_max])]);
  const spanX = Math.max(...corners.map(p => p[0])) - Math.min(...corners.map(p => p[0]));
  const spanY = Math.max(...corners.map(p => p[1])) - Math.min(...corners.map(p => p[1]));
  const scale = Math.min((w - 110) / Math.max(spanX, 1), (h - 100) / Math.max(spanY, 1)) * camera.zoom;
  const project = (p) => { const [x, y] = raw(p); return [w/2 + x*scale, h/2 + y*scale]; };
  return {project, scale, xs, ys};
}

function line(points, color, width=1, dash=[]) {
  if (points.length < 2) return;
  ctx.beginPath();
  ctx.moveTo(...points[0]);
  for (const p of points.slice(1)) ctx.lineTo(...p);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dash);
  ctx.stroke();
  ctx.setLineDash([]);
}

function frameAt(time) {
  const hist = result.history;
  let left = 0, right = hist.time.length - 1;
  while (left < right) {
    const mid = Math.ceil((left + right) / 2);
    if (hist.time[mid] <= time) left = mid;
    else right = mid - 1;
  }
  const next = Math.min(left + 1, hist.time.length - 1);
  const dt = hist.time[next] - hist.time[left];
  const fraction = dt > 0 ? (time - hist.time[left]) / dt : 0;
  const mix = (key) => hist[key][left].map((value, i) => value + fraction * (hist[key][next][i] - value));
  return {index: left, position: mix("position"), reference: mix("reference"), rotation: hist.rotation[left], error: hist.error[left] + fraction * (hist.error[next] - hist.error[left])};
}

function drawLab(project, xs, ys) {
  const lab = sceneGeometry();
  const floor = lab.floor.map(p => project([...p, lab.z_min]));
  const ceiling = lab.floor.map(p => project([...p, lab.z_max]));
  ctx.beginPath();
  ctx.moveTo(...floor[0]);
  for (const p of floor.slice(1)) ctx.lineTo(...p);
  ctx.closePath();
  ctx.fillStyle = "#1a304080";
  ctx.fill();
  ctx.save();
  ctx.clip();
  const [xmin, xmax, ymin, ymax] = [Math.floor(Math.min(...xs)), Math.ceil(Math.max(...xs)), Math.floor(Math.min(...ys)), Math.ceil(Math.max(...ys))];
  const step = Math.max(1,Math.ceil(Math.max(xmax-xmin,ymax-ymin)/30));
  for (let x = xmin; x <= xmax; x+=step) line([project([x, ymin, lab.z_min]), project([x, ymax, lab.z_min])], "#50768a27");
  for (let y = ymin; y <= ymax; y+=step) line([project([xmin, y, lab.z_min]), project([xmax, y, lab.z_min])], "#50768a27");
  ctx.restore();
  if (result.config.lab.enabled) {
    line([...floor, floor[0]], "#65899d99", 1);
    line([...ceiling, ceiling[0]], "#57748b66", 1, [4, 5]);
    for (let i = 0; i < floor.length; i++) line([floor[i], ceiling[i]], "#57748b44", 1, [3, 6]);
  }
  const base = [xmin + .5, ymin + .5, lab.z_min];
  const origin = project(base);
  for (const [axis, vector, color] of [["x", [1,0,0], "#b08078"], ["y", [0,1,0], "#759f86"], ["z", [0,0,1], "#789bb6"]]) {
    const end = project(base.map((v, i) => v + vector[i]));
    line([origin, end], color, 1.3);
    ctx.font = "10px system-ui";
    ctx.fillStyle = color;
    ctx.fillText(axis, end[0] + 4, end[1] + 4);
  }
}

function drawQuad(frame, project, scale) {
  const p = frame.position;
  const ground = project([p[0], p[1], sceneGeometry().z_min]);
  const screen = project(p);
  line([ground, screen], "#78dbc330", 1, [2, 5]);
  ctx.beginPath();ctx.ellipse(ground[0], ground[1], 13, 4, 0, 0, Math.PI*2);
  ctx.fillStyle = "#00000030";ctx.fill();
  const r = frame.rotation;
  const transform = (v) => project(p.map((value, i) => value + r[i].reduce((sum, entry, j) => sum + entry*v[j], 0)));
  // Schematic drawn larger than the physical 0.08 m arm for legibility.
  const arm = 0.25;
  const motors = [[arm,-arm,0], [-arm,arm,0], [arm,arm,0], [-arm,-arm,0]].map(transform);
  line([motors[0], motors[1]], "#e5f3f6", 3);
  line([motors[2], motors[3]], "#e5f3f6", 3);
  const ring = Math.max(4, Math.min(10, scale*.13));
  for (let i = 0; i < motors.length; i++) {
    const [x,y] = motors[i];
    ctx.beginPath();ctx.arc(x,y,ring,0,Math.PI*2);ctx.fillStyle="#182d3b";ctx.fill();
    ctx.strokeStyle=i===0||i===2?mint:"#c4d9e4";ctx.lineWidth=1.6;ctx.stroke();
    const angle = performance.now()/160 + i;
    line([[x-Math.cos(angle)*ring*.75,y-Math.sin(angle)*ring*.75],[x+Math.cos(angle)*ring*.75,y+Math.sin(angle)*ring*.75]], "#ffffff80", 1);
  }
  ctx.beginPath();ctx.arc(...screen,4,0,Math.PI*2);ctx.fillStyle=mint;ctx.fill();
  line([screen, transform([.45,0,0])], orange, 2);
}

function drawScene(frame) {
  const {w,h} = canvasSize(scene, ctx);
  ctx.clearRect(0,0,w,h);
  if (!result) {
    ctx.fillStyle="#7991a2";ctx.font="13px system-ui";ctx.textAlign="center";
    ctx.fillText("Computing your first flight…",w/2,h/2);ctx.textAlign="left";
    return;
  }
  const {project,scale,xs,ys} = projector(w,h);
  drawLab(project,xs,ys);
  line(result.history.reference.map(project), "#eea369aa", 1.4, [5,5]);
  line(result.history.position.map(project), "#78dbc325", 1.5);
  const trail = result.history.position.slice(0,frame.index+1).map(project);
  trail.push(project(frame.position));
  line(trail,mint,2);
  const ref = project(frame.reference);
  ctx.beginPath();ctx.arc(...ref,4,0,Math.PI*2);ctx.strokeStyle=orange;ctx.lineWidth=1.5;ctx.stroke();
  drawQuad(frame,project,scale);
}

function drawChart(frame) {
  const {w,h} = canvasSize(chart,chartCtx);
  chartCtx.clearRect(0,0,w,h);
  if (!result || w < 70) return;
  const hist = result.history, pad = {left:35, right:13, top:12, bottom:22};
  const maxY = Math.max(.05, Math.ceil(Math.max(...hist.error)*100/5)*.05);
  const project = (t,e) => [pad.left+(w-pad.left-pad.right)*t/result.config.simulation.duration, h-pad.bottom-(h-pad.top-pad.bottom)*e/maxY];
  chartCtx.font="9px system-ui";
  for (let i=0;i<=3;i++) {
    const val=maxY*i/3, y=project(0,val)[1];
    chartCtx.beginPath();chartCtx.moveTo(pad.left,y);chartCtx.lineTo(w-pad.right,y);chartCtx.strokeStyle="#e6ece8";chartCtx.lineWidth=1;chartCtx.stroke();
    chartCtx.fillStyle="#81928c";chartCtx.fillText(val.toFixed(2),1,y+3);
  }
  for (let i=0;i<=4;i++) {
    const t=result.config.simulation.duration*i/4, x=project(t,0)[0];
    chartCtx.fillText(t.toFixed(0),x-3,h-4);
  }
  chartCtx.beginPath();
  hist.time.forEach((t,i)=>{const [x,y]=project(t,hist.error[i]);i?chartCtx.lineTo(x,y):chartCtx.moveTo(x,y);});
  chartCtx.strokeStyle="#559e87";chartCtx.lineWidth=1.6;chartCtx.stroke();
  const point=project(playbackTime,frame.error);
  chartCtx.beginPath();chartCtx.moveTo(point[0],pad.top);chartCtx.lineTo(point[0],h-pad.bottom);chartCtx.strokeStyle="#9bc8b7";chartCtx.setLineDash([2,3]);chartCtx.stroke();chartCtx.setLineDash([]);
  chartCtx.beginPath();chartCtx.arc(...point,3,0,Math.PI*2);chartCtx.fillStyle="#3e977a";chartCtx.fill();
}

function tick(now) {
  const elapsed=Math.min((now-previousFrame)/1000,.1);
  previousFrame=now;
  if (result) {
    const duration=result.config.simulation.duration;
    if (playing && !dragging) {
      playbackTime+=elapsed*Number($("speed").value);
      if (playbackTime>duration) {
        if ($("loop").checked) playbackTime%=duration;
        else {playbackTime=duration;setPlaying(false);}
      }
    }
    const frame=frameAt(playbackTime);
    $("time").textContent=`${playbackTime.toFixed(2)} s`;
    $("scrub").value=Math.round(playbackTime/duration*1000);
    metric("current-error",frame.error);
    $("current-position").textContent=`x ${frame.position[0].toFixed(2)} · y ${frame.position[1].toFixed(2)} · z ${frame.position[2].toFixed(2)}`;
    drawScene(frame);drawChart(frame);
  } else drawScene(null);
  requestAnimationFrame(tick);
}

function setPlaying(value) {
  playing=value;
  $("play").textContent=value?"Ⅱ":"▶";
  $("play").setAttribute("aria-label",value?"Pause playback":"Play flight");
}

$("play").addEventListener("click",()=>setPlaying(!playing));
$("scrub").addEventListener("input",()=>{if(result)playbackTime=Number($("scrub").value)/1000*result.config.simulation.duration;});
$("rerun").addEventListener("click",async()=>{
  try {const response=await fetch("/api/rerun",{method:"POST"});if(!response.ok)throw new Error("Run failed");$("status").textContent="Recomputing experiment";}
  catch {$("status").textContent="Waiting for the local server";}
});
async function selectExperiment(mode,pattern="") {
  try {
    const response=await fetch("/api/experiment",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({mode,pattern:pattern||null})});
    if(!response.ok)throw new Error((await response.json()).error);
    selectedMode=mode;selectedPattern=pattern;
    $("status").textContent="Recomputing experiment";
  } catch(error) {$("status").textContent=error.message;}
}
document.querySelectorAll("[data-mode]").forEach(button=>button.addEventListener("click",()=>selectExperiment(button.dataset.mode)));
$("pattern").addEventListener("change",()=>selectExperiment("generic",$("pattern").value));
document.querySelectorAll("[data-view]").forEach(button=>button.addEventListener("click",()=>{
  camera.view=button.dataset.view;camera.zoom=1;
  if(camera.view==="top"){camera.azimuth=0;camera.elevation=Math.PI/2;}
  else if(camera.view==="side"){camera.azimuth=0;camera.elevation=0;}
  else {camera.azimuth=-.58;camera.elevation=.42;}
  document.querySelectorAll("[data-view]").forEach(b=>{b.classList.toggle("active",b===button);b.setAttribute("aria-pressed",String(b===button));});
}));
scene.addEventListener("pointerdown",event=>{dragging={x:event.clientX,y:event.clientY};scene.setPointerCapture(event.pointerId);});
scene.addEventListener("pointermove",event=>{
  if(!dragging)return;
  camera.azimuth+=(event.clientX-dragging.x)*.006;
  camera.elevation=Math.max(.02,Math.min(Math.PI/2,(camera.elevation+(event.clientY-dragging.y)*.004)));
  dragging={x:event.clientX,y:event.clientY};
  document.querySelectorAll("[data-view]").forEach(b=>{b.classList.remove("active");b.setAttribute("aria-pressed","false");});
});
for(const event of ["pointerup","pointercancel","lostpointercapture"])scene.addEventListener(event,()=>dragging=null);
scene.addEventListener("wheel",event=>{event.preventDefault();camera.zoom=Math.max(.5,Math.min(2.5,camera.zoom*Math.exp(-event.deltaY*.001)));},{passive:false});
poll();requestAnimationFrame(tick);
