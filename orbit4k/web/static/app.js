const $ = q => document.querySelector(q);
const formatParams = n => n ? `${(n/1e6).toFixed(2)}M` : '—';
const fmt = (n, d=4) => Number.isFinite(Number(n)) ? Number(n).toFixed(d) : '—';
const savedKeys = ['sourcePath','outputPath','prepareConfig','trainDataPath','runDir','trainConfig'];

function savePaths(){ savedKeys.forEach(k => localStorage.setItem(`orbit4k:${k}`, $(`#${k}`).value)); }
function restorePaths(){ savedKeys.forEach(k => { const v=localStorage.getItem(`orbit4k:${k}`); if(v) $(`#${k}`).value=v; }); }
function setText(id, value){ $(id).textContent = value ?? '—'; }
function logsText(lines){ return (lines && lines.length ? lines.slice(-80).join('\n') : '—'); }

async function api(url, options={}){
  const r = await fetch(url, options);
  let data = {};
  try{ data = await r.json(); }catch{}
  if(!r.ok) throw new Error(data.detail || data.error || `${r.status} ${r.statusText}`);
  return data;
}
async function postJson(url, body){ return api(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}); }

async function loadStatus(){
  const s = await api('/api/status');
  setText('#modelName', s.model);
  setText('#params', formatParams(s.parameters));
  setText('#audioFeature', `V${s.audio_feature_version} · ${s.audio_token_dim}d`);
  setText('#gpu', s.cuda?.available ? s.cuda.device : 'CPU / CUDA unavailable');
}

function buildGrid(){
  const root=$('#gridDemo');
  for(let i=0;i<64;i++){const c=document.createElement('div');c.className='cell';root.appendChild(c)}
  setInterval(()=>{[...root.children].forEach((c,i)=>c.classList.toggle('on', Math.random()<.10 && i>7));},520);
}

function renderLoss(records=[]){
  const values=records.map(r=>r.validation?.loss ?? r.train?.loss).filter(v=>Number.isFinite(Number(v))).map(Number);
  const line=$('#lossLine');
  if(values.length<2){line.setAttribute('points','');return;}
  const w=600,h=180,p=10,min=Math.min(...values),max=Math.max(...values),span=Math.max(max-min,1e-6);
  const pts=values.map((v,i)=>{
    const x=p+(w-2*p)*(i/(values.length-1));
    const y=h-p-(h-2*p)*((v-min)/span);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  line.setAttribute('points',pts);
}

async function showDatasetSummary(path){
  const box=$('#datasetSummary');
  try{
    const data=await api(`/api/dataset-summary?path=${encodeURIComponent(path)}`), s=data.summary;
    box.classList.remove('hidden');
    box.innerHTML=`<b>Dataset Ready</b><div class="summary-grid"><span>Accepted <strong>${s.accepted_charts}</strong></span><span>Rejected <strong>${s.rejected_charts}</strong></span><span>Unique Audio <strong>${s.unique_audio}</strong></span><span>Train / Val / Test <strong>${s.splits?.train ?? 0} / ${s.splits?.validation ?? 0} / ${s.splits?.test ?? 0}</strong></span></div>`;
  }catch{ box.classList.add('hidden'); }
}

function renderPrepare(job){
  setText('#prepareState', job.state);
  const p=job.progress || {};
  setText('#prepareStage', p.stage || '—');
  const percent=Number(p.percent ?? (p.stage==='complete'?100:0));
  $('#prepareProgress').style.width=`${Math.max(0,Math.min(100,percent))}%`;
  setText('#prepareCount', p.total ? `${p.current ?? 0} / ${p.total} (${fmt(percent,1)}%)` : '—');
  setText('#prepareAudio', p.unique_audio ?? '—');
  setText('#prepareAccepted', p.accepted!=null ? `${p.accepted} / ${p.rejected ?? 0}` : '—');
  $('#prepareLog').textContent=logsText(job.logs);
  if(job.paths?.output){
    $('#outputPath').value=job.paths.output;
    if(!job.running) $('#trainDataPath').value=job.paths.output;
  }
  if(job.state==='completed' && job.paths?.output) showDatasetSummary(job.paths.output);
}

function renderTrain(job){
  setText('#trainState', job.state);
  const records=job.records || [], last=records[records.length-1] || {};
  setText('#epoch', last.epoch ?? job.progress?.epoch ?? '—');
  setText('#trainLoss', fmt(last.train?.loss ?? job.progress?.train?.loss));
  setText('#valLoss', fmt(last.validation?.loss ?? job.progress?.validation?.loss));
  setText('#lr', last.learning_rate!=null ? Number(last.learning_rate).toExponential(2) : '—');
  $('#trainLog').textContent=logsText(job.logs);
  renderLoss(records);
  if(job.paths?.data) $('#trainDataPath').value=job.paths.data;
  if(job.paths?.run_dir) $('#runDir').value=job.paths.run_dir;
}

async function pollJobs(){
  try{const data=await api('/api/jobs');renderPrepare(data.prepare);renderTrain(data.train);}catch(e){console.error(e)}
}

$('#prepareStart').addEventListener('click', async()=>{
  savePaths();
  try{
    const body={source_path:$('#sourcePath').value,output_path:$('#outputPath').value,config_path:$('#prepareConfig').value};
    const data=await postJson('/api/prepare/start',body);
    $('#trainDataPath').value=data.output;
    await pollJobs();
  }catch(e){alert(e.message)}
});
$('#prepareStop').addEventListener('click', async()=>{try{await postJson('/api/prepare/stop',{});await pollJobs()}catch(e){alert(e.message)}});
$('#trainStart').addEventListener('click', async()=>{
  savePaths();
  try{
    await postJson('/api/train/start',{data_path:$('#trainDataPath').value,run_dir:$('#runDir').value,config_path:$('#trainConfig').value});
    await pollJobs();
  }catch(e){alert(e.message)}
});
$('#trainStop').addEventListener('click', async()=>{try{await postJson('/api/train/stop',{});await pollJobs()}catch(e){alert(e.message)}});

async function inspect(file){
  const box=$('#inspectResult'); box.classList.remove('hidden'); box.innerHTML='正在检查 4K / timing / 1/96 量化…';
  const fd=new FormData(); fd.append('file',file);
  try{
    const data=await api('/api/inspect-zip',{method:'POST',body:fd});
    if(!data.ok){box.textContent=data.error;return}
    const rows=data.accepted.map(x=>`<div class="map-row"><b>${x.version}</b><span>${x.bpm} BPM</span><span>${x.offset_ms} ms</span><span>${x.stars ?? 'SR ?'} ★</span><span>p95 ${x.p95_error_ms} ms</span></div>`).join('');
    box.innerHTML=`<b>通过 ${data.accepted.length} 张 · 拒绝 ${data.rejected.length} 张</b>${rows || '<p>没有可用的 mania 4K 谱。</p>'}${data.rejected.length?`<p class="note">首个拒绝原因：${data.rejected[0].reason}</p>`:''}`;
  }catch(e){box.textContent=e.message}
}
const input=$('#zipInput'), drop=$('#dropZone');
input.addEventListener('change',()=>input.files[0]&&inspect(input.files[0]));
['dragenter','dragover'].forEach(e=>drop.addEventListener(e,x=>{x.preventDefault();drop.classList.add('drag')}));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,x=>{x.preventDefault();drop.classList.remove('drag')}));
drop.addEventListener('drop',e=>e.dataTransfer.files[0]&&inspect(e.dataTransfer.files[0]));

restorePaths(); buildGrid(); loadStatus(); pollJobs(); setInterval(pollJobs,1000);
