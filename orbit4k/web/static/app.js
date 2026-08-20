const $ = q => document.querySelector(q);
const formatParams = n => n ? `${(n/1e6).toFixed(2)}M` : '—';
const fmt = (n, d=4) => Number.isFinite(Number(n)) ? Number(n).toFixed(d) : '—';
const savedKeys = [
  'sourcePath','outputPath','prepareConfig','trainArchitecture','trainDataPath','runDir','trainConfig','trainWarmStart',
  'genCheckpoint','genAudio','genOutput','genBpm','genOffset','genStars',
  'genMode','genTemperature','genMeasures','genWindowMeasures','genContextMeasures','genSeed'
];
let datasetReady = false;
let trainRunning = false;
let generateRunning = false;

function savePaths(){ savedKeys.forEach(k => { const el=$(`#${k}`); if(el) localStorage.setItem(`orbit4k:${k}`, el.value); }); }
function restorePaths(){ savedKeys.forEach(k => { const el=$(`#${k}`), v=localStorage.getItem(`orbit4k:${k}`); if(el&&v!=null) el.value=v; }); }
function setText(id, value){ const el=$(id); if(el) el.textContent = value ?? '—'; }
function logsText(lines){ return (lines && lines.length ? lines.slice(-80).join('\n') : '—'); }
function updateTrainButton(){ $('#trainStart').disabled = trainRunning || generateRunning || !datasetReady; }
function updateGenerateButton(){ $('#genStart').disabled = generateRunning || trainRunning; }
function updateGenerationModeUI(){
  const full=$('#genMode').value==='full_song';
  $('#previewSettings').classList.toggle('hidden', full);
  $('#fullSongSettings').classList.toggle('hidden', !full);
  $('#genStart').textContent=full?'生成整首谱面':'生成预览谱';
}
function updateTrainingArchitectureUI(forceDefaults=false){
  const v1=$('#trainArchitecture').value==='v1';
  $('#trainWarmStart').disabled=!v1;
  if(forceDefaults){
    $('#runDir').value=v1?'runs/v1':'runs/v0';
    $('#trainConfig').value=v1?'configs/v1.yaml':'configs/v0.yaml';
    if(v1 && !$('#trainWarmStart').value.trim()) $('#trainWarmStart').value='runs/v0/best.pt';
  }
}

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
    box.innerHTML=`<b>Dataset Ready</b><div class="summary-grid"><span>Accepted <strong>${s.accepted_charts}</strong></span><span>Rejected <strong>${s.rejected_charts}</strong></span><span>Unique Audio <strong>${s.unique_audio}</strong></span><span>Failed Audio <strong>${s.failed_unique_audio ?? 0}</strong></span><span>Train / Val / Test <strong>${s.splits?.train ?? 0} / ${s.splits?.validation ?? 0} / ${s.splits?.test ?? 0}</strong></span></div>`;
  }catch{ box.classList.add('hidden'); }
}

async function refreshDatasetReadiness(){
  const box=$('#datasetReadiness');
  const path=$('#trainDataPath').value.trim();
  if(!path){
    datasetReady=false;
    box.className='summary-box warning';
    box.innerHTML='<b>Dataset 路径为空</b><p>先指定清洗输出目录。</p>';
    updateTrainButton();
    return;
  }
  try{
    const data=await api(`/api/dataset-status?path=${encodeURIComponent(path)}`);
    datasetReady=Boolean(data.ready);
    const s=data.summary || {}, st=data.state || {};
    if(data.ready){
      box.className='summary-box ready';
      const arch=$('#trainArchitecture').value==='v1'?'V1 32×3 会直接复用这份 1/96 cache':'V0 flat 1/96';
      box.innerHTML=`<b>✓ Dataset Ready · ${arch}</b><div class="summary-grid"><span>Accepted <strong>${s.accepted_charts ?? '—'}</strong></span><span>Rejected <strong>${s.rejected_charts ?? '—'}</strong></span><span>Unique Audio <strong>${s.unique_audio ?? '—'}</strong></span><span>Train / Val / Test <strong>${s.splits?.train ?? 0} / ${s.splits?.validation ?? 0} / ${s.splits?.test ?? 0}</strong></span></div>`;
    }else{
      const processed=st.processed ?? 0, total=st.total_scanned_accepted ?? '—';
      const partial=(data.partial_index_exists || data.partial_rejected_exists) ? '有 partial checkpoint，可复用已有缓存' : '尚无 partial checkpoint';
      const err=st.last_error ? `<p><strong>Last error:</strong> ${st.last_error}</p>` : '';
      box.className='summary-box warning';
      box.innerHTML=`<b>⚠ Dataset ${String(data.status).toUpperCase()} · 训练已锁定</b><div class="summary-grid"><span>Processed <strong>${processed} / ${total}</strong></span><span>Final index <strong>${data.index_exists?'✓':'✗'}</strong></span><span>Summary <strong>${data.summary_exists?'✓':'✗'}</strong></span><span>${partial}</span></div>${err}`;
    }
  }catch(e){
    datasetReady=false;
    box.className='summary-box warning';
    box.innerHTML=`<b>⚠ Dataset 状态无法读取 · 训练已锁定</b><p>${e.message}</p>`;
  }
  updateTrainButton();
}

function renderPrepare(job){
  setText('#prepareState', job.state);
  const p=job.progress || {};
  setText('#prepareStage', p.stage || '—');
  const percent=Number(p.percent ?? (p.stage==='complete'?100:0));
  $('#prepareProgress').style.width=`${Math.max(0,Math.min(100,percent))}%`;
  setText('#prepareCount', p.total ? `${p.current ?? 0} / ${p.total} (${fmt(percent,1)}%)` : '—');
  setText('#prepareAudio', p.unique_audio ?? '—');
  const extra=p.failed_unique_audio ? ` · bad audio ${p.failed_unique_audio}` : '';
  setText('#prepareAccepted', p.accepted!=null ? `${p.accepted} / ${p.rejected ?? 0}${extra}` : '—');
  $('#prepareLog').textContent=logsText(job.logs);
  const failure=$('#prepareFailure');
  if(job.state==='failed'){
    failure.classList.remove('hidden');
    failure.innerHTML=`<b>Dataset Builder failed</b><p>${job.failure_reason || p.last_error || '查看下方日志获取异常原因。'}</p><small>已经写入的 audio/chart cache 与 partial checkpoint 会保留。</small>`;
  }else failure.classList.add('hidden');
  if(job.paths?.output){
    $('#outputPath').value=job.paths.output;
    if(!job.running) $('#trainDataPath').value=job.paths.output;
  }
  if(job.state==='completed' && job.paths?.output) showDatasetSummary(job.paths.output);
}

function renderTrain(job){
  trainRunning=Boolean(job.running);
  setText('#trainState', job.state);
  const records=job.records || [], last=records[records.length-1] || {};
  const arch=last.architecture ?? job.progress?.architecture ?? job.paths?.architecture ?? $('#trainArchitecture').value;
  const memory=last.gpu_memory ?? job.progress?.gpu_memory ?? {};
  setText('#trainArch', arch==='v1_32x3' || arch==='v1' ? 'V1 · 32×3' : (arch==='v0' ? 'V0 · 1/96' : arch || '—'));
  setText('#epoch', last.epoch ?? job.progress?.epoch ?? '—');
  setText('#trainLoss', fmt(last.train?.loss ?? job.progress?.train?.loss));
  setText('#valLoss', fmt(last.validation?.loss ?? job.progress?.validation?.loss));
  setText('#lr', last.learning_rate!=null ? Number(last.learning_rate).toExponential(2) : '—');
  setText('#peakVram', memory.peak_allocated_mb!=null ? `${fmt(memory.peak_allocated_mb,1)} MB` : '—');
  $('#trainLog').textContent=logsText(job.logs);
  renderLoss(records);
  if(job.paths?.architecture){ $('#trainArchitecture').value=job.paths.architecture; updateTrainingArchitectureUI(false); }
  if(job.paths?.data) $('#trainDataPath').value=job.paths.data;
  if(job.paths?.run_dir){
    $('#runDir').value=job.paths.run_dir;
    if(job.state==='completed' && job.paths?.architecture==='v0') $('#genCheckpoint').value=`${job.paths.run_dir}\\best.pt`;
  }
  if(job.paths?.config) $('#trainConfig').value=job.paths.config;
  if(job.paths?.warm_start_v0 && job.paths.warm_start_v0.trim()) $('#trainWarmStart').value=job.paths.warm_start_v0;
  updateTrainButton();
  updateGenerateButton();
}

function renderGenerate(job){
  generateRunning=Boolean(job.running);
  const p=job.progress || {}, stats=p.stats || {};
  const mode=p.mode || job.paths?.mode || $('#genMode').value || 'preview';
  const full=mode==='full_song';
  setText('#genState', job.state);
  setText('#genStage', p.stage || '—');
  const percent=Number(p.percent ?? (p.stage==='complete'?100:0));
  $('#genProgress').style.width=`${Math.max(0,Math.min(100,percent))}%`;
  if(full && p.total){
    const windowInfo=p.total_windows ? ` · window ${p.window ?? 0}/${p.total_windows}` : '';
    setText('#genTicks', `${p.current ?? 0} / ${p.total} (${fmt(percent,1)}%)${windowInfo}`);
  }else{
    setText('#genTicks', p.total ? `${p.current ?? 0} / ${p.total}` : (p.generated_ticks ?? '—'));
  }
  setText('#genKeydowns', stats.keydowns ?? '—');
  setText('#genRepairs', stats.repairs ?? '—');
  $('#genLog').textContent=logsText(job.logs);

  const result=$('#genResult'), failure=$('#genFailure');
  if(job.state==='completed' && p.output){
    const extra=full
      ? `<span>Length <strong>${fmt(stats.duration_seconds,1)} s / ${fmt(stats.generated_measures,1)} m</strong></span><span>Windows <strong>${p.windows ?? '—'}</strong></span>`
      : '';
    result.className='summary-box ready';
    result.innerHTML=`<b>✓ ${full?'Full Song':'Preview'} Generated</b><p><strong>.osu:</strong> ${p.output}</p><div class="summary-grid"><span>Checkpoint epoch <strong>${p.checkpoint_epoch ?? '—'}</strong></span><span>SR <strong>${p.stars ?? '—'}</strong></span><span>Keydowns <strong>${stats.keydowns ?? '—'}</strong></span><span>Tap / LN <strong>${stats.taps ?? 0} / ${stats.ln ?? 0}</strong></span><span>Chord ticks <strong>${stats.chord_ticks ?? 0}</strong></span><span>Repairs <strong>${stats.repairs ?? 0}</strong></span>${extra}</div>`;
    result.classList.remove('hidden');
  }else if(job.running){
    result.classList.add('hidden');
  }
  if(job.state==='failed'){
    failure.classList.remove('hidden');
    failure.innerHTML=`<b>${full?'Full-song':'Preview'} generation failed</b><p>${job.failure_reason || p.last_error || '查看生成日志。'}</p>`;
  }else failure.classList.add('hidden');

  if(job.paths?.mode){ $('#genMode').value=job.paths.mode; updateGenerationModeUI(); }
  if(job.paths?.checkpoint) $('#genCheckpoint').value=job.paths.checkpoint;
  if(job.paths?.audio) $('#genAudio').value=job.paths.audio;
  if(job.paths?.output_dir) $('#genOutput').value=job.paths.output_dir;
  updateTrainButton();
  updateGenerateButton();
}

async function pollJobs(){
  try{
    const data=await api('/api/jobs');
    renderPrepare(data.prepare);
    renderTrain(data.train);
    renderGenerate(data.generate || {state:'idle',progress:{},logs:[],paths:{}});
    await refreshDatasetReadiness();
  }catch(e){console.error(e)}
}

$('#prepareStart').addEventListener('click', async()=>{
  savePaths();
  datasetReady=false; updateTrainButton();
  $('#datasetReadiness').className='summary-box warning';
  $('#datasetReadiness').innerHTML='<b>Dataset Builder 正在启动…训练已锁定</b>';
  try{
    const body={source_path:$('#sourcePath').value,output_path:$('#outputPath').value,config_path:$('#prepareConfig').value};
    const data=await postJson('/api/prepare/start',body);
    $('#trainDataPath').value=data.output;
    await pollJobs();
  }catch(e){alert(e.message);await refreshDatasetReadiness()}
});
$('#prepareStop').addEventListener('click', async()=>{try{await postJson('/api/prepare/stop',{});await pollJobs()}catch(e){alert(e.message)}});

$('#trainArchitecture').addEventListener('change',()=>{
  updateTrainingArchitectureUI(true);
  savePaths();
  refreshDatasetReadiness();
});
$('#trainStart').addEventListener('click', async()=>{
  savePaths();
  if(!datasetReady) return;
  try{
    await postJson('/api/train/start',{
      architecture:$('#trainArchitecture').value,
      data_path:$('#trainDataPath').value,
      run_dir:$('#runDir').value,
      config_path:$('#trainConfig').value,
      warm_start_v0:$('#trainArchitecture').value==='v1' ? $('#trainWarmStart').value : null,
    });
    await pollJobs();
  }catch(e){alert(e.message)}
});
$('#trainStop').addEventListener('click', async()=>{try{await postJson('/api/train/stop',{});await pollJobs()}catch(e){alert(e.message)}});
$('#trainDataPath').addEventListener('change', refreshDatasetReadiness);
$('#trainDataPath').addEventListener('blur', refreshDatasetReadiness);

$('#genMode').addEventListener('change',()=>{ updateGenerationModeUI(); savePaths(); });
$('#genStart').addEventListener('click', async()=>{
  savePaths();
  $('#genResult').classList.add('hidden');
  try{
    await postJson('/api/generate/start',{
      mode:$('#genMode').value,
      checkpoint_path:$('#genCheckpoint').value,
      audio_path:$('#genAudio').value,
      output_dir:$('#genOutput').value,
      bpm:Number($('#genBpm').value),
      offset_ms:Number($('#genOffset').value),
      stars:Number($('#genStars').value),
      temperature:Number($('#genTemperature').value),
      measures:Number($('#genMeasures').value),
      window_measures:Number($('#genWindowMeasures').value),
      context_measures:Number($('#genContextMeasures').value),
      seed:Number($('#genSeed').value),
    });
    await pollJobs();
  }catch(e){alert(e.message)}
});
$('#genStop').addEventListener('click', async()=>{try{await postJson('/api/generate/stop',{});await pollJobs()}catch(e){alert(e.message)}});

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

const hadSavedArchitecture=localStorage.getItem('orbit4k:trainArchitecture')!=null;
restorePaths();
updateTrainingArchitectureUI(!hadSavedArchitecture);
updateGenerationModeUI();
buildGrid(); loadStatus(); pollJobs(); setInterval(pollJobs,1000);
