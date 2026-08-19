const $ = (q) => document.querySelector(q);
const formatParams = n => n ? `${(n/1e6).toFixed(2)}M` : '—';

async function loadStatus(){
  const r = await fetch('/api/status'); const s = await r.json();
  $('#modelName').textContent = s.model;
  $('#params').textContent = formatParams(s.parameters);
  $('#grid').textContent = s.grid;
  $('#checkpoint').textContent = s.checkpoint_ready ? 'Ready' : 'Not trained';
}

function buildGrid(){
  const root=$('#gridDemo');
  for(let i=0;i<64;i++){const c=document.createElement('div');c.className='cell';root.appendChild(c)}
  setInterval(()=>{
    [...root.children].forEach((c,i)=>c.classList.toggle('on', Math.random()<.10 && i>7));
  },520);
}

async function inspect(file){
  const box=$('#inspectResult'); box.classList.remove('hidden'); box.innerHTML='正在检查 4K / timing / 1/96 量化…';
  const fd=new FormData(); fd.append('file',file);
  const r=await fetch('/api/inspect-zip',{method:'POST',body:fd}); const data=await r.json();
  if(!data.ok){box.textContent=data.error;return}
  const rows=data.accepted.map(x=>`<div class="map-row"><b>${x.version}</b><span>${x.bpm} BPM</span><span>${x.offset_ms} ms</span><span>${x.stars ?? 'SR ?'} ★</span><span>p95 ${x.p95_error_ms} ms</span></div>`).join('');
  box.innerHTML=`<b>通过 ${data.accepted.length} 张 · 拒绝 ${data.rejected.length} 张</b>${rows || '<p>没有可用的 mania 4K 谱。</p>'}${data.rejected.length?`<p class="note">首个拒绝原因：${data.rejected[0].reason}</p>`:''}`;
}

const input=$('#zipInput'), drop=$('#dropZone');
input.addEventListener('change',()=>input.files[0]&&inspect(input.files[0]));
['dragenter','dragover'].forEach(e=>drop.addEventListener(e,x=>{x.preventDefault();drop.classList.add('drag')}));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,x=>{x.preventDefault();drop.classList.remove('drag')}));
drop.addEventListener('drop',e=>e.dataTransfer.files[0]&&inspect(e.dataTransfer.files[0]));
buildGrid(); loadStatus();
