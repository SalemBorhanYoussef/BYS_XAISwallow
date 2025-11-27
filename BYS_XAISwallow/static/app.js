/* static/app.js – Pfad-basierte Variante ohne Upload */
(() => {
  const $ = (id) => document.getElementById(id);
  const parseNum = (el, fb=0) => { const v = parseFloat(el.value); return isNaN(v)?fb:v; };
  const parseIntVal = (el, fb=0) => { const v = parseInt(el.value,10); return isNaN(v)?fb:v; };

  // Elemente
  const elCkpt = $("ckpt"), elModelType=$("modelType"), elTfast=$("tfast"), elAlpha=$("alpha"), elWinsz=$("winsz"), elStride=$("stride"), elRh=$("rh"), elRw=$("rw"), elFps=$("fps");
  const elYoloWeights=$("yoloWeights"), elYoloConf=$("yoloConf"), elRoiSize=$("roiSize"), elRoiMode=$("roiMode"), elFixedX=$("fixedX"), elFixedY=$("fixedY"), elYoloOverlay=$("yoloOverlay"), elFaceMaskMode=$("faceMaskMode");
  const elBtnLoad=$("btnLoad"), elLoadStatus=$("loadStatus"), elBtnUsePath=$("btnUsePath"), elPathInput=$("pathInput"), elBtnStop=$("btnStop");
  const elMetaFps=$("metaFps"), elMetaFrames=$("metaFrames"), elMetaDur=$("metaDur");
  const elBtnPlay=$("btnPlay"), elBtnPause=$("btnPause"), elBtnRestart=$("btnRestart"), elSeek=$("seek"), elTimeLabel=$("timeLabel"), elLastVal=$("lastVal"), elChart=$("chart");

  let meta={fps:32,frames:0,duration:0};
  let isPlaying=false;

  // Chart
  const chart=new Chart(elChart.getContext('2d'),{type:'scatter',data:{datasets:[
    {label:'Prediction p(t)',data:[],showLine:true,pointRadius:0,borderWidth:2,tension:0,borderColor:'#2563eb'},
    {label:'Seek marker',data:[],showLine:false,pointRadius:4,borderWidth:0,backgroundColor:'#dc2626'}]},
    options:{responsive:true,maintainAspectRatio:false,scales:{x:{type:'linear',title:{display:true,text:'t (s)'}},y:{min:0,max:1,title:{display:true,text:'p'},ticks:{stepSize:0.1}}},animation:false,plugins:{legend:{display:true},tooltip:{callbacks:{label:(c)=>`t=${c.raw.x.toFixed(2)}s  p=${c.raw.y.toFixed(3)}`}}}}});
  function chartAddPoint(t,p,kind='live'){ if(kind==='live'){ chart.data.datasets[0].data.push({x:t,y:p}); if(chart.data.datasets[0].data.length>6000){chart.data.datasets[0].data.splice(0,1000);} } else { chart.data.datasets[1].data=[{x:t,y:p}]; } chart.update('none'); }

  // Socket
  const socket=io();
  socket.on('meta',m=>{meta=m; elMetaFps.textContent=m.fps.toFixed(3); elMetaFrames.textContent=m.frames; elMetaDur.textContent=m.duration.toFixed(2); elSeek.max=m.duration.toFixed(3); tuneSeekResolution();});
  socket.on('pred',msg=>{
    if(typeof msg.t==='number' && typeof msg.p_smooth==='number'){
      chartAddPoint(msg.t,msg.p_smooth,'live');
      if(isPlaying){ elSeek.value=msg.t.toFixed(3); updateTimeLabel(); }
      const gt = (msg.gt===0||msg.gt===1)? msg.gt : '-';
      elLastVal.textContent = `t=${msg.t.toFixed(2)}s  p_raw=${(msg.p_raw||0).toFixed(3)}  p=${msg.p_smooth.toFixed(3)}  thr=${(msg.thresh||0).toFixed(2)}  event=${msg.event}  gt=${gt}`;
    }
  });

  function updateTimeLabel(){ const t=parseFloat(elSeek.value)||0; elTimeLabel.textContent=`t=${t.toFixed(2)}s`; }
  function tuneSeekResolution(){ const dur=meta.duration||0; let step=0.04; if(dur>600) step=0.2; else if(dur>300) step=0.1; else if(dur>120) step=0.05; elSeek.step=step.toFixed(3); }

  // Modell laden
  elBtnLoad.addEventListener('click',async()=>{
    elLoadStatus.textContent='Lade...';
    const body={
      ckpt:elCkpt.value.trim(),
      model:elModelType.value,
      fps:parseNum(elFps,32),
      t_fast:parseIntVal(elTfast,32),
      alpha:parseIntVal(elAlpha,4),
      window_size:parseIntVal(elWinsz,32),
      test_stride:parseIntVal(elStride,8),
      resize_h:244,
      resize_w:244,
      yolo_weights:elYoloWeights.value.trim(),
      yolo_conf:parseNum(elYoloConf,0.25),
      roi_size:244,
      roi_mode:'first_frame_pose',
      fixed_neck_xy:null,
      yolo_overlay:false,
      face_mask_mode:'off',
      smooth_prob:7
    };
    const r=await fetch('/load_model',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const j=await r.json();
    if(!j.ok){ elLoadStatus.textContent='Fehler: '+(j.error||r.status); return; }
    elLoadStatus.textContent='OK';
  });

  // Start per Pfad
  elBtnUsePath.addEventListener('click',async()=>{ const p=elPathInput.value.trim(); if(!p) return; if(!meta||!meta.fps){ /* no-op */ } isPlaying=true; const r=await fetch('/use_path',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:p})}); let j=null; try{ j=await r.json(); }catch(e){ alert('Antwort kein JSON'); return; } if(!j.ok){ alert('Start fehlgeschlagen: '+(j.error||r.status)); return;} setTimeout(refreshMeta,500); });
  async function refreshMeta(){ const r=await fetch('/video_meta'); if(r.ok){ const j=await r.json(); if(j.ok){ meta=j; elMetaFps.textContent=j.fps.toFixed(3); elMetaFrames.textContent=j.frames; elMetaDur.textContent=j.duration.toFixed(2); elSeek.max=j.duration.toFixed(3); tuneSeekResolution(); }}};

  // Player
  elBtnStop.addEventListener('click',async()=>{ await fetch('/stop',{method:'POST'}); isPlaying=false; });
  elBtnPlay.addEventListener('click',()=>{ isPlaying=true; socket.emit('control',{action:'play'}); });
  elBtnPause.addEventListener('click',()=>{ isPlaying=false; socket.emit('control',{action:'pause'}); });
  elBtnRestart.addEventListener('click',()=>{ socket.emit('control',{action:'replay'}); chart.data.datasets[0].data=[]; chart.data.datasets[1].data=[]; chart.update('none'); elSeek.value=0; updateTimeLabel(); });

  // Seek
  elSeek.addEventListener('input',updateTimeLabel);
  elSeek.addEventListener('change',()=>{ const t=parseFloat(elSeek.value)||0; socket.emit('control',{action:'seek',t}); });

  // ROI Mode fix XY enable
  function toggleFixed(){ const man=elRoiMode.value==='manual'; elFixedX.disabled=elFixedY.disabled=!man; document.getElementById('fixedXYWrap').style.opacity=man?'1':'.4'; }
  elRoiMode.addEventListener('change',toggleFixed); toggleFixed();

  updateTimeLabel();
})();
