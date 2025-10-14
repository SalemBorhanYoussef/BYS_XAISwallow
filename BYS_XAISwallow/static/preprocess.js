// static/preprocess.js – Auto Neck & Ellipse Mask
(() => {
  const state = { path:null, img:null, imgW:0, imgH:0, frames:0, fps:0, cx:-1, cy:-1, ellipse:null, autoEllipse:null };
  const $ = (q)=>document.querySelector(q);
  const btnFirst=$('#btnFirst'); // optional (kann entfallen)
  const little=$('#little'), fileInfo=$('#fileInfo'), meta=$('#meta');
  const cnv=$('#cnv'), ctx=cnv.getContext('2d');
  const cnvMasked=$('#cnvMasked'), ctxMasked=cnvMasked.getContext('2d');
  const cnvRoi=$('#cnvRoi'), ctxRoi=cnvRoi.getContext('2d');
  const ptLabel=$('#ptLabel'), roiSize=$('#roiSize'), btnExport=$('#btnExport'), statusEl=$('#status'), links=$('#links');
  const roiMaskExport=$('#roiMaskExport');
  const roiMaskShow=$('#roiMaskShow');
  const manualPath=$('#manualPath'), btnUseManual=$('#btnUseManual'), btnBrowse=$('#btnBrowse');
  const browserPanel=$('#browserPanel'), brCur=$('#brCur'), brList=$('#brList'), brUp=$('#brUp'), brFilter=$('#brFilter');
  const neckMode=$('#neckMode'), btnAuto=$('#btnAuto'), yoloWeights=$('#yoloWeights'), yoloConf=$('#yoloConf');
  const maskEnabled=$('#maskEnabled');
  const editModeRadios = [...document.querySelectorAll('input[name="editMode"]')];
  function currentEditMode(){ const r=editModeRadios.find(r=>r.checked); return r? r.value : 'roi'; }
  
  // Ellipse interaction state
  let ellDragging=false; let ellDragOffset={dx:0,dy:0};
  let ellRotating=false; // rotation via handle
  const FACE_RATIO = 1.41; // height / width (axes[1]/axes[0])
  function enforceFaceRatio(){
    if(!state.ellipse) return;
    let ax1 = state.ellipse.axes[0];
    if(ax1 < 5) ax1 = 5;
    let ax2 = ax1 * FACE_RATIO;
    state.ellipse.axes[0] = Math.round(ax1);
    state.ellipse.axes[1] = Math.round(Math.max(5, Math.min(2000, ax2)));
  }

  function redraw(){
    if(!state.img) return;
    const scaleX = cnv.width/state.imgW;
    const scaleY = cnv.height/state.imgH;
    // Base
    ctx.clearRect(0,0,cnv.width,cnv.height);
    ctx.drawImage(state.img,0,0,cnv.width,cnv.height);
    // ROI overlay
    const rSize = parseInt(roiSize.value,10)||224;
    if(state.cx>=0 && state.cy>=0){
      const half = rSize/2;
      const rx = state.cx * scaleX, ry = state.cy * scaleY;
      const x1 = (state.cx - half) * scaleX, y1 = (state.cy - half) * scaleY;
      ctx.strokeStyle='#3399ff'; ctx.lineWidth=2; ctx.setLineDash([6,4]);
      ctx.strokeRect(x1,y1, rSize*scaleX, rSize*scaleY);
      ctx.setLineDash([]);
      ctx.strokeStyle='#00ff00'; ctx.fillStyle='#00ff00';
      ctx.beginPath(); ctx.arc(rx,ry,6,0,Math.PI*2); ctx.fill();
      ctx.beginPath(); ctx.moveTo(rx,0); ctx.lineTo(rx,cnv.height); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0,ry); ctx.lineTo(cnv.width,ry); ctx.stroke();
    }
    // Ellipse overlay
    const ell = state.ellipse;
    if(maskEnabled.checked && ell){
      const [cxE,cyE]=ell.center; const [ax1E,ax2E]=ell.axes; const angDeg=ell.angle; const ang=angDeg*Math.PI/180;
      ctx.save(); ctx.translate(cxE*scaleX, cyE*scaleY); ctx.rotate(ang);
      ctx.strokeStyle='#ff00ff'; ctx.lineWidth=2;
      ctx.beginPath(); ctx.ellipse(0,0, ax1E*scaleX, ax2E*scaleY,0,0,Math.PI*2); ctx.stroke();
      // Draw cross axes for orientation
      ctx.strokeStyle='rgba(255,0,255,0.4)'; ctx.setLineDash([4,4]);
      ctx.beginPath(); ctx.moveTo(-ax1E*scaleX,0); ctx.lineTo(ax1E*scaleX,0); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0,-ax2E*scaleY); ctx.lineTo(0,ax2E*scaleY); ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
      // Rotation handle (at top of ellipse AFTER rotation)
      const hx = cxE + ax2E*Math.sin(ang); // rotated (0,-ax2E)
      const hy = cyE - ax2E*Math.cos(ang);
      const hxS = hx*scaleX, hyS=hy*scaleY;
      ctx.strokeStyle='#ffa500'; ctx.fillStyle= ellRotating ? '#ff8800' : '#ffa500';
      ctx.lineWidth=2;
      ctx.beginPath(); ctx.moveTo(cxE*scaleX, cyE*scaleY); ctx.lineTo(hxS, hyS); ctx.stroke();
      ctx.beginPath(); ctx.arc(hxS, hyS, 8, 0, Math.PI*2); ctx.fill(); ctx.stroke();
      // Angle label
      ctx.font='12px monospace'; ctx.fillStyle='#ffa500'; ctx.textAlign='center';
      ctx.fillText(`${Math.round(angDeg)}°`, hxS, hyS-12);
    }
    // Masked full image (no canvas resize each frame)
    ctxMasked.clearRect(0,0,cnvMasked.width,cnvMasked.height);
    ctxMasked.drawImage(state.img,0,0,cnvMasked.width,cnvMasked.height);
    if(maskEnabled.checked && ell){
      ctxMasked.save();
      ctxMasked.translate(ell.center[0]*scaleX, ell.center[1]*scaleY);
      ctxMasked.rotate(ell.angle*Math.PI/180);
      ctxMasked.beginPath();
      ctxMasked.ellipse(0,0, ell.axes[0]*scaleX, ell.axes[1]*scaleY,0,0,Math.PI*2);
      ctxMasked.fillStyle='#000'; ctxMasked.fill();
      ctxMasked.restore();
    }
    // ROI canvas draw via source rect
    const displayMax = 256;
    if(state.cx>=0 && state.cy>=0){
      const half = rSize/2;
      const sx = Math.max(0, Math.min(state.imgW - rSize, state.cx - half));
      const sy = Math.max(0, Math.min(state.imgH - rSize, state.cy - half));
      const scaleR = rSize>displayMax? displayMax/rSize : 1;
      const rw = Math.round(rSize*scaleR), rh=Math.round(rSize*scaleR);
      if(cnvRoi.width!==rw||cnvRoi.height!==rh){ cnvRoi.width=rw; cnvRoi.height=rh; }
      ctxRoi.clearRect(0,0,rw,rh);
      ctxRoi.imageSmoothingEnabled=true;
      ctxRoi.drawImage(state.img, sx, sy, rSize, rSize, 0,0,rw,rh);
      if(maskEnabled.checked && ell && roiMaskShow.checked){
        // If ellipse intersects ROI, paint its black region relative coordinates
        const [cxE,cyE]=ell.center; const [ax1E,ax2E]=ell.axes; const ang=ell.angle*Math.PI/180;
        const ellLeft=cxE-ax1E, ellRight=cxE+ax1E, ellTop=cyE-ax2E, ellBot=cyE+ax2E;
        if(!(ellRight < sx || ellLeft > sx+rSize || ellBot < sy || ellTop > sy+rSize)){
          ctxRoi.save();
          ctxRoi.translate( (cxE - sx)*scaleR, (cyE - sy)*scaleR );
          ctxRoi.rotate(ang);
          ctxRoi.beginPath();
          ctxRoi.ellipse(0,0, ax1E*scaleR, ax2E*scaleR,0,0,Math.PI*2);
          ctxRoi.fillStyle='#000'; ctxRoi.fill();
          ctxRoi.restore();
        }
      }
    } else {
      ctxRoi.clearRect(0,0,cnvRoi.width,cnvRoi.height);
    }
  }
  function setImage(url){
    const im=new Image();
    im.onload=()=>{
      const maxW=900;
      const scale=im.width>maxW?maxW/im.width:1;
      const cw=Math.round(im.width*scale);
      const ch=Math.round(im.height*scale);
      [cnv, cnvMasked].forEach(c=>{ c.width=cw; c.height=ch; c.style.width=cw+'px'; c.style.height=ch+'px'; });
      state.img=im;
      redraw();
    };
    im.src=url;
  }

  btnUseManual.addEventListener('click',()=>{ const p=manualPath.value.trim(); if(!p) return; selectPath(p); });
  btnBrowse.addEventListener('click',()=>{ browserPanel.style.display = browserPanel.style.display==='none'?'block':'none'; if(browserPanel.style.display==='block'){ loadDir(manualPath.value.trim()||'.'); }});
  brUp.addEventListener('click',()=>{ loadDir(brCur.dataset.pathParent || brCur.textContent); });
  brFilter.addEventListener('input',()=> filterBrowser());

  async function loadDir(p){ try{ const params=new URLSearchParams({path:p,ext:'.ravi,.mp4'}); const r=await fetch('/preprocess/browse?'+params.toString()); const j=await r.json(); if(!j.ok){ fileInfo.textContent='Browse Fehler: '+(j.error||''); return;} brCur.textContent=j.path; brCur.dataset.pathParent=j.parent; renderEntries(j.entries); }catch(e){ fileInfo.textContent='Browse Fehler'; }}
  function renderEntries(entries){ brList.innerHTML=''; const f=brFilter.value.trim().toLowerCase(); entries.forEach(ent=>{ if(f && !ent.name.toLowerCase().includes(f)) return; const li=document.createElement('li'); li.style.cursor='pointer'; li.textContent= ent.dir? '📁 '+ent.name : '📄 '+ent.name; li.dataset.path=ent.path; li.dataset.dir=ent.dir; li.addEventListener('click',()=>{ if(ent.dir){ loadDir(ent.path); } else { manualPath.value=ent.path; selectPath(ent.path,true); } }); brList.appendChild(li); }); }
  function filterBrowser(){ const items=[...brList.querySelectorAll('li')]; const f=brFilter.value.trim().toLowerCase(); items.forEach(li=>{ li.style.display = (!f || li.textContent.toLowerCase().includes(f)) ? '' : 'none'; }); }
  function selectPath(p, autoClose=false){ resetState(); state.path=p; manualPath.value=p; fileInfo.textContent='Pfad gesetzt'; if(btnFirst) btnFirst.disabled=false; // Direkt erstes Frame laden
    loadFirstFrame(); if(autoClose) browserPanel.style.display='none'; }

  function resetState(){ links.innerHTML=''; ptLabel.textContent='Kein Punkt gewählt'; state.cx=state.cy=-1; state.ellipse=null; state.autoEllipse=null; statusEl.textContent=''; }

  if(btnFirst) btnFirst.addEventListener('click',loadFirstFrame);
  async function loadFirstFrame(){ if(!state.path) return; if(btnFirst) btnFirst.disabled=true; meta.textContent='Lade erstes Frame…'; const r=await fetch('/preprocess/first_frame',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:state.path,little_endian:little.checked})}); const j=await r.json(); if(!j.ok){ meta.textContent='Fehler: '+j.error; if(btnFirst) btnFirst.disabled=false; return;} const {w,h,frames,fps}=j.meta; state.imgW=w; state.imgH=h; state.frames=frames; state.fps=fps; meta.textContent=`Quelle: ${w}×${h}, Frames=${frames}, FPS≈${fps.toFixed(3)}`; setImage(j.image); btnExport.disabled=false; if(btnFirst) btnFirst.disabled=false; btnAuto.disabled=false; }

  // Canvas Klick = ROI Center setzen (unabhängig von Neck-Modus, solange Edit=ROI)
  cnv.addEventListener('click',ev=>{
    if(!state.img) return;
    if(dragging || ellDragging) return;
    if(currentEditMode()!=='roi') return;
    const {sx,sy}=canvasToSrc(ev);
    setCenter(sx,sy,true);
  });

  // --- Manuelles ROI Drag ---
  let dragging=false; let dragOffset={dx:0,dy:0};
  function canvasToSrc(ev){
    const rect=cnv.getBoundingClientRect();
    const x=ev.clientX-rect.left; const y=ev.clientY-rect.top;
    const sx = x*(state.imgW/rect.width);
    const sy = y*(state.imgH/rect.height);
    return { sx, sy };
  }
  function setCenter(sx,sy,label=false){
    let rSize=parseInt(roiSize.value,10)||224;
    rSize = Math.min(rSize, state.imgW, state.imgH);
    const half=rSize/2;
    // Clamp so ROI stays fully inside image
    const minX=half, maxX=state.imgW-half, minY=half, maxY=state.imgH-half;
    state.cx=Math.round(Math.max(minX,Math.min(maxX,sx)));
    state.cy=Math.round(Math.max(minY,Math.min(maxY,sy)));
    // (Synchronisation Ellipse-Zentrum mit ROI entfernt – jetzt vollständig unabhängig)
    if(label) ptLabel.textContent=`Punkt: (${state.cx}, ${state.cy})`;
    redraw();
  }
  cnv.addEventListener('mousedown',ev=>{ if(currentEditMode()!=='roi') return; if(state.cx<0) return; const {sx,sy}=canvasToSrc(ev); let rSize=parseInt(roiSize.value,10)||224; rSize=Math.min(rSize,state.imgW,state.imgH); const half=rSize/2; if(Math.abs(sx-state.cx)<=half && Math.abs(sy-state.cy)<=half){ dragging=true; dragOffset.dx = sx - state.cx; dragOffset.dy = sy - state.cy; ev.preventDefault(); } });
  // Ellipse center drag (only manual_ellipse mode)
  cnv.addEventListener('mousedown',ev=>{ if(currentEditMode()!=='ellipse') return; if(!maskEnabled.checked) return; if(!state.ellipse) return;
    const {sx,sy}=canvasToSrc(ev); const ell=state.ellipse; const [cxE,cyE]=ell.center; const [ax1E,ax2E]=ell.axes; const ang=ell.angle*Math.PI/180;
    // First test rotation handle
    const hx = cxE + ax2E*Math.sin(ang); const hy = cyE - ax2E*Math.cos(ang); // handle world coords
    const distHandle = Math.hypot(sx-hx, sy-hy);
    if(distHandle <= ax2E*0.3 || distHandle <= 25){ // size heuristics
      ellRotating=true; ellAngleStart=ell.angle; // reference vector angle from center to cursor
      const vecAng = Math.atan2(sy - cyE, sx - cxE)*180/Math.PI; // vector angle deg (0 at +x)
      ellRotateRefAngle = vecAng; ev.preventDefault(); redraw(); return; }
    // Otherwise test drag inside ellipse interior
    const dx=sx-cxE, dy=sy-cyE; const cos=Math.cos(-ang), sin=Math.sin(-ang); const lx=dx*cos - dy*sin, ly=dx*sin + dy*cos;
    if((lx*lx)/(ax1E*ax1E)+(ly*ly)/(ax2E*ax2E) <=1){ ellDragging=true; ellDragOffset.dx=sx-cxE; ellDragOffset.dy=sy-cyE; ev.preventDefault(); }
  });
  window.addEventListener('mousemove',ev=>{ if(!dragging) return; const {sx,sy}=canvasToSrc(ev); setCenter(sx-dragOffset.dx, sy-dragOffset.dy,true); });
  window.addEventListener('mousemove',ev=>{ if(!ellDragging) return; const {sx,sy}=canvasToSrc(ev); let nx = sx - ellDragOffset.dx; let ny = sy - ellDragOffset.dy; // clamp ellipse center inside image
    nx=Math.max(0,Math.min(state.imgW-1,nx)); ny=Math.max(0,Math.min(state.imgH-1,ny));
    state.ellipse.center=[Math.round(nx),Math.round(ny)]; redraw(); });
  window.addEventListener('mousemove',ev=>{ if(!ellRotating) return; const {sx,sy}=canvasToSrc(ev); const ell=state.ellipse; if(!ell) return; const [cxE,cyE]=ell.center; const vecAng = Math.atan2(sy - cyE, sx - cxE)*180/Math.PI; // angle of vector center->cursor
    // ellipse angle should align so that handle vector (current) corresponds to top point ( -90 + angle )
    // Thus: vecAng = -90 + angle  => angle = vecAng + 90
    let newAng = (vecAng + 90 + 3600)%3600; newAng = Math.round(newAng)%360; ell.angle=newAng; redraw(); });
  window.addEventListener('mouseup',()=>{ dragging=false; });
  window.addEventListener('mouseup',()=>{ ellDragging=false; });
  window.addEventListener('mouseup',()=>{ if(ellRotating){ ellRotating=false; redraw(); }});
  // (Wheel-basierte ROI Größenänderung entfernt – feste Auswahl 244 oder 112)
  // Ellipse wheel interactions (scale / rotate) when manual ellipse mode active
  cnv.addEventListener('wheel',ev=>{ if(currentEditMode()!=='ellipse') return; if(!maskEnabled.checked) return; if(!state.ellipse) return; ev.preventDefault(); let changed=false; if(ev.shiftKey){ // rotate
      let ang=state.ellipse.angle + (ev.deltaY<0? -5: 5); ang=(ang+360)%360; state.ellipse.angle=ang; changed=true;
    } else if(ev.ctrlKey){ // adjust width, recompute height
      let ax1=state.ellipse.axes[0] + (ev.deltaY<0?5:-5); ax1=Math.max(5,Math.min(2000,ax1)); state.ellipse.axes[0]=ax1; enforceFaceRatio(); changed=true;
    } else if(ev.altKey){ // adjust height, recompute width from ratio
      let ax2=state.ellipse.axes[1] + (ev.deltaY<0?5:-5); ax2=Math.max(5,Math.min(2000,ax2)); // derive width
      let ax1 = ax2 / FACE_RATIO; if(ax1 < 5) ax1 = 5; if(ax1 > 2000) ax1=2000; state.ellipse.axes[0]=Math.round(ax1); state.ellipse.axes[1]=Math.round(ax1*FACE_RATIO); changed=true;
    } else { // uniform scale (keep ratio)
      let ax1=state.ellipse.axes[0] + (ev.deltaY<0?5:-5); ax1=Math.max(5,Math.min(2000,ax1)); state.ellipse.axes[0]=ax1; enforceFaceRatio(); changed=true;
    }
    if(changed) redraw();
  }, {passive:false});

  // Auto Neck
  btnAuto.addEventListener('click',async()=>{ if(!state.path) return; btnAuto.disabled=true; ptLabel.textContent='Auto erkenne…'; const r=await fetch('/preprocess/auto_point',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:state.path,little_endian:little.checked,yolo_weights:yoloWeights.value.trim(),yolo_conf:parseFloat(yoloConf.value)||0.25})}); const j=await r.json(); if(!j.ok){ ptLabel.textContent='Fehler Auto: '+j.error; btnAuto.disabled=false; return;} // Verwende NICHT j.preview, um Originalframe unverändert zu behalten (keine eingebackenen Annotationen)
    if(j.neck){ state.cx=j.neck.x; state.cy=j.neck.y; ptLabel.textContent=`Auto Neck: (${state.cx}, ${state.cy})`; } else { ptLabel.textContent='Kein Neck gefunden'; }
    if(j.ellipse){
      // Übernommene Auto-Ellipse an Gesichts-Ratio anpassen (nur wenn plausibel)
      const ae = {center:j.ellipse.center.slice(), axes:j.ellipse.axes.slice(), angle:j.ellipse.angle};
      // Normalisiere auf gewünschtes Verhältnis anhand aktueller Breite (axes[0])
      ae.axes[1] = Math.round(ae.axes[0] * FACE_RATIO);
      state.autoEllipse=ae;
      if(maskEnabled.checked && !state.ellipse){ state.ellipse={...state.autoEllipse}; }
    }
    redraw(); btnAuto.disabled=false; });

  maskEnabled.addEventListener('change',()=>{
    if(!maskEnabled.checked){ state.ellipse=null; }
    else {
      if(state.autoEllipse) state.ellipse={...state.autoEllipse};
      else if(!state.ellipse && state.img){ state.ellipse={center:[Math.round(state.imgW/2),Math.round(state.imgH/2)],axes:[50, Math.round(50*FACE_RATIO)],angle:0}; }
    }
    enforceFaceRatio();
    redraw();
  });

  if(roiMaskShow){ roiMaskShow.addEventListener('change', redraw); }
  const roiSizeSel = document.getElementById('roiSize');
  if(roiSizeSel){ roiSizeSel.addEventListener('change', ()=>{ redraw(); }); }

  // Export
  btnExport.addEventListener('click',async()=>{ if(!state.path||state.cx<0) return; btnExport.disabled=true; statusEl.textContent='Export läuft…'; links.innerHTML=''; const previewCard=document.getElementById('exportPreview'); const vidFull=document.getElementById('vidFull'); const vidRoi=document.getElementById('vidRoi'); if(previewCard){ previewCard.style.display='none'; }
    let ellipsePayload=null; if(maskEnabled.checked && state.ellipse) ellipsePayload=state.ellipse; const r=await fetch('/preprocess/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:state.path,cx:state.cx,cy:state.cy,roi_size:parseInt(roiSize.value,10)||224,little_endian:little.checked,ellipse:ellipsePayload,roi_mask: !!(roiMaskExport && roiMaskExport.checked)})}); const j=await r.json(); if(!j.ok){ statusEl.textContent='Fehler: '+j.error; btnExport.disabled=false; return;} statusEl.textContent=`OK (${j.frames} Frames, ~${j.duration.toFixed(2)}s, Maske=${j.ellipse_used}, ${(j.ms_per_frame? j.ms_per_frame.toFixed(1)+' ms/F':'')})`; links.innerHTML=`<a class="btn" href="${j.full_url}" target="_blank">Vollbild MP4</a> <a class="btn" href="${j.roi_url}" target="_blank">ROI MP4</a>`; if(previewCard && vidFull && vidRoi){ const ts=Date.now(); vidFull.src = j.full_url + '?t='+ts; vidRoi.src = j.roi_url + '?t='+ts; vidFull.currentTime=0; vidRoi.currentTime=0; previewCard.style.display='block'; }
    btnExport.disabled=false; });

  // Init
  redraw();
})();
