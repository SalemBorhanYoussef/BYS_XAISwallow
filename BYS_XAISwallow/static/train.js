(() => {
  const qs = (s) => document.querySelector(s);
  const logsEl = qs('#logs');
  const statusEl = qs('#status');
  const startBtn = qs('#btnStart');
  const stopBtn = qs('#btnStop');
  let logCursor = 0;
  let pollTimer = null;

  function readCfg() {
    return {
      python: qs('#pyExe').value.trim() || 'python',
      script: qs('#script').value.trim() || 'train_multi_new.py',
      workdir: qs('#workdir').value.trim(),
      config: {
        dataset_dir: qs('#dsTrain').value.trim(),
        dataset_test_dir: qs('#dsTest').value.trim(),
        out_dir: qs('#outDir').value.trim(),
        batch_size: parseInt(qs('#batchSize').value, 10) || 8,
        epochs: parseInt(qs('#epochs').value, 10) || 30,
        lr: parseFloat(qs('#lr').value) || 1e-4,
        scheduler: qs('#scheduler').value,
        use_amp: qs('#useAmp').value === 'true',
        model_type: qs('#modelType').value,
      },
    };
  }

  async function start() {
    startBtn.disabled = true;
    try {
      const r = await fetch('/train/api/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(readCfg()),
      });
      const j = await r.json();
      if (!j.ok) {
        statusEl.textContent = 'Fehler: ' + (j.error || 'start');
        startBtn.disabled = false;
        return;
      }
      statusEl.textContent = `Running (pid=${j.pid})`;
      logsEl.textContent = '';
      logCursor = 0;
      startPolling();
    } catch (e) {
      statusEl.textContent = 'Netzfehler beim Start';
      startBtn.disabled = false;
    }
  }

  async function stop() {
    try {
      const r = await fetch('/train/api/stop', { method: 'POST' });
      const j = await r.json();
      statusEl.textContent = 'Gestoppt';
      stopPolling();
      startBtn.disabled = false;
    } catch (e) {
      statusEl.textContent = 'Stop fehlgeschlagen';
    }
  }

  async function pollOnce() {
    try {
      const [stR, lgR] = await Promise.all([
        fetch('/train/api/status'),
        fetch('/train/api/logs?since=' + logCursor + '&max=500'),
      ]);
      const st = await stR.json();
      if (st.ok) {
        statusEl.textContent = st.running ? `Running (pid=${st.pid}, t=${st.uptime_sec.toFixed(1)}s)` : 'Idle';
        if (!st.running) startBtn.disabled = false;
      }
      const lg = await lgR.json();
      if (lg.ok) {
        if (lg.lines && lg.lines.length) {
          logsEl.textContent += lg.lines.join('\n') + '\n';
          logsEl.scrollTop = logsEl.scrollHeight;
        }
        logCursor = lg.to || logCursor;
      }
    } catch (e) {
      // ignore
    }
  }

  function startPolling() {
    stopPolling();
    pollTimer = setInterval(pollOnce, 1000);
  }
  function stopPolling() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = null;
  }

  startBtn.addEventListener('click', start);
  stopBtn.addEventListener('click', stop);
  // initial status
  pollOnce();
})();
