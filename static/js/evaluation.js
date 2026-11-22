(function () {
  const form = document.getElementById('evalForm'); if (!form) return;
  form.addEventListener('submit', async e => {
    e.preventDefault(); const btn = document.getElementById('evalBtn'), msg = document.getElementById('evalMsg'); btn.disabled = true; msg.textContent = 'Running…';
    const r = await postJSON(form.dataset.url);
    if (!r.data.ok) { msg.textContent = r.data.error; btn.disabled = false; return; }
    const f = await pollStatus(form.dataset.status);
    if (f.job && f.job.status === 'completed') location.reload(); else { msg.textContent = 'Failed: ' + ((f.job && f.job.error) || 'unknown'); btn.disabled = false; }
  });
})();
