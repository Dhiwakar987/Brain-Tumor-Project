(function () {
  const form = document.getElementById('radForm');
  if (form) form.addEventListener('submit', async e => {
    e.preventDefault(); const btn = document.getElementById('radBtn'), msg = document.getElementById('radMsg'); btn.disabled = true;
    const fd = new FormData(form);
    if (!fd.getAll('regions').length || !fd.getAll('modalities').length) { msg.textContent = 'Select at least one region and one modality.'; btn.disabled = false; return; }
    const r = await postJSON(form.dataset.url, fd);
    if (!r.data.ok) { msg.textContent = r.data.error || 'Could not start'; btn.disabled = false; return; }
    msg.textContent = 'Running…';
    const f = await pollStatus(form.dataset.status, s => { msg.textContent = 'Running: ' + ((s.job && s.job.stage) || '…'); });
    if (f.job && f.job.status === 'completed') location.href = location.pathname + '?source=' + fd.get('source'); else { msg.textContent = 'Failed: ' + ((f.job && f.job.error) || 'unknown error'); btn.disabled = false; }
  });
  const table = document.getElementById('featTable'); if (!table) return;
  const sel = id => document.getElementById(id);
  function apply() {
    const reg = sel('fRegion').value, mod = sel('fModality').value, fam = sel('fFamily').value, txt = sel('fText').value.toLowerCase(); let n = 0; const vals = [];
    table.querySelectorAll('tbody tr').forEach(tr => {
      const show = (!reg || tr.dataset.region === reg) && (!mod || tr.dataset.modality === mod) && (!fam || tr.dataset.family === fam) && (!txt || tr.cells[0].textContent.toLowerCase().includes(txt));
      tr.style.display = show ? '' : 'none'; if (show) { n++; const v = parseFloat(tr.cells[1].textContent); if (isFinite(v)) vals.push(v); }
    });
    sel('visibleCount').textContent = n + ' features shown';
    Plotly.react('distPlot', [{ x: vals, type: 'histogram' }], { margin: { t: 10 }, xaxis: { title: 'feature value (mixed units)' }, yaxis: { title: 'count' } }, { responsive: true, displaylogo: false });
  }
  ['fRegion', 'fModality', 'fFamily'].forEach(i => sel(i).addEventListener('change', apply)); sel('fText').addEventListener('input', apply); apply();
})();

// shape chart + category cards (appended)
(function () {
  const t = document.getElementById('shapeTable'); if (!t || !window.Plotly) return;
  const rows = [...t.querySelectorAll('tbody tr')].map(r => ({ region: r.dataset.region, name: r.dataset.name, v: parseFloat(r.dataset.value) }));
  const names = ['VoxelVolume', 'MeshVolume', 'SurfaceArea']; const regs = ['WT', 'TC', 'ET'];
  const cols = { WT: '#12305f', TC: '#14b8a6', ET: '#d97706' };
  Plotly.newPlot('shapeChart', regs.map(rg => ({ type: 'bar', name: rg, marker: { color: cols[rg] }, x: names.filter(n => rows.some(r => r.name === n)), y: names.filter(n => rows.some(r => r.name === n)).map(n => { const r = rows.find(x => x.region === rg && x.name === n); return r ? r.v : null; }) })),
    { barmode: 'group', margin: { t: 10, l: 60, r: 10, b: 40 }, yaxis: { title: 'mm³ (volumes) / mm² (surface area)', type: 'log' }, legend: { orientation: 'h' } }, { responsive: true, displaylogo: false });
  document.querySelectorAll('.cat-card').forEach(c => c.addEventListener('click', () => { const s = document.getElementById('fFamily'); if (s) { s.value = c.dataset.fam; s.dispatchEvent(new Event('change')); document.getElementById('featTable').scrollIntoView({ behavior: 'smooth' }); } }));
})();
