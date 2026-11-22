(function () {
  const form = document.getElementById('uploadForm'), errBox = document.getElementById('errorBox');
  const bar = document.getElementById('progBar'), wrap = document.getElementById('progWrap'), btn = document.getElementById('submitBtn'), st = document.getElementById('statusText');
  const cards = {}; document.querySelectorAll('.up-card').forEach(c => cards[c.dataset.role] = c);
  const fmt = b => b > 1048576 ? (b / 1048576).toFixed(1) + ' MB' : (b / 1024).toFixed(0) + ' KB';
  const okExt = n => window.ALLOWED.some(e => n.toLowerCase().endsWith(e));
  const esc = t => t.replace(/[<>&]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]));
  const showErr = list => { errBox.innerHTML = '<ul class="mb-0">' + list.map(e => '<li>' + esc(e) + '</li>').join('') + '</ul>'; errBox.classList.remove('d-none'); errBox.scrollIntoView({behavior: 'smooth', block: 'center'}); };
  const setState = (card, cls, msg, icon) => { card.classList.remove('valid', 'invalid'); if (cls) card.classList.add(cls); card.querySelector('.state').textContent = msg; card.querySelector('.state').className = 'state mt-1 ' + (cls === 'valid' ? 'text-success' : cls === 'invalid' ? 'text-danger' : 'text-muted'); if (icon) card.querySelector('.ico i').className = 'bi ' + icon; };
  // cheap client-side check of the first bytes (gzip magic or NIfTI sizeof_hdr); the server re-validates fully
  async function sniff(file) {
    const b = new Uint8Array(await file.slice(0, 4).arrayBuffer());
    if (file.name.toLowerCase().endsWith('.gz')) return b[0] === 0x1f && b[1] === 0x8b;
    const dv = new DataView(b.buffer); return dv.getInt32(0, true) === 348 || dv.getInt32(0, false) === 348 || dv.getInt32(0, true) === 540 || dv.getInt32(0, false) === 540;
  }
  async function check(role, file) {
    const c = cards[role]; c.querySelector('.fname').textContent = file ? file.name : 'No file selected'; c.querySelector('.fsize').textContent = file ? fmt(file.size) : '';
    if (!file) return setState(c, '', role === 'ground_truth' ? 'Optional' : 'Waiting for file', 'bi-cloud-arrow-up');
    if (!okExt(file.name)) return setState(c, 'invalid', 'Must be .nii or .nii.gz', 'bi-x-circle');
    if (file.size === 0) return setState(c, 'invalid', 'File is empty', 'bi-x-circle');
    if (!(await sniff(file))) return setState(c, 'invalid', 'Does not look like a NIfTI file', 'bi-x-circle');
    setState(c, 'valid', 'Format OK – full validation on upload', 'bi-check-circle');
  }
  Object.entries(cards).forEach(([role, c]) => {
    const inp = c.querySelector('input'); inp.addEventListener('change', () => check(role, inp.files[0]));
    ['dragenter', 'dragover'].forEach(ev => c.addEventListener(ev, e => { e.preventDefault(); c.classList.add('drag'); }));
    ['dragleave', 'drop'].forEach(ev => c.addEventListener(ev, e => { e.preventDefault(); c.classList.remove('drag'); }));
    c.addEventListener('drop', e => { if (e.dataTransfer.files.length) { inp.files = e.dataTransfer.files; check(role, inp.files[0]); } });
  });
  form.addEventListener('submit', e => {
    e.preventDefault(); errBox.classList.add('d-none');
    const problems = []; let total = 0;
    ['t1', 't1ce', 't2', 'flair'].forEach(k => { const f = form.elements[k].files[0]; if (!f) problems.push('Missing ' + k.toUpperCase() + ' file'); else { total += f.size; if (cards[k].classList.contains('invalid')) problems.push(k.toUpperCase() + ': ' + cards[k].querySelector('.state').textContent); } });
    const gt = form.elements['ground_truth'].files[0]; if (gt) { total += gt.size; if (cards.ground_truth.classList.contains('invalid')) problems.push('Ground truth: ' + cards.ground_truth.querySelector('.state').textContent); }
    if (total > window.MAX_MB * 1048576) problems.push('Total size ' + fmt(total) + ' exceeds ' + window.MAX_MB + ' MB');
    if (problems.length) return showErr(problems);
    const xhr = new XMLHttpRequest(); xhr.open('POST', form.action);
    xhr.setRequestHeader('X-Requested-With', 'fetch'); xhr.setRequestHeader('X-CSRFToken', csrfToken());
    wrap.classList.remove('d-none'); btn.disabled = true; st.textContent = 'Uploading…';
    Object.values(cards).forEach(c => { if (c.querySelector('input').files[0]) c.querySelector('.progress').classList.remove('d-none'); });
    xhr.upload.onprogress = ev => { if (!ev.lengthComputable) return; const p = Math.round(ev.loaded / ev.total * 100); bar.style.width = p + '%'; bar.textContent = p + '%';
      document.querySelectorAll('.up-card .progress-bar').forEach(b => b.style.width = p + '%'); if (p === 100) st.textContent = 'Validating on server (shape, affine, orientation, spacing)…'; };
    xhr.onload = () => { let d = {}; try { d = JSON.parse(xhr.responseText); } catch (x) {}
      if (xhr.status === 200 && d.ok) { Object.entries(cards).forEach(([r, c]) => { if (c.querySelector('input').files[0]) setState(c, 'valid', 'Validated on server', 'bi-check-circle-fill'); }); st.textContent = 'Validated – opening viewer…'; window.location = d.redirect; }
      else { btn.disabled = false; st.textContent = ''; wrap.classList.add('d-none'); document.querySelectorAll('.up-card .progress').forEach(p => p.classList.add('d-none')); showErr(d.errors || ['Upload failed (HTTP ' + xhr.status + ')']); } };
    xhr.onerror = () => { btn.disabled = false; st.textContent = ''; showErr(['Network error during upload']); };
    xhr.send(new FormData(form));
  });
})();
