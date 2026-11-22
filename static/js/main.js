// Shared helpers
window.csrfToken = () => document.querySelector('meta[name="csrf-token"]').content;
window.postJSON = async (url, bodyObj) => {
  const opts = { method: 'POST', headers: { 'X-CSRFToken': csrfToken(), 'X-Requested-With': 'fetch' } };
  if (bodyObj instanceof FormData) opts.body = bodyObj;
  const r = await fetch(url, opts);
  let data = {}; try { data = await r.json(); } catch (e) {}
  return { status: r.status, data };
};
// Poll /status until the latest job finishes. cb(statusObj) on each tick; resolves with final status.
window.pollStatus = (url, cb, interval = 1500) => new Promise(resolve => {
  const tick = async () => {
    try {
      const r = await fetch(url); const s = await r.json();
      if (cb) cb(s);
      if (s.job && (s.job.status === 'queued' || s.job.status === 'running')) return setTimeout(tick, interval);
      resolve(s);
    } catch (e) { setTimeout(tick, 3000); }
  };
  tick();
});
// Confirm before deleting an examination
document.addEventListener('submit', e => {
  if (e.target.classList.contains('delete-form') && !confirm('Delete this examination and all its files? This cannot be undone.')) e.preventDefault();
});
// Mobile sidebar toggle
(function () {
  const sb = document.getElementById('sidebar'), sc = document.getElementById('scrim'), b = document.getElementById('menuBtn');
  if (!b) return; const t = o => { sb.classList.toggle('open', o); sc.classList.toggle('show', o); };
  b.addEventListener('click', () => t(!sb.classList.contains('open'))); sc.addEventListener('click', () => t(false));
})();
