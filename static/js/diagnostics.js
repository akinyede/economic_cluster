(function(){
  function fmt(val, suffix){
    if (val === null || val === undefined) return 'N/A';
    var num = Number(val);
    if (!isFinite(num)) return String(val);
    if (suffix === '%') return num.toFixed(2) + '%';
    if (suffix === '') return num.toFixed(0);
    return num.toFixed(2);
  }

  async function loadDiagnostics(){
    const card = document.getElementById('diagnosticsCard');
    if(!card) return;
    const spinner = document.getElementById('diagnosticsSpinner');
    const err = document.getElementById('diagnosticsError');
    if (spinner) spinner.style.display = 'inline-block';
    if (err) err.style.display = 'none';
    try{
      const resp = await fetch('/api/diagnostics');
      const json = await resp.json();
      if (!json || json.status !== 'ok') throw new Error(json && json.message || 'Request failed');
      const data = json.data || {};
      const back = data.backtest || {};
      const abl = data.ablation || {};
      // Safe text injections
      if (window.SecurityHelpers){
        const SH = window.SecurityHelpers;
        SH.safeText('#diagGdpMapePre', fmt(back.gdp_mape_pre, '%'));
        SH.safeText('#diagGdpMapePost', fmt(back.gdp_mape_post, '%'));
        SH.safeText('#diagJobsMapePre', fmt(back.jobs_mape_pre, '%'));
        SH.safeText('#diagJobsMapePost', fmt(back.jobs_mape_post, '%'));
        SH.safeText('#diagAvgRateGDP', fmt(back.avg_rate_gdp, ''));
        SH.safeText('#diagAvgRateJobs', fmt(back.avg_rate_jobs, ''));
        SH.safeText('#diagDeltaGDP', fmt(abl.delta_gdp_pct, '%'));
        SH.safeText('#diagDeltaJobs', fmt(abl.delta_jobs_pct, '%'));
        const ts = data.generated_at ? new Date(data.generated_at*1000).toLocaleString() : '—';
        SH.safeText('#diagnosticsUpdatedAt', ts);
      }
    }catch(e){
      if (err){
        err.style.display = 'block';
        err.textContent = 'Failed to load diagnostics: ' + (e && e.message ? e.message : e);
      }
    }finally{
      if (spinner) spinner.style.display = 'none';
    }
  }

  document.addEventListener('DOMContentLoaded', function(){
    loadDiagnostics();
    var btn = document.getElementById('refreshDiagnosticsBtn');
    if (btn){ btn.addEventListener('click', loadDiagnostics); }
  });
})();
