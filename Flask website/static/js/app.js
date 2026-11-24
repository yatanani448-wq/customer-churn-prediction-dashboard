function renderPlot(id, figure) {
  const target = document.getElementById(id);
  if (!target) return;
  Plotly.newPlot(target, figure.data, figure.layout, {responsive: true, displaylogo: false});
}

async function loadAnalytics() {
  const selector = document.getElementById('dimension');
  const target = document.getElementById('analytics-chart');
  if (!selector || !target) return;
  target.innerHTML = '<p class="loading">Loading real customer data…</p>';
  const response = await fetch(`/api/analytics?dimension=${encodeURIComponent(selector.value)}`);
  if (!response.ok) { target.innerHTML = '<p>Unable to load this chart.</p>'; return; }
  const figure = await response.json();
  renderPlot('analytics-chart', figure);
}
