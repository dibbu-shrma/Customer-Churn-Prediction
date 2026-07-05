// -------- Page navigation --------
  const navItems = document.querySelectorAll(".nav-item");
  const pages = document.querySelectorAll(".page");

  navItems.forEach(function (item) {
    item.addEventListener("click", function () {
      navItems.forEach(function (n) { n.classList.remove("active"); });
      pages.forEach(function (p) { p.classList.remove("active"); });

      item.classList.add("active");
      document.getElementById("page-" + item.dataset.page).classList.add("active");
    });
  });

  // -------- Model performance chart (placeholder data) --------
  // Replace these numbers with your real GridSearchCV / comparison results.
  const modelResults = [
    { name: "Logistic Regression", score: 0.80, isWinner: true },
    { name: "Random Forest",       score: 0.79, isWinner: false },
    { name: "XGBoost",             score: 0.78, isWinner: false },
    { name: "SVM",                 score: 0.77, isWinner: false },
    { name: "KNN",                 score: 0.75, isWinner: false }
  ];

  const chartEl = document.getElementById("perf-chart");
  modelResults.forEach(function (m) {
    const row = document.createElement("div");
    row.className = "bar-row";

    const label = document.createElement("div");
    label.className = "bar-row-label";
    label.innerHTML = m.name + (m.isWinner ? '<span class="winner-tag">Winner</span>' : "");

    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = (m.score * 100) + "%";
    fill.style.background = m.isWinner
      ? "linear-gradient(90deg, var(--accent-pink), var(--accent-purple))"
      : "var(--accent-purple)";
    fill.style.opacity = m.isWinner ? "1" : "0.55";
    track.appendChild(fill);

    const value = document.createElement("div");
    value.className = "bar-value";
    value.textContent = m.score.toFixed(2);

    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(value);
    chartEl.appendChild(row);
  });

  // -------- Risk prediction form logic --------
  // Relative path works because Flask now serves both the frontend and the API
  // from the same origin -- no CORS needed, no hardcoded domain.
  const API_URL = "/predict";
  const form = document.getElementById("predict-form");
  const runBtn = document.getElementById("run-btn");
  const resultBox = document.getElementById("result");
  const errorBox = document.getElementById("error-box");

  form.addEventListener("submit", async function (e) {
    e.preventDefault();
    errorBox.style.display = "none";
    resultBox.style.display = "none";
    runBtn.disabled = true;
    runBtn.textContent = "Analyzing...";

    const formData = new FormData(form);
    const payload = {};
    formData.forEach(function (value, key) { payload[key] = value; });

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error("Server responded with status " + response.status);
      }

      const data = await response.json();
      renderResult(data);

    } catch (err) {
      errorBox.textContent = "Couldn't reach the prediction server: " + err.message +
        ". Make sure your Flask backend is running at " + API_URL + ".";
      errorBox.style.display = "block";
    } finally {
      runBtn.disabled = false;
      runBtn.textContent = "Check churn risk";
    }
  });

  function renderResult(data) {
    const probability = data.churn_probability;
    const percent = Math.round(probability * 100);
    const willChurn = data.will_churn;

    const verdictEl = document.getElementById("verdict-text");
    verdictEl.textContent = willChurn ? "Likely to churn" : "Likely to stay";
    verdictEl.className = "verdict " + (willChurn ? "risk-high" : "risk-low");

    document.getElementById("prob-number").innerHTML = percent + "<span>%</span>";

    const reasons = data.reasons.slice();

    function impactSize(item) {
      if (item.impact < 0) return -item.impact;
      return item.impact;
    }

    reasons.sort(function (a, b) { return impactSize(b) - impactSize(a); });
    const topReasons = reasons.slice(0, 6);

    let maxImpact = 0;
    topReasons.forEach(function (r) {
      const size = impactSize(r);
      if (size > maxImpact) maxImpact = size;
    });
    if (maxImpact === 0) maxImpact = 1;

    const listEl = document.getElementById("reasons-list");
    listEl.innerHTML = "";

    topReasons.forEach(function (r) {
      const pushesChurn = r.impact > 0;
      const widthPercent = (impactSize(r) / maxImpact) * 50;

      const row = document.createElement("div");
      row.className = "reason-row";

      const nameEl = document.createElement("div");
      nameEl.textContent = formatFeatureName(r.feature);

      const trackEl = document.createElement("div");
      trackEl.className = "reason-bar-track";
      const fillEl = document.createElement("div");
      fillEl.className = "reason-bar-fill " + (pushesChurn ? "push" : "pull");
      fillEl.style.width = widthPercent + "%";
      trackEl.appendChild(fillEl);

      const impactEl = document.createElement("div");
      impactEl.className = "reason-impact " + (pushesChurn ? "up" : "down");
      impactEl.textContent = (pushesChurn ? "↑ " : "↓ ") + Math.abs(r.impact).toFixed(3);

      row.appendChild(nameEl);
      row.appendChild(trackEl);
      row.appendChild(impactEl);
      listEl.appendChild(row);
    });

    resultBox.style.display = "block";
    resultBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function formatFeatureName(rawName) {
    let name = rawName.replace("_", ": ");
    name = name.replace(/([a-z])([A-Z])/g, "$1 $2");
    return name;
  }