import "./style.css";

type DamageIssue = {
  issue: string;
  category: string;
  severity: string;
  penalty_pct: number;
  estimated_impact: number;
};

type AnalysisResponse = {
  input_summary: Record<string, string | number | null>;
  base_prediction: number;
  adjusted_prediction: number;
  price_range: { low: number; high: number };
  confidence: string;
  confidence_score: number;
  asking_price_comparison: string | null;
  extracted_damage: {
    cleaned_text: string;
    severity_score: number;
    total_penalty_pct: number;
    confidence_hint: string;
    issues: DamageIssue[];
    category_counts: Record<string, number>;
  };
  explanation_points: string[];
  negotiation_tip: string;
  model_metrics: Record<string, string | number>;
};

type ChatResponse = {
  answer: string;
  bullets: string[];
};

type ChatEntry = {
  role: "user" | "assistant";
  text: string;
  bullets?: string[];
};

const app = document.querySelector<HTMLDivElement>("#app");
const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, "") || "http://127.0.0.1:8000";

if (!app) {
  throw new Error("App root not found.");
}

app.innerHTML = `
  <div class="dashboard-shell">
    <header class="dashboard-header">
      <div class="header-copy">
        <p class="eyebrow">CarSight AI</p>
        <h1>Damage-aware used car pricing for clearer buying decisions.</h1>
        <p class="header-note">
          CarSight helps turn vague listing language into a price range you can actually work with.
        </p>
        <div class="header-actions">
          <p id="api-status" class="status-text">Checking backend...</p>
          <button id="open-report-btn" class="ghost-btn" type="button" disabled>View report</button>
        </div>
      </div>

      <div class="story-panel">
        <div class="story-grid">
          <figure class="story-photo">
            <img src="/slide-damaged-car.jpg" alt="Damaged vehicle with heavy front-end impact">
            <figcaption>What the listing may be hiding</figcaption>
          </figure>
          <div class="story-seller">
            <img src="/slide-seller.jpg" alt="Smiling seller in a suit">
            <div class="story-bubble">
              <p>"Minor scratches. Runs fine. Easy fix."</p>
            </div>
          </div>
        </div>
        <p class="story-instruction">Enter the vehicle details, paste the seller's wording, and check the range before you trust the price.</p>
      </div>
    </header>

    <main class="dashboard-grid">
      <aside class="control-panel">
        <div class="panel-heading">
          <h2>Vehicle input</h2>
          <p>Fill in the car details and paste the seller's wording exactly as it appears.</p>
        </div>

        <form id="analysis-form" class="analysis-form">
          <div class="field-grid">
            <label>
              <span>Make</span>
              <input name="make" value="Honda" required />
            </label>
            <label>
              <span>Model</span>
              <input name="model" value="Accord" required />
            </label>
            <label>
              <span>Year</span>
              <input name="year" type="number" value="2018" min="1980" max="2030" required />
            </label>
            <label>
              <span>Mileage</span>
              <input name="mileage" type="number" value="82000" min="0" required />
            </label>
            <label>
              <span>Engine</span>
              <input name="engine" value="2.0L I-4 252HP" required />
            </label>
            <label>
              <span>Color</span>
              <input name="color" value="Gray" />
            </label>
            <label>
              <span>Fuel type</span>
              <select name="fuel_type">
                <option selected>Gasoline</option>
                <option>Hybrid</option>
                <option>Electric</option>
                <option>Diesel</option>
              </select>
            </label>
            <label>
              <span>Transmission</span>
              <select name="transmission">
                <option selected>Automatic</option>
                <option>Manual</option>
                <option>CVT</option>
              </select>
            </label>
            <label>
              <span>Drivetrain</span>
              <select name="drivetrain">
                <option selected>Front-wheel Drive</option>
                <option>Rear-wheel Drive</option>
                <option>All-wheel Drive</option>
                <option>Four-wheel Drive</option>
              </select>
            </label>
            <label>
              <span>Asking price</span>
              <input name="asking_price" type="number" value="14900" min="0" />
            </label>
          </div>

          <label class="textarea-field">
            <span>Damage description</span>
            <textarea name="damage_description" rows="6" required>Rear bumper cracked and engine knocks when cold. Minor scratches on driver-side door.</textarea>
          </label>

          <div class="button-row">
            <button id="analyze-btn" type="submit">Run estimate</button>
            <button id="sample-btn" class="ghost-btn" type="button">Reset sample</button>
          </div>
        </form>

        <div class="panel-footer">
          <p class="footnote-label">Method in this build</p>
          <p class="footnote-copy">Best result came from a 60/40 blend of TF-IDF Ridge NLP and HistGradientBoosting, then the app narrows the report range using held-out residual quantiles.</p>
        </div>
      </aside>

      <section class="insight-panel">
        <div id="result-empty" class="empty-state">
          <h2>Run the estimate to open the dashboard</h2>
          <p>You will see a tighter fair-price band, a simple range visual, damage cost findings, and a printable estimate report.</p>
        </div>
        <div id="result-content" class="result-stack hidden"></div>
      </section>
    </main>
  </div>

  <button id="chat-launch" class="chat-launch" type="button">Ask CarSight</button>

  <div id="chat-overlay" class="chat-overlay hidden"></div>
  <aside id="chat-sidebar" class="chat-sidebar">
    <div class="chat-sidebar-inner">
      <div class="chat-header">
        <div>
          <p class="eyebrow">Chat assistant</p>
          <h2>Ask CarSight</h2>
        </div>
        <button id="chat-close" class="ghost-btn" type="button">Close</button>
      </div>
      <p class="chat-subtitle">Follow-up questions are based on the latest vehicle estimate.</p>
      <div id="chat-history" class="chat-history">
        <div class="chat-placeholder">Run an estimate first, then ask whether the car looks fair, risky, or worth negotiating on.</div>
      </div>
      <div class="quick-prompts">
        <button type="button" class="ghost-btn quick-prompt" data-question="Is this overpriced for the condition?">Overpriced?</button>
        <button type="button" class="ghost-btn quick-prompt" data-question="Should I negotiate on this car?">Negotiate?</button>
        <button type="button" class="ghost-btn quick-prompt" data-question="What is the main risk here?">Main risk?</button>
      </div>
      <form id="sidebar-chat-form" class="sidebar-chat-form">
        <textarea id="sidebar-chat-input" rows="3" placeholder="Ask about price, damage, or what to check before buying."></textarea>
        <button id="sidebar-chat-submit" type="submit">Send question</button>
      </form>
    </div>
  </aside>

  <div id="report-modal" class="report-modal hidden">
    <div class="report-dialog">
      <div class="report-toolbar">
        <div>
          <p class="eyebrow">Estimate report</p>
          <h2>Professional valuation summary</h2>
        </div>
        <div class="report-toolbar-actions">
          <button id="print-report-btn" class="ghost-btn" type="button">Print report</button>
          <button id="close-report-btn" class="ghost-btn" type="button">Close</button>
        </div>
      </div>
      <div id="report-body" class="report-body"></div>
    </div>
  </div>
`;

const form = document.querySelector<HTMLFormElement>("#analysis-form");
const analyzeBtn = document.querySelector<HTMLButtonElement>("#analyze-btn");
const sampleBtn = document.querySelector<HTMLButtonElement>("#sample-btn");
const statusText = document.querySelector<HTMLParagraphElement>("#api-status");
const resultEmpty = document.querySelector<HTMLDivElement>("#result-empty");
const resultContent = document.querySelector<HTMLDivElement>("#result-content");
const openReportBtn = document.querySelector<HTMLButtonElement>("#open-report-btn");
const reportModal = document.querySelector<HTMLDivElement>("#report-modal");
const reportBody = document.querySelector<HTMLDivElement>("#report-body");
const closeReportBtn = document.querySelector<HTMLButtonElement>("#close-report-btn");
const printReportBtn = document.querySelector<HTMLButtonElement>("#print-report-btn");
const chatLaunch = document.querySelector<HTMLButtonElement>("#chat-launch");
const chatClose = document.querySelector<HTMLButtonElement>("#chat-close");
const chatOverlay = document.querySelector<HTMLDivElement>("#chat-overlay");
const chatSidebar = document.querySelector<HTMLDivElement>("#chat-sidebar");
const chatHistoryNode = document.querySelector<HTMLDivElement>("#chat-history");
const chatForm = document.querySelector<HTMLFormElement>("#sidebar-chat-form");
const chatInput = document.querySelector<HTMLTextAreaElement>("#sidebar-chat-input");
const quickPrompts = document.querySelectorAll<HTMLButtonElement>(".quick-prompt");

let latestAnalysis: AnalysisResponse | null = null;
let chatHistory: ChatEntry[] = [];

function escapeHtml(value: string) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function money(value: number) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value);
}

function formatDate() {
  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  }).format(new Date());
}

function clamp(value: number, min = 0, max = 100) {
  return Math.min(Math.max(value, min), max);
}

function getAskingPrice(result: AnalysisResponse) {
  return typeof result.input_summary.asking_price === "number" ? result.input_summary.asking_price : null;
}

function severityClass(severity: string) {
  return `severity severity-${severity.toLowerCase()}`;
}

function rangeSpan(result: AnalysisResponse) {
  return result.price_range.high - result.price_range.low;
}

function recommendationLabel(result: AnalysisResponse) {
  const askingPrice = getAskingPrice(result);
  if (askingPrice === null) return "Use this range as the negotiation anchor.";
  if (askingPrice > result.price_range.high) return "Listing looks above fair market value.";
  if (askingPrice < result.price_range.low) return "Listing looks attractive, but verify why it is priced low.";
  return "Listing sits inside the estimated fair market zone.";
}

function dealStatus(result: AnalysisResponse) {
  const askingPrice = getAskingPrice(result);

  if (askingPrice === null) {
    return {
      label: "Fair range",
      tone: "fair",
      detail: "No asking price was entered, so the meter is centered on the estimate.",
    };
  }

  if (askingPrice < result.price_range.low) {
    return {
      label: "Good deal",
      tone: "good",
      detail: "The asking price is sitting below the estimated fair band.",
    };
  }

  if (askingPrice > result.price_range.high) {
    return {
      label: "Bad deal",
      tone: "bad",
      detail: "The asking price is sitting above the estimated fair band.",
    };
  }

  return {
    label: "Fair deal",
    tone: "fair",
    detail: "The asking price sits inside the estimated fair band.",
  };
}

function buildDamageHeadline(result: AnalysisResponse) {
  if (!result.extracted_damage.issues.length) {
    return "The text does not show a strong damage signal, so the estimate leans more on market data.";
  }

  const topIssue = [...result.extracted_damage.issues].sort((left, right) => right.estimated_impact - left.estimated_impact)[0];
  return `${topIssue.category} looks like the biggest hit in the description, mainly from "${topIssue.issue}".`;
}

function buildDealMeter(result: AnalysisResponse) {
  const askingPrice = getAskingPrice(result);
  const displayValue = askingPrice ?? result.adjusted_prediction;
  const status = dealStatus(result);
  const span = Math.max(rangeSpan(result), 1);
  const paddedLow = Math.max(result.price_range.low - span * 0.4, 0);
  const paddedHigh = result.price_range.high + span * 0.4;
  const ratio = clamp(((displayValue - paddedLow) / Math.max(paddedHigh - paddedLow, 1)) * 100);
  const needleAngle = -82 + (ratio / 100) * 164;

  return `
    <div class="deal-meter-card">
      <div class="deal-meter-shell">
        <div class="deal-meter-arc" style="--needle-angle:${needleAngle}deg;">
          <div class="deal-meter-ring"></div>
          <div class="deal-meter-hole"></div>
          <div class="deal-meter-needle"></div>
          <div class="deal-meter-cap"></div>
          <div class="deal-meter-center">
            <p class="deal-meter-kicker">${askingPrice === null ? "Adjusted estimate" : "Listing price"}</p>
            <h3>${money(displayValue)}</h3>
            <span class="deal-badge deal-badge-${status.tone}">${status.label}</span>
          </div>
        </div>
      </div>

      <div class="deal-meter-scale">
        <span>${money(result.price_range.low)}</span>
        <span>${money(result.price_range.high)}</span>
      </div>

      <div class="deal-meter-meta">
        <div>
          <strong>Fair band</strong>
          <span>${money(result.price_range.low)} - ${money(result.price_range.high)}</span>
        </div>
        <div>
          <strong>Adjusted midpoint</strong>
          <span>${money(result.adjusted_prediction)}</span>
        </div>
      </div>

      <p class="deal-meter-note">${status.detail}</p>
    </div>
  `;
}

function buildRangeVisual(result: AnalysisResponse, mode: "dashboard" | "report" = "dashboard") {
  const askingPrice = getAskingPrice(result);
  const domainLow = Math.min(result.price_range.low, askingPrice ?? result.price_range.low);
  const domainHigh = Math.max(result.price_range.high, askingPrice ?? result.price_range.high);
  const span = Math.max(domainHigh - domainLow, 1);
  const bandStart = ((result.price_range.low - domainLow) / span) * 100;
  const bandWidth = ((result.price_range.high - result.price_range.low) / span) * 100;
  const midpointPosition = ((result.adjusted_prediction - domainLow) / span) * 100;
  const askingPosition = askingPrice === null ? null : ((askingPrice - domainLow) / span) * 100;
  const compactClass = mode === "report" ? "range-visual report-mode" : "range-visual";

  return `
    <div class="${compactClass}">
      <div class="range-axis">
        <span>${money(domainLow)}</span>
        <span>${money(domainHigh)}</span>
      </div>
      <div class="range-track">
        <div class="range-band" style="left:${clamp(bandStart)}%; width:${clamp(bandWidth, 2, 100)}%;"></div>
        <div class="range-marker midpoint" style="left:${clamp(midpointPosition)}%;">
          <span>Adjusted ${money(result.adjusted_prediction)}</span>
        </div>
        ${
          askingPosition === null
            ? ""
            : `
          <div class="range-marker asking" style="left:${clamp(askingPosition)}%;">
            <span>Ask ${money(askingPrice)}</span>
          </div>
        `
        }
      </div>
      <div class="range-foot">
        <div><strong>Low</strong><span>${money(result.price_range.low)}</span></div>
        <div><strong>High</strong><span>${money(result.price_range.high)}</span></div>
      </div>
    </div>
  `;
}

function buildIssueRows(result: AnalysisResponse) {
  if (!result.extracted_damage.issues.length) {
    return `<p class="muted-copy">No specific damage issue was detected from the text, so the estimate leans mostly on vehicle market data.</p>`;
  }

  return result.extracted_damage.issues
    .map(
      (issue) => `
        <article class="issue-row">
          <div class="issue-copy">
            <p class="issue-title">${escapeHtml(issue.category)}</p>
            <p>${escapeHtml(issue.issue)}</p>
          </div>
          <div class="issue-cost">
            <span class="${severityClass(issue.severity)}">${escapeHtml(issue.severity)}</span>
            <strong>${money(issue.estimated_impact)}</strong>
          </div>
        </article>
      `,
    )
    .join("");
}

function buildModelSummary(result: AnalysisResponse) {
  const modelName = escapeHtml(String(result.model_metrics.model_name ?? "Hybrid model"));
  const methodology = escapeHtml(String(result.model_metrics.methodology ?? ""));
  const selectionNote = escapeHtml(String(result.model_metrics.selection_note ?? ""));
  const blendWeight = Number(result.model_metrics.blend_weight ?? 0.6);

  return `
    <div class="method-block">
      <p class="section-label">Chosen method</p>
      <h3>${modelName}</h3>
      <p class="muted-copy">${methodology}</p>
      <dl class="mini-metrics">
        <div><dt>Ridge NLP weight</dt><dd>${Math.round(blendWeight * 100)}%</dd></div>
        <div><dt>Test MAE</dt><dd>${money(Number(result.model_metrics.mae ?? 0))}</dd></div>
        <div><dt>Test RMSE</dt><dd>${money(Number(result.model_metrics.rmse ?? 0))}</dd></div>
      </dl>
      <p class="method-note">${selectionNote}</p>
    </div>
  `;
}

function executiveSummary(result: AnalysisResponse) {
  const make = escapeHtml(String(result.input_summary.make ?? ""));
  const model = escapeHtml(String(result.input_summary.model ?? ""));
  const year = escapeHtml(String(result.input_summary.year ?? ""));
  const recommendation = escapeHtml(recommendationLabel(result));

  return `${year} ${make} ${model} is currently estimated at ${money(result.price_range.low)} to ${money(result.price_range.high)}. ${recommendation}`;
}

function buildReportMarkup(result: AnalysisResponse) {
  const askingPrice = getAskingPrice(result);
  const summaryLines = [
    executiveSummary(result),
    result.asking_price_comparison ?? "No asking price was provided for a direct listing comparison.",
    result.negotiation_tip,
  ];

  return `
    <article class="report-page">
      <div class="report-cover">
        <div>
          <p class="section-label">CarSight AI estimate report</p>
          <h3>${escapeHtml(String(result.input_summary.year ?? ""))} ${escapeHtml(String(result.input_summary.make ?? ""))} ${escapeHtml(String(result.input_summary.model ?? ""))}</h3>
          <p class="muted-copy">Prepared on ${formatDate()}</p>
        </div>
        <div class="report-score">
          <span>${result.confidence}</span>
          <small>${Math.round(result.confidence_score * 100)}% confidence</small>
        </div>
      </div>

      <section class="report-section">
        <div>
          <p class="section-label">Executive summary</p>
          <p class="report-summary">${escapeHtml(summaryLines[0])}</p>
        </div>
        ${buildRangeVisual(result, "report")}
      </section>

      <section class="report-grid">
        <div class="report-section">
          <p class="section-label">Vehicle profile</p>
          <dl class="report-list">
            <div><dt>Year</dt><dd>${escapeHtml(String(result.input_summary.year ?? ""))}</dd></div>
            <div><dt>Mileage</dt><dd>${Number(result.input_summary.mileage ?? 0).toLocaleString()} miles</dd></div>
            <div><dt>Engine</dt><dd>${escapeHtml(String(result.input_summary.engine ?? ""))}</dd></div>
            <div><dt>Fuel</dt><dd>${escapeHtml(String(result.input_summary.fuel_type ?? ""))}</dd></div>
            <div><dt>Transmission</dt><dd>${escapeHtml(String(result.input_summary.transmission ?? ""))}</dd></div>
            <div><dt>Asking price</dt><dd>${askingPrice === null ? "Not provided" : money(askingPrice)}</dd></div>
          </dl>
        </div>

        <div class="report-section">
          <p class="section-label">Estimate findings</p>
          <dl class="report-list">
            <div><dt>Base market value</dt><dd>${money(result.base_prediction)}</dd></div>
            <div><dt>Adjusted midpoint</dt><dd>${money(result.adjusted_prediction)}</dd></div>
            <div><dt>Range width</dt><dd>${money(rangeSpan(result))}</dd></div>
            <div><dt>Damage adjustment</dt><dd>${(result.extracted_damage.total_penalty_pct * 100).toFixed(1)}%</dd></div>
          </dl>
        </div>
      </section>

      <section class="report-grid">
        <div class="report-section">
          <p class="section-label">Damage findings</p>
          <div class="report-issues">${buildIssueRows(result)}</div>
        </div>

        <div class="report-section">
          <p class="section-label">Method and model</p>
          <p class="muted-copy">${escapeHtml(String(result.model_metrics.methodology ?? ""))}</p>
          <ul class="report-bullets">
            <li>${escapeHtml(String(summaryLines[1]))}</li>
            <li>${escapeHtml(String(summaryLines[2]))}</li>
            <li>${escapeHtml(String(result.model_metrics.selection_note ?? ""))}</li>
          </ul>
        </div>
      </section>
    </article>
  `;
}

function renderChatHistory() {
  if (!chatHistoryNode) return;

  if (!chatHistory.length) {
    chatHistoryNode.innerHTML = `<div class="chat-placeholder">Run an estimate first, then ask whether the car looks fair, risky, or worth negotiating on.</div>`;
    return;
  }

  chatHistoryNode.innerHTML = chatHistory
    .map((entry) => {
      const bubbleClass = entry.role === "assistant" ? "chat-bubble assistant" : "chat-bubble user";
      const bullets = entry.bullets?.length
        ? `<ul>${entry.bullets.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`
        : "";
      return `
        <article class="${bubbleClass}">
          <p>${escapeHtml(entry.text)}</p>
          ${bullets}
        </article>
      `;
    })
    .join("");

  chatHistoryNode.scrollTop = chatHistoryNode.scrollHeight;
}

function seedChat(result: AnalysisResponse) {
  chatHistory = [
    {
      role: "assistant",
      text: `Your latest estimate is ${money(result.price_range.low)} to ${money(result.price_range.high)} with ${result.confidence.toLowerCase()} confidence.`,
      bullets: [
        result.asking_price_comparison ?? "No asking price comparison is available yet.",
        result.negotiation_tip,
      ],
    },
  ];
  renderChatHistory();
}

function setChatOpen(open: boolean) {
  if (!chatSidebar || !chatOverlay) return;
  chatSidebar.classList.toggle("is-open", open);
  chatOverlay.classList.toggle("hidden", !open);
}

function openReportModal() {
  if (!latestAnalysis || !reportModal || !reportBody) return;
  reportBody.innerHTML = buildReportMarkup(latestAnalysis);
  reportModal.classList.remove("hidden");
}

function printReport() {
  if (!latestAnalysis) return;

  const reportWindow = window.open("", "_blank", "width=1040,height=880");
  if (!reportWindow) return;

  reportWindow.document.write(`
    <html>
      <head>
        <title>CarSight Estimate Report</title>
        <style>
          body { font-family: "Segoe UI", Arial, sans-serif; margin: 0; background: #ffffff; color: #1f2933; }
          .report-page { max-width: 920px; margin: 0 auto; padding: 32px; }
          .report-cover, .report-section, .report-grid > div { border: 1px solid #d7dde7; border-radius: 8px; background: #ffffff; }
          .report-cover { display: flex; justify-content: space-between; gap: 24px; align-items: flex-start; padding: 20px; margin-bottom: 16px; }
          .report-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px; }
          .report-section { padding: 18px; }
          .section-label { margin: 0 0 8px; font-size: 12px; font-weight: 700; text-transform: uppercase; color: #9f1239; }
          .muted-copy { color: #5c6675; }
          .report-score { display: grid; justify-items: end; gap: 6px; }
          .report-score span { display: inline-flex; align-items: center; justify-content: center; min-height: 36px; padding: 0 12px; border-radius: 8px; background: #fff1f2; color: #9f1239; font-weight: 700; }
          .report-summary { font-size: 18px; line-height: 1.45; }
          .report-list, .mini-metrics { display: grid; gap: 10px; margin: 0; }
          .report-list div, .mini-metrics div { display: flex; justify-content: space-between; gap: 12px; padding: 10px 0; border-bottom: 1px solid #eceff4; }
          .report-list div:last-child, .mini-metrics div:last-child { border-bottom: 0; }
          dt { font-weight: 700; } dd { margin: 0; text-align: right; }
          .issue-row { display: flex; justify-content: space-between; gap: 14px; padding: 12px 0; border-bottom: 1px solid #eceff4; }
          .issue-row:last-child { border-bottom: 0; }
          .issue-title { margin: 0 0 4px; font-weight: 700; text-transform: capitalize; }
          .issue-copy p:last-child { margin: 0; color: #5c6675; }
          .issue-cost { display: grid; justify-items: end; gap: 8px; }
          .severity { display: inline-flex; align-items: center; justify-content: center; min-height: 26px; min-width: 72px; border-radius: 8px; padding: 0 8px; font-size: 12px; font-weight: 700; text-transform: capitalize; }
          .severity-low { background: #dcfce7; color: #166534; }
          .severity-medium { background: #fef3c7; color: #92400e; }
          .severity-high { background: #fee2e2; color: #991b1b; }
          .range-axis, .range-foot { display: flex; justify-content: space-between; font-size: 13px; color: #5c6675; }
          .range-track { position: relative; height: 18px; border-radius: 999px; background: #eceff4; margin: 14px 0 26px; }
          .range-band { position: absolute; top: 0; height: 18px; border-radius: 999px; background: linear-gradient(90deg, #f97316, #dc2626); }
          .range-marker { position: absolute; top: -8px; width: 2px; height: 34px; background: #111827; }
          .range-marker span { position: absolute; top: -26px; left: 50%; transform: translateX(-50%); white-space: nowrap; font-size: 12px; font-weight: 700; color: #111827; }
          .range-marker.asking { background: #15803d; }
          .range-marker.asking span { color: #166534; top: 26px; }
          .range-foot div { display: grid; gap: 4px; }
          .report-bullets { margin: 12px 0 0; padding-left: 18px; }
        </style>
      </head>
      <body>
        ${buildReportMarkup(latestAnalysis)}
      </body>
    </html>
  `);
  reportWindow.document.close();
  reportWindow.focus();
  reportWindow.print();
}

function renderAnalysis(result: AnalysisResponse) {
  latestAnalysis = result;
  resultEmpty?.classList.add("hidden");
  resultContent?.classList.remove("hidden");
  openReportBtn?.removeAttribute("disabled");

  const modelName = escapeHtml(String(result.model_metrics.model_name ?? "Hybrid model"));
  const explanationList = result.explanation_points.map((point) => `<li>${escapeHtml(point)}</li>`).join("");

  resultContent!.innerHTML = `
    <section class="valuation-section">
      <div class="section-heading section-heading-inline">
        <div>
          <p class="section-label">Price meter</p>
          <h2>See where the listing lands before you trust the price.</h2>
        </div>
        <div class="section-actions">
          <button id="report-preview-btn" class="ghost-btn" type="button">View report</button>
          <button id="chat-open-inline-btn" class="ghost-btn" type="button">Open chat</button>
        </div>
      </div>

      <div class="deal-layout">
        ${buildDealMeter(result)}

        <div class="deal-side-panel">
          <div class="deal-info-card">
            <p class="section-label">Listing check</p>
            <h3>${escapeHtml(dealStatus(result).label)}</h3>
            <p class="summary-copy">${escapeHtml(result.asking_price_comparison ?? "No asking price was entered, so the estimate is shown as a stand-alone fair range.")}</p>
          </div>

          <div class="deal-info-card">
            <p class="section-label">Negotiation angle</p>
            <h3>How to use it</h3>
            <p class="summary-copy">${escapeHtml(result.negotiation_tip)}</p>
          </div>

          <div class="deal-info-card">
            <p class="section-label">Damage read</p>
            <h3>Main impact</h3>
            <p class="summary-copy">${escapeHtml(buildDamageHeadline(result))}</p>
          </div>
        </div>
      </div>
    </section>

    <section class="detail-grid">
      <div class="detail-section">
        <div class="section-heading section-heading-inline">
          <div>
            <p class="section-label">Damage findings</p>
            <h3>Detected issues and estimated impact</h3>
          </div>
        </div>
        ${buildIssueRows(result)}
      </div>

      <div class="detail-section">
        <div class="section-heading">
          <p class="section-label">Why the estimate moved</p>
          <h3>What influenced the result</h3>
        </div>
        <ul class="explanation-list">${explanationList}</ul>
      </div>
    </section>

    <section class="detail-grid">
      <div class="detail-section">
        <div class="section-heading section-heading-inline">
          <div>
            <p class="section-label">Professional report</p>
            <h3>One-page valuation summary</h3>
          </div>
          <button id="report-open-secondary-btn" class="ghost-btn" type="button">Open report</button>
        </div>
        <div class="report-preview">
          <p class="report-preview-copy">Open the clean one-page summary with the estimate range, listing check, and damage findings.</p>
          <ul class="preview-bullets">
            <li>${escapeHtml(recommendationLabel(result))}</li>
            <li>${escapeHtml(result.negotiation_tip)}</li>
            <li>Damage estimate applied: ${(result.extracted_damage.total_penalty_pct * 100).toFixed(1)}%</li>
          </ul>
        </div>
      </div>

      <div class="detail-section">
        <div class="section-heading">
          <p class="section-label">Method used</p>
          <h3>Best-performing model after comparison</h3>
        </div>
        ${buildModelSummary(result)}
        <p class="muted-copy method-comparison">Compared baseline Ridge NLP, expanded TF-IDF Ridge NLP, HistGradientBoosting, and the final hybrid blend. ${modelName} produced the lowest held-out error.</p>
      </div>
    </section>
  `;

  document.querySelector<HTMLButtonElement>("#report-preview-btn")?.addEventListener("click", openReportModal);
  document.querySelector<HTMLButtonElement>("#report-open-secondary-btn")?.addEventListener("click", openReportModal);
  document.querySelector<HTMLButtonElement>("#chat-open-inline-btn")?.addEventListener("click", () => setChatOpen(true));

  seedChat(result);
}

async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE}/health`);
    const body = await response.json();
    if (body.model === "ready") {
      statusText!.textContent = "Backend connected. Updated model is ready.";
      statusText!.className = "status-text ok";
      return;
    }
    statusText!.textContent = "Backend connected, but model artifacts are missing.";
    statusText!.className = "status-text warn";
  } catch (_error) {
    statusText!.textContent = "Backend is not running yet.";
    statusText!.className = "status-text warn";
  }
}

function resetSample() {
  if (!form) return;
  form.reset();
  (form.elements.namedItem("make") as HTMLInputElement).value = "Honda";
  (form.elements.namedItem("model") as HTMLInputElement).value = "Accord";
  (form.elements.namedItem("year") as HTMLInputElement).value = "2018";
  (form.elements.namedItem("mileage") as HTMLInputElement).value = "82000";
  (form.elements.namedItem("engine") as HTMLInputElement).value = "2.0L I-4 252HP";
  (form.elements.namedItem("color") as HTMLInputElement).value = "Gray";
  (form.elements.namedItem("asking_price") as HTMLInputElement).value = "14900";
  (form.elements.namedItem("damage_description") as HTMLTextAreaElement).value =
    "Rear bumper cracked and engine knocks when cold. Minor scratches on driver-side door.";
}

async function submitAnalysis(event: SubmitEvent) {
  event.preventDefault();
  if (!form) return;

  analyzeBtn!.disabled = true;
  analyzeBtn!.textContent = "Running...";
  statusText!.textContent = "Running estimate...";
  statusText!.className = "status-text";

  const formData = new FormData(form);
  const payload = {
    make: String(formData.get("make") || ""),
    model: String(formData.get("model") || ""),
    year: Number(formData.get("year") || 0),
    mileage: Number(formData.get("mileage") || 0),
    engine: String(formData.get("engine") || ""),
    fuel_type: String(formData.get("fuel_type") || ""),
    transmission: String(formData.get("transmission") || ""),
    drivetrain: String(formData.get("drivetrain") || ""),
    color: String(formData.get("color") || ""),
    damage_description: String(formData.get("damage_description") || ""),
    asking_price: formData.get("asking_price") ? Number(formData.get("asking_price")) : null,
  };

  try {
    const response = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const body = await response.json();
      throw new Error(body.detail || "Analysis failed.");
    }

    const result = (await response.json()) as AnalysisResponse;
    renderAnalysis(result);
    statusText!.textContent = "Estimate ready.";
    statusText!.className = "status-text ok";
  } catch (error) {
    statusText!.textContent = error instanceof Error ? error.message : "Something went wrong.";
    statusText!.className = "status-text error";
  } finally {
    analyzeBtn!.disabled = false;
    analyzeBtn!.textContent = "Run estimate";
  }
}

async function askCarSight(question: string) {
  if (!latestAnalysis) {
    chatHistory = [
      {
        role: "assistant",
        text: "Run an estimate first so I have a real vehicle result to talk through.",
      },
    ];
    renderChatHistory();
    setChatOpen(true);
    return;
  }

  chatHistory.push({ role: "user", text: question });
  chatHistory.push({ role: "assistant", text: "Thinking through the latest estimate..." });
  renderChatHistory();

  try {
    const response = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, analysis: latestAnalysis }),
    });
    const body = (await response.json()) as ChatResponse;
    chatHistory.pop();
    chatHistory.push({ role: "assistant", text: body.answer, bullets: body.bullets });
    renderChatHistory();
  } catch (_error) {
    chatHistory.pop();
    chatHistory.push({
      role: "assistant",
      text: "The chatbot could not answer right now. Try again after re-running the estimate.",
    });
    renderChatHistory();
  }
}

form?.addEventListener("submit", submitAnalysis);
sampleBtn?.addEventListener("click", resetSample);
openReportBtn?.addEventListener("click", openReportModal);
closeReportBtn?.addEventListener("click", () => reportModal?.classList.add("hidden"));
printReportBtn?.addEventListener("click", printReport);
chatLaunch?.addEventListener("click", () => setChatOpen(true));
chatClose?.addEventListener("click", () => setChatOpen(false));
chatOverlay?.addEventListener("click", () => setChatOpen(false));
chatForm?.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = chatInput?.value.trim();
  if (!question) return;
  chatInput!.value = "";
  void askCarSight(question);
});

quickPrompts.forEach((button) => {
  button.addEventListener("click", () => {
    const question = button.dataset.question;
    if (!question) return;
    if (chatInput) chatInput.value = question;
    void askCarSight(question);
  });
});

resetSample();
renderChatHistory();
void checkHealth();
