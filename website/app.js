// The website's behaviour: the results viewer (video, events, timeline and risk in step), the
// operator's view, scores, clips, the data section's charts, the team and the links. Data:
// data/*.json (tools/website_data.py); videos and clips: SITE.media (tools/render_samples.py).
"use strict";

const COLOURS = {  // the same colours as the annotated videos (src/visualize.py)
  jaywalking: "#ffa500",
  failure_to_yield: "#e63c3c",
  red_light: "#b52222",
  stop_line: "#e0b000",
  solid_line_crossing: "#3c78e6",
};
const OTHER = "#8a919c";
const LIGHT = { red: "#dc3545", amber: "#f5a300", green: "#2f9e62", unknown: "#c4c9d1" };
const VIDEOS = [
  ["C3896", "C3896", "morning"],
  ["C3897", "C3897", "morning"],
  ["C3902", "C3902", "afternoon, labelled"],
  ["C3905", "C3905", "evening, labelled"],
];
const GROUP_COLOURS = { people: "#ffa500", cars: "#3c78e6", "buses and trucks": "#9b4dca", "two-wheelers": "#2f9e62" };
const ALARM = 0.5;  // the metric's alarm threshold (evaluate.py)
const PLOT = { responsive: true, displayModeBar: false };
const MARGIN = { l: 150, r: 16, t: 36, b: 40 };  // the same left margin keeps the charts' time axes aligned

const $ = (selector) => document.querySelector(selector);
const colour = (label) => COLOURS[label] || OTHER;
const mmss = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
const chip = (label) => `<span class="chip" style="background:${colour(label)}">${label}</span>`;

async function json(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.json();
}

function overlap(a, b) {  // temporal IoU, as evaluate.py computes it
  const inter = Math.max(0, Math.min(a[1], b[1]) - Math.max(a[0], b[0]));
  const union = Math.max(a[1], b[1]) - Math.min(a[0], b[0]);
  return union > 0 ? inter / union : 0;
}

function bestOverlap(segment, others) {
  return Math.max(0, ...others.filter((o) => o[2] === segment[2]).map((o) => overlap(segment, o)));
}

function valueAt(curve, t) {  // the curve's value at or before t (binary search)
  if (!curve.length || t < curve[0][0]) return 0;
  let lo = 0;
  let hi = curve.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (curve[mid][0] <= t) lo = mid; else hi = mid - 1;
  }
  return curve[lo][1];
}

function alarms(curve) {  // runs at or above the threshold; runs under 2 s apart merge (evaluate.py)
  const runs = [];
  let start = null;
  let last = null;
  for (const [t, s] of curve) {
    if (s >= ALARM) {
      if (start === null) start = t;
      last = t;
    } else if (start !== null) {
      runs.push([start, last]);
      start = null;
    }
  }
  if (start !== null) runs.push([start, last]);
  return runs.reduce((merged, run) => {
    const previous = merged[merged.length - 1];
    if (previous && run[0] - previous[1] < 2) previous[1] = run[1]; else merged.push(run);
    return merged;
  }, []);
}

const playhead = (t) => ({
  type: "line", x0: t, x1: t, y0: 0, y1: 1, yref: "paper", line: { color: "#16181d", width: 1.5 },
});

// ---------------------------------------------------------------- the results viewer
const viewer = { entry: null, lights: [], duration: 0 };

function selectVideo(stem, data) {
  const entry = data.results.videos.find((v) => v.video.startsWith(stem));
  const facts = data.eda.videos.find((v) => v.video.startsWith(stem));
  viewer.entry = entry;
  viewer.lights = facts ? facts.lights : [];
  viewer.duration = facts ? facts.duration : entry.risk[entry.risk.length - 1][0];
  for (const button of document.querySelectorAll("#video-tabs button")) {
    button.setAttribute("aria-selected", String(button.dataset.stem === stem));
  }
  const video = $("#player");
  video.poster = `img/posters/${stem}.jpg`;
  video.src = `${SITE.media}${stem}.mp4`;
  drawTimeline(entry);
  drawRisk(entry);
  listEvents(entry);
  showNow(0);
}

function drawTimeline(entry) {
  const labels = entry.labels || [];
  const classes = [...new Set([...entry.events, ...labels].map((e) => e[2]))].sort();
  const traces = [];
  if (labels.length) {
    traces.push({
      type: "bar", orientation: "h", name: "our labels", width: 0.8,
      y: labels.map((e) => e[2]), x: labels.map((e) => Math.max(e[1] - e[0], 0.6)),
      base: labels.map((e) => e[0]), customdata: labels.map((e) => [e[0], e[1]]),
      marker: { color: "rgba(22,24,29,0.08)", line: { color: "rgba(22,24,29,0.45)", width: 1 } },
      hovertemplate: "label %{customdata[0]:.1f}–%{customdata[1]:.1f} s<extra></extra>",
    });
  }
  for (const label of classes) {
    const events = entry.events.filter((e) => e[2] === label);
    if (!events.length) continue;
    traces.push({
      type: "bar", orientation: "h", name: label, width: labels.length ? 0.42 : 0.6,
      y: events.map(() => label), x: events.map((e) => Math.max(e[1] - e[0], 0.6)),
      base: events.map((e) => e[0]), customdata: events.map((e) => [e[0], e[1]]),
      marker: { color: colour(label) },
      hovertemplate: `${label} %{customdata[0]:.1f}–%{customdata[1]:.1f} s<extra></extra>`,
    });
  }
  const layout = {
    barmode: "overlay", showlegend: false, height: 96 + 44 * Math.max(classes.length, 1),
    margin: MARGIN, plot_bgcolor: "#fff", paper_bgcolor: "#fff",
    title: { text: labels.length ? "Events: ours (colour) over our labels (grey)" : "Events", x: 0, font: { size: 14 } },
    xaxis: { range: [0, viewer.duration], title: { text: "seconds", font: { size: 12 } }, zeroline: false },
    yaxis: { categoryorder: "array", categoryarray: classes.slice().reverse(), automargin: true },
    shapes: [playhead(0)],
  };
  Plotly.react("timeline", traces, layout, PLOT);
  onClick("timeline", (point) => seek(point.customdata[0] - 1));
}

function drawRisk(entry) {
  const bands = alarms(entry.risk).map(([a, b]) => ({
    type: "rect", x0: a - 0.5, x1: b + 0.5, y0: 0, y1: 1, yref: "paper",
    fillcolor: "rgba(214,69,45,0.16)", line: { width: 0 },
  }));
  const threshold = {
    type: "line", x0: 0, x1: 1, xref: "paper", y0: ALARM, y1: ALARM,
    line: { color: "#8a919c", dash: "dash", width: 1 },
  };
  const trace = {
    type: "scatter", mode: "lines", x: entry.risk.map((r) => r[0]), y: entry.risk.map((r) => r[1]),
    fill: "tozeroy", fillcolor: "rgba(214,69,45,0.12)", line: { color: "#d6452d", width: 1.5 },
    hovertemplate: "%{x:.1f} s: risk %{y:.2f}<extra></extra>",
  };
  const layout = {
    height: 230, margin: MARGIN, showlegend: false, plot_bgcolor: "#fff", paper_bgcolor: "#fff",
    title: { text: "Accident risk: the chance an accident starts within 5 s (dashed: alarm)", x: 0, font: { size: 14 } },
    xaxis: { range: [0, viewer.duration], title: { text: "seconds", font: { size: 12 } } },
    yaxis: { range: [0, 1], tickvals: [0, 0.5, 1] },
    shapes: [playhead(0), threshold, ...bands],
  };
  Plotly.react("risk", [trace], layout, PLOT);
  onClick("risk", (point) => seek(point.x - 1));
}

function onClick(id, handler) {  // one click handler per chart, whichever video is shown
  const div = document.getElementById(id);
  if (div.removeAllListeners) div.removeAllListeners("plotly_click");
  div.on("plotly_click", (event) => handler(event.points[0]));
}

function listEvents(entry) {
  const labels = entry.labels;
  const ours = entry.events.map((e) => ({ start: e[0], end: e[1], label: e[2], best: labels ? bestOverlap(e, labels) : null }));
  const missed = labels ? labels.filter((l) => bestOverlap(l, entry.events) < 0.3) : [];
  $("#list-title").textContent = `${ours.length} events`;
  $("#list-note").textContent = labels
    ? `${ours.filter((e) => e.best >= 0.5).length} match a label · ${missed.length} labels missed`
    : "no labels for this video";
  const list = $("#events");
  list.innerHTML = "";
  for (const event of ours) {
    const item = document.createElement("li");
    item.dataset.start = event.start;
    item.dataset.end = event.end;
    let match = "";
    if (event.best !== null) {
      const kind = event.best >= 0.5 ? "ok" : event.best < 0.3 ? "miss" : "";
      const text = event.best >= 0.5 ? `matches a label (overlap ${event.best.toFixed(2)})`
        : event.best >= 0.3 ? `partly matches a label (overlap ${event.best.toFixed(2)})`
          : "no matching label: a false alarm";
      match = `<span class="match ${kind}">${text}</span>`;
    }
    item.innerHTML = `${chip(event.label)}<span class="when">${mmss(event.start)}–${mmss(event.end)}</span>${match}`;
    item.addEventListener("click", () => seek(event.start - 1));
    list.appendChild(item);
  }
}

function showNow(t) {
  const entry = viewer.entry;
  if (!entry) return;
  $("#now-time").textContent = `${t.toFixed(1)} s`;
  const under = entry.events.filter((e) => e[0] <= t && t < e[1]);
  $("#now-events").innerHTML = under.length ? under.map((e) => chip(e[2])).join("") : '<span class="none">no event under way</span>';
  const risk = valueAt(entry.risk, t);
  $("#now-risk").textContent = risk.toFixed(2);
  const gauge = $("#now-gauge");
  gauge.style.width = `${Math.min(100, risk * 100)}%`;
  gauge.classList.toggle("alarm", risk >= ALARM);
  const phase = viewer.lights.find((l) => l[0] <= t && t < l[1]);
  const light = $("#now-light");
  light.style.background = LIGHT[phase ? phase[2] : "unknown"];
  light.title = `The top road's light as read: ${phase ? phase[2] : "not read"}`;
  for (const item of document.querySelectorAll("#events li")) {
    item.classList.toggle("active", Number(item.dataset.start) <= t && t < Number(item.dataset.end));
  }
}

function seek(t) {
  const video = $("#player");
  video.currentTime = Math.max(0, t);
  video.play().catch(() => {});
  const box = video.getBoundingClientRect();
  if (box.top < 0 || box.bottom > window.innerHeight) video.scrollIntoView({ behavior: "smooth", block: "center" });
}

function follow() {  // the charts' playheads and the "now" panel follow the video
  const video = $("#player");
  let pending = false;
  video.addEventListener("timeupdate", () => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      const t = video.currentTime;
      showNow(t);
      for (const id of ["timeline", "risk"]) Plotly.relayout(id, { "shapes[0].x0": t, "shapes[0].x1": t });
    });
  });
}

// ---------------------------------------------------------------- the operator's view, scores, clips
function operatorView(data) {
  const classes = [...new Set(data.results.videos.flatMap((v) => v.events.map((e) => e[2])))].sort();
  const names = data.results.videos.map((v) => v.video.replace(".MP4", ""));
  const traces = classes.map((label) => ({
    type: "bar", name: label, x: names, marker: { color: colour(label) },
    y: data.results.videos.map((v) => v.events.filter((e) => e[2] === label).length),
    hovertemplate: `%{x}: %{y} × ${label}<extra></extra>`,
  }));
  Plotly.react("per-video", traces, {
    barmode: "stack", height: 330, margin: { l: 40, r: 10, t: 36, b: 40 },
    title: { text: "Events per video, by type", x: 0, font: { size: 14 } },
    legend: { orientation: "h", y: -0.18 }, plot_bgcolor: "#fff", paper_bgcolor: "#fff",
  }, PLOT);
  const cards = $("#video-cards");
  for (const v of data.results.videos) {
    const facts = data.eda.videos.find((f) => f.video === v.video);
    const peak = Math.max(...v.risk.map((r) => r[1]));
    const runs = alarms(v.risk).length;
    const perMinute = v.events.length / ((facts ? facts.duration : 60) / 60);
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `<b>${v.video.replace(".MP4", "")}</b><small>${facts ? facts.time_of_day : ""}</small>
      <div class="big">${v.events.length}</div><small>events, ${perMinute.toFixed(1)} a minute</small>
      <small>highest risk <b>${peak.toFixed(2)}</b> · ${runs} alarm${runs === 1 ? "" : "s"}</small>`;
    cards.appendChild(card);
  }
}

function scoresTable(scores) {
  const rows = Object.entries(scores.classes).sort((a, b) => b[1].mean - a[1].mean);
  const cell = (value) => `<td class="num">${value.toFixed(2)}</td>`;
  $("#scores").innerHTML = `<thead><tr><th>Class</th><th class="num">IoU 0.3</th><th class="num">IoU 0.5</th><th class="num">IoU 0.7</th><th class="num">Mean F1</th></tr></thead>
    <tbody>${rows.map(([label, f]) => `<tr><td>${chip(label)}</td>${cell(f["0.3"])}${cell(f["0.5"])}${cell(f["0.7"])}
      <td class="num bar-cell"><i style="width:${Math.round(f.mean * 100)}%"></i><span>${f.mean.toFixed(3)}</span></td></tr>`).join("")}
    <tr class="total"><td>Score A (${scores.videos.map((v) => v.replace(".MP4", "")).join(" + ")})</td><td></td><td></td><td></td><td class="num">${scores.score_a.toFixed(3)}</td></tr></tbody>`;
}

function clipCards(list, target, folder) {
  for (const clip of list) {
    const card = document.createElement("article");
    card.className = "clip";
    const head = folder === "examples"
      ? `${chip(clip.id)} <span class="overlap">overlap with our label ${clip.overlap.toFixed(2)}</span>`
      : `<b>${clip.title}</b>`;
    card.innerHTML = `<video controls muted playsinline preload="metadata" src="${SITE.media}${folder}/${clip.id}.mp4#t=0.5"></video>
      <div>${head}<p>${clip.caption}</p><small>${clip.video.replace(".MP4", "")}, ${mmss(clip.start)}–${mmss(clip.end)}</small></div>`;
    target.appendChild(card);
  }
}

// ---------------------------------------------------------------- the data section
function factsTable(eda) {
  const rows = eda.videos.map((v) => `<tr><td><b>${v.video.replace(".MP4", "")}</b></td><td>${v.time_of_day}</td>
    <td>${v.width} × ${v.height}, ${v.fps} fps</td><td>${mmss(v.duration)}</td><td>${v.brightness ?? "–"}</td>
    <td>${v.camera_shift_px} px, ${v.camera_rotation_deg}°</td><td>${v.cycle_sec ? `${v.cycle_sec} s` : "–"}</td>
    <td>${v.tracks.people} people, ${v.tracks.cars} cars, ${v.tracks["buses and trucks"]} buses and trucks</td></tr>`);
  $("#facts").innerHTML = `<thead><tr><th>Video</th><th>Light</th><th>Format</th><th>Length</th><th>Brightness</th>
    <th>Camera off the reference</th><th>Signal cycle</th><th>Tracked</th></tr></thead><tbody>${rows.join("")}</tbody>`;
}

function edaCharts(facts) {
  const t = facts.counts.people.map((_, i) => i * facts.bins_sec);
  const lines = Object.keys(GROUP_COLOURS).map((group) => ({
    type: "scatter", mode: "lines", name: group, x: t, y: facts.counts[group],
    line: { color: GROUP_COLOURS[group], width: 2, shape: "spline", smoothing: 0.6 },
    hovertemplate: `%{x} s: %{y:.1f} ${group}<extra></extra>`,
  }));
  Plotly.react("counts", lines, {
    height: 300, margin: { l: 50, r: 10, t: 36, b: 40 }, plot_bgcolor: "#fff", paper_bgcolor: "#fff",
    title: { text: `Road users in view over time, ${facts.video.replace(".MP4", "")} (average per frame, 5 s bins)`, x: 0, font: { size: 14 } },
    xaxis: { title: { text: "seconds", font: { size: 12 } } }, legend: { orientation: "h", y: -0.25 },
  }, PLOT);
  const phases = facts.lights.filter((l) => l[2] !== "unknown").map(([a, b, name]) => ({
    type: "rect", x0: a, x1: b, y0: 0, y1: 1, yref: "paper", line: { width: 0 }, layer: "below",
    fillcolor: { red: "rgba(220,53,69,0.13)", amber: "rgba(245,163,0,0.2)", green: "rgba(47,158,98,0.13)" }[name],
  }));
  Plotly.react("density", [{
    type: "bar", x: t, y: facts.counts["vehicles standing"], marker: { color: "#16181d" }, opacity: 0.75,
    hovertemplate: "%{x} s: %{y:.1f} vehicles standing<extra></extra>",
  }], {
    height: 260, margin: { l: 50, r: 10, t: 36, b: 40 }, plot_bgcolor: "#fff", paper_bgcolor: "#fff", bargap: 0.05,
    title: { text: "Vehicles standing, over the top road's light as read (red, amber, green bands)", x: 0, font: { size: 14 } },
    xaxis: { title: { text: "seconds", font: { size: 12 } } }, shapes: phases,
  }, PLOT);
}

// ---------------------------------------------------------------- team, links, demo
function teamCards(team) {
  const target = $("#team-cards");
  for (const person of team.members) {
    const links = [["GitHub", person.github], ["LinkedIn", person.linkedin], ["Portfolio", person.portfolio]]
      .filter(([, url]) => url).map(([name, url]) => `<a href="${url}" target="_blank" rel="noopener">${name}</a>`).join("");
    const card = document.createElement("article");
    card.className = "person";
    card.innerHTML = `<h4>${person.name}</h4><div class="role">${person.role}</div>
      <ul>${person.did.map((d) => `<li>${d}</li>`).join("")}</ul>
      ${person.projects.length ? `<small><b>Projects:</b> ${person.projects.join(", ")}</small>` : ""}
      <div class="person-links">${links}</div>`;
    target.appendChild(card);
  }
}

function linkList() {
  const tree = `${SITE.repo}/tree/${SITE.tag}`;
  const blob = `${SITE.repo}/blob/${SITE.tag}`;
  const links = [
    ["The repository", `${SITE.repo}`, `code, README; the submission is tag ${SITE.tag}`],
    ["Model weights", `${tree}/weights`, "YOLO26m, TorchScript FP16, 41.5 MB"],
    ["predictions_samples.json", `${blob}/predictions_samples.json`, "our output on the four samples"],
    ["The live demo", SITE.demo, "upload a clip, get its events back"],
    ["Annotated videos and clips", SITE.mediaPage, "on Hugging Face"],
    ["The report", "#report", "what worked, what didn't, what next"],
  ];
  $("#link-list").innerHTML = links.map(([name, url, note]) => `<li><a href="${url}"${url.startsWith("#") ? "" : ' target="_blank" rel="noopener"'}>${name}<small>${note}</small></a></li>`).join("");
}

function wireStatic() {
  for (const el of document.querySelectorAll("[data-team]")) el.textContent = SITE.team;
  for (const el of document.querySelectorAll("[data-demo-link]")) el.href = SITE.demo;
  for (const el of document.querySelectorAll("[data-repo-link]")) el.href = SITE.repo;
  $("#demo-load").addEventListener("click", () => {
    $("#demo-frame").innerHTML = `<iframe src="${SITE.demo}/?embed=true" title="The live demo" loading="lazy" allow="fullscreen"></iframe>`;
  });
}

// ---------------------------------------------------------------- start
document.addEventListener("DOMContentLoaded", async () => {
  wireStatic();
  linkList();
  let data;
  try {
    const [results, eda, clips, team] = await Promise.all(
      ["data/results.json", "data/eda.json", "data/clips.json", "data/team.json"].map(json));
    data = { results, eda, clips, team };
  } catch (error) {
    $("#results .intro").insertAdjacentHTML("afterend", `<p class="match miss">The data didn't load (${error.message}).</p>`);
    return;
  }
  const tabs = $("#video-tabs");
  for (const [stem, name, note] of VIDEOS) {
    const button = document.createElement("button");
    button.dataset.stem = stem;
    button.setAttribute("role", "tab");
    button.innerHTML = `${name}<small>${note}</small>`;
    button.addEventListener("click", () => selectVideo(stem, data));
    tabs.appendChild(button);
  }
  follow();
  selectVideo("C3902", data);
  operatorView(data);
  scoresTable(data.results.scores);
  clipCards(data.clips.examples, $("#examples"), "examples");
  clipCards(data.clips.failures, $("#failures"), "failures");
  factsTable(data.eda);
  const edaTabs = $("#eda-tabs");
  for (const facts of data.eda.videos) {
    const button = document.createElement("button");
    button.textContent = facts.video.replace(".MP4", "");
    button.addEventListener("click", () => {
      for (const b of edaTabs.children) b.setAttribute("aria-selected", String(b === button));
      edaCharts(facts);
    });
    edaTabs.appendChild(button);
  }
  edaTabs.children[0].click();
  teamCards(data.team);
});
