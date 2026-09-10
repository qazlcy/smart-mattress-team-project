const $ = (selector) => document.querySelector(selector);
let replay;
let frame = 0;
let timer;

const regionColors = ["#44dbc8", "#f7d85a", "#ff8b4a", "#ee5a75", "#9ad66b", "#8fb8ff"];

function color(value) {
  const n = Math.min(1, value / 300);
  const r = Math.round(255 * Math.min(1, n * 2));
  const g = Math.round(210 * Math.sin(n * Math.PI));
  const b = Math.round(110 + 145 * (1 - n));
  return `rgb(${r},${g},${b})`;
}

function drawHeatmap(data, metrics) {
  const canvas = $("#heatmap");
  const context = canvas.getContext("2d");
  const cellWidth = canvas.width / 24;
  const cellHeight = canvas.height / 56;
  context.clearRect(0, 0, canvas.width, canvas.height);
  data.forEach((row, rowIndex) => {
    row.forEach((value, colIndex) => {
      context.fillStyle = color(value);
      context.fillRect(colIndex * cellWidth, rowIndex * cellHeight, Math.ceil(cellWidth), Math.ceil(cellHeight));
    });
  });
  metrics.bodyRegions.forEach((region, index) => {
    context.strokeStyle = regionColors[index % regionColors.length];
    context.lineWidth = 3;
    context.strokeRect(
      region.startCol * cellWidth,
      region.startRow * cellHeight,
      (region.endCol - region.startCol) * cellWidth,
      (region.endRow - region.startRow) * cellHeight
    );
  });
}

function renderRegions(regions) {
  $("#regions").innerHTML = regions.map((region, index) => `
    <div class="region-row">
      <i style="background:${regionColors[index % regionColors.length]}"></i>
      <span>${region.label}</span>
      <b>${region.loadShare}%</b>
    </div>
  `).join("");
}

function renderWeight(stable, instant) {
  $("#userType").textContent = stable.userType.label;
  $("#userType").dataset.kind = stable.userType.key;
  if (stable.available) {
    $("#stableWeight").textContent = `${stable.prediction.kg} kg`;
    $("#stableWeightRange").textContent = stable.prediction.interval.label;
    $("#stableWeightSource").textContent = stable.prediction.source;
    $("#stableWeightEvidence").textContent = `${stable.frameCount} 帧多序列特征中位数 · 正式评价口径`;
  } else {
    $("#stableWeight").textContent = "暂无聚合结果";
    $("#stableWeightRange").textContent = "--";
    $("#stableWeightSource").textContent = "需要本地静态采集数据";
    $("#stableWeightEvidence").textContent = "当前仅可展示即时估计";
  }
  $("#weight").textContent = instant.kg == null ? "未检测到载荷" : `${instant.kg} kg`;
  $("#weightRange").textContent = instant.interval ? instant.interval.label : "--";
}

function draw() {
  if (!replay) return;
  const data = replay.frames[frame];
  const metrics = replay.metrics[frame];
  drawHeatmap(data, metrics);
  $("#frame").textContent = `第 ${frame + 1} / ${replay.frames.length} 帧`;
  $("#posture").textContent = metrics.posture;
  $("#postureSource").textContent = metrics.postureSource;
  $("#max").textContent = metrics.maxPressure;
  $("#avg").textContent = metrics.averagePressure;
  $("#area").textContent = `${metrics.contactAreaIndex}%`;
  $("#regionSource").textContent = metrics.bodyRegionSource;
  $("#baseline").textContent = metrics.emptyBaselineApplied ? "已应用空载校正" : "未检测到空载基线";
  $("#airbags").innerHTML = metrics.airbags.map((airbag) => `
    <div class="airbag">
      <b><i class="airbag-dot" style="background:${airbag.color}"></i>${airbag.name}</b>
      <div class="bar"><i style="width:${Math.min(100, airbag.pressure)}%; background:${airbag.color}"></i></div>
      <span>${airbag.state}</span>
    </div>
  `).join("");
  renderRegions(metrics.bodyRegions);
  renderWeight(replay.stableWeightPrediction, metrics.weightPrediction);
}

async function load() {
  clearInterval(timer);
  timer = null;
  $("#play").textContent = "播放";
  frame = 0;
  const user = $("#user").value;
  const sequence = $("#sequence").value;
  replay = await fetch(`/api/replay?user=${encodeURIComponent(user)}&sequence=${sequence}`).then((response) => response.json());
  $("#source").textContent = replay.source;
  draw();
}

async function init() {
  const data = await fetch("/api/users").then((response) => response.json());
  $("#user").innerHTML = data.users.map((user) => `<option>${user}</option>`).join("");
  await load();
}

$("#load").onclick = load;
$("#play").onclick = () => {
  if (timer) {
    clearInterval(timer);
    timer = null;
    $("#play").textContent = "播放";
  } else {
    timer = setInterval(() => {
      frame = (frame + 1) % replay.frames.length;
      draw();
    }, 160);
    $("#play").textContent = "暂停";
  }
};
init();
