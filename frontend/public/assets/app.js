const $ = id => document.getElementById(id);
let classNames = [];
const colors = ['#b42046', '#21845a', '#6946ad'];
let latest = null, selected = null, records = [], data = null;
let scatterView = null, scatterFrame = null, scatterPan = null;
let scatterScroll = null;
const dataReady = fetch('/api/dataset').then(r => {
    if (!r.ok)
        throw Error('分布データを取得できません');
    return r.json();
}).then(d => {
    data = d;
    classNames = d.class_names;
    $('dataset-name').textContent = `${d.name} / ${d.variables.length} Features`;
    document.querySelectorAll('.legend span').forEach((node,i)=>node.textContent=classNames[i]);
    const examples=document.querySelectorAll('[data-prompt]');
    examples[1].textContent=`${classNames[0]}の特徴`;
    examples[1].dataset.prompt=`${classNames[0]}の特徴を教えて`;
    examples[2].textContent=`${classNames[0]}と${classNames[1]}を比較`;
    examples[2].dataset.prompt=`${classNames[0]}と${classNames[1]}を比較して`;
    examples[3].textContent=`${classNames[0]}の内部構造の特性が見たい`;
    examples[3].dataset.prompt=`${classNames[0]}の内部構造の特性が見たい`;
    $('prompt').placeholder=`例：${classNames[2]}の特徴を教えて`;
    return d;
});
dataReady.catch(e => {
    $('status').textContent = e.message;
});
const el = (tag, text, cls) => {
    const e = document.createElement(tag);
    if (text !== undefined)
        e.textContent = text;
    if (cls)
        e.className = cls;
    return e;
};
const svgEl = (tag, attrs = {}, text) => {
    const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [k, v] of Object.entries(attrs))
        e.setAttribute(k, v);
    if (text !== undefined)
        e.textContent = text;
    return e;
};
function table(container, headers, rows) {
    const t = el('table'), head = el('tr');
    headers.forEach(h => head.append(el('th', h)));
    t.append(head);
    rows.forEach(row => {
        const tr = el('tr');
        row.forEach(v => tr.append(el('td', v)));
        t.append(tr);
    });
    container.replaceChildren(t);
}
const fmt = x => Number(x).toFixed(4);
const fmtParameter = x => Number(x).toPrecision(2);
function parameters() {
    const container = $('weights');
    container.classList.remove('empty');
    container.replaceChildren();
    for (let k = 0; k < 3; k++) {
        const card = el('div', undefined, 'weight-card');
        card.style.setProperty('--color', colors[k]);
        card.append(el('strong', classNames[k]));
        for (const key of ['w_tg', 'w_bg', 'w_bw']) {
            const row = el('label', undefined, 'manual-weight');
            row.append(el('span', key));
            const input = el('input');
            input.type = 'range';
            input.min = '0';
            input.max = '1';
            input.step = '0.01';
            input.value = latest.parameters[key][k];
            input.dataset.parameter = key;
            input.dataset.label = k;
            const value = el('output', fmtParameter(latest.parameters[key][k]));
            input.oninput = () => value.textContent = fmtParameter(Number(input.value));
            row.append(input, value);
            card.append(row);
        }
        container.append(card);
    }
    const alpha = el('label', undefined, 'manual-weight alpha-control');
    alpha.append(el('span', 'alpha'));
    const alphaInput = el('input');
    alphaInput.type = 'range';
    alphaInput.min = '0';
    alphaInput.max = '20';
    alphaInput.step = '0.1';
    alphaInput.value = latest.parameters.alpha;
    alphaInput.dataset.parameter = 'alpha';
    const alphaValue = el('output', fmtParameter(latest.parameters.alpha));
    alphaInput.oninput = () => alphaValue.textContent = fmtParameter(Number(alphaInput.value));
    alpha.append(alphaInput, alphaValue);
    container.append(alpha, el('button', 'パラメータを適用', 'apply-parameters'));
    container.querySelector('.apply-parameters').onclick = applyManualParameters;
}
function scatter() {
    if (!latest)
        return;
    const points = latest.coordinates;
    const xmin = Math.min(...points.map(p => p[0])), xmax = Math.max(...points.map(p => p[0]));
    const ymin = Math.min(...points.map(p => p[1])), ymax = Math.max(...points.map(p => p[1]));
    const W = Math.max(120, $('scatter').clientWidth), H = Math.max(120, $('scatter').clientHeight);
    // Scale each axis independently so narrow projections remain readable.
    // The underlying coordinates and tick values remain unchanged.
    const xPad = Math.max((xmax - xmin) * .05, .05), yPad = Math.max((ymax - ymin) * .05, .05);
    if (!scatterView)
        scatterView = { cx: (xmin+xmax)/2, cy: (ymin+ymax)/2, zoom: 1 };
    const viewWidth = (xmax-xmin+2*xPad)/scatterView.zoom;
    const viewHeight = (ymax-ymin+2*yPad)/scatterView.zoom;
    const x0 = scatterView.cx-viewWidth/2, x1 = scatterView.cx+viewWidth/2;
    const y0 = scatterView.cy-viewHeight/2, y1 = scatterView.cy+viewHeight/2;
    const left = 45, right = W - 45, top = 30, bottom = H - 35;
    const plotWidth = right - left, plotHeight = bottom - top;
    const xRange = Math.max(x1 - x0, 1e-6), yRange = Math.max(y1 - y0, 1e-6);
    const sx = plotWidth / xRange, sy = plotHeight / yRange;
    const x = v => left + (v - x0) * sx, y = v => bottom - (v - y0) * sy;
    const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'ULCA散布図' });
    scatterFrame = { W, H, left, right, top, bottom, sx, sy, x0, y1 };
    $('scatter-zoom').textContent = `${Math.round(scatterView.zoom*100)}%`;
    const defs = svgEl('defs'), clip = svgEl('clipPath', { id: 'scatter-clip' });
    clip.append(svgEl('rect', { x: left, y: top, width: plotWidth, height: plotHeight }));
    defs.append(clip);
    svg.append(defs);
    const marks = svgEl('g', { 'clip-path': 'url(#scatter-clip)' });
    const tick = value => Math.abs(value) < 1e-12 ? '0' : Number(value.toPrecision(3)).toString();
    for (let i = 0; i < 6; i++) {
        const xv = x0 + xRange * i / 5, yv = y0 + yRange * i / 5;
        svg.append(svgEl('line', { x1: x(xv), x2: x(xv), y1: top, y2: bottom, stroke: '#e6eaf2' }),
            svgEl('text', { x: x(xv), y: bottom + 15, 'text-anchor': 'middle', fill: '#8c97ac', 'font-size': 9 }, tick(xv)),
            svgEl('line', { y1: y(yv), y2: y(yv), x1: left, x2: right, stroke: '#e6eaf2' }),
            svgEl('text', { x: left - 7, y: y(yv) + 3, 'text-anchor': 'end', fill: '#8c97ac', 'font-size': 9 }, tick(yv)));
    }
    points.forEach((p, i) => {
        const circle = svgEl('circle', { cx: x(p[0]), cy: y(p[1]), r: 3.2, fill: colors[latest.labels[i]], opacity: .82 });
        circle.append(svgEl('title', {}, `${classNames[latest.labels[i]]} · sample ${i} · (${p.map(fmt).join(', ')})`));
        marks.append(circle);
    });
    svg.append(marks);
    drawGeometry(svg, W, H, x, y, sx, sy, x0, y1, left, top, bottom, marks);
    svg.append(svgEl('text', { x: W / 2, y: H - 4, 'text-anchor': 'middle', 'font-size': 10, fill: '#8390a7' }, 'ULCA 1 (X)'), svgEl('text', { x: 13, y: H / 2, transform: `rotate(-90 13 ${H / 2})`, 'text-anchor': 'middle', 'font-size': 10, fill: '#8390a7' }, 'ULCA 2 (Y)'));
    $('scatter').classList.remove('empty');
    $('scatter').replaceChildren(svg);
    syncScatterScrollbars(xmin-xPad, xmax+xPad, ymin-yPad, ymax+yPad, viewWidth, viewHeight);
}
function syncScatterScrollbars(baseX0, baseX1, baseY0, baseY1, width, height) {
    const horizontal = $('scatter-scroll-x'), vertical = $('scatter-scroll-y');
    // Keep room around the full dataset so the bars can pan at any zoom level.
    const marginX = Math.max(baseX1-baseX0, width), marginY = Math.max(baseY1-baseY0, height);
    const minX = Math.min(baseX0-marginX, scatterView.cx-width/2);
    const maxX = Math.max(baseX1+marginX, scatterView.cx+width/2);
    const minY = Math.min(baseY0-marginY, scatterView.cy-height/2);
    const maxY = Math.max(baseY1+marginY, scatterView.cy+height/2);
    const pixelX = horizontal.clientWidth/width, pixelY = vertical.clientHeight/height;
    horizontal.firstElementChild.style.width = `${(maxX-minX)*pixelX}px`;
    vertical.firstElementChild.style.height = `${(maxY-minY)*pixelY}px`;
    horizontal.scrollLeft = (scatterView.cx-width/2-minX)*pixelX;
    vertical.scrollTop = (maxY-scatterView.cy-height/2)*pixelY;
    scatterScroll = { minX, maxY, width, height, pixelX, pixelY,
                      x: horizontal.scrollLeft, y: vertical.scrollTop };
}
function scrollScatter(axis) {
    if (!scatterScroll || !scatterView)
        return;
    const s = scatterScroll;
    const value = axis === 'x' ? $('scatter-scroll-x').scrollLeft : $('scatter-scroll-y').scrollTop;
    if (Math.abs(value-s[axis]) < .5)
        return;
    if (axis === 'x')
        scatterView.cx = s.minX+value/s.pixelX+s.width/2;
    else
        scatterView.cy = s.maxY-value/s.pixelY-s.height/2;
    scatter();
}
$('scatter-scroll-x').addEventListener('scroll', () => scrollScatter('x'));
$('scatter-scroll-y').addEventListener('scroll', () => scrollScatter('y'));
function ensureGeometry() {
    if (!latest.target_geometry)
        latest.target_geometry = latest.clusters.map(c => {
            const [[vx, cov], [, vy]] = c.covariance, spread = Math.hypot(vx - vy, 2 * cov);
            const major = Math.max(0, (vx+vy+spread)/2), minor = Math.max(0, (vx+vy-spread)/2);
            const angle = .5*Math.atan2(2*cov, vx-vy), cos = Math.cos(angle), sin = Math.sin(angle);
            const floor = Math.max(major*1e-12, 1e-24);
            // Preserve covariance orientation/aspect; size the contour to contain
            // at least 95% of this cluster's actual samples, without a normality assumption.
            const distances = latest.coordinates.filter((p,i) => latest.labels[i] === c.class).map(p => {
                const dx = p[0]-c.centroid[0], dy = p[1]-c.centroid[1];
                return (cos*dx+sin*dy)**2/Math.max(major,floor)
                     + (-sin*dx+cos*dy)**2/Math.max(minor,floor);
            }).sort((a,b) => a-b);
            const coverageScale = Math.sqrt(Math.max(distances[Math.max(0,Math.ceil(.95*distances.length)-1)] || 0, 1e-12));
            return { label: c.class, center: [...c.centroid], rx: coverageScale*Math.sqrt(Math.max(major,floor)), ry: coverageScale*Math.sqrt(Math.max(minor,floor)), angle: -angle, coverageScale };
        });
    return latest.target_geometry;
}
function drawGeometry(svg, W, H, x, y, sx, sy, x0, y1, left, top, bottom, marks) {
    for (const g of ensureGeometry()) {
        const color = colors[g.label], draw = () => {
            const cx = x(g.center[0]), cy = y(g.center[1]), angle = g.angle * 180 / Math.PI;
            ellipse.setAttribute('cx', 0);
            ellipse.setAttribute('cy', 0);
            ellipse.setAttribute('rx', g.rx);
            ellipse.setAttribute('ry', g.ry);
            ellipse.setAttribute('transform', `translate(${cx} ${cy}) scale(${sx} ${sy}) rotate(${angle})`);
            center.setAttribute('cx', cx);
            center.setAttribute('cy', cy);
        };
        // Compensate for the data-to-screen scale (including unequal X/Y scales)
        // so the outline and dashes stay thin in screen pixels.
        const ellipse = svgEl('ellipse', { fill: color, 'fill-opacity': .03, stroke: color, 'stroke-width': 1.2, 'vector-effect': 'non-scaling-stroke', 'stroke-dasharray': '5 3', cursor: 'nwse-resize', 'pointer-events': 'visibleStroke' }), center = svgEl('circle', { r: 7, fill: 'white', stroke: color, 'stroke-width': 2, cursor: 'move' });
        ellipse.append(svgEl('title', {}, `${classNames[g.label]}の分布楕円（計算結果の点の95%以上を含む）。輪郭をドラッグして分散の目標を変更`));
        center.append(svgEl('title', {}, `${classNames[g.label]}の重心をドラッグして移動`));
        marks.append(ellipse, center);
        draw();
        const bind = (element, mode) => {
            element.addEventListener('pointerdown', event => {
                event.preventDefault();
                event.stopPropagation();
                element.setPointerCapture(event.pointerId);
                const rect = svg.getBoundingClientRect(), point = e => [(e.clientX - rect.left) * W / rect.width, (e.clientY - rect.top) * H / rect.height], original = { center: [...g.center], rx: g.rx, ry: g.ry, angle: g.angle };
                const move = e => {
                    const p = point(e);
                    if (mode === 'move') {
                        g.center = [x0 + (p[0] - left) / sx, y1 - (p[1] - top) / sy];
                    }
                    else {
                        const dx = (p[0] - x(original.center[0])) / sx, dy = (p[1] - y(original.center[1])) / sy;
                        const cos = Math.cos(original.angle), sin = Math.sin(original.angle);
                        const localX = cos * dx + sin * dy, localY = -sin * dx + cos * dy;
                        const scale = Math.max(.05, Math.hypot(localX / Math.max(original.rx, 1e-6), localY / Math.max(original.ry, 1e-6)));
                        g.rx = original.rx * scale;
                        g.ry = original.ry * scale;
                    }
                    draw();
                };
                const end = () => {
                    element.removeEventListener('pointermove', move);
                    element.removeEventListener('pointerup', end);
                    element.removeEventListener('pointercancel', end);
                    if (element.hasPointerCapture(event.pointerId))
                        element.releasePointerCapture(event.pointerId);
                    sendInverseAdjustment(g.label, mode);
                };
                element.addEventListener('pointermove', move);
                element.addEventListener('pointerup', end);
                element.addEventListener('pointercancel', end);
            });
        };
        bind(center, 'move');
        bind(ellipse, 'scale');
    }
}
function adjustmentPayload() {
    return { pattern: latest.intent.pattern, targets: latest.intent.targets, parameters: latest.parameters, reference_coordinates: latest.coordinates, initial_coordinates: latest.initial_coordinates, history: latest.history };
}
function acceptAdjustment(updated) {
    Object.assign(latest, updated);
    scatterView = null;
    delete latest.target_geometry;
    parameters();
    scatter();
    loadings();
    history();
    if (selected)
        distribution();
    $('status').textContent = updated.inverse_adjustment ? `逆調整${updated.inverse_adjustment.success ? '改善' : '改善なし'} · ${updated.inverse_adjustment.iterations}反復 · 目的関数 ${fmt(updated.inverse_adjustment.objective_before)} → ${fmt(updated.inverse_adjustment.objective)}` : '手動パラメータを反映しました';
}
async function applyManualParameters() {
    if (!latest)
        return;
    const params = { w_tg: [0, 1, 2].map(k => Number(document.querySelector(`[data-parameter="w_tg"][data-label="${k}"]`).value)), w_bg: [0, 1, 2].map(k => Number(document.querySelector(`[data-parameter="w_bg"][data-label="${k}"]`).value)), w_bw: [0, 1, 2].map(k => Number(document.querySelector(`[data-parameter="w_bw"][data-label="${k}"]`).value)), alpha: Number(document.querySelector('[data-parameter="alpha"]').value) };
    const body = adjustmentPayload();
    body.parameters = params;
    await sendAdjustment('/api/update', body, '手動変更を計算しています…');
}
async function sendInverseAdjustment(label, interaction) {
    if (!latest)
        return;
    const geometry = ensureGeometry().map(g => ({ label: g.label, center: g.center, area: Math.max(1e-8, g.rx * g.ry / g.coverageScale**2) }));
    await sendAdjustment('/api/inverse', { ...adjustmentPayload(), geometry, updated_label: label, interaction, max_iterations: 50 }, '重心・分散の指定からパラメータを逆調整しています…');
}
async function sendAdjustment(url, body, message) {
    $('status').textContent = message;
    try {
        const response = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        const result = await response.json();
        if (!response.ok)
            throw Error(result.detail || `調整に失敗しました (${response.status})`);
        acceptAdjustment(result);
    }
    catch (error) {
        $('status').textContent = error.message;
    }
}
function pixelHeatmap(container, axis) {
    // Both axes share a symmetric scale so their colors remain comparable.
    const limit = Math.max(...latest.loadings.flat().map(Math.abs), 1e-12);
    const svg = svgEl('svg', {viewBox:'0 0 320 320', role:'group', 'aria-label':`28×28画素の寄与度・${axis===0?'X':'Y'}軸`});
    const label = (x,y,text) => svgEl('text',{x,y,'text-anchor':'middle','font-size':10,fill:'#68748c'},text);
    svg.append(label(170,12,'列'),label(18,24,'行'));
    for (const n of [1,7,14,21,28]) {
        svg.append(label(30+(n-.5)*10,25,String(n)),label(18,30+(n-.5)*10+3,String(n)));
    }
    latest.variables.forEach((name,i)=>{
        const value=latest.loadings[i][axis], intensity=Math.min(1,Math.abs(value)/limit);
        const end=value<0?[232,117,49]:[71,138,198];
        const rgb=end.map(v=>Math.round(255+(v-255)*intensity));
        const cell=svgEl('rect',{x:30+(i%28)*10,y:30+Math.floor(i/28)*10,width:10,height:10,
            fill:`rgb(${rgb.join(',')})`,role:'button',tabindex:0,'aria-label':`${name}の寄与度 ${fmt(value)}。クリックして分布を表示`,
            'aria-pressed':String(selected===name),'data-pixel-index':i,stroke:selected===name?'#20283b':'none','stroke-width':2});
        cell.append(svgEl('title',{},`${name}：${fmt(value)}`));
        cell.onclick=()=>selectFeature(name);
        cell.onkeydown=event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();selectFeature(name);}};
        svg.append(cell);
    });
    const legend=el('div',undefined,'heatmap-legend');
    legend.append(el('span',`−${limit.toFixed(3)}`),el('i'),el('span',`+${limit.toFixed(3)}`));
    container.append(svg,legend,el('div','負 ← 0（白）→ 正','heatmap-note'));
}
function loadings() {
    if (!latest)
        return;
    const container = $('loadings'), axis = Number($('axis').value);
    const isImage = latest.dataset_id === 'mnist069' && latest.variables.length === 784;
    $('loading-order').hidden = isImage;
    container.classList.toggle('heatmap',isImage);
    if(isImage){container.classList.remove('empty');container.replaceChildren();pixelHeatmap(container,axis);return;}
    const order = $('loading-order').value, items = latest.variables.map((name, index) => ({ name, index, value: latest.loadings[index][axis] }));
    if (order === 'magnitude')
        items.sort((a, b) => Math.abs(b.value) - Math.abs(a.value) || a.index - b.index);
    const max = Math.max(...items.map(item => Math.abs(item.value)), 1e-12);
    container.classList.remove('empty');
    container.replaceChildren();
    for (const item of items) {
        const { name, index, value } = item, button = el('button', undefined, 'bar-row' + (selected === name ? ' selected' : ''));
        button.title = `${name}：${fmt(value)}。クリックして分布を表示`;
        button.setAttribute('aria-label', `${name}の寄与度 ${fmt(value)}。クリックして分布を表示`);
        button.setAttribute('aria-pressed', String(selected === name));
        const track = el('span', undefined, 'bar-track'), fill = el('i');
        fill.style.left = '0';
        fill.style.width = `${Math.abs(value) / max * 100}%`;
        fill.style.background = value < 0 ? '#e87531' : '#478ac6';
        track.append(fill);
        button.append(el('span', name, 'bar-name'), track, el('span', value.toFixed(2), 'bar-value'));
        button.onclick = () => selectFeature(name);
        container.append(button);
    }
}
function groups() {
    const ids = data.labels.map((_, i) => i);
    return [0, 1, 2].map(k => ({ name: classNames[k], ids: ids.filter(i => data.labels[i] === k) }));
}
function distribution() {
    if (!latest || !selected || !data)
        return;
    const j = data.variables.indexOf(selected);
    if (j < 0)
        return;
    const rows = groups(), values = data.raw.map(row => row[j]), used = rows.flatMap(g => g.ids.map(i => values[i]));
    let min = Math.min(...used), max = Math.max(...used), pad = (max - min || 1) * .08;
    min -= pad;
    max += pad;
    const W = 300, H = rows.length * 100 + 65, x = v => 24 + (v - min) / (max - min) * (W - 48), svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `${selected}の箱ひげ図・元の値` });
    const quantile = (sorted, q) => {
        const index = (sorted.length - 1) * q, lower = Math.floor(index), fraction = index - lower;
        return sorted[lower] + (sorted[Math.min(lower + 1, sorted.length - 1)] - sorted[lower]) * fraction;
    };
    rows.forEach((g, row) => {
        const cy = 56 + row * 100, sorted = g.ids.map(i => values[i]).sort((a, b) => a - b), q1 = quantile(sorted, .25), median = quantile(sorted, .5), q3 = quantile(sorted, .75), iqr = q3 - q1, lowFence = q1 - 1.5 * iqr, highFence = q3 + 1.5 * iqr, lower = sorted.find(value => value >= lowFence), upper = [...sorted].reverse().find(value => value <= highFence);
        svg.append(svgEl('text', { x: 24, y: cy - 28, 'font-size': 11, fill: '#66718a' }, g.name));
        svg.append(svgEl('line', { x1: x(lower), x2: x(q1), y1: cy, y2: cy, stroke: '#53627a', 'stroke-width': 1.5 }), svgEl('line', { x1: x(q3), x2: x(upper), y1: cy, y2: cy, stroke: '#53627a', 'stroke-width': 1.5 }), svgEl('line', { x1: x(lower), x2: x(lower), y1: cy - 9, y2: cy + 9, stroke: '#53627a', 'stroke-width': 1.5 }), svgEl('line', { x1: x(upper), x2: x(upper), y1: cy - 9, y2: cy + 9, stroke: '#53627a', 'stroke-width': 1.5 }), svgEl('rect', { x: x(q1), y: cy - 13, width: Math.max(1, x(q3) - x(q1)), height: 26, fill: colors[data.labels[g.ids[0]]], 'fill-opacity': .2, stroke: colors[data.labels[g.ids[0]]], 'stroke-width': 1.5 }), svgEl('line', { x1: x(median), x2: x(median), y1: cy - 13, y2: cy + 13, stroke: '#263044', 'stroke-width': 2 }));
        g.ids.forEach(id => {
            if (values[id] < lowFence || values[id] > highFence) {
                const dot = svgEl('circle', { cx: x(values[id]), cy: cy + ((id % 5) - 2) * 2, r: 2.8, fill: colors[data.labels[id]], 'data-outlier-sample': id });
                svg.append(dot);
            }
        });
    });
    const baseline = H - 36;
    svg.append(svgEl('line', { x1: 24, x2: W - 24, y1: baseline, y2: baseline, stroke: '#b2bace' }));
    for (let i = 0; i < 5; i++) {
        const v = min + (max - min) * i / 4;
        svg.append(svgEl('text', { x: x(v), y: baseline + 17, 'text-anchor': 'middle', 'font-size': 9, fill: '#8590a6' }, Number(v.toPrecision(3)).toString()));
    }
    $('distribution').replaceChildren(svg);
    $('feature-name').textContent = selected;
    $('inspection-empty').hidden = true;
    $('inspection-content').hidden = false;
}
async function selectFeature(name) {
    try {
        await dataReady;
        if (!latest || JSON.stringify(data.labels) !== JSON.stringify(latest.labels))
            throw Error('分布データと分析結果の対応を確認できません');
        selected = name;
        loadings();
        document.querySelectorAll('.feature-chip').forEach(b => {
            b.classList.toggle('selected', b.dataset.feature === name);
            b.setAttribute('aria-pressed', String(b.dataset.feature === name));
        });
        distribution();
    }
    catch (e) {
        $('status').textContent = e.message;
    }
}
function addMessage(text, kind, result) {
    if (typeof text !== 'string' || !text.trim())
        return;
    const box = el('div', undefined, 'message ' + kind);
    if (result) {
        const names = [...result.variables].sort((a, b) => b.length - a.length), pattern = new RegExp(names.map(n => n.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|'), 'gi');
        let pos = 0;
        for (const match of text.matchAll(pattern)) {
            box.append(document.createTextNode(text.slice(pos, match.index)));
            const name = names.find(n => n.toLowerCase() === match[0].toLowerCase()), b = el('button', match[0], 'feature-chip');
            b.dataset.feature = name;
            b.onclick = () => {
                if (latest !== result)
                    activate(result);
                selectFeature(name);
            };
            box.append(b);
            pos = match.index + match[0].length;
        }
        box.append(document.createTextNode(text.slice(pos)));
    }
    else
        box.textContent = text;
    $('messages').append(el('div', kind === 'user' ? 'You' : 'Assistant', 'message-label'), box);
    $('messages').scrollTop = $('messages').scrollHeight;
}
function history() {
    const h = $('history');
    h.replaceChildren();
    records.forEach((r, i) => {
        const names = r.class_names || classNames;
        const targets = r.intent.targets.map(k => names[k]);
        const title = r.intent.pattern === 'target' ? `${targets[0]}の特徴把握`
            : r.intent.pattern === 'diversity' ? `${targets[0]}の内部変動`
            : r.intent.pattern === 'pair' ? `${targets.join('と')}の比較`
            : r.intent.name;
        const b = el('button', `${i + 1}. ${title}`, 'history-entry' + (r === latest ? ' active' : ''));
        b.title = r.prompt;
        b.onclick = () => activate(r);
        h.append(b);
    });
}
function activate(r) {
    latest = r;
    scatterView = null;
    selected = null;
    $('comparison').textContent = r.intent.comparison;
    $('inspection-empty').hidden = false;
    $('inspection-content').hidden = true;
    document.querySelectorAll('.feature-chip').forEach(b => b.classList.remove('selected'));
    parameters();
    scatter();
    loadings();
    history();
}
function scatterPosition(event) {
    const rect = $('scatter').getBoundingClientRect();
    return [(event.clientX-rect.left)*scatterFrame.W/rect.width,
            (event.clientY-rect.top)*scatterFrame.H/rect.height];
}
function zoomScatter(factor, point) {
    if (!scatterView || scatterPan)
        return;
    const f = scatterFrame;
    const [px, py] = point || [(f.left+f.right)/2, (f.top+f.bottom)/2];
    const zoom = Math.max(.1, Math.min(100, scatterView.zoom*factor));
    const ratio = scatterView.zoom/zoom;
    const anchorX = f.x0+(px-f.left)/f.sx;
    const anchorY = f.y1-(py-f.top)/f.sy;
    scatterView.cx = anchorX+(scatterView.cx-anchorX)*ratio;
    scatterView.cy = anchorY+(scatterView.cy-anchorY)*ratio;
    scatterView.zoom = zoom;
    scatter();
}
$('scatter-zoom-in').onclick = () => zoomScatter(1.25);
$('scatter-zoom-out').onclick = () => zoomScatter(1/1.25);
$('scatter-reset').onclick = () => { scatterView = null; scatter(); };
$('scatter').addEventListener('wheel', event => {
    if (!scatterFrame || !latest)
        return;
    const point = scatterPosition(event), f = scatterFrame;
    if (point[0] < f.left || point[0] > f.right || point[1] < f.top || point[1] > f.bottom)
        return;
    event.preventDefault();
    const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? f.H : 1;
    const dx = (event.shiftKey ? event.deltaY : event.deltaX)*unit;
    const dy = (event.shiftKey ? 0 : event.deltaY)*unit;
    scatterView.cx += dx/f.sx;
    scatterView.cy -= dy/f.sy;
    scatter();
}, { passive: false });
$('scatter').addEventListener('pointerdown', event => {
    if (!scatterView || event.button !== 0 || scatterPan)
        return;
    const [px, py] = scatterPosition(event), f = scatterFrame;
    if (px < f.left || px > f.right || py < f.top || py > f.bottom)
        return;
    event.preventDefault();
    scatterPan = { id: event.pointerId, px, py, cx: scatterView.cx, cy: scatterView.cy, sx: f.sx, sy: f.sy };
    $('scatter').setPointerCapture(event.pointerId);
    $('scatter').classList.add('panning');
});
$('scatter').addEventListener('pointermove', event => {
    if (!scatterPan || event.pointerId !== scatterPan.id)
        return;
    const [px, py] = scatterPosition(event);
    scatterView.cx = scatterPan.cx-(px-scatterPan.px)/scatterPan.sx;
    scatterView.cy = scatterPan.cy+(py-scatterPan.py)/scatterPan.sy;
    scatter();
});
function endScatterPan(event) {
    if (!scatterPan || event.pointerId !== scatterPan.id)
        return;
    scatterPan = null;
    $('scatter').classList.remove('panning');
    if ($('scatter').hasPointerCapture(event.pointerId))
        $('scatter').releasePointerCapture(event.pointerId);
}
$('scatter').addEventListener('pointerup', endScatterPan);
$('scatter').addEventListener('pointercancel', endScatterPan);
$('scatter').addEventListener('lostpointercapture', endScatterPan);
window.addEventListener('resize', scatter);
$('axis').onchange = loadings;
$('loading-order').onchange = loadings;
document.querySelectorAll('[data-prompt]').forEach(b => b.onclick = () => {
    $('prompt').value = b.dataset.prompt;
    $('prompt').focus();
});
$('analyze').onsubmit = async (event) => {
    event.preventDefault();
    const prompt = $('prompt').value.trim();
    if (!prompt)
        return;
    $('submit').disabled = true;
    addMessage(prompt, 'user');
    $('status').textContent = '分析目的を確認し、ULCAを最適化しています…';
    try {
        const response = await fetch('/api/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ prompt, seed: 17 }) });
        if (!response.ok)
            throw Error((await response.json()).detail || `分析に失敗しました (${response.status})`);
        const r = await response.json();
        if (r.status === 'clarification') {
            addMessage(r.message, 'assistant');
            $('status').textContent = '比較対象を確認してください';
            return;
        }
        records.push(r);
        activate(r);
        addMessage(r.explanation, 'assistant', r);
        if (r.warnings && r.warnings.length)
            addMessage(r.warnings.join('\n'), 'assistant');
        $('status').textContent = '';
        $('prompt').value = '';
    }
    catch (e) {
        $('status').textContent = e.message;
    }
    finally {
        $('submit').disabled = false;
    }
};
$('details-open').onclick = () => {
    const body = $('details-body');
    body.replaceChildren();
    if (latest) {
        const diversity = latest.intent.pattern === 'diversity';
        body.append(el('p', latest.intent.comparison), el('p', `${diversity ? '群内分散比' : 'シルエット'} ${fmt(latest.initial_metrics.score)} → ${fmt(latest.metrics.score)} ／ ${latest.trials}試行 ／ 探索 ${latest.optimization_seconds.toFixed(2)}秒`));
        if (diversity) {
            body.append(el('p', `対象群の分散 ${fmt(latest.metrics.target_variance)} ／ 他群の平均分散 ${fmt(latest.metrics.other_variance)}`));
            body.append(el('p', latest.metrics.feasible ? '対象群の広がりは初期以上です。' : '手動・逆調整後の対象群の広がりは初期を下回っています。'));
        }
        const timing = latest.llm_timing;
        if (timing) {
            const status = {complete:'応答完了',failed:'失敗・定型処理へ切替',skipped:'未使用'};
            const rows = ['classification','explanation'].map((key,i)=>[
                i===0?'プロンプト分類':'結果の説明',
                timing[key].status==='skipped'?'—':`${timing[key].seconds.toFixed(2)}秒`,status[timing[key].status]]);
            const timingTable = el('div');
            body.append(el('h3','LLM応答時間'));
            table(timingTable,['処理','所要時間','状態'],rows);
            body.append(timingTable,el('p',`LLM合計 ${timing.total_seconds.toFixed(2)}秒（通信・待機を含む。ULCA計算時間は除く）`));
            for (const key of ['classification', 'explanation']) {
                if (timing[key].error) body.append(el('p', `${key === 'classification' ? '分類' : '説明'}の失敗理由：${timing[key].error}`));
            }
        } else {
            body.append(el('p',latest.trials===0?'LLM応答時間：未使用（初期表示）':'LLM応答時間：記録なし'));
        }
        const target = el('div');
        table(target, [diversity ? '群内分散（X＋Y）' : '群', '初期', '最適化後'], Object.keys(latest.metrics.groups).map(k => [k, fmt(latest.initial_metrics.groups[k]), fmt(latest.metrics.groups[k])]));
        body.append(target);
        const img = el('img');
        img.src = latest.images.convergence;
        img.alt = '探索の推移';
        body.append(img, el('p', diversity ? '対象群の群内分散 ÷（他の各群の群内分散の平均＋1e-8）を最大化。対象群の分散は初期以上を条件とします。手動・逆調整ではこの条件を強制しません。' : '調整と評価には同じデータを使用。各群の平均シルエットを等重みで集計しています。'));
    }
    else
        body.textContent = '分析後に表示されます。';
    $('details').showModal();
};
$('details-close').onclick = () => $('details').close();
async function initializeWorkspace() {
    $('submit').disabled = true;
    $('status').textContent = '初期設定のULCAを読み込んでいます…';
    try {
        await dataReady;
        const response = await fetch('/api/initial');
        if (!response.ok)
            throw Error('初期表示の取得に失敗しました。再読み込みしてください。');
        const result = await response.json();
        activate(result);
        $('status').textContent = '';
    }
    catch (error) {
        $('status').textContent = error.message;
    }
    finally {
        $('submit').disabled = false;
    }
}
initializeWorkspace();
