"use strict";
(() => {
  const $ = (s) => document.querySelector(s);
  const NS = "http://www.w3.org/2000/svg";
  const width = 345, height = 130;
  const kinds = { input: "Input", model: "ComfyUI generation", python: "Python / FFmpeg", gate: "Qwen / validation", manual: "Human review / layout", hold: "Hold" };
  let data, zoom = 1;
  function svg(tag, attrs, text) {
    const n = document.createElementNS(NS, tag);
    for (const [k,v] of Object.entries(attrs || {})) n.setAttribute(k, v);
    if (text !== undefined) n.textContent = text;
    return n;
  }
  function html(tag, text) { const n = document.createElement(tag); if (text !== undefined) n.textContent = text; return n; }
  function section(title, values, code = false) {
    const fragment = document.createDocumentFragment(); fragment.append(html("h3", title));
    const list = html("ul");
    values.forEach(value => { const li = html("li"); li.append(html(code ? "code" : "span", value)); list.append(li); });
    fragment.append(list); return fragment;
  }
  function choose(id, focus = false) {
    const node = data.nodes.find(n => n.id === id); if (!node) return;
    $("#node-picker").value = id;
    document.querySelectorAll(".graph-node").forEach(n => { const active = n.dataset.id === id; n.classList.toggle("selected", active); n.setAttribute("aria-pressed", String(active)); });
    document.querySelectorAll(".graph-edge").forEach(n => n.classList.toggle("selected", n.dataset.source === id || n.dataset.target === id));
    $("#inspector-kind").textContent = kinds[node.kind]; $("#inspector-title").textContent = node.title; $("#inspector-summary").textContent = node.detail;
    const content = $("#inspector-content"); content.replaceChildren();
    if (node.reused) { const note = html("p", "Reused upstream input / evidence in the accepted Bamburgh run."); note.className = "node-note"; content.append(note); }
    content.append(section("Custom Python / implementation", node.script, true));
    for (const key of node.models) {
      const model = data.models[key]; content.append(section(model.title, model.weights, true)); content.append(html("p", model.note));
      if (model.digest) content.append(section("Pinned local model digest", [model.digest], true));
    }
    const incoming = data.edges.filter(e => e.target === id).map(e => data.nodes.find(n => n.id === e.source).title + (e.label ? " — " + e.label : ""));
    const outgoing = data.edges.filter(e => e.source === id).map(e => data.nodes.find(n => n.id === e.target).title + (e.label ? " — " + e.label : ""));
    if (incoming.length) content.append(section("Receives from", incoming)); if (outgoing.length) content.append(section("Feeds into", outgoing));
    if (node.comfy_nodes.length) {
      if (node.model_loaders?.length) content.append(section("Captured loader settings", node.model_loaders.map(n => `${n.id} · ${n.type}: ${JSON.stringify(n.settings)}`), true));
      const details = html("details"); details.append(html("summary", `Actual ComfyUI nodes (${node.comfy_nodes.length})`));
      details.append(section("Captured graph node IDs / class names", node.comfy_nodes.map(n => `${n.id} · ${n.type}`), true)); content.append(details);
    } else content.append(html("p", "Runs outside the ComfyUI graph; no ComfyUI node IDs for this step."));
    if (focus) document.querySelector(`.graph-node[data-id="${id}"]`).scrollIntoView({ block: "nearest", inline: "center" });
  }
  function setZoom(value) {
    zoom = Math.max(.1, Math.min(2, value)); $("#graph-svg").style.width = `${1610 * zoom}px`; $("#zoom-level").textContent = `${Math.round(zoom*100)}%`;
    $("#zoom-out").disabled = zoom <= .1; $("#zoom-in").disabled = zoom >= 2;
  }
  function draw() {
    const root = $("#graph-svg"), defs = svg("defs");
    for (const [id, color] of [["arrow", "#89977c"], ["hold-arrow", "#b45549"]]) { const marker = svg("marker", { id, viewBox:"0 0 10 10", refX:9, refY:5, markerWidth:7, markerHeight:7, orient:"auto-start-reverse" }); marker.append(svg("path",{d:"M0 0L10 5L0 10Z", fill:color})); defs.append(marker); }
    root.append(defs);
    data.columns.forEach((name,i) => { root.append(svg("text", { x:35+i*395, y:55, class:"graph-column" }, name)); });
    const byId = Object.fromEntries(data.nodes.map(n => [n.id,n]));
    for (const e of data.edges) {
      const a=byId[e.source], b=byId[e.target]; let d, lx, ly;
      if (e.source === "flight" && ["opening","assemble"].includes(e.target)) {
        const mid = e.target === "opening" ? 1195 : 1210, routeY = e.target === "opening" ? 990 : 1008;
        d=`M${a.x+width/2} ${a.y+height} V${routeY} H${mid} V${b.y+height/2} H${b.x}`; lx=mid-180; ly=routeY-7;
      } else if (e.via) {
        const ax=e.via>a.x+width/2?a.x+width:a.x, bx=e.via>b.x+width/2?b.x+width:b.x;
        d=`M${ax} ${a.y+height/2} H${e.via} V${b.y+height/2} H${bx}`; lx=e.via+6; ly=(a.y+b.y)/2+height/2;
      } else if (a.x === b.x) {
        d=`M${a.x+width/2} ${a.y+height} V${b.y}`; lx=a.x+width/2+8; ly=(a.y+height+b.y)/2;
      } else {
        d=`M${a.x+width} ${a.y+height/2} H${b.x}`; lx=(a.x+width+b.x)/2; ly=a.y+height/2-10;
      }
      root.append(svg("path", { d, class:"graph-edge", "data-kind":e.kind, "data-source":e.source, "data-target":e.target, "marker-end":`url(#${e.kind==="hold"?"hold-arrow":"arrow"})` }));
      if (e.label) root.append(svg("text",{ x:lx,y:ly,class:"graph-edge-label","data-kind":e.kind },e.label));
    }
    for (const n of data.nodes) {
      const group=svg("g",{class:`graph-node${n.reused?" reused":""}`,transform:`translate(${n.x},${n.y})`,"data-id":n.id,"data-kind":n.kind,role:"button",tabindex:0,"aria-label":`${n.title}. ${n.summary}. Select for implementation details.`,"aria-pressed":"false"});
      group.append(svg("title",{},n.title+": "+n.summary),svg("rect",{width,height,rx:11,class:"node-box"}),svg("rect",{width:5,height:94,x:0,y:18,rx:2,class:"node-accent"}));
      group.append(svg("text",{x:18,y:25,class:"node-kind"},kinds[n.kind].toUpperCase()));
      group.append(svg("text",{x:18,y:54,class:"node-title"},n.title),svg("text",{x:18,y:78,class:"node-summary"},n.summary),svg("text",{x:18,y:110,class:"node-hint"},n.reused?"Reused input · inspect details →":"Inspect scripts, models and connections →"));
      group.addEventListener("click",()=>choose(n.id)); group.addEventListener("keydown",e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();choose(n.id);}}); root.append(group);
      const option=html("option",n.title); option.value=n.id; $("#node-picker").append(option);
    }
    const textList=html("ul"); for(const e of data.edges) textList.append(html("li",`${byId[e.source].title} → ${byId[e.target].title}${e.label?" ("+e.label+")":""}`)); $("#graph-text-content").append(textList);
    $("#node-picker").addEventListener("change",e=>choose(e.target.value,true));
    $("#zoom-in").onclick=()=>setZoom(zoom+.15); $("#zoom-out").onclick=()=>setZoom(zoom-.15); $("#zoom-reset").onclick=()=>setZoom(1); $("#zoom-fit").onclick=()=>setZoom(Math.min($("#workflow-graph").clientWidth/1610, $("#workflow-graph").clientHeight/1585));
    choose("arrivalgate");
  }
  fetch("/static/workflow-graph.json", {cache: "no-store"}).then(r=>{if(!r.ok)throw Error("Graph data unavailable");return r.json();}).then(value=>{data=value;draw();}).catch(()=>{const error=$("#graph-error");error.textContent="The workflow graph could not load. Reload this page or download the graph data above.";error.hidden=false;});
})();
