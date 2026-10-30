"use strict";

const $ = (s) => document.querySelector(s);
const state = { config: null, destination: null, subject: null, references: [], base: "", framing: "medium", saving: false, dirty: false, processing: 0, referenceBusy: false,
  automatic: [], referenceSet: "", photoLoading: false, photoRequest: null, photoMessage: "", creating: false };
let config = null;
let passWait = null;
let searchVersion = 0;
let routeVersion = 0;
let portraitVersion = 0;
let referenceVersion = 0;
let discoveryVersion = 0;

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

async function api(path, options = {}, binary = false) {
  const r = await fetch(path, { credentials: "same-origin", ...options });
  if (r.status === 401 && config?.passcode_required) {
    await askPasscode();
    return api(path, options, binary);
  }
  if (!r.ok) {
    let message = "Something went wrong. Please try again.";
    try { const data = await r.json(); if (typeof data.detail === "string") message = data.detail; } catch (_) { /* no JSON */ }
    const error = new Error(message); error.status = r.status; throw error;
  }
  return r.status === 204 ? null : binary ? r.blob() : r.json();
}

function askPasscode() {
  if (passWait) return passWait;
  passWait = new Promise((resolve, reject) => {
    const dialog = $("#pass");
    $("#pass-error").hidden = true;
    const cancel = () => { dialog.close(); reject(new Error("Enter the passcode to open this studio.")); };
    dialog.oncancel = (e) => { e.preventDefault(); cancel(); };
    $("#pass-cancel").onclick = cancel;
    $("#pass-form").onsubmit = async (e) => {
      e.preventDefault();
      try {
        const response = await fetch("/api/login", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ passcode: $("#pass-input").value }) });
        if (response.ok) { dialog.close(); resolve(); } else { $("#pass-error").hidden = false; }
      } catch (_) { $("#pass-error").textContent = "Couldn't connect. Please try again."; $("#pass-error").hidden = false; }
    };
    dialog.showModal();
  }).finally(() => { passWait = null; });
  return passWait;
}

function dirty() {
  state.dirty = true;
  $("#save-state").textContent = "Unsaved changes";
  $("#saved").hidden = true;
  $("#form-error").hidden = true;
  updateSave();
}

function showFormError(message) {
  $("#form-error").textContent = message;
  $("#form-error").hidden = false;
}

function compatible() { return !["twirl", "arms-open"].includes($("#ending").value) || ["full", "three-quarter"].includes(state.framing); }

function updateSave() {
  $("#setup-fields").disabled = state.saving || state.creating;
  $("#save").disabled = state.saving || state.creating || state.processing > 0 || !state.destination || !state.subject || !compatible();
  if (!state.saving) $("#save span").textContent = state.processing > 0 ? "Preparing photos…" : "Save setup";
  $("#create").disabled = !state.config?.render_available || state.creating || state.saving || state.processing > 0 || !state.destination || !state.subject || !compatible();
  $("#create").textContent = state.creating ? "Starting…" : "Create film";
  $("#availability").textContent = state.config?.render_available
    ? "Your setup is saved before generation. You can leave and return using the job link."
    : "Generation worker offline. You can still save your setup and return later.";
}

function release(item) { if (item?.file && item.url) URL.revokeObjectURL(item.url); }

function radioKeys(container, onSelect) {
  container.addEventListener("keydown", (event) => {
    if (!["ArrowRight", "ArrowLeft", "ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) return;
    const items = [...container.querySelectorAll('[role="radio"]')];
    const at = items.indexOf(document.activeElement);
    if (at < 0) return;
    event.preventDefault();
    const index = event.key === "Home" ? 0 : event.key === "End" ? items.length - 1 : (at + (["ArrowRight", "ArrowDown"].includes(event.key) ? 1 : -1) + items.length) % items.length;
    onSelect(items[index].dataset.id);
    items[index].focus();
  });
}

function setupDestinations() {
  const select = $("#destination");
  const custom = el("option", "", "Custom place…"); custom.value = ""; select.append(custom);
  const groups = new Map();
  for (const place of state.config.destinations) {
    if (!groups.has(place.group)) {
      const group = el("optgroup"); group.label = place.group;
      select.append(group); groups.set(place.group, group);
    }
    const option = el("option", "", `${place.title} · ${place.country}`); option.value = place.id;
    groups.get(place.group).append(option);
  }
  select.onchange = () => {
    const place = state.config.destinations.find((p) => p.id === select.value);
    choosePlace(place || null);
    if (!place) $("#place-search").focus();
  };
}

function choosePlace(place, markDirty = true, preserveInput = false, discover = true) {
  place = state.config.destinations.find((p) => p.id === place?.id) || place;
  const same = state.destination && JSON.stringify(state.destination) === JSON.stringify(place);
  if (!same) {
    referenceVersion += 1; discoveryVersion += 1;
    state.references.forEach(release); state.references = []; $("#reference-input").value = "";
    state.automatic = []; state.referenceSet = ""; state.photoMessage = ""; state.photoLoading = false; state.photoRequest = null;
  }
  state.destination = place;
  searchVersion += 1;
  $("#search-panel").hidden = true;
  if (!preserveInput) $("#place-search").value = place?.id === "custom" ? place.title : "";
  $("#destination").value = place?.id === "custom" ? "" : place?.id || "";
  $("#chosen").textContent = place ? [place.title, place.region || place.country].filter(Boolean).join(" · ") : "Enter a custom place name, or choose a landmark.";
  $("#reference-details").open = true;
  renderReferences();
  if (markDirty) dirty();
  updateSave();
  if (!same && discover && place?.latitude !== undefined) findPhotos();
}

function findPhotos(retry = false) {
  if (!state.destination) return Promise.resolve();
  if (state.photoLoading) return state.photoRequest;
  const version = ++discoveryVersion;
  state.photoLoading = true; state.photoMessage = "Finding location photos on Wikimedia Commons…"; renderReferences();
  state.photoRequest = (async () => {
    try {
      const data = await api("/api/studio/location-photos", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ destination: state.destination, retry }) });
      if (version !== discoveryVersion) return;
      state.referenceSet = data.id; state.automatic = data.photos; state.photoMessage = data.message;
    } catch (error) {
      if (version !== discoveryVersion) return;
      state.photoMessage = "Couldn't fetch location photos just now. Try again, or add your own below.";
    } finally {
      if (version === discoveryVersion) { state.photoLoading = false; renderReferences(); }
    }
  })();
  return state.photoRequest;
}

function selectExample(place, play = false) {
  const video = $("#example-video");
  video.pause();
  video.poster = place.poster;
  $("#example-select").value = place.id;
  $("#preview-method").textContent = place.method || "";
  $("#preview-title").textContent = place.title;
  $("#preview-region").textContent = [place.region, place.country].filter(Boolean).join(", ");
  $("#preview-duration").textContent = place.duration;
  $("#example-missing").hidden = !!place.example_url;
  $("#play-example").hidden = !place.example_url || play;
  if (place.example_url) {
    if (video.getAttribute("src") !== place.example_url) { video.src = place.example_url; video.load(); }
    video.controls = play;
    if (play) video.play().catch(() => { $("#play-example").hidden = false; });
  } else { video.removeAttribute("src"); video.load(); video.controls = false; }
}

function renderReferences() {
  $("#reference-count").textContent = state.photoLoading ? "Finding…" : String(state.automatic.length + state.references.length);
  $("#reference-hint").textContent = !state.destination ? "Choose a place and we'll find its reference photos."
    : state.photoMessage || "Search for your place to find photos automatically, or use Find location photos below. We'll also look when you save.";
  $("#find-photos").disabled = !state.destination || state.photoLoading;
  $("#find-photos").textContent = state.photoLoading ? "Finding photos…" : state.referenceSet ? "Find photos again" : "Find location photos";
  const automatic = $("#automatic-previews"); automatic.replaceChildren();
  state.automatic.forEach((item, index) => {
    const frame = el("div", "reference-image");
    const image = el("img"); image.src = item.image_url; image.alt = item.title;
    const button = el("button", "", "×"); button.type = "button"; button.setAttribute("aria-label", `Remove found reference ${index + 1}`);
    button.onclick = () => { state.automatic.splice(index, 1); renderReferences(); dirty(); };
    const credit = el("small", "reference-credit");
    const source = el("a", "", "Commons"); source.href = item.url; source.target = "_blank"; source.rel = "noreferrer";
    const license = el("a", "", item.license); license.href = item.license_url; license.target = "_blank"; license.rel = "noreferrer";
    credit.append(`${item.artist} · `, source, " · ", license);
    frame.append(image, button, credit); automatic.append(frame);
  });
  const box = $("#reference-previews"); box.replaceChildren();
  state.references.forEach((item, index) => {
    const frame = el("div", "reference-image");
    const image = el("img"); image.src = item.url; image.alt = `Your location reference ${index + 1}`;
    const button = el("button", "", "×"); button.type = "button"; button.setAttribute("aria-label", `Remove location reference ${index + 1}`);
    button.onclick = () => { release(item); state.references.splice(index, 1); renderReferences(); dirty(); };
    frame.append(image, button); box.append(frame);
  });
  $("#reference-input").disabled = state.referenceBusy || state.references.length >= state.config.max_references;
  $("#reference-status").hidden = !state.referenceBusy;
}

async function validFile(file) {
  const heif = /\.(heic|heif|hif)$/i.test(file.name) || /^image\/(heic|heif)(-sequence)?$/i.test(file.type);
  const supportedType = ["image/jpeg", "image/png", "image/webp"].includes(file.type);
  if (!heif && !supportedType && !/\.(jpe?g|png|webp)$/i.test(file.name)) throw new Error("Please choose a JPG, PNG, WebP or HEIC/HEIF photo.");
  if (file.size > state.config.max_upload_mb * 1024 * 1024) throw new Error(`Each photo must be under ${state.config.max_upload_mb} MB.`);
  // Chrome cannot reliably display a HEIF blob. Preview the server's sanitized
  // JPEG/PNG; keep the original File for the normal saved-upload validation.
  let preview = file;
  if (heif) {
    const form = new FormData(); form.append("file", file);
    preview = await api("/api/studio/images/normalize", { method: "POST", body: form }, true);
  }
  return { file, name: file.name, url: URL.createObjectURL(preview) };
}

function renderSubject() {
  const image = $("#photo-preview");
  image.hidden = !state.subject;
  $(".upload-symbol").hidden = !!state.subject;
  $("#photo-remove").hidden = !state.subject;
  $("#photo-label").textContent = state.subject ? "Photo added" : "Choose a photo";
  $("#photo-note").textContent = state.subject ? "Choose another photo to replace" : "JPG, PNG, WebP or HEIC/HEIF";
  if (state.subject) image.src = state.subject.url; else image.removeAttribute("src");
  updateSave();
}

function setupUploads() {
  const setPhoto = async (file) => {
    if (!file || state.saving) return;
    const version = ++portraitVersion;
    state.processing += 1; updateSave();
    $("#photo-note").textContent = "Preparing photo preview…";
    try {
      const next = await validFile(file);
      if (version !== portraitVersion) { release(next); return; }
      release(state.subject); state.subject = next; renderSubject(); dirty();
    }
    catch (error) { if (version === portraitVersion) showFormError(error.message); }
    finally {
      state.processing -= 1;
      if (version === portraitVersion) { $("#photo").value = ""; renderSubject(); }
      updateSave();
    }
  };
  $("#photo").onchange = (event) => setPhoto(event.target.files[0]);
  $("#photo-remove").onclick = () => { portraitVersion += 1; release(state.subject); state.subject = null; renderSubject(); dirty(); };
  const drop = $("#drop");
  for (const name of ["dragenter", "dragover"]) drop.addEventListener(name, (event) => { event.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (event) => { event.preventDefault(); drop.classList.remove("over"); if (event.dataTransfer.files.length > 1) { showFormError("Choose one portrait for the person."); return; } setPhoto(event.dataTransfer.files[0]); });
  $("#reference-input").onchange = async (event) => {
    const files = [...event.target.files];
    if (state.references.length + files.length > state.config.max_references) { showFormError("Add up to four location photos."); event.target.value = ""; return; }
    const added = [];
    const version = ++referenceVersion;
    state.referenceBusy = true; state.processing += 1; renderReferences(); updateSave();
    try {
      for (const file of files) added.push(await validFile(file));
      if (version !== referenceVersion) { added.forEach(release); return; }
      state.references.push(...added); renderReferences(); dirty();
    }
    catch (error) { added.forEach(release); if (version === referenceVersion) showFormError(error.message); }
    finally { event.target.value = ""; state.referenceBusy = false; state.processing -= 1; renderReferences(); updateSave(); }
  };
}

function setupEndings() {
  for (const ending of state.config.endings) {
    const option = el("option", "", ending.label + (ending.trial ? " (trial)" : "")); option.value = ending.id;
    $("#ending").append(option);
  }
  $("#ending").onchange = () => { renderEnding(); dirty(); };
  const box = $("#framings");
  for (const framing of state.config.framings) {
    const button = el("button", "framing"); button.type = "button"; button.dataset.id = framing.id;
    button.setAttribute("role", "radio"); button.setAttribute("aria-label", `${framing.label}: ${framing.description}`);
    const height = { full: 70, "three-quarter": 53, medium: 38, close: 23 }[framing.id];
    // A simple crop diagram, not a pose or scene reference sent to the model.
    button.innerHTML = `<svg viewBox="0 0 40 ${height}" aria-hidden="true"><circle cx="20" cy="9" r="6" fill="currentColor"/><path d="M12 20 Q20 15 28 20 L30 39 H25 L25 65 H21 L20 43 L19 65 H15 L15 39 H10Z" fill="currentColor"/></svg>`;
    button.append(el("span", "", framing.label));
    button.onclick = () => chooseFraming(framing.id);
    box.append(button);
  }
  radioKeys(box, chooseFraming);
  chooseFraming("medium", false);
  renderEnding();
}

function chooseFraming(value, markDirty = true) {
  state.framing = value;
  for (const button of $("#framings").children) {
    const on = button.dataset.id === value;
    button.setAttribute("aria-checked", String(on)); button.tabIndex = on ? 0 : -1;
  }
  renderEnding();
  if (markDirty) dirty();
}

function renderEnding() {
  const ending = state.config.endings.find((e) => e.id === $("#ending").value);
  $("#ending-hint").textContent = ending ? ending.description + (ending.trial ? " Trial: fails more often." : "") : "";
  $("#framing-hint").textContent = !compatible() ? "Choose full body or three-quarter to leave room for this movement."
    : ending?.id === "twirl" ? "A twirl needs a longer ending, with room to turn." : "";
  updateSave();
}

function setupSearch() {
  const input = $("#place-search");
  const useName = $("#use-name");
  const manual = () => { choosePlace({ id: "custom", title: input.value.trim(), region: "", country: "" }); findPhotos(); };
  useName.onclick = manual;
  input.oninput = () => {
    searchVersion += 1;
    $("#search-results").replaceChildren();
    const q = input.value.trim();
    choosePlace(q.length >= 3 ? { id: "custom", title: q, region: "", country: "" } : null, true, true);
    $("#search-panel").hidden = q.length < 3;
    $("#search-status").textContent = "Search to choose the right place and find location photos.";
    useName.textContent = `Find photos for “${q}”`;
    useName.hidden = q.length < 3;
  };
  const search = async () => {
    const q = input.value.trim();
    if (q.length < 3) return;
    const version = ++searchVersion;
    $("#search-panel").hidden = false;
    $("#search-status").textContent = "Searching places…";
    $("#search-results").replaceChildren();
    try {
      const data = await api("/api/studio/search?q=" + encodeURIComponent(q));
      if (version !== searchVersion) return;
      $("#search-status").textContent = data.results.length ? "Choose a place:" : "No matches. You can use the name and add location photos.";
      for (const result of data.results) {
        const li = el("li"); const button = el("button", "", result.label || result.name); button.type = "button";
        button.onclick = () => choosePlace({ id: "custom", title: result.name || q, region: result.state || "", country: "", latitude: result.latitude, longitude: result.longitude });
        li.append(button); $("#search-results").append(li);
      }
    } catch (error) { if (version === searchVersion) $("#search-status").textContent = error.message; }
  };
  $("#search-go").onclick = search;
  input.onkeydown = (event) => { if (event.key === "Enter") { event.preventDefault(); search(); } };
}

function showSaved(data) {
  $("#save-state").textContent = "Saved";
  $("#saved").hidden = false;
  $("#saved-message").textContent = data.preparation === "needs_location_photos"
    ? "Setup saved. Add at least two location photos before the scene can be reviewed."
    : "Setup saved. Your photos and choices will guide the film and its reference checks.";
  $("#saved-link").value = `${location.origin}/#setup=${data.id}`;
  $("#copy-link").textContent = "Copy link";
  let latest = $("#saved-job");
  if (!latest) { latest = el("p", "fine"); latest.id = "saved-job"; $("#saved").append(latest); }
  latest.replaceChildren();
  if (data.latest_job) {
    const link = el("a", "", data.latest_job.status === "done" ? "View your film" : ["queued", "running"].includes(data.latest_job.status) ? "View generation progress" : "View the last take");
    link.href = "#job=" + data.latest_job.id; latest.append(link);
  }
}

function restore(data) {
  portraitVersion += 1; referenceVersion += 1;
  release(state.subject); state.references.forEach(release);
  state.base = data.id;
  state.subject = { name: data.subject.name, url: data.subject.url };
  state.references = [];
  const known = state.config.destinations.find((p) => p.id === data.destination.id);
  choosePlace(known || data.destination, false, false, false);
  discoveryVersion += 1; state.photoLoading = false; state.photoRequest = null;
  state.referenceSet = data.reference_set || "";
  state.automatic = (data.default_references || []).filter((p) => p.image_url);
  state.photoMessage = state.referenceSet ? (state.automatic.length ? "Saved location references. Check that they show the setting you want."
    : "No automatically found photos in this setup. Try again, or add your own.") : "";
  state.references = data.references.map((item) => ({ name: item.name, url: item.url }));
  $("#ending").value = data.ending; $("#wearing").value = data.wearing;
  chooseFraming(data.framing, false);
  renderReferences(); renderSubject();
  state.dirty = false;
  state.config.render_available = data.render_available;
  updateSave();
  showSaved(data);
  if (!state.referenceSet) findPhotos();
}

async function save(event) {
  event.preventDefault();
  if (!state.subject || !state.destination || state.saving || state.processing > 0 || !compatible()) return;
  state.saving = true;
  $("#setup-fields").disabled = true;
  $("#save span").textContent = "Saving…";
  $("#form-error").hidden = true;
  if (state.photoLoading) await state.photoRequest;
  else if (!state.referenceSet) await findPhotos();
  const form = new FormData();
  form.append("destination_json", JSON.stringify(state.destination));
  form.append("ending", $("#ending").value); form.append("framing", state.framing); form.append("wearing", $("#wearing").value);
  form.append("base", state.base);
  form.append("reference_set", state.referenceSet);
  if (state.referenceSet) form.append("automatic", JSON.stringify(state.automatic.map((r) => r.sha256)));
  form.append("retained", JSON.stringify(state.references.filter((r) => !r.file).map((r) => r.name)));
  if (state.subject.file) form.append("subject", state.subject.file);
  for (const reference of state.references) if (reference.file) form.append("references", reference.file);
  try {
    const data = await api("/api/studio/setups", { method: "POST", body: form });
    restore(data);
    location.hash = "setup=" + data.id;
    return data;
  } catch (error) { showFormError(error.message); }
  finally { state.saving = false; $("#setup-fields").disabled = false; $("#save span").textContent = "Save setup"; updateSave(); }
}

async function createVideo() {
  if (state.creating || state.saving || state.processing || !state.subject || !state.destination || !compatible()) return;
  state.creating = true; updateSave();
  try {
    if (state.dirty || !state.base) {
      const data = await save({ preventDefault() {} });
      if (!data) return;
    }
    const job = await api(`/api/studio/setups/${state.base}/render`, { method: "POST" });
    state.dirty = false;
    location.hash = "job=" + job.id;
  } catch (error) { showFormError(error.message); }
  finally { state.creating = false; updateSave(); }
}

async function setupExamples() {
  const fallback = state.config.landmarks.filter(p => p.example_url);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 8000);
  try {
    const options = { cache: "no-store", signal: controller.signal };
    const [catalog, gallery] = await Promise.all([api("/static/reviewed-examples.json", options), api("/api/gallery", options)]);
    const installed = new Set(gallery.items.map(item => item.url));
    state.examples = catalog.filter(item => installed.has(item.example_url));
  } catch (_) { state.examples = []; }
  finally { clearTimeout(timeout); }
  if (!state.examples.length) state.examples = fallback;
  const select = $("#example-select"); select.replaceChildren();
  for (const place of state.examples) {
    const option = el("option", "", place.title); option.value = place.id; select.append(option);
  }
  select.disabled = !state.examples.length;
  if (!state.examples.length) select.append(el("option", "", "No example films installed"));
  select.onchange = () => {
    const place = state.examples.find(item => item.id === select.value);
    if (place) selectExample(place);
  };
}

function renderGallery() {
  const items = state.examples;
  $("#gallery-empty").hidden = items.length > 0;
  for (const place of items) {
    const button = el("button", "film-card"); button.type = "button"; button.setAttribute("aria-label", `Play example: ${place.title}`);
    const poster = el("div", "film-poster"); const image = el("img"); image.src = place.poster; image.alt = ""; image.loading = "lazy";
    poster.append(image, el("span", "film-play", "▶"));
    const caption = el("div", "film-card-title"); caption.append(el("span", "", place.title), el("small", "", place.duration));
    button.append(poster, caption);
    button.onclick = async () => {
      if ($("#make").hidden) { location.hash = "make"; await route(); }
      selectExample(place, true); $("#screen").scrollIntoView({ behavior: "smooth", block: "center" });
    };
    $("#gallery-items").append(button);
  }
  const credits = [...state.config.landmarks.map(place => ({ ...place.credit, title: place.title })),
    ...state.examples.flatMap(place => place.credits || [])];
  const credited = new Set();
  for (const credit of credits) {
    if (credited.has(credit.url)) continue;
    credited.add(credit.url);
    const p = el("p"); const link = el("a", "", credit.title); link.href = credit.url; link.target = "_blank"; link.rel = "noreferrer";
    const license = el("a", "", credit.license); license.href = credit.license_url; license.target = "_blank"; license.rel = "noreferrer";
    p.append(link, ` — ${credit.artist}, `, license, "."); $("#photo-credits").append(p);
  }
}

const STAGE_LABELS = { layout: "Sketching the regional layout", texture: "Creating the overhead texture", expand: "Widening the overhead view", project: "Wrapping onto the curved Earth", orbit: "Finishing the view from orbit", opening: "Opening from orbit", source: "Loading original film", refine: "Refining photographic detail", assemble: "Joining refined sections", verify: "Checking original timing", references: "Landmark photos, landing photo and portrait", arrival: "Creating the landing scene", arrival_check: "Checking ground contact", portrait: "Creating your moving portrait", flight: "Flight down through the cloud", pacing: "Timing the descent", review: "Qwen3.5 review of the finished film", terrain: "Preparing the terrain", stills: "Creating scene images", opener: "Orbit", dive: "Descent", cloud: "Cloud passage", swoop: "Final approach", body: "Flight movement", end: "Living portrait", face: "Portrait refinement", sound: "Finishing the film" };
// Typical minutes per Create stage on an RTX 4090: the median of four generic-orbit builds (29 Oct, 22-24 min). The bar fills by elapsed time,
// held inside the band of the stage that is actually running, so it never runs ahead of or behind the real stage.
const STAGE_MINUTES = { references: 6.5, flight: 12.5, opening: 4.5, review: 0.5 };
const STATE_LABELS = { pending: "", running: "working…", done: "done", cached: "reused", failed: "failed here", stopped: "stopped", held: "needs review" };
const PREVIEW_LABELS = { arrival: "Landing scene", approach: "Final approach", portrait: "Portrait", orbit: "From orbit", cloud: "Into the cloud", end: "Arrival" };

let canonicalEvidence = "";
let canonicalCheckTime = 0;
async function showCanonicalCheck(job) {
  const canonical = job.workflow === "photographic-grounded-v1";
  $("#canonical-check").hidden = !canonical;
  if (!canonical || (Date.now() - canonicalCheckTime < 4000 && !["done", "held", "failed", "cancelled"].includes(job.status))) return;
  canonicalCheckTime = Date.now();
  try {
    const data = await api(`/api/jobs/${job.id}/checkpoint`);
    if (location.hash !== `#job=${job.id}`) return;
    const check = data.checkpoint;
    $("#canonical-check-status").textContent = !check ? (["done", "held", "failed", "cancelled"].includes(job.status) ? "See the saved review report for this take’s checks and limitations." : "Preparing photographic inputs. Checks run automatically.") : `${check.title}: ${data.decision ? (data.decision.continue ? "passed" : "held") + ". " + data.decision.reason : "automatic check in progress…"}`;
    if (check && canonicalEvidence !== check.evidence_sha256) {
      canonicalEvidence = check.evidence_sha256;
      const assets = $("#canonical-check-assets"); assets.replaceChildren();
      for (const asset of check.assets) {
        const media = el(asset.name.endsWith(".mp4") ? "video" : "img");
        media.src = asset.url; media.alt = check.title;
        if (asset.name.endsWith(".mp4")) { media.controls = true; media.preload = "none"; media.playsInline = true; }
        assets.append(media);
      }
    }
  } catch (_) { $("#canonical-check-status").textContent = "Stage evidence is temporarily unavailable."; }
}

let enhancementTimer = null;
let enhancementParent = null;
let enhancementBusy = false;
function stopEnhancement() {
  clearTimeout(enhancementTimer); enhancementTimer = null; enhancementParent = null;
  $("#enhancement").hidden = true;
}
async function refreshEnhancement(id) {
  if (enhancementParent !== id) return;
  try {
    const data = await api(`/api/jobs/${id}/enhancement`);
    if (enhancementParent !== id) return;
    const job = data.job;
    const active = job && ["queued", "running"].includes(job.status);
    $("#enhancement").hidden = !data.eligible || (!data.available && !job);
    const complete = job?.status === "done";
    $("#enhance-button").hidden = Boolean(active || complete);
    $("#enhance-button").disabled = !data.available || enhancementBusy;
    $("#enhance-button").textContent = job ? "Try 4K again" : "Refine to 4K";
    $("#enhance-status").textContent = complete ? "Enhanced film ready. Check the face and motion before sharing." : active ? (job.message || "Enhancement queued. You can leave this page and return to this film.") : job ? `${job.status === "cancelled" ? "Enhancement cancelled." : "Enhancement did not complete."} Your original is unchanged.${data.available ? "" : " The enhancement worker is offline."}` : data.available ? "" : "The enhancement worker is offline. Try again when it is available.";
    const links = $("#enhance-links"); links.replaceChildren();
    if (job) {
      const view = el("a", "ghost", complete ? "View enhanced film" : "View enhancement");
      view.href = `#job=${job.id}`; links.append(view);
      if (job.video) { const download = el("a", "primary", "Download enhanced MP4"); download.href = job.video + "?download=1"; download.download = ""; links.append(download); }
    }
    $("#enhance-button").onclick = async () => {
      enhancementBusy = true; $("#enhance-button").disabled = true;
      try { await api(`/api/jobs/${id}/enhance`, {method: "POST"}); }
      catch (error) { if (enhancementParent === id) $("#enhance-status").textContent = error.message; }
      finally { enhancementBusy = false; if (enhancementParent === id) { clearTimeout(enhancementTimer); enhancementTimer = setTimeout(() => refreshEnhancement(id), 1500); } }
    };
  } catch (error) {
    if (enhancementParent === id) $("#enhance-status").textContent = error.message;
  }
  if (enhancementParent === id) { clearTimeout(enhancementTimer); enhancementTimer = setTimeout(() => refreshEnhancement(id), 15000); }
}

let source = null;
let pollTimer = null;
function stopFollowing() { if (source) { source.close(); source = null; } if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; } if (clockTimer) { clearInterval(clockTimer); clockTimer = null; } }
function showJob(id) {
  stopEnhancement();
  canonicalEvidence = ""; canonicalCheckTime = 0; $("#canonical-check").hidden = true; $("#canonical-check-assets").replaceChildren();
  reviewLoadedJob = "";
  $("#make").hidden = true; $("#job").hidden = false;
  $("#result").hidden = true; $("#previews").replaceChildren();
  $("#review-note").hidden = true; $("#review-findings").replaceChildren(); $("#review-findings").hidden = true; $("#job-setup-link").replaceChildren(); $("#job-error-details").hidden = true; $("#stages").replaceChildren(); $("#job-clock").hidden = true; $("#monitor").hidden = true; monitorJob = null;
  $("#job-title").textContent = "Your Earthfall"; $("#job-status").textContent = "Opening the job…";
  $("#job-link").replaceChildren(); const link = el("a", "", "Link to this job"); link.href = location.href; $("#job-link").append(link);
  $("#cancel").disabled = false;
  $("#cancel").onclick = async () => { $("#cancel").disabled = true; try { await api(`/api/jobs/${id}/cancel`, { method: "POST" }); } catch (error) { $("#job-status").textContent = error.message; $("#cancel").disabled = false; } };
  stopFollowing();
  if (window.EventSource) { source = new EventSource(`/api/jobs/${id}/events`); source.onmessage = (event) => renderJob(JSON.parse(event.data)); source.onerror = () => { stopFollowing(); poll(id); }; } else poll(id);
}
function poll(id) {
  const tick = async () => {
    try { const job = await api(`/api/jobs/${id}`); renderJob(job); if (["done", "failed", "cancelled", "held"].includes(job.status)) return; }
    catch (error) { $("#job-status").textContent = error.message; if (error.status === 404) return; }
    pollTimer = setTimeout(tick, 3000);
  }; tick();
}
let reviewLoadedJob = "";
async function showReviewFindings(job) {
  if (reviewLoadedJob === job.id || !job.review_url || !["held", "done"].includes(job.status)) return;
  reviewLoadedJob = job.id;
  try {
    const review = await api(job.review_url);
    if (location.hash !== "#job=" + job.id) return;
    const labels = { foot_contact: "Feet on the ground", support_context: "Standing surface", surface_continuity: "Scene continuity", appearance: "Appearance", scene_match: "Location", framing: "Final framing", ending: "Pose and movement", orbit: "View from orbit", high: "High aerial view", cloud_exit: "View below the clouds", camera_movement: "Camera movement" };
    const checks = { ...(review.arrival_review?.answer?.checks || {}), ...(review.final_review?.answer?.checks || {}), ...(review.ending_review || {}) };
    if (review.orbital_scale_review) checks.orbit = review.orbital_scale_review;
    for (const [name, finding] of Object.entries(review.scale_reference_reviews || {})) {
      if (finding.status !== "pass") checks[name] = finding;
    }
    const exitFinding = review.cloud_exit_reviews?.find(finding => finding.status !== "pass");
    if (exitFinding) checks.cloud_exit = exitFinding;
    if (review.motion_review) checks.camera_movement = review.motion_review;
    const box = $("#review-findings"); box.replaceChildren();
    for (const [name, finding] of Object.entries(checks)) {
      if (finding.status === "pass") continue;
      const detail = String(finding.reason || finding.evidence || "Needs a closer look.");
      box.append(el("li", "", `${labels[name] || name}: ${detail.length > 240 ? detail.slice(0, 237) + "…" : detail}`));
    }
    box.hidden = !box.children.length;
  } catch (_) { /* The general job message remains available if the report cannot be loaded. */ }
}

let clockTimer = null, monitorJob = null, monitorOrder = [];
function clock(seconds) { const s = Math.max(0, Math.floor(seconds)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; }
function monitorTick() {
  const job = monitorJob; if (!job) return;
  const running = job.status === "running";
  const elapsed = job.started ? ((running ? Date.now() / 1000 : job.finished || Date.now() / 1000) - job.started) : 0;
  $("#job-clock").hidden = !job.started; $("#job-clock").textContent = clock(elapsed);
  const weights = monitorOrder.map(name => STAGE_MINUTES[name] || 0);
  const known = weights.every(w => w > 0), total = known ? weights.reduce((a, b) => a + b, 0) : monitorOrder.length;
  let before = 0, current = -1;
  monitorOrder.forEach((name, i) => { const w = known ? weights[i] : 1; const v = job.stages[name]; if (v === "done" || v === "cached") before += w; else if (v === "running" && current < 0) current = i; });
  let fraction = before / total;
  if (job.status === "done") fraction = 1;
  else if (running && current >= 0) {
    const band = (known ? weights[current] : 1) / total;
    const timed = known ? elapsed / 60 / total : fraction;          // where a typical build would be by now
    fraction = Math.min(Math.max(timed, fraction), fraction + band * 0.95);
  }
  const pct = Math.round(fraction * 100);
  $("#job-bar").style.width = `${pct}%`; $("#monitor .bar").setAttribute("aria-valuenow", pct);
  $("#job-estimate").textContent = running && known ? `A typical build takes about ${Math.round(total)} minutes on an RTX 4090. The bar is estimated from typical stage times.` : job.status === "queued" ? "Waiting for the GPU." : "";
}
function showMonitor(job, order) {
  monitorJob = job; monitorOrder = order;
  const monitor = $("#monitor");
  monitor.hidden = job.status === "done";
  monitor.className = `monitor ${job.status === "failed" ? "failed" : ["cancelled", "held"].includes(job.status) ? "stopped" : ""}`;
  monitorTick();
  if (job.status === "running" && !clockTimer) clockTimer = setInterval(monitorTick, 1000);
  if (job.status !== "running" && clockTimer) { clearInterval(clockTimer); clockTimer = null; }
}

function renderJob(job) {
  $("#job-title").textContent = job.parent_id ? `4K${job.title ? " · " + job.title : ""}` : job.title ? `Earthfall at ${job.title}` : "Your Earthfall";
  const statuses = { queued: job.position <= 1 ? "Next in line. Waiting for the GPU…" : `In the queue: position ${job.position}.`, running: job.message || "Creating your film…", done: "Done.", failed: "The build failed. Open Details for more information.", cancelled: "Cancelled.", held: job.message || "This take needs review before it can be accepted." };
  $("#job-status").textContent = job.cancel_requested && job.status === "running" ? "Cancelling…" : statuses[job.status];
  $("#cancel").hidden = !["queued", "running"].includes(job.status);
  const order = Object.keys(job.stages).length ? Object.keys(job.stages) : job.parent_id ? ["source", "refine", "assemble", "verify"] : job.workflow === "photographic-grounded-v1" ? ["references", "flight", "opening", "review"] : job.workflow === "photographic-v1" ? ["references", "arrival", "arrival_check", "portrait", "flight", "pacing", "review"] : ["terrain", "stills", "opener", "dive", "cloud", "swoop", "body", "end", "face", "sound"];
  const stages = $("#stages"); stages.replaceChildren();
  for (const name of order) {
    let value = job.stages[name] || "pending";
    if (value === "running" && ["failed", "cancelled", "held"].includes(job.status)) value = job.status === "failed" ? "failed" : "stopped";
    const li = el("li", value); li.append(el("span", "dot"), el("span", "what", STAGE_LABELS[name] || name), el("span", "state", STATE_LABELS[value] ?? value)); stages.append(li);
  }
  showMonitor(job, order);
  const previews = $("#previews"); previews.replaceChildren();
  for (const preview of job.previews) {
    const figure = el("figure"); const image = el("img"); image.src = preview.url; image.alt = PREVIEW_LABELS[preview.name.split(".")[0]] || preview.name;
    figure.append(image, el("figcaption", "", image.alt)); previews.append(figure);
  }
  const video = job.video || job.review_video;
  $("#result").hidden = !video;
  if (video) { if ($("#video").getAttribute("src") !== video) $("#video").src = video; $("#download").href = video + "?download=1"; $("#download").textContent = job.status === "held" ? "Download review take" : "Download MP4"; }
  $("#review-note").hidden = !["done", "held"].includes(job.status) || !["photographic-v1", "photographic-grounded-v1"].includes(job.workflow);
  $("#review-note").textContent = job.status === "held" ? "This film did not pass its automatic checks. You can still watch and download it, or try again from your setup." : "The automatic checks passed. Watch the face and motion yourself before sharing.";
  $("#job-setup-link").replaceChildren();
  if (job.parent_id) { const link = el("a", "", "Back to original film"); link.href = `#job=${job.parent_id}`; $("#job-setup-link").append(link); }
  if (job.status === "done" && !job.parent_id && enhancementParent !== job.id) {
    enhancementParent = job.id; refreshEnhancement(job.id);
  }
  if (job.setup_url) { const link = el("a", "", "Return to your saved setup"); link.href = job.setup_url; $("#job-setup-link").append(link); }
  if (job.review_url && ["done", "held", "failed"].includes(job.status)) { const link = el("a", "", "View checks"); link.href = job.review_url; link.target = "_blank"; $("#job-setup-link").append(" · ", link); }
  $("#job-error-details").hidden = !["failed", "held"].includes(job.status);
  $("#job-error").textContent = job.error || "";
  showReviewFindings(job);
  showCanonicalCheck(job);
  if (["done", "failed", "cancelled", "held"].includes(job.status)) stopFollowing();
}

async function route() {
  const version = ++routeVersion;
  const job = location.hash.match(/^#job=([A-Za-z0-9]+)$/);
  if (job) { $("#example-video").pause(); showJob(job[1]); return; }
  stopFollowing(); stopEnhancement(); $("#video").pause(); $("#job").hidden = true; $("#make").hidden = false;
  const setup = location.hash.match(/^#setup=([a-f0-9]{32})$/);
  if (setup && setup[1] !== state.base) {
    try { const data = await api(`/api/studio/setups/${setup[1]}`); if (version === routeVersion) restore(data); }
    catch (error) { if (version === routeVersion) showFormError(error.message); }
  }
}

async function start() {
  config = await api("/api/config");
  state.config = await api("/api/studio/config");
  if (config.mock) $("#mock-banner").hidden = false;
  setupDestinations(); setupEndings(); setupSearch(); setupUploads();
  await setupExamples(); renderGallery();
  $("#find-photos").onclick = () => { dirty(); findPhotos(true); };
  choosePlace(state.config.destinations[0] || state.config.landmarks[0], false);
  if (state.examples.length) selectExample(state.examples[0]);
  $("#form").onsubmit = save;
  $("#create").onclick = createVideo;
  setInterval(async () => {
    try { const available = await api("/api/studio/availability"); state.config.render_available = available.render_available; updateSave(); }
    catch (_) { state.config.render_available = false; updateSave(); }
  }, 15000);
  $("#wearing").oninput = dirty;
  $("#copy-link").onclick = async () => {
    try { await navigator.clipboard.writeText($("#saved-link").value); $("#copy-link").textContent = "Copied"; }
    catch (_) { $("#saved-link").focus(); $("#saved-link").select(); }
  };
  $("#play-example").onclick = () => { $("#example-video").play().catch(() => { $("#example-missing").hidden = false; }); };
  $("#example-video").onplay = () => { $("#example-video").controls = true; $("#play-example").hidden = true; };
  $("#example-video").onerror = () => { $("#example-missing").hidden = false; $("#play-example").hidden = true; };
  document.addEventListener("play", (event) => { if (event.target.tagName === "VIDEO") document.querySelectorAll("video").forEach((video) => { if (video !== event.target) video.pause(); }); }, true);
  window.addEventListener("hashchange", route);
  window.addEventListener("beforeunload", (event) => { if (state.dirty && state.subject) { event.preventDefault(); event.returnValue = ""; } });
  await route();
}

start().catch((error) => { $("#boot-error").textContent = `${error.message} Reload this page to retry.`; $("#boot-error").hidden = false; });
