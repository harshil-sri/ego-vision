/* ego-kitchen web UI — display + REST/WS calls only.
 * All decisions live in the backend engine; this file renders snapshots
 * (GET /api/state at 2 Hz) and events (WS /ws with reconnect backoff).
 */
(function () {
  "use strict";

  /* ------------------------------------------------------------ helpers */

  var $ = function (id) { return document.getElementById(id); };

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function fmt(n, d) {
    if (n == null || isNaN(n)) return "\u2013";
    return Number(n).toFixed(d == null ? 2 : d);
  }

  /* Rebuild an element's innerHTML only when its signature changed. */
  function swap(id, sig, html) {
    var el = $(id);
    if (!el || el._sig === sig) return false;
    el._sig = sig;
    el.innerHTML = html;
    return true;
  }

  /* -------------------------------------------------------------- state */

  var UI = {
    snap: null,
    ws: "connecting",          // connecting | open | closed
    pollOk: true,
    stopped: false,            // local: user pressed "End session"
    thumbs: {},                // ingredient -> [dataURI, ...] (capped)
    events: [],                // {kind, at, short} newest first, max 12
    warnUntil: 0, warnMsg: "",
    noteUntil: 0, noteMsg: "",
    nudgeUntil: 0, nudgeMsg: "",
    metrics: null,             // last metrics_tick payload
    sampleFrames: [],          // last window_sample thumbs (T frames)
    scanClock: { ing: null, max: 0 },
    lastCurrentIdx: -2,
    helpOpen: false
  };

  /* --------------------------------------------------------- networking */

  function toast(msg, isErr) {
    var box = $("toasts");
    var t = document.createElement("div");
    t.className = "toast" + (isErr ? " toast-error" : "");
    t.textContent = msg;
    box.appendChild(t);
    setTimeout(function () { t.remove(); }, isErr ? 4000 : 3200);
  }

  function post(path, body) {
    return fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {})
    }).then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      pollSoon();
    }).catch(function (e) {
      toast(path + " failed: " + e.message, true);
    });
  }

  var pollTimer = null;
  function pollSoon() {
    if (pollTimer) return;
    pollTimer = setTimeout(function () { pollTimer = null; pollState(); }, 60);
  }

  function pollState() {
    fetch("/api/state").then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    }).then(function (s) {
      UI.pollOk = true;
      UI.snap = s;
      render();
    }).catch(function () {
      if (UI.pollOk) { UI.pollOk = false; }
      render();
    });
  }

  /* WebSocket with exponential backoff 1s..10s. */
  var wsRetryMs = 1000;
  function connectWS() {
    var proto = location.protocol === "https:" ? "wss://" : "ws://";
    var ws;
    try { ws = new WebSocket(proto + location.host + "/ws"); }
    catch (e) { scheduleReconnect(); return; }
    UI.ws = "connecting";
    ws.onopen = function () {
      UI.ws = "open";
      wsRetryMs = 1000;
      renderConn();
    };
    ws.onmessage = function (m) {
      try { handleEvent(JSON.parse(m.data)); } catch (e) { /* ignore */ }
    };
    ws.onclose = function () {
      UI.ws = "closed";
      renderConn();
      scheduleReconnect();
    };
    ws.onerror = function () { try { ws.close(); } catch (e) { /* noop */ } };
  }
  function scheduleReconnect() {
    setTimeout(connectWS, wsRetryMs);
    wsRetryMs = Math.min(wsRetryMs * 2, 10000);
  }

  /* --------------------------------------------------------- WS events */

  function summarize(kind, p) {
    p = p || {};
    switch (kind) {
      case "session_started": return (p.recipe || "") + " / " + (p.mode || "");
      case "session_stopped": return "stopped (" + (p.phase || "?") + ")";
      case "scan_progress":
        return (p.ingredient || "?") + " " + (p.captured || 0) + "/" + (p.target || "?") +
               (p.skipped ? " skipped" : "") + (p.done ? " done" : "");
      case "scan_complete": return "enrolled: " + (p.enrolled || []).join(", ");
      case "ingredient_matched": return (p.ingredient || "?") + " " + fmt(p.score);
      case "verb_predicted": return (p.verb || "?") + " " + fmt(p.confidence);
      case "metrics_tick": return (p.fps || 0) + " fps, " + (p.inference_ms || 0) + " ms";
      case "window_sample": return "T=" + (p.thumbs || []).length + " frames";
      case "error": return p.message || "";
      default:
        if (p.message) return p.message;
        try { return JSON.stringify(p).slice(0, 60); } catch (e) { return ""; }
    }
  }

  function logEvent(kind, at, payload) {
    UI.events.unshift({ kind: kind, at: (at || "").slice(11, 19), short: summarize(kind, payload) });
    if (UI.events.length > 12) UI.events.length = 12;
  }

  function handleEvent(ev) {
    var k = ev.kind, p = ev.payload || {};
    logEvent(k, ev.at, p);

    switch (k) {
      case "ping":
        return;                                   // keepalive, no re-render
      case "session_started":
        UI.stopped = false;
        $("stoppedOverlay").hidden = true;
        pollSoon();
        break;
      case "session_stopped":
        UI.stopped = true;
        $("stoppedOverlay").hidden = false;
        break;
      case "scan_progress":
        if (p.thumb && p.ingredient) {
          var list = UI.thumbs[p.ingredient] || (UI.thumbs[p.ingredient] = []);
          list.push(p.thumb);
          if (list.length > 18) list.splice(0, list.length - 18);
        }
        if (p.skipped && p.ingredient) { UI.thumbs[p.ingredient] = []; }
        pollSoon();
        break;
      case "scan_complete":
        if (p.all_done) toast("All ingredients enrolled \u2014 ready to cook");
        pollSoon();
        break;
      case "wrong_order":
        UI.warnUntil = Date.now() + 3200;
        UI.warnMsg = p.message || "Unexpected step order";
        pollSoon();
        break;
      case "step_advanced":
        UI.noteUntil = Date.now() + 4000;
        UI.noteMsg = p.message || "Step done";
        pollSoon();
        break;
      case "repeat":
      case "idle_nudged":
      case "done":
        UI.noteUntil = Date.now() + 4000;
        UI.noteMsg = p.message || k;
        if (k === "done") toast(p.message || "Recipe complete");
        pollSoon();
        break;
      case "idle":
        UI.nudgeUntil = Date.now() + 9000;
        UI.nudgeMsg = p.message || "Waiting\u2026";
        break;
      case "error":
        toast(p.message || "error", true);
        break;
      case "metrics_tick":
        UI.metrics = p;
        break;
      case "window_sample":
        UI.sampleFrames = p.thumbs || [];
        break;
      default:
        pollSoon();
        break;
    }
    render();
  }

  /* ------------------------------------------------------- view models */

  function enrolledList() {
    var s = UI.snap;
    if (!s || !s.scan) return [];
    return s.scan.enrolled || [];
  }

  function activeScan() {
    var s = UI.snap;
    if (!s || !s.scan || !s.scan.ingredient) return null;
    return s.scan;
  }

  function nextIngredient() {
    var s = UI.snap;
    if (!s) return null;
    var en = enrolledList();
    for (var i = 0; i < s.recipe.ingredients.length; i++) {
      if (en.indexOf(s.recipe.ingredients[i]) === -1) return s.recipe.ingredients[i];
    }
    return null;
  }

  /* Fraction of the 10 s capture window remaining, for the countdown ring. */
  function scanFraction() {
    var sc = activeScan();
    if (!sc || sc.remaining_s == null) return null;
    if (UI.scanClock.ing !== sc.ingredient) {
      UI.scanClock = { ing: sc.ingredient, max: sc.remaining_s };
    } else if (sc.remaining_s > UI.scanClock.max) {
      UI.scanClock.max = sc.remaining_s;
    }
    var m = UI.scanClock.max || 1;
    return Math.max(0, Math.min(1, sc.remaining_s / m));
  }

  /* ------------------------------------------------------------- views */

  var ICO_CHECK = '<svg class="ico-check" viewBox="0 0 24 24" aria-hidden="true" focusable="false">' +
    '<path d="M4.5 12.5l5 5 10-11" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  var ICO_DONE_BIG = '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">' +
    '<path d="M4 12.5l5.5 5.5L20 6.5" fill="none" stroke="currentColor" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"/></svg>';

  function statusIcon(enrolled) {
    return enrolled ? ICO_CHECK : '<span class="ico-circle" aria-hidden="true"></span>';
  }

  /* --- idle / welcome card --- */
  function idleCardHtml() {
    var s = UI.snap, r = s.recipe;
    var next = nextIngredient();
    var rows = r.steps.map(function (st, i) {
      return '<li><span class="step-n">' + (i + 1) + '</span><span>' +
        esc(st.verb) + " " + esc(st.ingredient) +
        (st.note ? ' <span class="tl-note">(' + esc(st.note) + ")</span>" : "") +
        "</span></li>";
    }).join("");
    return '<div class="idle-card">' +
      '<div><h2 class="idle-title">Waiting to start</h2>' +
      '<p class="idle-sub">The engine is running. Enroll each ingredient, then cook.</p></div>' +
      '<div class="idle-meta">' +
        '<span class="chip">' + r.steps.length + " steps</span>" +
        '<span class="chip">' + r.ingredients.length + " ingredients</span>" +
        '<span class="chip">mode: ' + esc(s.mode) + "</span>" +
      "</div>" +
      '<h2>Recipe \u2014 ' + esc(r.name) + "</h2>" +
      '<ul class="idle-steps">' + rows + "</ul>" +
      (next
        ? '<button class="btn btn-primary" data-act="begin" data-arg="' + esc(next) + '">' +
            "Start \u2014 hold up " + esc(next) + "</button>"
        : '<button class="btn btn-primary" data-act="finish">Start cooking</button>') +
      "</div>";
  }

  /* --- scan wizard --- */
  function wizardHtml() {
    var s = UI.snap, r = s.recipe;
    var en = enrolledList();
    var act = activeScan();
    var next = nextIngredient();
    var total = r.ingredients.length;
    var pct = total ? Math.round(100 * en.length / total) : 0;

    var rows = r.ingredients.map(function (ing) {
      var isIn = en.indexOf(ing) !== -1;
      var isActive = act && act.ingredient === ing;
      var thumbs = UI.thumbs[ing] || [];
      var lastThumb = thumbs.length ? thumbs[thumbs.length - 1] : "";
      return '<li class="ing-row" data-enrolled="' + isIn + '">' +
        '<span class="ing-row-status">' + statusIcon(isIn) +
          "<span>" + (isIn ? "enrolled" : isActive ? "capturing" : "pending") + "</span></span>" +
        '<span class="ing-row-name">' + esc(ing) + "</span>" +
        (lastThumb ? '<img class="ing-thumb" src="' + lastThumb + '" alt="last thumbnail">' : "") +
        '<span class="ing-row-actions">' +
          '<button class="btn-ghost-sm" data-act="retake" data-arg="' + esc(ing) + '"' +
            (isIn || isActive ? "" : " disabled") + ">Retake</button>" +
          '<button class="btn-ghost-sm" data-act="skip"' +
            (isActive ? ' data-arg="' + esc(ing) + '"' : " disabled") + ">Skip</button>" +
        "</span></li>";
    }).join("");

    var card;
    if (act) {
      card =
        '<div class="capture-card" data-active="true">' +
          '<div class="ring" aria-hidden="true">' +
            '<svg width="92" height="92" viewBox="0 0 92 92">' +
              '<circle class="ring-track" cx="46" cy="46" r="40" fill="none" stroke-width="7"/>' +
              '<circle id="ringFill" class="ring-fill" cx="46" cy="46" r="40" fill="none" stroke-width="7" stroke-linecap="round" stroke-dasharray="251.3" stroke-dashoffset="0"/>' +
            "</svg>" +
            '<span class="ring-num" id="ringNum"></span>' +
          "</div>" +
          '<div class="capture-main">' +
            '<div class="capture-title">Hold up <em>' + esc(act.ingredient) + "</em></div>" +
            '<div class="capture-sub" id="captureCount">capturing\u2026</div>' +
            '<div class="capture-actions">' +
              '<button class="btn-ghost-sm" data-act="skip" data-arg="' + esc(act.ingredient) + '">Skip</button>' +
            "</div>" +
          "</div>" +
        "</div>";
    } else if (next) {
      card =
        '<div class="capture-card" data-active="false">' +
          '<div class="ring"><span class="ring-num ring-idle-mark" aria-hidden="true">' +
            (total - en.length) + "</span></div>" +
          '<div class="capture-main">' +
            '<div class="capture-title">Next: <em>' + esc(next) + "</em></div>" +
            '<div class="capture-sub">Hold it in the highlighted zone, then start the 10 s capture.</div>' +
            '<div class="capture-actions">' +
              '<button class="btn btn-primary" data-act="begin" data-arg="' + esc(next) + '">Capture ' + esc(next) + "</button>" +
            "</div>" +
          "</div>" +
        "</div>";
    } else {
      card =
        '<div class="capture-card" data-active="false">' +
          '<div class="capture-main">' +
            '<div class="capture-title">All ingredients enrolled</div>' +
            '<div class="capture-sub">Retake any ingredient, or start cooking.</div>' +
          "</div>" +
        "</div>";
    }

    return '<div class="wizard">' +
      '<div class="wiz-head"><h2>Ingredient scan</h2>' +
        '<span class="wiz-count">' + en.length + " / " + total + " enrolled</span></div>" +
      '<div class="wiz-bar" role="progressbar" aria-valuenow="' + en.length + '" aria-valuemin="0" aria-valuemax="' + total + '">' +
        '<div class="wiz-bar-fill" style="width:' + pct + '%"></div></div>' +
      card +
      '<div id="thumbGrid" class="thumb-grid"></div>' +
      '<ul class="ing-rows">' + rows + "</ul>" +
      '<button class="btn btn-primary wiz-finish" data-act="finish"' +
        (en.length > 0 ? "" : " disabled") + ">Start cooking (" + en.length +
        "/" + total + " enrolled)</button>" +
      "</div>";
  }

  /* --- recipe timeline (cooking / done) --- */
  function timelineHtml(steps) {
    var cur = -1, i;
    for (i = 0; i < steps.length; i++) if (steps[i].current) cur = i;
    var rows = steps.map(function (st, idx) {
      var state = st.done ? "done" : (st.current ? "current" : "upcoming");
      var icon = st.done ? ICO_CHECK :
        (st.current ? '<span class="tl-ico-current" aria-hidden="true"></span>'
                    : '<span class="tl-ico-upcoming" aria-hidden="true"></span>');
      var nextLine = "";
      if (st.current && idx + 1 < steps.length) {
        var nx = steps[idx + 1];
        nextLine = '<span class="tl-next">Next: ' + esc(nx.verb) + " " + esc(nx.ingredient) + "</span>";
      }
      return '<li class="tl-row" data-state="' + state + '">' +
        '<span class="tl-num">' + (idx + 1) + "</span>" +
        '<span class="tl-ico">' + icon + "</span>" +
        '<span class="tl-text"><span class="tl-main">' + esc(st.verb) + " " + esc(st.ingredient) + "</span>" +
          (st.note ? '<span class="tl-note">' + esc(st.note) + "</span>" : "") +
          nextLine + "</span></li>";
    }).join("");
    return '<ul class="timeline">' + rows + "</ul>";
  }

  function verbRowHtml() {
    var s = UI.snap;
    if (s.mode !== "keyboard" || s.phase !== "cooking") return "";
    var btns = s.recipe.verbs.slice(0, 9).map(function (v, i) {
      return '<button class="verb-btn" data-act="verb" data-arg="' + esc(v) + '">' +
        '<span class="verb-key" aria-hidden="true">' + (i + 1) + "</span>" + esc(v) + "</button>";
    }).join("");
    return '<div class="verb-row-wrap"><h2>Verbs (keyboard mode)</h2>' +
      '<div class="verb-row">' + btns + "</div></div>";
  }

  function cookingHtml() {
    var s = UI.snap;
    var cur = -1;
    s.recipe.steps.forEach(function (st, i) { if (st.current) cur = i; });
    if (cur !== UI.lastCurrentIdx) { UI.lastCurrentIdx = cur; }
    return '<h2>Recipe \u2014 ' + esc(s.recipe.name) + "</h2>" +
      timelineHtml(s.recipe.steps) + verbRowHtml();
  }

  function doneHtml() {
    var s = UI.snap;
    return '<div class="done-card">' +
      '<div class="done-mark" aria-hidden="true">' + ICO_DONE_BIG + "</div>" +
      '<h2 class="done-title">Recipe complete</h2>' +
      '<p class="done-sub">' + esc(s.recipe.name) + " \u2014 nicely done.</p>" +
      "</div>" +
      timelineHtml(s.recipe.steps);
  }

  /* ------------------------------------------------------- main render */

  function viewMode() {
    if (UI.stopped) return "stopped";
    var s = UI.snap;
    if (!s) return "loading";
    if (s.phase === "done") return "done";
    if (s.phase === "cooking") return "cooking";
    /* engine stays 'idle' during scanning: key off the scan fields */
    if (activeScan()) return "scanning";
    if (enrolledList().length > 0) return "scanning";
    return "idle";
  }

  function renderPanel() {
    var mode = viewMode();
    var s = UI.snap, sig, html;
    if (mode === "loading") return;
    if (mode === "stopped") return;                       // overlay covers UI
    if (mode === "done") {
      sig = "done:" + s.recipe.name;
      html = doneHtml();
    } else if (mode === "cooking") {
      sig = "cook:" + JSON.stringify(s.recipe.steps) + ":" + s.mode + ":" + s.phase;
      html = cookingHtml();
    } else if (mode === "scanning") {
      sig = "scan:" + (activeScan() ? activeScan().ingredient : "-") + ":" +
            JSON.stringify(enrolledList());
      html = wizardHtml();
    } else {
      sig = "idle:" + JSON.stringify(s.recipe.ingredients);
      html = idleCardHtml();
    }
    var changed = swap("panel", sig, html);

    /* live values that must not wait for a structural change */
    if (mode === "scanning") updateCaptureLive(changed);
    if (mode === "cooking" && changed) {
      var row = document.querySelector(".tl-row[data-state='current']");
      if (row && UI.lastCurrentIdx >= 0) row.scrollIntoView({ block: "nearest" });
    }
  }

  function updateCaptureLive(rebuilt) {
    var act = activeScan();
    var grid = $("thumbGrid");
    if (grid) {
      var thumbs = act ? (UI.thumbs[act.ingredient] || []) : [];
      var gsig = (act ? act.ingredient : "-") + ":" + thumbs.length;
      swap("thumbGrid", gsig, thumbs.slice(-9).map(function (t) {
        return '<img src="' + t + '" alt="scan thumbnail">';
      }).join(""));
    }
    var frac = act ? scanFraction() : null;
    var ring = $("ringFill");
    if (ring && frac != null) {
      var C = 251.3;
      ring.setAttribute("stroke-dashoffset", String(C * (1 - frac)));
    }
    var num = $("ringNum");
    if (num) {
      num.textContent = act && act.remaining_s != null
        ? Math.ceil(act.remaining_s) + "s" : "";
    }
    var cnt = $("captureCount");
    if (cnt && act) {
      cnt.textContent = "captured " + act.captured + " / " + act.target +
                        " frames \u00b7 " + fmt(act.remaining_s, 1) + " s left";
    }
  }

  function renderChips() {
    var s = UI.snap;
    if (!s) return;
    $("verbChip").textContent = s.verb.name
      ? s.verb.name + " " + fmt(s.verb.confidence) : "\u2013";
    $("ingChip").textContent = s.ingredient.name && s.ingredient.name !== "unknown"
      ? s.ingredient.name + " " + fmt(s.ingredient.score) : "unknown";
  }

  function renderTop() {
    var s = UI.snap;
    if (!s) return;
    $("recipeName").textContent = "\u00b7 " + s.recipe.name;
    var m = viewMode();
    var label = { loading: "starting", idle: "waiting", scanning: "scanning",
                  cooking: "cooking", done: "done", stopped: "ended" }[m];
    var chip = $("phaseChip");
    chip.dataset.phase = m;
    $("phaseText").textContent = label;
    $("modeChip").textContent = "mode: " + s.mode;
  }

  function renderWarning() {
    var s = UI.snap, msg = null;
    if (UI.warnUntil > Date.now()) msg = UI.warnMsg;
    else if (s && s.warning) msg = s.warning.message;
    var el = $("warnBanner");
    if (msg) { $("warnText").textContent = msg; el.hidden = false; }
    else el.hidden = true;

    var note = $("noteLine");
    if (UI.noteUntil > Date.now() && UI.noteMsg) {
      note.textContent = UI.noteMsg; note.hidden = false;
    } else note.hidden = true;

    var nug = $("idleNudge");
    if (UI.nudgeUntil > Date.now() && UI.nudgeMsg) {
      nug.textContent = UI.nudgeMsg; nug.hidden = false;
    } else nug.hidden = true;
  }

  function renderDebug() {
    var s = UI.snap;
    if (!s) return;
    var m = (s && s.metrics) || {};
    var mt = UI.metrics || {};
    $("dbgVerb").textContent = mt.verb || s.verb.name || "\u2013";
    $("dbgConf").textContent = fmt(s.verb.confidence);
    $("dbgIng").textContent = mt.ingredient || s.ingredient.name || "\u2013";
    $("dbgScore").textContent = fmt(s.ingredient.score);
    $("dbgFps").textContent = fmt(mt.fps != null ? mt.fps : m.fps, 1);
    $("dbgInf").textContent = fmt(m.inference_ms, 1) + " ms";
    $("dbgDrop").textContent = String(m.dropped != null ? m.dropped : 0);
    $("dbgSmooth").textContent = m.smoother || "\u2013";
    $("dbgMinConf").textContent = fmt(m.min_conf);

    var strip = UI.sampleFrames.map(function (t) {
      return "<img class=\"dbg-frame\" src=\"" + t + "\" alt=\"sampled frame\">";
    }).join("");
    swap("dbgFrames", String(UI.sampleFrames.length),
      strip || '<span class="dbg-frames-empty">no sampled-frame strip yet</span>');

    var lis = UI.events.map(function (e) {
      return "<li><span class=\"ev-kind\">" + esc(e.kind) + "</span>" +
        "<span class=\"ev-time\">" + esc(e.at) + "</span>" +
        "<span>" + esc(e.short) + "</span></li>";
    }).join("");
    swap("dbgEvents", String(UI.events.length) + ":" + (UI.events[0] && UI.events[0].kind) +
      ":" + (UI.events[0] && UI.events[0].at),
      lis || '<li class="dbg-events-empty">no events yet</li>');
  }

  function renderConn() {
    var banner = $("connBanner");
    if (UI.ws === "open" && UI.pollOk) { banner.hidden = true; return; }
    banner.hidden = false;
    $("connBannerText").textContent = !UI.pollOk && UI.ws !== "open"
      ? "offline \u2014 server unreachable, retrying\u2026"
      : "reconnecting\u2026";
  }

  function render() {
    renderTop();
    renderChips();
    renderPanel();
    renderWarning();
    renderDebug();
    renderConn();
  }

  /* ------------------------------------------------------ interactions */

  $("panel").addEventListener("click", function (e) {
    var b = e.target.closest("button[data-act]");
    if (!b || b.disabled) return;
    var act = b.dataset.act, arg = b.dataset.arg;
    if (act === "begin" || act === "retake") {
      post("/api/scan/" + act, { ingredient: arg });
    } else if (act === "skip") {
      post("/api/scan/skip", {});
    } else if (act === "finish") {
      post("/api/scan/finish", {});
    } else if (act === "verb") {
      post("/api/verb", { verb: arg });
    }
  });

  $("stopBtn").addEventListener("click", function () {
    fetch("/api/session/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}"
    }).then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      UI.stopped = true;
      $("stoppedOverlay").hidden = false;
      render();
    }).catch(function (e) {
      toast("could not stop session: " + e.message, true);
    });
  });

  $("restartBtn").addEventListener("click", function () {
    fetch("/api/session/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}"
    }).then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      UI.stopped = false;              // session_started (WS) also clears it
      $("stoppedOverlay").hidden = true;
      pollSoon();
      render();
    }).catch(function (e) {
      toast("could not restart session: " + e.message, true);
    });
  });

  /* debug drawer */
  function toggleDebug() {
    var d = $("debugDrawer");
    var open = d.dataset.open !== "open";
    d.dataset.open = open ? "open" : "closed";
    $("debugBody").hidden = !open;
    $("debugToggle").setAttribute("aria-expanded", String(open));
  }
  $("debugToggle").addEventListener("click", toggleDebug);

  /* presentation mode */
  function togglePresentation() {
    document.body.classList.toggle("presentation");
  }

  /* help overlay */
  function toggleHelp(force) {
    UI.helpOpen = force != null ? force : !$("helpOverlay").hidden;
    $("helpOverlay").hidden = !UI.helpOpen;
  }
  $("helpBtn").addEventListener("click", function () { toggleHelp(); });
  $("helpClose").addEventListener("click", function () { toggleHelp(false); });

  /* keyboard shortcuts */
  document.addEventListener("keydown", function (e) {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === "Escape") { toggleHelp(false); return; }
    if (e.key === "?") { toggleHelp(); return; }
    if (UI.helpOpen) return;
    if (e.key === "d" || e.key === "D") { toggleDebug(); return; }
    if (e.key === "p" || e.key === "P") { togglePresentation(); return; }
    if (/^[1-9]$/.test(e.key)) {
      var s = UI.snap;
      if (s && s.mode === "keyboard" && s.phase === "cooking") {
        var v = s.recipe.verbs[Number(e.key) - 1];
        if (v) post("/api/verb", { verb: v });
      }
    }
  });

  /* video stream fallback message */
  $("video").addEventListener("error", function () {
    this.style.display = "none";
    var w = this.parentElement;
    w.style.minHeight = "260px";
    w.setAttribute("role", "img");
    w.setAttribute("aria-label", "video stream unavailable");
    w.insertAdjacentHTML("beforeend",
      '<div class="capture-sub" style="position:absolute;inset:0;display:flex;' +
      'align-items:center;justify-content:center;color:#aab1ba">' +
      "video stream unavailable</div>");
  });

  /* ------------------------------------------------------------- start */

  pollState();
  setInterval(pollState, 500);      // 2 Hz fallback + remaining_s source
  connectWS();
})();
