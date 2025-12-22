(function () {
    const endpoint = "/api/tools/run";
    const HAS_PLUS = window.rrHasPlus === true || window.rrHasPlus === "true";
    window.__rrToolState = window.__rrToolState || {};

    async function runToolRequest(tool, input = {}) {
        const payload = { tool, input };

        try {
            const response = await fetch(endpoint, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            const data = await response.json().catch(() => null);
            if (data) return data;

            return {
                ok: false,
                error: { code: "INVALID_RESPONSE", message: "Unable to parse server response." },
                data: null,
                request_id: null,
            };
        } catch (error) {
            return {
                ok: false,
                error: { code: "NETWORK_ERROR", message: error?.message || "Network error" },
                data: null,
                request_id: null,
            };
        }
    }

    // ---- Daily Phrase TTS (client only) ----
    let cachedVoices = [];
    function loadVoices() {
        if (!("speechSynthesis" in window)) return [];
        const voices = window.speechSynthesis.getVoices();
        if (voices && voices.length) {
            cachedVoices = voices;
        }
        return cachedVoices;
    }
    if ("speechSynthesis" in window) {
        loadVoices();
        window.speechSynthesis.onvoiceschanged = loadVoices;
    }

    function langCodeFor(language) {
        const map = {
            Spanish: "es-ES",
            French: "fr-FR",
            German: "de-DE",
            Italian: "it-IT",
            Japanese: "ja-JP",
        };
        return map[language] || "en-US";
    }

    function normalizePhraseValue(text) {
        if (text === undefined || text === null) return "";
        const value = String(text).trim().replace(/^["']+|["']+$/g, "");
        const match = value.match(/^(?:phrase|translation|example)\s*:\s*(.+)$/i);
        const withoutLabel = match ? match[1].trim() : value;
        return withoutLabel.replace(/^[\-\u2013\u2014]\s*/, "").trim();
    }

    function renderAudioControls(form) {
        const outputEl = document.querySelector("[data-tool-output]");
        if (!outputEl) return;
        let audioWrap = outputEl.querySelector(".tool-audio");
        const state = window.__rrToolState["daily-phrase"];
        const phraseText = normalizePhraseValue(state?.phrase);
        if (!state || !phraseText) return;
        if (!("speechSynthesis" in window) || !("SpeechSynthesisUtterance" in window)) {
            if (!audioWrap) {
                audioWrap = document.createElement("div");
                audioWrap.className = "tool-audio";
                const note = document.createElement("div");
                note.className = "tool-audio-note";
                note.textContent = "Audio not supported on this device.";
                audioWrap.append(note);
                outputEl.append(audioWrap);
            }
            return;
        }
        if (!audioWrap) {
            audioWrap = document.createElement("div");
            audioWrap.className = "tool-audio";
            const playBtn = document.createElement("button");
            playBtn.type = "button";
            playBtn.textContent = "Play";
            playBtn.onclick = () => speakPhrase(state);
            const stopBtn = document.createElement("button");
            stopBtn.type = "button";
            stopBtn.textContent = "Stop";
            stopBtn.onclick = () => window.speechSynthesis.cancel();
            const note = document.createElement("div");
            note.className = "tool-audio-note";
            note.textContent = "Same phrase for everyone today.";
            audioWrap.append(playBtn, stopBtn, note);
            outputEl.append(audioWrap);
        }
    }

    function speakPhrase(state) {
        if (!state) return;
        if (!("speechSynthesis" in window) || !("SpeechSynthesisUtterance" in window)) return;
        const phraseText = normalizePhraseValue(state.phrase);
        if (!phraseText) return;
        window.speechSynthesis.cancel();
        const langCode = langCodeFor(state.language);
        const utter = new SpeechSynthesisUtterance(phraseText);
        utter.lang = langCode;
        const voices = loadVoices();
        if (voices && voices.length) {
            const match = voices.find((v) => v.lang && v.lang.toLowerCase().startsWith(langCode.split("-")[0]));
            if (match) utter.voice = match;
        }
        utter.rate = 0.95;
        utter.pitch = 1.0;
        window.speechSynthesis.speak(utter);
    }

    function initAudioControls(form) {
        renderAudioControls(form);
        window.addEventListener("beforeunload", () => {
            if ("speechSynthesis" in window) window.speechSynthesis.cancel();
        });
    }

    // ----- Daily Check-In (client only) -----
    const DAILY_KEY_PREFIX = "rr_tools_daily_checkin::";
    const WORKOUT_LOG_KEY = "rr_workout_log_v1";
    const COUNTDOWN_KEY = "rr_countdowns_v1";

    function normalizeHabitName(raw = "") {
        return raw.trim().toLowerCase().replace(/\s+/g, " ");
    }

    function storageKeyForHabit(name) {
        const normalized = normalizeHabitName(name);
        return `${DAILY_KEY_PREFIX}${normalized}`;
    }

    function getStoredHabit(name) {
        const key = storageKeyForHabit(name);
        const raw = localStorage.getItem(key);
        if (!raw) return { entries: {} };
        try {
            const parsed = JSON.parse(raw);
            return parsed && typeof parsed === "object" ? parsed : { entries: {} };
        } catch (err) {
            return { entries: {} };
        }
    }

    function saveHabit(name, data) {
        const key = storageKeyForHabit(name);
        localStorage.setItem(key, JSON.stringify(data));
    }

    function clearHabit(name) {
        const key = storageKeyForHabit(name);
        localStorage.removeItem(key);
    }

    // ----- Workout Log (client only) -----
    function loadWorkoutStore() {
        const raw = localStorage.getItem(WORKOUT_LOG_KEY);
        if (!raw) return { version: 1, workouts: [] };
        try {
            const parsed = JSON.parse(raw);
            if (!parsed || typeof parsed !== "object") return { version: 1, workouts: [] };
            if (!parsed.version) {
                // migration guard
                const workouts = Array.isArray(parsed.workouts) ? parsed.workouts : [];
                return { version: 1, workouts };
            }
            if (parsed.version !== 1) return { version: 1, workouts: Array.isArray(parsed.workouts) ? parsed.workouts : [] };
            return { version: 1, workouts: Array.isArray(parsed.workouts) ? parsed.workouts : [] };
        } catch (e) {
            return { version: 1, workouts: [] };
        }
    }

    function saveWorkoutStore(store) {
        localStorage.setItem(WORKOUT_LOG_KEY, JSON.stringify(store));
    }

    function todayISO() {
        const d = new Date();
        const yyyy = d.getFullYear();
        const mm = String(d.getMonth() + 1).padStart(2, "0");
        const dd = String(d.getDate()).padStart(2, "0");
        return `${yyyy}-${mm}-${dd}`;
    }

    function makeId() {
        if (window.crypto && typeof window.crypto.randomUUID === "function") return window.crypto.randomUUID();
        return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    }

    function escapeCsv(value) {
        if (value === null || value === undefined) return "";
        const s = String(value);
        if (/[",\n\r]/.test(s)) {
            return `"${s.replace(/"/g, '""')}"`;
        }
        return s;
    }

    function downloadText(filename, text, mime = "text/plain") {
        const blob = new Blob([text], { type: mime });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 250);
    }

    function parsePositiveIntOrEmpty(raw) {
        const s = String(raw || "").trim();
        if (!s) return null;
        const n = Number(s);
        if (!Number.isFinite(n)) return null;
        const asInt = Math.trunc(n);
        if (asInt <= 0) return null;
        return asInt;
    }

    function parseNumberOrEmpty(raw) {
        const s = String(raw || "").trim();
        if (!s) return null;
        const n = Number(s);
        if (!Number.isFinite(n)) return null;
        return n;
    }

    function toAbsoluteUrl(pathOrUrl) {
        const raw = (pathOrUrl || "").trim();
        if (!raw) return "";
        try {
            return new URL(raw, window.location.origin).toString();
        } catch (e) {
            return raw;
        }
    }

    function initWorkoutLog(form, outputEl, statusEl) {
        const slug = form?.dataset?.toolSlug;
        if (slug !== "workout-log") return;

        const runBtn = form.querySelector("[data-run-button]") || form.querySelector("button[type='submit']");
        if (runBtn) runBtn.style.display = "none";

        const outputPre = outputEl ? outputEl.querySelector(".tool-output-pre") : null;
        if (outputPre) outputPre.textContent = "";

        const custom = document.getElementById("tool-custom-ui");
        if (!custom) return;
        custom.innerHTML = "";

        const dateInput = form.querySelector('[name="workout_date"]');
        const nameInput = form.querySelector('[name="workout_name"]');
        if (dateInput && !dateInput.value) dateInput.value = todayISO();

        // Build sets UI
        const setsWrap = document.createElement("div");
        setsWrap.className = "wl-sets";
        const setsTitle = document.createElement("div");
        setsTitle.className = "wl-section-title";
        setsTitle.textContent = "Sets";

        const table = document.createElement("div");
        table.className = "wl-table";
        const head = document.createElement("div");
        head.className = "wl-row wl-head";
        head.innerHTML =
            "<div>Exercise</div><div>Sets</div><div>Reps</div><div>Weight</div><div>Unit</div><div></div>";
        table.appendChild(head);

        const body = document.createElement("div");
        body.className = "wl-body";
        table.appendChild(body);

        function addRow(prefill) {
            const row = document.createElement("div");
            row.className = "wl-row wl-item";

            const ex = document.createElement("input");
            ex.type = "text";
            ex.placeholder = "Bench press";
            ex.value = prefill?.exercise || "";

            const sets = document.createElement("input");
            sets.type = "number";
            sets.min = "1";
            sets.step = "1";
            sets.placeholder = "3";
            sets.value = prefill?.sets ?? "";

            const reps = document.createElement("input");
            reps.type = "number";
            reps.min = "1";
            reps.step = "1";
            reps.placeholder = "8";
            reps.value = prefill?.reps ?? "";

            const weight = document.createElement("input");
            weight.type = "number";
            weight.step = "0.5";
            weight.min = "0";
            weight.placeholder = "135";
            weight.value = prefill?.weight ?? "";

            const unit = document.createElement("select");
            ["lb", "kg"].forEach((u) => {
                const opt = document.createElement("option");
                opt.value = u;
                opt.textContent = u;
                if ((prefill?.unit || "lb") === u) opt.selected = true;
                unit.appendChild(opt);
            });

            const remove = document.createElement("button");
            remove.type = "button";
            remove.className = "wl-remove";
            remove.textContent = "Remove";
            remove.onclick = () => row.remove();

            row.append(ex, sets, reps, weight, unit, remove);
            body.appendChild(row);
        }

        addRow();

        const addBtn = document.createElement("button");
        addBtn.type = "button";
        addBtn.className = "tool-run-btn wl-add";
        addBtn.textContent = "Add row";
        addBtn.onclick = () => addRow();

        setsWrap.append(setsTitle, table, addBtn);

        // Actions
        const actions = document.createElement("div");
        actions.className = "wl-actions";

        const saveBtn = document.createElement("button");
        saveBtn.type = "button";
        saveBtn.className = "tool-run-btn";
        saveBtn.textContent = "Save workout";

        const clearBtn = document.createElement("button");
        clearBtn.type = "button";
        clearBtn.className = "tool-clear-btn";
        clearBtn.textContent = "Clear form";

        const exportBtn = document.createElement("button");
        exportBtn.type = "button";
        exportBtn.className = "tool-copy-btn wl-export";
        exportBtn.style.display = "inline-block";
        exportBtn.textContent = "Export CSV";
        exportBtn.dataset.plusRequired = "true";

        actions.append(saveBtn, clearBtn, exportBtn);

        // Output: recent workouts
        const recentWrap = document.createElement("div");
        recentWrap.className = "wl-recent";
        const recentTitle = document.createElement("div");
        recentTitle.className = "wl-section-title";
        recentTitle.textContent = "Recent workouts";
        const list = document.createElement("div");
        list.className = "wl-list";
        const detail = document.createElement("div");
        detail.className = "wl-detail";
        detail.style.display = "none";
        recentWrap.append(recentTitle, list, detail);

        function setStatus(text, type = "") {
            if (!statusEl) return;
            statusEl.textContent = text || "";
            statusEl.className = "tool-status";
            if (type) statusEl.classList.add(type);
        }

        function summarize(workout) {
            const rows = Array.isArray(workout.rows) ? workout.rows : [];
            const unique = new Set(rows.map((r) => (r.exercise || "").trim()).filter(Boolean));
            const totalSets = rows.reduce((sum, r) => sum + (Number(r.sets) || 0), 0);
            const exCount = unique.size || rows.length;
            return `${exCount} exercises, ${totalSets} sets`;
        }

        function renderDetail(workout) {
            detail.innerHTML = "";
            if (!workout) {
                detail.style.display = "none";
                return;
            }
            detail.style.display = "block";
            const header = document.createElement("div");
            header.className = "wl-detail-header";
            header.textContent = `${workout.date} • ${workout.name}`;

            const close = document.createElement("button");
            close.type = "button";
            close.className = "wl-detail-close";
            close.textContent = "Close";
            close.onclick = () => renderDetail(null);

            const rowsWrap = document.createElement("div");
            rowsWrap.className = "wl-detail-rows";
            (workout.rows || []).forEach((r) => {
                const row = document.createElement("div");
                row.className = "wl-detail-row";
                row.innerHTML = `<div class="wl-detail-ex">${escapeHtml(r.exercise || "")}</div>
<div class="wl-detail-meta">${escapeHtml(String(r.sets || ""))}×${escapeHtml(String(r.reps || ""))} @ ${escapeHtml(String(r.weight ?? ""))}${escapeHtml(r.unit || "")}</div>`;
                rowsWrap.appendChild(row);
            });

            const wrap = document.createElement("div");
            wrap.className = "wl-detail-card";
            wrap.append(header, close, rowsWrap);
            detail.appendChild(wrap);
        }

        function escapeHtml(s) {
            return String(s).replace(/[&<>"']/g, (ch) => {
                const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
                return map[ch] || ch;
            });
        }

        function renderList() {
            const store = loadWorkoutStore();
            const workouts = Array.isArray(store.workouts) ? store.workouts : [];
            list.innerHTML = "";
            if (!workouts.length) {
                const empty = document.createElement("div");
                empty.className = "wl-empty";
                empty.textContent = "No workouts saved yet.";
                list.appendChild(empty);
                renderDetail(null);
                return;
            }
            workouts.slice(0, 10).forEach((w) => {
                const item = document.createElement("div");
                item.className = "wl-item-row";
                const left = document.createElement("div");
                left.className = "wl-item-left";
                const title = document.createElement("div");
                title.className = "wl-item-title";
                title.textContent = `${w.date || "—"} • ${w.name || "Workout"}`;
                const sub = document.createElement("div");
                sub.className = "wl-item-sub";
                sub.textContent = summarize(w);
                left.append(title, sub);

                const buttons = document.createElement("div");
                buttons.className = "wl-item-actions";
                const view = document.createElement("button");
                view.type = "button";
                view.className = "wl-small";
                view.textContent = "View";
                view.onclick = () => renderDetail(w);
                const del = document.createElement("button");
                del.type = "button";
                del.className = "wl-small danger";
                del.textContent = "Delete";
                del.onclick = () => {
                    const next = loadWorkoutStore();
                    next.workouts = (next.workouts || []).filter((x) => x.id !== w.id);
                    saveWorkoutStore(next);
                    setStatus("Deleted.", "success");
                    renderList();
                };
                buttons.append(view, del);

                item.append(left, buttons);
                list.appendChild(item);
            });
        }

        function readRows() {
            const out = [];
            const rows = body.querySelectorAll(".wl-item");
            rows.forEach((rowEl) => {
                const [exerciseEl, setsEl, repsEl, weightEl, unitEl] = rowEl.querySelectorAll("input, select");
                const exercise = (exerciseEl?.value || "").trim();
                const sets = parsePositiveIntOrEmpty(setsEl?.value);
                const reps = parsePositiveIntOrEmpty(repsEl?.value);
                const weight = parseNumberOrEmpty(weightEl?.value);
                const unit = unitEl?.value || "lb";
                if (!exercise && !sets && !reps && (weight === null || weight === 0)) return;
                out.push({ exercise, sets: sets || 0, reps: reps || 0, weight: weight ?? "", unit });
            });
            return out;
        }

        saveBtn.onclick = () => {
            const name = (nameInput?.value || "").trim();
            if (!name) {
                setStatus("Workout name is required.", "error");
                return;
            }
            const date = (dateInput?.value || "").trim() || todayISO();
            const rows = readRows();
            if (!rows.length) {
                setStatus("Add at least one set row.", "error");
                return;
            }
            const workout = {
                id: makeId(),
                date,
                name,
                created_at: new Date().toISOString(),
                rows,
            };
            const store = loadWorkoutStore();
            store.workouts = [workout, ...(store.workouts || [])];
            saveWorkoutStore(store);
            setStatus("Saved locally.", "success");
            renderList();
        };

        clearBtn.onclick = () => {
            if (nameInput) nameInput.value = "";
            if (dateInput) dateInput.value = todayISO();
            body.innerHTML = "";
            addRow();
            setStatus("Cleared.", "success");
        };

        exportBtn.onclick = () => {
            if (!HAS_PLUS) {
                if (typeof window.openPlusPrompt === "function") {
                    window.openPlusPrompt("Export is a Plus feature", "Unlock CSV and PDF exports, plus saved history.");
                    return;
                }
            }
            const store = loadWorkoutStore();
            const workouts = Array.isArray(store.workouts) ? store.workouts : [];
            if (!workouts.length) {
                setStatus("Nothing to export yet.", "error");
                return;
            }
            const header = ["date", "workout_name", "exercise", "sets", "reps", "weight", "unit", "created_at"];
            const lines = [header.join(",")];
            workouts
                .slice()
                .reverse()
                .forEach((w) => {
                    (w.rows || []).forEach((r) => {
                        const row = [
                            escapeCsv(w.date || ""),
                            escapeCsv(w.name || ""),
                            escapeCsv(r.exercise || ""),
                            escapeCsv(r.sets ?? ""),
                            escapeCsv(r.reps ?? ""),
                            escapeCsv(r.weight ?? ""),
                            escapeCsv(r.unit || ""),
                            escapeCsv(w.created_at || ""),
                        ];
                        lines.push(row.join(","));
                    });
                });
            const csv = lines.join("\n");
            downloadText(`workout-log-${todayISO()}.csv`, csv, "text/csv");
            setStatus("CSV downloaded.", "success");
        };

        // Prevent any backend submission for this tool
        form.addEventListener("submit", (e) => {
            e.preventDefault();
            setStatus("Use Save workout to store locally.", "success");
        });

        custom.append(setsWrap, actions, recentWrap);
        renderList();
    }

    // ----- Grocery List (shared via token) -----
    function initGroceryList(form, outputEl, statusEl) {
        const slug = form?.dataset?.toolSlug;
        if (slug !== "grocery-list") return;

        const runBtn = form.querySelector("[data-run-button]") || form.querySelector("button[type='submit']");
        if (runBtn) runBtn.style.display = "none";

        if (outputEl) {
            // This tool is interactive; don't use the generic output box.
            outputEl.style.display = "none";
        }

        const custom = document.getElementById("tool-custom-ui");
        if (!custom) return;
        custom.innerHTML = "";

        const nameField = form.querySelector('[name="list_name"]');

        const setStatus = (text, type = "") => {
            if (!statusEl) return;
            statusEl.textContent = text || "";
            statusEl.className = "tool-status";
            if (type) statusEl.classList.add(type);
        };

        const parseTokenFromUrl = () => {
            const path = window.location.pathname || "";
            const m = path.match(/^\/tools\/grocery\/([^\/?#]+)/);
            if (m && m[1]) return decodeURIComponent(m[1]);
            const params = new URLSearchParams(window.location.search);
            const from = params.get("from");
            return from ? from.trim() : "";
        };

        const state = {
            token: parseTokenFromUrl(),
            shareUrl: "",
            payload: { name: (nameField?.value || "").trim() || "Groceries", items: [] },
            saving: false,
            saveTimer: null,
            lastSent: "",
        };

        const categories = ["Produce", "Dairy", "Meat", "Pantry", "Frozen", "Drinks", "Snacks", "Household", "Other"];

        const shareBox = document.createElement("div");
        shareBox.className = "gl-share";
        shareBox.style.display = "none";
        const shareLabel = document.createElement("div");
        shareLabel.className = "gl-share-label";
        shareLabel.textContent = "Share link";
        const shareRow = document.createElement("div");
        shareRow.className = "gl-share-row";
        const shareInput = document.createElement("input");
        shareInput.type = "text";
        shareInput.readOnly = true;
        shareInput.className = "gl-share-input";
        shareInput.value = "";
        const shareOpen = document.createElement("button");
        shareOpen.type = "button";
        shareOpen.className = "tool-copy-btn gl-open";
        shareOpen.textContent = "Open";
        const shareCopy = document.createElement("button");
        shareCopy.type = "button";
        shareCopy.className = "tool-run-btn gl-copy";
        shareCopy.textContent = "Copy";
        shareOpen.disabled = true;
        shareCopy.disabled = true;
        const shareFeedback = document.createElement("div");
        shareFeedback.className = "gl-share-feedback";
        shareFeedback.setAttribute("aria-live", "polite");
        let shareCopyTimer = null;
        shareOpen.onclick = () => {
            if (!state.shareUrl) return;
            window.open(state.shareUrl, "_blank", "noopener,noreferrer");
        };
        shareCopy.onclick = async () => {
            if (!shareInput.value) return;
            if (shareCopyTimer) clearTimeout(shareCopyTimer);
            try {
                await navigator.clipboard.writeText(shareInput.value);
                shareCopy.textContent = "Copied";
                shareFeedback.textContent = "Copied";
                shareCopyTimer = setTimeout(() => {
                    shareCopy.textContent = "Copy";
                    shareFeedback.textContent = "";
                }, 1500);
            } catch (e) {
                shareCopy.textContent = "Copy failed";
                shareFeedback.textContent = "Copy failed";
                shareCopyTimer = setTimeout(() => {
                    shareCopy.textContent = "Copy";
                    shareFeedback.textContent = "";
                }, 1500);
            }
        };
        shareRow.append(shareInput, shareOpen, shareCopy);
        const shareHelp = document.createElement("div");
        shareHelp.className = "gl-share-help";
        shareHelp.textContent = "Anyone with this link can add and check items.";
        shareBox.append(shareLabel, shareRow, shareHelp, shareFeedback);

        const controls = document.createElement("div");
        controls.className = "gl-controls";

        const createBtn = document.createElement("button");
        createBtn.type = "button";
        createBtn.className = "tool-run-btn";
        createBtn.textContent = "Create list";

        const createNote = document.createElement("div");
        createNote.className = "gl-note";
        createNote.textContent = "Create a shared link so anyone with it can add/check items.";

        const addRow = document.createElement("div");
        addRow.className = "gl-add-row";
        const addInput = document.createElement("input");
        addInput.type = "text";
        addInput.className = "gl-add-input";
        addInput.placeholder = "Add an item (e.g., milk, bananas)…";
        addInput.maxLength = 80;
        const addBtn = document.createElement("button");
        addBtn.type = "button";
        addBtn.className = "tool-run-btn";
        addBtn.textContent = "Add";
        addRow.append(addInput, addBtn);

        const actions = document.createElement("div");
        actions.className = "gl-actions";
        const clearCheckedBtn = document.createElement("button");
        clearCheckedBtn.type = "button";
        clearCheckedBtn.className = "tool-clear-btn";
        clearCheckedBtn.textContent = "Clear checked";
        actions.append(clearCheckedBtn);

        const listWrap = document.createElement("div");
        listWrap.className = "gl-list";

        function setShare(url) {
            const normalized = toAbsoluteUrl(url);
            state.shareUrl = normalized;
            shareInput.value = normalized;
            shareInput.title = normalized || "Share link";
            shareBox.style.display = normalized ? "block" : "none";
            shareOpen.disabled = !normalized;
            shareCopy.disabled = !normalized;
            shareFeedback.textContent = "";
            if (shareCopyTimer) {
                clearTimeout(shareCopyTimer);
                shareCopyTimer = null;
            }
            shareCopy.textContent = "Copy";
        }

        function setToken(token) {
            state.token = token || "";
            if (state.token) {
                // Keep the editable page shareable by query param
                const params = new URLSearchParams(window.location.search);
                if (window.location.pathname === "/tools/grocery-list") {
                    params.set("from", state.token);
                    const next = `${window.location.pathname}?${params.toString()}`;
                    window.history.replaceState({}, "", next);
                }
            }
        }

        function render() {
            if (nameField) {
                const current = (nameField.value || "").trim();
                if (!current && state.payload.name) nameField.value = state.payload.name;
            }

            listWrap.innerHTML = "";
            const items = Array.isArray(state.payload.items) ? state.payload.items : [];

            const byCat = new Map();
            categories.forEach((c) => byCat.set(c, []));
            items.forEach((it) => {
                const cat = categories.includes(it.category) ? it.category : "Other";
                if (!byCat.has(cat)) byCat.set(cat, []);
                byCat.get(cat).push(it);
            });

            categories.forEach((cat) => {
                const catItems = byCat.get(cat) || [];
                if (!catItems.length) return;
                const section = document.createElement("div");
                section.className = "gl-section";
                const head = document.createElement("div");
                head.className = "gl-section-head";
                head.textContent = cat;
                section.appendChild(head);

                catItems.forEach((it) => {
                    const row = document.createElement("div");
                    row.className = "gl-item";
                    const check = document.createElement("input");
                    check.type = "checkbox";
                    check.checked = !!it.checked;
                    check.className = "gl-check";
                    check.onchange = () => {
                        it.checked = check.checked;
                        scheduleSave();
                        render();
                    };
                    const text = document.createElement("div");
                    text.className = "gl-text";
                    text.textContent = it.text || "";
                    if (it.checked) row.classList.add("checked");
                    const del = document.createElement("button");
                    del.type = "button";
                    del.className = "gl-del";
                    del.textContent = "✕";
                    del.onclick = () => {
                        state.payload.items = (state.payload.items || []).filter((x) => x.id !== it.id);
                        scheduleSave();
                        render();
                    };
                    row.append(check, text, del);
                    section.appendChild(row);
                });

                listWrap.appendChild(section);
            });

            if (!items.length) {
                const empty = document.createElement("div");
                empty.className = "gl-empty";
                empty.textContent = state.token ? "Add your first item." : "Create a list to start adding items.";
                listWrap.appendChild(empty);
            }
        }

        async function api(action, extra = {}) {
            const input = {
                action,
                list_name: nameField ? (nameField.value || "").trim() : state.payload.name,
                ...extra,
            };
            return await runToolRequest("grocery-list", input);
        }

        async function load() {
            if (!state.token) {
                setShare("");
                render();
                return;
            }
            setStatus("Loading…");
            const res = await api("get", { token: state.token });
            if (!res.ok) {
                setStatus(res?.error?.message || "Unable to load list.", "error");
                return;
            }
            const out = res.data?.output || {};
            setToken(out.token || state.token);
            setShare(out.share_url || "");
            state.payload = out.payload || state.payload;
            if (nameField) nameField.value = state.payload.name || nameField.value;
            setStatus("Synced.", "success");
            render();
        }

        async function createList() {
            setStatus("Creating…");
            createBtn.disabled = true;
            const res = await api("create", {});
            createBtn.disabled = false;
            if (!res.ok) {
                setStatus(res?.error?.message || "Unable to create list.", "error");
                return;
            }
            const out = res.data?.output || {};
            setToken(out.token || "");
            setShare(out.share_url || "");
            state.payload = out.payload || state.payload;
            if (nameField) nameField.value = state.payload.name || nameField.value;
            setStatus("List created.", "success");
            render();
        }

        function normalizeItemsForUpdate() {
            const items = Array.isArray(state.payload.items) ? state.payload.items : [];
            return items
                .map((it) => ({
                    id: it.id,
                    text: String(it.text || "").trim(),
                    checked: !!it.checked,
                }))
                .filter((it) => it.text);
        }

        function scheduleSave() {
            if (!state.token) return;
            if (state.saveTimer) clearTimeout(state.saveTimer);
            state.saveTimer = setTimeout(saveNow, 500);
        }

        async function saveNow() {
            if (!state.token) return;
            const name = nameField ? (nameField.value || "").trim() : state.payload.name;
            const items = normalizeItemsForUpdate();
            const snapshot = JSON.stringify({ name, items });
            if (snapshot === state.lastSent) return;
            state.lastSent = snapshot;
            setStatus("Saving…");
            const res = await api("update", { token: state.token, name, items });
            if (!res.ok) {
                setStatus(res?.error?.message || "Save failed.", "error");
                return;
            }
            const out = res.data?.output || {};
            state.payload = out.payload || state.payload;
            setShare(out.share_url || state.shareUrl);
            if (nameField) nameField.value = state.payload.name || nameField.value;
            setStatus("Saved.", "success");
            render();
        }

        function addItem() {
            const text = (addInput.value || "").trim();
            if (!text) return;
            if (!state.token) {
                setStatus("Create the list first to get a share link.", "error");
                return;
            }
            const item = {
                id: makeId(),
                text,
                category: "Other",
                checked: false,
                created_at: new Date().toISOString(),
            };
            state.payload.items = [item, ...(state.payload.items || [])];
            addInput.value = "";
            scheduleSave();
            render();
        }

        addBtn.onclick = addItem;
        addInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                e.preventDefault();
                addItem();
            }
        });
        clearCheckedBtn.onclick = () => {
            if (!state.token) return;
            state.payload.items = (state.payload.items || []).filter((it) => !it.checked);
            scheduleSave();
            render();
        };
        createBtn.onclick = createList;

        if (nameField) {
            nameField.addEventListener("input", () => {
                state.payload.name = (nameField.value || "").trim();
                scheduleSave();
            });
        }

        // Controls visibility
        controls.append(createNote, createBtn);

        custom.append(shareBox, controls, addRow, actions, listWrap);

        // If already have token, hide create CTA
        if (state.token) {
            createBtn.style.display = "none";
            createNote.style.display = "none";
        }

        load();
        render();
    }

    // ----- Countdown (local + optional share) -----
    function loadCountdownStore() {
        const raw = localStorage.getItem(COUNTDOWN_KEY);
        if (!raw) return { version: 1, countdowns: [] };
        try {
            const parsed = JSON.parse(raw);
            if (!parsed || typeof parsed !== "object") return { version: 1, countdowns: [] };
            const countdowns = Array.isArray(parsed.countdowns) ? parsed.countdowns : [];
            return { version: 1, countdowns };
        } catch (e) {
            return { version: 1, countdowns: [] };
        }
    }

    function saveCountdownStore(store) {
        localStorage.setItem(COUNTDOWN_KEY, JSON.stringify(store));
    }

    function parseYyyyMmDd(dateStr) {
        const s = String(dateStr || "").trim();
        const m = s.match(/^(\d{4})-(\d{2})-(\d{2})$/);
        if (!m) return null;
        const y = Number(m[1]);
        const mo = Number(m[2]);
        const d = Number(m[3]);
        if (!y || mo < 1 || mo > 12 || d < 1 || d > 31) return null;
        return { y, mo, d, s };
    }

    function getTimeZoneOffsetMs(timeZone, date) {
        const dtf = new Intl.DateTimeFormat("en-US", {
            timeZone,
            hour12: false,
            year: "numeric",
            month: "2-digit",
            day: "2-digit",
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
        });
        const parts = dtf.formatToParts(date);
        const lookup = {};
        parts.forEach((p) => (lookup[p.type] = p.value));
        const asUtc = Date.parse(
            `${lookup.year}-${lookup.month}-${lookup.day}T${lookup.hour}:${lookup.minute}:${lookup.second}Z`
        );
        return asUtc - date.getTime();
    }

    function eventDateToUtcMs(dateStr, tz) {
        const parsed = parseYyyyMmDd(dateStr);
        if (!parsed) return null;
        const { y, mo, d } = parsed;
        if (tz === "UTC") {
            return Date.parse(`${parsed.s}T00:00:00Z`);
        }
        if (!tz || tz === "Local") {
            const local = new Date(`${parsed.s}T00:00:00`);
            return local.getTime();
        }
        // Use midday offset to avoid DST boundary surprises
        const noonUtc = new Date(Date.UTC(y, mo - 1, d, 12, 0, 0));
        const offset = getTimeZoneOffsetMs(tz, noonUtc);
        const wallMidnightAsUtc = Date.UTC(y, mo - 1, d, 0, 0, 0);
        return wallMidnightAsUtc - offset;
    }

    function computeCountdownParts(targetUtcMs, nowMs) {
        const diffMs = targetUtcMs - nowMs;
        if (!isFinite(diffMs) || diffMs <= 0) {
            return { isPast: true, days: 0, hours: 0, minutes: 0, diffMs: diffMs || 0 };
        }
        const totalMinutes = Math.floor(diffMs / 60000);
        const minutes = totalMinutes % 60;
        const totalHours = Math.floor(totalMinutes / 60);
        const hours = totalHours % 24;
        const days = Math.floor(totalHours / 24);
        return { isPast: false, days, hours, minutes, diffMs };
    }

    function initCountdown(form, outputEl, statusEl) {
        const slug = form?.dataset?.toolSlug;
        if (slug !== "countdown") return;

        const runBtn = form.querySelector("[data-run-button]") || form.querySelector("button[type='submit']");
        if (runBtn) runBtn.style.display = "none";

        if (outputEl) outputEl.style.display = "none";
        const outputPre = outputEl ? outputEl.querySelector(".tool-output-pre") : null;
        if (outputPre) outputPre.textContent = "";

        const custom = document.getElementById("tool-custom-ui");
        if (!custom) return;
        custom.innerHTML = "";

        const nameEl = form.querySelector('[name="event_name"]');
        const dateEl = form.querySelector('[name="event_date"]');
        const tzEl = form.querySelector('[name="timezone"]');
        if (dateEl) {
            try {
                dateEl.type = "date";
            } catch (e) {
                // ignore
            }
        }

        const setStatus = (text, type = "") => {
            if (!statusEl) return;
            statusEl.textContent = text || "";
            statusEl.className = "tool-status";
            if (type) statusEl.classList.add(type);
        };

        const parseTokenFromUrl = () => {
            const path = window.location.pathname || "";
            const m = path.match(/^\/tools\/countdown\/([^\/?#]+)/);
            if (m && m[1]) return decodeURIComponent(m[1]);
            const params = new URLSearchParams(window.location.search);
            const from = params.get("from");
            return from ? from.trim() : "";
        };

        const state = {
            activeId: null,
            shareToken: parseTokenFromUrl(),
            shareUrl: "",
            sharedPayload: null,
            tickTimer: null,
        };
        const initialShareUrl = (form?.dataset?.shareInitial || (typeof window !== "undefined" ? window.SHARE_URL_INITIAL : "") || "").trim();
        if (form && form.dataset.sharedPayload) {
            try {
                state.sharedPayload = JSON.parse(form.dataset.sharedPayload);
            } catch (_) {
                state.sharedPayload = null;
            }
        }
        if (!state.sharedPayload && form && form.dataset.savedInput) {
            try {
                state.sharedPayload = JSON.parse(form.dataset.savedInput);
            } catch (_) {
                // ignore
            }
        }
        if (!state.sharedPayload && typeof window !== "undefined" && window.SHARED_PAYLOAD) {
            state.sharedPayload = window.SHARED_PAYLOAD;
        }

        const shareBox = document.createElement("div");
        shareBox.className = "cd-share";
        shareBox.style.display = "none";
        const shareLabel = document.createElement("div");
        shareLabel.className = "cd-share-label";
        shareLabel.textContent = "Share link";
        const shareRow = document.createElement("div");
        shareRow.className = "cd-share-row";
        const shareInput = document.createElement("input");
        shareInput.type = "text";
        shareInput.readOnly = true;
        shareInput.className = "cd-share-input";
        const shareCopy = document.createElement("button");
        shareCopy.type = "button";
        shareCopy.className = "tool-run-btn";
        shareCopy.textContent = "Copy";
        shareCopy.onclick = async () => {
            if (!shareInput.value) return;
            try {
                await navigator.clipboard.writeText(shareInput.value);
                shareCopy.textContent = "Copied";
                setTimeout(() => (shareCopy.textContent = "Copy"), 1200);
            } catch (e) {
                shareCopy.textContent = "Copy failed";
                setTimeout(() => (shareCopy.textContent = "Copy"), 1200);
            }
        };
        shareRow.append(shareInput, shareCopy);
        shareBox.append(shareLabel, shareRow);

        const card = document.createElement("div");
        card.className = "cd-card";
        const cardTitle = document.createElement("div");
        cardTitle.className = "cd-title";
        const cardBig = document.createElement("div");
        cardBig.className = "cd-big";
        const cardSmall = document.createElement("div");
        cardSmall.className = "cd-small";
        card.append(cardTitle, cardBig, cardSmall);

        const actions = document.createElement("div");
        actions.className = "cd-actions";
        const saveBtn = document.createElement("button");
        saveBtn.type = "button";
        saveBtn.className = "tool-run-btn";
        saveBtn.textContent = "Save locally";
        const shareBtn = document.createElement("button");
        shareBtn.type = "button";
        shareBtn.className = "tool-copy-btn";
        shareBtn.style.display = "inline-block";
        shareBtn.textContent = "Create share link";
        const clearBtn = document.createElement("button");
        clearBtn.type = "button";
        clearBtn.className = "tool-clear-btn";
        clearBtn.textContent = "Clear";
        actions.append(saveBtn, shareBtn, clearBtn);

        const listWrap = document.createElement("div");
        listWrap.className = "cd-list";
        const listTitle = document.createElement("div");
        listTitle.className = "cd-section-title";
        listTitle.textContent = "Saved countdowns";
        const list = document.createElement("div");
        list.className = "cd-items";
        listWrap.append(listTitle, list);

        function setShare(url) {
            const abs = toAbsoluteUrl(url);
            state.shareUrl = abs;
            shareInput.value = abs;
            shareBox.style.display = abs ? "block" : "none";
            shareCopy.disabled = !abs;
        }

        if (initialShareUrl) {
            setShare(initialShareUrl);
        } else if (state.shareToken) {
            setShare(toAbsoluteUrl(window.location.pathname));
        }

        function applySharedPayload() {
            if (!state.sharedPayload) return;
            if (nameEl) nameEl.value = state.sharedPayload.name || "";
            if (dateEl) {
                const raw = state.sharedPayload.date || "";
                const cleaned = raw.includes("/") ? raw.replace(/\//g, "-") : raw;
                dateEl.value = cleaned;
            }
            if (tzEl) tzEl.value = state.sharedPayload.timezone || "Local";
        }

        function getActiveCountdown() {
            if (state.sharedPayload) return state.sharedPayload;
            const store = loadCountdownStore();
            const items = store.countdowns || [];
            if (state.activeId) {
                return items.find((c) => c.id === state.activeId) || null;
            }
            const pinned = items.find((c) => c.pinned);
            return pinned || items[0] || null;
        }

        function renderCard() {
            const active = getActiveCountdown();
            const name = active?.name || (nameEl?.value || "").trim() || "Countdown";
            const date = active?.date || (dateEl?.value || "").trim() || "";
            const tz = active?.timezone || (tzEl?.value || "Local");
            cardTitle.textContent = name;

            const targetMs = eventDateToUtcMs(date, tz);
            if (!date || !targetMs) {
                cardBig.textContent = "—";
                cardSmall.textContent = "Pick a date to start.";
                return;
            }

            const parts = computeCountdownParts(targetMs, Date.now());
            if (parts.isPast) {
                cardBig.textContent = "Today";
                cardSmall.textContent = "It’s here.";
                return;
            }

            cardBig.textContent = `${parts.days} day${parts.days === 1 ? "" : "s"} left`;
            cardSmall.textContent = `${parts.hours}h ${parts.minutes}m • Timezone: ${tz}`;
        }

        function scheduleTick() {
            if (state.tickTimer) clearTimeout(state.tickTimer);
            const now = new Date();
            const msToNextMinute = (60 - now.getSeconds()) * 1000 - now.getMilliseconds() + 5;
            state.tickTimer = setTimeout(() => {
                renderCard();
                scheduleTick();
            }, Math.max(500, msToNextMinute));
        }

        function renderList() {
            const store = loadCountdownStore();
            const items = Array.isArray(store.countdowns) ? store.countdowns : [];
            const sorted = items.slice().sort((a, b) => {
                const ap = a.pinned ? 1 : 0;
                const bp = b.pinned ? 1 : 0;
                if (ap !== bp) return bp - ap;
                return String(a.date || "").localeCompare(String(b.date || ""));
            });

            list.innerHTML = "";
            if (!sorted.length) {
                const empty = document.createElement("div");
                empty.className = "cd-empty";
                empty.textContent = "No saved countdowns yet.";
                list.appendChild(empty);
                return;
            }

            sorted.forEach((c) => {
                const row = document.createElement("div");
                row.className = "cd-item";
                if (c.id === state.activeId) row.classList.add("active");
                const left = document.createElement("div");
                left.className = "cd-item-left";
                const t = document.createElement("div");
                t.className = "cd-item-title";
                t.textContent = c.name;
                const sub = document.createElement("div");
                sub.className = "cd-item-sub";
                const targetMs = eventDateToUtcMs(c.date, c.timezone || "Local");
                const p = targetMs ? computeCountdownParts(targetMs, Date.now()) : null;
                const quick = p ? (p.isPast ? "Today" : `${p.days}d`) : "";
                sub.textContent = `${c.date} • ${c.timezone || "Local"}${quick ? ` • ${quick}` : ""}`;
                left.append(t, sub);

                const btns = document.createElement("div");
                btns.className = "cd-item-actions";
                const view = document.createElement("button");
                view.type = "button";
                view.className = "wl-small";
                view.textContent = "View";
                view.onclick = () => {
                    state.sharedPayload = null;
                    state.activeId = c.id;
                    renderList();
                    renderCard();
                    setShare("");
                    setStatus("");
                };
                const pin = document.createElement("button");
                pin.type = "button";
                pin.className = "wl-small";
                pin.textContent = c.pinned ? "Unpin" : "Pin";
                pin.onclick = () => {
                    const next = loadCountdownStore();
                    next.countdowns = (next.countdowns || []).map((x) =>
                        x.id === c.id ? { ...x, pinned: !x.pinned } : { ...x, pinned: false }
                    );
                    saveCountdownStore(next);
                    setStatus(c.pinned ? "Unpinned." : "Pinned.", "success");
                    renderList();
                };
                const del = document.createElement("button");
                del.type = "button";
                del.className = "wl-small danger";
                del.textContent = "Delete";
                del.onclick = () => {
                    const next = loadCountdownStore();
                    next.countdowns = (next.countdowns || []).filter((x) => x.id !== c.id);
                    saveCountdownStore(next);
                    if (state.activeId === c.id) state.activeId = null;
                    setStatus("Deleted.", "success");
                    renderList();
                    renderCard();
                };
                btns.append(view, pin, del);
                row.append(left, btns);
                list.appendChild(row);
            });
        }

        function saveCurrent() {
            const name = (nameEl?.value || "").trim();
            const date = (dateEl?.value || "").trim();
            const tz = (tzEl?.value || "Local").trim() || "Local";
            if (!name) {
                setStatus("Event name is required.", "error");
                return;
            }
            if (!parseYyyyMmDd(date)) {
                setStatus("Event date must be YYYY-MM-DD.", "error");
                return;
            }
            const store = loadCountdownStore();
            const entry = {
                id: makeId(),
                name,
                date,
                timezone: tz,
                created_at: new Date().toISOString(),
                pinned: false,
            };
            store.countdowns = [entry, ...(store.countdowns || [])];
            saveCountdownStore(store);
            state.sharedPayload = null;
            state.activeId = entry.id;
            setStatus("Saved locally.", "success");
            renderList();
            renderCard();
        }

        async function createShareLink() {
            const active = getActiveCountdown() || {
                name: (nameEl?.value || "").trim(),
                date: (dateEl?.value || "").trim(),
                timezone: (tzEl?.value || "Local").trim() || "Local",
            };
            if (!active.name) {
                setStatus("Event name is required.", "error");
                return;
            }
            if (!parseYyyyMmDd(active.date)) {
                setStatus("Event date must be YYYY-MM-DD.", "error");
                return;
            }
            setStatus("Creating share link…");
            shareBtn.disabled = true;
            let res;
            try {
                res = await runToolRequest("countdown", {
                    action: "create_share",
                    event_name: active.name,
                    event_date: active.date,
                    timezone: active.timezone || "Local",
                });
            } catch (e) {
                setStatus("Couldn't reach the server. Check your connection and try again.", "error");
                shareBtn.disabled = false;
                return;
            }
            shareBtn.disabled = false;
            if (!res.ok) {
                const ref = res?.ref || res?.error?.ref;
                const message = res?.message || res?.error?.message || "Please refresh and try again. If it keeps happening, contact support.";
                const refText = ref ? ` Reference: ${ref}` : "";
                if (res?.error === "PLUS_REQUIRED" || res?.error?.code === "PLUS_REQUIRED") {
                    if (typeof window.openPlusPrompt === "function") {
                        window.openPlusPrompt("Want multiple share links?", "Plus lets you create more than one share link and keep history.");
                    }
                    setStatus("Multiple share links are a Plus feature.", "error");
                    return;
                }
                setStatus(`Could not create share link. ${message}${refText}`, "error");
                return;
            }
            const out = res.data?.output || {};
            setShare(out.share_url || "");
            setStatus("Share link created.", "success");
        }

        async function loadShared() {
            if (!state.shareToken) return;
            if (!state.sharedPayload) {
                setStatus("Loading shared countdown…");
                const res = await runToolRequest("countdown", { action: "get", token: state.shareToken });
                if (!res.ok) {
                    setStatus(res?.error?.message || "Unable to load countdown.", "error");
                    return;
                }
                const out = res.data?.output || {};
                const payload = out.payload || null;
                if (!payload) {
                    setStatus("Countdown not found.", "error");
                    return;
                }
                state.sharedPayload = {
                    name: payload.name,
                    date: payload.date,
                    timezone: payload.timezone || "Local",
                };
                setShare(out.share_url || "");
                setStatus("Loaded.", "success");
            }
            state.activeId = null;
            applySharedPayload();
            if (!state.shareUrl) {
                setShare(initialShareUrl || toAbsoluteUrl(window.location.pathname));
            }
            renderList();
            renderCard();
        }

        saveBtn.onclick = saveCurrent;
        shareBtn.onclick = createShareLink;
        clearBtn.onclick = () => {
            if (nameEl) nameEl.value = "";
            if (dateEl) dateEl.value = "";
            if (tzEl) tzEl.value = "Local";
            state.sharedPayload = null;
            state.activeId = null;
            setShare("");
            setStatus("Cleared.", "success");
            renderCard();
        };

        const liveInputs = [nameEl, dateEl, tzEl].filter(Boolean);
        liveInputs.forEach((el) => el.addEventListener("input", renderCard));
        liveInputs.forEach((el) => el.addEventListener("change", renderCard));

        // Prevent default submit; this tool uses local save and optional share button
        form.addEventListener("submit", (e) => e.preventDefault());

        custom.append(shareBox, card, actions, listWrap);
        renderList();
        applySharedPayload();
        renderCard();
        scheduleTick();
        loadShared();
    }

    function formatDate(date) {
        const yyyy = date.getFullYear();
        const mm = String(date.getMonth() + 1).padStart(2, "0");
        const dd = String(date.getDate()).padStart(2, "0");
        return `${yyyy}-${mm}-${dd}`;
    }

    function getLastNDates(n, anchorDate) {
        const dates = [];
        for (let i = 0; i < n; i++) {
            const d = new Date(anchorDate);
            d.setDate(d.getDate() - i);
            dates.push(formatDate(d));
        }
        return dates;
    }

    function isConsecutive(prevDate, currDate) {
        const prev = new Date(prevDate);
        const curr = new Date(currDate);
        const diff = (curr - prev) / (1000 * 60 * 60 * 24);
        return diff === 1;
    }

    function computeStreaks(entries) {
        const dates = Object.keys(entries).sort();
        let currentStreak = 0;
        let bestStreak = 0;
        let prevDate = null;

        for (const date of dates) {
            const status = entries[date];
            if (status !== "Done") {
                prevDate = date;
                continue;
            }
            if (prevDate && isConsecutive(prevDate, date)) {
                currentStreak += 1;
            } else {
                currentStreak = 1;
            }
            bestStreak = Math.max(bestStreak, currentStreak);
            prevDate = date;
        }

        return { bestStreak };
    }

    function computeCurrentStreak(entries) {
        const today = formatDate(new Date());
        const yesterday = formatDate(new Date(new Date().setDate(new Date().getDate() - 1)));
        const todayStatus = entries[today];
        const startDate = todayStatus === "Done" ? today : yesterday;

        if (!startDate || entries[startDate] !== "Done") return 0;

        let streak = 0;
        let cursor = new Date(startDate);
        while (true) {
            const dateStr = formatDate(cursor);
            if (entries[dateStr] === "Done") {
                streak += 1;
                cursor.setDate(cursor.getDate() - 1);
            } else {
                break;
            }
        }
        return streak;
    }

    function renderDailyCheckinOutput(outputPre, statusEl, habitName, status, entries) {
        const last7 = getLastNDates(7, new Date());
        const lines = [];
        lines.push(`Habit: ${habitName}`);
        lines.push(`Today: ${status}`);

        const currentStreak = computeCurrentStreak(entries);
        const { bestStreak } = computeStreaks(entries);
        lines.push(`Current streak: ${currentStreak}`);
        lines.push(`Best streak: ${bestStreak}`);
        lines.push("Last 7 days:");
        last7.forEach((date) => {
            const val = entries[date] || "—";
            lines.push(`${date}: ${val}`);
        });

        if (outputPre) outputPre.textContent = lines.join("\n");
        if (statusEl) {
            statusEl.textContent = "Saved locally.";
            statusEl.className = "tool-status success";
        }
    }

    function handleDailyCheckin(form, statusEl, outputPre, clearBtn) {
        const habitInput = form.querySelector('[name="habit_name"]');
        const statusInput = form.querySelector('[name="status"]');
        if (!habitInput || !statusInput) return;

        const updateForHabit = () => {
            const habitName = habitInput.value.trim();
            if (!habitName) {
                if (outputPre) outputPre.textContent = "";
                if (statusEl) {
                    statusEl.textContent = "Enter a habit name to view history.";
                    statusEl.className = "tool-status";
                }
                return;
            }
            const stored = getStoredHabit(habitName);
            renderDailyCheckinOutput(outputPre, statusEl, habitName, statusInput.value || "Done", stored.entries || {});
        };

        habitInput.addEventListener("input", updateForHabit);
        statusInput.addEventListener("change", updateForHabit);

        if (clearBtn) {
            clearBtn.style.display = "inline-block";
            clearBtn.addEventListener("click", () => {
                const habitName = habitInput.value.trim();
                if (!habitName) return;
                clearHabit(habitName);
                const stored = getStoredHabit(habitName);
                renderDailyCheckinOutput(outputPre, statusEl, habitName, statusInput.value || "Done", stored.entries || {});
            });
        }

        form.addEventListener("submit", (event) => {
            event.preventDefault();
            const habitName = habitInput.value.trim();
            const statusVal = statusInput.value || "";
            if (!habitName) {
                if (statusEl) {
                    statusEl.textContent = "Please enter a habit name.";
                    statusEl.className = "tool-status error";
                }
                return;
            }
            if (!statusVal) {
                if (statusEl) {
                    statusEl.textContent = "Please select a status.";
                    statusEl.className = "tool-status error";
                }
                return;
            }

            const stored = getStoredHabit(habitName);
            const entries = stored.entries || {};
            const today = formatDate(new Date());
            entries[today] = statusVal;
            saveHabit(habitName, { entries });
            renderDailyCheckinOutput(outputPre, statusEl, habitName, statusVal, entries);
        });

        updateForHabit();
    }

    // ---- Trip Planner (structured UI) ----
    const defaultBudgetCategories = ["Flight", "Lodging", "Food", "Activities", "Transport", "Other"];

    function createEl(tag, opts = {}) {
        const el = document.createElement(tag);
        if (opts.className) el.className = opts.className;
        if (opts.text) el.textContent = opts.text;
        if (opts.type) el.type = opts.type;
        if (opts.placeholder) el.placeholder = opts.placeholder;
        if (opts.value !== undefined) el.value = opts.value;
        return el;
    }

    function computeTripSummary(state) {
        const balances = {};
        state.people.forEach((p) => {
            balances[p] = { paid: 0, owes: 0 };
        });

        state.expensesPaid.forEach((exp) => {
            const shareList = exp.split_with && exp.split_with.length ? exp.split_with : state.people.slice();
            const share = exp.amount / shareList.length;
            if (balances[exp.payer]) balances[exp.payer].paid += exp.amount;
            shareList.forEach((p) => {
                if (balances[p]) balances[p].owes += share;
            });
        });

        const perPerson = Object.entries(balances).map(([name, data]) => {
            const paid = round2(data.paid);
            const owes = round2(data.owes);
            return { name, paid, owes, net: round2(paid - owes) };
        });

        // Settlement
        const debtors = [];
        const creditors = [];
        perPerson.forEach((p) => {
            if (Math.abs(p.net) < 0.01) return;
            if (p.net < 0) debtors.push([p.name, -p.net]);
            else creditors.push([p.name, p.net]);
        });
        debtors.sort((a, b) => a[1] - b[1]);
        creditors.sort((a, b) => b[1] - a[1]);
        const settlements = [];
        let i = 0,
            j = 0;
        while (i < debtors.length && j < creditors.length) {
            const pay = Math.min(debtors[i][1], creditors[j][1]);
            settlements.push([debtors[i][0], creditors[j][0], round2(pay)]);
            debtors[i][1] = round2(debtors[i][1] - pay);
            creditors[j][1] = round2(creditors[j][1] - pay);
            if (debtors[i][1] <= 0.01) i++;
            if (creditors[j][1] <= 0.01) j++;
        }

        const spentByCat = {};
        state.expensesPaid.forEach((e) => {
            spentByCat[e.category] = (spentByCat[e.category] || 0) + e.amount;
        });
        const plannedByCat = {};
        state.itemsPlanned.forEach((i) => {
            plannedByCat[i.category] = (plannedByCat[i.category] || 0) + i.amount;
        });
        const budgetMap = {};
        state.budgets.forEach((b) => {
            budgetMap[b.category] = (budgetMap[b.category] || 0) + b.amount;
        });
        const cats = new Set([
            ...Object.keys(spentByCat),
            ...Object.keys(plannedByCat),
            ...Object.keys(budgetMap),
        ]);
        const budgetSummary = [];
        cats.forEach((cat) => {
            const planned = round2(budgetMap[cat] || 0);
            const spent = round2(spentByCat[cat] || 0);
            const upcoming = round2(plannedByCat[cat] || 0);
            budgetSummary.push({
                category: cat,
                planned,
                spent,
                upcoming,
                remaining: round2(planned - (spent + upcoming)),
            });
        });

        return { perPerson, settlements, budgetSummary };
    }

    function round2(num) {
        return Math.round((num + Number.EPSILON) * 100) / 100;
    }

    function initTripPlanner(form) {
        const custom = document.getElementById("tool-custom-ui");
        if (!custom) return;
        const statusEl = document.querySelector("[data-tool-status]");
        const outputEl = document.querySelector("[data-tool-output]");
        const outputPre = outputEl ? outputEl.querySelector(".tool-output-pre") : null;
        const shareBlock = outputEl ? outputEl.querySelector("[data-share-block]") : null;
        const shareUrlEl = outputEl ? outputEl.querySelector("[data-share-url]") : null;
        const copyShareBtn = outputEl ? outputEl.querySelector("[data-copy-share]") : null;

        const defaults = defaultBudgetCategories.map((c) => ({ category: c, amount: 0 }));

        const state = {
            people: [],
            budgets: defaults.slice(),
            expensesPaid: [],
            itemsPlanned: [],
        };
        window.tripPlannerState = state;

        const peopleSection = createEl("div", { className: "tp-section" });
        const peopleHeader = createEl("h3", { text: "People" });
        const peopleRow = createEl("div", { className: "tp-inline" });
        const peopleInput = createEl("input", { type: "text", placeholder: "Add person" });
        const peopleAdd = createEl("button", { type: "button", className: "tool-run-btn tp-add-btn" });
        peopleAdd.textContent = "Add";
        peopleRow.append(peopleInput, peopleAdd);
        const chipsWrap = createEl("div", { className: "chip-row" });
        peopleSection.append(peopleHeader, peopleRow, chipsWrap);

        const budgetSection = createEl("div", { className: "tp-section" });
        budgetSection.append(createEl("h3", { text: "Budgets" }));
        const budgetList = createEl("div", { className: "share-table" });
        const addBudgetBtn = createEl("button", { type: "button", className: "tool-run-btn" });
        addBudgetBtn.textContent = "Add category";
        addBudgetBtn.style.width = "auto";
        budgetSection.append(budgetList, addBudgetBtn);

        const expensesSection = createEl("div", { className: "tp-section" });
        expensesSection.append(createEl("h3", { text: "Paid expenses" }));
        const expensesList = createEl("div", { className: "share-sections" });
        const addExpenseBtn = createEl("button", { type: "button", className: "tool-run-btn" });
        addExpenseBtn.textContent = "Add expense";
        addExpenseBtn.style.width = "auto";
        expensesSection.append(expensesList, addExpenseBtn);

        const plannedSection = createEl("div", { className: "tp-section" });
        plannedSection.append(createEl("h3", { text: "Planned items" }));
        plannedSection.append(createEl("p", { className: "muted", text: "Track what isn’t booked yet—estimate cost, date, and who’s on point." }));
        const plannedList = createEl("div", { className: "share-sections" });
        const addPlannedBtn = createEl("button", { type: "button", className: "tool-run-btn" });
        addPlannedBtn.textContent = "Add planned item";
        addPlannedBtn.style.width = "auto";
        plannedSection.append(plannedList, addPlannedBtn);

        custom.append(peopleSection, budgetSection, expensesSection, plannedSection);

        function renderPeople() {
            chipsWrap.innerHTML = "";
            state.people.forEach((p, idx) => {
                const chip = createEl("span", { className: "chip" });
                chip.textContent = p + " ✕";
                chip.style.cursor = "pointer";
                chip.onclick = () => {
                    state.people.splice(idx, 1);
                    renderPeople();
                    renderExpenses();
                    renderPlanned();
                    renderOutput();
                };
                chipsWrap.append(chip);
            });
            renderBudgets();
            renderExpenses();
            renderPlanned();
            renderOutput();
        }

        peopleAdd.onclick = () => {
            const name = peopleInput.value.trim();
            if (!name) return;
            if (!state.people.includes(name)) {
                state.people.push(name);
                renderPeople();
            }
            peopleInput.value = "";
        };

        function renderBudgets() {
            budgetList.innerHTML = "";
            state.budgets.forEach((b, idx) => {
                const row = createEl("div", { className: "share-row" });
                const catInput = createEl("input", { type: "text", value: b.category });
                const amtInput = createEl("input", { type: "number", value: b.amount });
                amtInput.step = "0.01";
                const remove = createEl("button", { type: "button", className: "tp-remove-btn" });
                remove.textContent = "✕";
                remove.onclick = () => {
                    state.budgets.splice(idx, 1);
                    renderBudgets();
                    renderExpenses();
                    renderOutput();
                };
                catInput.oninput = () => {
                    state.budgets[idx].category = catInput.value.trim() || "Other";
                    renderExpenses();
                };
                amtInput.oninput = () => {
                    state.budgets[idx].amount = parseFloat(amtInput.value || "0") || 0;
                    renderOutput();
                };
                row.append(catInput, amtInput, remove);
                budgetList.append(row);
            });
        }

        addBudgetBtn.onclick = () => {
            state.budgets.push({ category: "New", amount: 0 });
            renderBudgets();
        };

        function renderExpenses() {
            expensesList.innerHTML = "";
            // Header
            const header = createEl("div", { className: "share-row share-head tp-exp-grid" });
            ["Payer", "Amount", "Category", "Description", "Split with", ""].forEach((h) => {
                const span = createEl("span", { text: h });
                header.append(span);
            });
            expensesList.append(header);

            state.expensesPaid.forEach((exp, idx) => {
                const wrap = createEl("div", { className: "share-row tp-exp-grid" });
                const payer = createEl("select");
                state.people.forEach((p) => {
                    const opt = createEl("option", { text: p });
                    opt.value = p;
                    if (p === exp.payer) opt.selected = true;
                    payer.append(opt);
                });
                payer.onchange = () => {
                    exp.payer = payer.value;
                    renderOutput();
                };
                const amount = createEl("input", { type: "number", value: exp.amount || 0 });
                amount.step = "0.01";
                amount.min = "0";
                amount.oninput = () => {
                    exp.amount = parseFloat(amount.value || "0") || 0;
                    renderOutput();
                };
                const cat = createEl("select");
                state.budgets.forEach((b) => {
                    const opt = createEl("option", { text: b.category });
                    opt.value = b.category;
                    if (b.category === exp.category) opt.selected = true;
                    cat.append(opt);
                });
                cat.onchange = () => {
                    exp.category = cat.value;
                    renderOutput();
                };
                const desc = createEl("input", { type: "text", value: exp.description || "", placeholder: "What was this?" });
                desc.oninput = () => (exp.description = desc.value);
                const splitWrap = createEl("div");
                state.people.forEach((p) => {
                    const lbl = createEl("label");
                    const cb = createEl("input", { type: "checkbox" });
                    cb.checked = !exp.split_with || exp.split_with.includes(p);
                    cb.onchange = () => {
                        const list = exp.split_with || [];
                        if (cb.checked) {
                            if (!list.includes(p)) list.push(p);
                        } else {
                            exp.split_with = list.filter((n) => n !== p);
                        }
                        exp.split_with = list.filter((n) => n);
                        renderOutput();
                    };
                    lbl.append(cb, document.createTextNode(" " + p));
                    splitWrap.append(lbl, document.createElement("br"));
                });
                const remove = createEl("button", { type: "button", className: "tp-remove-btn" });
                remove.textContent = "✕";
                remove.onclick = () => {
                    state.expensesPaid.splice(idx, 1);
                    renderExpenses();
                    renderOutput();
                };
                wrap.append(payer, amount, cat, desc, splitWrap, remove);
                expensesList.append(wrap);
            });
        }

        addExpenseBtn.onclick = () => {
            if (!state.people.length) {
                if (statusEl) {
                    statusEl.textContent = "Add at least one person first.";
                    statusEl.className = "tool-status error";
                }
                return;
            }
            state.expensesPaid.push({
                payer: state.people[0],
                amount: 0,
                category: state.budgets[0]?.category || "Other",
                description: "",
                split_with: state.people.slice(),
            });
            renderExpenses();
            renderOutput();
        };

        function renderPlanned() {
            plannedList.innerHTML = "";
            // Header
            const header = createEl("div", { className: "share-row share-head tp-grid" });
            ["Category", "Estimate", "Description", "Date", "Assigned to", ""].forEach((h) => {
                const span = createEl("span", { text: h });
                header.append(span);
            });
            plannedList.append(header);
            state.itemsPlanned.forEach((item, idx) => {
                const wrap = createEl("div", { className: "share-row tp-grid" });
                const cat = createEl("select");
                state.budgets.forEach((b) => {
                    const opt = createEl("option", { text: b.category });
                    opt.value = b.category;
                    if (b.category === item.category) opt.selected = true;
                    cat.append(opt);
                });
                cat.onchange = () => {
                    item.category = cat.value;
                    renderOutput();
                };
                const amount = createEl("input", { type: "number", value: item.amount || 0, placeholder: "Estimate" });
                amount.step = "0.01";
        amount.min = "0";
                amount.oninput = () => {
                    item.amount = parseFloat(amount.value || "0") || 0;
                    renderOutput();
                };
                const desc = createEl("input", { type: "text", value: item.description || "", placeholder: "What’s left to book?" });
                desc.oninput = () => (item.description = desc.value);
                const due = createEl("input", { type: "date", value: item.due_date || "" });
                due.oninput = () => (item.due_date = due.value);
                const assign = createEl("select");
                const noneOpt = createEl("option", { text: "Unassigned" });
                noneOpt.value = "";
                assign.append(noneOpt);
                state.people.forEach((p) => {
                    const opt = createEl("option", { text: p });
                    opt.value = p;
                    if (p === item.assigned_to) opt.selected = true;
                    assign.append(opt);
                });
                assign.onchange = () => (item.assigned_to = assign.value || null);
                const remove = createEl("button", { type: "button", className: "tp-remove-btn" });
                remove.textContent = "✕";
                remove.onclick = () => {
                    state.itemsPlanned.splice(idx, 1);
                    renderPlanned();
                    renderOutput();
                };
                wrap.append(cat, amount, desc, due, assign, remove);
                plannedList.append(wrap);
            });
        }

        addPlannedBtn.onclick = () => {
            state.itemsPlanned.push({
                category: state.budgets[0]?.category || "Other",
                amount: 0,
                description: "",
                due_date: "",
                assigned_to: null,
            });
            renderPlanned();
        };

        function renderOutput() {
            const summary = computeTripSummary(state);
            const lines = [];
            const tripName = form.querySelector('[name="trip_name"]')?.value || "Trip";
            lines.push(`Trip: ${tripName}`);
            lines.push(`People: ${state.people.join(", ") || "None"}`);
            lines.push("");
            lines.push("Budgets:");
            if (summary.budgetSummary.length) {
                summary.budgetSummary.forEach((b) => {
                    lines.push(
                        `${b.category}: planned ${b.planned.toFixed(2)}, spent ${b.spent.toFixed(2)}, upcoming ${b.upcoming.toFixed(2)}, remaining ${b.remaining.toFixed(2)}`
                    );
                });
            } else {
                lines.push("None set.");
            }
            lines.push("");
            lines.push("Totals:");
            summary.perPerson.forEach((p) => {
                lines.push(`${p.name}: paid ${p.paid.toFixed(2)}, owes ${p.owes.toFixed(2)}, net ${p.net.toFixed(2)}`);
            });
            lines.push("");
            lines.push("Settle:");
            if (summary.settlements.length) {
                summary.settlements.forEach((s) => {
                    lines.push(`${s[0]} pays ${s[1]} ${s[2].toFixed(2)}`);
                });
            } else {
                lines.push("No transfers needed.");
            }
            if (outputPre) outputPre.textContent = lines.join("\n");
        }

        function showShareLocal(url) {
            if (!shareBlock || !shareUrlEl) return;
            const full = url.startsWith("http") ? url : `${window.location.origin}${url}`;
            shareUrlEl.textContent = full;
            shareBlock.style.display = "block";
            if (copyShareBtn) {
                copyShareBtn.onclick = async () => {
                    try {
                        await navigator.clipboard.writeText(full);
                        statusEl.textContent = "Link copied.";
                        statusEl.className = "tool-status success";
                    } catch (e) {
                        statusEl.textContent = "Unable to copy.";
                        statusEl.className = "tool-status error";
                    }
                };
            }
        }

        // Expose payload builder for submit
        window.tripPlannerBuildPayload = () => {
            const tripName = form.querySelector('[name="trip_name"]')?.value.trim();
            const currency = form.querySelector('[name="currency"]')?.value || "USD";
            const notes = form.querySelector('[name="notes"]')?.value || "";
            if (!tripName) return { ok: false, error: "Trip name is required." };
            if (!state.people.length) return { ok: false, error: "Add at least one person." };
            // basic amount validation
            const expensesValid = state.expensesPaid.every((e) => e.payer && e.amount > 0);
            if (!expensesValid) return { ok: false, error: "Expenses need payer and positive amount." };
            return {
                ok: true,
                payload: {
                    trip_name: tripName,
                    currency,
                    notes,
                    people: state.people,
                    budgets: state.budgets,
                    expenses_paid: state.expensesPaid,
                    items_planned: state.itemsPlanned,
                },
            };
        };

        // Prefill from ?from=token
        const urlParams = new URLSearchParams(window.location.search);
        const fromToken = urlParams.get("from");
        async function loadFromToken(token) {
            try {
                const res = await fetch(`/api/tools/trip/${token}`);
                const data = await res.json();
                if (!data.ok) return;
                const p = data.data;
                if (p.people) state.people = p.people;
                if (p.budgets) state.budgets = p.budgets;
                if (p.expenses_paid) state.expensesPaid = p.expenses_paid;
                if (p.items_planned) state.itemsPlanned = p.items_planned;
                const tripInput = form.querySelector('[name="trip_name"]');
                if (tripInput && p.trip_name) tripInput.value = p.trip_name;
                const currencyInput = form.querySelector('[name="currency"]');
                if (currencyInput && p.currency) currencyInput.value = p.currency;
                const notesInput = form.querySelector('[name="notes"]');
                if (notesInput && p.notes) notesInput.value = p.notes;
                renderPeople();
            } catch (e) {
                // ignore
            }
        }

        renderPeople();
        renderBudgets();
        renderExpenses();
        renderPlanned();
        renderOutput();

        if (fromToken) {
            loadFromToken(fromToken);
        }

        // expose for share block updates
        window.tripPlannerShowShare = showShareLocal;
    }
    function serializeForm(form) {
        const fields = form.querySelectorAll("[data-input-field]");
        const payload = {};
        fields.forEach((field) => {
            const name = field.name;
            const type = field.dataset.type || field.type;
            if (!name) return;
            const wrapper = field.closest("[data-depends-field]");
            if (wrapper && wrapper.dataset.conditionalHidden === "true") {
                return;
            }
            if (field.disabled) return;
            if (!name) return;
            let value = field.value || "";
            if (type === "text" || type === "textarea") {
                value = value.trim();
            }
            payload[name] = value;
        });
        return payload;
    }

    function attachToolForm() {
        const form = document.querySelector("[data-tool-form]");
        if (!form) return;
        const conditionalControls = form.querySelectorAll("[data-depends-field]");

        const applyConditionalVisibility = () => {
            conditionalControls.forEach((wrapper) => {
                const dependsField = wrapper.dataset.dependsField;
                const dependsValue = wrapper.dataset.dependsValue;
                if (!dependsField) return;
                const controller = form.querySelector(`[name="${dependsField}"]`);
                const current = controller ? controller.value : "";
                const shouldShow = current === dependsValue;
                wrapper.dataset.conditionalHidden = shouldShow ? "false" : "true";
                if (shouldShow) {
                    wrapper.style.display = "";
                    const input = wrapper.querySelector("[data-input-field]");
                    if (input) input.disabled = false;
                } else {
                    wrapper.style.display = "none";
                    const input = wrapper.querySelector("[data-input-field]");
                    if (input) {
                        input.disabled = true;
                        input.value = "";
                    }
                }
            });
        };

        const conditionalControllers = new Set();
        conditionalControls.forEach((wrapper) => {
            const dependsField = wrapper.dataset.dependsField;
            if (dependsField) {
                const controller = form.querySelector(`[name="${dependsField}"]`);
                if (controller) conditionalControllers.add(controller);
            }
        });
        conditionalControllers.forEach((controller) => {
            controller.addEventListener("change", applyConditionalVisibility);
        });
        applyConditionalVisibility();

        const runBtn = form.querySelector("[data-run-button]") || form.querySelector("button[type='submit']");
        const statusEl = document.querySelector("[data-tool-status]");
        const outputEl = document.querySelector("[data-tool-output]");
        const outputPre = outputEl ? outputEl.querySelector(".tool-output-pre") : null;
        const shareBlock = outputEl ? outputEl.querySelector("[data-share-block]") : null;
        const shareUrlEl = outputEl ? outputEl.querySelector("[data-share-url]") : null;
        const copyShareBtn = outputEl ? outputEl.querySelector("[data-copy-share]") : null;
        const copyBtn = form.querySelector("[data-copy-button]");
        const viewMode = form.dataset.viewMode === "true";
        const savedInput = (() => {
            const raw = form.dataset.savedInput;
            if (!raw) return null;
            try {
                return JSON.parse(raw);
            } catch (e) {
                return null;
            }
        })();
        let lockedView = viewMode;

        const setStatus = (text, type = "") => {
            if (!statusEl) return;
            statusEl.textContent = text || "";
            statusEl.className = "tool-status";
            if (type) statusEl.classList.add(type);
        };

        const setLoading = (isLoading) => {
            if (runBtn) {
                runBtn.disabled = isLoading;
                runBtn.textContent = isLoading ? "Running…" : "Run tool";
            }
        };

        const clearOutputCards = () => {
            if (!outputEl) return;
            outputEl.querySelectorAll(".result-card, .compare-panel, .nudge-panel").forEach((el) => el.remove());
        };

        const pillClassForVerdict = (label) => {
            const normalized = String(label || "").toLowerCase();
            if (normalized.includes("buy") || normalized.includes("worth")) return "success";
            if (normalized.includes("maybe")) return "warn";
            if (normalized.includes("skip") || normalized.includes("not")) return "danger";
            return "warn";
        };

        const formatNumber = (val, decimals = 2) => {
            const num = typeof val === "number" ? val : Number(val);
            if (!isFinite(num)) return "";
            return num.toFixed(decimals).replace(/\.00$/, "");
        };

        const renderWorthIt = (data) => {
            clearOutputCards();
            if (outputPre) outputPre.textContent = "";

            const card = document.createElement("div");
            card.className = "result-card";

            const title = document.createElement("div");
            title.className = "result-title";
            title.textContent = data.title || "Result";
            card.appendChild(title);

            const primary = document.createElement("div");
            primary.className = "result-primary";
            const primaryLabel = document.createElement("div");
            primaryLabel.className = "result-primary-label";
            primaryLabel.textContent = data.primary?.metric_label || "Primary metric";
            const primaryValue = document.createElement("div");
            primaryValue.className = "result-primary-value";
            primaryValue.textContent = data.primary?.metric_display || "";
            primary.appendChild(primaryLabel);
            primary.appendChild(primaryValue);
            card.appendChild(primary);

            const breakEven = document.createElement("div");
            breakEven.className = "result-secondary";
            const breakLabel = document.createElement("div");
            breakLabel.textContent = "Break-even usage";
            const perWeek = formatNumber(data.break_even?.per_week, 2);
            const perMonth = formatNumber(data.break_even?.per_month, 2);
            const breakValue = document.createElement("div");
            breakValue.className = "result-secondary-value";
            breakValue.textContent = `${perWeek}/week • ${perMonth}/month`;
            breakEven.appendChild(breakLabel);
            breakEven.appendChild(breakValue);
            card.appendChild(breakEven);

            if (data.break_even?.friendly) {
                const sub = document.createElement("div");
                sub.className = "result-subtext";
                sub.textContent = data.break_even.friendly;
                card.appendChild(sub);
            }

            const verdict = document.createElement("div");
            verdict.className = "pill";
            verdict.classList.add(pillClassForVerdict(data.verdict?.label));
            verdict.textContent = data.verdict?.label || "";
            card.appendChild(verdict);

            if (data.verdict?.reason) {
                const sub = document.createElement("div");
                sub.className = "result-subtext";
                sub.textContent = data.verdict.reason;
                card.appendChild(sub);
            }

            if (data.expected && data.expected.adjusted_display) {
                const adj = document.createElement("div");
                adj.className = "result-subtext";
                const ft = data.expected.probability_use_pct;
                const quit = data.expected.chance_quit_pct;
                const bits = [];
                if (typeof ft === "number") bits.push(`${ft}% follow-through`);
                if (typeof quit === "number") bits.push(`${quit}% quit chance`);
                adj.textContent = `Adjusted: ${data.expected.adjusted_display}${bits.length ? ` (${bits.join(", ")})` : ""}`;
                card.appendChild(adj);
            }

            const chipRow = document.createElement("div");
            chipRow.className = "chip-row";
            const chips = [];
            const totals = data.totals || {};
            if (totals.frequency) chips.push(`Frequency: ${totals.frequency}`);
            if (totals.timeframe_months) chips.push(`Timeframe: ${totals.timeframe_months} mo`);
            if (totals.sessions_total !== undefined) chips.push(`Uses: ${formatNumber(totals.sessions_total, 0)}`);
            if (totals.total_hours !== undefined) chips.push(`Hours: ${formatNumber(totals.total_hours, 1)}`);
            if (totals.total_cost !== undefined) chips.push(`Total: $${formatNumber(totals.total_cost, 2)}`);
            chips.forEach((text) => {
                const c = document.createElement("div");
                c.className = "chip";
                c.textContent = text;
                chipRow.appendChild(c);
            });
            card.appendChild(chipRow);

            // Optional time_saved extras
            if (data.time_saved) {
                const rows = document.createElement("div");
                rows.className = "detail-rows";
                const items = [
                    ["Value per use", data.time_saved.value_per_use_display],
                    ["Break-even uses", String(data.time_saved.break_even_uses ?? "")],
                ];
                if (data.time_saved.break_even_uses_per_frequency !== null && data.time_saved.break_even_uses_per_frequency !== undefined) {
                    items.push(["Break-even uses / period", String(data.time_saved.break_even_uses_per_frequency)]);
                }
                items.forEach(([label, value]) => {
                    if (!value) return;
                    const r = document.createElement("div");
                    r.className = "detail-row";
                    const l = document.createElement("span");
                    l.textContent = label;
                    const v = document.createElement("span");
                    v.textContent = value;
                    r.append(l, v);
                    rows.appendChild(r);
                });
                card.appendChild(rows);
            }

            outputEl.appendChild(card);

            // Optional compare panel (if present)
            if (data.compare && data.compare.enabled) {
                const compare = document.createElement("div");
                compare.className = "compare-panel";
                const titleRow = document.createElement("div");
                titleRow.className = "compare-title";
                titleRow.textContent = "Option A vs B";
                compare.appendChild(titleRow);

                const grid = document.createElement("div");
                grid.className = "result-grid";
                const aCol = document.createElement("div");
                aCol.className = "compare-col";
                if (data.compare.winner === "A") aCol.classList.add("winner");
                const aName = document.createElement("div");
                aName.className = "compare-name";
                aName.textContent = data.compare.a?.name || "Option A";
                const aMetric = document.createElement("div");
                aMetric.className = "compare-metric";
                aMetric.textContent = data.compare.a?.display || "";
                aCol.append(aName, aMetric);

                const bCol = document.createElement("div");
                bCol.className = "compare-col";
                if (data.compare.winner === "B") bCol.classList.add("winner");
                const bName = document.createElement("div");
                bName.className = "compare-name";
                bName.textContent = data.compare.b?.name || "Option B";
                const bMetric = document.createElement("div");
                bMetric.className = "compare-metric";
                bMetric.textContent = data.compare.b?.display || "";
                bCol.append(bName, bMetric);

                grid.append(aCol, bCol);
                compare.appendChild(grid);

                const note = document.createElement("div");
                note.className = "compare-note";
                const diff = typeof data.compare.difference_per_hour === "number" ? `$${formatNumber(data.compare.difference_per_hour, 2)}/hr` : "";
                note.textContent = diff ? `Difference: ${diff}. ${data.compare.note || ""}` : (data.compare.note || "");
                compare.appendChild(note);
                outputEl.appendChild(compare);
            }
        };

        const renderSocialPostPolisher = (data) => {
            clearOutputCards();
            if (outputPre) outputPre.textContent = "";

            const card = document.createElement("div");
            card.className = "result-card sp-result";

            const title = document.createElement("div");
            title.className = "result-title";
            title.textContent = "Polished Post";
            card.appendChild(title);

            const meta = document.createElement("div");
            meta.className = "sp-meta";
            const platform = data.platform ? `Platform: ${data.platform}` : "";
            const counts =
                typeof data.original_length === "number" && typeof data.polished_length === "number"
                    ? `Chars: ${data.original_length} → ${data.polished_length}`
                    : "";
            meta.textContent = [platform, counts].filter(Boolean).join(" • ");
            card.appendChild(meta);

            const box = document.createElement("div");
            box.className = "sp-copybox";
            const textEl = document.createElement("pre");
            textEl.className = "sp-text";
            textEl.textContent = data.polished_post || "";
            box.appendChild(textEl);

            const actions = document.createElement("div");
            actions.className = "sp-actions";
            const copyBtn = document.createElement("button");
            copyBtn.type = "button";
            copyBtn.className = "tool-run-btn sp-copy-btn";
            copyBtn.textContent = "Copy";
            copyBtn.onclick = async () => {
                const text = data.polished_post || "";
                if (!text) return;
                try {
                    await navigator.clipboard.writeText(text);
                    copyBtn.textContent = "Copied";
                    setTimeout(() => (copyBtn.textContent = "Copy"), 1200);
                } catch (e) {
                    copyBtn.textContent = "Copy failed";
                    setTimeout(() => (copyBtn.textContent = "Copy"), 1200);
                }
            };
            actions.appendChild(copyBtn);
            card.appendChild(box);
            card.appendChild(actions);

            // Secondary: what changed
            const changesCard = document.createElement("div");
            changesCard.className = "compare-panel sp-secondary";
            const cTitle = document.createElement("div");
            cTitle.className = "compare-title";
            cTitle.textContent = "What Changed";
            changesCard.appendChild(cTitle);

            const changes = data.summary?.changes || [];
            if (Array.isArray(changes) && changes.length) {
                const ul = document.createElement("ul");
                ul.className = "sp-changes";
                changes.slice(0, 6).forEach((c) => {
                    const li = document.createElement("li");
                    li.textContent = String(c);
                    ul.appendChild(li);
                });
                changesCard.appendChild(ul);
            } else {
                const empty = document.createElement("div");
                empty.className = "compare-note";
                empty.textContent = "Tightened wording and improved readability.";
                changesCard.appendChild(empty);
            }

            // Alternate versions (optional)
            const alt = data.alt_versions || {};
            const altShort = (alt.short || "").trim();
            const altHook = (alt.hook_first || "").trim();
            if (altShort || altHook) {
                const toggle = document.createElement("button");
                toggle.type = "button";
                toggle.className = "tool-clear-btn sp-toggle";
                toggle.textContent = "Show alternate versions";

                const panel = document.createElement("div");
                panel.className = "sp-alt";
                panel.style.display = "none";

                const addAlt = (label, text) => {
                    const section = document.createElement("div");
                    section.className = "sp-alt-section";
                    const h = document.createElement("div");
                    h.className = "sp-alt-title";
                    h.textContent = label;
                    const pre = document.createElement("pre");
                    pre.className = "sp-text sp-alt-text";
                    pre.textContent = text;
                    const btn = document.createElement("button");
                    btn.type = "button";
                    btn.className = "tool-run-btn sp-copy-btn";
                    btn.textContent = "Copy";
                    btn.onclick = async () => {
                        try {
                            await navigator.clipboard.writeText(text);
                            btn.textContent = "Copied";
                            setTimeout(() => (btn.textContent = "Copy"), 1200);
                        } catch (e) {
                            btn.textContent = "Copy failed";
                            setTimeout(() => (btn.textContent = "Copy"), 1200);
                        }
                    };
                    section.append(h, pre, btn);
                    panel.appendChild(section);
                };

                if (altShort) addAlt("Short version", altShort);
                if (altHook) addAlt("Hook-first version", altHook);

                toggle.onclick = () => {
                    const isHidden = panel.style.display === "none";
                    panel.style.display = isHidden ? "block" : "none";
                    toggle.textContent = isHidden ? "Hide alternate versions" : "Show alternate versions";
                };

                changesCard.append(toggle, panel);
            }

            outputEl.appendChild(card);
            outputEl.appendChild(changesCard);
        };

        const renderOutput = (data, slug) => {
            if (!outputEl) return;
            clearOutputCards();
            if (outputPre) outputPre.textContent = "";

            const isWorthItCard = slug === "worth-it" && data && typeof data === "object" && data.primary && data.primary.metric_display;
            if (isWorthItCard) {
                renderWorthIt(data);
                return;
            }

            const isSocialPostPolisher =
                slug === "social-post-polisher" && data && typeof data === "object" && typeof data.polished_post === "string";
            if (isSocialPostPolisher) {
                renderSocialPostPolisher(data);
                return;
            }

            const isShareCard = data && typeof data === "object" && data.share_card;
            if (!isShareCard) {
                if (outputPre) outputPre.textContent = typeof data === "string" ? data : "";
                return;
            }

            const cardData = data.share_card;
            const details = data.details || [];
            const card = document.createElement("div");
            card.className = "result-card";

            const title = document.createElement("div");
            title.className = "result-title";
            title.textContent = cardData.title || "Result";
            card.appendChild(title);

            const primary = document.createElement("div");
            primary.className = "result-primary";
            const primaryLabel = document.createElement("div");
            primaryLabel.className = "result-primary-label";
            primaryLabel.textContent = cardData.primary_metric_label || "Primary metric";
            const primaryValue = document.createElement("div");
            primaryValue.className = "result-primary-value";
            primaryValue.textContent = cardData.primary_metric_value || "";
            primary.appendChild(primaryLabel);
            primary.appendChild(primaryValue);
            card.appendChild(primary);

            const secondary = document.createElement("div");
            secondary.className = "result-secondary";
            const breakLabel = document.createElement("div");
            breakLabel.textContent = cardData.break_even_label || "Break-even";
            const breakValue = document.createElement("div");
            breakValue.className = "result-secondary-value";
            breakValue.textContent = cardData.break_even_value || "";
            secondary.appendChild(breakLabel);
            secondary.appendChild(breakValue);
            card.appendChild(secondary);

            const verdict = document.createElement("div");
            verdict.className = "pill";
            if (cardData.pill_class) verdict.classList.add(cardData.pill_class);
            verdict.textContent = cardData.verdict || "";
            card.appendChild(verdict);

            if (cardData.subtext) {
                const sub = document.createElement("div");
                sub.className = "result-subtext";
                sub.textContent = cardData.subtext;
                card.appendChild(sub);
            }

            if (details.length) {
                const rows = document.createElement("div");
                rows.className = "detail-rows";
                details.forEach((row) => {
                    const r = document.createElement("div");
                    r.className = "detail-row";
                    const l = document.createElement("span");
                    l.textContent = row.label;
                    const v = document.createElement("span");
                    v.textContent = row.value;
                    r.append(l, v);
                    rows.appendChild(r);
                });
                card.appendChild(rows);
            }

            outputEl.appendChild(card);

            const comparison = data.comparison;
            if (comparison && comparison.enabled) {
                const compare = document.createElement("div");
                compare.className = "compare-panel";
                const titleRow = document.createElement("div");
                titleRow.className = "compare-title";
                titleRow.textContent = "Option A vs B";
                compare.appendChild(titleRow);

                const grid = document.createElement("div");
                grid.className = "result-grid";
                const aCol = document.createElement("div");
                aCol.className = "compare-col";
                const aName = document.createElement("div");
                aName.className = "compare-name";
                aName.textContent = comparison.a?.name || "Option A";
                const aMetric = document.createElement("div");
                aMetric.className = "compare-metric";
                aMetric.textContent = typeof comparison.a?.primary_metric_value === "number" ? comparison.a.primary_metric_value.toFixed(2) : (comparison.a?.primary_metric_value || "");
                if (comparison.winner === "A") aCol.classList.add("winner");
                aCol.append(aName, aMetric);

                const bCol = document.createElement("div");
                bCol.className = "compare-col";
                const bName = document.createElement("div");
                bName.className = "compare-name";
                bName.textContent = comparison.b?.name || "Option B";
                const bMetric = document.createElement("div");
                bMetric.className = "compare-metric";
                bMetric.textContent = typeof comparison.b?.primary_metric_value === "number" ? comparison.b.primary_metric_value.toFixed(2) : (comparison.b?.primary_metric_value || "");
                if (comparison.winner === "B") bCol.classList.add("winner");
                bCol.append(bName, bMetric);

                grid.append(aCol, bCol);
                compare.appendChild(grid);

                const note = document.createElement("div");
                note.className = "compare-note";
                note.textContent = comparison.break_even_note || "";
                compare.appendChild(note);

                outputEl.appendChild(compare);
            }

            const nudge = data.nudge;
            if (nudge && nudge.enabled) {
                const nudgePanel = document.createElement("div");
                nudgePanel.className = "nudge-panel";
                const nTitle = document.createElement("div");
                nTitle.className = "compare-title";
                nTitle.textContent = "Decision nudge";
                nudgePanel.appendChild(nTitle);

                const rows = document.createElement("div");
                rows.className = "detail-rows";
                const multRow = document.createElement("div");
                multRow.className = "detail-row";
                multRow.innerHTML = `<span>Expected usage multiplier</span><span>${nudge.expected_usage_multiplier || ""}</span>`;
                rows.appendChild(multRow);
                const adjRow = document.createElement("div");
                adjRow.className = "detail-row";
                adjRow.innerHTML = `<span>${nudge.adjusted_primary_metric_label || "Adjusted"}</span><span>${nudge.adjusted_primary_metric_value || ""}</span>`;
                rows.appendChild(adjRow);
                if (nudge.return_window_days) {
                    const rwRow = document.createElement("div");
                    rwRow.className = "detail-row";
                    rwRow.innerHTML = `<span>Return window</span><span>${nudge.return_window_days} days</span>`;
                    rows.appendChild(rwRow);
                }
                nudgePanel.appendChild(rows);

                if (nudge.note) {
                    const note = document.createElement("div");
                    note.className = "compare-note";
                    note.textContent = nudge.note;
                    nudgePanel.appendChild(note);
                }

                outputEl.appendChild(nudgePanel);
            }
        };

        const initWorthItFormUX = () => {
            const slug = form.dataset.toolSlug;
            if (slug !== "worth-it") return;

            const modeEl = form.querySelector('[name="mode"]');
            const freqEl = form.querySelector('[name="frequency"]');
            const minutesEl = form.querySelector('[name="minutes_per_use"]');
            const usesEl = form.querySelector('[name="uses_per_frequency"]');

            const setFieldLabel = (inputEl, text) => {
                if (!inputEl) return;
                const wrapper = inputEl.closest(".tool-field");
                if (!wrapper) return;
                const labelEl = wrapper.querySelector("label");
                if (!labelEl) return;
                const star = inputEl.required ? "*" : "";
                labelEl.textContent = `${text}${star}`;
            };

            const apply = () => {
                const mode = modeEl ? modeEl.value : "enjoyment";
                const freq = freqEl ? freqEl.value : "One-time";

                if (minutesEl) {
                    if (mode === "time_saved") {
                        setFieldLabel(minutesEl, "Minutes saved per use");
                        minutesEl.placeholder = "10";
                    } else {
                        setFieldLabel(minutesEl, "Minutes of enjoyment per use");
                        minutesEl.placeholder = "60";
                    }
                }

                if (usesEl) {
                    setFieldLabel(usesEl, "Uses per frequency");
                    if (freq === "Daily") usesEl.placeholder = "1 (times per day)";
                    else if (freq === "Weekly") usesEl.placeholder = "3 (times per week)";
                    else if (freq === "Biweekly") usesEl.placeholder = "6 (times per 2 weeks)";
                    else if (freq === "Monthly") usesEl.placeholder = "12 (times per month)";
                    else if (freq === "Yearly") usesEl.placeholder = "180 (times per year)";
                    else usesEl.placeholder = "50 (total uses)";
                }
            };

            if (modeEl) modeEl.addEventListener("change", apply);
            if (freqEl) freqEl.addEventListener("change", apply);
            apply();
        };

        const showShare = (url) => {
            if (!shareBlock || !shareUrlEl) return;
            const full = url.startsWith("http") ? url : `${window.location.origin}${url}`;
            shareUrlEl.textContent = full;
            shareBlock.style.display = "block";
            if (copyShareBtn) {
                copyShareBtn.onclick = async () => {
                    try {
                        await navigator.clipboard.writeText(full);
                        setStatus("Link copied.", "success");
                    } catch (e) {
                        setStatus("Unable to copy link.", "error");
                    }
                };
            }
        };

        const setViewMode = (isView) => {
            lockedView = isView;
            const fields = form.querySelectorAll("[data-input-field]");
            fields.forEach((f) => {
                f.disabled = isView;
            });
            if (runBtn) runBtn.style.display = isView ? "none" : "inline-block";
            if (copyBtn) copyBtn.style.display = isView ? "inline-block" : "none";
        };

        if (savedInput) {
            const fields = form.querySelectorAll("[data-input-field]");
            fields.forEach((f) => {
                if (savedInput[f.name] !== undefined) {
                    f.value = savedInput[f.name];
                }
            });
            applyConditionalVisibility();
            initWorthItFormUX();
        }

        if (viewMode) {
            setViewMode(true);
        }

        const initialShare = form.dataset.shareInitial;
        if (initialShare) {
            showShare(initialShare);
        }

        // Local-only tool init (no API calls)
        initWorkoutLog(form, outputEl, statusEl);
        initGroceryList(form, outputEl, statusEl);
        initCountdown(form, outputEl, statusEl);

        if (copyBtn) {
            copyBtn.addEventListener("click", () => {
                setViewMode(false);
                setStatus("Copied to new. Edit and re-run to generate a fresh link.", "success");
            });
        }

        initWorthItFormUX();

        form.addEventListener("submit", async (event) => {
            event.preventDefault();
            const slug = form.dataset.toolSlug;
            if (!slug) return;

            if (slug === "workout-log") {
                // handled fully client-side
                return;
            }
            if (slug === "grocery-list") {
                // handled by autosave UI
                return;
            }
            if (slug === "countdown") {
                // handled client-side (share uses explicit button)
                return;
            }

            if (lockedView) {
                setStatus("Use Copy to new before editing this shared view.", "error");
                return;
            }

            if (slug === "daily-checkin") {
                return;
            }

            if (slug === "trip-planner") {
                if (!window.tripPlannerState || !window.tripPlannerBuildPayload) {
                    setStatus("Trip planner not initialized.", "error");
                    return;
                }
                const { ok, payload, error } = window.tripPlannerBuildPayload();
                if (!ok) {
                    setStatus(error || "Please complete the required fields.", "error");
                    return;
                }
                setStatus("");
                setLoading(true);
                const response = await runToolRequest(slug, payload);
                if (!response || !response.ok) {
                    const message = response?.error?.message || "Something went wrong.";
                    setStatus(message, "error");
                    setLoading(false);
                    return;
                }
                const output = response.data?.output || "No output returned.";
                renderOutput(output, slug);
                if (response.data?.share_url) {
                    showShare(response.data.share_url);
                }
                setStatus("Saved and shared.", "success");
                setLoading(false);
                return;
            }

            setStatus("");
            setLoading(true);

            const input = serializeForm(form);
            const response = await runToolRequest(slug, input);

            if (!response || !response.ok) {
                const message = response?.error?.message || "Something went wrong.";
                setStatus(message, "error");
                setLoading(false);
                return;
            }

            const output = response.data?.output || "No output returned.";
            renderOutput(output, slug);
            if (response.data?.share_url) {
                showShare(response.data.share_url);
            }
            if (slug === "daily-phrase") {
                const phrase = normalizePhraseValue(response.data?.phrase);
                window.__rrToolState["daily-phrase"] = {
                    phrase,
                    translation: normalizePhraseValue(response.data?.translation),
                    example: normalizePhraseValue(response.data?.example),
                    language: form.querySelector('[name="language"]')?.value || "",
                    level: form.querySelector('[name="level"]')?.value || "",
                };
                renderAudioControls(form);
                setStatus("Come back tomorrow for a new phrase.", "success");
                if (runBtn) runBtn.disabled = true;
            } else {
                setStatus("Done.", "success");
            }
            setLoading(false);
        });
    }

    function attachToolSpecificBehavior() {
        const form = document.querySelector("[data-tool-form]");
        if (!form) return;
        const slug = form.dataset.toolSlug;
        if (!slug) return;

        if (slug === "daily-checkin") {
            const statusEl = document.querySelector("[data-tool-status]");
            const outputEl = document.querySelector("[data-tool-output]");
            const outputPre = outputEl ? outputEl.querySelector(".tool-output-pre") : null;
            const clearBtn = form.querySelector("[data-clear-button]");
            handleDailyCheckin(form, statusEl, outputPre, clearBtn);
            return;
        }

        if (slug === "trip-planner") {
            initTripPlanner(form);
        }

        if (slug === "daily-phrase") {
            initAudioControls(form);
        }
    }

    window.runToolRequest = runToolRequest;

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", () => {
            attachToolForm();
            attachToolSpecificBehavior();
        });
    } else {
        attachToolForm();
        attachToolSpecificBehavior();
    }
})();
