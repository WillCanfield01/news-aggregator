(function () {
    const endpoint = "/api/tools/run";
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

        const renderOutput = (text) => {
            if (outputPre) {
                outputPre.textContent = text || "";
            }
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
        }

        if (viewMode) {
            setViewMode(true);
        }

        const initialShare = form.dataset.shareInitial;
        if (initialShare) {
            showShare(initialShare);
        }

        if (copyBtn) {
            copyBtn.addEventListener("click", () => {
                setViewMode(false);
                setStatus("Copied to new. Edit and re-run to generate a fresh link.", "success");
            });
        }

        form.addEventListener("submit", async (event) => {
            event.preventDefault();
            const slug = form.dataset.toolSlug;
            if (!slug) return;

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
                renderOutput(output);
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
            renderOutput(output);
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
