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
