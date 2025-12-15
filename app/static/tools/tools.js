(function () {
    const endpoint = "/api/tools/run";

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

    function serializeForm(form) {
        const fields = form.querySelectorAll("[data-input-field]");
        const payload = {};
        fields.forEach((field) => {
            const name = field.name;
            const type = field.dataset.type || field.type;
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

        const runBtn = form.querySelector("[data-run-button]") || form.querySelector("button[type='submit']");
        const statusEl = document.querySelector("[data-tool-status]");
        const outputEl = document.querySelector("[data-tool-output]");
        const outputPre = outputEl ? outputEl.querySelector(".tool-output-pre") : null;

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

        form.addEventListener("submit", async (event) => {
            event.preventDefault();
            const slug = form.dataset.toolSlug;
            if (!slug) return;

            if (slug === "daily-checkin") {
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
            setStatus("Done.", "success");
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
