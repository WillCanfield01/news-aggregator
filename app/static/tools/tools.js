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

    window.runToolRequest = runToolRequest;

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", attachToolForm);
    } else {
        attachToolForm();
    }
})();
