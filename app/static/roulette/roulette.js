(function () {
  const cardsEl = document.getElementById("cards");
  const promptEl = document.getElementById("prompt");
  const dateNoteEl = document.getElementById("dateNote");
  const resultEl = document.getElementById("result");
  const sourceLink = document.getElementById("sourceLink");
  const nextBtn = document.getElementById("nextBtn");
  const restartBtn = document.getElementById("restartBtn");
  const sessionScore = document.getElementById("sessionScore");
  const recapEl = document.getElementById("recap");

  let state = { step: 1, score: 0, payload: null, locked: false };

  const escapeHtml = (s = "") =>
    s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");

  function setHeader(payload) {
    promptEl.textContent = payload.prompt || "Pick the real event.";
    dateNoteEl.textContent = payload.date_label
      ? `Based on an event from ${payload.date_label}.`
      : "";
  }

  function renderCards(payload) {
    cardsEl.innerHTML = "";
    sourceLink.style.display = "none";
    resultEl.classList.remove("is-visible");
    resultEl.textContent = "";
    state.locked = false;
    const type = payload.type;

    payload.cards.forEach((card, idx) => {
      const cardEl = document.createElement("article");
      cardEl.className = "choice-card";
      cardEl.setAttribute("role", "button");
      cardEl.setAttribute("tabindex", "0");
      cardEl.dataset.idx = idx;
      cardEl.setAttribute("aria-pressed", "false");

      if (type === "quote") {
        cardEl.classList.add("is-quote");
        cardEl.innerHTML = `
          <div class="choice-media">
            <div class="image-frame">
              <img src="${escapeHtml(card.image_url || "")}" alt="" loading="lazy">
            </div>
          </div>
          <div class="choice-body quote-body">
            <div class="choice-meta">Choice ${idx + 1}</div>
            <div class="quote-block">
              <p class="quote-text">${escapeHtml(card.quote || card.title || "")}</p>
              <div class="quote-author">&#8212; ${escapeHtml(card.author || "Unknown")}</div>
            </div>
          </div>
        `;
      } else {
        cardEl.classList.add(type === "image" ? "is-image" : "is-headline");
        const blurb = card.blurb || "";
        cardEl.innerHTML = `
          <div class="choice-media">
            <div class="image-frame">
              <img src="${escapeHtml(card.image_url || "")}" alt="" loading="lazy">
            </div>
          </div>
          <div class="choice-body">
            <div class="choice-meta">Choice ${idx + 1}</div>
            <h3 class="choice-title">${escapeHtml(card.title || "")}</h3>
            ${blurb ? `<p class="choice-blurb">${escapeHtml(blurb)}</p>` : ""}
          </div>
        `;
      }

      const select = () => handleGuess(idx, payload.correct_index);
      cardEl.addEventListener("click", select);
      cardEl.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          select();
        }
      });
      cardsEl.appendChild(cardEl);
    });
  }

  function lockCards(correctIdx, pickedIdx) {
    const cards = cardsEl.querySelectorAll(".choice-card");
    cards.forEach((card, i) => {
      card.classList.add("is-locked");
      card.classList.toggle("is-selected", i === pickedIdx);
      card.classList.toggle("is-correct", i === correctIdx);
      card.setAttribute("aria-pressed", i === pickedIdx ? "true" : "false");
    });
  }

  async function fetchSession() {
    const res = await fetch("/roulette/session");
    if (!res.ok) throw new Error("session fetch failed");
    const data = await res.json();
    state.step = data.step;
    state.score = data.score;
    state.payload = data.payload;
    const played = Math.max(0, Math.min(state.step - 1, 3));
    sessionScore.textContent = `Score: ${state.score}/${played}`;
    setHeader(data.payload);
    resultEl.classList.remove("is-visible");
    nextBtn.style.display = "none";
    restartBtn.style.display = "none";
    recapEl.style.display = "none";
    renderCards(data.payload);
  }

  async function handleGuess(idx, correctIdx) {
    if (state.locked) return;
    state.locked = true;
    lockCards(correctIdx, idx);
    try {
      const res = await fetch("/roulette/session/guess", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ choice: idx }),
      });
      const data = await res.json();
      if (data.ok) {
        const msg = data.is_correct ? "You got it." : "Good try - the real pick is highlighted.";
        resultEl.textContent = msg;
        resultEl.classList.add("is-visible");
        state.score = data.score;
        sessionScore.textContent = `Score: ${Math.min(state.score, 3)}/${Math.min(state.step, 3)}`;
        if (state.payload && state.payload.source_url) {
          sourceLink.href = state.payload.source_url;
          sourceLink.style.display = "inline-flex";
        }
        nextBtn.style.display = "inline-flex";
      }
    } catch (e) {
      resultEl.textContent = "Unable to record that pick.";
      resultEl.classList.add("is-visible");
    }
  }

  async function nextStep() {
    const res = await fetch("/roulette/session/next", { method: "POST" });
    const data = await res.json();
    if (!data.ok) return;
    if (data.rounds) {
      showRecap(data);
      return;
    }
    state.step = data.step;
    state.score = data.score;
    state.payload = data.payload;
    const played = Math.max(0, Math.min(state.step - 1, 3));
    sessionScore.textContent = `Score: ${Math.min(state.score, 3)}/${played}`;
    setHeader(data.payload);
    resultEl.classList.remove("is-visible");
    nextBtn.style.display = "none";
    renderCards(data.payload);
    state.locked = false;
  }

  function showRecap(data) {
    recapEl.innerHTML = `<h3>Recap (Score: ${data.score}/3)</h3>`;
    const sections = [
      ["text", "Headlines"],
      ["image", "Photos"],
      ["quote", "Quotes"],
    ];
    sections.forEach(([key, label]) => {
      const r = data.rounds[key];
      if (!r || !r.payload) return;
      const block = document.createElement("div");
      block.className = "recap-block";
      block.innerHTML = `<div class="choice-meta">${label}</div>`;
      r.payload.cards.forEach((c, i) => {
        const line = document.createElement("div");
        line.className = "recap-line";
        const mark = i === r.payload.correct_index ? "✅" : i === r.guess ? "✕" : "•";
        const text = c.quote || c.title || c.raw_text || `Choice ${i + 1}`;
        line.textContent = `${mark} ${text}`;
        block.appendChild(line);
      });
      recapEl.appendChild(block);
    });
    recapEl.style.display = "block";
    nextBtn.style.display = "none";
    restartBtn.style.display = "inline-flex";
    sourceLink.style.display = "none";
  }

  restartBtn.addEventListener("click", async () => {
    try {
      await fetch("/roulette/session/reset", { method: "POST" });
    } catch (e) {
      // ignore
    }
    state.locked = false;
    await fetchSession();
  });
  nextBtn.addEventListener("click", () => {
    state.locked = false;
    nextStep();
  });

  fetchSession().catch(() => {
    promptEl.textContent = "Unable to start game right now.";
  });
})();
