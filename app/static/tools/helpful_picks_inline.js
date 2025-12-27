document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".helpful-picks-chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      const slug = btn.getAttribute("data-tool-slug");
      const panel = document.getElementById(`helpful-picks-${slug}`);
      if (!panel) return;
      const isHidden = panel.hasAttribute("hidden");
      if (isHidden) {
        panel.removeAttribute("hidden");
      } else {
        panel.setAttribute("hidden", "hidden");
      }
    });
  });
});
