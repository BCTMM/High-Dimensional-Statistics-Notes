document$.subscribe(({ body }) => {
  renderMathInElement(body, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true },
    ],
    macros: {
      "\\R": "\\mathbb{R}",
      "\\E": "\\mathbb{E}",
      "\\P": "\\mathbb{P}",
      "\\Var": "\\operatorname{Var}",
      "\\Cov": "\\operatorname{Cov}",
      "\\tr": "\\operatorname{tr}",
      "\\diag": "\\operatorname{diag}",
    },
    throwOnError: false,
  });
});
