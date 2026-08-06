function piplsMermaidThemeCss() {
  const isSlate = document.body?.getAttribute("data-md-color-scheme") === "slate";
  const background = isSlate ? "#5a4514" : "#fff0bd";
  const foreground = isSlate ? "#fff1bd" : "#3b2d00";

  return `
.edgeLabel .label,
.edgeLabel p,
.edgeLabel span {
  background-color: ${background} !important;
  color: ${foreground} !important;
}

.edgeLabel p,
.edgeLabel span {
  margin: 0 !important;
  padding: 1px 5px !important;
  border-radius: 3px !important;
  line-height: 1.15 !important;
}

.edgeLabel .label rect,
.edgeLabel .labelBkg,
.edgeLabel rect,
.labelBkg {
  fill: ${background} !important;
  opacity: 1 !important;
}
`;
}

function withMermaidRuntime(callback, attempts = 40) {
  if (typeof window.mermaid !== "undefined") {
    callback(window.mermaid);
    return;
  }
  if (attempts <= 0) {
    console.warn("Mermaid runtime was not available for custom initialization.");
    return;
  }
  window.setTimeout(() => withMermaidRuntime(callback, attempts - 1), 50);
}

function renderPiplsMermaidDiagrams() {
  withMermaidRuntime((mermaid) => {
    mermaid.initialize({
      startOnLoad: false,
      securityLevel: "loose",
      themeCSS: piplsMermaidThemeCss(),
    });
    mermaid.run({ querySelector: ".mermaid" });
  });
}

if (typeof window.document$ !== "undefined") {
  window.document$.subscribe(renderPiplsMermaidDiagrams);
} else {
  document.addEventListener("DOMContentLoaded", renderPiplsMermaidDiagrams);
}
