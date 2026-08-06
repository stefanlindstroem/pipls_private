import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";

const EDGE_LABEL_PALETTES = {
  default: {
    background: "#fff0bd",
    foreground: "#3b2d00",
  },
  slate: {
    background: "#5a4514",
    foreground: "#fff1bd",
  },
};

const PIPLS_MERMAID_THEME_CSS = `
/* Ordinary rectangular flowchart nodes */
.node rect {
  rx: 5px !important;
  ry: 5px !important;
}

/* Node text rendered as HTML */
.nodeLabel,
.nodeLabel p,
.nodeLabel span {
  font-size: 13px !important;
  line-height: 1.15 !important;
}

/* Node text rendered as SVG */
.node text,
.node tspan {
  font-size: 13px !important;
}

/* Optional compact label layout */
.nodeLabel p,
.nodeLabel span {
  margin: 0 !important;
}

.edgeLabel {
  background: transparent !important;
  color: var(--pipls-mermaid-edge-label-fg) !important;
}

/* HTML-rendered edge labels */
.edgeLabel > p {
  display: inline-flex !important;
  align-items: center !important;
  box-sizing: border-box !important;

  margin: 0 !important;
  padding: 2px 5px 0 !important;

  background-color: var(--pipls-mermaid-edge-label-bg) !important;
  color: var(--pipls-mermaid-edge-label-fg) !important;

  font-size: 13px !important;
  line-height: 1 !important;
  border-radius: 5px !important;
}

.edgeLabel > p span {
  margin: 0 !important;
  padding: 0 !important;
  background: transparent !important;
  color: inherit !important;
  font-size: inherit !important;
  line-height: inherit !important;
}

/* SVG-rendered edge labels */
.edgeLabel .label,
.edgeLabel .label text,
.edgeLabel .label tspan,
.edgeLabel text,
.edgeLabel tspan,
.flowchart-label .text-outer-tspan {
  font-size: 13px !important;
  line-height: 1.0 !important;
  color: var(--pipls-mermaid-edge-label-fg) !important;
  fill: var(--pipls-mermaid-edge-label-fg) !important;
  dominant-baseline: central !important;
  alignment-baseline: central !important;
}

/* Edge-label backing shapes */
.edgeLabel .labelBkg,
.edgeLabel rect,
.labelBkg {
  fill: var(--pipls-mermaid-edge-label-bg) !important;
  opacity: 1 !important;
  rx: 5px;
  ry: 5px;
}
`;

const PIPLS_FLOWCHART_CONFIG = {
  diagramPadding: 20,
  padding: 5,
  nodeSpacing: 24,
  rankSpacing: 24,
};

function applyEdgeLabelPalette() {
  const scheme = document.body?.getAttribute("data-md-color-scheme");
  const palette = scheme === "slate" ? EDGE_LABEL_PALETTES.slate : EDGE_LABEL_PALETTES.default;
  const root = document.documentElement;

  root.style.setProperty("--pipls-mermaid-edge-label-bg", palette.background);
  root.style.setProperty("--pipls-mermaid-edge-label-fg", palette.foreground);
}

applyEdgeLabelPalette();

if (document.body) {
  new MutationObserver(applyEdgeLabelPalette).observe(document.body, {
    attributes: true,
    attributeFilter: ["data-md-color-scheme"],
  });
}

const initializeMermaid = mermaid.initialize.bind(mermaid);

mermaid.initialize = (configuration = {}) => {
  const materialThemeCss = configuration.themeCSS ?? "";
  return initializeMermaid({
    ...configuration,
    flowchart: {
      ...(configuration.flowchart ?? {}),
      ...PIPLS_FLOWCHART_CONFIG,
    },
    themeCSS: `${materialThemeCss}
${PIPLS_MERMAID_THEME_CSS}`,
  });
};

// Material for MkDocs uses this global instance when it mounts Mermaid blocks.
window.mermaid = mermaid;
