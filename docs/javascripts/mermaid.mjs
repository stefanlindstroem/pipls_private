import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";

const PIPLS_MERMAID_HOST_CSS = `
.md-typeset div.mermaid {
  box-sizing: border-box;

  /* Space outside the entire chart */
  margin: 0 0 0 40px;

  /* Space between the host boundary and the SVG */
  padding: 0;
}
`;

function installPiplsMermaidHostCss() {
  const id = "pipls-mermaid-host-css";

  if (document.getElementById(id)) {
    return;
  }

  const style = document.createElement("style");
  style.id = id;
  style.textContent = PIPLS_MERMAID_HOST_CSS;
  document.head.append(style);
}

installPiplsMermaidHostCss();

const EDGE_LABEL_PALETTES = {
  default: {
    background: "#e7e7e7",
    foreground: "#3b2d00",
  },
  slate: {
    background: "#5a4514",
    foreground: "#fff1bd",
  },
};

const NODE_PALETTES = {
  default: {
    fill: "#4051b5",
    stroke: "#000044",
    text: "#ffffff",
    strokeWidth: "0px",
  },
  slate: {
    fill: "#2b3038",
    stroke: "#9aa4b2",
    text: "#f5f7fa",
    strokeWidth: "1.25px",
  },
};

const PIPLS_MERMAID_THEME_CSS = `
/* Standard Mermaid flowchart node shapes */
.node rect,
.node polygon,
.node circle,
.node ellipse,
.node path {
  fill: var(--pipls-mermaid-node-fill) !important;
  stroke: var(--pipls-mermaid-node-stroke) !important;
  stroke-width: var(--pipls-mermaid-node-stroke-width) !important;
}

/* Rounded corners for rectangular nodes */
.node rect {
  rx: 0.5ex !important;
  ry: 0.5ex !important;
}

/* Node text rendered as HTML */
.nodeLabel,
.nodeLabel p,
.nodeLabel span,
.node foreignObject {
  color: var(--pipls-mermaid-node-text) !important;
  font-size: small !important;
  line-height: 1.15 !important;
}

/* Node text rendered as SVG */
.node text,
.node tspan {
  color: var(--pipls-mermaid-node-text) !important;
  fill: var(--pipls-mermaid-node-text) !important;
  font-size: small !important;
}

/* Keep HTML node labels compact */
.nodeLabel p,
.nodeLabel span {
  margin: 0 !important;
  background: transparent !important;
}

/* The outer Mermaid edge label is only a container */
.edgeLabel {
  background: transparent !important;
  color: var(--pipls-mermaid-edge-label-fg) !important;
}

/* HTML-rendered edge-label box */
.edgeLabel > p {
  display: inline-flex !important;
  align-items: center !important;
  box-sizing: border-box !important;

  margin: 0 !important;
  padding: 0.5ex 0.5ex !important;

  background-color: var(--pipls-mermaid-edge-label-bg) !important;
  color: var(--pipls-mermaid-edge-label-fg) !important;

  font-size: small !important;
  line-height: 1 !important;
  border-radius: 0.5ex !important;
}

/* Do not let nested inline elements create a second box */
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
  color: var(--pipls-mermaid-edge-label-fg) !important;
  fill: var(--pipls-mermaid-edge-label-fg) !important;
  font-size: small !important;
  line-height: 1 !important;
}

/* SVG-backed edge-label background */
.edgeLabel .labelBkg,
.edgeLabel rect,
.labelBkg {
  fill: var(--pipls-mermaid-edge-label-bg) !important;
  opacity: 1 !important;
  rx: 0.5ex;
  ry: 0.5ex;
}
`;

const PIPLS_FLOWCHART_CONFIG = {
  diagramPadding: 30,
  padding: 5,
  nodeSpacing: 24,
  rankSpacing: 24,
};

function applyMermaidPalette() {
  const scheme = document.body?.getAttribute("data-md-color-scheme");
  const paletteName = scheme === "slate" ? "slate" : "default";

  const edgeLabelPalette = EDGE_LABEL_PALETTES[paletteName];
  const nodePalette = NODE_PALETTES[paletteName];
  const root = document.documentElement;

  root.style.setProperty(
    "--pipls-mermaid-edge-label-bg",
    edgeLabelPalette.background,
  );
  root.style.setProperty(
    "--pipls-mermaid-edge-label-fg",
    edgeLabelPalette.foreground,
  );

  root.style.setProperty("--pipls-mermaid-node-fill", nodePalette.fill);
  root.style.setProperty("--pipls-mermaid-node-stroke", nodePalette.stroke);
  root.style.setProperty("--pipls-mermaid-node-text", nodePalette.text);
  root.style.setProperty(
    "--pipls-mermaid-node-stroke-width",
    nodePalette.strokeWidth,
  );
}

applyMermaidPalette();

if (document.body) {
  new MutationObserver(applyMermaidPalette).observe(document.body, {
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

/* Material for MkDocs uses this instance when mounting Mermaid blocks. */
window.mermaid = mermaid;

