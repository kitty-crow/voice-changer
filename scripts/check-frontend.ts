import { readFile } from "node:fs/promises";
import { assertMobileViewport } from "../vendor/pages/src/mobile.ts";

const read = (path: string) => readFile(path, "utf8");
const requireMatch = (source: string, pattern: RegExp, message: string) => {
    if (!pattern.test(source)) throw new Error(message);
};
const requireText = (source: string, text: string, message: string) => {
    if (!source.includes(text)) throw new Error(message);
};

const html = await read("client/demo/public/index.html");
assertMobileViewport(html, "client/demo/public/index.html");
requireMatch(html, /<html\b[^>]*\blang=["']en-GB["']/i, "HTML root must declare lang=\"en-GB\".");
requireMatch(html, /<meta\b[^>]*\bname=["']description["'][^>]*\bcontent=["'][^"']+["']/i, "HTML must provide a non-empty description meta tag.");
requireMatch(html, /<meta\b[^>]*\bname=["']theme-color["'][^>]*\bcontent=["'][^"']+["']/i, "HTML must provide a theme-colour meta tag.");
requireMatch(html, /<meta\b[^>]*\bname=["']color-scheme["'][^>]*\bcontent=["']light dark["']/i, "HTML must advertise light/dark colour-scheme support.");
requireMatch(html, /<title>[^<]+<\/title>/i, "HTML must provide a document title.");
requireMatch(html, /<div\b[^>]*\bid=["']app["'][^>]*><\/div>/i, "HTML must keep the React #app mount point.");
if (/<(?:html|body|div)\b[^>]*\bstyle=["'][^"']*(?:width|height|overflow)/i.test(html)) {
    throw new Error("Responsive shell dimensions must live in CSS, not inline HTML styles.");
}

const entry = await read("client/demo/src/000_index.tsx");
requireText(entry, 'import "./css/Modern.css";', "Modern.css must load after the legacy compatibility stylesheet.");

const header = await read("client/demo/src/components/demo/components2/001_HeaderArea.tsx");
requireText(header, "<header", "The application shell must use a semantic header element.");
requireText(header, "data-theme-toggle", "The header must expose the theme control.");

const layout = await read("client/demo/src/components/demo/b00_ModelSlotControl.tsx");
requireText(layout, "<main", "The primary application controls must live inside a semantic main element.");
requireText(layout, "<section", "Primary control groups must use semantic sections.");

const demo = await read("client/demo/src/components/demo/010_Demo.tsx");
requireText(demo, "<footer", "The application shell must include a semantic footer.");
requireText(demo, "appGuiSettingState.version", "The footer must display the runtime application version.");

const responsive = await read("client/demo/src/css/modern/responsive.css");
for (const breakpoint of ["1100px", "820px", "560px"]) {
    requireText(responsive, breakpoint, `Responsive CSS must retain the ${breakpoint} breakpoint.`);
}
requireText(responsive, "100dvh", "Responsive CSS must account for dynamic mobile viewport height.");

const shell = await read("client/demo/src/css/modern/shell.css");
requireText(shell, "min-width: 0", "The modern shell must explicitly protect grid/flex children from horizontal clipping.");
requireText(shell, "overflow-x: hidden", "The modern shell must prevent accidental page-level horizontal overflow.");

console.log("Front-end standards check passed.");
