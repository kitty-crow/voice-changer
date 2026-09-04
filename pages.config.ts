import { definePages } from "./vendor/pages/src/index.ts";

export default definePages({
    source: "client/demo/public",
    out: ".frontend-pages-check",
    copySource: false,
    noJekyll: false,
    pages: [{ from: "index.html", route: "/" }],
});
