# Vendored icons

Twenty-two SVGs from [Lucide](https://lucide.dev) (`lucide-icons/lucide`, `icons/` folder). Twenty-one were fetched
2026-11-13 and `quote` (the Audiobooks Dialogue marker) on 2026-11-23, all from release tag 1.72.0. They are copied
unmodified; colour is applied at load time by `icons.py`, because Qt Style Sheets do not support `currentColor`.

Licence: Lucide is ISC; `chevron-down`, `chevron-right`, `download`, `info`, `square` and
`circle-exclamation-point` (formerly `alert-circle` / `circle-alert`) are derived from Feather (MIT). Both notices are in `LICENSE` in this folder and must stay with the assets.

Only icons the application uses are vendored. To add one, copy its SVG here, reference it from the
code, and add its name to `_USED_ICONS` in `tests_ui.py` (a test fails if an unused icon is left here).
