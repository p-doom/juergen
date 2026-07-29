# Spec: videocua_key_map.json

Produce `videocua_key_map.json` in this directory. It drives normalization of VideoCUA's messy
human-annotated key names into the crowd-cast rdev-style key vocabulary (see
`crowdcast_key_vocab.json` for the observed target vocabulary; census of raw VideoCUA values is in
`census_prelim/census.json` under `key_name_vocab`).

Structure:
{
  "aliases":   { "<lowercased, whitespace-collapsed alias>": <entry>, ... },
  "modifiers": { "<alias>": "<vocab modifier>", ... },
  "chars":     { "<single char>": <entry>, ... }
}
where <entry> is either a vocab string ("Return") or a 2-list ["ShiftLeft","<base>"] for shifted keys.

Target vocab names (rdev style, use EXACTLY these spellings): KeyA..KeyZ, Num0..Num9, Space,
Return, Backspace, Tab, Escape, ForwardDelete, Home, End, PageUp, PageDown, UpArrow, DownArrow,
LeftArrow, RightArrow, F1..F12, MetaLeft, ControlLeft, ShiftLeft, Alt, AltGr, CapsLock, Minus,
Equal, LeftBracket, RightBracket, SemiColon, Quote, BackQuote, BackSlash, Comma, Dot, Slash,
Insert, PrintScreen.

"modifiers" (lowercased alias -> vocab): ctrl, control, strg -> ControlLeft; shift -> ShiftLeft;
alt, option -> Alt; altgr -> AltGr; cmd, command, meta, super, win, windows, windows key, winkey,
windowskey -> MetaLeft.

"aliases" must cover at least (all keys lowercased; value shown after ->):
- enter, return -> Return; esc, escape -> Escape; backspace, back space -> Backspace; tab -> Tab
- space, spacebar, space bar -> Space
- delete, del, delete button, delete key, forward delete -> ForwardDelete
- up, up arrow, arrow up, up arrow key, arrowup, up_arrow -> UpArrow (same pattern for down/left/right,
  including "down arrow key", "arrow down", "down_arrow", "down" etc.)
- page down, pagedown, pgdn, page down key -> PageDown; page up, pageup, pgup -> PageUp
- home -> Home; end -> End; insert -> Insert
- print screen, prt sc, prtsc, prtscn, printscreen -> PrintScreen
- f1..f12 -> F1..F12
- caps lock, capslock -> CapsLock
- comma -> Comma; dot, period -> Dot; minus, dash, hyphen -> Minus; plus -> ["ShiftLeft","Equal"];
  equals, equal -> Equal; slash -> Slash; backslash -> BackSlash; semicolon -> SemiColon;
  quote, apostrophe -> Quote; backquote, backtick, grave -> BackQuote
- single punctuation aliases: "." -> Dot, "," -> Comma, "-" -> Minus, "=" -> Equal, "/" -> Slash,
  "\\" -> BackSlash, ";" -> SemiColon, "'" -> Quote, "`" -> BackQuote, "[" -> LeftBracket,
  "]" -> RightBracket, "*" -> ["ShiftLeft","Num8"]
- also map the modifier words themselves in aliases (ctrl -> ControlLeft etc.) so bare PRESS "Ctrl" works.
DO NOT add single letters or digits to aliases (the converter handles those programmatically).

"chars" must cover every printable ASCII punctuation char for US layout typing:
unshifted: `-=[]\;',./ -> BackQuote Minus Equal LeftBracket RightBracket BackSlash SemiColon Quote
Comma Dot Slash; space -> Space.
shifted (2-list with ShiftLeft): ~!@#$%^&*()_+{}|:"<>? -> BackQuote Num1 Num2 Num3 Num4 Num5 Num6
Num7 Num8 Num9 Num0 Minus Equal LeftBracket RightBracket BackSlash SemiColon Quote Comma Dot Slash.

Also write a short validation script `validate_key_map.py` that loads the JSON, asserts every value
resolves to the target vocab list above, asserts the chars section covers all of
string.punctuation, and prints a coverage report of `census_prelim/census.json` key_name_vocab
values (for PRESS/HOTKEY/KEY_DOWN/KEY_UP): how many occurrences resolve through
modifiers/aliases (after lowercasing + whitespace collapse + strip of quotes) vs remain unmapped,
listing the top 30 unmapped. Run it and iterate until the only unmapped census values are genuine
garbage (e.g. "Project Proposal", "123456", "scholarships for international students",
"command + v: Featuring 100 crochet strip patterns for modern striped blankets.",
"Enter + left arrow" — multi-key sequences and typed-text-as-PRESS are handled by the converter's
chord/typing fallback, not the map). Aim for >= 95% occurrence-weighted coverage of PRESS and
KEY_DOWN/KEY_UP single-key values.
