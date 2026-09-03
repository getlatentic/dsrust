#!/usr/bin/env python3
"""Measure expat's character classes from the pinned Python, one parse per code point.

Writes the ranges as a fixture and as the Rust tables `adapter/xml/names.rs` reads at runtime;
a unit test there holds the two equal.
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "crates/dsrust/tests/conformance/adapter/xml_names.json"
TABLES = ROOT / "crates/dsrust/src/adapter/xml/names.rs"
XML_WHITESPACE = {0x20, 0x9, 0xD, 0xA}


def accepted(text: str) -> bool:
    try:
        ET.fromstring(text)
    except ET.ParseError:
        return False
    return True


def ranges(points: list[int]) -> list[list[int]]:
    out: list[list[int]] = []
    for point in points:
        if out and out[-1][1] == point - 1:
            out[-1][1] = point
        else:
            out.append([point, point])
    return out


def measure() -> dict[str, list[list[int]]]:
    start, name, char = [], [], []
    for point in range(0x110000):
        if 0xD800 <= point <= 0xDFFF:
            continue
        c = chr(point)
        if accepted(f"<{c}/>"):
            start.append(point)
        if point not in XML_WHITESPACE and accepted(f"<a{c}/>"):
            name.append(point)
        if accepted(f"<a>&#{point};</a>"):
            char.append(point)
    return {"name_start": ranges(start), "name_char": ranges(name), "char": ranges(char)}


def rust_table(name: str, table: list[list[int]]) -> str:
    """The ranges, several to a line: one per line is 500 lines of data in a source tree that
    budgets 400, and a table of pairs reads no worse wrapped."""
    entries = [f"({lo:#x}, {hi:#x})" for lo, hi in table]
    lines, row = [], "   "
    for entry in entries:
        candidate = f"{row} {entry},"
        if len(candidate) > 98:
            lines.append(row)
            row = f"    {entry},"
        else:
            row = candidate
    lines.append(row)
    body = "\n".join(lines)
    # `rustfmt::skip` because this is data, not code: the formatter puts one pair per line, which
    # is 500 lines of table in a tree that budgets 400 per file.
    return f"#[rustfmt::skip]\npub(super) const {name}: &[(u32, u32)] = &[\n{body}\n];\n"


def main() -> None:
    tables = measure()
    FIXTURE.write_text(json.dumps(tables, separators=(",", ":")) + "\n")
    body = "\n".join(
        (
            "//! expat's character classes, measured from the pinned Python by",
            "//! `scripts/generate_xml_name_tables.py`: inclusive code-point ranges.",
            "",
            rust_table("NAME_START", tables["name_start"]),
            rust_table("NAME_CHAR", tables["name_char"]),
            rust_table("CHAR", tables["char"]),
            "/// Whether `point` falls in one of `table`'s inclusive ranges.",
            "pub(super) fn contains(table: &[(u32, u32)], point: char) -> bool {",
            "    let point = point as u32;",
            "    table",
            "        .binary_search_by(|(lo, hi)| {",
            "            if *hi < point {",
            "                std::cmp::Ordering::Less",
            "            } else if *lo > point {",
            "                std::cmp::Ordering::Greater",
            "            } else {",
            "                std::cmp::Ordering::Equal",
            "            }",
            "        })",
            "        .is_ok()",
            "}",
            "",
            "#[cfg(test)]",
            "mod tests {",
            "    use super::*;",
            "",
            "    #[test]",
            "    fn the_tables_are_the_measured_ranges() {",
            "        let fixture: serde_json::Value =",
            '            serde_json::from_str(include_str!("../../../tests/conformance/adapter/xml_names.json"))',
            '                .expect("fixture parses");',
            '        for (name, table) in [("name_start", NAME_START), ("name_char", NAME_CHAR), ("char", CHAR)] {',
            "            let measured: Vec<(u32, u32)> = fixture[name]",
            "                .as_array()",
            '                .expect("ranges")',
            "                .iter()",
            "                .map(|pair| (pair[0].as_u64().unwrap_or(0) as u32, pair[1].as_u64().unwrap_or(0) as u32))",
            "                .collect();",
            '            assert_eq!(table, measured.as_slice(), "{name}");',
            "        }",
            "    }",
            "",
            "    #[test]",
            "    fn membership_follows_the_ranges() {",
            "        assert!(contains(NAME_START, 'a') && !contains(NAME_START, '1') && !contains(NAME_START, ':'));",
            "        assert!(contains(NAME_CHAR, '1') && contains(NAME_CHAR, '-') && !contains(NAME_CHAR, ' '));",
            "        assert!(contains(CHAR, '\\u{9}') && !contains(CHAR, '\\u{1}') && !contains(CHAR, '\\u{ffff}'));",
            "    }",
            "}",
        )
    )
    TABLES.write_text(body + "\n")
    for name, table in tables.items():
        print(name, len(table), "ranges", file=sys.stderr)


if __name__ == "__main__":
    main()
