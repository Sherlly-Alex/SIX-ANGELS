#!/usr/bin/env python3
"""Export simulation Markdown summary and state-machine timing into XLSX."""
from __future__ import annotations
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG = "http://schemas.openxmlformats.org/package/2006/relationships"
ET.register_namespace("", MAIN)
ET.register_namespace("r", REL)

def q(name):
    return "{" + MAIN + "}" + name

def col(index):
    result = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        result = chr(65 + rem) + result
    return result

def markdown_table(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and not all(re.fullmatch(r"[: -]+", cell) for cell in cells):
            rows.append(cells)
    if not rows:
        raise ValueError("no table in " + str(path))
    return rows[0], rows[1:]

EVENT = re.compile(r"^\[[A-Z]+\] \[([0-9.]+)\].*controller=([a-z_]+) task=([0-9]+) attempt=([0-9]+) stage=([a-z_\-]+) score=([0-9]+): (.*)$")

def parse_run(run_dir):
    client = run_dir / "client.log"
    if not client.exists():
        return [], [], []
    raw = []
    for line in client.read_text(encoding="utf-8", errors="replace").splitlines():
        match = EVENT.match(line)
        if match:
            raw.append({"time": float(match.group(1)), "kind": match.group(2), "task": int(match.group(3)), "attempt": int(match.group(4)), "stage": match.group(5), "score": int(match.group(6)), "message": match.group(7)})
    if not raw:
        return [], [], []
    base = raw[0]["time"]
    entries = [item for item in raw if item["kind"] == "executing_stage" and item["stage"] != "-" and " entering " in item["message"]]
    terminals = [item for item in raw if item["kind"] in {"waiting_for_referee", "blocked", "finished"}]
    timeline = []
    task_groups = defaultdict(list)
    for index, item in enumerate(entries):
        key = (item["task"], item["attempt"])
        next_time = None
        for candidate in entries[index + 1:]:
            if (candidate["task"], candidate["attempt"]) == key:
                next_time = candidate["time"]
                break
        if next_time is None:
            for candidate in terminals:
                if (candidate["task"], candidate["attempt"]) == key and candidate["time"] > item["time"]:
                    next_time = candidate["time"]
                    break
        duration = "" if next_time is None else round(next_time - item["time"], 3)
        timeline.append([run_dir.name, item["task"], item["attempt"], item["stage"], round(item["time"] - base, 3), duration, item["score"], item["message"]])
        task_groups[key].append(item)
    summaries = []
    for (task, attempt), items in sorted(task_groups.items()):
        start = items[0]["time"]
        related = [item for item in terminals if (item["task"], item["attempt"]) == (task, attempt) and item["time"] >= start]
        terminal = related[0] if related else None
        end = terminal["time"] if terminal else items[-1]["time"]
        outcome = terminal["kind"] if terminal else "in_progress"
        score = max(item["score"] for item in raw if item["task"] == task)
        summaries.append([run_dir.name, task, attempt, round(start - base, 3), round(end - base, 3), round(end - start, 3), len(items), score, outcome, "" if terminal is None else terminal["message"]])
    faults = []
    for line_no, line in enumerate(client.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if "runtime input stale" in line:
            faults.append([run_dir.name, "runtime_input_stale", line_no, line])
        elif "controller=blocked" in line:
            faults.append([run_dir.name, "state_machine_blocked", line_no, line])
    return timeline, summaries, faults

def sheet_xml(headers, rows):
    root = ET.Element(q("worksheet"))
    data = ET.SubElement(root, q("sheetData"))
    for row_no, values in enumerate([headers] + rows, start=1):
        row = ET.SubElement(data, q("row"), {"r": str(row_no)})
        for col_no, value in enumerate(values):
            attrs = {"r": col(col_no) + str(row_no)}
            if row_no == 1:
                attrs["s"] = "1"
            value = str(value)
            if re.fullmatch(r"-?[0-9]+(?:[.][0-9]+)?", value):
                cell = ET.SubElement(row, q("c"), attrs)
                ET.SubElement(cell, q("v")).text = value
            else:
                attrs["t"] = "inlineStr"
                cell = ET.SubElement(row, q("c"), attrs)
                ET.SubElement(ET.SubElement(cell, q("is")), q("t")).text = value
    cols = ET.SubElement(root, q("cols"))
    for i, header in enumerate(headers, start=1):
        width = min(max(12, len(str(header)) + 4), 55)
        ET.SubElement(cols, q("col"), {"min": str(i), "max": str(i), "width": str(width), "customWidth": "1"})
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)

def write_xlsx(markdown, output):
    headers, summary = markdown_table(markdown)
    timeline, tasks, faults = [], [], []
    for run_dir in sorted(markdown.parent.glob("run_*")):
        tl, ts, fs = parse_run(run_dir)
        timeline += tl
        tasks += ts
        faults += fs
    books = [
        ("Run Summary", headers, summary),
        ("State Timeline", ["Run", "Task", "Attempt", "Stage", "Start from client(s)", "Stage duration(s)", "Score at entry", "Transition message"], timeline),
        ("Task Summary", ["Run", "Task", "Attempt", "Start(s)", "End(s)", "Duration(s)", "Stages", "Max score", "Outcome", "Terminal message"], tasks),
        ("Protection Events", ["Run", "Type", "Client log line", "Message"], faults),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    types = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>', '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">', '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>', '<Default Extension="xml" ContentType="application/xml"/>', '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>', '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>']
    workbook_sheets, relations = [], []
    for index, (name, _, _) in enumerate(books, start=1):
        types.append('<Override PartName="/xl/worksheets/sheet%d.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' % index)
        workbook_sheets.append('<sheet name="%s" sheetId="%d" r:id="rId%d"/>' % (name, index, index))
        relations.append('<Relationship Id="rId%d" Type="%s/worksheet" Target="worksheets/sheet%d.xml"/>' % (index, REL, index))
    types.append('</Types>')
    workbook = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="%s" xmlns:r="%s"><sheets>%s</sheets></workbook>' % (MAIN, REL, "".join(workbook_sheets))
    wb_rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="%s">%s<Relationship Id="rId%d" Type="%s/styles" Target="styles.xml"/></Relationships>' % (PKG, "".join(relations), len(books) + 1, REL)
    styles = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="%s"><fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts><fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills><borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0"/></cellXfs></styleSheet>' % MAIN
    root_rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="%s"><Relationship Id="rId1" Type="%s/officeDocument" Target="xl/workbook.xml"/></Relationships>' % (PKG, REL)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "".join(types))
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", wb_rels)
        archive.writestr("xl/styles.xml", styles)
        for index, (_, headers, rows) in enumerate(books, start=1):
            archive.writestr("xl/worksheets/sheet%d.xml" % index, sheet_xml(headers, rows))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: export_simulation_xlsx.py REPORT.md REPORT.xlsx")
    write_xlsx(Path(sys.argv[1]), Path(sys.argv[2]))
