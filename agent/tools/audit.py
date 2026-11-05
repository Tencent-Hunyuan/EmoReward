#!/usr/bin/env python3
"""Static source/HTML audit. No model imports, downloads, or training."""
import argparse
import ast
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit


class Document(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.ids, self.manifest = [], set(), []
        self.capture = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        self.links.extend(attrs[key] for key in ("href", "src") if attrs.get(key))
        if tag == "script" and attrs.get("id") == "emoreward-agent-manifest":
            self.capture = True

    def handle_endtag(self, tag):
        if tag == "script":
            self.capture = False

    def handle_data(self, data):
        if self.capture:
            self.manifest.append(data)


def definitions(tree, prefix=""):
    found = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            name = prefix + node.name
            found.add(name)
            if isinstance(node, ast.ClassDef):
                found.update(definitions(node, name + "."))
    return found


def constant(tree, name, owner=None):
    body = tree.body
    if owner:
        body = next(node.body for node in body if isinstance(node, ast.ClassDef) and node.name == owner)
    for node in body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return list(ast.literal_eval(node.value))
    raise ValueError("Constant not found: " + name)


def audit(root):
    root = root.resolve()
    checks = []

    def record(name, passed, detail):
        checks.append({"check": name, "status": "pass" if passed else "fail", "detail": str(detail)})

    agent = root / "agent"
    hub = Document()
    hub.feed((agent / "index.html").read_text(encoding="utf-8"))
    manifest = json.loads("".join(hub.manifest))
    record("manifest-format", manifest.get("format") == "emoreward-agent-hub/v1", manifest.get("format"))
    for row in manifest["sources"]:
        path = (root / row["path"]).resolve()
        exists = path.is_relative_to(root) and path.is_file()
        record("source:" + row["id"], exists, row["path"])
        if not exists:
            continue
        record("fingerprint:" + row["id"], hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], row["path"])
        if row["symbols"]:
            found = definitions(ast.parse(path.read_text(encoding="utf-8")))
            missing = sorted(set(row["symbols"]) - found)
            record("symbols:" + row["id"], not missing, missing or "all source symbols found")
    for rel in manifest["documents"]:
        record("document:" + rel, (agent / rel).is_file(), rel)
    for route in manifest["routes"]:
        record("skill:" + route["skill"], (agent / route["skill"]).is_file(), route["intent"])
    for path in sorted(agent.rglob("*")):
        if path.suffix not in (".html", ".md"):
            continue
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".html":
            parser = Document()
            parser.feed(text)
            links = parser.links
        else:
            links = re.findall(r"\[[^\]]*\]\(([^)]+)\)", text)
        for link in links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            target = (path.parent / unquote(url.path)).resolve() if url.path else path
            valid = target.is_relative_to(root) and target.is_file()
            if valid and url.fragment and target.suffix == ".html":
                doc = Document()
                doc.feed(target.read_text(encoding="utf-8"))
                valid = unquote(url.fragment) in doc.ids
            record("link:" + str(path.relative_to(root)), valid, link)
    scorer = ast.parse((root / "hyvideo/models/reward_models/emo_reward/react_reward.py").read_text())
    exporter = ast.parse((root / "scripts/export_jit_gate_table.py").read_text())
    memory = ast.parse((root / "hyvideo/models/reward_models/emo_reward/reward_memory.py").read_text())
    runtime_keys = constant(scorer, "_GATE_TOOLS", "ReflectionRewarder")
    record("runtime-exporter-provider-keys", runtime_keys == constant(exporter, "GATE_TOOLS") == manifest["gate_tools"], runtime_keys)
    record("analysis-provider-keys", constant(memory, "TOOL_KEYS") == manifest["analysis_tools"], manifest["analysis_tools"])
    return {"format": "emoreward-static-audit/v1", "status": "pass" if all(row["status"] == "pass" for row in checks) else "fail", "source_commit": manifest["source_commit"], "checks": checks, "real_scoring": "not_run", "host_training": "not_run"}


def render(report):
    rows = "\n".join("<tr>" + "".join("<td>" + html.escape(str(row[key])) + "</td>" for key in ("check", "status", "detail")) + "</tr>" for row in report["checks"])
    data = json.dumps(report, ensure_ascii=False, indent=2).replace("<", "\\u003c")
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><title>EmoReward static audit</title></head><body>'
            '<h1>EmoReward static audit: ' + report["status"] + '</h1><p>Source/HTML checks only. Real scoring and host training: not_run.</p>'
            '<table><thead><tr><th>Check</th><th>Status</th><th>Evidence</th></tr></thead><tbody>' + rows + '</tbody></table>'
            '<script type="application/json" id="emoreward-audit">' + data + '</script></body></html>\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2], help="Upstream EmoReward checkout root")
    parser.add_argument("--output", default="-", help="HTML report path, or - for stdout")
    args = parser.parse_args()
    try:
        report = audit(args.repo_root.resolve())
    except (OSError, ValueError, KeyError, SyntaxError, StopIteration) as exc:
        report = {"format": "emoreward-static-audit/v1", "status": "fail", "checks": [{"check": "audit-input", "status": "fail", "detail": str(exc)}], "real_scoring": "not_run", "host_training": "not_run"}
    output = render(report)
    if args.output == "-":
        sys.stdout.write(output)
    else:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(output, encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": len(report["checks"]), "real_scoring": "not_run", "host_training": "not_run"}), file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
