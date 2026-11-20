#!/usr/bin/env python3
"""superpowers-skill-router: deterministic routing-reachability audit for Agent Skills.

The tool parses ``SKILL.md`` frontmatter (the same ``name`` / ``description``
fields a host agent uses to auto-trigger a skill), builds a deterministic
lexical proxy of description matching, and reports two failure modes that the
upstream project reports but cannot measure:

* unreachable skills  -- a skill that never ranks first for its own trigger text
* routing collisions  -- two skills that tie on the same request

It is a *comparative* analyser, not a model of any particular vendor LLM.
Scores are lexical (IDF-weighted token overlap plus phrase bonuses); every
number is reproducible offline from local files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

TOOL = "superpowers-skill-router"
VERSION = "1.0.0"
EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_USAGE = 2

STOPWORDS = frozenset(
    """a an and any are as at be before but by can could do does for from had has
    have how i if in into is it its just like may me more most must no not of on
    or other our out over same should so some such than that the their them then
    there these they this those to too up use used uses using very was we were
    what when where which while who why will with without you your""".split()
)
STRENGTH_MARKERS = ("must", "always", "required", "never")


# --------------------------------------------------------------------------
# frontmatter
# --------------------------------------------------------------------------
def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        inner = value[1:-1]
        if value[0] == '"':
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                inner = inner.replace('\\"', '"').replace("\\\\", "\\")
        return inner.replace("''", "'")
    return value


def parse_frontmatter(text: str) -> dict:
    """Parse the YAML frontmatter subset used by SKILL.md files.

    Supports plain scalars, quoted strings, and folded/literal block scalars
    (``>``, ``|`` and their ``-`` chomping variants). Raises ValueError when
    the document has no frontmatter block.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("missing frontmatter opening '---'")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        raise ValueError("missing frontmatter closing '---'") from None

    data: dict[str, str] = {}
    i = 1
    while i < end:
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        if ":" not in line:
            raise ValueError(f"invalid frontmatter line {i + 1}: {line!r}")
        key, _, raw = line.partition(":")
        key = key.strip()
        raw = raw.strip()
        if raw in (">", "|", ">-", "|-", ">+", "|+"):
            block: list[str] = []
            i += 1
            while i < end and (not lines[i].strip() or lines[i][:1].isspace()):
                block.append(lines[i].strip())
                i += 1
            sep = "\n" if raw.startswith("|") else " "
            data[key] = sep.join(b for b in block if b).strip()
            continue
        if raw == "":
            # plain multi-line continuation
            cont: list[str] = []
            i += 1
            while i < end and lines[i][:1].isspace() and lines[i].strip():
                cont.append(lines[i].strip())
                i += 1
            data[key] = " ".join(cont)
            continue
        data[key] = _unquote(raw)
        i += 1
    return data


# --------------------------------------------------------------------------
# tokenisation / scoring
# --------------------------------------------------------------------------
def stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for raw in re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", text.lower()):
        tok = stem(raw.replace("'s", "").strip("'-"))
        if len(tok) < 2 or tok in STOPWORDS or tok.isdigit():
            continue
        out.append(tok)
    return out


def bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}


def strength(description: str) -> float:
    low = description.lower()
    return 1.15 if any(re.search(rf"\b{m}\b", low) for m in STRENGTH_MARKERS) else 1.0


@dataclass
class Skill:
    name: str
    description: str
    path: str
    tokens: list[str]
    phrases: set[tuple[str, str]]
    weight: float
    tf: dict[str, int]

    @property
    def token_set(self) -> set[str]:
        return set(self.tokens)


def load_skills(skills_dir: Path) -> list[Skill]:
    if not skills_dir.is_dir():
        raise ValueError(f"skills directory not found: {skills_dir}")
    skills: list[Skill] = []
    for path in sorted(skills_dir.glob("*/SKILL.md")):
        try:
            meta = parse_frontmatter(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
        name = meta.get("name", "").strip()
        desc = meta.get("description", "").strip()
        if not name:
            raise ValueError(f"{path}: missing required 'name'")
        if not desc:
            raise ValueError(f"{path}: missing required 'description'")
        # A host matches both the skill name and its description; explicit name
        # mentions are repeated twice so they outrank generic description words.
        toks = tokenize(desc) + tokenize(name) + tokenize(name)
        tf: dict[str, int] = {}
        for tok in toks:
            tf[tok] = tf.get(tok, 0) + 1
        skills.append(Skill(name, desc, str(path), toks, bigrams(toks), strength(desc), tf))
    if not skills:
        raise ValueError(f"no */SKILL.md found under {skills_dir}")
    names = [s.name for s in skills]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"duplicate skill names: {', '.join(dupes)}")
    return skills


class Router:
    """Deterministic IDF-weighted lexical router over skill descriptions."""

    def __init__(self, skills: list[Skill]):
        self.skills = skills
        df: dict[str, int] = {}
        for s in skills:
            for tok in s.token_set:
                df[tok] = df.get(tok, 0) + 1
        n = len(skills)
        self.idf = {t: round(math.log((n + 1) / (c + 0.5)) + 1.0, 6) for t, c in df.items()}
        self.default_idf = round(math.log((n + 1) / 0.5) + 1.0, 6)

    def score(self, query: str) -> list[tuple[Skill, float]]:
        q = tokenize(query)
        if not q:
            return []
        qb = bigrams(q)
        counts: dict[str, int] = {}
        for tok in q:
            counts[tok] = counts.get(tok, 0) + 1
        qset = set(q)
        ranked: list[tuple[Skill, float]] = []
        for skill in self.skills:
            stoks = skill.token_set
            total = 0.0
            for tok, qtf in counts.items():
                if tok in stoks:
                    doc_tf = skill.tf.get(tok, 1)
                    total += (
                        self.idf.get(tok, self.default_idf)
                        * (1.0 + 0.5 * (doc_tf - 1))
                        * (1.0 + 0.25 * (qtf - 1))
                    )
            total += 0.75 * len(qb & skill.phrases)
            # An explicit name mention is the strongest routing signal a host
            # has; apply it only when *this* skill's name is in the query.
            name_toks = set(tokenize(skill.name))
            if name_toks and name_toks <= qset:
                total *= 1.35
            ranked.append((skill, round(total * skill.weight, 4)))
        ranked.sort(key=lambda item: (-item[1], item[0].name))
        return [(s, sc) for s, sc in ranked if sc > 0]


# --------------------------------------------------------------------------
# trigger queries
# --------------------------------------------------------------------------
_TRIGGER_START = re.compile(
    r"(?:you\s+must\s+use\s+this\s+before|use\s+when|use\s+this\s+when|"
    r"use\s+before|use\s+after|use\s+for|use\s+if)",
    re.IGNORECASE,
)
_TRIGGER_STOP = re.compile(r"\s+(?:-|—|--)\s+|;\s*")


def _mask_name(text: str, name: str) -> str:
    parts = [name] + [p for p in re.split(r"[-_]", name) if len(p) > 2]
    for part in sorted(set(parts), key=len, reverse=True):
        text = re.sub(rf"(?i)\b{re.escape(part)}\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def derive_trigger_query(skill: Skill, max_tokens: int = 16) -> str:
    """Build the self-retrieval query from a skill's own trigger clause.

    The skill name is masked so the probe measures description semantics, not
    name leakage.
    """
    desc = _mask_name(skill.description, skill.name)
    match = _TRIGGER_START.search(desc)
    clause = desc[match.start():] if match else desc
    clause = _TRIGGER_STOP.split(clause)[0]
    toks = clause.split()
    if len(toks) > max_tokens:
        clause = " ".join(toks[:max_tokens])
    return clause.strip(" .,:") or _mask_name(skill.description, skill.name)


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------
def analyse(
    skills: list[Skill],
    extra_queries: list[dict] | None = None,
    min_margin: float = 0.15,
    top_k: int = 3,
    trigger_queries: bool = True,
) -> dict:
    router = Router(skills)
    queries: list[dict] = []
    if trigger_queries:
        for skill in skills:
            queries.append(
                {
                    "text": derive_trigger_query(skill),
                    "source": "derived:description",
                    "expected": skill.name,
                }
            )
    for item in extra_queries or []:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        queries.append(
            {
                "text": text,
                "source": item.get("source", "file"),
                "expected": item.get("expected", ""),
            }
        )

    results: list[dict] = []
    reachable: set[str] = set()
    collisions = 0
    margins: list[float] = []
    for q in queries:
        ranked = router.score(q["text"])
        top = ranked[:top_k]
        if not top:
            results.append({**q, "ranking": [], "top1": None, "margin": 0.0, "collision": False})
            continue
        first, second = top[0], (top[1] if len(top) > 1 else None)
        margin = 1.0 if second is None or first[1] <= 0 else round((first[1] - second[1]) / first[1], 4)
        both_strong = bool(second) and first[0].weight > 1.0 and second[0].weight > 1.0
        collision = bool(second) and (margin < min_margin or (both_strong and margin < 0.35))
        if collision:
            collisions += 1
        margins.append(margin)
        reachable.add(first[0].name)
        results.append(
            {
                **q,
                "ranking": [{"skill": s.name, "score": sc} for s, sc in top],
                "top1": first[0].name,
                "margin": margin,
                "collision": collision,
                "hit": first[0].name == q["expected"],
            }
        )

    unreachable = [s.name for s in skills if s.name not in reachable]
    suggestions: dict[str, list[str]] = {}
    winners = {r["top1"] for r in results if r["top1"]}
    for name in unreachable:
        skill = next(s for s in skills if s.name == name)
        rival_tokens: set[str] = set()
        for s in skills:
            if s.name in winners and s.name != name:
                rival_tokens |= s.token_set
        own = [t for t in dict.fromkeys(skill.tokens) if t not in rival_tokens]
        own.sort(key=lambda t: (-(router.idf.get(t, router.default_idf)), t))
        suggestions[name] = own[:5]

    report = {
        "tool": TOOL,
        "version": VERSION,
        "model": {
            "kind": "deterministic-lexical-proxy",
            "idf": "log((N+1)/(df+0.5))+1",
            "phrase_bonus": 0.75,
            "strength_multiplier": 1.15,
            "min_margin": min_margin,
        },
        "skills": [
            {
                "name": s.name,
                "description": s.description,
                "tokens": len(s.tokens),
                "weight": s.weight,
            }
            for s in skills
        ],
        "queries": results,
        "summary": {
            "skills": len(skills),
            "queries": len(results),
            "reachable": len(skills) - len(unreachable),
            "unreachable": len(unreachable),
            "collisions": collisions,
            "mean_margin": round(sum(margins) / len(margins), 4) if margins else 0.0,
            "expected_hits": sum(1 for r in results if r.get("hit")),
            "expected_total": sum(1 for r in results if r.get("expected")),
        },
        "unreachable": unreachable,
        "suggestions": suggestions,
    }
    report["digest"] = digest(report)
    return report


def canonical(obj: dict) -> str:
    payload = {k: v for k, v in obj.items() if k != "digest"}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(obj: dict) -> str:
    return "sha256:" + hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def load_queries(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"query file not found: {path}")
    items: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"{path}:{lineno}: expected a JSON object")
        items.append(obj)
    if not items:
        raise ValueError(f"{path}: no queries found")
    return items


def render_text(report: dict) -> str:
    s = report["summary"]
    lines = [
        f"{TOOL} {VERSION}",
        f"skills={s['skills']} queries={s['queries']} reachable={s['reachable']} "
        f"unreachable={s['unreachable']} collisions={s['collisions']} mean_margin={s['mean_margin']}",
        f"expected_hits={s['expected_hits']}/{s['expected_total']}",
        f"digest={report['digest']}",
    ]
    if report["unreachable"]:
        lines.append("")
        lines.append("UNREACHABLE (never ranked first for their own trigger text):")
        for name in report["unreachable"]:
            hint = ", ".join(report["suggestions"].get(name, [])) or "(no discriminative token)"
            lines.append(f"  - {name}  -> add distinctive terms: {hint}")
    colliding = [q for q in report["queries"] if q["collision"]]
    if colliding:
        lines.append("")
        lines.append(f"COLLISIONS (margin < {report['model']['min_margin']}):")
        for q in colliding:
            pair = " vs ".join(f"{r['skill']}({r['score']})" for r in q["ranking"][:2])
            lines.append(f"  - {q['top1']}  |  {pair}  |  {q['text'][:90]}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="skill_router.py",
        description="Deterministic routing-reachability audit for Agent Skills (SKILL.md).",
    )
    p.add_argument("--skills", required=True, help="directory containing <skill>/SKILL.md")
    p.add_argument("--queries", help="JSONL file of extra queries: {text, expected, source}")
    p.add_argument("--report", help="write the full JSON report to this path")
    p.add_argument("--json", action="store_true", help="print JSON to stdout")
    p.add_argument("--min-margin", type=float, default=0.15, help="collision margin threshold (default 0.15)")
    p.add_argument("--top", type=int, default=3, help="ranking depth (default 3)")
    p.add_argument("--no-trigger-queries", action="store_true", help="skip derived self-retrieval queries")
    p.add_argument("--only-unreachable", action="store_true", help="print only unreachable skills")
    p.add_argument("--version", action="version", version=f"{TOOL} {VERSION}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.min_margin < 0 or args.top < 1:
        print("error: --min-margin must be >= 0 and --top >= 1", file=sys.stderr)
        return EXIT_USAGE
    try:
        skills = load_skills(Path(args.skills))
        extra = load_queries(Path(args.queries)) if args.queries else []
        report = analyse(
            skills,
            extra_queries=extra,
            min_margin=args.min_margin,
            top_k=args.top,
            trigger_queries=not args.no_trigger_queries,
        )
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.only_unreachable:
        for name in report["unreachable"]:
            print(name)
    else:
        print(render_text(report))
    findings = report["summary"]["unreachable"] + report["summary"]["collisions"]
    return EXIT_FINDINGS if findings else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
