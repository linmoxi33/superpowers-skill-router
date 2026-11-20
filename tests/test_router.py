import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "bin" / "skill_router.py"
spec = importlib.util.spec_from_file_location("skill_router", MODULE_PATH)
sr = importlib.util.module_from_spec(spec)
sys.modules["skill_router"] = sr
spec.loader.exec_module(sr)

REAL_SKILLS = ROOT / "examples" / "in" / "skills"
REAL_QUERIES = ROOT / "examples" / "in" / "real_prompts.jsonl"


def write_skill(root: Path, name: str, description: str, body: str = "body") -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {json.dumps(description)}\n---\n\n# {name}\n\n{body}\n",
        encoding="utf-8",
    )
    return d


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(MODULE_PATH), *args],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )


class TestFrontmatter(unittest.TestCase):
    def test_plain_and_quoted(self):
        meta = sr.parse_frontmatter('---\nname: foo\ndescription: "Use when a: b"\n---\nbody')
        self.assertEqual(meta["name"], "foo")
        self.assertEqual(meta["description"], "Use when a: b")

    def test_block_scalar_folded(self):
        meta = sr.parse_frontmatter(
            "---\nname: foo\ndescription: >\n  Use when testing\n  things\n---\n"
        )
        self.assertEqual(meta["description"], "Use when testing things")

    def test_missing_frontmatter_raises(self):
        with self.assertRaises(ValueError):
            sr.parse_frontmatter("# no frontmatter\n")

    def test_unclosed_frontmatter_raises(self):
        with self.assertRaises(ValueError):
            sr.parse_frontmatter("---\nname: foo\n")


class TestTokens(unittest.TestCase):
    def test_stopwords_and_stemming(self):
        self.assertEqual(sr.tokenize("Use when implementing any features"), ["implement", "feature"])

    def test_bigrams(self):
        self.assertEqual(sr.bigrams(["a", "b", "c"]), {("a", "b"), ("b", "c")})

    def test_strength_marker(self):
        self.assertGreater(sr.strength("You MUST use this"), 1.0)
        self.assertEqual(sr.strength("Use when debugging"), 1.0)


class TestLoader(unittest.TestCase):
    def test_missing_description_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "x"
            d.mkdir()
            (d / "SKILL.md").write_text("---\nname: x\n---\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                sr.load_skills(Path(tmp))

    def test_duplicate_names_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_skill(root, "a", "one two")
            (root / "b").mkdir()
            (root / "b" / "SKILL.md").write_text(
                "---\nname: a\ndescription: three four\n---\n", encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                sr.load_skills(root)


class TestRouter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write_skill(self.root, "alpha-lint", "Use when linting alpha configuration files")
        write_skill(self.root, "beta-deploy", "Use when deploying beta services to production")
        self.skills = sr.load_skills(self.root)
        self.router = sr.Router(self.skills)

    def tearDown(self):
        self.tmp.cleanup()

    def test_ranking_is_deterministic(self):
        first = self.router.score("linting alpha configuration")
        second = self.router.score("linting alpha configuration")
        self.assertEqual([(s.name, v) for s, v in first], [(s.name, v) for s, v in second])

    def test_explicit_name_wins(self):
        ranked = self.router.score("beta deploy please")
        self.assertEqual(ranked[0][0].name, "beta-deploy")

    def test_derive_trigger_masks_name(self):
        skill = next(s for s in self.skills if s.name == "alpha-lint")
        query = sr.derive_trigger_query(skill)
        self.assertNotIn("alpha", query.lower())
        self.assertIn("linting", query.lower())

    def test_analyse_summary_and_digest(self):
        report = sr.analyse(self.skills)
        self.assertEqual(report["summary"]["skills"], 2)
        self.assertEqual(report["summary"]["queries"], 2)
        self.assertTrue(report["digest"].startswith("sha256:"))
        self.assertEqual(report["digest"], sr.digest(report))
        again = sr.analyse(self.skills)
        self.assertEqual(report["digest"], again["digest"])

    def test_unreachable_detection(self):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        write_skill(root, "gamma-one", "Use when validating gamma widgets")
        write_skill(root, "gamma-two", "Use when validating gamma widgets")
        report = sr.analyse(sr.load_skills(root))
        self.assertEqual(report["summary"]["unreachable"], 1)
        self.assertEqual(len(report["unreachable"]), 1)
        tmp.cleanup()

    def test_collision_flag_on_equal_scores(self):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        write_skill(root, "delta-a", "Use when checking delta payloads")
        write_skill(root, "delta-b", "Use when checking delta payloads")
        report = sr.analyse(sr.load_skills(root))
        self.assertGreaterEqual(report["summary"]["collisions"], 1)
        tmp.cleanup()


class TestCli(unittest.TestCase):
    def test_exit_zero_on_clean_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_skill(root, "only-skill", "Use when doing the only job in this fixture")
            proc = run_cli("--skills", str(root), "--json")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(json.loads(proc.stdout)["summary"]["unreachable"], 0)

    def test_exit_one_and_json_parses_on_real_fixture(self):
        proc = run_cli("--skills", str(REAL_SKILLS), "--queries", str(REAL_QUERIES), "--json")
        self.assertEqual(proc.returncode, 1, proc.stderr)
        report = json.loads(proc.stdout)
        self.assertEqual(report["summary"]["skills"], 15)
        self.assertGreaterEqual(report["summary"]["queries"], 20)

    def test_exit_two_on_missing_dir(self):
        proc = run_cli("--skills", "/nonexistent/skills")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("error:", proc.stderr)

    def test_exit_two_on_bad_query_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            q = Path(tmp) / "q.jsonl"
            q.write_text("{not json}\n", encoding="utf-8")
            proc = run_cli("--skills", str(REAL_SKILLS), "--queries", str(q))
            self.assertEqual(proc.returncode, 2)

    def test_only_unreachable_output(self):
        proc = run_cli("--skills", str(REAL_SKILLS), "--only-unreachable")
        # Derived self-retrieval queries leave every skill reachable on this fixture.
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("COLLISIONS", proc.stdout)

    def test_report_file_written_and_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1, p2 = Path(tmp) / "r1.json", Path(tmp) / "r2.json"
            run_cli("--skills", str(REAL_SKILLS), "--queries", str(REAL_QUERIES), "--report", str(p1))
            run_cli("--skills", str(REAL_SKILLS), "--queries", str(REAL_QUERIES), "--report", str(p2))
            r1 = json.loads(p1.read_text())
            r2 = json.loads(p2.read_text())
            self.assertEqual(r1["digest"], r2["digest"])

    def test_real_corpus_explicit_name_accuracy(self):
        report = sr.analyse(
            sr.load_skills(REAL_SKILLS),
            extra_queries=sr.load_queries(REAL_QUERIES),
        )
        s = report["summary"]
        self.assertGreaterEqual(s["expected_hits"], 18)
        self.assertEqual(s["expected_total"], 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
