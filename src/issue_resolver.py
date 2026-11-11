#!/usr/bin/env python3
"""
issue_resolver.py - resolves the oldest open GitHub issue with an LLM agent.

Flow:
  1. Reads secrets and settings from ./.secrets (KEY=VALUE per line).
  2. Reads the coding instructions from ./.openai/openai-instructions.json and
     sends them to the model verbatim inside <instructions>...</instructions>,
     followed by the issue.
  3. Uses the local clone given on the command line; the GitHub repository
     (owner/name) is taken from its `origin` remote. Refuses to run if
     ./.secrets is inside that repository or the clone has uncommitted changes.
  4. Finds the oldest open issue that has no open fix pull request yet.
  5. Fetches origin and creates branch issue-<n>-fix from
     origin/<base branch> (default: dev).
  6. Runs a tool-calling loop against any OpenAI-compatible server
     (OpenAI, llama.cpp llama-server, Ollama, LM Studio). The model can list,
     read, search, write, edit and delete files inside the project and run
     pytest.
  7. If nothing changed, deletes the branch and stops. Otherwise stages
     everything (minus caches/databases), commits "Fix #<n>: <title>",
     pushes, opens a PR into the base branch with "Closes #<n>" and comments
     on the issue with the PR link.

.secrets keys
  GITHUB_TOKEN       required
  GITHUB_REPOSITORY  optional owner/name; must match the clone's origin
  OPENAI_API_KEY     optional for local servers (any value works)
  OPENAI_BASE_URL    e.g. http://localhost:8080/v1 for llama-server
                     (OPENAI_API_BASE is accepted too)
  OPENAI_MODEL       model name the server expects
  BASE_BRANCH        default: dev
  MAX_STEPS          default: 60
Each setting can also come from an environment variable of the same name or a
command-line flag (flags win, then environment, then .secrets).

Usage:
    python issue_resolver.py --repo path/to/project-clone [--dry-run] [--no-pytest]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests
from openai import OpenAI

GITHUB_API = "https://api.github.com"
SECRETS_FILE = ".secrets"
INSTRUCTIONS_FILE = Path(".openai") / "openai-instructions.json"

MAX_READ_CHARS = 100_000
MAX_LIST_ENTRIES = 2_000
MAX_SEARCH_HITS = 200
MAX_TOOL_OUTPUT = 30_000
PYTEST_TIMEOUT = 300

# Files the model must never touch inside the project.
PROTECTED_NAMES = {".secrets"}
PROTECTED_DIRS = {".git", ".openai"}

# New files matching these are never committed (caches, databases, envs).
JUNK_PATTERNS = [
    r"(^|/)__pycache__/", r"(^|/)\.pytest_cache/", r"\.py[co]$",
    r"\.(db|sqlite|sqlite3)$", r"(^|/)instance/", r"\.log$",
    r"(^|/)\.?venv/", r"(^|/)\.DS_Store$",
]


class AgentError(Exception):
    """Fatal error: the resolver stops."""


class ToolError(Exception):
    """Error reported back to the model; the run continues."""


def log(msg: str) -> None:
    print(f"[resolver] {msg}", flush=True)


def branch_name(number: int) -> str:
    return f"issue-{number}-fix"


# --------------------------------------------------------------------------
# Secrets, settings and instructions
# --------------------------------------------------------------------------

def load_secrets(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines. Blank lines, '#' comments and 'export ' are ok."""
    if not path.is_file():
        raise AgentError(f"{path} not found; create it with GITHUB_TOKEN=... in it")
    if os.name == "posix" and path.stat().st_mode & 0o077:
        log(f"warning: {path} is readable by other users; run: chmod 600 {path}")

    secrets: dict[str, str] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            raise AgentError(f"{path}:{lineno}: expected KEY=VALUE")
        key, value = (part.strip() for part in line.split("=", 1))
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        secrets[key] = value
    return secrets


@dataclass
class Settings:
    token: str
    repository: str
    api_key: str
    base_url: str | None
    model: str
    base_branch: str
    max_steps: int
    allow_pytest: bool
    dry_run: bool


def load_settings(args: argparse.Namespace, secrets: dict[str, str]) -> Settings:
    def pick(cli: Any, key: str, default: Any = None) -> Any:
        if cli not in (None, ""):
            return cli
        return os.environ.get(key) or secrets.get(key) or default

    token = secrets.get("GITHUB_TOKEN", "")
    if not token:
        raise AgentError(f"GITHUB_TOKEN is missing from {SECRETS_FILE}")

    repository = str(pick(args.github_repo, "GITHUB_REPOSITORY", "")).strip()
    if repository and not re.fullmatch(r"[\w.-]+/[\w.-]+", repository):
        raise AgentError(f"GITHUB_REPOSITORY must look like owner/name, got {repository!r}")

    base_url = pick(args.base_url, "OPENAI_BASE_URL") or pick(None, "OPENAI_API_BASE")
    api_key = pick(None, "OPENAI_API_KEY")
    if not api_key:
        if not base_url:
            raise AgentError("set OPENAI_API_KEY, or OPENAI_BASE_URL for a local server")
        api_key = "local"           # local servers accept any key

    return Settings(
        token=token,
        repository=repository,
        api_key=api_key,
        base_url=base_url,
        model=pick(args.model, "OPENAI_MODEL", "gpt-4o"),
        base_branch=pick(args.base_branch, "BASE_BRANCH", "dev"),
        max_steps=int(pick(args.max_steps, "MAX_STEPS", 60)),
        allow_pytest=not args.no_pytest,
        dry_run=args.dry_run,
    )


def load_instructions(path: Path) -> str:
    """Return the instructions file as pretty JSON text (validated)."""
    if not path.is_file():
        raise AgentError(f"{path} not found")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise AgentError(f"{path} is not valid JSON: {e}") from e
    return json.dumps(data, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------
# git
# --------------------------------------------------------------------------

def clean_env() -> dict[str, str]:
    """Environment without anything that looks like a credential."""
    return {k: v for k, v in os.environ.items()
            if not re.search(r"TOKEN|SECRET|PASSWORD|API_KEY", k, re.I)}


def git(args: list[str], cwd: Path, token: str | None = None,
        check: bool = True) -> subprocess.CompletedProcess:
    """Run git. The token, if given, reaches git only via env for this call."""
    env = clean_env()
    env["GIT_TERMINAL_PROMPT"] = "0"
    if token:
        cred = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        env["GIT_CONFIG_COUNT"] = "1"
        env["GIT_CONFIG_KEY_0"] = "http.https://github.com/.extraheader"
        env["GIT_CONFIG_VALUE_0"] = f"AUTHORIZATION: basic {cred}"
    result = subprocess.run(["git", *args], cwd=cwd, env=env,
                            text=True, capture_output=True)
    if check and result.returncode != 0:
        raise AgentError(f"git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result


def remote_to_full_name(url: str) -> str | None:
    m = re.search(r"github\.com[:/]+([^/\s]+)/([^/\s]+?)(?:\.git)?/?$", url.strip())
    return f"{m.group(1)}/{m.group(2)}".lower() if m else None


def repo_remotes(path: Path) -> set[str]:
    out = git(["remote", "-v"], cwd=path, check=False).stdout
    return {name for line in out.splitlines()
            if len(parts := line.split()) >= 2
            and (name := remote_to_full_name(parts[1]))}


def git_toplevel(path: Path) -> Path | None:
    r = git(["rev-parse", "--show-toplevel"], cwd=path, check=False)
    return Path(r.stdout.strip()).resolve() if r.returncode == 0 else None


def assert_secrets_outside_project(secrets_path: Path, project_root: Path,
                                   repository: str) -> None:
    """Refuse to run if .secrets is (or could end up) inside the project repo."""
    secrets_path = secrets_path.resolve()
    project_root = project_root.resolve()
    if secrets_path.is_relative_to(project_root):
        raise AgentError(
            f"refusing to run: {secrets_path} is inside the project checkout "
            f"{project_root}. Keep .secrets and the resolver outside the repo.")

    top = git_toplevel(secrets_path.parent)
    if top is None:
        return
    if repository.lower() in repo_remotes(top):
        raise AgentError(
            f"refusing to run: {secrets_path} is inside {top}, a checkout of "
            f"{repository}. Run the resolver from a folder outside the project.")
    if git(["check-ignore", "-q", str(secrets_path)], cwd=top, check=False).returncode:
        log(f"warning: {secrets_path} is inside git repo {top} and not git-ignored")


# --------------------------------------------------------------------------
# GitHub REST API
# --------------------------------------------------------------------------

class GitHub:
    def __init__(self, token: str, repository: str):
        self.repo = repository
        self.owner = repository.split("/")[0]
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "issue-resolver",
        })

    def _request(self, method: str, path: str, ok_404: bool = False, **kw) -> Any:
        r = self.session.request(method, f"{GITHUB_API}{path}", timeout=30, **kw)
        if ok_404 and r.status_code == 404:
            return None
        if r.status_code >= 400:
            raise AgentError(f"GitHub {method} {path} -> {r.status_code}: {r.text[:500]}")
        return r.json() if r.content else None

    def branch_exists(self, branch: str) -> bool:
        return self._request("GET", f"/repos/{self.repo}/branches/{branch}",
                             ok_404=True) is not None

    def has_open_pr(self, branch: str) -> bool:
        return bool(self._request("GET", f"/repos/{self.repo}/pulls", params={
            "head": f"{self.owner}:{branch}", "state": "open"}))

    def oldest_open_issue(self, skip: Callable[[dict], bool]) -> dict | None:
        page = 1
        while True:
            items = self._request("GET", f"/repos/{self.repo}/issues", params={
                "state": "open", "sort": "created", "direction": "asc",
                "per_page": 100, "page": page})
            for item in items:
                if "pull_request" not in item and not skip(item):
                    return item
            if len(items) < 100:
                return None
            page += 1

    def create_pr(self, title: str, head: str, base: str, body: str) -> dict:
        return self._request("POST", f"/repos/{self.repo}/pulls", json={
            "title": title, "head": head, "base": base, "body": body})

    def comment(self, number: int, body: str) -> dict:
        return self._request("POST", f"/repos/{self.repo}/issues/{number}/comments",
                             json={"body": body})


# --------------------------------------------------------------------------
# Sandboxed tools for the model
# --------------------------------------------------------------------------

def find_pytest() -> list[str]:
    """pytest from the virtual environment the resolver runs in."""
    exe = shutil.which("pytest", path=str(Path(sys.executable).parent))
    return [exe] if exe else [sys.executable, "-m", "pytest"]


class Workspace:
    def __init__(self, root: Path, secrets: list[str], allow_pytest: bool):
        self.root = root.resolve()
        self.secrets = [s for s in secrets if len(s) >= 6]
        self.allow_pytest = allow_pytest

    # -- path handling ------------------------------------------------------
    def _inside(self, p: Path) -> bool:
        if not p.is_relative_to(self.root):
            return False
        parts = p.relative_to(self.root).parts
        return not (set(parts) & PROTECTED_DIRS or p.name in PROTECTED_NAMES)

    def _path(self, rel: Any) -> Path:
        if not isinstance(rel, str) or not rel.strip():
            raise ToolError("path must be a non-empty string")
        p = (self.root / rel).resolve()          # also resolves symlinks
        if not self._inside(p):
            raise ToolError(f"access denied: {rel}")
        return p

    def _rel(self, p: Path) -> str:
        return p.relative_to(self.root).as_posix() or "."

    def _walk(self, base: Path):
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames
                                 if d not in PROTECTED_DIRS | {"__pycache__"})
            for name in sorted(filenames):
                f = Path(dirpath) / name
                if self._inside(f.resolve()):
                    yield f

    # -- tools --------------------------------------------------------------
    def list_files(self, path: str = ".") -> str:
        base = self._path(path)
        if not base.is_dir():
            raise ToolError(f"not a directory: {path}")
        out = []
        for f in self._walk(base):
            out.append(self._rel(f))
            if len(out) >= MAX_LIST_ENTRIES:
                return "\n".join(out) + "\n... (truncated)"
        return "\n".join(out) or "(empty)"

    def read_file(self, path: str) -> str:
        p = self._path(path)
        if not p.is_file():
            raise ToolError(f"not a file: {path}")
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ToolError(f"{path} is not a UTF-8 text file")
        return text if len(text) <= MAX_READ_CHARS else text[:MAX_READ_CHARS] + "\n... (truncated)"

    def write_file(self, path: str, content: str) -> str:
        p = self._path(path)
        if not isinstance(content, str):
            raise ToolError("content must be a string")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"wrote {self._rel(p)} ({len(content)} chars)"

    def replace_in_file(self, path: str, old: str, new: str) -> str:
        p = self._path(path)
        text = self.read_file(path)
        count = text.count(old)
        if count != 1:
            raise ToolError(f"'old' must occur exactly once in {path}; found {count}")
        p.write_text(text.replace(old, new, 1), encoding="utf-8")
        return f"edited {self._rel(p)}"

    def delete_file(self, path: str) -> str:
        p = self._path(path)
        if not p.is_file():
            raise ToolError(f"not a file: {path}")
        p.unlink()
        return f"deleted {self._rel(p)}"

    def search(self, query: str, path: str = ".", regex: bool = False) -> str:
        base = self._path(path)
        try:
            pattern = re.compile(query if regex else re.escape(query))
        except re.error as e:
            raise ToolError(f"bad regex: {e}")
        files = [base] if base.is_file() else self._walk(base)
        hits = []
        for f in files:
            try:
                lines = f.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            for i, line in enumerate(lines, 1):
                if pattern.search(line):
                    hits.append(f"{self._rel(f)}:{i}: {line[:300]}")
                    if len(hits) >= MAX_SEARCH_HITS:
                        return "\n".join(hits) + "\n... (truncated)"
        return "\n".join(hits) or "no matches"

    def run_pytest(self, args: str = "") -> str:
        if not self.allow_pytest:
            raise ToolError("running pytest is disabled for this run; check tests by reading them")
        try:
            argv = shlex.split(args or "")
        except ValueError as e:
            raise ToolError(f"bad arguments: {e}")
        simple = {"-q", "-v", "-vv", "-x", "-s", "-l", "-rA", "-ra"}
        prefixes = ("--tb=", "--maxfail=")
        i = 0
        while i < len(argv):
            a = argv[i]
            if a == "-k":
                i += 2                                 # next item is an expression
                continue
            if a.startswith("-"):
                if a not in simple and not a.startswith(prefixes):
                    raise ToolError(f"pytest option not allowed: {a}")
            else:
                self._path(a.split("::")[0])           # test path must be in the repo
            i += 1

        env = clean_env()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        cmd = [*find_pytest(), "-p", "no:cacheprovider", *argv]
        try:
            r = subprocess.run(cmd, cwd=self.root, env=env, text=True,
                               capture_output=True, timeout=PYTEST_TIMEOUT)
        except subprocess.TimeoutExpired:
            return f"pytest timed out after {PYTEST_TIMEOUT}s"
        output = (r.stdout + r.stderr)[-MAX_TOOL_OUTPUT:]
        return f"pytest exit code {r.returncode}\n{output}"

    def dispatch(self, name: str, args: dict) -> str:
        handlers = {
            "list_files": self.list_files, "read_file": self.read_file,
            "write_file": self.write_file, "replace_in_file": self.replace_in_file,
            "delete_file": self.delete_file, "search": self.search,
            "run_pytest": self.run_pytest,
        }
        if name not in handlers:
            return f"error: unknown tool {name}"
        try:
            result = handlers[name](**args)
        except ToolError as e:
            result = f"error: {e}"
        except TypeError as e:
            result = f"error: bad arguments: {e}"
        for s in self.secrets:                       # never echo a secret back
            result = result.replace(s, "[REDACTED]")
        return result[:MAX_TOOL_OUTPUT]


def _fn(name: str, description: str, props: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": props, "required": required}}}


_S = {"type": "string"}
TOOLS = [
    _fn("list_files", "List files recursively under a directory (like ls/find).",
        {"path": {**_S, "description": "Directory relative to the repo root; default '.'"}}, []),
    _fn("read_file", "Read a text file (like cat).", {"path": _S}, ["path"]),
    _fn("search", "Search file contents (like grep). Plain text unless regex=true.",
        {"query": _S, "path": _S, "regex": {"type": "boolean"}}, ["query"]),
    _fn("write_file", "Create or overwrite a file with its full content.",
        {"path": _S, "content": _S}, ["path", "content"]),
    _fn("replace_in_file", "Replace one exact snippet that occurs once in a file.",
        {"path": _S, "old": _S, "new": _S}, ["path", "old", "new"]),
    _fn("delete_file", "Delete a file.", {"path": _S}, ["path"]),
    _fn("run_pytest", "Run `pytest` from the repo root. Optional args such as "
        "'-q', '-x', 'tests/test_x.py', '-k name'.", {"args": _S}, []),
    _fn("finish", "Call once at the end with your final plain-text summary.",
        {"summary": _S}, ["summary"]),
]

TOOL_GUIDE = """\
You work through function calls only; you have no shell.
- ls / find  -> list_files      - cat  -> read_file      - grep -> search
- create or edit files -> write_file / replace_in_file / delete_file
- pytest -> run_pytest
git, pip, network access and running the application are not available, as
the instructions describe. When you are done, call `finish` with the final
summary the instructions ask for. Follow the instructions in the user message.
"""


def build_prompt(instructions: str, issue: dict, tree: str) -> str:
    return (
        f"<instructions>\n{instructions}\n</instructions>\n\n"
        f"Issue #{issue['number']}: {issue['title']}\n\n"
        f"{issue.get('body') or '(no description)'}\n\n"
        f"--- Files currently in the repository ---\n{tree[:20_000]}\n"
    )


def run_agent(client: OpenAI, settings: Settings, ws: Workspace,
              instructions: str, issue: dict) -> str:
    """Run the tool loop and return the model's final summary."""
    messages: list[dict] = [
        {"role": "system", "content": TOOL_GUIDE},
        {"role": "user", "content": build_prompt(instructions, issue, ws.list_files("."))},
    ]
    tools_used = 0

    for step in range(1, settings.max_steps + 1):
        msg = client.chat.completions.create(
            model=settings.model, messages=messages, tools=TOOLS).choices[0].message

        entry: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            entry["tool_calls"] = [{
                "id": tc.id, "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            } for tc in msg.tool_calls]
        messages.append(entry)

        if not msg.tool_calls:
            if msg.content and msg.content.strip() and tools_used:
                return msg.content.strip()          # plain-text final summary
            messages.append({"role": "user", "content":
                             "Use the tools to resolve the issue, then call finish."})
            continue

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
                if not isinstance(args, dict):
                    raise ValueError
            except ValueError:
                result = "error: arguments must be a JSON object"
            else:
                if name == "finish":
                    return str(args.get("summary", "")).strip()
                shown = ", ".join(f"{k}={str(v)[:60]!r}" for k, v in args.items()
                                  if k not in ("content", "old", "new"))
                log(f"step {step}: {name}({shown})")
                result = ws.dispatch(name, args)
                tools_used += 1
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    raise AgentError(
        f"the model did not finish within {settings.max_steps} steps. Its partial "
        f"changes are left in {ws.root}; review them or discard them with "
        "`git reset --hard && git clean -fd` there before the next run.")


# --------------------------------------------------------------------------
# Main flow
# --------------------------------------------------------------------------

def open_repository(path_arg: str) -> Path:
    """Return the top-level folder of the git clone at path_arg."""
    path = Path(path_arg).expanduser().resolve()
    if not path.is_dir():
        raise AgentError(f"{path} is not a folder")
    top = git_toplevel(path)
    if top is None:
        raise AgentError(f"{path} is not inside a git repository")
    if top != path:
        log(f"using repository root {top}")
    return top


def origin_repository(root: Path, expected: str) -> str:
    """owner/name of the clone's origin remote, checked against `expected`."""
    r = git(["config", "--get", "remote.origin.url"], root, check=False)
    name = remote_to_full_name(r.stdout) if r.returncode == 0 else None
    if not name:
        raise AgentError(f"{root} has no GitHub remote named origin")
    if expected and expected.lower() != name:
        raise AgentError(f"origin of {root} is {name}, but GITHUB_REPOSITORY is {expected}")
    return name


def prepare_branch(root: Path, settings: Settings, branch: str) -> str:
    """Create the fix branch from origin/<base>; return the branch to go back to."""
    if git(["status", "--porcelain"], root).stdout.strip():
        raise AgentError(f"{root} has uncommitted or untracked changes; "
                         "commit, stash or remove them first")
    original = git(["rev-parse", "--abbrev-ref", "HEAD"], root).stdout.strip()
    log(f"fetching origin/{settings.base_branch}")
    git(["fetch", "origin", settings.base_branch], root, token=settings.token)
    git(["checkout", "-q", "-B", branch, f"origin/{settings.base_branch}"], root)
    return original


def unstage_junk(root: Path) -> None:
    added = git(["diff", "--cached", "--name-only", "--diff-filter=A"], root).stdout.split("\n")
    junk = [p for p in added if p and any(re.search(rx, p) for rx in JUNK_PATTERNS)]
    for path in junk:
        git(["rm", "--cached", "-q", "--", path], root)
    if junk:
        log(f"not committing {len(junk)} generated file(s): {', '.join(junk[:5])}")


def drop_branch(root: Path, settings: Settings, branch: str, original: str) -> None:
    """Return to the branch the clone was on and delete the fix branch."""
    if original and original not in ("HEAD", branch):
        git(["checkout", "-q", original], root)
    else:
        git(["checkout", "-q", "--detach", f"origin/{settings.base_branch}"], root)
    git(["branch", "-q", "-D", branch], root)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Resolve the oldest open GitHub issue.")
    ap.add_argument("--repo", dest="repo_path", metavar="FOLDER",
                    help="folder of the local clone of the project (required)")
    ap.add_argument("repo_folder", nargs="?", help=argparse.SUPPRESS)
    ap.add_argument("--github-repo", metavar="OWNER/NAME",
                    help="optional; must match the clone's origin (else GITHUB_REPOSITORY)")
    ap.add_argument("--model", help="model name (else OPENAI_MODEL)")
    ap.add_argument("--base-url", help="OpenAI-compatible server URL (else OPENAI_BASE_URL)")
    ap.add_argument("--base-branch", help="PR target branch (else BASE_BRANCH, default dev)")
    ap.add_argument("--max-steps", type=int, help="tool-call rounds (else MAX_STEPS, default 60)")
    ap.add_argument("--no-pytest", action="store_true",
                    help="do not let the model run pytest")
    ap.add_argument("--dry-run", action="store_true",
                    help="commit locally only; no push, PR or comment")
    args = ap.parse_args()
    args.repo_path = args.repo_path or args.repo_folder
    if not args.repo_path:
        ap.error("pass the project folder with --repo FOLDER")
    return args


def main() -> int:
    args = parse_args()
    home = Path.cwd().resolve()
    secrets_path = home / SECRETS_FILE

    secrets = load_secrets(secrets_path)
    settings = load_settings(args, secrets)
    instructions = load_instructions(home / INSTRUCTIONS_FILE)

    project_root = open_repository(args.repo_path)
    settings.repository = origin_repository(project_root, settings.repository)
    assert_secrets_outside_project(secrets_path, project_root, settings.repository)
    log(f"repository {settings.repository} at {project_root}")

    gh = GitHub(settings.token, settings.repository)
    if not gh.branch_exists(settings.base_branch):
        raise AgentError(f"branch '{settings.base_branch}' does not exist in {settings.repository}")

    issue = gh.oldest_open_issue(lambda i: gh.has_open_pr(branch_name(i["number"])))
    if issue is None:
        log("no open issues without a fix pull request")
        return 0
    number, title = issue["number"], issue["title"]
    branch = branch_name(number)
    log(f"resolving #{number}: {title}")

    original = prepare_branch(project_root, settings, branch)

    log(f"model {settings.model} at {settings.base_url or 'https://api.openai.com/v1 (OpenAI)'}")
    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url or None)
    ws = Workspace(project_root, list(secrets.values()), settings.allow_pytest)
    summary = run_agent(client, settings, ws, instructions, issue)
    print("\n===== Final summary =====\n" + (summary or "(no summary)") + "\n", flush=True)

    git(["add", "-A"], project_root)
    unstage_junk(project_root)
    if not git(["diff", "--cached", "--name-only"], project_root).stdout.strip():
        git(["reset", "-q", "--hard"], project_root)
        git(["clean", "-qfd"], project_root)
        drop_branch(project_root, settings, branch, original)
        log("no changes; deleted the fix branch")
        return 0
    staged = git(["diff", "--cached"], project_root).stdout
    if any(len(s) >= 6 and s in staged for s in secrets.values()):
        raise AgentError("the staged changes contain a value from .secrets; aborting")

    git(["commit", "-q", "-m", f"Fix #{number}: {title}"], project_root)
    git(["clean", "-qfd"], project_root)              # drop unstaged generated files
    log(f"committed on {branch}")

    if settings.dry_run:
        log(f"dry run: the commit is in {project_root}")
        return 0

    git(["push", "--force", "--set-upstream", "origin", branch], project_root,
        token=settings.token)
    log(f"pushed {branch}")

    body = (f"Closes #{number}\n\n{summary}\n\n---\n"
            "_Opened automatically by issue_resolver.py._")
    pr = gh.create_pr(f"Fix #{number}: {title}", branch, settings.base_branch, body)
    log(f"opened {pr['html_url']}")

    gh.comment(number, f"A pull request for this issue is ready for review: {pr['html_url']}")
    log(f"commented on #{number}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AgentError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)