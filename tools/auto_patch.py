#!/usr/bin/env python3
"""Apply llama.cpp plugin host patches from plugin manifests.

This tool is intentionally conservative:
- patch files are normal unified diffs
- Git performs all patch application and reverse application
- no source file is rewritten by Python
- destructive Git commands are never used
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SCHEMA = "llama-plugin-patch/v1"


class PatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class PatchFile:
    id: str
    path: Path


@dataclass(frozen=True)
class PluginManifest:
    id: str
    name: str
    manifest_path: Path
    plugin_dir: Path
    patches: tuple[PatchFile, ...]
    dll: str | None
    flags: tuple[str, ...]


@dataclass(frozen=True)
class GitResult:
    ok: bool
    stdout: str
    stderr: str
    returncode: int


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_plugins_dir() -> Path:
    return repo_root() / "plugins"


def run_git(target: Path, args: list[str], check: bool = False) -> GitResult:
    proc = subprocess.run(
        ["git", *args],
        cwd=target,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    result = GitResult(proc.returncode == 0, proc.stdout, proc.stderr, proc.returncode)
    if check and not result.ok:
        raise PatchError(format_git_failure(args, result))
    return result


def format_git_failure(args: list[str], result: GitResult) -> str:
    parts = [f"git {' '.join(args)} failed with exit code {result.returncode}"]
    if result.stdout.strip():
        parts.append("stdout:\n" + result.stdout.rstrip())
    if result.stderr.strip():
        parts.append("stderr:\n" + result.stderr.rstrip())
    return "\n".join(parts)


def ensure_git_worktree(target: Path) -> None:
    if not target.exists():
        raise PatchError(f"target does not exist: {target}")
    if not target.is_dir():
        raise PatchError(f"target is not a directory: {target}")
    result = run_git(target, ["rev-parse", "--is-inside-work-tree"])
    if not result.ok or result.stdout.strip() != "true":
        raise PatchError(f"target is not a Git worktree: {target}")


def load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise PatchError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise PatchError(f"manifest must be a JSON object: {path}")
    return data


def normalize_patch_entries(plugin_id: str, plugin_dir: Path, data: dict) -> tuple[PatchFile, ...]:
    entries: list[PatchFile] = []

    if "patch" in data:
        value = data["patch"]
        if not isinstance(value, str):
            raise PatchError(f"{plugin_id}: 'patch' must be a string")
        entries.append(PatchFile(plugin_id, (plugin_dir / value).resolve()))

    raw_patches = data.get("patches", [])
    if raw_patches is None:
        raw_patches = []
    if not isinstance(raw_patches, list):
        raise PatchError(f"{plugin_id}: 'patches' must be a list")

    for i, item in enumerate(raw_patches, start=1):
        if isinstance(item, str):
            patch_id = f"{plugin_id}:{i}"
            patch_file = item
        elif isinstance(item, dict):
            patch_id_raw = item.get("id")
            patch_file_raw = item.get("file")
            if not isinstance(patch_id_raw, str) or not patch_id_raw:
                raise PatchError(f"{plugin_id}: patch entry {i} missing string id")
            if not isinstance(patch_file_raw, str) or not patch_file_raw:
                raise PatchError(f"{plugin_id}: patch entry {i} missing string file")
            patch_id = patch_id_raw
            patch_file = patch_file_raw
        else:
            raise PatchError(f"{plugin_id}: patch entry {i} must be string or object")

        entries.append(PatchFile(patch_id, (plugin_dir / patch_file).resolve()))

    if not entries:
        raise PatchError(f"{plugin_id}: manifest has no patch or patches entries")

    return tuple(entries)


def load_manifest(path: Path) -> PluginManifest:
    data = load_json(path)
    schema = data.get("schema")
    if schema != SCHEMA:
        raise PatchError(f"{path}: unsupported schema {schema!r}; expected {SCHEMA!r}")

    plugin_data = data.get("plugin")
    if isinstance(plugin_data, dict):
        plugin_id = plugin_data.get("id")
        name = plugin_data.get("name") or plugin_id
    else:
        plugin_id = data.get("id")
        name = data.get("name") or plugin_id

    if not isinstance(plugin_id, str) or not plugin_id:
        raise PatchError(f"{path}: missing plugin id")
    if not isinstance(name, str) or not name:
        name = plugin_id

    plugin_dir = path.parent.resolve()
    patches = normalize_patch_entries(plugin_id, plugin_dir, data)

    runtime = data.get("runtime")
    if isinstance(runtime, dict):
        dll = runtime.get("dll")
        flags_raw = runtime.get("flags", [])
    else:
        dll = data.get("dll")
        flags_raw = data.get("flags", [])

    if dll is not None and not isinstance(dll, str):
        raise PatchError(f"{plugin_id}: dll must be a string")
    if flags_raw is None:
        flags_raw = []
    if not isinstance(flags_raw, list) or any(not isinstance(flag, str) for flag in flags_raw):
        raise PatchError(f"{plugin_id}: flags must be a list of strings")

    return PluginManifest(
        id=plugin_id,
        name=name,
        manifest_path=path.resolve(),
        plugin_dir=plugin_dir,
        patches=patches,
        dll=dll,
        flags=tuple(flags_raw),
    )


def discover_manifests(plugins_dir: Path) -> dict[str, PluginManifest]:
    manifests: dict[str, PluginManifest] = {}
    if not plugins_dir.exists():
        return manifests

    for path in sorted(plugins_dir.glob("*/patch.json")):
        manifest = load_manifest(path)
        if manifest.id in manifests:
            prev = manifests[manifest.id].manifest_path
            raise PatchError(f"duplicate plugin id {manifest.id!r}: {prev} and {path}")
        manifests[manifest.id] = manifest
    return manifests


def select_plugins(manifests: dict[str, PluginManifest], ids: list[str]) -> list[PluginManifest]:
    if not ids:
        raise PatchError("select at least one plugin id")
    selected: list[PluginManifest] = []
    for plugin_id in ids:
        manifest = manifests.get(plugin_id)
        if manifest is None:
            available = ", ".join(sorted(manifests)) or "none"
            raise PatchError(f"unknown plugin {plugin_id!r}; available: {available}")
        selected.append(manifest)
    return selected


def iter_patch_files(plugins: Iterable[PluginManifest]) -> list[tuple[PluginManifest, PatchFile]]:
    seen: set[Path] = set()
    ordered: list[tuple[PluginManifest, PatchFile]] = []
    for plugin in plugins:
        for patch in plugin.patches:
            key = patch.path.resolve()
            if key in seen:
                continue
            seen.add(key)
            ordered.append((plugin, patch))
    return ordered


def git_apply_check(target: Path, patch: Path, reverse: bool = False) -> GitResult:
    args = ["apply"]
    if reverse:
        args.append("--reverse")
    args.extend(["--check", str(patch)])
    return run_git(target, args)


def patch_status(target: Path, patch: Path) -> str:
    if not patch.exists():
        return "missing"
    if git_apply_check(target, patch, reverse=True).ok:
        return "applied"
    if git_apply_check(target, patch).ok:
        return "pending"
    return "conflict"


def apply_patch_file(target: Path, patch: Path, use_3way: bool) -> None:
    result = run_git(target, ["apply", str(patch)])
    if result.ok:
        return
    if not use_3way:
        raise PatchError(format_git_failure(["apply", str(patch)], result))

    check_3way = run_git(target, ["apply", "--3way", "--check", str(patch)])
    if not check_3way.ok:
        msg = format_git_failure(["apply", str(patch)], result)
        msg += "\n\n3-way preflight also failed:\n"
        msg += format_git_failure(["apply", "--3way", "--check", str(patch)], check_3way)
        raise PatchError(msg)

    merge_result = run_git(target, ["apply", "--3way", str(patch)])
    if not merge_result.ok:
        msg = format_git_failure(["apply", str(patch)], result)
        msg += "\n\n3-way fallback also failed:\n"
        msg += format_git_failure(["apply", "--3way", str(patch)], merge_result)
        raise PatchError(msg)


def revert_patch_file(target: Path, patch: Path) -> None:
    result = run_git(target, ["apply", "--reverse", str(patch)])
    if not result.ok:
        raise PatchError(format_git_failure(["apply", "--reverse", str(patch)], result))


def parse_touched_files(patch: Path) -> set[str]:
    files: set[str] = set()
    try:
        lines = patch.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return files

    for line in lines:
        if not line.startswith("diff --git "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        for raw in parts[2:4]:
            if raw.startswith("a/") or raw.startswith("b/"):
                path = raw[2:]
                if path != "/dev/null":
                    files.add(path)
    return files


def dirty_files(target: Path, patches: Iterable[Path]) -> list[str]:
    touched: set[str] = set()
    for patch in patches:
        touched.update(parse_touched_files(patch))
    if not touched:
        return []

    result = run_git(target, ["status", "--porcelain", "--", *sorted(touched)])
    if not result.ok:
        raise PatchError(format_git_failure(["status", "--porcelain", "--", *sorted(touched)], result))
    return [line for line in result.stdout.splitlines() if line.strip()]


def format_env_path(path: Path, base: Path) -> str:
    try:
        rel = path.resolve().relative_to(base.resolve())
        text = ".\\" + str(rel)
    except ValueError:
        text = str(path.resolve())
    if os.name == "nt":
        return text.replace("/", "\\")
    return text.replace("\\", "/")


def plugin_env_entries(plugins: Iterable[PluginManifest], env_file: Path) -> list[str]:
    base = env_file.parent
    entries: list[str] = []
    for plugin in plugins:
        if not plugin.dll:
            continue
        dll_path = (plugin.plugin_dir / plugin.dll).resolve()
        entries.append(format_env_path(dll_path, base))
    return entries


def read_env_lines(env_file: Path) -> list[str]:
    if not env_file.exists():
        return []
    return env_file.read_text(encoding="utf-8", errors="replace").splitlines()


def write_env_lines(env_file: Path, lines: list[str]) -> None:
    env_file.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines).rstrip()
    if text:
        text += "\n"
    env_file.write_text(text, encoding="utf-8")


def update_plugins_env(env_file: Path, entries: list[str], remove: bool = False, dry_run: bool = False) -> None:
    if not entries:
        return

    existing = read_env_lines(env_file)
    normalized_existing = {line.strip().lower() for line in existing if line.strip()}
    normalized_entries = {entry.strip().lower() for entry in entries}

    if remove:
        new_lines = [line for line in existing if line.strip().lower() not in normalized_entries]
    else:
        new_lines = existing[:]
        for entry in entries:
            if entry.strip().lower() not in normalized_existing:
                new_lines.append(entry)
                normalized_existing.add(entry.strip().lower())

    action = "remove from" if remove else "add to"
    print(f"Would {action} {env_file}:") if dry_run else None
    for entry in entries:
        print(f"  {entry}") if dry_run else None
    if not dry_run:
        write_env_lines(env_file, new_lines)


def command_list(args: argparse.Namespace) -> int:
    manifests = discover_manifests(args.plugins_dir)
    if not manifests:
        print(f"No patch manifests found in {args.plugins_dir}")
        return 0

    for manifest in manifests.values():
        patch_count = len(manifest.patches)
        dll = manifest.dll or "no dll entry"
        print(f"{manifest.id}: {manifest.name} ({patch_count} patch file(s), {dll})")
    return 0


def command_plan(args: argparse.Namespace) -> int:
    manifests = discover_manifests(args.plugins_dir)
    plugins = select_plugins(manifests, args.plugins)

    print("Selected plugins:")
    for plugin in plugins:
        print(f"  {plugin.id}: {plugin.name}")
        if plugin.flags:
            print("    flags: " + ", ".join(plugin.flags))

    print("Patch order:")
    for plugin, patch in iter_patch_files(plugins):
        print(f"  {patch.id}: {patch.path} ({plugin.id})")

    entries = plugin_env_entries(plugins, args.env_file)
    if entries:
        print(f"plugins.env entries for {args.env_file}:")
        for entry in entries:
            print(f"  {entry}")
    return 0


def command_check(args: argparse.Namespace) -> int:
    ensure_git_worktree(args.target)
    manifests = discover_manifests(args.plugins_dir)
    plugins = select_plugins(manifests, args.plugins)

    failed = False
    for plugin, patch in iter_patch_files(plugins):
        status = patch_status(args.target, patch.path)
        print(f"{plugin.id} / {patch.id}: {status}")
        if status in {"missing", "conflict"}:
            failed = True

    return 1 if failed else 0


def command_apply(args: argparse.Namespace) -> int:
    ensure_git_worktree(args.target)
    manifests = discover_manifests(args.plugins_dir)
    plugins = select_plugins(manifests, args.plugins)
    ordered = iter_patch_files(plugins)

    pending: list[tuple[PluginManifest, PatchFile]] = []
    for plugin, patch in ordered:
        status = patch_status(args.target, patch.path)
        print(f"{plugin.id} / {patch.id}: {status}")
        if status == "missing":
            raise PatchError(f"missing patch file: {patch.path}")
        if status == "conflict":
            raise PatchError(f"patch cannot be applied cleanly: {patch.path}")
        if status == "pending":
            pending.append((plugin, patch))

    if pending and not args.allow_dirty:
        dirty = dirty_files(args.target, [patch.path for _, patch in pending])
        if dirty:
            joined = "\n".join(f"  {line}" for line in dirty)
            raise PatchError(
                "target has local changes in files touched by pending patches. "
                "Use --allow-dirty only if you have reviewed them.\n" + joined
            )

    if args.dry_run:
        print("Dry run: no patches applied")
    else:
        for plugin, patch in pending:
            print(f"Applying {plugin.id} / {patch.id}")
            apply_patch_file(args.target, patch.path, use_3way=args.three_way)

    if args.write_env:
        entries = plugin_env_entries(plugins, args.env_file)
        update_plugins_env(args.env_file, entries, remove=False, dry_run=args.dry_run)

    if not pending:
        print("All selected patches were already applied")
    return 0


def command_revert(args: argparse.Namespace) -> int:
    ensure_git_worktree(args.target)
    manifests = discover_manifests(args.plugins_dir)
    plugins = select_plugins(manifests, args.plugins)
    ordered = list(reversed(iter_patch_files(plugins)))

    applied: list[tuple[PluginManifest, PatchFile]] = []
    for plugin, patch in ordered:
        status = patch_status(args.target, patch.path)
        print(f"{plugin.id} / {patch.id}: {status}")
        if status == "missing":
            raise PatchError(f"missing patch file: {patch.path}")
        if status == "conflict":
            raise PatchError(
                f"patch is neither cleanly applicable nor cleanly reversible: {patch.path}"
            )
        if status == "applied":
            applied.append((plugin, patch))

    for plugin, patch in applied:
        check = git_apply_check(args.target, patch.path, reverse=True)
        if not check.ok:
            raise PatchError(format_git_failure(["apply", "--reverse", "--check", str(patch.path)], check))

    if args.dry_run:
        print("Dry run: no patches reverted")
    else:
        for plugin, patch in applied:
            print(f"Reverting {plugin.id} / {patch.id}")
            revert_patch_file(args.target, patch.path)

    if args.write_env:
        entries = plugin_env_entries(plugins, args.env_file)
        update_plugins_env(args.env_file, entries, remove=True, dry_run=args.dry_run)

    if not applied:
        print("No selected patches were applied")
    return 0


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--plugins-dir",
        type=Path,
        default=default_plugins_dir(),
        help="directory containing plugin folders with patch.json manifests",
    )


def add_target_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("plugins", nargs="+", help="plugin id(s) to operate on")
    parser.add_argument(
        "--target",
        type=Path,
        default=repo_root() / "llama.cpp",
        help="target llama.cpp Git worktree",
    )


def add_env_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--env-file",
        type=Path,
        default=repo_root() / "plugins.env",
        help="plugins.env file to update",
    )
    parser.add_argument(
        "--write-env",
        action="store_true",
        help="update plugins.env entries for selected plugins",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply llama.cpp plugin host patches safely")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list patchable plugins")
    add_common_args(p_list)
    p_list.set_defaults(func=command_list)

    p_plan = sub.add_parser("plan", help="show selected plugins, patches, and env entries")
    add_common_args(p_plan)
    p_plan.add_argument("plugins", nargs="+", help="plugin id(s) to plan")
    p_plan.add_argument(
        "--env-file",
        type=Path,
        default=repo_root() / "plugins.env",
        help="plugins.env path used for preview entries",
    )
    p_plan.set_defaults(func=command_plan)

    p_check = sub.add_parser("check", help="check whether patches apply or are already applied")
    add_common_args(p_check)
    add_target_args(p_check)
    p_check.set_defaults(func=command_check)

    p_apply = sub.add_parser("apply", help="apply selected plugin patches")
    add_common_args(p_apply)
    add_target_args(p_apply)
    add_env_args(p_apply)
    p_apply.add_argument("--dry-run", action="store_true", help="show actions without applying")
    p_apply.add_argument("--allow-dirty", action="store_true", help="allow touched files to have local changes")
    p_apply.add_argument("--3way", dest="three_way", action="store_true", help="opt in to git apply --3way fallback")
    p_apply.set_defaults(func=command_apply)

    p_revert = sub.add_parser("revert", help="reverse selected plugin patches")
    add_common_args(p_revert)
    add_target_args(p_revert)
    add_env_args(p_revert)
    p_revert.add_argument("--dry-run", action="store_true", help="show actions without reverting")
    p_revert.set_defaults(func=command_revert)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except PatchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
