"""Versioned strategy rules learned from prior site interactions."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_PATH = Path("site_memory.json")
RULESET_VERSION = 1


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _normalize_domain(domain: str) -> str:
    return domain.lower().strip().replace("https://", "").replace("http://", "").split("/")[0]


def _domain_candidates(domain: str) -> list[str]:
    parts = [p for p in _normalize_domain(domain).split(".") if p]
    if len(parts) < 2:
        return [_normalize_domain(domain)]
    full = ".".join(parts)
    parent = ".".join(parts[-2:])
    if parent == full:
        return [full]
    return [full, parent]


class SiteMemory:
    """Persistent store of versioned domain strategy rules."""

    def __init__(self, path: Path = DEFAULT_MEMORY_PATH):
        self.path = path
        self._data: dict[str, Any] = {"ruleset_version": RULESET_VERSION, "rules": []}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
        except Exception as e:
            logger.warning(f"Failed to load site memory: {e}")
            return

        if isinstance(raw, dict) and "rules" in raw:
            self._data = raw
            self._data.setdefault("ruleset_version", RULESET_VERSION)
            self._data.setdefault("rules", [])
            return

        # Legacy format migration: {domain: {needs_stealth: true, ...}}
        if isinstance(raw, dict):
            logger.info("Migrating legacy site_memory format to versioned rule packs")
            migrated = {"ruleset_version": RULESET_VERSION, "rules": []}
            for domain, workaround in raw.items():
                if isinstance(workaround, dict):
                    rule_id = self._legacy_to_rule(domain, workaround)
                    if rule_id:
                        migrated["rules"].append(self._find_rule(rule_id))
            migrated["rules"] = [r for r in migrated["rules"] if r]
            self._data = migrated
            self._save()

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._data, indent=2))

    def _find_rule(self, rule_id: str) -> dict[str, Any] | None:
        for rule in self._data.get("rules", []):
            if rule.get("rule_id") == rule_id:
                return rule
        return None

    def _legacy_to_rule(self, domain: str, workaround: dict[str, Any]) -> str | None:
        return self.save_workaround(domain, workaround)

    def get(self, domain: str) -> dict[str, Any] | None:
        """Compatibility helper: returns highest-confidence matching rule for domain."""
        rules = self.get_matching_rules(domain)
        return rules[0] if rules else None

    def get_all(self) -> dict[str, Any]:
        """Return full ruleset."""
        return {
            "ruleset_version": self._data.get("ruleset_version", RULESET_VERSION),
            "rules": list(self._data.get("rules", [])),
        }

    def save_rule(self, rule_pack: dict[str, Any]) -> str:
        """Save a versioned strategy rule pack."""
        pre = dict(rule_pack.get("preconditions", {}))
        domain = _normalize_domain(pre.get("domain", ""))
        if not domain:
            raise ValueError("rule_pack.preconditions.domain is required")
        pre["domain"] = domain
        pre.setdefault("task_types", [])
        pre.setdefault("error_signatures", [])

        actions = list(rule_pack.get("actions", []))
        guardrails = dict(rule_pack.get("guardrails", {}))
        guardrails.setdefault("max_attempts", 3)
        guardrails.setdefault("timeout_sec", 60)
        guardrails.setdefault("abort_conditions", [])

        meta = dict(rule_pack.get("metadata", {}))
        base_id = str(rule_pack.get("rule_id") or f"{domain.replace('.', '_')}_strategy")
        existing_versions = []
        for r in self._data.get("rules", []):
            rid = str(r.get("rule_id", ""))
            if rid.startswith(f"{base_id}_v"):
                try:
                    existing_versions.append(int(rid.split("_v")[-1]))
                except ValueError:
                    continue
        version = int(meta.get("version") or ((max(existing_versions) + 1) if existing_versions else 1))
        rule_id = f"{base_id}_v{version}"

        now = _now_iso()
        meta.setdefault("confidence", 0.6)
        meta.setdefault("sample_count", 1)
        meta.setdefault("notes", "")
        meta["version"] = version
        meta.setdefault("created_at", now)
        meta["updated_at"] = now

        rule = {
            "rule_id": rule_id,
            "preconditions": pre,
            "actions": actions,
            "guardrails": guardrails,
            "metadata": meta,
        }

        # Keep immutable history by appending new versions.
        self._data.setdefault("rules", []).append(rule)
        self._save()
        logger.info(f"Site memory: saved rule pack {rule_id}")
        return rule_id

    def save_workaround(self, domain: str, workaround: dict[str, Any]) -> str | None:
        """Legacy compatibility: convert workaround dict into a versioned rule pack."""
        normalized = _normalize_domain(domain)
        actions: list[dict[str, Any]] = []
        if workaround.get("needs_headers"):
            actions.append({
                "tool": "configure_browser",
                "input": {
                    "action": "set_headers",
                    "value": json.dumps(workaround["needs_headers"]),
                },
            })
        if workaround.get("needs_stealth"):
            actions.append({
                "tool": "configure_browser",
                "input": {"action": "enable_stealth"},
            })
        if workaround.get("user_agent"):
            actions.append({
                "tool": "configure_browser",
                "input": {"action": "set_user_agent", "value": str(workaround["user_agent"])},
            })
        if workaround.get("clear_cookies"):
            actions.append({
                "tool": "configure_browser",
                "input": {"action": "clear_cookies"},
            })
        if not actions and workaround.get("workaround"):
            # Keep as metadata-only rule when we don't have executable actions.
            actions = []

        error_signatures = []
        if workaround.get("blocks_headless"):
            error_signatures.extend(["403", "automated tool"])
        if workaround.get("rate_limited"):
            error_signatures.extend(["429", "too many requests"])

        rule_pack = {
            "rule_id": f"{normalized.replace('.', '_')}_legacy",
            "preconditions": {
                "domain": normalized,
                "task_types": [],
                "error_signatures": error_signatures,
            },
            "actions": actions,
            "guardrails": {
                "max_attempts": int(workaround.get("max_attempts", 3)),
                "timeout_sec": int(workaround.get("timeout_sec", 60)),
                "abort_conditions": list(workaround.get("abort_conditions", [])),
            },
            "metadata": {
                "confidence": float(workaround.get("confidence", 0.6)),
                "sample_count": int(workaround.get("sample_count", 1)),
                "notes": str(workaround.get("workaround", "")),
            },
        }
        return self.save_rule(rule_pack)

    def get_matching_rules(
        self,
        domain: str,
        task_type: str | None = None,
        error_signature: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find matching rules ordered by confidence/version."""
        candidates = set(_domain_candidates(domain))
        matched: list[dict[str, Any]] = []

        for rule in self._data.get("rules", []):
            pre = rule.get("preconditions", {})
            rule_domain = _normalize_domain(str(pre.get("domain", "")))
            if rule_domain and rule_domain not in candidates:
                continue

            task_types = [str(x) for x in pre.get("task_types", [])]
            if task_type and task_types and task_type not in task_types:
                continue

            signatures = [str(x).lower() for x in pre.get("error_signatures", [])]
            if error_signature and signatures:
                lowered = error_signature.lower()
                if not any(sig in lowered for sig in signatures):
                    continue

            matched.append(rule)

        matched.sort(
            key=lambda r: (
                float(r.get("metadata", {}).get("confidence", 0.0)),
                int(r.get("metadata", {}).get("version", 0)),
            ),
            reverse=True,
        )
        return matched

    def get_auto_actions_for_url(
        self,
        url: str,
        task_type: str | None = None,
        error_signature: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        """Return configured actions for the best-matching rule for this URL."""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
        except Exception:
            return [], None

        matches = self.get_matching_rules(domain, task_type=task_type, error_signature=error_signature)
        if not matches:
            return [], None
        top = matches[0]
        return list(top.get("actions", [])), top

    def get_hint_for_url(self, url: str) -> str | None:
        """Return a short hint summarizing best known rule for URL domain."""
        actions, rule = self.get_auto_actions_for_url(url)
        if not rule:
            return None
        pre = rule.get("preconditions", {})
        meta = rule.get("metadata", {})
        parts = [f"Rule {rule.get('rule_id')} matches domain {pre.get('domain')}:"]
        if pre.get("error_signatures"):
            parts.append(f"- error signatures: {pre['error_signatures']}")
        if actions:
            action_names = [a.get("input", {}).get("action", "?") for a in actions]
            parts.append(f"- actions: {action_names}")
        parts.append(
            f"- confidence={meta.get('confidence', 0.0)} sample_count={meta.get('sample_count', 0)}"
        )
        if meta.get("notes"):
            parts.append(f"- notes: {meta['notes']}")
        return "\n".join(parts)

    def record_rule_outcome(self, rule_id: str, success: bool) -> None:
        """Update confidence/sample count for an existing rule."""
        rule = self._find_rule(rule_id)
        if rule is None:
            return

        meta = rule.setdefault("metadata", {})
        current_conf = float(meta.get("confidence", 0.6))
        sample_count = int(meta.get("sample_count", 0)) + 1
        target = 1.0 if success else 0.0
        updated_conf = ((current_conf * (sample_count - 1)) + target) / sample_count
        meta["confidence"] = round(updated_conf, 4)
        meta["sample_count"] = sample_count
        meta["updated_at"] = _now_iso()
        self._save()
