"""
Prompt Registry with versioning and diff support.

Manages the lifecycle of prompt templates: registration, versioning,
retrieval, diffing, rollback, and rendering. Prompts are versioned
immutably — each update creates a new version.

Uses an in-memory store with JSON serialization for persistence.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any

from agentops.prompts.state import (
    DEFAULT_PROMPTS,
    PromptCategory,
    PromptDiff,
    PromptTemplate,
    PromptVersion,
)


class PromptRegistry:
    """Versioned registry for prompt templates.

    Each prompt has:
    - A template definition (name, description, category, variables)
    - An ordered list of immutable versions
    - Version history with changelogs

    Usage:
        reg = PromptRegistry()
        reg.register("Summarize documents: {{docs}}", name="summarizer")
        reg.update("summarizer", "Summarize briefly: {{docs}}", changelog="Added brevity")
        v = reg.get("summarizer")  # latest version
        rendered = reg.render("summarizer", docs="doc content")
    """

    def __init__(self, storage_path: str | None = None):
        self._templates: dict[str, PromptTemplate] = {}
        self._versions: dict[str, list[PromptVersion]] = {}
        self._storage_path = Path(storage_path) if storage_path else None

        # Register built-in defaults
        for template in DEFAULT_PROMPTS.values():
            self.register_template(template, author="agentops", changelog="Initial default")

    # ── Registration ──────────────────────────────────────────────

    def register(
        self,
        content: str,
        name: str,
        description: str = "",
        category: str | PromptCategory = PromptCategory.CUSTOM,
        variables: list[str] | None = None,
        author: str = "agentops",
        changelog: str = "Initial version",
        metadata: dict[str, Any] | None = None,
    ) -> PromptVersion:
        """Register a new prompt and create version 1.

        Args:
            content: The prompt template content with {{variables}}
            name: Unique prompt name
            description: Human-readable description
            category: Prompt category for organization
            variables: Explicit variable list (auto-extracted if None)
            author: Who created this version
            changelog: Description of changes
            metadata: Additional key-value metadata

        Returns:
            The newly created PromptVersion (v1)

        Raises:
            ValueError: If a prompt with this name already exists
        """
        if name in self._templates:
            raise ValueError(f"Prompt '{name}' already exists. Use update() to create a new version.")

        if isinstance(category, str):
            category = PromptCategory(category)

        template = PromptTemplate(
            name=name,
            content=content,
            description=description,
            category=category,
            variables=variables or [],
            metadata=metadata or {},
        )

        return self.register_template(template, author=author, changelog=changelog)

    def register_template(
        self,
        template: PromptTemplate,
        author: str = "agentops",
        changelog: str = "Initial version",
    ) -> PromptVersion:
        """Register a pre-built template. Creates version 1.

        If a template with this name already exists, acts like update().
        """
        if template.name in self._templates:
            return self.update(
                template.name,
                template.content,
                author=author,
                changelog=changelog,
                metadata=template.metadata,
            )

        self._templates[template.name] = template
        version = PromptVersion(
            prompt_name=template.name,
            version=1,
            content=template.content,
            author=author,
            changelog=changelog,
            metadata=template.metadata,
        )
        self._versions[template.name] = [version]
        return version

    # ── Versioning ─────────────────────────────────────────────────

    def update(
        self,
        name: str,
        new_content: str,
        author: str = "agentops",
        changelog: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> PromptVersion:
        """Create a new version of an existing prompt.

        Args:
            name: Prompt name to update
            new_content: New prompt content
            author: Who made this change
            changelog: Description of what changed
            metadata: Updated metadata

        Returns:
            The newly created PromptVersion

        Raises:
            KeyError: If prompt name doesn't exist
        """
        if name not in self._templates:
            raise KeyError(f"Prompt '{name}' not found. Use register() first.")

        template = self._templates[name]
        version_number = len(self._versions[name]) + 1

        # Update template metadata
        template.content = new_content
        template.variables = template._extract_variables()
        if metadata:
            template.metadata.update(metadata)

        version = PromptVersion(
            prompt_name=name,
            version=version_number,
            content=new_content,
            author=author,
            changelog=changelog,
            metadata=template.metadata,
        )
        self._versions[name].append(version)
        return version

    # ── Retrieval ──────────────────────────────────────────────────

    def get(self, name: str, version: int | None = None) -> PromptVersion:
        """Get a specific version of a prompt. Defaults to latest.

        Raises KeyError if prompt or version doesn't exist.
        """
        if name not in self._versions:
            raise KeyError(f"Prompt '{name}' not found")
        versions = self._versions[name]
        if version is None:
            return versions[-1]
        if version < 1 or version > len(versions):
            raise KeyError(
                f"Version {version} not found for prompt '{name}'. "
                f"Available: 1-{len(versions)}"
            )
        return versions[version - 1]

    def get_template(self, name: str) -> PromptTemplate:
        """Get the prompt template definition."""
        if name not in self._templates:
            raise KeyError(f"Prompt '{name}' not found")
        return self._templates[name]

    def list_prompts(self) -> list[dict[str, Any]]:
        """List all registered prompts with their current state."""
        result = []
        for name, template in sorted(self._templates.items()):
            versions = self._versions.get(name, [])
            latest = versions[-1] if versions else None
            result.append({
                "name": name,
                "description": template.description,
                "category": template.category.value,
                "variables": template.variables,
                "current_version": len(versions),
                "latest_hash": latest.content_hash if latest else None,
                "version_count": len(versions),
            })
        return result

    def list_versions(self, name: str) -> list[dict[str, Any]]:
        """List all versions of a specific prompt."""
        if name not in self._versions:
            raise KeyError(f"Prompt '{name}' not found")
        return [v.to_dict() for v in self._versions[name]]

    # ── Diff ───────────────────────────────────────────────────────

    def diff(self, name: str, version_a: int, version_b: int | None = None) -> PromptDiff:
        """Compute the diff between two prompt versions.

        If version_b is None, compares version_a against the previous version.
        """
        self._versions[name]
        if version_b is None:
            version_b = version_a
            version_a = version_b - 1

        v_a = self.get(name, version_a)
        v_b = self.get(name, version_b)

        a_lines = v_a.content.splitlines(keepends=True)
        b_lines = v_b.content.splitlines(keepends=True)

        differ = difflib.unified_diff(
            a_lines, b_lines,
            fromfile=f"v{version_a}", tofile=f"v{version_b}",
            lineterm="",
        )

        added: list[str] = []
        removed: list[str] = []
        unchanged = 0

        for line in differ:
            if line.startswith("@@") or line.startswith("---") or line.startswith("+++"):
                continue
            if line.startswith("+") and not line.startswith("+++"):
                added.append(line[1:])
            elif line.startswith("-") and not line.startswith("---"):
                removed.append(line[1:])
            elif line.startswith(" "):
                unchanged += 1

        return PromptDiff(
            prompt_name=name,
            version_a=version_a,
            version_b=version_b,
            lines_added=added,
            lines_removed=removed,
            lines_unchanged=unchanged,
        )

    # ── Rollback ───────────────────────────────────────────────────

    def rollback(self, name: str, target_version: int, author: str = "agentops") -> PromptVersion:
        """Rollback to a previous version by creating a new version with
        the target version's content.

        Args:
            name: Prompt name
            target_version: Version number to rollback to
            author: Who performed the rollback

        Returns:
            New PromptVersion with the rollback content
        """
        target = self.get(name, target_version)
        return self.update(
            name,
            target.content,
            author=author,
            changelog=f"Rollback to v{target_version}",
        )

    # ── Render ─────────────────────────────────────────────────────

    def render(self, name: str, version: int | None = None, **kwargs) -> str:
        """Render a prompt with variable values.

        Gets the specified version (or latest) and fills in variables.
        """
        prompt_version = self.get(name, version)
        template = self._templates[name]

        # Use the template for variable validation, version content for rendering
        missing = set(template.variables) - set(kwargs.keys())
        if missing:
            raise ValueError(
                f"Missing variables for prompt '{name}': {sorted(missing)}"
            )

        result = prompt_version.content
        for var, value in kwargs.items():
            result = result.replace(f"{{{{{var}}}}}", str(value))
        return result

    # ── Persistence ────────────────────────────────────────────────

    def save(self, path: str | None = None) -> str:
        """Persist the registry to a JSON file."""
        target = Path(path) if path else self._storage_path
        if target is None:
            raise ValueError("No storage path configured. Provide a path or set storage_path.")

        data = {
            "templates": {
                name: {
                    "description": t.description,
                    "category": t.category.value,
                    "variables": t.variables,
                    "metadata": t.metadata,
                    "current_content": t.content,
                }
                for name, t in self._templates.items()
            },
            "versions": {
                name: [v.to_dict() for v in versions]
                for name, versions in self._versions.items()
            },
        }

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2))
        return str(target)

    @classmethod
    def load(cls, path: str) -> PromptRegistry:
        """Load a registry from a JSON file."""
        data = json.loads(Path(path).read_text())
        registry = cls(storage_path=path)

        # Clear defaults since we'll load from file
        registry._templates.clear()
        registry._versions.clear()

        for name, t_data in data.get("templates", {}).items():
            template = PromptTemplate(
                name=name,
                content=t_data.get("current_content", ""),
                description=t_data.get("description", ""),
                category=PromptCategory(t_data.get("category", "custom")),
                variables=t_data.get("variables", []),
                metadata=t_data.get("metadata", {}),
            )
            registry._templates[name] = template

        for name, versions in data.get("versions", {}).items():
            registry._versions[name] = [
                PromptVersion(**v) for v in versions
            ]

        return registry

    # ── Stats ───────────────────────────────────────────────────────

    @property
    def prompt_count(self) -> int:
        return len(self._templates)

    @property
    def total_versions(self) -> int:
        return sum(len(v) for v in self._versions.values())

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        categories: dict[str, int] = {}
        for t in self._templates.values():
            cat = t.category.value
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "total_prompts": self.prompt_count,
            "total_versions": self.total_versions,
            "categories": categories,
            "prompts_with_multiple_versions": sum(
                1 for v in self._versions.values() if len(v) > 1
            ),
        }
