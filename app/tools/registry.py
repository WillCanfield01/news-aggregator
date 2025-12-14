from __future__ import annotations

from typing import Any, Dict, List, Optional

ToolInput = Dict[str, Any]
ToolEntry = Dict[str, Any]


def get_tools_registry() -> List[ToolEntry]:
    """Return the current list of registered tools."""
    return [
        {
            "slug": "resume-bullets",
            "title": "Resume Bullet Improver",
            "description": "Turn rough experience notes into strong, measurable resume bullets.",
            "category": "Career",
            "is_enabled": True,
            "output_label": "Improved bullets",
            "inputs": [
                {
                    "name": "experience_text",
                    "label": "Experience notes",
                    "type": "textarea",
                    "required": True,
                    "max_chars": 3000,
                    "placeholder": "Paste raw experience notes, tasks, or accomplishments to improve.",
                },
                {
                    "name": "target_role",
                    "label": "Target role",
                    "type": "text",
                    "required": False,
                    "max_chars": 120,
                    "placeholder": "e.g., IT Support Specialist",
                },
                {
                    "name": "job_description",
                    "label": "Job description (optional)",
                    "type": "textarea",
                    "required": False,
                    "max_chars": 3000,
                    "placeholder": "Paste the job post text to align keywords (optional).",
                },
                {
                    "name": "tone",
                    "label": "Tone",
                    "type": "select",
                    "required": False,
                    "max_chars": 50,
                    "placeholder": "",
                    "options": ["Direct", "Professional", "Technical"],
                    "default": "Professional",
                },
            ],
        },
    ]


def get_enabled_tools() -> List[ToolEntry]:
    return [tool for tool in get_tools_registry() if tool.get("is_enabled")]


def get_tool_by_slug(slug: str) -> Optional[ToolEntry]:
    for tool in get_tools_registry():
        if tool.get("slug") == slug:
            return tool
    return None
