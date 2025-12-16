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
        {
            "slug": "daily-checkin",
            "title": "Daily Habit Check-In",
            "description": "Track one habit per day with a simple done or skipped check-in.",
            "category": "Habits",
            "is_enabled": True,
            "output_label": "Status",
            "inputs": [
                {
                    "name": "habit_name",
                    "label": "Habit name",
                    "type": "text",
                    "required": True,
                    "max_chars": 80,
                    "placeholder": "e.g., Walk 20 minutes",
                },
                {
                    "name": "status",
                    "label": "Status",
                    "type": "select",
                    "required": True,
                    "max_chars": 10,
                    "placeholder": "",
                    "options": ["Done", "Skipped"],
                    "default": "Done",
                },
            ],
        },
        {
            "slug": "expense-splitter",
            "title": "Expense Splitter",
            "description": "Create a simple shared expense sheet and see who owes who.",
            "category": "Finance",
            "is_enabled": False,
            "output_label": "Settlement",
            "inputs": [
                {
                    "name": "event_name",
                    "label": "Event name",
                    "type": "text",
                    "required": True,
                    "max_chars": 80,
                    "placeholder": "e.g., Tahoe trip",
                },
                {
                    "name": "participants",
                    "label": "Participants",
                    "type": "text",
                    "required": True,
                    "max_chars": 400,
                    "placeholder": "e.g., Will, Sam, Jordan",
                },
                {
                    "name": "expenses",
                    "label": "Expenses",
                    "type": "textarea",
                    "required": True,
                    "max_chars": 5000,
                    "placeholder": "Format: payer | amount | description | split(optional)\nExamples:\nWill | 60 | Groceries\nSam | 120 | Airbnb | Will, Sam, Jordan\nJordan | 30 | Gas | Sam, Jordan",
                },
            ],
        },
        {
            "slug": "trip-planner",
            "title": "Trip Planner",
            "description": "Plan a trip, track budgets, and settle shared expenses with a shareable link.",
            "category": "Travel",
            "is_enabled": True,
            "output_label": "Trip summary",
            "inputs": [
                {
                    "name": "trip_name",
                    "label": "Trip name",
                    "type": "text",
                    "required": True,
                    "max_chars": 80,
                    "placeholder": "e.g., Tahoe trip",
                },
                {
                    "name": "currency",
                    "label": "Currency",
                    "type": "select",
                    "required": True,
                    "max_chars": 3,
                    "placeholder": "",
                    "options": ["USD", "EUR", "GBP"],
                    "default": "USD",
                },
                {
                    "name": "notes",
                    "label": "Notes",
                    "type": "textarea",
                    "required": False,
                    "max_chars": 1000,
                    "placeholder": "Key details, packing notes, or reminders.",
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
