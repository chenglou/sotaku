"""Static exclusions for Modal images that upload the project directory."""

PROJECT_IGNORE = [
    "venv/", ".venv/", "**/__pycache__/", "**/*.pyc",
    ".git/", ".claude/", ".codex/", ".agents/",
    ".env", ".env.*", "temp-side-convo.txt",
    "logs/", "runs/", "runs_modal/", "**/*.pt", "**/*.log",
    "release/validation/", "release/v2/*.zip",
]
