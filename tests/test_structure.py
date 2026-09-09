from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_trajectory_project_structure():
    required = [
        ROOT / "config" / "project_config.json",
        ROOT / "trajectory" / "agent.py",
        ROOT / "evaluate.py",
        ROOT / "main.py",
    ]
    assert all(path.exists() for path in required)
