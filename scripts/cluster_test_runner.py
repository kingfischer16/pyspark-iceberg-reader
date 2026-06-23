"""Entry point for running @pytest.mark.cluster tests on Databricks serverless.

Invoked as spark_python_task by the integration_tests job in databricks.yml.
All arguments after the script name are forwarded to pytest.

Usage (from databricks.yml spark_python_task parameters):
    --tb=short -v --no-header
"""
from __future__ import annotations

import subprocess
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Install test dependencies and the package from the uploaded source tree.
# Done here rather than in the environment spec to avoid serverless getcwd issues.
# pyarrow is skipped — DBR already ships it; installing a duplicate causes conflicts.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet",
     "pytest>=8", "pyiceberg[sql-sqlite]>=0.7", str(ROOT)],
    check=True,
)

result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        str(ROOT / "tests"),
        "-m", "cluster",
        *sys.argv[1:],
    ],
    cwd=ROOT,
)
sys.exit(result.returncode)
