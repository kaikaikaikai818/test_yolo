"""Backward-compatible launcher for the staged grasp test program.

Use grasp_test.py for new runs. The future text-command production entry point
will be a separate main.py and will not include this test workflow.
"""
from grasp_test_workflow import *  # Re-export existing helper API for tests/tools.
from grasp_test_workflow import main

if __name__ == "__main__":
    main()
