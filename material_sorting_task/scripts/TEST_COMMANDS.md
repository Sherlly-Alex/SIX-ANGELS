# Formal + research one-shot commands (Windows PowerShell)
#
# Formal tests (no ROS required for unittest; excludes semantic_research):
#   bash scripts/run_formal_tests.sh
#
# Full scheduler/costmap/learning + existing regression suite:
#   PYTHONPATH=.;examples/material_sorting python -m pytest -q
# Windows PowerShell:
#   $env:PYTHONPATH='.;examples/material_sorting'; D:\anaconda\python.exe -m pytest -q
#
# Research unit tests (may need sklearn):
#   bash scripts/run_semantic_research_tests.sh
#
# Offline research eval (independent of run_client.sh):
#   bash scripts/run_semantic_research_eval.sh
