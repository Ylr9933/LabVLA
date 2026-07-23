#!/usr/bin/env bash
# Deploy the JAKA + AGV 10-D LabVLA policy.

set -euo pipefail
SERVE_ENTRYPOINT="serve_jaka_mobile.py"
export SERVE_ENTRYPOINT
exec "$(dirname "${BASH_SOURCE[0]}")/deploy_jaka.sh" "$@"
