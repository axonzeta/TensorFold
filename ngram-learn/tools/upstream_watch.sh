#!/usr/bin/env bash
# What changed upstream that matters to ngram-learn: commits on ashhart/TensorFold main and zig-flashnext since the
# last check, PRs updated since then (all, flagged when they touch our areas), and MiaAI-Lab's Flash Next recipe.
# State in .upstream_since; ./upstream_watch.sh [--since 2026-10-10T00:00:00Z]
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
state="$here/.upstream_since"
since=${2:-$(cat "$state" 2>/dev/null || echo 2026-10-09T00:00:00Z)}
[[ "${1:-}" == --since ]] && since=$2
now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
areas='flash ?next|flashnext|qwen4|ngram|n-gram|ple|living|slide|learn|lora|drafter|overlay|cuda|spark|gb10|train'
echo "== upstream since $since"
for br in main zig-flashnext; do
  echo "-- ashhart/TensorFold $br"
  gh api "repos/ashhart/TensorFold/commits?sha=$br&since=$since&per_page=100" --jq '.[] | "\(.sha[0:7]) \(.commit.author.date[0:16]) \(.commit.message | split("\n")[0])"' | head -60
done
echo "-- PRs updated (* = touches our areas)"
gh pr list -R ashhart/TensorFold --state all --limit 100 --search "updated:>=${since:0:10}" --json number,state,title,updatedAt \
  --jq '.[] | "\(.number) \(.state) \(.updatedAt[0:16]) \(.title)"' | while read -r line; do
  if echo "$line" | grep -qiE "$areas"; then echo "* $line"; else echo "  $line"; fi
done
echo "-- releases"
gh release list -R ashhart/TensorFold --limit 3
echo "-- MiaAI-Lab Flash Next recipe"
gh api "repos/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/commits?since=$since&per_page=20" --jq '.[] | "\(.sha[0:7]) \(.commit.author.date[0:16]) \(.commit.message | split("\n")[0])"'
echo "$now" > "$state"
