#!/usr/bin/env bash
# Download nflverse play-by-play and weekly player stats (2015–2024) into data/raw.
set -euo pipefail
dest="${PLAYPARSE_DATA_DIR:-$(dirname "$0")/../data/raw}"
mkdir -p "$dest"
base=https://github.com/nflverse/nflverse-data/releases/download
for y in $(seq 2015 2024); do
  curl -sfL -o "$dest/pbp_$y.parquet" "$base/pbp/play_by_play_$y.parquet" &
  curl -sfL -o "$dest/stats_player_week_$y.parquet" "$base/stats_player/stats_player_week_$y.parquet" &
done
wait
ls -la "$dest"
