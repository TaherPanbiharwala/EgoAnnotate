#!/usr/bin/env bash
# Create a short, web-friendly MP4 preview (and optionally a poster image).
# Requires: ffmpeg and ffprobe

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/extract_preview.sh INPUT_VIDEO [OPTIONS]

Create a 5–10 second H.264 MP4 preview suitable for a blog or website.

Options:
  -s, --start SECONDS       Start offset in seconds (default: 0)
  -d, --duration SECONDS    Preview duration: 5 through 10 seconds (default: 8)
  -o, --output PATH         Output MP4 path (default: INPUT_preview.mp4)
  -p, --poster PATH         Also write a JPEG poster image at the preview start
  -h, --help                Show this help text

Examples:
  # Eight-second preview from the beginning.
  scripts/extract_preview.sh /path/to/video.mp4

  # Five-second preview beginning at 1 minute 20 seconds, with a poster.
  scripts/extract_preview.sh /path/to/video.mp4 --start 80 --duration 5 \
    --output public/media/shop-preview.mp4 --poster public/media/shop-preview.jpg
EOF
}

fail() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

input=''
start='0'
duration='8'
output=''
poster=''

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--start)
      [[ $# -ge 2 ]] || fail "$1 requires a value"
      start="$2"
      shift 2
      ;;
    -d|--duration)
      [[ $# -ge 2 ]] || fail "$1 requires a value"
      duration="$2"
      shift 2
      ;;
    -o|--output)
      [[ $# -ge 2 ]] || fail "$1 requires a value"
      output="$2"
      shift 2
      ;;
    -p|--poster)
      [[ $# -ge 2 ]] || fail "$1 requires a value"
      poster="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      fail "unknown option: $1"
      ;;
    *)
      [[ -z "$input" ]] || fail "only one input video may be supplied"
      input="$1"
      shift
      ;;
  esac
done

[[ -n "$input" ]] || { usage >&2; exit 1; }
[[ -f "$input" ]] || fail "input video does not exist: $input"
command -v ffmpeg >/dev/null 2>&1 || fail "ffmpeg is not installed or not on PATH"
command -v ffprobe >/dev/null 2>&1 || fail "ffprobe is not installed or not on PATH"

is_nonnegative_number() {
  [[ "$1" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]
}

is_nonnegative_number "$start" || fail "start must be a non-negative number of seconds"
is_nonnegative_number "$duration" || fail "duration must be a number of seconds"
awk -v value="$duration" 'BEGIN { exit !(value >= 5 && value <= 10) }' \
  || fail "duration must be between 5 and 10 seconds"

if [[ -z "$output" ]]; then
  base="${input%.*}"
  [[ "$base" != "$input" ]] || base="$input"
  output="${base}_preview.mp4"
fi

input_duration="$(ffprobe -v error -show_entries format=duration -of default=nokey=1:noprint_wrappers=1 "$input")"
[[ -n "$input_duration" && "$input_duration" != "N/A" ]] \
  || fail "could not determine input duration"
awk -v offset="$start" -v length="$duration" -v total="$input_duration" \
  'BEGIN { exit !(offset + length <= total + 0.001) }' \
  || fail "requested preview extends beyond the input duration (${input_duration}s)"

mkdir -p "$(dirname "$output")"
printf 'Creating %ss preview starting at %ss:\n  %s\n' "$duration" "$start" "$output"
ffmpeg -hide_banner -y -ss "$start" -i "$input" -t "$duration" \
  -map 0:v:0 -map 0:a? -c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p \
  -movflags +faststart -c:a aac "$output"

if [[ -n "$poster" ]]; then
  mkdir -p "$(dirname "$poster")"
  printf 'Creating poster image:\n  %s\n' "$poster"
  ffmpeg -hide_banner -y -ss "$start" -i "$input" -frames:v 1 -update 1 -q:v 2 "$poster"
fi

printf 'Done.\n'
