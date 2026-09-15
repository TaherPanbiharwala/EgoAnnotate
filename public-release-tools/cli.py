#!/usr/bin/env python3
"""CLI for the historical EgoBlur-redacted-video annotate/publish pipeline.

    uv run public-release-tools/cli.py <command> --help

Four commands, all superseded by the active curate-original /
annotate-curated-original pipeline (see egoannote-run --help) but kept
working as a public reference for the Hugging Face publishing mechanism:
`annotate` (MediaPipe + dense VLM captioning on an EgoBlur-redacted video),
`approve-redaction` (record a private human approval of one redaction),
`publish-hf` (upload the privacy-safe bundle), and `prepare-public-release`
(build an owner-approved public dataset folder from curated face-free
artifacts). See public-release-tools/README.md.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from annotate_redacted import (
    VideoInput,
    _load_batch,
    annotate_video,
    create_redaction_review,
    publish_run,
)
from egoannote.pipeline import _parse_windows, _validate_video_id
from public_release import build_public_release

log = logging.getLogger("public_release_tools.cli")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    annotate = sub.add_parser("annotate", help="run MediaPipe and/or dense VLM captioning")
    annotate.add_argument("--run-dir", type=Path, required=True)
    annotate.add_argument("--batch-manifest", type=Path)
    annotate.add_argument("--redacted-video", type=Path)
    annotate.add_argument("--original-video", type=Path)
    annotate.add_argument("--blur-manifest", type=Path)
    annotate.add_argument(
        "--redaction-review",
        type=Path,
        help="private hash-bound human approval required for a non-PASS EgoBlur manifest",
    )
    annotate.add_argument("--video-id")
    annotate.add_argument("--model", action="append", default=[])
    annotate.add_argument("--models-toml", type=Path)
    annotate.add_argument("--max-spend-per-model", type=float, default=10.0)
    annotate.add_argument("--workers", type=int, default=1)
    annotate.add_argument("--pilot-windows")
    annotate.add_argument("--skip-hands", action="store_true")
    annotate.add_argument("--skip-captions", action="store_true")
    annotate.add_argument("--prune-caption-frames", action="store_true")

    publish = sub.add_parser("publish-hf", help="upload only the privacy-safe bundle")
    publish.add_argument("--run-dir", type=Path, required=True)
    publish.add_argument("--repo-id", required=True)
    publish.add_argument("--public", action="store_true")
    publish.add_argument("--allow-unverified-redaction", action="store_true")
    publish.add_argument("--license-file", type=Path)
    publish.add_argument("--license-name")

    public_release = sub.add_parser(
        "prepare-public-release",
        help="build an owner-approved public dataset folder from curated face-free artifacts",
    )
    public_release.add_argument(
        "--children-dir",
        type=Path,
        required=True,
        help="private face-free child directory containing one <video-id> directory per clip",
    )
    public_release.add_argument(
        "--annotation-run-dir",
        type=Path,
        action="append",
        required=True,
        help="ordered private run directory; first hash-bound complete annotation set wins",
    )
    public_release.add_argument(
        "--annotated-video-dir",
        type=Path,
        action="append",
        required=True,
        help="ordered directory containing final hand-and-caption overlay MP4s",
    )
    public_release.add_argument("--output-dir", type=Path, required=True)
    public_release.add_argument("--video-id", action="append", required=True)
    public_release.add_argument("--model", required=True)
    public_release.add_argument("--approved-by", required=True)
    public_release.add_argument("--release-version", default="v1.0")

    approve = sub.add_parser(
        "approve-redaction",
        help="record named private human approval of one reviewed redacted video",
    )
    approve.add_argument("--run-dir", type=Path, required=True)
    approve.add_argument("--video-id", required=True)
    approve.add_argument("--redacted-video", type=Path, required=True)
    approve.add_argument("--blur-manifest", type=Path, required=True)
    approve.add_argument("--hand-suppression-report", type=Path, required=True)
    approve.add_argument("--yunet-report", type=Path, required=True)
    approve.add_argument("--reviewer", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "annotate":
        if args.batch_manifest:
            if args.redacted_video:
                parser.error("use either --batch-manifest or --redacted-video, not both")
            inputs = _load_batch(args.batch_manifest)
        else:
            if not args.redacted_video:
                parser.error("--redacted-video is required without --batch-manifest")
            inputs = [
                VideoInput(
                    redacted_video=args.redacted_video,
                    original_video=args.original_video,
                    blur_manifest=args.blur_manifest,
                    redaction_review=args.redaction_review,
                    video_id=args.video_id,
                )
            ]
        pilot = _parse_windows(args.pilot_windows)
        for item in inputs:
            video_id = annotate_video(
                item,
                run_dir=args.run_dir.resolve(),
                model_ids=args.model,
                registry_path=args.models_toml,
                max_spend_per_model=args.max_spend_per_model,
                workers=args.workers,
                pilot_windows=pilot,
                run_hands=not args.skip_hands,
                run_captions=not args.skip_captions,
                prune_caption_frames=args.prune_caption_frames,
            )
            print(f"complete: {video_id}")
        return 0

    if args.command == "approve-redaction":
        output = create_redaction_review(
            run_dir=args.run_dir,
            video_id=args.video_id,
            redacted_video=args.redacted_video,
            blur_manifest=args.blur_manifest,
            hand_suppression_report=args.hand_suppression_report,
            yunet_report=args.yunet_report,
            reviewer=args.reviewer,
        )
        print(f"approved: {args.video_id} -> {output}")
        return 0

    if args.command == "publish-hf":
        publish_run(
            args.run_dir.resolve(),
            repo_id=args.repo_id,
            private=not args.public,
            allow_unverified_redaction=args.allow_unverified_redaction,
            license_file=args.license_file,
            license_name=args.license_name,
        )
        return 0

    release = build_public_release(
        children_dir=args.children_dir,
        annotation_run_dirs=args.annotation_run_dir,
        annotated_video_dirs=args.annotated_video_dir,
        output_dir=args.output_dir,
        video_ids=[_validate_video_id(video_id) for video_id in args.video_id],
        model_id=args.model,
        approved_by=args.approved_by,
        release_version=args.release_version,
    )
    print(
        f"complete: {len(release['video_ids'])} public release video(s) -> "
        f"{release['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
