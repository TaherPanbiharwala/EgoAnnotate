#!/usr/bin/env python3
"""CLI for EgoBlur's pre/post-redaction support tooling.

    uv run egoblur/cli.py <command> --help

Five commands, all private/local, none wired into the main egoannote-run
pipeline: `pose-prior` and `hand-prior` build pre-redaction MediaPipe priors
used by job.py's wearer-hand suppression; `verify-yunet` is an independent
post-redaction face audit; `init-yunet-decisions` and
`decisions-to-forced-boxes` turn a YuNet report into a human review template
and then into job.py's `--forced-boxes` input. This folder is parked but
kept working as a reference — see egoblur/README.md.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from egoblur import hand_prior as hand_prior_layer
from egoblur import pose_prior as pose_prior_layer
from egoblur.verify_yunet import create_decision_template, decisions_to_forced_boxes, verify_yunet

log = logging.getLogger("egoblur.cli")

_VIDEO_ID_RE = re.compile(r"^[\w.-]+$", flags=re.UNICODE)


def _validate_video_id(video_id: str) -> str:
    """Keep a video identifier confined to its run directory and Drive prefix."""
    if (
        not video_id
        or video_id in {".", ".."}
        or not _VIDEO_ID_RE.fullmatch(video_id)
        or "/" in video_id
        or "\\" in video_id
    ):
        raise ValueError(
            "video_id must contain only letters, numbers, underscore, dot, or hyphen and "
            "must not be '.' or '..'"
        )
    return video_id


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    yunet = sub.add_parser(
        "verify-yunet",
        help="CPU-only independent face audit on a redacted video; writes a private report",
    )
    yunet.add_argument("--redacted-video", type=Path, required=True)
    yunet.add_argument("--blur-manifest", type=Path, required=True)
    yunet.add_argument("--checkpoint-dir", type=Path, required=True)
    yunet.add_argument("--yunet-model", type=Path, required=True)
    yunet.add_argument(
        "--job-script",
        type=Path,
        help="defaults to job.py next to this cli.py (egoblur/job.py)",
    )
    yunet.add_argument("--ffmpeg", default="ffmpeg")
    yunet.add_argument("--report", type=Path, required=True)
    yunet.add_argument(
        "--preview-video",
        type=Path,
        help="optional redacted-only local overlay; path must include private or DO-NOT-SHIP",
    )
    yunet.add_argument(
        "--pose-prior",
        type=Path,
        help="optional private pre-redaction pose prior; ranks likely wearer-limb false positives",
    )
    yunet.add_argument(
        "--hand-prior",
        type=Path,
        help=(
            "optional private 30-fps pre-redaction Hand Landmarker prior; filters likely "
            "wearer-hand YuNet noise using expanded amber/pink hand regions"
        ),
    )
    yunet.add_argument(
        "--candidate-contact-sheet-dir",
        type=Path,
        help="optional private redacted-only contact sheets, one per temporal YuNet candidate",
    )

    pose = sub.add_parser(
        "pose-prior",
        help="private pre-redaction MediaPipe Pose pass; creates a shadow-only limb prior",
    )
    pose.add_argument("--original-video", type=Path, required=True)
    pose.add_argument("--video-id", required=True)
    pose.add_argument("--output", type=Path, required=True)
    pose.add_argument("--models-dir", type=Path, default=Path("private/models"))
    pose.add_argument(
        "--preview-video",
        type=Path,
        help="optional private original-only pose overlay (amber wearer candidate; blue other pose)",
    )
    pose.add_argument("--detect-hz", type=float, default=pose_prior_layer.POSE_DETECT_HZ)
    pose.add_argument("--num-poses", type=int, default=pose_prior_layer.POSE_NUM_POSES)
    pose.add_argument("--confidence", type=float, default=pose_prior_layer.POSE_MIN_CONFIDENCE)

    hand_prior = sub.add_parser(
        "hand-prior",
        help="private pre-redaction MediaPipe Hand Landmarker pass for wearer-hand review",
    )
    hand_prior.add_argument("--original-video", type=Path, required=True)
    hand_prior.add_argument("--video-id", required=True)
    hand_prior.add_argument("--output", type=Path, required=True)
    hand_prior.add_argument("--models-dir", type=Path, default=Path("private/models"))
    hand_prior.add_argument(
        "--preview-video",
        type=Path,
        help="optional private original-only hand overlay (amber stable wearer; magenta provisional; blue other)",
    )
    hand_prior.add_argument("--detect-hz", type=float, default=hand_prior_layer.HAND_DETECT_HZ)
    hand_prior.add_argument(
        "--num-hands",
        type=int,
        default=hand_prior_layer.HAND_NUM_HANDS,
        help=(
            "maximum hands to detect (default: 2 for the pinned active policy; "
            f"use {hand_prior_layer.HAND_REVIEW_NUM_HANDS} for private multi-hand review)"
        ),
    )
    hand_prior.add_argument("--confidence", type=float, default=hand_prior_layer.HAND_MIN_CONFIDENCE)

    decisions = sub.add_parser(
        "init-yunet-decisions",
        help="create a private all-uncertain review-decision template from a YuNet report",
    )
    decisions.add_argument("--report", type=Path, required=True)
    decisions.add_argument("--output", type=Path, required=True)

    forced = sub.add_parser(
        "decisions-to-forced-boxes",
        help="convert only confirmed private YuNet candidate tracks into EgoBlur --forced-boxes JSON",
    )
    forced.add_argument("--report", type=Path, required=True)
    forced.add_argument("--decisions", type=Path, required=True)
    forced.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "verify-yunet":
        job_script = (args.job_script or Path(__file__).resolve().parent / "job.py").resolve()
        result = verify_yunet(
            redacted_video=args.redacted_video.resolve(),
            blur_manifest=args.blur_manifest.resolve(),
            checkpoint_dir=args.checkpoint_dir.resolve(),
            yunet_model=args.yunet_model.resolve(),
            job_script=job_script,
            ffmpeg=args.ffmpeg,
            report=args.report.resolve(),
            preview_video=args.preview_video.resolve() if args.preview_video else None,
            pose_prior_path=args.pose_prior.resolve() if args.pose_prior else None,
            hand_prior_path=args.hand_prior.resolve() if args.hand_prior else None,
            candidate_contact_sheet_dir=(
                args.candidate_contact_sheet_dir.resolve()
                if args.candidate_contact_sheet_dir else None
            ),
        )
        print(f"{result['review_status']}: {args.report}")
        return 0

    if args.command == "pose-prior":
        video_id = _validate_video_id(args.video_id)
        artifact = pose_prior_layer.build_pose_prior(
            original_video=args.original_video,
            video_id=video_id,
            output=args.output,
            models_dir=args.models_dir,
            detect_hz=args.detect_hz,
            num_poses=args.num_poses,
            confidence=args.confidence,
            preview_video=args.preview_video,
        )
        print(f"complete: {artifact['source']['video_id']} -> {args.output}")
        return 0

    if args.command == "hand-prior":
        video_id = _validate_video_id(args.video_id)
        artifact = hand_prior_layer.build_hand_prior(
            original_video=args.original_video,
            video_id=video_id,
            output=args.output,
            models_dir=args.models_dir,
            detect_hz=args.detect_hz,
            num_hands=args.num_hands,
            confidence=args.confidence,
            preview_video=args.preview_video,
        )
        print(f"complete: {artifact['source']['video_id']} -> {args.output}")
        return 0

    if args.command == "init-yunet-decisions":
        template = create_decision_template(args.report.resolve(), args.output.resolve())
        print(f"review template: {len(template['decisions'])} candidate(s) -> {args.output}")
        return 0

    result = decisions_to_forced_boxes(
        args.report.resolve(), args.decisions.resolve(), args.output.resolve()
    )
    print(f"forced boxes: {result['n_forced_boxes']} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
