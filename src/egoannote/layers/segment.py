"""Segmentation: turn per-window WindowCaption records (each holding one or
more ordered actions) into variable-length Segment records.

STATUS: design finalized (plan "S1-S5" + cycle-2 "C1 RESOLUTION"), NOT YET
IMPLEMENTED. This is deliberate, not an oversight: two of this module's
constants are explicitly "measure before you build" per the plan --

  - config.SEG_SNAP_RADIUS_SECONDS_PROVISIONAL must be replaced with the
    measured p90 of |proposed - human| boundary error (plan S3), which
    requires the Phase 6.5 pilot review to exist first.
  - The velocity-minima snap-support scorer must be re-tuned against
    MediaPipe output from the S6-fixed hand assignment (layers/hands.py) --
    the ORIGINAL measurement (1.23 minima per +-0.75s window, ~6.8:1 vs
    assumed boundaries) was taken on the pre-fix, corrupted parquet and is
    explicitly flagged provisional in the plan (S2).

Writing the scoring thresholds now, before either measurement exists, would
produce code that has to be substantially reworked once real numbers land --
exactly the kind of premature-precision mistake this project's whole
methodology argues against. Land Phases 0-5 first (get real captions +
real hand tracks), run Phase 6.5's pilot review, THEN implement this against
measured constants.

The design that IS settled, for whoever picks this up:

    E(t) is not a fused continuous curve. Boundary CANDIDATES come from
    s_label (every action boundary the VLM reported -- both intra-window,
    from one WindowCaption's action list, and at window seams, comparing
    the last action of window N to the first action of window N+1, using
    the SAME comparison function either way since every action item shares
    one shape -- see parse.ActionItem).

    Each candidate is then SCORED, not relocated:

        score(b) = w_label * label_significance(b)
                 + w_motion * snap_support(b)   # 0 if no qualifying minimum
                 + w_visual * visual_delta(b)

    snap_support is a validator, not a locator: prominence x exp(-|dt|/tau),
    scored against a MEASURED threshold, contributing 0 (not a relocation)
    when no velocity minimum clears it. label_significance is the field-
    weighted comparison kept from v1 (task_step=1.0, both hands=0.8, one
    hand=0.4), applied identically to intra-window and window-seam
    comparisons because both compare two ActionItem-shaped objects.

    Stamp `boundary_source` ("intra_window" | "window_seam") and
    `boundary_method` ("label+motion+visual" | "label+visual" |
    "label+visual_unsnapped") on every Segment -- these are NOT decorative;
    downstream verification (accuracy, calibration) must stratify by both,
    per the plan's Eng review findings A1-5 and S2.

See docs/SCHEMA.md (once written) for the full Segment field list, and
schema.Segment for the current dataclass shape.
"""

from __future__ import annotations

# Intentionally no implementation yet -- see module docstring.
