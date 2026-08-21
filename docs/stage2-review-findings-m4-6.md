# Stage II Milestones 4-6 review findings

Recorded 2026-08-21. Scope: `git diff c7b4005..HEAD` — Milestone 4 (Stage
I-only rendering and technical verification), Milestone 5 (private review,
labels, and operator workflow), and the in-progress Milestone 6 preparation
(offline GPU smoke, verified snapshot loading, SAM frame-window extraction).
Milestones 1-3 already went through a separate review pass (fix commits
`a31e65c`, `ce08d73`, `c5609ca`) and were treated as a reviewed baseline, not
re-audited here.

Method: 9 parallel finder passes (5 correctness angles, 3 cleanup angles, 1
altitude angle; no project `CLAUDE.md` exists, so the conventions angle was
skipped), independent verification of every surviving candidate, one gap-sweep
pass, and verification of the sweep's own candidates. Two candidates were
investigated and refuted: an unbounded `frame_end` check in
`register_private_evidence` (no downstream consumer ever reads the value) and
a `fingerprint()`-bypass in private-artifact hashing (no code path actually
reuses a stale private artifact across a code-version bump). Neither is listed
below.

All 15 findings below were independently verified (CONFIRMED). All 15 are now
fixed, on this same branch, each in its own commit with a mutation-tested
regression test (revert the fix, confirm the new test fails; restore it,
confirm the suite passes). Commits, in the same order as the findings below:

1. `f402ee8` — also covers #2
2. `f402ee8`
3. `32dab4e`
4. `f8f85a0`
5. `1f1095c`
6. `1edf808`
7. `fc727b3`
8. `ec0d030`
9. `5f0382f`
10. `2334a4d` (Python side mutation-tested; the shell-script half sits on the
    real-GPU-pod path, the same pre-existing testability boundary as the rest
    of that script, so it's verified by code review and `bash -n` rather than
    an automated test)
11. `88badd8`
12. `1462af9` (test-only fix; verified by reproducing the exact cross-test
    leak with a throwaway probe rather than a kept regression test)
13. `21a747b`
14. `d772acc`
15. `459f7ae`

Full suite: 473 passed (up from 453 before this pass). This file is kept as
the historical record of the review; treat `git log` on this branch as
authoritative for exact diffs.

## 1. `invalidate_from` erases review history with no audit trail

**File:** `jobs/20_deidentify_stage2.py:6668` (also `_remove_layer_path`,
`:6625-6649`)

`invalidate_from()` unconditionally deletes the entire `reviews_dir` on any
`--recompute-from` choice (`dino`/`sam`/`render`), bypassing
`create_review_record`'s content-hash immutability guard entirely. No
history/audit-log mechanism anywhere in the file preserves what was deleted.

**Failure scenario:** A reviewer records `REJECTED` for a clip. An operator
runs `resume --recompute-from render` with unchanged inputs; since rendering
is deterministic and reproduces byte-identical output/manifest hashes,
`invalidate_from` wipes the reviews directory, and a fresh
`review --decision ACCEPTED` can then be recorded for the exact same content
with zero trace anywhere that it was previously rejected.

## 2. Release gate sorts review timestamps as strings

**File:** `jobs/20_deidentify_stage2.py:5918`

`effective_review_status()` picks the "most recent" review via `max()` on the
raw `reviewed_at` ISO-8601 string rather than a parsed datetime, so two
records with different fractional-second precision for the same
manifest/output hash can sort out of chronological order.

**Failure scenario:** A `REJECTED` review timestamped
`"2026-08-20T12:00:00Z"` and a later `ACCEPTED` review timestamped
`"2026-08-20T12:00:00.500000Z"` (500ms afterward) sort with the `REJECTED`
one as string-max, because `.` sorts before `Z` in ASCII. `release_check()`,
the fail-closed publication gate, can therefore pick the stale `REJECTED`
record over the newer `ACCEPTED` one (or vice versa) — no test exercises two
records with differing precision to catch this.

## 3. DINO snapshot verification is far weaker than SAM's

**File:** `jobs/20_deidentify_stage2.py:2319`

`TransformersGroundingDinoAdapter`'s "verified persistent snapshot" path
hashes only `model.safetensors` (`config.json`/`preprocessor_config.json`/
tokenizer files load unverified), and skips hash verification entirely when
`model_path` isn't passed — unlike SAM2's directory-wide
`sha256_directory_tree` + `SamRuntimeIdentity` binding threaded into
`sam_fingerprint`.

**Failure scenario:** A tampered or mismatched config/preprocessor/tokenizer
file sitting alongside a correctly-hashed `model.safetensors` loads unverified
and silently changes DINO's preprocessing or label decoding.
`run_offline_gpu_smoke` (the real Milestone 6 entry point) always passes
`model_path`, so this narrower-than-SAM verification is the live production
behavior, not just a theoretical gap — Milestone 6's claimed "verified
persistent DINO snapshot binding" is not actually equivalent to SAM's
protection.

## 4. `release_check` leak scan omits DINO/SAM artifacts

**File:** `jobs/20_deidentify_stage2.py:5964`

`release_check`'s internal-artifact leak-detection set only binds the render
artifact's hash — `dino_artifact` and `sam_mask_shards`, both already present
in the same in-scope manifest dict, are never added, even though both live
outside `DO-NOT-SHIP` as siblings of `render/` in the run root.

**Failure scenario:** An operator stages a release candidate by copying the
run root minus `DO-NOT-SHIP/` — the pattern `release_check`'s own design
anticipates — and naturally sweeps in `dino/` and `sam/` alongside `render/`.
`release_check` passes even though those artifacts encode precise per-frame
face bounding boxes and binary face-silhouette masks derived from the
unredacted original video, including faces Stage I's automated blur may have
missed.

## 5. `audit_status` hides a skipped Stage I YuNet check

**File:** `jobs/20_deidentify_stage2.py:5558`

`finalize_verified_processing`'s `audit_status` computation only
special-cases `stage1_status == "NEEDS_REVIEW"`, silently collapsing the
weaker `PASS_AUTOMATED_NO_YUNET` Stage I status into the same
`PASS_AUTOMATED_TECHNICAL` result as a full `PASS_AUTOMATED` — contradicting
`AGENTS.md`'s own "a check that didn't run is never a pass" rule, which
`jobs/10_blur_egoblur.py` already enforces for Stage I itself.

**Failure scenario:** A clip whose Stage I redaction skipped the independent
YuNet cross-check ends up with the exact same Stage II `audit_status` as one
verified by the full check. No downstream consumer (the `status` command,
release-check, review workflow) surfaces the weaker claim, so an operator
reading Stage II's own audit trail has no way to know this clip's upstream
verification was weaker.

## 6. `window_loader` arity mismatch blocks real wiring

**File:** `jobs/20_deidentify_stage2.py:3163`

`extract_attested_sam_window`'s all-keyword-only signature
(`def extract_attested_sam_window(*, stage1, paths, window)`) is incompatible
with `generate_sam_mask_shards`' `window_loader(window)` positional-call
contract at line 4346 — wiring the real Milestone 6 extractor into the
Milestone 3 orchestration the obvious way raises `TypeError` on the very
first window.

**Failure scenario:** Passing `window_loader=extract_attested_sam_window` to
`generate_sam_mask_shards` — mirroring how `run_fake_pipeline` wires its fake
lambda today — crashes immediately with `TypeError:
extract_attested_sam_window() takes 0 positional arguments but 1 was given`.
No test or code path currently exercises this integration point, so it will
surface only when someone finishes wiring the real (non-fake) pipeline.

## 7. SAM cache-hit path still requires the archived source video

**File:** `jobs/20_deidentify_stage2.py:3174`

`generate_sam_mask_shards` calls the real window loader before checking
shard-cache existence, and `extract_attested_sam_window` unconditionally
stats the original source video via a dead `file_stamp(source_path)` call
that's never reused on the cache-hit branch — regressing the "reuse without
touching the source video" guarantee. The test that used to enforce this
(asserting the loader is never called on a cache hit) was rewritten in the
same commit to assert the opposite.

**Failure scenario:** After the original source video is archived or deleted
from the pod's workspace (a normal step once shards are cached), re-running
Stage II with every shard already cached now hits the dead
`file_stamp(source_path)` stat inside `extract_attested_sam_window`, raises
`Stage2Error` (caught per-window), and silently discards the fully-verified
cached shard in favor of regenerating a degraded DINO-fallback-only shard
with a review flag.

## 8. TOCTOU between hash and bytes written to evidence

**File:** `jobs/20_deidentify_stage2.py:2871`

`register_private_evidence` hashes the source file once via
`artifact_ref(source)` and embeds that hash in the persisted evidence
artifact's filename, then performs a completely separate, unguarded
`source.read_bytes()` call whose result is what actually gets written — with
no re-verification that the file still matches the hash already baked into
the filename.

**Failure scenario:** If the source file (e.g. a face-crop export another
tool is concurrently producing) changes between the two reads, the persisted
private evidence file's content no longer matches the sha256 embedded in its
own filename — breaking this codebase's content-addressing invariant for
exactly this one artifact class, unlike every other ingestion path
(`extract_attested_sam_window`, `artifact_ref` itself) which re-verifies via
`file_stamp()` immediately before trusting content.

## 9. No stop-check inside real DINO/SAM/render loops

**File:** `jobs/20_deidentify_stage2.py:2186` (also `:4338`, `:4924`/`:5312`)

`ensure_run_not_stopped` is only called between layers inside
`run_fake_pipeline` (all 4 call sites in the whole file live there) — the
real per-anchor DINO loop (`generate_dino_proposals`), per-window SAM loop
(`generate_sam_mask_shards`, line 4338), and per-frame render loop
(`_render_frames`/`render_stage2_video`, lines 4924/5312) never check the
stop marker internally.

**Failure scenario:** Once real GPU execution is wired for Milestone 6+, an
operator running `stop` mid-clip during, say, anchor 50 of 300 DINO anchors,
or partway through a full-clip render, gets no response until that entire
layer's real per-item work finishes — potentially the majority of the run's
GPU-billed time — because no loop body ever re-checks the stop marker.

## 10. Render/GPU-smoke promotion skips fsync before rename

**File:** `jobs/20_deidentify_stage2.py:5280`; also
`scripts/runpod_setup_stage2.sh:288`

`_promote_render_output` never fsyncs the rendered video's content
before/during promotion, unlike the sibling `extract_attested_sam_window`
(same milestone-era code) which reopens and fsyncs every produced file
first. `scripts/runpod_setup_stage2.sh`'s GPU-smoke-evidence `mv` has the
identical gap while the script's own nearby asset-manifest write correctly
fsyncs first.

**Failure scenario:** A crash or power loss at the exact moment between the
ffmpeg render finishing and `_promote_render_output`'s `os.replace` (or
between the `doctor --load-models` subprocess finishing and the shell
script's `mv`) can leave the promoted "immutable" render output or GPU-smoke
evidence file truncated or holding stale buffered data that was never
flushed to disk, silently breaking the file's own recorded hash on next
read.

## 11. Deleted private artifact misreported as escaped

**File:** `jobs/20_deidentify_stage2.py:5608` (also `finalize_verified_processing`, `:5527-5547`)

`_manifest_private_artifacts` and `finalize_verified_processing` both catch
`FileNotFoundError` (from `path.resolve(strict=True)`) and a genuine
`ValueError` from `relative_to()` in the same except clause, raising the
identical `PRIVATE_ARTIFACT_OUTSIDE_DO_NOT_SHIP` error for both — a routine
deleted-file case is indistinguishable from an artifact actually relocated
outside the private directory.

**Failure scenario:** An operator deletes one stale private label JSON still
referenced by the manifest. Running `status`, `review`, or `release-check`
fails with an alarming "escaped DO-NOT-SHIP" error that reads like a
possible privacy/data-exfiltration incident, when the actual cause is a
routine missing file that just needs restoring or recomputing.

## 12. Test leaks offline env vars into later tests

**File:** `jobs/20_deidentify_stage2.py:6864`; `tests/test_stage2_gpu.py:281-304`

`run_offline_gpu_smoke` sets `os.environ['TRANSFORMERS_OFFLINE']`/
`['HF_HUB_OFFLINE']` via direct assignment instead of `monkeypatch`; the
covering test's `monkeypatch.delenv(..., raising=False)` is a no-op when the
var wasn't already present, so pytest's teardown never unsets it — the
forced-offline flags persist in the real process environment for the rest of
the session.

**Failure scenario:** In a normal CI/dev run where these vars aren't
pre-set, running `tests/test_stage2_gpu.py`'s offline-smoke test permanently
forces offline mode for every test that runs afterward in the same pytest
invocation — any later Transformers/HF-touching test's behavior now silently
depends on test execution order rather than its own content.

## 13. `sweep --thresholds` parsed but never validated

**File:** `jobs/20_deidentify_stage2.py:7090`

`sweep.add_argument('--thresholds', ...)` is referenced nowhere else in the
7294-line file — `operator_dry_run` never surfaces it in dry-run JSON
output, and no code path range-validates the values.

**Failure scenario:** An operator sanity-checking before paying for GPU time
runs `--dry-run sweep GX010057 --thresholds 0.15 0.20 0.25 3.0` (typo for
`0.30`); argparse silently accepts the out-of-range float and the dry-run
output gives no feedback on thresholds at all. Lower severity since `sweep`
is still fully gated behind `_deferred_gpu_error`, so this is scaffolding for
an unwired feature rather than a live bug today.

## 14. Render path re-hashes multi-GB video redundantly

**File:** `jobs/20_deidentify_stage2.py:5324`

`render_stage2_video` and `_promote_render_output` SHA-256 the full Stage I
source video up to 3 times and the full rendered output up to 4 times within
a single render call as repeated re-verification, even though the file's own
cheap `file_stamp()` (mtime+size) idiom — already used elsewhere in this
file for exactly this kind of recheck — could avoid re-hashing unchanged
large files.

**Failure scenario:** On a per-hour-billed GPU pod processing full-length
clips, each `render_stage2_video` call pays for 3-4 full-content SHA-256
passes over multi-gigabyte video files that haven't changed between checks,
adding real wall-clock and dollar cost on every run and every reuse-path
invocation.

## 15. `color_fields` tuple duplicated between render/verify

**File:** `jobs/20_deidentify_stage2.py:4895` (also `verify_rendered_video`, `:5073`)

The identical 5-item `color_fields` tuple (`color_range`/`color_space`/
`color_transfer`/`color_primaries`/`chroma_location`) and its extraction
comprehension are independently defined in both `_render_frames` (the write
path, line 4895) and `verify_rendered_video` (the verification path, line
5073), with no shared module-level constant.

**Failure scenario:** Adding or removing a color-fact field in
`_render_frames` but forgetting the identical change in
`verify_rendered_video` (or vice versa) would make the renderer record a
color fact that verification never checks, or make verification demand a
fact the renderer never records — silently weakening the exact
color-fidelity check this fail-closed verification step exists to enforce.

## Refuted candidates (not fixed, not real bugs)

- **`register_private_evidence`'s unbounded `frame_end`** — weaker than
  `validate_face_event_labels`'s bound, but nothing downstream ever reads the
  value back; it's write-only metadata under `DO-NOT-SHIP`.
- **Private-artifact hashing bypassing `fingerprint()`** — a real stylistic
  inconsistency (raw `hashlib.sha256(canonical_json_bytes(...))` instead of
  the canonical helper), but every write and load path re-runs current-code
  validation regardless, so no stale private artifact is ever silently
  trusted across a code-version bump.
