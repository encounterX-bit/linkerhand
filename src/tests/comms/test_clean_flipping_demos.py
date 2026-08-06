from pathlib import Path

import numpy as np

from scripts.clean_flipping_demos import (
    FaceRun,
    all_face_changes,
    candidate_segments,
    parse_route,
    smooth_faces,
    stable_face_runs,
)


def _row(value: int) -> dict:
    action = [255.0] * 20
    action[5] = float(value)
    action[10] = float(value)
    return {"last_action": action}


def test_parse_route_builds_explicit_directed_edges():
    assert parse_route("0>4, 2>5") == {(0, 4), (2, 5)}


def test_all_face_changes_contains_every_nonself_directed_edge():
    route = all_face_changes()
    assert len(route) == 30
    assert (0, 5) in route
    assert (5, 0) in route
    assert all(source != destination for source, destination in route)


def test_face_smoothing_and_stable_runs_remove_short_detection_glitch():
    indices = list(range(0, 45, 5))
    faces = [2, 2, 2, 4, 2, 5, 5, 5, 5]

    runs = stable_face_runs(
        indices,
        smooth_faces(faces, radius=1),
        minimum_samples=2,
    )

    assert [run.face for run in runs] == [2, 5]


def test_candidate_segment_has_bounded_context_and_metrics():
    rows = [_row(value) for value in np.linspace(255, 0, 200)]
    runs = [FaceRun(2, 0, 80, 17), FaceRun(5, 120, 199, 16)]

    segments = candidate_segments(
        Path("episode_000"),
        rows,
        runs,
        context_before=30,
        context_after=40,
    )

    assert len(segments) == 1
    segment = segments[0]
    assert (segment.from_face, segment.to_face) == (2, 5)
    assert (segment.start_frame, segment.end_frame) == (50, 160)
    assert segment.frame_count == 111
    assert segment.action_reversals == 0
    assert segment.thumb_route_reversals == 0
    assert segment.total_action_variation > 0


def test_final_transition_keeps_only_last_stable_pair_and_full_runs():
    rows = [_row(value) for value in np.linspace(255, 0, 300)]
    runs = [
        FaceRun(2, 0, 79, 16),
        FaceRun(4, 80, 159, 16),
        FaceRun(2, 160, 239, 16),
        FaceRun(5, 240, 299, 12),
    ]

    segments = candidate_segments(
        Path("episode_000"),
        rows,
        runs,
        context_before=0,
        context_after=0,
        selection="final-transition",
    )

    assert len(segments) == 1
    segment = segments[0]
    assert (segment.from_face, segment.to_face) == (2, 5)
    assert (segment.start_frame, segment.end_frame) == (160, 299)
