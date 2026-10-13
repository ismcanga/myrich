import json
import os
from pathlib import Path


ROOT = Path("artifacts/experiment-001")

GROUPS_INPUT = ROOT / "failure_overlap_groups.json"
DESCRIPTORS_INPUT = ROOT / "failure_field_descriptions_v2.json"

OUTPUT = ROOT / "failure_overlap_group_descriptions.json"


def frame_by_index(transition, relative_index):
    for frame in transition["frames"]:
        if (
            frame["relative_measurement_index"]
            == relative_index
        ):
            return frame

    return None


def sign(value):
    if value is None:
        return None

    if value > 0:
        return 1

    if value < 0:
        return -1

    return 0


def sign_reversal(
    value_before,
    value_after,
):
    before_sign = sign(
        value_before
    )
    after_sign = sign(
        value_after
    )

    if (
        before_sign is None
        or after_sign is None
    ):
        return None

    if (
        before_sign == 0
        or after_sign == 0
    ):
        return False

    return (
        before_sign
        != after_sign
    )


def sign_change(
    value_before,
    value_after,
):
    before_sign = sign(
        value_before
    )
    after_sign = sign(
        value_after
    )

    if (
        before_sign is None
        or after_sign is None
    ):
        return None

    return (
        before_sign
        != after_sign
    )


def difference(
    value_before,
    value_after,
):
    if (
        value_before is None
        or value_after is None
    ):
        return None

    return (
        value_after
        - value_before
    )


def magnitude_difference(
    value_before,
    value_after,
):
    if (
        value_before is None
        or value_after is None
    ):
        return None

    return (
        abs(value_after)
        - abs(value_before)
    )


def direction_from_change(
    change,
):
    if change is None:
        return None

    if change > 0:
        return "increase"

    if change < 0:
        return "decrease"

    return "unchanged"


def describe_transition(
    transition,
):
    frame_0 = frame_by_index(
        transition,
        0,
    )

    frame_1 = frame_by_index(
        transition,
        1,
    )

    if (
        frame_0 is None
        or frame_1 is None
    ):
        return {
            "transition_id": (
                transition[
                    "transition_id"
                ]
            ),
            "comparison_available": False,
            "reason": (
                "step 0 or step 1 "
                "descriptor frame missing"
            ),
        }
    
    support_0 = frame_0[
        "support"
    ][
        "valid_cell_refs"
    ]

    support_1 = frame_1[
        "support"
    ][
        "valid_cell_refs"
    ]

    if support_0 != support_1:
        return {
            "transition_id": transition[
                "transition_id"
            ],
            "comparison_available": False,
            "reason": (
                "step 0 and step 1 "
                "valid spatial support differs"
            ),
        }

    mean_u_change = difference(
        frame_0[
            "mean_u_mps"
        ],
        frame_1[
            "mean_u_mps"
        ],
    )

    mean_v_change = difference(
        frame_0[
            "mean_v_mps"
        ],
        frame_1[
            "mean_v_mps"
        ],
    )

    mean_speed_change = difference(
        frame_0[
            "mean_speed_mps"
        ],
        frame_1[
            "mean_speed_mps"
        ],
    )

    speed_spread_change = difference(
        frame_0[
            "speed_spread_mps"
        ],
        frame_1[
            "speed_spread_mps"
        ],
    )

    coherence_change = difference(
        frame_0[
            "vector_coherence"
        ],
        frame_1[
            "vector_coherence"
        ],
    )

    vorticity_change = difference(
        frame_0[
            "vorticity_per_s"
        ],
        frame_1[
            "vorticity_per_s"
        ],
    )

    vorticity_magnitude_change = (
        magnitude_difference(
            frame_0[
                "vorticity_per_s"
            ],
            frame_1[
                "vorticity_per_s"
            ],
        )
    )

    divergence_change = difference(
        frame_0[
            "divergence_per_s"
        ],
        frame_1[
            "divergence_per_s"
        ],
    )

    return {
        "transition_id": transition[
            "transition_id"
        ],
        "comparison_available": True,
        "mean_u_sign_reversal": (
            sign_reversal(
                frame_0[
                    "mean_u_mps"
                ],
                frame_1[
                    "mean_u_mps"
                ],
            )
        ),
        "mean_v_sign_reversal": (
            sign_reversal(
                frame_0[
                    "mean_v_mps"
                ],
                frame_1[
                    "mean_v_mps"
                ],
            )
        ),
        "divergence_sign_change": (
            sign_change(
                frame_0[
                    "divergence_per_s"
                ],
                frame_1[
                    "divergence_per_s"
                ],
            )
        ),
        "mean_u_change_mps": (
            mean_u_change
        ),
        "mean_v_change_mps": (
            mean_v_change
        ),
        "mean_speed_change_mps": (
            mean_speed_change
        ),
        "speed_spread_change_mps": (
            speed_spread_change
        ),
        "coherence_change": (
            coherence_change
        ),
        "vorticity_change_per_s": (
            vorticity_change
        ),
        "vorticity_magnitude_change_per_s": (
            vorticity_magnitude_change
        ),
        "divergence_change_per_s": (
            divergence_change
        ),
        "speed_change_direction": (
            direction_from_change(
                mean_speed_change
            )
        ),
        "coherence_change_direction": (
            direction_from_change(
                coherence_change
            )
        ),
        "vorticity_magnitude_change_direction": (
            direction_from_change(
                vorticity_magnitude_change
            )
        ),
    }


def empty_counts():
    return {
        "comparison_available": 0,
        "comparison_unavailable": 0,
        "mean_u_sign_reversal": 0,
        "mean_u_sign_reversal_valid": 0,
        "mean_u_sign_reversal_unknown": 0,
        "mean_v_sign_reversal": 0,
        "mean_v_sign_reversal_valid": 0,
        "mean_v_sign_reversal_unknown": 0,
        "speed_increase": 0,
        "speed_decrease": 0,
        "speed_unchanged": 0,
        "speed_change_valid": 0,
        "speed_change_unknown": 0,
        "coherence_increase": 0,
        "coherence_decrease": 0,
        "coherence_unchanged": 0,
        "coherence_change_valid": 0,
        "coherence_change_unknown": 0,
        "vorticity_magnitude_increase": 0,
        "vorticity_magnitude_decrease": 0,
        "vorticity_magnitude_unchanged": 0,
        "vorticity_magnitude_change_valid": 0,
        "vorticity_magnitude_change_unknown": 0,
        "divergence_sign_change": 0,
        "divergence_sign_change_valid": 0,
        "divergence_sign_change_unknown": 0,
    }


def update_counts(
    counts,
    description,
):
    if not description[
        "comparison_available"
    ]:
        counts[
            "comparison_unavailable"
        ] += 1
        return

    counts[
        "comparison_available"
    ] += 1

    for metric in (
        "mean_u_sign_reversal",
        "mean_v_sign_reversal",
        "divergence_sign_change",
    ):
        value = description[metric]

        if value is None:
            counts[
                f"{metric}_unknown"
            ] += 1
        else:
            counts[
                f"{metric}_valid"
            ] += 1

            if value:
                counts[metric] += 1

    for metric, direction_key in (
        (
            "speed_change",
            "speed_change_direction",
        ),
        (
            "coherence_change",
            "coherence_change_direction",
        ),
        (
            "vorticity_magnitude_change",
            "vorticity_magnitude_change_direction",
        ),
    ):
        direction = description[
            direction_key
        ]

        if direction is None:
            counts[
                f"{metric}_unknown"
            ] += 1
            continue

        counts[
            f"{metric}_valid"
        ] += 1

        if metric == "speed_change":
            prefix = "speed"
        elif metric == "coherence_change":
            prefix = "coherence"
        else:
            prefix = (
                "vorticity_magnitude"
            )

        counts[
            f"{prefix}_{direction}"
        ] += 1


def fraction(
    numerator,
    denominator,
):
    if denominator == 0:
        return None

    return numerator / denominator


def behavior_fractions(
    counts,
):
    return {
        "mean_u_sign_reversal_fraction": fraction(
            counts["mean_u_sign_reversal"],
            counts[
                "mean_u_sign_reversal_valid"
            ],
        ),
        "mean_v_sign_reversal_fraction": fraction(
            counts["mean_v_sign_reversal"],
            counts[
                "mean_v_sign_reversal_valid"
            ],
        ),
        "speed_increase_fraction": fraction(
            counts["speed_increase"],
            counts["speed_change_valid"],
        ),
        "speed_decrease_fraction": fraction(
            counts["speed_decrease"],
            counts["speed_change_valid"],
        ),
        "coherence_increase_fraction": fraction(
            counts["coherence_increase"],
            counts[
                "coherence_change_valid"
            ],
        ),
        "coherence_decrease_fraction": fraction(
            counts["coherence_decrease"],
            counts[
                "coherence_change_valid"
            ],
        ),
        "vorticity_magnitude_increase_fraction": fraction(
            counts[
                "vorticity_magnitude_increase"
            ],
            counts[
                "vorticity_magnitude_change_valid"
            ],
        ),
        "vorticity_magnitude_decrease_fraction": fraction(
            counts[
                "vorticity_magnitude_decrease"
            ],
            counts[
                "vorticity_magnitude_change_valid"
            ],
        ),
        "divergence_sign_change_fraction": fraction(
            counts[
                "divergence_sign_change"
            ],
            counts[
                "divergence_sign_change_valid"
            ],
        ),
    }


def describe_group(
    group,
    descriptor_lookup,
):
    descriptions = []
    counts = empty_counts()

    missing_transition_ids = []

    for transition_id in group[
        "transition_ids"
    ]:
        transition = (
            descriptor_lookup.get(
                transition_id
            )
        )

        if transition is None:
            missing_transition_ids.append(
                transition_id
            )
            continue

        description = (
            describe_transition(
                transition
            )
        )

        descriptions.append(
            description
        )

        update_counts(
            counts,
            description,
        )

    return {
        "group_id": group[
            "group_id"
        ],
        "slice": group[
            "slice"
        ],
        "transition_count": group[
            "transition_count"
        ],
        "descriptor_transition_count": (
            len(descriptions)
        ),
        "missing_transition_count": (
            len(
                missing_transition_ids
            )
        ),
        "missing_transition_ids": (
            missing_transition_ids
        ),
        "behavior_counts": counts,
        "behavior_fractions": behavior_fractions(
            counts
        ),
        "transitions": descriptions,
    }


def main():
    if not GROUPS_INPUT.exists():
        raise SystemExit(
            f"Missing overlap group artifact: "
            f"{GROUPS_INPUT}"
        )

    if not DESCRIPTORS_INPUT.exists():
        raise SystemExit(
            f"Missing descriptor artifact: "
            f"{DESCRIPTORS_INPUT}"
        )

    groups_data = json.loads(
        GROUPS_INPUT.read_text(
            encoding="utf-8"
        )
    )

    descriptors_data = json.loads(
        DESCRIPTORS_INPUT.read_text(
            encoding="utf-8"
        )
    )

    descriptor_lookup = {
        transition[
            "transition_id"
        ]: transition
        for transition
        in descriptors_data[
            "failures"
        ]
    }

    groups = [
        describe_group(
            group,
            descriptor_lookup,
        )
        for group in groups_data[
            "groups"
        ]
    ]

    output = {
        "schema_version": 4,
        "experiment": "001",
        "source_artifacts": [
            GROUPS_INPUT.name,
            DESCRIPTORS_INPUT.name,
        ],
        "purpose": (
            "describe step-0 to step-1 "
            "field behaviour within overlap "
            "groups while preserving each "
            "transition separately"
            " and reporting metric-specific validity"
        ),
        "interpretation_rule": (
            "behaviour counts summarize "
            "descriptor changes within "
            "overlap groups. They do not "
            "identify cause, mechanism, "
            "physical episode, or Delta."
        ),
        "comparison_rule": (
            "compare descriptor frames with "
            "relative_measurement_index 0 "
            "and 1 only when their valid "
            "spatial support is identical"
        ),
        "group_count": len(
            groups
        ),
        "groups": groups,
    }

    output_text = (
        json.dumps(
            output,
            indent=2,
        )
        + "\n"
    )

    temporary_output = OUTPUT.with_name(
        OUTPUT.name + ".tmp"
    )

    temporary_output.write_text(
        output_text,
        encoding="utf-8",
    )

    os.replace(
        temporary_output,
        OUTPUT,
    )

    print(
        json.dumps(
            {
                "experiment": "001",
                "source_groups": (
                    GROUPS_INPUT.name
                ),
                "source_descriptors": (
                    DESCRIPTORS_INPUT.name
                ),
                "output": OUTPUT.name,
                "group_count": len(
                    groups
                ),
                "missing_transition_count": sum(
                    group[
                        "missing_transition_count"
                    ]
                    for group in groups
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()