import json
from pathlib import Path

import math

ROOT = Path("artifacts/experiment-001")

INPUT = ROOT / "failure_localization.json"
OUTPUT = ROOT / "failure_overlap_inventory.json"


def transition_id(
    slice_name,
    failure,
):
    return (
        f"{slice_name}:"
        f"{failure['time_index']}:"
        f"{failure['lat_index']}:"
        f"{failure['lon_index']}:"
        f"{failure['start']}:"
        f"{failure['end']}"
    )


def measurement_ref(
    slice_name,
    timestamp,
    cell,
):
    return (
        slice_name,
        timestamp,
        float(cell["latitude"]),
        float(cell["longitude"]),
    )


def window_refs(
    slice_name,
    failure,
):
    refs = set()

    for frame in failure[
        "measurement_window"
    ]["frames"]:
        timestamp = frame["timestamp"]

        for cell in frame["cells"]:
            refs.add(
                measurement_ref(
                    slice_name,
                    timestamp,
                    cell,
                )
            )

    return refs

def valid_measurement_cell(cell):
    if cell.get("valid") is not True:
        return False

    ugos = cell.get("ugos_mps")
    vgos = cell.get("vgos_mps")

    return (
        isinstance(ugos, (int, float))
        and not isinstance(ugos, bool)
        and math.isfinite(ugos)
        and isinstance(vgos, (int, float))
        and not isinstance(vgos, bool)
        and math.isfinite(vgos)
    )


def valid_measurement_refs(
    slice_name,
    failure,
):
    refs = set()

    for frame in failure[
        "measurement_window"
    ]["frames"]:
        timestamp = frame["timestamp"]

        for cell in frame["cells"]:
            if not valid_measurement_cell(
                cell
            ):
                continue

            refs.add(
                measurement_ref(
                    slice_name,
                    timestamp,
                    cell,
                )
            )

    return refs


def focal_identity(
    slice_name,
    failure,
):
    return (
        slice_name,
        failure["lat_index"],
        failure["lon_index"],
    )


def build_transition(
    slice_name,
    selection_rank,
    failure,
):
    return {
        "transition_id": transition_id(
            slice_name,
            failure,
        ),
        "slice": slice_name,
        "selection_rank": selection_rank,
        "time_index": failure[
            "time_index"
        ],
        "lat_index": failure[
            "lat_index"
        ],
        "lon_index": failure[
            "lon_index"
        ],
        "start": failure["start"],
        "end": failure["end"],
        "window_refs": window_refs(
            slice_name,
            failure,
        ),
        "valid_measurement_refs": (
            valid_measurement_refs(
                slice_name,
                failure,
            )
        ),
        "focal_identity": focal_identity(
            slice_name,
            failure,
        ),
    }


def shared_timestamps(
    shared_refs,
):
    return sorted(
        {
            ref[1]
            for ref in shared_refs
        }
    )


def same_focal_cell(
    transition_a,
    transition_b,
):
    return (
        transition_a[
            "focal_identity"
        ]
        == transition_b[
            "focal_identity"
        ]
    )


def consecutive_focal_transitions(
    transition_a,
    transition_b,
):
    if not same_focal_cell(
        transition_a,
        transition_b,
    ):
        return False

    return (
        transition_a["end"]
        == transition_b["start"]
        or transition_b["end"]
        == transition_a["start"]
    )


def build_edges(transitions):
    edges = []

    for index_a in range(
        len(transitions)
    ):
        transition_a = transitions[
            index_a
        ]

        for index_b in range(
            index_a + 1,
            len(transitions),
        ):
            transition_b = transitions[
                index_b
            ]

            # Different NOAA slices cannot share
            # one localized measurement identity.
            if (
                transition_a["slice"]
                != transition_b["slice"]
            ):
                continue

            shared_window_refs = (
                transition_a[
                    "window_refs"
                ]
                & transition_b[
                    "window_refs"
                ]
            )

            shared_valid_refs = (
                transition_a[
                    "valid_measurement_refs"
                ]
                & transition_b[
                    "valid_measurement_refs"
                ]
            )

            if not shared_window_refs:
                continue

            edges.append(
                {
                    "transition_a": (
                        transition_a[
                            "transition_id"
                        ]
                    ),
                    "transition_b": (
                        transition_b[
                            "transition_id"
                        ]
                    ),
                    "shared_window_cell_count": (
                        len(shared_window_refs)
                    ),
                    "shared_valid_measurement_count": (
                        len(shared_valid_refs)
                    ),
                    "shared_window_timestamps": (
                        shared_timestamps(
                            shared_window_refs
                        )
                    ),
                    "shared_valid_measurement_timestamps": (
                        shared_timestamps(
                            shared_valid_refs
                        )
                    ),
                    "same_focal_cell": (
                        same_focal_cell(
                            transition_a,
                            transition_b,
                        )
                    ),
                    "consecutive_focal_transitions": (
                        consecutive_focal_transitions(
                            transition_a,
                            transition_b,
                        )
                    ),
                }
            )

    return edges


def build_overlap_summary(
    transitions,
    edges,
):
    summary = {
        transition[
            "transition_id"
        ]: {
            "direct_overlap_count": 0,
            "same_focal_cell_overlap_count": 0,
            "consecutive_focal_overlap_count": 0,
            "maximum_shared_window_cell_count": 0,
            "maximum_shared_valid_measurement_count": 0,
            "directly_overlapping_transition_ids": [],
        }
        for transition in transitions
    }

    localized_window_count_by_id = {
        transition["transition_id"]: len(
            transition["window_refs"]
        )
        for transition in transitions
    }

    localized_valid_count_by_id = {
        transition["transition_id"]: len(
            transition[
                "valid_measurement_refs"
            ]
        )
        for transition in transitions
    }

    for edge in edges:
        transition_a = edge[
            "transition_a"
        ]
        transition_b = edge[
            "transition_b"
        ]

        for current_id, other_id in (
            (
                transition_a,
                transition_b,
            ),
            (
                transition_b,
                transition_a,
            ),
        ):
            item = summary[current_id]

            item[
                "direct_overlap_count"
            ] += 1

            if edge[
                "same_focal_cell"
            ]:
                item[
                    "same_focal_cell_overlap_count"
                ] += 1

            if edge[
                "consecutive_focal_transitions"
            ]:
                item[
                    "consecutive_focal_overlap_count"
                ] += 1

            item[
                "maximum_shared_window_cell_count"
            ] = max(
                item[
                    "maximum_shared_window_cell_count"
                ],
                edge[
                    "shared_window_cell_count"
                ],
            )

            item[
                "maximum_shared_valid_measurement_count"
            ] = max(
                item[
                    "maximum_shared_valid_measurement_count"
                ],
                edge[
                    "shared_valid_measurement_count"
                ],
            )

            item[
                "directly_overlapping_transition_ids"
            ].append(
                other_id
            )

    for transition_id, item in summary.items():
        localized_window_count = (
            localized_window_count_by_id[
                transition_id
            ]
        )
        localized_valid_count = (
            localized_valid_count_by_id[
                transition_id
            ]
        )

        item["window_overlap_fraction_max"] = (
            item[
                "maximum_shared_window_cell_count"
            ]
            / localized_window_count
            if localized_window_count
            else None
        )

        item[
            "valid_measurement_overlap_fraction_max"
        ] = (
            item[
                "maximum_shared_valid_measurement_count"
            ]
            / localized_valid_count
            if localized_valid_count
            else None
        )

        item[
            "directly_overlapping_transition_ids"
        ].sort()

    return summary


def transition_summary(
    transition,
    overlap_summary,
):
    return {
        "transition_id": transition[
            "transition_id"
        ],
        "slice": transition["slice"],
        "selection_rank": transition[
            "selection_rank"
        ],
        "time_index": transition[
            "time_index"
        ],
        "lat_index": transition[
            "lat_index"
        ],
        "lon_index": transition[
            "lon_index"
        ],
        "start": transition["start"],
        "end": transition["end"],
        "localized_window_cell_count": len(
            transition["window_refs"]
        ),
        "localized_valid_measurement_count": len(
            transition[
                "valid_measurement_refs"
            ]
        ),
        "overlap": overlap_summary,
    }


def main():
    if not INPUT.exists():
        raise SystemExit(
            f"Missing input artifact: "
            f"{INPUT}"
        )

    data = json.loads(
        INPUT.read_text(
            encoding="utf-8"
        )
    )

    transitions = []

    for (
        slice_name,
        slice_data,
    ) in data["slices"].items():

        for rank, failure in enumerate(
            slice_data["failures"],
            start=1,
        ):
            transitions.append(
                build_transition(
                    slice_name,
                    rank,
                    failure,
                )
            )

    edges = build_edges(
        transitions
    )

    overlap_summary_by_id = (
        build_overlap_summary(
            transitions,
            edges,
        )
    )

    output = {
        "schema_version": 2,
        "experiment": "001",
        "source_artifact": INPUT.name,
        "purpose": (
            "inventory shared localized window "
            "coverage and shared valid measurements "
            "between selected failure transitions"
        ),
        "interpretation_rule": (
            "shared window coverage establishes "
            "geometric-temporal overlap only. "
            "Shared valid measurements establish "
            "shared measured evidence. Missing or "
            "invalid cells never count as measured "
            "overlap. Neither establishes a common "
            "physical cause or one physical episode."
        ),
        "window_cell_identity": (
            "slice + timestamp + latitude "
            "+ longitude"
        ),
        "valid_measurement_identity": (
            "slice + timestamp + latitude "
            "+ longitude, requiring valid=true and "
            "finite numeric non-boolean ugos_mps "
            "and vgos_mps"
        ),
        "transition_count": len(
            transitions
        ),
        "overlap_edge_count": len(
            edges
        ),
        "transitions": [
            transition_summary(
                transition,
                overlap_summary_by_id[
                    transition[
                        "transition_id"
                    ]
                ],
            )
            for transition in transitions
        ],
        "edges": edges,
    }

    OUTPUT.write_text(
        json.dumps(
            output,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "experiment": "001",
                "source": INPUT.name,
                "output": OUTPUT.name,
                "transition_count": len(
                    transitions
                ),
                "overlap_edge_count": len(
                    edges
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()