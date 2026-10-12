import json
from pathlib import Path


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


def measurement_refs(
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
        "measurement_refs": measurement_refs(
            slice_name,
            failure,
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

            shared_refs = (
                transition_a[
                    "measurement_refs"
                ]
                & transition_b[
                    "measurement_refs"
                ]
            )

            if not shared_refs:
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
                    "shared_measurement_count": (
                        len(shared_refs)
                    ),
                    "shared_timestamps": (
                        shared_timestamps(
                            shared_refs
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
            "maximum_shared_measurement_count": 0,
            "directly_overlapping_transition_ids": [],
        }
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
                "maximum_shared_measurement_count"
            ] = max(
                item[
                    "maximum_shared_measurement_count"
                ],
                edge[
                    "shared_measurement_count"
                ],
            )

            item[
                "directly_overlapping_transition_ids"
            ].append(
                other_id
            )

    for item in summary.values():
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
        "localized_measurement_count": len(
            transition[
                "measurement_refs"
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
        "schema_version": 1,
        "experiment": "001",
        "source_artifact": INPUT.name,
        "purpose": (
            "inventory direct overlap "
            "between selected localized "
            "failure windows"
        ),
        "interpretation_rule": (
            "shared measurements establish "
            "shared evidence only. They do "
            "not establish a common physical "
            "cause or one physical episode."
        ),
        "measurement_identity": (
            "slice + timestamp + latitude "
            "+ longitude"
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