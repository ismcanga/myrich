import json
import hashlib
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path("artifacts/experiment-001")

INPUT = ROOT / "failure_overlap_inventory.json"
OUTPUT = ROOT / "failure_overlap_groups.json"

def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def membership_fingerprint(
    slice_name,
    component_ids,
):
    canonical_membership = {
        "slice": slice_name,
        "transition_ids": sorted(
            component_ids
        ),
    }

    encoded = json.dumps(
        canonical_membership,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return sha256_bytes(encoded)


def build_transition_lookup(data):
    return {
        item["transition_id"]: item
        for item in data["transitions"]
    }


def build_adjacency(data):
    adjacency = defaultdict(set)

    for transition in data["transitions"]:
        adjacency[
            transition["transition_id"]
        ]

    for edge in data["edges"]:
        transition_a = edge[
            "transition_a"
        ]
        transition_b = edge[
            "transition_b"
        ]

        adjacency[
            transition_a
        ].add(
            transition_b
        )

        adjacency[
            transition_b
        ].add(
            transition_a
        )

    return adjacency


def connected_components(
    transition_ids,
    adjacency,
):
    visited = set()
    components = []

    for transition_id in sorted(
        transition_ids
    ):
        if transition_id in visited:
            continue

        queue = deque(
            [transition_id]
        )

        visited.add(
            transition_id
        )

        component = []

        while queue:
            current = queue.popleft()

            component.append(
                current
            )

            for neighbour in sorted(
                adjacency[current]
            ):
                if neighbour in visited:
                    continue

                visited.add(
                    neighbour
                )

                queue.append(
                    neighbour
                )

        components.append(
            sorted(component)
        )

    return components


def edge_count_for_component(
    component_ids,
    edges,
):
    component_set = set(
        component_ids
    )

    return sum(
        1
        for edge in edges
        if (
            edge["transition_a"]
            in component_set
            and edge["transition_b"]
            in component_set
        )
    )


def group_id(
    slice_name,
    group_number,
):
    return (
        f"{slice_name}-"
        f"{group_number:03d}"
    )


def summarize_group(
    slice_name,
    group_number,
    component_ids,
    transition_lookup,
    edges,
):
    transitions = [
        transition_lookup[
            transition_id
        ]
        for transition_id
        in component_ids
    ]

    starts = [
        transition["start"]
        for transition in transitions
    ]

    ends = [
        transition["end"]
        for transition in transitions
    ]

    lat_indices = [
        transition["lat_index"]
        for transition in transitions
    ]

    lon_indices = [
        transition["lon_index"]
        for transition in transitions
    ]

    selection_ranks = sorted(
        transition[
            "selection_rank"
        ]
        for transition in transitions
    )

    transition_count = len(
        transitions
    )

    possible_edge_count = (
        transition_count
        * (transition_count - 1)
        // 2
    )

    edge_count = (
        edge_count_for_component(
            component_ids,
            edges,
        )
    )

    edge_density = (
        edge_count
        / possible_edge_count
        if possible_edge_count
        else 0.0
    )

    return {
        "group_id": group_id(
            slice_name,
            group_number,
        ),
        "membership_fingerprint": (
            membership_fingerprint(
                slice_name,
                component_ids,
            )
        ),
        "slice": slice_name,
        "transition_count": transition_count,
        "edge_count": edge_count,
        "possible_edge_count": possible_edge_count,
        "edge_density": edge_density,
        "selection_ranks": (
            selection_ranks
        ),
        "start_min": min(
            starts
        ),
        "end_max": max(
            ends
        ),
        "lat_index_min": min(
            lat_indices
        ),
        "lat_index_max": max(
            lat_indices
        ),
        "lon_index_min": min(
            lon_indices
        ),
        "lon_index_max": max(
            lon_indices
        ),
        "transition_ids": sorted(
            component_ids
        ),
    }


def build_groups_for_slice(
    slice_name,
    transitions,
    edges,
):
    transition_ids = [
        transition[
            "transition_id"
        ]
        for transition in transitions
        if transition["slice"]
        == slice_name
    ]

    slice_edges = [
        edge
        for edge in edges
        if (
            edge["transition_a"]
            in transition_ids
            and edge["transition_b"]
            in transition_ids
        )
    ]

    adjacency = defaultdict(set)

    for transition_id in transition_ids:
        adjacency[
            transition_id
        ]

    for edge in slice_edges:
        adjacency[
            edge["transition_a"]
        ].add(
            edge["transition_b"]
        )

        adjacency[
            edge["transition_b"]
        ].add(
            edge["transition_a"]
        )

    components = connected_components(
        transition_ids,
        adjacency,
    )

    transition_lookup = {
        transition[
            "transition_id"
        ]: transition
        for transition in transitions
    }

    groups = []

    for group_number, component_ids in enumerate(
        components,
        start=1,
    ):
        groups.append(
            summarize_group(
                slice_name,
                group_number,
                component_ids,
                transition_lookup,
                slice_edges,
            )
        )

    return groups


def main():
    if not INPUT.exists():
        raise SystemExit(
            f"Missing input artifact: "
            f"{INPUT}"
        )

    input_bytes = INPUT.read_bytes()

    data = json.loads(
        input_bytes.decode("utf-8")
    )

    source_sha256 = sha256_bytes(
        input_bytes
    )

    transitions = data[
        "transitions"
    ]

    edges = data[
        "edges"
    ]

    groups = []

    for slice_name in sorted(
        {
            transition["slice"]
            for transition
            in transitions
        }
    ):
        groups.extend(
            build_groups_for_slice(
                slice_name,
                transitions,
                edges,
            )
        )

    output = {
        "schema_version": 2,
        "experiment": "001",
        "source_artifact": (
            INPUT.name
        ),
        "provenance": {
            "source_artifact": INPUT.name,
            "source_schema_version": data.get(
                "schema_version"
            ),
            "source_sha256": source_sha256,
        },
        "purpose": (
            "organize selected transitions "
            "into connected overlap groups "
            "for navigation"
        ),
        "interpretation_rule": (
            "an overlap group is a connected "
            "component of transitions sharing "
            "localized window coverage. Shared "
            "window coverage is geometric-temporal "
            "overlap only and does not establish "
            "shared valid measurements, one physical "
            "event, episode, mechanism, or Delta."
        ),
        "grouping_rule": (
            "nodes are selected transitions; "
            "edges are direct shared-window-coverage "
            "overlaps from the inventory; connected "
            "components form overlap groups"
        ),
        "transition_count": len(
            transitions
        ),
        "edge_count": len(
            edges
        ),
        "group_count": len(
            groups
        ),
        "groups": groups,
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
                "edge_count": len(
                    edges
                ),
                "group_count": len(
                    groups
                ),
                "groups_by_slice": {
                    slice_name: sum(
                        1
                        for group in groups
                        if group["slice"]
                        == slice_name
                    )
                    for slice_name in sorted(
                        {
                            group["slice"]
                            for group in groups
                        }
                    )
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()