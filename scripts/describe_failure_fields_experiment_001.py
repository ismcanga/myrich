import json
from math import cos, radians, sqrt
from pathlib import Path


ROOT = Path("artifacts/experiment-001")

INPUT = ROOT / "failure_localization.json"
OUTPUT = ROOT / "failure_field_descriptions_v2.json"

METRES_PER_DEGREE_LATITUDE = 111_320.0


def mean(values):
    if not values:
        return None

    return sum(values) / len(values)


def population_std(values):
    if not values:
        return None

    avg = mean(values)

    variance = sum(
        (value - avg) ** 2
        for value in values
    ) / len(values)

    return sqrt(variance)


def magnitude(u, v):
    return sqrt(u * u + v * v)


def vector_coherence(vectors):
    """
    Magnitude of the mean vector divided by
    mean individual vector magnitude.

    Near 1 means the vectors are strongly aligned.
    Near 0 means they substantially cancel.

    Undefined for an all-zero field.
    """

    if not vectors:
        return None

    mean_u = mean([u for u, _ in vectors])
    mean_v = mean([v for _, v in vectors])

    mean_vector_speed = magnitude(
        mean_u,
        mean_v,
    )

    mean_individual_speed = mean(
        [
            magnitude(u, v)
            for u, v in vectors
        ]
    )

    if mean_individual_speed == 0:
        return None

    return (
        mean_vector_speed
        / mean_individual_speed
    )


def frame_grids(frame):
    """
    Reconstruct the localized field without
    modifying measurement values.

    Expected interior support is 3x3, but boundary
    truncation or missing measurements remain visible.
    """

    cells = frame["cells"]

    latitudes = sorted(
        {
            cell["latitude"]
            for cell in cells
        }
    )

    longitudes = sorted(
        {
            cell["longitude"]
            for cell in cells
        }
    )

    cell_by_position = {
        (
            cell["latitude"],
            cell["longitude"],
        ): cell
        for cell in cells
    }

    u_grid = []
    v_grid = []
    valid_refs = []

    for latitude in latitudes:
        u_row = []
        v_row = []

        for longitude in longitudes:
            cell = cell_by_position.get(
                (latitude, longitude)
            )

            if (
                cell is None
                or not cell.get(
                    "valid",
                    False,
                )
            ):
                u_row.append(None)
                v_row.append(None)
                continue

            u_row.append(
                float(cell["ugos_mps"])
            )
            v_row.append(
                float(cell["vgos_mps"])
            )

            valid_refs.append(
                {
                    "latitude": latitude,
                    "longitude": longitude,
                }
            )

        u_grid.append(u_row)
        v_grid.append(v_row)

    return {
        "latitudes": latitudes,
        "longitudes": longitudes,
        "u_grid": u_grid,
        "v_grid": v_grid,
        "valid_refs": valid_refs,
        "boundary_truncated": (
            len(latitudes) != 3
            or len(longitudes) != 3
        ),
    }


def flatten_valid(u_grid, v_grid):
    vectors = []

    for row_index in range(len(u_grid)):
        for column_index in range(
            len(u_grid[row_index])
        ):
            u = u_grid[
                row_index
            ][column_index]

            v = v_grid[
                row_index
            ][column_index]

            if u is None or v is None:
                continue

            vectors.append((u, v))

    return vectors


def local_spacing_metres(
    latitudes,
    longitudes,
):
    """
    Local Cartesian approximation.

    dy uses latitude separation directly.

    dx scales longitude separation by cos(latitude).
    """

    if (
        len(latitudes) != 3
        or len(longitudes) != 3
    ):
        return None, None

    center_latitude = latitudes[1]

    dy_metres = abs(
        (
            latitudes[2]
            - latitudes[0]
        )
        / 2.0
        * METRES_PER_DEGREE_LATITUDE
    )

    metres_per_degree_longitude = (
        METRES_PER_DEGREE_LATITUDE
        * cos(
            radians(
                center_latitude
            )
        )
    )

    dx_metres = abs(
        (
            longitudes[2]
            - longitudes[0]
        )
        / 2.0
        * metres_per_degree_longitude
    )

    return dx_metres, dy_metres


def center_difference(
    left,
    right,
    spacing,
):
    if (
        left is None
        or right is None
        or not spacing
    ):
        return None

    return (
        right - left
    ) / (
        2.0 * spacing
    )


def local_derivatives(
    u_grid,
    v_grid,
    latitudes,
    longitudes,
):
    """
    Centered finite-difference approximation
    at the centre of a complete 3x3 field.

    divergence:
        du/dx + dv/dy

    vertical vorticity:
        dv/dx - du/dy
    """

    if (
        len(u_grid) != 3
        or len(v_grid) != 3
    ):
        return {
            "divergence_per_s": None,
            "vorticity_per_s": None,
            "stencil_complete": False,
        }

    if any(
        len(row) != 3
        for row in (
            u_grid + v_grid
        )
    ):
        return {
            "divergence_per_s": None,
            "vorticity_per_s": None,
            "stencil_complete": False,
        }

    dx_metres, dy_metres = (
        local_spacing_metres(
            latitudes,
            longitudes,
        )
    )

    if (
        not dx_metres
        or not dy_metres
    ):
        return {
            "divergence_per_s": None,
            "vorticity_per_s": None,
            "stencil_complete": False,
        }

    west_u = u_grid[1][0]
    east_u = u_grid[1][2]

    south_v = v_grid[0][1]
    north_v = v_grid[2][1]

    west_v = v_grid[1][0]
    east_v = v_grid[1][2]

    south_u = u_grid[0][1]
    north_u = u_grid[2][1]

    stencil_values = (
        west_u,
        east_u,
        south_v,
        north_v,
        west_v,
        east_v,
        south_u,
        north_u,
    )

    stencil_complete = all(
        value is not None
        for value in stencil_values
    )

    if not stencil_complete:
        return {
            "divergence_per_s": None,
            "vorticity_per_s": None,
            "stencil_complete": False,
        }

    du_dx = center_difference(
        west_u,
        east_u,
        dx_metres,
    )

    dv_dy = center_difference(
        south_v,
        north_v,
        dy_metres,
    )

    dv_dx = center_difference(
        west_v,
        east_v,
        dx_metres,
    )

    du_dy = center_difference(
        south_u,
        north_u,
        dy_metres,
    )

    return {
        "divergence_per_s": (
            du_dx + dv_dy
        ),
        "vorticity_per_s": (
            dv_dx - du_dy
        ),
        "stencil_complete": True,
    }


def describe_frame(
    frame,
    start_timestamp,
):
    grids = frame_grids(frame)

    u_grid = grids["u_grid"]
    v_grid = grids["v_grid"]

    vectors = flatten_valid(
        u_grid,
        v_grid,
    )

    speeds = [
        magnitude(u, v)
        for u, v in vectors
    ]

    mean_u = mean(
        [u for u, _ in vectors]
    )

    mean_v = mean(
        [v for _, v in vectors]
    )

    derivatives = local_derivatives(
        u_grid,
        v_grid,
        grids["latitudes"],
        grids["longitudes"],
    )

    frame_time = (
        frame["timestamp"]
    )

    from datetime import datetime

    start_dt = datetime.fromisoformat(
        start_timestamp.replace(
            "Z",
            "+00:00",
        )
    )

    frame_dt = datetime.fromisoformat(
        frame_time.replace(
            "Z",
            "+00:00",
        )
    )

    offset_seconds = int(
        (
            frame_dt - start_dt
        ).total_seconds()
    )

    return {
        "timestamp": frame_time,
        "relative_measurement_index": (
            frame[
                "relative_measurement_index"
            ]
        ),
        "offset_seconds": offset_seconds,
        "support": {
            "available_cell_count": len(
                frame["cells"]
            ),
            "valid_cell_count": len(
                vectors
            ),
            "includes_center": any(
                cell.get(
                    "center",
                    False,
                )
                for cell in frame["cells"]
            ),
            "boundary_truncated": grids[
                "boundary_truncated"
            ],
            "derivative_stencil_complete": (
                derivatives[
                    "stencil_complete"
                ]
            ),
        },
        "mean_u_mps": mean_u,
        "mean_v_mps": mean_v,
        "mean_speed_mps": mean(
            speeds
        ),
        "speed_spread_mps": (
            population_std(
                speeds
            )
        ),
        "vector_coherence": (
            vector_coherence(
                vectors
            )
        ),
        "divergence_per_s": (
            derivatives[
                "divergence_per_s"
            ]
        ),
        "vorticity_per_s": (
            derivatives[
                "vorticity_per_s"
            ]
        ),
    }


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


def describe_failure(
    slice_name,
    rank,
    failure,
):
    frames = [
        describe_frame(
            frame,
            failure["start"],
        )
        for frame in failure[
            "measurement_window"
        ]["frames"]
    ]

    return {
        "transition_id": transition_id(
            slice_name,
            failure,
        ),
        "slice": slice_name,
        "selection_rank": rank,
        "time_index": (
            failure["time_index"]
        ),
        "lat_index": (
            failure["lat_index"]
        ),
        "lon_index": (
            failure["lon_index"]
        ),
        "start": failure["start"],
        "end": failure["end"],
        "roles": {
            "A": (
                "focal_initial_state"
            ),
            "B": (
                "evolution_interval"
            ),
            "C": (
                "focal_resulting_state"
            ),
            "D": (
                "surrounding_measured_field"
            ),
        },
        "calculation": (
            "persistence_v1"
        ),
        "physical_mechanism_status": (
            "unresolved"
        ),
        "frames": frames,
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

    failures = []

    for (
        slice_name,
        slice_data,
    ) in data["slices"].items():

        for rank, failure in enumerate(
            slice_data["failures"],
            start=1,
        ):
            failures.append(
                describe_failure(
                    slice_name,
                    rank,
                    failure,
                )
            )

    output = {
        "schema_version": 2,
        "experiment": "001",
        "source_artifact": (
            INPUT.name
        ),
        "purpose": (
            "describe the local measured "
            "vector field around selected "
            "failed calculations"
        ),
        "interpretation_rule": (
            "these descriptors summarize "
            "measured field behaviour. "
            "They do not identify cause "
            "or Delta."
        ),
        "four_unit_structure": {
            "A": (
                "focal initial state"
            ),
            "B": (
                "evolution over the "
                "specified interval"
            ),
            "C": (
                "focal resulting state"
            ),
            "D": (
                "surrounding measured "
                "field and history"
            ),
        },
        "assumptions": {
            "geometry": (
                "local Cartesian "
                "approximation"
            ),
            "metres_per_degree_latitude": (
                METRES_PER_DEGREE_LATITUDE
            ),
            "longitude_spacing": (
                "scaled by cos(center latitude)"
            ),
            "derivative_method": (
                "centered finite difference "
                "over a complete 3x3 field"
            ),
            "rounding": "none",
        },
        "descriptors": {
            "mean_u_mps": (
                "mean eastward measured velocity"
            ),
            "mean_v_mps": (
                "mean northward measured velocity"
            ),
            "mean_speed_mps": (
                "mean speed across valid cells"
            ),
            "speed_spread_mps": (
                "population standard deviation "
                "of cell speeds"
            ),
            "vector_coherence": (
                "magnitude of mean vector divided "
                "by mean individual vector magnitude"
            ),
            "divergence_per_s": (
                "local Cartesian centered estimate "
                "of du/dx + dv/dy"
            ),
            "vorticity_per_s": (
                "local Cartesian centered estimate "
                "of dv/dx - du/dy"
            ),
        },
        "failures": failures,
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
                "schema_version": 2,
                "source": INPUT.name,
                "output": OUTPUT.name,
                "failure_count": len(
                    failures
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()