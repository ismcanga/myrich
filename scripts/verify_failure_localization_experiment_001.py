import hashlib
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, num2date


ROOT = Path("artifacts/experiment-001")

LOCALIZATION = ROOT / "failure_localization.json"
CHECKSUMS = ROOT / "SHA256SUMS.txt"
RECEIPT = ROOT / (
    "failure_localization_verification.json"
)

FILES = {
    "west": ROOT / "noaa_np_west_2025-01.nc",
    "east": ROOT / "noaa_np_east_2025-01.nc",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_expected_checksums():
    if not CHECKSUMS.exists():
        raise SystemExit(
            f"Missing checksum file: {CHECKSUMS}"
        )

    expected = {}

    for line in CHECKSUMS.read_text(
        encoding="utf-8"
    ).splitlines():
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 2:
            raise SystemExit(
                f"Invalid checksum line: {line}"
            )

        checksum, filename = parts
        expected[filename] = checksum

    return expected


def verify_checksums():
    expected = load_expected_checksums()

    verified = 0
    verified_files = {}

    for path in FILES.values():
        if not path.exists():
            raise SystemExit(
                f"Missing NetCDF file: {path}"
            )

        expected_hash = expected.get(
            path.name
        )

        if expected_hash is None:
            raise SystemExit(
                f"No checksum recorded for "
                f"{path.name}"
            )

        actual_hash = sha256_file(path)

        if actual_hash != expected_hash:
            raise SystemExit(
                f"Checksum mismatch for "
                f"{path.name}: "
                f"expected {expected_hash}, "
                f"got {actual_hash}"
            )

        verified_files[path.name] = {
            "expected_sha256": (
                expected_hash
            ),
            "actual_sha256": (
                actual_hash
            ),
        }

        verified += 1

    return verified, verified_files


def load_times(ds):
    time_var = ds.variables["time"]

    return list(
        num2date(
            time_var[:],
            units=time_var.units,
            calendar=getattr(
                time_var,
                "calendar",
                "standard",
            ),
        )
    )


def iso_time(value):
    return value.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def verify_number(
    actual,
    expected,
    label,
    tolerance=0.0,
):
    if tolerance == 0.0:
        if actual != expected:
            raise AssertionError(
                f"{label}: expected "
                f"{expected!r}, got "
                f"{actual!r}"
            )
        return

    if not np.isclose(
        actual,
        expected,
        rtol=0.0,
        atol=tolerance,
    ):
        raise AssertionError(
            f"{label}: expected "
            f"{expected!r}, got "
            f"{actual!r}"
        )


def verify_slice(
    slice_name,
    slice_data,
    dataset_path,
):
    frames_checked = 0
    cells_checked = 0
    coordinate_mismatches = 0
    time_mismatches = 0
    value_mismatches = 0

    with Dataset(dataset_path) as ds:
        times = load_times(ds)

        lats = np.asarray(
            ds.variables["latitude"][:],
            dtype=np.float64,
        )

        lons = np.asarray(
            ds.variables["longitude"][:],
            dtype=np.float64,
        )

        ugos = np.ma.array(
            ds.variables["ugos"][:]
        )

        vgos = np.ma.array(
            ds.variables["vgos"][:]
        )

        for failure in slice_data[
            "failures"
        ]:
            time_index = failure[
                "time_index"
            ]
            lat_index = failure[
                "lat_index"
            ]
            lon_index = failure[
                "lon_index"
            ]

            expected_start = iso_time(
                times[time_index]
            )
            expected_end = iso_time(
                times[time_index + 1]
            )

            if (
                failure["start"]
                != expected_start
            ):
                time_mismatches += 1
                raise AssertionError(
                    f"{slice_name} failure "
                    f"start mismatch at "
                    f"time_index={time_index}"
                )

            if (
                failure["end"]
                != expected_end
            ):
                time_mismatches += 1
                raise AssertionError(
                    f"{slice_name} failure "
                    f"end mismatch at "
                    f"time_index={time_index}"
                )

            try:
                verify_number(
                    failure["latitude"],
                    float(lats[lat_index]),
                    (
                        f"{slice_name} "
                        f"failure latitude"
                    ),
                )

                verify_number(
                    failure["longitude"],
                    float(lons[lon_index]),
                    (
                        f"{slice_name} "
                        f"failure longitude"
                    ),
                )
            except AssertionError:
                coordinate_mismatches += 1
                raise

            current_u = ugos[
                time_index,
                lat_index,
                lon_index,
            ]

            current_v = vgos[
                time_index,
                lat_index,
                lon_index,
            ]

            future_u = ugos[
                time_index + 1,
                lat_index,
                lon_index,
            ]

            future_v = vgos[
                time_index + 1,
                lat_index,
                lon_index,
            ]

            if (
                np.ma.is_masked(current_u)
                or np.ma.is_masked(current_v)
                or np.ma.is_masked(future_u)
                or np.ma.is_masked(future_v)
            ):
                raise AssertionError(
                    f"{slice_name} focal "
                    f"measurement unexpectedly "
                    f"masked"
                )

            try:
                verify_number(
                    failure[
                        "measurement_at_t"
                    ]["ugos_mps"],
                    float(current_u),
                    (
                        f"{slice_name} "
                        f"measurement_at_t "
                        f"ugos"
                    ),
                )

                verify_number(
                    failure[
                        "measurement_at_t"
                    ]["vgos_mps"],
                    float(current_v),
                    (
                        f"{slice_name} "
                        f"measurement_at_t "
                        f"vgos"
                    ),
                )

                verify_number(
                    failure[
                        "measurement_at_t_plus_24h"
                    ]["ugos_mps"],
                    float(future_u),
                    (
                        f"{slice_name} "
                        f"measurement_at_t_plus_24h "
                        f"ugos"
                    ),
                )

                verify_number(
                    failure[
                        "measurement_at_t_plus_24h"
                    ]["vgos_mps"],
                    float(future_v),
                    (
                        f"{slice_name} "
                        f"measurement_at_t_plus_24h "
                        f"vgos"
                    ),
                )
            except AssertionError:
                value_mismatches += 1
                raise

            for frame in failure[
                "measurement_window"
            ]["frames"]:
                frames_checked += 1

                timestamp = frame[
                    "timestamp"
                ]

                matching_indices = [
                    index
                    for index, value
                    in enumerate(times)
                    if iso_time(value)
                    == timestamp
                ]

                if len(
                    matching_indices
                ) != 1:
                    time_mismatches += 1
                    raise AssertionError(
                        f"{slice_name} frame "
                        f"timestamp not uniquely "
                        f"found: {timestamp}"
                    )

                frame_time_index = (
                    matching_indices[0]
                )

                for cell in frame[
                    "cells"
                ]:
                    cells_checked += 1

                    latitude = cell[
                        "latitude"
                    ]
                    longitude = cell[
                        "longitude"
                    ]

                    lat_matches = np.where(
                        lats == latitude
                    )[0]

                    lon_matches = np.where(
                        lons == longitude
                    )[0]

                    if (
                        len(lat_matches) != 1
                        or len(lon_matches)
                        != 1
                    ):
                        coordinate_mismatches += 1
                        raise AssertionError(
                            f"{slice_name} cell "
                            f"coordinate not found: "
                            f"{latitude}, "
                            f"{longitude}"
                        )

                    yi = int(
                        lat_matches[0]
                    )
                    xi = int(
                        lon_matches[0]
                    )

                    source_u = ugos[
                        frame_time_index,
                        yi,
                        xi,
                    ]

                    source_v = vgos[
                        frame_time_index,
                        yi,
                        xi,
                    ]

                    source_valid = not (
                        np.ma.is_masked(
                            source_u
                        )
                        or np.ma.is_masked(
                            source_v
                        )
                    )

                    artifact_valid = cell[
                        "valid"
                    ]

                    if (
                        source_valid
                        != artifact_valid
                    ):
                        value_mismatches += 1
                        raise AssertionError(
                            f"{slice_name} cell "
                            f"validity mismatch at "
                            f"{timestamp} "
                            f"{latitude}, "
                            f"{longitude}"
                        )

                    if not source_valid:
                        continue

                    try:
                        verify_number(
                            cell[
                                "ugos_mps"
                            ],
                            float(source_u),
                            (
                                f"{slice_name} "
                                f"cell ugos "
                                f"{timestamp} "
                                f"{latitude},"
                                f"{longitude}"
                            ),
                        )

                        verify_number(
                            cell[
                                "vgos_mps"
                            ],
                            float(source_v),
                            (
                                f"{slice_name} "
                                f"cell vgos "
                                f"{timestamp} "
                                f"{latitude},"
                                f"{longitude}"
                            ),
                        )
                    except AssertionError:
                        value_mismatches += 1
                        raise

    return {
        "frames_checked": (
            frames_checked
        ),
        "cells_checked": (
            cells_checked
        ),
        "coordinate_mismatches": (
            coordinate_mismatches
        ),
        "time_mismatches": (
            time_mismatches
        ),
        "value_mismatches": (
            value_mismatches
        ),
    }

def write_json_atomic(path, payload):
    temporary = path.with_name(
        path.name + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def main():
    if not LOCALIZATION.exists():
        raise SystemExit(
            f"Missing localization artifact: "
            f"{LOCALIZATION}"
        )

    checksum_count, verified_files = (
        verify_checksums()
    )

    localization_bytes = (
        LOCALIZATION.read_bytes()
    )

    localization = json.loads(
        localization_bytes.decode(
            "utf-8"
        )
    )

    localization_sha256 = hashlib.sha256(
        localization_bytes
    ).hexdigest()

    checksums_sha256 = sha256_file(
        CHECKSUMS
    )

    totals = {
        "frames_checked": 0,
        "cells_checked": 0,
        "coordinate_mismatches": 0,
        "time_mismatches": 0,
        "value_mismatches": 0,
    }

    for slice_name, path in (
        FILES.items()
    ):
        slice_data = localization[
            "slices"
        ][slice_name]

        result = verify_slice(
            slice_name,
            slice_data,
            path,
        )

        for key in totals:
            totals[key] += result[key]

    summary = {
        "schema_version": 1,
        "experiment": "001",
        "verification_target": (
            LOCALIZATION.name
        ),
        "provenance": {
            "localization_sha256": (
                localization_sha256
            ),
            "checksum_manifest": (
                CHECKSUMS.name
            ),
            "checksum_manifest_sha256": (
                checksums_sha256
            ),
            "verified_source_files": (
                verified_files
            ),
        },
        "source_checksums_verified": (
            checksum_count
        ),
        "localized_frames_checked": (
            totals["frames_checked"]
        ),
        "localized_cells_checked": (
            totals["cells_checked"]
        ),
        "coordinate_mismatches": (
            totals[
                "coordinate_mismatches"
            ]
        ),
        "time_mismatches": (
            totals["time_mismatches"]
        ),
        "value_mismatches": (
            totals["value_mismatches"]
        ),
        "status": "verified",
    }

    write_json_atomic(
        RECEIPT,
        summary,
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()