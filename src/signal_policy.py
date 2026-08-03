from __future__ import annotations

import re
from collections.abc import Iterable


META_COLUMNS = {
    "time",
    "sample",
    "anomaly",
    "category",
    "setting",
    "action",
    "active",
}

MEASURED_ROBOT_COLUMNS = {
    "robot_voltage",
    "robot_current",
    "io_current",
    "system_current",
}

MEASURED_AXIS_PATTERNS = (
    r"motor_position_\d+",
    r"motor_velocity_\d+",
    r"joint_position_\d+",
    r"joint_velocity_\d+",
    r"motor_torque_\d+",
    r"torque_sensor_a_\d+",
    r"torque_sensor_b_\d+",
    r"motor_iq_\d+",
    r"motor_id_\d+",
    r"power_motor_el_\d+",
    r"power_motor_mech_\d+",
    r"power_load_mech_\d+",
    r"motor_voltage_\d+",
    r"supply_voltage_\d+",
    r"brake_voltage_\d+",
)


def is_measured_signal(column: str) -> bool:
    """Return True when a column is a measured robot signal."""
    if column in META_COLUMNS:
        return False

    if column in MEASURED_ROBOT_COLUMNS:
        return True

    return any(
        re.fullmatch(pattern, column)
        for pattern in MEASURED_AXIS_PATTERNS
    )


def select_measured_signals(
    columns: Iterable[str],
) -> list[str]:
    """Select measured-only signals for the primary experiment."""
    selected = [
        column
        for column in columns
        if is_measured_signal(column)
    ]

    if not selected:
        raise ValueError(
            "No measured signals were found in the dataset schema."
        )

    return selected


def select_machine_signals(
    columns: Iterable[str],
) -> list[str]:
    """Select all non-metadata machine signals."""
    selected = [
        column
        for column in columns
        if column not in META_COLUMNS
    ]

    if not selected:
        raise ValueError(
            "No machine signals were found in the dataset schema."
        )

    return selected