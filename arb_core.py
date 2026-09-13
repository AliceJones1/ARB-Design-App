"""Engineering calculations for a blade-adjustable anti-roll bar.

Internal units are SI. Rotational stiffness is exposed as N*m/degree because
that is the convention used by the ARG26 report and the vehicle targets.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

DEG_PER_RAD = 180.0 / math.pi
NM_PER_FTLB = 1.3558179483
M_PER_IN = 0.0254


def ft_lb_per_deg_to_nm_per_deg(value: float) -> float:
    return value * NM_PER_FTLB


def nm_per_deg_to_ft_lb_per_deg(value: float) -> float:
    return value / NM_PER_FTLB


def roll_to_twist_stiffness(
    k_roll: float,
    motion_ratio_roll_per_twist: float,
) -> float:
    """Convert roll-referenced rate to twist-referenced rate using energy."""
    if motion_ratio_roll_per_twist <= 0:
        raise ValueError("Motion ratio must be positive")

    return k_roll * motion_ratio_roll_per_twist**2


def chassis_roll_deg(
    roll_gradient_deg_per_g: float,
    lateral_accel_g: float,
) -> float:
    return roll_gradient_deg_per_g * lateral_accel_g


def arb_twist_deg(
    chassis_roll: float,
    motion_ratio_roll_per_twist: float,
) -> float:
    if motion_ratio_roll_per_twist <= 0:
        raise ValueError("Motion ratio must be positive")

    return chassis_roll / motion_ratio_roll_per_twist


def polar_moment_hollow(od_m: float, id_m: float) -> float:
    if od_m <= 0 or id_m < 0 or id_m >= od_m:
        raise ValueError("Require 0 <= ID < OD")

    return math.pi * (od_m**4 - id_m**4) / 32.0


def tube_stiffness_nm_per_deg(
    od_m: float,
    id_m: float,
    length_m: float,
    shear_modulus_pa: float,
) -> float:
    if length_m <= 0 or shear_modulus_pa <= 0:
        raise ValueError("Length and shear modulus must be positive")

    k_nm_per_rad = (
        shear_modulus_pa
        * polar_moment_hollow(od_m, id_m)
        / length_m
    )

    return k_nm_per_rad / DEG_PER_RAD


def tube_twist_deg(
    torque_nm: float,
    tube_k_nm_per_deg: float,
) -> float:
    if tube_k_nm_per_deg <= 0:
        raise ValueError("Tube stiffness must be positive")

    return torque_nm / tube_k_nm_per_deg


def tube_shear_stress_pa(
    torque_nm: float,
    od_m: float,
    id_m: float,
) -> float:
    return (
        abs(torque_nm)
        * (od_m / 2.0)
        / polar_moment_hollow(od_m, id_m)
    )


def required_blade_pair_stiffness(
    total_k: float,
    tube_k: float,
) -> float:
    """Equivalent stiffness of the two-blade compliance in N*m/degree."""
    if total_k <= 0 or tube_k <= total_k:
        raise ValueError(
            "Tube stiffness must exceed required total stiffness"
        )

    return total_k * tube_k / (tube_k - total_k)


def series_stiffness(*stiffnesses: float) -> float:
    if any(k <= 0 for k in stiffnesses):
        raise ValueError("All stiffnesses must be positive")

    return 1.0 / sum(1.0 / k for k in stiffnesses)


def blade_pair_stiffness_from_single(
    single_blade_k: float,
) -> float:
    return single_blade_k / 2.0


def rotated_rectangular_inertia(
    width_m: np.ndarray | float,
    thickness_m: float,
    angle_deg: float,
):
    """Second moment about the active bending axis; 0 deg is weak-axis/soft."""
    angle_rad = math.radians(angle_deg)

    i_weak = np.asarray(width_m) * thickness_m**3 / 12.0
    i_strong = thickness_m * np.asarray(width_m) ** 3 / 12.0

    return (
        i_weak * math.cos(angle_rad) ** 2
        + i_strong * math.sin(angle_rad) ** 2
    )


@dataclass(frozen=True)
class BladeResult:
    linear_stiffness_n_per_m: float
    single_rot_stiffness_nm_per_deg: float
    pair_rot_stiffness_nm_per_deg: float
    tip_deflection_m: float
    max_stress_pa: float
    max_stress_x_m: float


def tapered_blade_analysis(
    force_n: float,
    length_m: float,
    root_width_m: float,
    tip_width_m: float,
    thickness_m: float,
    youngs_modulus_pa: float,
    angle_deg: float,
    stations: int = 801,
) -> BladeResult:
    """Euler-Bernoulli tapered rectangular blade under a tip point load."""
    if min(
        length_m,
        root_width_m,
        tip_width_m,
        thickness_m,
        youngs_modulus_pa,
    ) <= 0:
        raise ValueError(
            "Blade dimensions and modulus must be positive"
        )

    x = np.linspace(
        0.0,
        length_m,
        max(101, int(stations)),
    )

    width = (
        root_width_m
        + (tip_width_m - root_width_m) * x / length_m
    )

    inertia = rotated_rectangular_inertia(
        width,
        thickness_m,
        angle_deg,
    )

    compliance = np.trapezoid(
        (length_m - x) ** 2
        / (youngs_modulus_pa * inertia),
        x,
    )

    k_linear = 1.0 / compliance
    delta = force_n / k_linear

    single_k_rad = k_linear * length_m**2
    single_k_deg = single_k_rad / DEG_PER_RAD

    moment = abs(force_n) * (length_m - x)
    angle_rad = math.radians(angle_deg)

    c_eff = 0.5 * (
        np.abs(width * math.sin(angle_rad))
        + abs(thickness_m * math.cos(angle_rad))
    )

    stress = moment * c_eff / inertia
    idx = int(np.nanargmax(stress))

    return BladeResult(
        linear_stiffness_n_per_m=k_linear,
        single_rot_stiffness_nm_per_deg=single_k_deg,
        pair_rot_stiffness_nm_per_deg=single_k_deg / 2.0,
        tip_deflection_m=delta,
        max_stress_pa=float(stress[idx]),
        max_stress_x_m=float(x[idx]),
    )


def blade_tip_force(
    torque_nm: float,
    effective_arm_m: float,
) -> float:
    if effective_arm_m <= 0:
        raise ValueError("Effective arm must be positive")

    return abs(torque_nm) / effective_arm_m


def drop_link_checks(
    force_n: float,
    length_m: float,
    diameter_m: float,
    youngs_modulus_pa: float,
    yield_pa: float,
):
    if min(
        length_m,
        diameter_m,
        youngs_modulus_pa,
        yield_pa,
    ) <= 0:
        raise ValueError("Drop-link inputs must be positive")

    area = math.pi * diameter_m**2 / 4.0
    inertia = math.pi * diameter_m**4 / 64.0

    stress = abs(force_n) / area

    p_cr_pinned = (
        math.pi**2
        * youngs_modulus_pa
        * inertia
        / length_m**2
    )

    return {
        "axial_stress_pa": stress,
        "yield_fos": yield_pa / stress if stress else math.inf,
        "euler_buckling_n": p_cr_pinned,
        "buckling_fos": (
            p_cr_pinned / abs(force_n)
            if force_n
            else math.inf
        ),
    }


def point_distance(p1, p2) -> float:
    a = np.asarray(p1, dtype=float)
    b = np.asarray(p2, dtype=float)

    if a.shape != (3,) or b.shape != (3,):
        raise ValueError(
            "Points must have x, y, z coordinates"
        )

    return float(np.linalg.norm(b - a))


def linear_regression(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 3:
        raise ValueError(
            "At least three finite samples are required"
        )

    slope, intercept = np.polyfit(
        x[mask],
        y[mask],
        1,
    )

    pred = slope * x[mask] + intercept

    ss_res = float(
        np.sum((y[mask] - pred) ** 2)
    )

    ss_tot = float(
        np.sum(
            (y[mask] - np.mean(y[mask])) ** 2
        )
    )

    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot
        else float("nan")
    )

    return float(slope), float(intercept), r2
