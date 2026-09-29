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
    """Convert ft·lbf/degree to N·m/degree."""
    return value * NM_PER_FTLB


def nm_per_deg_to_ft_lb_per_deg(value: float) -> float:
    """Convert N·m/degree to ft·lbf/degree."""
    return value / NM_PER_FTLB


def roll_to_twist_stiffness(
    k_roll: float,
    motion_ratio_roll_per_twist: float,
) -> float:
    """Convert roll-referenced stiffness to twist-referenced stiffness.

    The motion-ratio convention is:

        motion ratio = chassis roll / ARB twist

    Therefore:

        k_twist = k_roll * motion_ratio**2
    """
    if motion_ratio_roll_per_twist <= 0:
        raise ValueError("Motion ratio must be positive")

    return (
        k_roll
        * motion_ratio_roll_per_twist**2
    )


def chassis_roll_deg(
    roll_gradient_deg_per_g: float,
    lateral_accel_g: float,
) -> float:
    """Calculate chassis roll from roll gradient and lateral acceleration."""
    return (
        roll_gradient_deg_per_g
        * lateral_accel_g
    )


def arb_twist_deg(
    chassis_roll: float,
    motion_ratio_roll_per_twist: float,
) -> float:
    """Calculate relative ARB twist from chassis roll.

    The motion-ratio convention is:

        motion ratio = chassis roll / ARB twist
    """
    if motion_ratio_roll_per_twist <= 0:
        raise ValueError(
            "Motion ratio must be positive"
        )

    return (
        chassis_roll
        / motion_ratio_roll_per_twist
    )


def polar_moment_hollow(
    od_m: float,
    id_m: float,
) -> float:
    """Calculate the polar second moment of area of a hollow circular tube."""
    if (
        od_m <= 0
        or id_m < 0
        or id_m >= od_m
    ):
        raise ValueError(
            "Require 0 <= ID < OD"
        )

    return (
        math.pi
        * (od_m**4 - id_m**4)
        / 32.0
    )


def tube_stiffness_nm_per_deg(
    od_m: float,
    id_m: float,
    length_m: float,
    shear_modulus_pa: float,
) -> float:
    """Calculate hollow-tube torsional stiffness in N·m/degree."""
    if (
        length_m <= 0
        or shear_modulus_pa <= 0
    ):
        raise ValueError(
            "Length and shear modulus must be positive"
        )

    polar_moment = polar_moment_hollow(
        od_m,
        id_m,
    )

    stiffness_nm_per_rad = (
        shear_modulus_pa
        * polar_moment
        / length_m
    )

    return (
        stiffness_nm_per_rad
        / DEG_PER_RAD
    )


def tube_twist_deg(
    torque_nm: float,
    tube_k_nm_per_deg: float,
) -> float:
    """Calculate tube twist in degrees."""
    if tube_k_nm_per_deg <= 0:
        raise ValueError(
            "Tube stiffness must be positive"
        )

    return (
        torque_nm
        / tube_k_nm_per_deg
    )


def tube_shear_stress_pa(
    torque_nm: float,
    od_m: float,
    id_m: float,
) -> float:
    """Calculate maximum nominal torsional shear stress."""
    radius = od_m / 2.0

    polar_moment = polar_moment_hollow(
        od_m,
        id_m,
    )

    return (
        abs(torque_nm)
        * radius
        / polar_moment
    )


def tube_mass_kg(
    od_m: float,
    id_m: float,
    length_m: float,
    density_kg_m3: float,
) -> float:
    """Calculate the mass of the uniform hollow portion of a torsion tube."""
    if (
        length_m <= 0
        or density_kg_m3 <= 0
    ):
        raise ValueError(
            "Length and density must be positive"
        )

    if (
        od_m <= 0
        or id_m < 0
        or id_m >= od_m
    ):
        raise ValueError(
            "Require 0 <= ID < OD"
        )

    cross_section_area = (
        math.pi
        * (od_m**2 - id_m**2)
        / 4.0
    )

    volume = (
        cross_section_area
        * length_m
    )

    return (
        density_kg_m3
        * volume
    )


def required_blade_pair_stiffness(
    total_k: float,
    tube_k: float,
) -> float:
    """Calculate the required equivalent stiffness of the two blades.

    The torsion tube and equivalent blade pair act in series:

        1 / total_k = 1 / tube_k + 1 / blade_pair_k

    Rearranging gives:

        blade_pair_k = total_k * tube_k / (tube_k - total_k)
    """
    if (
        total_k <= 0
        or tube_k <= total_k
    ):
        raise ValueError(
            "Tube stiffness must exceed required total stiffness"
        )

    return (
        total_k
        * tube_k
        / (tube_k - total_k)
    )


def series_stiffness(
    *stiffnesses: float,
) -> float:
    """Combine any number of stiffnesses acting in series."""
    if any(
        stiffness <= 0
        for stiffness in stiffnesses
    ):
        raise ValueError(
            "All stiffnesses must be positive"
        )

    return (
        1.0
        / sum(
            1.0 / stiffness
            for stiffness in stiffnesses
        )
    )


def blade_pair_stiffness_from_single(
    single_blade_k: float,
) -> float:
    """Convert one-blade stiffness into equivalent two-blade stiffness.

    Two identical blades both add compliance:

        1 / k_pair = 1 / k_left + 1 / k_right

    Therefore:

        k_pair = k_single / 2
    """
    return (
        single_blade_k
        / 2.0
    )


def rotated_rectangular_inertia(
    width_m: np.ndarray | float,
    thickness_m: float,
    angle_deg: float,
):
    """Calculate blade inertia about the active bending axis.

    An angle of 0 degrees represents the weak-axis soft position.
    An angle of 90 degrees represents the strong-axis stiff position.
    """
    angle_rad = math.radians(
        angle_deg
    )

    width_array = np.asarray(
        width_m
    )

    weak_axis_inertia = (
        width_array
        * thickness_m**3
        / 12.0
    )

    strong_axis_inertia = (
        thickness_m
        * width_array**3
        / 12.0
    )

    return (
        weak_axis_inertia
        * math.cos(angle_rad) ** 2
        + strong_axis_inertia
        * math.sin(angle_rad) ** 2
    )


@dataclass(frozen=True)
class BladeResult:
    """Results returned by the tapered blade analysis."""

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
    """Analyse a tapered rectangular blade under a tip point load.

    The calculation uses Euler-Bernoulli beam theory and numerical
    integration. Blade width varies linearly from the root to the tip.
    """
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

    number_of_stations = max(
        101,
        int(stations),
    )

    x = np.linspace(
        0.0,
        length_m,
        number_of_stations,
    )

    width = (
        root_width_m
        + (
            tip_width_m
            - root_width_m
        )
        * x
        / length_m
    )

    inertia = rotated_rectangular_inertia(
        width,
        thickness_m,
        angle_deg,
    )

    compliance_integrand = (
        (length_m - x) ** 2
        / (
            youngs_modulus_pa
            * inertia
        )
    )

    compliance = np.trapezoid(
        compliance_integrand,
        x,
    )

    linear_stiffness = (
        1.0
        / compliance
    )

    tip_deflection = (
        force_n
        / linear_stiffness
    )

    single_rotational_stiffness_nm_per_rad = (
        linear_stiffness
        * length_m**2
    )

    single_rotational_stiffness_nm_per_deg = (
        single_rotational_stiffness_nm_per_rad
        / DEG_PER_RAD
    )

    bending_moment = (
        abs(force_n)
        * (length_m - x)
    )

    angle_rad = math.radians(
        angle_deg
    )

    effective_extreme_fibre_distance = (
        0.5
        * (
            np.abs(
                width
                * math.sin(angle_rad)
            )
            + abs(
                thickness_m
                * math.cos(angle_rad)
            )
        )
    )

    stress = (
        bending_moment
        * effective_extreme_fibre_distance
        / inertia
    )

    maximum_stress_index = int(
        np.nanargmax(stress)
    )

    maximum_stress = float(
        stress[maximum_stress_index]
    )

    maximum_stress_location = float(
        x[maximum_stress_index]
    )

    pair_rotational_stiffness_nm_per_deg = (
        single_rotational_stiffness_nm_per_deg
        / 2.0
    )

    return BladeResult(
        linear_stiffness_n_per_m=linear_stiffness,
        single_rot_stiffness_nm_per_deg=(
            single_rotational_stiffness_nm_per_deg
        ),
        pair_rot_stiffness_nm_per_deg=(
            pair_rotational_stiffness_nm_per_deg
        ),
        tip_deflection_m=tip_deflection,
        max_stress_pa=maximum_stress,
        max_stress_x_m=maximum_stress_location,
    )


def tapered_blade_pair_mass_kg(
    length_m: float,
    root_width_m: float,
    tip_width_m: float,
    thickness_m: float,
    density_kg_m3: float,
) -> float:
    """Calculate the mass of two linearly tapered rectangular blades."""
    if min(
        length_m,
        root_width_m,
        tip_width_m,
        thickness_m,
        density_kg_m3,
    ) <= 0:
        raise ValueError(
            "Blade dimensions and density must be positive"
        )

    average_width = (
        root_width_m
        + tip_width_m
    ) / 2.0

    single_blade_volume = (
        thickness_m
        * length_m
        * average_width
    )

    single_blade_mass = (
        density_kg_m3
        * single_blade_volume
    )

    return (
        2.0
        * single_blade_mass
    )


def blade_tip_force(
    torque_nm: float,
    effective_arm_m: float,
) -> float:
    """Convert ARB torque into blade-tip/drop-link force."""
    if effective_arm_m <= 0:
        raise ValueError(
            "Effective arm must be positive"
        )

    return (
        abs(torque_nm)
        / effective_arm_m
    )


def drop_link_checks(
    force_n: float,
    length_m: float,
    diameter_m: float,
    youngs_modulus_pa: float,
    yield_pa: float,
):
    """Check a solid circular drop link for yield and Euler buckling."""
    if min(
        length_m,
        diameter_m,
        youngs_modulus_pa,
        yield_pa,
    ) <= 0:
        raise ValueError(
            "Drop-link inputs must be positive"
        )

    cross_section_area = (
        math.pi
        * diameter_m**2
        / 4.0
    )

    second_moment_of_area = (
        math.pi
        * diameter_m**4
        / 64.0
    )

    axial_stress = (
        abs(force_n)
        / cross_section_area
    )

    pinned_euler_buckling_load = (
        math.pi**2
        * youngs_modulus_pa
        * second_moment_of_area
        / length_m**2
    )

    if axial_stress:
        yield_factor_of_safety = (
            yield_pa
            / axial_stress
        )
    else:
        yield_factor_of_safety = math.inf

    if force_n:
        buckling_factor_of_safety = (
            pinned_euler_buckling_load
            / abs(force_n)
        )
    else:
        buckling_factor_of_safety = math.inf

    return {
        "axial_stress_pa": axial_stress,
        "yield_fos": yield_factor_of_safety,
        "euler_buckling_n": (
            pinned_euler_buckling_load
        ),
        "buckling_fos": (
            buckling_factor_of_safety
        ),
    }


def point_distance(
    point_1,
    point_2,
) -> float:
    """Calculate the three-dimensional distance between two points."""
    point_1_array = np.asarray(
        point_1,
        dtype=float,
    )

    point_2_array = np.asarray(
        point_2,
        dtype=float,
    )

    if (
        point_1_array.shape != (3,)
        or point_2_array.shape != (3,)
    ):
        raise ValueError(
            "Points must have x, y, z coordinates"
        )

    return float(
        np.linalg.norm(
            point_2_array
            - point_1_array
        )
    )


def linear_regression(
    x,
    y,
):
    """Fit a straight line and return slope, intercept and R-squared."""
    x_array = np.asarray(
        x,
        dtype=float,
    )

    y_array = np.asarray(
        y,
        dtype=float,
    )

    finite_mask = (
        np.isfinite(x_array)
        & np.isfinite(y_array)
    )

    if finite_mask.sum() < 3:
        raise ValueError(
            "At least three finite samples are required"
        )

    slope, intercept = np.polyfit(
        x_array[finite_mask],
        y_array[finite_mask],
        1,
    )

    predictions = (
        slope
        * x_array[finite_mask]
        + intercept
    )

    residual_sum_of_squares = float(
        np.sum(
            (
                y_array[finite_mask]
                - predictions
            )
            ** 2
        )
    )

    total_sum_of_squares = float(
        np.sum(
            (
                y_array[finite_mask]
                - np.mean(
                    y_array[finite_mask]
                )
            )
            ** 2
        )
    )

    if total_sum_of_squares:
        r_squared = (
            1.0
            - residual_sum_of_squares
            / total_sum_of_squares
        )
    else:
        r_squared = float("nan")

    return (
        float(slope),
        float(intercept),
        r_squared,
    )