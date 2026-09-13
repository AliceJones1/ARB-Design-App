import math

from arb_core import *


def test_report_chain():
    assert math.isclose(
        chassis_roll_deg(0.6, 2.0),
        1.2,
    )

    assert math.isclose(
        arb_twist_deg(1.2, 0.8),
        1.5,
    )

    assert math.isclose(
        roll_to_twist_stiffness(400, 0.8),
        256,
    )


def test_series_allocation():
    blade_pair_k = required_blade_pair_stiffness(
        500,
        1000,
    )

    assert math.isclose(
        blade_pair_k,
        1000,
    )

    assert math.isclose(
        series_stiffness(
            1000,
            blade_pair_k,
        ),
        500,
    )


def test_tube_units_and_stress():
    tube_k = tube_stiffness_nm_per_deg(
        od_m=0.0127,
        id_m=0.0097536,
        length_m=0.5,
        shear_modulus_pa=79.3e9,
    )

    assert tube_k > 0

    tube_stress = tube_shear_stress_pa(
        torque_nm=100,
        od_m=0.0127,
        id_m=0.0097536,
    )

    assert tube_stress > 0


def test_uniform_blade_matches_closed_form():
    E = 200e9
    L = 0.1
    b = 0.02
    t = 0.004
    F = 1000

    result = tapered_blade_analysis(
        force_n=F,
        length_m=L,
        root_width_m=b,
        tip_width_m=b,
        thickness_m=t,
        youngs_modulus_pa=E,
        angle_deg=0,
    )

    inertia = b * t**3 / 12

    expected_deflection = (
        F * L**3
        / (3 * E * inertia)
    )

    assert math.isclose(
        result.tip_deflection_m,
        expected_deflection,
        rel_tol=2e-5,
    )
