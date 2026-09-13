from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from arb_core import (
    M_PER_IN,
    arb_twist_deg,
    blade_tip_force,
    chassis_roll_deg,
    drop_link_checks,
    ft_lb_per_deg_to_nm_per_deg,
    linear_regression,
    nm_per_deg_to_ft_lb_per_deg,
    point_distance,
    required_blade_pair_stiffness,
    roll_to_twist_stiffness,
    series_stiffness,
    tapered_blade_analysis,
    tube_shear_stress_pa,
    tube_stiffness_nm_per_deg,
    tube_twist_deg,
)


st.set_page_config(
    page_title="ARB Design Studio",
    page_icon="🏎️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        max-width: 1500px;
    }

    .stMetric {
        background: #11182708;
        border: 1px solid #64748b33;
        border-radius: 12px;
        padding: 12px;
    }

    .small-note {
        color: #64748b;
        font-size: 0.88rem;
    }

    .pass {
        color: #138a4b;
        font-weight: 700;
    }

    .fail {
        color: #c63b3b;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("ARB Design Studio")

st.caption(
    "Requirements → stiffness allocation → "
    "component sizing → validation"
)


with st.sidebar:
    st.header("Project")

    axle = st.radio(
        "Axle",
        ["Front", "Rear"],
        horizontal=True,
    )

    unit_rate = st.selectbox(
        "Rate input unit",
        ["N·m/deg", "ft·lbf/deg"],
    )

    st.info(
        "All calculations use SI internally. "
        "Motion ratio is chassis roll / ARB twist, "
        "matching the ARG26 report."
    )

    st.markdown("**Design gates**")

    st.caption(
        "1. Meet soft and stiff rates\n\n"
        "2. Remain elastic with fatigue margin\n\n"
        "3. Respect deflection and packaging\n\n"
        "4. Correlate analytical, FEA and test"
    )


tabs = st.tabs(
    [
        "1 Requirements",
        "2 Geometry",
        "3 Stiffness allocation",
        "4 Torsion tube",
        "5 Blades",
        "6 Drop link",
        "7 Tolerances",
        "8 Validation",
        "9 Telemetry",
        "10 Export",
    ]
)


def rate_to_si(value):
    if unit_rate.startswith("ft"):
        return ft_lb_per_deg_to_nm_per_deg(value)

    return value


def rate_from_si(value):
    if unit_rate.startswith("ft"):
        return nm_per_deg_to_ft_lb_per_deg(value)

    return value


def fnum(value, suffix="", digits=3):
    return f"{value:,.{digits}f}{suffix}"


with tabs[0]:
    st.subheader(
        "Vehicle-level inputs and ARB requirements"
    )

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        rate_reference = st.radio(
            "Given rates are referenced to",
            ["ARB twist", "Chassis roll"],
            horizontal=True,
        )

        kmin_in = st.number_input(
            f"Minimum ARB rate ({unit_rate})",
            min_value=0.001,
            value=250.0,
            step=10.0,
        )

        kmax_in = st.number_input(
            f"Maximum ARB rate ({unit_rate})",
            min_value=0.002,
            value=500.0,
            step=10.0,
        )

    with col_b:
        motion_ratio = st.number_input(
            "ARB motion ratio (roll / twist)",
            min_value=0.001,
            value=0.800,
            step=0.01,
            format="%.4f",
        )

        roll_gradient = st.number_input(
            "Target roll gradient (deg/g)",
            min_value=0.0,
            value=0.60,
            step=0.01,
        )

        ay_max = st.number_input(
            "Maximum lateral acceleration (g)",
            min_value=0.0,
            value=2.00,
            step=0.05,
        )

    with col_c:
        load_factor = st.number_input(
            "Structural load factor",
            min_value=1.0,
            value=1.25,
            step=0.05,
        )

        use_twist_override = st.checkbox(
            "Override maximum ARB twist"
        )

        twist_override = st.number_input(
            "Maximum ARB twist (deg)",
            min_value=0.0,
            value=1.50,
            disabled=not use_twist_override,
        )

    kmin_given = rate_to_si(kmin_in)
    kmax_given = rate_to_si(kmax_in)

    if kmax_given <= kmin_given:
        st.error(
            "Maximum rate must exceed minimum rate."
        )
        st.stop()

    if rate_reference == "Chassis roll":
        kmin = roll_to_twist_stiffness(
            kmin_given,
            motion_ratio,
        )

        kmax = roll_to_twist_stiffness(
            kmax_given,
            motion_ratio,
        )

    else:
        kmin = kmin_given
        kmax = kmax_given

    phi_max = chassis_roll_deg(
        roll_gradient,
        ay_max,
    )

    if use_twist_override:
        theta_max = twist_override
    else:
        theta_max = arb_twist_deg(
            phi_max,
            motion_ratio,
        )

    torque_soft = (
        kmin
        * theta_max
        * load_factor
    )

    torque_stiff = (
        kmax
        * theta_max
        * load_factor
    )

    metric_columns = st.columns(5)

    metric_values = [
        (
            "Twist rate: soft",
            kmin,
            "N·m/deg",
        ),
        (
            "Twist rate: stiff",
            kmax,
            "N·m/deg",
        ),
        (
            "Chassis roll",
            phi_max,
            "deg",
        ),
        (
            "ARB twist",
            theta_max,
            "deg",
        ),
        (
            "Design torque",
            torque_stiff,
            "N·m",
        ),
    ]

    for column, (label, value, unit) in zip(
        metric_columns,
        metric_values,
    ):
        column.metric(
            label,
            fnum(value, f" {unit}", 2),
        )

    st.caption(
        "Design torque includes the structural load "
        "factor. Use the stiff rate for the governing "
        "torque unless your load cases show otherwise."
    )


with tabs[1]:
    st.subheader(
        "Geometry from suspension hardpoints"
    )

    st.caption(
        "Coordinates may use any consistent length "
        "unit; select the unit below for conversion."
    )

    geometry_unit = st.selectbox(
        "Coordinate unit",
        ["mm", "m", "in"],
        index=0,
    )

    geometry_scale = {
        "mm": 0.001,
        "m": 1.0,
        "in": M_PER_IN,
    }[geometry_unit]

    geometry_columns = st.columns(3)

    def point_inputs(column, title, defaults):
        with column:
            st.markdown(f"**{title}**")

            return [
                st.number_input(
                    f"{title} {axis}",
                    value=float(value),
                    key=f"{axle}_{title}_{axis}",
                )
                for axis, value in zip(
                    "xyz",
                    defaults,
                )
            ]

    pivot_point = point_inputs(
        geometry_columns[0],
        "ARB_PVT",
        [0, 0, 0],
    )

    blade_tip_point = point_inputs(
        geometry_columns[1],
        "ARB_DL",
        [0, 150, 0],
    )

    rocker_point = point_inputs(
        geometry_columns[2],
        "RK_ARB",
        [0, 150, 180],
    )

    blade_length_geometry = point_distance(
        np.array(pivot_point) * geometry_scale,
        np.array(blade_tip_point) * geometry_scale,
    )

    drop_link_length_geometry = point_distance(
        np.array(blade_tip_point) * geometry_scale,
        np.array(rocker_point) * geometry_scale,
    )

    geometry_metric_1, geometry_metric_2 = (
        st.columns(2)
    )

    geometry_metric_1.metric(
        "Blade pivot-to-tip length",
        fnum(
            blade_length_geometry * 1000,
            " mm",
            1,
        ),
    )

    geometry_metric_2.metric(
        "Drop-link length",
        fnum(
            drop_link_length_geometry * 1000,
            " mm",
            1,
        ),
    )

    st.warning(
        "Point distances alone do not determine motion "
        "ratio. Confirm the supplied motion ratio with "
        "a suspension kinematic sweep through bump, "
        "rebound and roll."
    )


with tabs[2]:
    st.subheader(
        "Torsion-bar / blade stiffness allocation"
    )

    sweep_columns = st.columns(3)

    sweep_min = sweep_columns[0].number_input(
        "Sweep start (× stiff target)",
        min_value=1.01,
        value=1.10,
        step=0.05,
    )

    sweep_max = sweep_columns[1].number_input(
        "Sweep end (× stiff target)",
        min_value=1.02,
        value=4.00,
        step=0.25,
    )

    sweep_count = int(
        sweep_columns[2].number_input(
            "Candidates",
            min_value=6,
            max_value=200,
            value=40,
        )
    )

    factors = np.linspace(
        sweep_min,
        max(sweep_max, sweep_min + 0.01),
        sweep_count,
    )

    sweep_rows = []

    for factor in factors:
        tube_candidate_k = kmax * factor

        blade_pair_soft_k = (
            required_blade_pair_stiffness(
                kmin,
                tube_candidate_k,
            )
        )

        blade_pair_stiff_k = (
            required_blade_pair_stiffness(
                kmax,
                tube_candidate_k,
            )
        )

        sweep_rows.append(
            {
                "Tube factor": factor,
                "Tube k (N·m/deg)": tube_candidate_k,
                "Blade pair soft": blade_pair_soft_k,
                "Blade pair stiff": blade_pair_stiff_k,
                "Blade ratio": (
                    blade_pair_stiff_k
                    / blade_pair_soft_k
                ),
                "Tube twist @ design T (deg)": (
                    tube_twist_deg(
                        torque_stiff,
                        tube_candidate_k,
                    )
                ),
                "Blade soft rotation @ design T (deg)": (
                    torque_stiff
                    / blade_pair_soft_k
                ),
                "Blade stiff rotation @ design T (deg)": (
                    torque_stiff
                    / blade_pair_stiff_k
                ),
            }
        )

    sweep_df = pd.DataFrame(sweep_rows)

    stiffness_figure = go.Figure()

    stiffness_series = [
        (
            "Tube k (N·m/deg)",
            "Tube",
        ),
        (
            "Blade pair soft",
            "Blade pair: soft",
        ),
        (
            "Blade pair stiff",
            "Blade pair: stiff",
        ),
    ]

    for y_column, name in stiffness_series:
        stiffness_figure.add_trace(
            go.Scatter(
                x=sweep_df["Tube factor"],
                y=sweep_df[y_column],
                name=name,
            )
        )

    stiffness_figure.update_layout(
        xaxis_title=(
            "Tube stiffness / required stiff ARB rate"
        ),
        yaxis_title="Stiffness (N·m/deg)",
        height=420,
        legend_orientation="h",
    )

    st.plotly_chart(
        stiffness_figure,
        width="stretch",
    )

    selected_factor = st.slider(
        "Selected tube stiffness factor",
        float(sweep_min),
        float(max(sweep_max, sweep_min + 0.01)),
        min(
            2.0,
            float(max(
                sweep_max,
                sweep_min + 0.01,
            )),
        ),
        0.01,
    )

    selected_tube_k = kmax * selected_factor

    required_blade_soft_k = (
        required_blade_pair_stiffness(
            kmin,
            selected_tube_k,
        )
    )

    required_blade_stiff_k = (
        required_blade_pair_stiffness(
            kmax,
            selected_tube_k,
        )
    )

    allocation_metrics = st.columns(4)

    allocation_metrics[0].metric(
        "Selected tube k",
        fnum(
            selected_tube_k,
            " N·m/deg",
            1,
        ),
    )

    allocation_metrics[1].metric(
        "Required blade pair: soft",
        fnum(
            required_blade_soft_k,
            " N·m/deg",
            1,
        ),
    )

    allocation_metrics[2].metric(
        "Required blade pair: stiff",
        fnum(
            required_blade_stiff_k,
            " N·m/deg",
            1,
        ),
    )

    allocation_metrics[3].metric(
        "Required blade ratio",
        fnum(
            required_blade_stiff_k
            / required_blade_soft_k,
            "×",
            2,
        ),
    )

    with st.expander("Candidate table"):
        st.dataframe(
            sweep_df.style.format("{:.3f}"),
            width="stretch",
            height=320,
        )


with tabs[3]:
    st.subheader("Hollow torsion-tube sizing")

    tube_columns = st.columns(4)

    default_od = (
        15.875
        if axle == "Rear"
        else 12.700
    )

    default_wall = (
        0.711
        if axle == "Rear"
        else 1.473
    )

    od_mm = tube_columns[0].number_input(
        "Outer diameter (mm)",
        min_value=0.1,
        value=default_od,
        step=0.1,
    )

    wall_mm = tube_columns[1].number_input(
        "Wall thickness (mm)",
        min_value=0.05,
        value=default_wall,
        step=0.05,
    )

    tube_length_mm = tube_columns[2].number_input(
        "Effective compliant length (mm)",
        min_value=1.0,
        value=500.0,
        step=5.0,
    )

    shear_modulus_gpa = (
        tube_columns[3].number_input(
            "Shear modulus (GPa)",
            min_value=0.1,
            value=79.3,
            step=0.1,
        )
    )

    strength_columns = st.columns(2)

    shear_yield_mpa = (
        strength_columns[0].number_input(
            "Shear yield strength (MPa)",
            min_value=1.0,
            value=460.0,
            step=10.0,
        )
    )

    fatigue_allowable_mpa = (
        strength_columns[1].number_input(
            "Corrected shear fatigue allowable (MPa)",
            min_value=1.0,
            value=250.0,
            step=10.0,
        )
    )

    od_m = od_mm / 1000
    wall_m = wall_mm / 1000
    id_m = od_m - 2 * wall_m

    if id_m <= 0:
        st.error(
            "Wall thickness must be less than half "
            "the OD."
        )
        st.stop()

    actual_tube_k = tube_stiffness_nm_per_deg(
        od_m=od_m,
        id_m=id_m,
        length_m=tube_length_mm / 1000,
        shear_modulus_pa=(
            shear_modulus_gpa * 1e9
        ),
    )

    tube_shear = tube_shear_stress_pa(
        torque_nm=torque_stiff,
        od_m=od_m,
        id_m=id_m,
    )

    tube_metrics = st.columns(5)

    tube_metric_values = [
        (
            "Tube ID",
            id_m * 1000,
            "mm",
        ),
        (
            "Calculated tube k",
            actual_tube_k,
            "N·m/deg",
        ),
        (
            "Target tube k",
            selected_tube_k,
            "N·m/deg",
        ),
        (
            "Peak shear",
            tube_shear / 1e6,
            "MPa",
        ),
        (
            "Yield FoS",
            shear_yield_mpa / (tube_shear / 1e6),
            "",
        ),
    ]

    for column, (label, value, unit) in zip(
        tube_metrics,
        tube_metric_values,
    ):
        column.metric(
            label,
            fnum(value, f" {unit}", 2),
        )

    tube_error = (
        100
        * (actual_tube_k - selected_tube_k)
        / selected_tube_k
    )

    st.progress(
        min(abs(tube_error) / 50, 1.0),
        text=(
            f"Tube stiffness error: "
            f"{tube_error:+.1f}%"
        ),
    )

    if (
        tube_shear / 1e6
        <= fatigue_allowable_mpa
    ):
        fatigue_margin = (
            fatigue_allowable_mpa
            / (tube_shear / 1e6)
        )

        st.success(
            "Peak nominal shear is below the entered "
            "fatigue allowable "
            f"(margin {fatigue_margin:.2f})."
        )

    else:
        st.error(
            "Peak nominal shear exceeds the entered "
            "fatigue allowable."
        )

    st.caption(
        "Nominal pure-torsion result only. Apply "
        "stress-concentration and weld/insert effects "
        "separately."
    )


with tabs[4]:
    st.subheader(
        "Adjustable tapered-blade model"
    )

    blade_columns = st.columns(4)

    blade_length_mm = (
        blade_columns[0].number_input(
            "Effective blade length (mm)",
            min_value=1.0,
            value=max(
                100.0,
                blade_length_geometry * 1000,
            ),
            step=1.0,
        )
    )

    root_width_mm = (
        blade_columns[1].number_input(
            "Root width (mm)",
            min_value=0.1,
            value=22.0,
            step=0.5,
        )
    )

    tip_width_mm = (
        blade_columns[2].number_input(
            "Tip width (mm)",
            min_value=0.1,
            value=9.0,
            step=0.5,
        )
    )

    blade_thickness_mm = (
        blade_columns[3].number_input(
            "Blade thickness (mm)",
            min_value=0.1,
            value=4.0,
            step=0.1,
        )
    )

    blade_property_columns = st.columns(3)

    blade_modulus_gpa = (
        blade_property_columns[0].number_input(
            "Blade Young's modulus (GPa)",
            min_value=0.1,
            value=205.0,
            step=1.0,
        )
    )

    blade_allowable_mpa = (
        blade_property_columns[1].number_input(
            "Blade allowable stress (MPa)",
            min_value=1.0,
            value=1000.0,
            step=25.0,
        )
    )

    angle_steps = int(
        blade_property_columns[2].number_input(
            "Adjustment positions",
            min_value=2,
            max_value=91,
            value=7,
        )
    )

    blade_length_m = blade_length_mm / 1000

    blade_force = blade_tip_force(
        torque_nm=torque_stiff,
        effective_arm_m=blade_length_m,
    )

    angles = np.linspace(
        0,
        90,
        angle_steps,
    )

    blade_rows = []

    for angle in angles:
        blade_result = tapered_blade_analysis(
            force_n=blade_force,
            length_m=blade_length_m,
            root_width_m=root_width_mm / 1000,
            tip_width_m=tip_width_mm / 1000,
            thickness_m=(
                blade_thickness_mm / 1000
            ),
            youngs_modulus_pa=(
                blade_modulus_gpa * 1e9
            ),
            angle_deg=angle,
        )

        achieved_rate = series_stiffness(
            actual_tube_k,
            blade_result
            .pair_rot_stiffness_nm_per_deg,
        )

        blade_rows.append(
            {
                "Angle (deg)": angle,
                "Blade pair k": (
                    blade_result
                    .pair_rot_stiffness_nm_per_deg
                ),
                "Total ARB k": achieved_rate,
                "Tip deflection (mm)": (
                    blade_result.tip_deflection_m
                    * 1000
                ),
                "Max nominal stress (MPa)": (
                    blade_result.max_stress_pa
                    / 1e6
                ),
                "Stress FoS": (
                    blade_allowable_mpa
                    / (
                        blade_result.max_stress_pa
                        / 1e6
                    )
                ),
            }
        )

    blade_df = pd.DataFrame(blade_rows)

    blade_figure = go.Figure()

    blade_figure.add_trace(
        go.Scatter(
            x=blade_df["Angle (deg)"],
            y=blade_df["Total ARB k"],
            name="Predicted total ARB",
        )
    )

    blade_figure.add_hline(
        y=kmin,
        line_dash="dash",
        annotation_text="Soft target",
    )

    blade_figure.add_hline(
        y=kmax,
        line_dash="dash",
        annotation_text="Stiff target",
    )

    blade_figure.update_layout(
        xaxis_title=(
            "Blade angle (0° soft, 90° stiff)"
        ),
        yaxis_title="Total rate (N·m/deg)",
        height=410,
    )

    st.plotly_chart(
        blade_figure,
        width="stretch",
    )

    soft_result = blade_df.iloc[0]
    stiff_result = blade_df.iloc[-1]

    blade_metrics = st.columns(5)

    blade_metric_values = [
        (
            "Blade force",
            blade_force,
            "N",
        ),
        (
            "Predicted soft rate",
            soft_result["Total ARB k"],
            "N·m/deg",
        ),
        (
            "Predicted stiff rate",
            stiff_result["Total ARB k"],
            "N·m/deg",
        ),
        (
            "Soft tip deflection",
            soft_result["Tip deflection (mm)"],
            "mm",
        ),
        (
            "Worst stress",
            blade_df[
                "Max nominal stress (MPa)"
            ].max(),
            "MPa",
        ),
    ]

    for column, (label, value, unit) in zip(
        blade_metrics,
        blade_metric_values,
    ):
        column.metric(
            label,
            fnum(value, f" {unit}", 2),
        )

    soft_rate_error = (
        100
        * (
            soft_result["Total ARB k"]
            - kmin
        )
        / kmin
    )

    stiff_rate_error = (
        100
        * (
            stiff_result["Total ARB k"]
            - kmax
        )
        / kmax
    )

    st.write(
        "Endpoint errors: "
        f"soft **{soft_rate_error:+.1f}%**, "
        f"stiff **{stiff_rate_error:+.1f}%**"
    )

    st.dataframe(
        blade_df.style.format("{:.3f}"),
        width="stretch",
    )

    st.caption(
        "Beam-model stress is nominal. Root features, "
        "holes, transitions and contact require FEA. "
        "Blade rotation changes both stiffness and the "
        "extreme-fibre approximation."
    )


with tabs[5]:
    st.subheader(
        "Drop-link axial yield and buckling"
    )

    drop_link_columns = st.columns(4)

    drop_link_length_mm = (
        drop_link_columns[0].number_input(
            "Pin-to-pin length (mm)",
            min_value=1.0,
            value=max(
                100.0,
                drop_link_length_geometry * 1000,
            ),
            step=1.0,
        )
    )

    drop_link_diameter_mm = (
        drop_link_columns[1].number_input(
            "Minimum/net diameter (mm)",
            min_value=0.1,
            value=5.0,
            step=0.1,
        )
    )

    drop_link_modulus_gpa = (
        drop_link_columns[2].number_input(
            "Young's modulus (GPa)",
            min_value=0.1,
            value=71.7,
            step=1.0,
        )
    )

    drop_link_yield_mpa = (
        drop_link_columns[3].number_input(
            "Yield strength (MPa)",
            min_value=1.0,
            value=500.0,
            step=10.0,
        )
    )

    drop_link_result = drop_link_checks(
        force_n=blade_force,
        length_m=drop_link_length_mm / 1000,
        diameter_m=(
            drop_link_diameter_mm / 1000
        ),
        youngs_modulus_pa=(
            drop_link_modulus_gpa * 1e9
        ),
        yield_pa=(
            drop_link_yield_mpa * 1e6
        ),
    )

    drop_link_metrics = st.columns(4)

    drop_link_metric_values = [
        (
            "Axial stress",
            (
                drop_link_result[
                    "axial_stress_pa"
                ]
                / 1e6
            ),
            "MPa",
        ),
        (
            "Yield FoS",
            drop_link_result["yield_fos"],
            "",
        ),
        (
            "Pinned Euler load",
            drop_link_result[
                "euler_buckling_n"
            ],
            "N",
        ),
        (
            "Buckling FoS",
            drop_link_result["buckling_fos"],
            "",
        ),
    ]

    for column, (label, value, unit) in zip(
        drop_link_metrics,
        drop_link_metric_values,
    ):
        column.metric(
            label,
            fnum(value, f" {unit}", 2),
        )

    st.caption(
        "Use the minimum threaded/root section. Euler "
        "buckling assumes an ideal straight "
        "pinned-pinned member and excludes rod-end "
        "limitations and eccentricity."
    )


with tabs[6]:
    st.subheader(
        "Manufacturing sensitivity and "
        "Monte Carlo tolerance"
    )

    st.caption(
        "Propagates OD, wall and blade-section "
        "tolerances through the analytical model."
    )

    tolerance_columns = st.columns(4)

    od_tolerance_mm = (
        tolerance_columns[0].number_input(
            "Tube OD tolerance ± (mm)",
            min_value=0.0,
            value=0.05,
            step=0.01,
        )
    )

    wall_tolerance_mm = (
        tolerance_columns[1].number_input(
            "Wall tolerance ± (mm)",
            min_value=0.0,
            value=0.05,
            step=0.01,
        )
    )

    width_tolerance_mm = (
        tolerance_columns[2].number_input(
            "Blade width tolerance ± (mm)",
            min_value=0.0,
            value=0.05,
            step=0.01,
        )
    )

    thickness_tolerance_mm = (
        tolerance_columns[3].number_input(
            "Blade thickness tolerance ± (mm)",
            min_value=0.0,
            value=0.025,
            step=0.005,
        )
    )

    sample_count = 3000
    random_generator = np.random.default_rng(27)
    simulated_rates = []

    for _ in range(sample_count):
        simulated_od = (
            od_mm
            + random_generator.uniform(
                -od_tolerance_mm,
                od_tolerance_mm,
            )
        ) / 1000

        simulated_wall = (
            wall_mm
            + random_generator.uniform(
                -wall_tolerance_mm,
                wall_tolerance_mm,
            )
        ) / 1000

        simulated_id = (
            simulated_od
            - 2 * simulated_wall
        )

        simulated_root_width = (
            root_width_mm
            + random_generator.uniform(
                -width_tolerance_mm,
                width_tolerance_mm,
            )
        ) / 1000

        simulated_tip_width = (
            tip_width_mm
            + random_generator.uniform(
                -width_tolerance_mm,
                width_tolerance_mm,
            )
        ) / 1000

        simulated_thickness = (
            blade_thickness_mm
            + random_generator.uniform(
                -thickness_tolerance_mm,
                thickness_tolerance_mm,
            )
        ) / 1000

        if (
            simulated_id <= 0
            or min(
                simulated_root_width,
                simulated_tip_width,
                simulated_thickness,
            ) <= 0
        ):
            continue

        simulated_tube_k = (
            tube_stiffness_nm_per_deg(
                od_m=simulated_od,
                id_m=simulated_id,
                length_m=tube_length_mm / 1000,
                shear_modulus_pa=(
                    shear_modulus_gpa * 1e9
                ),
            )
        )

        simulated_blade = (
            tapered_blade_analysis(
                force_n=blade_force,
                length_m=blade_length_m,
                root_width_m=(
                    simulated_root_width
                ),
                tip_width_m=(
                    simulated_tip_width
                ),
                thickness_m=(
                    simulated_thickness
                ),
                youngs_modulus_pa=(
                    blade_modulus_gpa * 1e9
                ),
                angle_deg=90,
            )
        )

        simulated_rate = series_stiffness(
            simulated_tube_k,
            simulated_blade
            .pair_rot_stiffness_nm_per_deg,
        )

        simulated_rates.append(simulated_rate)

    simulated_rates = np.array(
        simulated_rates
    )

    tolerance_figure = px.histogram(
        x=simulated_rates,
        nbins=45,
        labels={
            "x": (
                "Predicted stiff-setting rate "
                "(N·m/deg)"
            )
        },
    )

    tolerance_figure.add_vline(
        x=kmax,
        line_dash="dash",
        annotation_text="Target",
    )

    tolerance_figure.update_layout(
        height=400,
        yaxis_title="Samples",
    )

    st.plotly_chart(
        tolerance_figure,
        width="stretch",
    )

    tolerance_metrics = st.columns(4)

    tolerance_metric_values = [
        (
            "Mean",
            simulated_rates.mean(),
            "N·m/deg",
        ),
        (
            "5th percentile",
            np.percentile(
                simulated_rates,
                5,
            ),
            "N·m/deg",
        ),
        (
            "95th percentile",
            np.percentile(
                simulated_rates,
                95,
            ),
            "N·m/deg",
        ),
        (
            "Spread",
            np.ptp(
                np.percentile(
                    simulated_rates,
                    [5, 95],
                )
            ),
            "N·m/deg",
        ),
    ]

    for column, (label, value, unit) in zip(
        tolerance_metrics,
        tolerance_metric_values,
    ):
        column.metric(
            label,
            fnum(value, f" {unit}", 2),
        )


with tabs[7]:
    st.subheader(
        "Analytical ↔ FEA ↔ measured correlation"
    )

    default_correlation = pd.DataFrame(
        {
            "Quantity": [
                "Soft assembly stiffness",
                "Stiff assembly stiffness",
                "Blade max stress",
                "Blade tip deflection",
                "Torsion-bar twist",
            ],
            "Analytical": [
                float(
                    soft_result["Total ARB k"]
                ),
                float(
                    stiff_result["Total ARB k"]
                ),
                float(
                    blade_df[
                        "Max nominal stress (MPa)"
                    ].max()
                ),
                float(
                    soft_result[
                        "Tip deflection (mm)"
                    ]
                ),
                tube_twist_deg(
                    torque_stiff,
                    actual_tube_k,
                ),
            ],
            "FEA": [np.nan] * 5,
            "Measured": [np.nan] * 5,
            "Unit": [
                "N·m/deg",
                "N·m/deg",
                "MPa",
                "mm",
                "deg",
            ],
        }
    )

    correlation = st.data_editor(
        default_correlation,
        num_rows="dynamic",
        width="stretch",
    )

    for source in ["FEA", "Measured"]:
        correlation[
            f"{source} difference (%)"
        ] = (
            (
                correlation[source]
                - correlation["Analytical"]
            )
            / correlation["Analytical"]
            * 100
        )

    st.dataframe(
        correlation,
        width="stretch",
    )

    st.download_button(
        "Download correlation CSV",
        correlation.to_csv(
            index=False
        ).encode(),
        f"{axle.lower()}_arb_correlation.csv",
        "text/csv",
    )

    st.caption(
        "A 5–10% stiffness agreement is a useful "
        "initial correlation goal; investigate fixture "
        "compliance, contact, friction and boundary "
        "conditions when disagreement is larger."
    )


with tabs[8]:
    st.subheader(
        "On-car roll-gradient correlation"
    )

    st.write(
        "Upload a CSV containing lateral acceleration "
        "in g and chassis roll in degrees, or use the "
        "demo data."
    )

    uploaded_file = st.file_uploader(
        "Telemetry CSV",
        type="csv",
    )

    if uploaded_file is not None:
        telemetry = pd.read_csv(
            uploaded_file
        )

    else:
        demo_ay = np.linspace(
            -2,
            2,
            201,
        )

        random_generator = (
            np.random.default_rng(7)
        )

        telemetry = pd.DataFrame(
            {
                "LatAcc_g": demo_ay,
                "Roll_deg": (
                    0.62 * demo_ay
                    + 0.03
                    + random_generator.normal(
                        0,
                        0.035,
                        len(demo_ay),
                    )
                ),
            }
        )

    st.dataframe(
        telemetry.head(),
        width="stretch",
    )

    numeric_columns = list(
        telemetry.select_dtypes(
            include=np.number
        ).columns
    )

    if len(numeric_columns) >= 2:
        telemetry_columns = st.columns(3)

        ay_column = (
            telemetry_columns[0].selectbox(
                "Lateral acceleration column",
                numeric_columns,
                index=0,
            )
        )

        roll_column = (
            telemetry_columns[1].selectbox(
                "Roll angle column",
                numeric_columns,
                index=min(
                    1,
                    len(numeric_columns) - 1,
                ),
            )
        )

        minimum_abs_ay = (
            telemetry_columns[2].number_input(
                "Minimum |ay| included (g)",
                min_value=0.0,
                value=0.3,
                step=0.1,
            )
        )

        telemetry_mask = (
            telemetry[ay_column].abs()
            >= minimum_abs_ay
        )

        try:
            slope, intercept, r_squared = (
                linear_regression(
                    telemetry.loc[
                        telemetry_mask,
                        ay_column,
                    ],
                    telemetry.loc[
                        telemetry_mask,
                        roll_column,
                    ],
                )
            )

            telemetry_figure = px.scatter(
                telemetry.loc[telemetry_mask],
                x=ay_column,
                y=roll_column,
                opacity=0.45,
            )

            fit_x = np.array(
                [
                    telemetry.loc[
                        telemetry_mask,
                        ay_column,
                    ].min(),
                    telemetry.loc[
                        telemetry_mask,
                        ay_column,
                    ].max(),
                ]
            )

            telemetry_figure.add_trace(
                go.Scatter(
                    x=fit_x,
                    y=(
                        slope * fit_x
                        + intercept
                    ),
                    name="Fit",
                    line={"width": 3},
                )
            )

            telemetry_figure.update_layout(
                height=430
            )

            st.plotly_chart(
                telemetry_figure,
                width="stretch",
            )

            telemetry_metrics = st.columns(3)

            telemetry_metrics[0].metric(
                "Measured roll gradient",
                fnum(
                    slope,
                    " deg/g",
                    3,
                ),
            )

            telemetry_metrics[1].metric(
                "Intercept",
                fnum(
                    intercept,
                    " deg",
                    3,
                ),
            )

            telemetry_metrics[2].metric(
                "R²",
                fnum(
                    r_squared,
                    "",
                    4,
                ),
            )

        except ValueError as error:
            st.warning(str(error))


with tabs[9]:
    st.subheader(
        "Design snapshot and exports"
    )

    summary = {
        "project": {
            "axle": axle,
            "motion_ratio_roll_per_twist": (
                motion_ratio
            ),
            "load_factor": load_factor,
        },
        "requirements": {
            "k_soft_nm_per_deg_twist": kmin,
            "k_stiff_nm_per_deg_twist": kmax,
            "max_chassis_roll_deg": phi_max,
            "max_arb_twist_deg": theta_max,
            "design_torque_nm": torque_stiff,
        },
        "geometry": {
            "blade_length_mm": blade_length_mm,
            "drop_link_length_mm": (
                drop_link_length_mm
            ),
        },
        "tube": {
            "od_mm": od_mm,
            "wall_mm": wall_mm,
            "length_mm": tube_length_mm,
            "calculated_k_nm_per_deg": (
                actual_tube_k
            ),
            "peak_shear_mpa": (
                tube_shear / 1e6
            ),
        },
        "blade": {
            "root_width_mm": root_width_mm,
            "tip_width_mm": tip_width_mm,
            "thickness_mm": (
                blade_thickness_mm
            ),
            "force_n": blade_force,
            "predicted_soft_rate_nm_per_deg": (
                float(
                    soft_result["Total ARB k"]
                )
            ),
            "predicted_stiff_rate_nm_per_deg": (
                float(
                    stiff_result["Total ARB k"]
                )
            ),
            "worst_nominal_stress_mpa": (
                float(
                    blade_df[
                        "Max nominal stress (MPa)"
                    ].max()
                )
            ),
        },
        "drop_link": drop_link_result,
    }

    st.json(summary)

    export_columns = st.columns(3)

    export_columns[0].download_button(
        "Download design JSON",
        json.dumps(
            summary,
            indent=2,
        ).encode(),
        f"{axle.lower()}_arb_design.json",
        "application/json",
    )

    export_columns[1].download_button(
        "Download stiffness sweep CSV",
        sweep_df.to_csv(
            index=False
        ).encode(),
        f"{axle.lower()}_stiffness_sweep.csv",
        "text/csv",
    )

    export_columns[2].download_button(
        "Download blade settings CSV",
        blade_df.to_csv(
            index=False
        ).encode(),
        f"{axle.lower()}_blade_settings.csv",
        "text/csv",
    )

    checklist = pd.DataFrame(
        {
            "Gate": [
                "Soft stiffness target",
                "Stiff stiffness target",
                "Tube yield FoS ≥ 1.5",
                "Tube fatigue margin ≥ 1.0",
                "Blade nominal FoS ≥ 1.0",
                "Drop-link yield FoS ≥ 1.5",
                "Drop-link buckling FoS ≥ 2.0",
            ],
            "Pass": [
                abs(soft_rate_error) <= 10,
                abs(stiff_rate_error) <= 10,
                (
                    shear_yield_mpa
                    / (tube_shear / 1e6)
                    >= 1.5
                ),
                (
                    fatigue_allowable_mpa
                    / (tube_shear / 1e6)
                    >= 1.0
                ),
                (
                    blade_df[
                        "Stress FoS"
                    ].min()
                    >= 1.0
                ),
                (
                    drop_link_result[
                        "yield_fos"
                    ]
                    >= 1.5
                ),
                (
                    drop_link_result[
                        "buckling_fos"
                    ]
                    >= 2.0
                ),
            ],
        }
    )

    st.dataframe(
        checklist,
        width="stretch",
        hide_index=True,
    )

    st.warning(
        "Passing this preliminary checklist does not "
        "replace detailed joint, weld, fatigue, contact, "
        "bearing, mount or chassis-interface analysis."
    )
