from __future__ import annotations

import json
from pathlib import Path

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
    tapered_blade_pair_mass_kg,
    tube_mass_kg,
    tube_shear_stress_pa,
    tube_stiffness_nm_per_deg,
    tube_twist_deg,
)

st.set_page_config(
    page_title="ARB Design Studio",
    page_icon="🏎️",
    layout="wide",
)

SAVE_PATH = Path(__file__).with_name("saved_design.json")

# Only these widget values are written to a design file. Derived results are
# recalculated when the design is loaded.
INPUT_KEYS = [
    "axle",
    "unit_rate",
    "rate_reference",
    "kmin_in",
    "kmax_in",
    "mr",
    "roll_grad",
    "ay_max",
    "load_factor",
    "use_twist_override",
    "twist_override",
    "geom_unit",
    "ARB_PVT_x",
    "ARB_PVT_y",
    "ARB_PVT_z",
    "ARB_DL_x",
    "ARB_DL_y",
    "ARB_DL_z",
    "RK_ARB_x",
    "RK_ARB_y",
    "RK_ARB_z",
    "sweep_min",
    "sweep_max",
    "sweep_n",
    "selected_factor",
    "od_mm",
    "wall_mm",
    "tube_len_mm",
    "G_gpa",
    "shear_yield_mpa",
    "fatigue_allow_mpa",
    "blade_len_mm",
    "root_w_mm",
    "tip_w_mm",
    "thick_mm",
    "E_gpa",
    "blade_allow_mpa",
    "angle_steps",
    "dl_len",
    "dl_d",
    "dl_E",
    "dl_yield",
    "od_tol",
    "wall_tol",
    "width_tol",
    "thick_tol",
    "opt_tube_density",
    "opt_blade_density",
    "opt_rate_tol",
    "opt_tube_fos",
    "opt_blade_fos",
    "opt_soft_defl",
    "opt_stiff_defl",
    "opt_od_min",
    "opt_od_max",
    "opt_wall_min",
    "opt_wall_max",
    "opt_root_min",
    "opt_root_max",
    "opt_tip_min",
    "opt_tip_max",
    "opt_thick_min",
    "opt_thick_max",
    "opt_grid_n",
    "opt_tube_length",
    "opt_blade_length",
    "opt_G",
    "opt_tube_yield",
    "opt_tube_fatigue",
    "opt_E",
    "opt_blade_allow",
]


def apply_saved_inputs(data: dict) -> None:
    """Apply a saved design before the keyed widgets are constructed."""
    inputs = data.get("inputs", data)

    if not isinstance(inputs, dict):
        raise ValueError(
            "The design file does not contain an input dictionary."
        )

    restored = 0

    for key in INPUT_KEYS:
        if key in inputs:
            st.session_state[key] = inputs[key]
            restored += 1

    if restored == 0:
        raise ValueError(
            "No compatible ARB input values were found in the file."
        )


# Restore the last design automatically when a new Streamlit session starts in
# the same Codespace.
if "_local_design_checked" not in st.session_state:
    st.session_state._local_design_checked = True

    if SAVE_PATH.exists():
        try:
            apply_saved_inputs(
                json.loads(SAVE_PATH.read_text(encoding="utf-8"))
            )
            st.session_state._restore_notice = (
                "Restored the last Codespace design."
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            st.session_state._restore_notice = (
                f"Could not restore saved_design.json: {exc}"
            )

if "_candidate_to_apply" in st.session_state:
    candidate = st.session_state.pop("_candidate_to_apply")

    for key, value in candidate.items():
        st.session_state[key] = value

    st.session_state._restore_notice = (
        "Applied the selected mass-optimised candidate."
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
        font-size: .88rem;
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
    "Requirements → stiffness allocation → component sizing → validation"
)

with st.sidebar:
    st.header("Project")

    uploaded_design = st.file_uploader(
        "Load design JSON",
        type="json",
        key="design_json_upload",
    )

    if st.button(
        "Load uploaded values",
        disabled=uploaded_design is None,
        width="stretch",
    ):
        try:
            apply_saved_inputs(
                json.loads(
                    uploaded_design.getvalue().decode("utf-8")
                )
            )
            st.session_state._restore_notice = (
                f"Loaded {uploaded_design.name}."
            )
            st.rerun()
        except (
            UnicodeDecodeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            st.error(f"Could not load design: {exc}")

    if "_restore_notice" in st.session_state:
        st.success(st.session_state.pop("_restore_notice"))

    axle = st.radio(
        "Axle",
        ["Front", "Rear"],
        horizontal=True,
        key="axle",
    )

    unit_rate = st.selectbox(
        "Rate input unit",
        ["N·m/deg", "ft·lbf/deg"],
        key="unit_rate",
    )

    st.info(
        "All calculations use SI internally. Motion ratio is chassis "
        "roll / ARB twist, matching the ARG26 report."
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
        "4 Mass optimisation",
        "5 Torsion tube",
        "6 Blades",
        "7 Drop link",
        "8 Tolerances",
        "9 Validation",
        "10 Telemetry",
        "11 Export",
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
    st.subheader("Vehicle-level inputs and ARB requirements")

    column_a, column_b, column_c = st.columns(3)

    with column_a:
        rate_reference = st.radio(
            "Given rates are referenced to",
            ["ARB twist", "Chassis roll"],
            horizontal=True,
            key="rate_reference",
        )

        kmin_in = st.number_input(
            f"Minimum ARB rate ({unit_rate})",
            min_value=0.001,
            value=250.0,
            step=10.0,
            key="kmin_in",
        )

        kmax_in = st.number_input(
            f"Maximum ARB rate ({unit_rate})",
            min_value=0.002,
            value=500.0,
            step=10.0,
            key="kmax_in",
        )

    with column_b:
        mr = st.number_input(
            "ARB motion ratio (roll / twist)",
            min_value=0.001,
            value=0.800,
            step=0.01,
            format="%.4f",
            key="mr",
        )

        roll_grad = st.number_input(
            "Target roll gradient (deg/g)",
            min_value=0.0,
            value=0.60,
            step=0.01,
            key="roll_grad",
        )

        ay_max = st.number_input(
            "Maximum lateral acceleration (g)",
            min_value=0.0,
            value=2.00,
            step=0.05,
            key="ay_max",
        )

    with column_c:
        load_factor = st.number_input(
            "Structural load factor",
            min_value=1.0,
            value=1.25,
            step=0.05,
            key="load_factor",
        )

        use_twist_override = st.checkbox(
            "Override maximum ARB twist",
            key="use_twist_override",
        )

        twist_override = st.number_input(
            "Maximum ARB twist (deg)",
            min_value=0.0,
            value=1.50,
            disabled=not use_twist_override,
            key="twist_override",
        )

    kmin_given = rate_to_si(kmin_in)
    kmax_given = rate_to_si(kmax_in)

    if kmax_given <= kmin_given:
        st.error("Maximum rate must exceed minimum rate.")
        st.stop()

    if rate_reference == "Chassis roll":
        kmin = roll_to_twist_stiffness(kmin_given, mr)
        kmax = roll_to_twist_stiffness(kmax_given, mr)
    else:
        kmin = kmin_given
        kmax = kmax_given

    phi_max = chassis_roll_deg(roll_grad, ay_max)

    if use_twist_override:
        theta_max = twist_override
    else:
        theta_max = arb_twist_deg(phi_max, mr)

    torque_soft = kmin * theta_max * load_factor
    torque_stiff = kmax * theta_max * load_factor

    columns = st.columns(5)

    results = [
        ("Twist rate: soft", kmin, "N·m/deg"),
        ("Twist rate: stiff", kmax, "N·m/deg"),
        ("Chassis roll", phi_max, "deg"),
        ("ARB twist", theta_max, "deg"),
        ("Design torque", torque_stiff, "N·m"),
    ]

    for column, (label, value, unit) in zip(columns, results):
        column.metric(
            label,
            fnum(value, f" {unit}", 2),
        )

    st.caption(
        "Design torque includes the structural load factor. Use the stiff "
        "rate for the governing torque unless your load cases show otherwise."
    )


with tabs[1]:
    st.subheader("Geometry from suspension hardpoints")

    st.caption(
        "Coordinates may use any consistent length unit. Select the unit "
        "below for conversion."
    )

    geom_unit = st.selectbox(
        "Coordinate unit",
        ["mm", "m", "in"],
        index=0,
        key="geom_unit",
    )

    scale = {
        "mm": 0.001,
        "m": 1.0,
        "in": M_PER_IN,
    }[geom_unit]

    geometry_column_1, geometry_column_2, geometry_column_3 = st.columns(3)

    def point_inputs(column, title, defaults):
        with column:
            st.markdown(f"**{title}**")

            return [
                st.number_input(
                    f"{title} {axis}",
                    value=float(value),
                    key=f"{title}_{axis}",
                )
                for axis, value in zip("xyz", defaults)
            ]

    pvt = point_inputs(
        geometry_column_1,
        "ARB_PVT",
        [0, 0, 0],
    )

    dl = point_inputs(
        geometry_column_2,
        "ARB_DL",
        [0, 150, 0],
    )

    rk = point_inputs(
        geometry_column_3,
        "RK_ARB",
        [0, 150, 180],
    )

    blade_len_geom = point_distance(
        np.array(pvt) * scale,
        np.array(dl) * scale,
    )

    drop_len_geom = point_distance(
        np.array(dl) * scale,
        np.array(rk) * scale,
    )

    geometry_result_1, geometry_result_2 = st.columns(2)

    geometry_result_1.metric(
        "Blade pivot-to-tip length",
        fnum(blade_len_geom * 1000, " mm", 1),
    )

    geometry_result_2.metric(
        "Drop-link length",
        fnum(drop_len_geom * 1000, " mm", 1),
    )

    st.warning(
        "Point distances alone do not determine motion ratio. Confirm the "
        "supplied motion ratio with a suspension kinematic sweep through "
        "bump, rebound and roll."
    )


with tabs[2]:
    st.subheader("Torsion-bar / blade stiffness allocation")

    sweep_column_1, sweep_column_2, sweep_column_3 = st.columns(3)

    sweep_min = sweep_column_1.number_input(
        "Sweep start (× stiff target)",
        min_value=1.01,
        value=1.10,
        step=0.05,
        key="sweep_min",
    )

    sweep_max = sweep_column_2.number_input(
        "Sweep end (× stiff target)",
        min_value=1.02,
        value=4.00,
        step=0.25,
        key="sweep_max",
    )

    

    sweep_n = int(
        sweep_column_3.number_input(
            "Candidates",
            min_value=6,
            max_value=200,
            value=40,
            key="sweep_n",
        )
    )

    factors = np.linspace(
        sweep_min,
        max(sweep_max, sweep_min + 0.01),
        sweep_n,
    )

    rows = []

    for factor in factors:
        tube_stiffness = kmax * factor

        blade_soft_stiffness = required_blade_pair_stiffness(
            kmin,
            tube_stiffness,
        )

        blade_stiff_stiffness = required_blade_pair_stiffness(
            kmax,
            tube_stiffness,
        )

        rows.append(
            {
                "Tube factor": factor,
                "Tube k (N·m/deg)": tube_stiffness,
                "Blade pair soft": blade_soft_stiffness,
                "Blade pair stiff": blade_stiff_stiffness,
                "Blade ratio": (
                    blade_stiff_stiffness
                    / blade_soft_stiffness
                ),
                "Tube twist @ design T (deg)": tube_twist_deg(
                    torque_stiff,
                    tube_stiffness,
                ),
                "Blade soft rotation @ design T (deg)": (
                    torque_stiff / blade_soft_stiffness
                ),
                "Blade stiff rotation @ design T (deg)": (
                    torque_stiff / blade_stiff_stiffness
                ),
            }
        )

    sweep_df = pd.DataFrame(rows)

    stiffness_figure = go.Figure()

    stiffness_series = [
        ("Tube k (N·m/deg)", "Tube"),
        ("Blade pair soft", "Blade pair: soft"),
        ("Blade pair stiff", "Blade pair: stiff"),
    ]

    for y_value, name in stiffness_series:
        stiffness_figure.add_trace(
            go.Scatter(
                x=sweep_df["Tube factor"],
                y=sweep_df[y_value],
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

    slider_min = float(sweep_min)
    slider_max = float(
        max(sweep_max, sweep_min + 0.01)
    )

    default_factor = min(
        2.0,
        slider_max,
    )

    current_factor = float(
        st.session_state.get(
            "selected_factor",
            default_factor,
        )
    )

    # Keep restored or optimiser-selected values inside the slider range.
    clamped_factor = min(
        max(current_factor, slider_min),
        slider_max,
    )

    st.session_state["selected_factor"] = clamped_factor

    selected_factor = st.slider(
        "Selected tube stiffness factor",
        min_value=slider_min,
        max_value=slider_max,
        value=clamped_factor,
        step=0.01,
        key="selected_factor",
    )

    selected_kt = kmax * selected_factor

    req_kb_soft = required_blade_pair_stiffness(
        kmin,
        selected_kt,
    )

    req_kb_stiff = required_blade_pair_stiffness(
        kmax,
        selected_kt,
    )

    allocation_columns = st.columns(4)

    allocation_columns[0].metric(
        "Selected tube k",
        fnum(selected_kt, " N·m/deg", 1),
    )

    allocation_columns[1].metric(
        "Required blade pair: soft",
        fnum(req_kb_soft, " N·m/deg", 1),
    )

    allocation_columns[2].metric(
        "Required blade pair: stiff",
        fnum(req_kb_stiff, " N·m/deg", 1),
    )

    allocation_columns[3].metric(
        "Required blade ratio",
        fnum(req_kb_stiff / req_kb_soft, "×", 2),
    )

    with st.expander("Candidate table"):
        st.dataframe(
            sweep_df.style.format("{:.3f}"),
            width="stretch",
            height=320,
        )


with tabs[3]:
    st.subheader("Minimum-mass stiffness allocation")

    st.caption(
        "The search combines a hollow uniform tube with two ideal linearly "
        "tapered rectangular blades. It ranks only candidates that meet both "
        "endpoint rates and the entered structural constraints."
    )

    material_column_1, material_column_2, material_column_3 = st.columns(3)

    opt_tube_density = material_column_1.number_input(
        "Tube density (kg/m³)",
        min_value=1.0,
        value=7850.0,
        step=50.0,
        key="opt_tube_density",
    )

    opt_blade_density = material_column_2.number_input(
        "Blade density (kg/m³)",
        min_value=1.0,
        value=7850.0,
        step=50.0,
        key="opt_blade_density",
    )

    opt_rate_tol = material_column_3.number_input(
        "Endpoint rate tolerance (±%)",
        min_value=0.1,
        max_value=25.0,
        value=5.0,
        step=0.5,
        key="opt_rate_tol",
    )

    with st.expander(
        "Lengths and material properties",
        expanded=True,
    ):
        property_column_1, property_column_2, property_column_3 = (
            st.columns(3)
        )

        opt_tube_length = property_column_1.number_input(
            "Tube compliant length (mm)",
            min_value=1.0,
            value=500.0,
            step=5.0,
            key="opt_tube_length",
        )

        opt_G = property_column_1.number_input(
            "Tube shear modulus (GPa)",
            min_value=0.1,
            value=79.3,
            step=0.1,
            key="opt_G",
        )

        opt_tube_yield = property_column_2.number_input(
            "Tube shear yield (MPa)",
            min_value=1.0,
            value=460.0,
            step=10.0,
            key="opt_tube_yield",
        )

        opt_tube_fatigue = property_column_2.number_input(
            "Tube shear fatigue allowable (MPa)",
            min_value=1.0,
            value=250.0,
            step=10.0,
            key="opt_tube_fatigue",
        )

        opt_blade_length = property_column_3.number_input(
            "Blade effective length (mm)",
            min_value=1.0,
            value=max(100.0, blade_len_geom * 1000),
            step=1.0,
            key="opt_blade_length",
        )

        opt_E = property_column_3.number_input(
            "Blade Young's modulus (GPa)",
            min_value=0.1,
            value=205.0,
            step=1.0,
            key="opt_E",
        )

        opt_blade_allow = property_column_3.number_input(
            "Blade allowable stress (MPa)",
            min_value=1.0,
            value=1000.0,
            step=25.0,
            key="opt_blade_allow",
        )

    (
        limit_column_1,
        limit_column_2,
        limit_column_3,
        limit_column_4,
    ) = st.columns(4)

    opt_tube_fos = limit_column_1.number_input(
        "Minimum tube yield FoS",
        min_value=1.0,
        value=1.5,
        step=0.1,
        key="opt_tube_fos",
    )

    opt_blade_fos = limit_column_2.number_input(
        "Minimum blade nominal FoS",
        min_value=1.0,
        value=1.5,
        step=0.1,
        key="opt_blade_fos",
    )

    opt_soft_defl = limit_column_3.number_input(
        "Maximum soft deflection (mm)",
        min_value=0.01,
        value=5.08,
        step=0.1,
        key="opt_soft_defl",
    )

    opt_stiff_defl = limit_column_4.number_input(
        "Maximum stiff deflection (mm)",
        min_value=0.01,
        value=1.27,
        step=0.1,
        key="opt_stiff_defl",
    )

    with st.expander(
        "Tube search bounds",
        expanded=True,
    ):
        tube_bound_1, tube_bound_2, tube_bound_3, tube_bound_4 = (
            st.columns(4)
        )

        opt_od_min = tube_bound_1.number_input(
            "Minimum OD (mm)",
            min_value=0.1,
            value=10.0,
            step=0.5,
            key="opt_od_min",
        )

        opt_od_max = tube_bound_2.number_input(
            "Maximum OD (mm)",
            min_value=0.2,
            value=25.0,
            step=0.5,
            key="opt_od_max",
        )

        opt_wall_min = tube_bound_3.number_input(
            "Minimum wall (mm)",
            min_value=0.05,
            value=0.5,
            step=0.1,
            key="opt_wall_min",
        )

        opt_wall_max = tube_bound_4.number_input(
            "Maximum wall (mm)",
            min_value=0.1,
            value=3.0,
            step=0.1,
            key="opt_wall_max",
        )

    with st.expander(
        "Blade search bounds",
        expanded=True,
    ):
        blade_bound_1, blade_bound_2, blade_bound_3 = st.columns(3)

        opt_root_min = blade_bound_1.number_input(
            "Minimum root width (mm)",
            min_value=0.1,
            value=12.0,
            step=1.0,
            key="opt_root_min",
        )

        opt_root_max = blade_bound_1.number_input(
            "Maximum root width (mm)",
            min_value=0.2,
            value=35.0,
            step=1.0,
            key="opt_root_max",
        )

        opt_tip_min = blade_bound_2.number_input(
            "Minimum tip width (mm)",
            min_value=0.1,
            value=5.0,
            step=0.5,
            key="opt_tip_min",
        )

        opt_tip_max = blade_bound_2.number_input(
            "Maximum tip width (mm)",
            min_value=0.2,
            value=20.0,
            step=0.5,
            key="opt_tip_max",
        )

        opt_thick_min = blade_bound_3.number_input(
            "Minimum thickness (mm)",
            min_value=0.1,
            value=2.0,
            step=0.25,
            key="opt_thick_min",
        )

        opt_thick_max = blade_bound_3.number_input(
            "Maximum thickness (mm)",
            min_value=0.2,
            value=10.0,
            step=0.25,
            key="opt_thick_max",
        )

        opt_grid_n = int(
            blade_bound_3.number_input(
                "Grid points per dimension",
                min_value=4,
                max_value=20,
                value=8,
                key="opt_grid_n",
            )
        )

    invalid_bounds = (
        opt_od_max <= opt_od_min
        or opt_wall_max <= opt_wall_min
        or opt_root_max <= opt_root_min
        or opt_tip_max <= opt_tip_min
        or opt_thick_max <= opt_thick_min
    )

    if invalid_bounds:
        st.error(
            "Every maximum search bound must be greater than its minimum."
        )
        optimisation_df = pd.DataFrame()

    else:
        blade_candidates = []

        opt_blade_length_m = opt_blade_length / 1000

        force_soft = blade_tip_force(
            torque_soft,
            opt_blade_length_m,
        )

        force_stiff = blade_tip_force(
            torque_stiff,
            opt_blade_length_m,
        )

        for root_mm in np.linspace(
            opt_root_min,
            opt_root_max,
            opt_grid_n,
        ):
            for tip_mm in np.linspace(
                opt_tip_min,
                opt_tip_max,
                opt_grid_n,
            ):
                if tip_mm > root_mm:
                    continue

                for blade_t_mm in np.linspace(
                    opt_thick_min,
                    opt_thick_max,
                    opt_grid_n,
                ):
                    soft_blade = tapered_blade_analysis(
                        force_soft,
                        opt_blade_length_m,
                        root_mm / 1000,
                        tip_mm / 1000,
                        blade_t_mm / 1000,
                        opt_E * 1e9,
                        0,
                    )

                    stiff_blade = tapered_blade_analysis(
                        force_stiff,
                        opt_blade_length_m,
                        root_mm / 1000,
                        tip_mm / 1000,
                        blade_t_mm / 1000,
                        opt_E * 1e9,
                        90,
                    )

                    worst_stress = max(
                        soft_blade.max_stress_pa,
                        stiff_blade.max_stress_pa,
                    ) / 1e6

                    if (
                        worst_stress <= 0
                        or opt_blade_allow / worst_stress
                        < opt_blade_fos
                    ):
                        continue

                    if (
                        soft_blade.tip_deflection_m * 1000
                        > opt_soft_defl
                        or stiff_blade.tip_deflection_m * 1000
                        > opt_stiff_defl
                    ):
                        continue

                    blade_candidates.append(
                        {
                            "root_mm": root_mm,
                            "tip_mm": tip_mm,
                            "blade_t_mm": blade_t_mm,
                            "kb_soft": (
                                soft_blade
                                .pair_rot_stiffness_nm_per_deg
                            ),
                            "kb_stiff": (
                                stiff_blade
                                .pair_rot_stiffness_nm_per_deg
                            ),
                            "blade_pair_mass_kg": (
                                tapered_blade_pair_mass_kg(
                                    opt_blade_length_m,
                                    root_mm / 1000,
                                    tip_mm / 1000,
                                    blade_t_mm / 1000,
                                    opt_blade_density,
                                )
                            ),
                            "blade_stress_mpa": worst_stress,
                            "soft_defl_mm": (
                                soft_blade.tip_deflection_m * 1000
                            ),
                            "stiff_defl_mm": (
                                stiff_blade.tip_deflection_m * 1000
                            ),
                        }
                    )

        results = []

        if blade_candidates:
            blade_candidate_df = pd.DataFrame(blade_candidates)

            for tube_od_mm in np.linspace(
                opt_od_min,
                opt_od_max,
                opt_grid_n,
            ):
                for tube_wall_mm in np.linspace(
                    opt_wall_min,
                    opt_wall_max,
                    opt_grid_n,
                ):
                    tube_id_mm = (
                        tube_od_mm - 2 * tube_wall_mm
                    )

                    if tube_id_mm <= 0:
                        continue

                    tube_stiffness = tube_stiffness_nm_per_deg(
                        tube_od_mm / 1000,
                        tube_id_mm / 1000,
                        opt_tube_length / 1000,
                        opt_G * 1e9,
                    )

                    if tube_stiffness <= kmax:
                        continue

                    tube_stress = (
                        tube_shear_stress_pa(
                            torque_stiff,
                            tube_od_mm / 1000,
                            tube_id_mm / 1000,
                        )
                        / 1e6
                    )

                    if tube_stress:
                        yield_fos = (
                            opt_tube_yield / tube_stress
                        )
                        fatigue_margin = (
                            opt_tube_fatigue / tube_stress
                        )
                    else:
                        yield_fos = np.inf
                        fatigue_margin = np.inf

                    if (
                        yield_fos < opt_tube_fos
                        or fatigue_margin < 1.0
                    ):
                        continue

                    total_soft = 1 / (
                        1 / tube_stiffness
                        + 1 / blade_candidate_df["kb_soft"]
                    )

                    total_stiff = 1 / (
                        1 / tube_stiffness
                        + 1 / blade_candidate_df["kb_stiff"]
                    )

                    soft_error = (
                        (total_soft - kmin).abs()
                        / kmin
                        * 100
                    )

                    stiff_error = (
                        (total_stiff - kmax).abs()
                        / kmax
                        * 100
                    )

                    feasible = blade_candidate_df[
                        (soft_error <= opt_rate_tol)
                        & (stiff_error <= opt_rate_tol)
                    ].copy()

                    if feasible.empty:
                        continue

                    tube_length = opt_tube_length / 1000

                    tube_mass = tube_mass_kg(
                        tube_od_mm / 1000,
                        tube_id_mm / 1000,
                        tube_length,
                        opt_tube_density,
                    )

                    feasible["total_mass_kg"] = (
                        tube_mass
                        + feasible["blade_pair_mass_kg"]
                    )

                    best = feasible.sort_values(
                        "total_mass_kg"
                    ).iloc[0]

                    results.append(
                        {
                            "Total mass (g)": (
                                best["total_mass_kg"] * 1000
                            ),
                            "Tube mass (g)": tube_mass * 1000,
                            "Blade pair mass (g)": (
                                best["blade_pair_mass_kg"]
                                * 1000
                            ),
                            "Tube OD (mm)": tube_od_mm,
                            "Tube wall (mm)": tube_wall_mm,
                            "Tube k (N·m/deg)": tube_stiffness,
                            "Tube compliance at stiff (%)": (
                                kmax / tube_stiffness * 100
                            ),
                            "Root width (mm)": best["root_mm"],
                            "Tip width (mm)": best["tip_mm"],
                            "Blade thickness (mm)": (
                                best["blade_t_mm"]
                            ),
                            "Blade ratio": (
                                best["kb_stiff"]
                                / best["kb_soft"]
                            ),
                            "Soft rate (N·m/deg)": (
                                total_soft.loc[best.name]
                            ),
                            "Stiff rate (N·m/deg)": (
                                total_stiff.loc[best.name]
                            ),
                            "Tube yield FoS": yield_fos,
                            "Tube fatigue margin": fatigue_margin,
                            "Blade nominal FoS": (
                                opt_blade_allow
                                / best["blade_stress_mpa"]
                            ),
                            "Soft deflection (mm)": (
                                best["soft_defl_mm"]
                            ),
                            "Stiff deflection (mm)": (
                                best["stiff_defl_mm"]
                            ),
                        }
                    )

        if results:
            optimisation_df = (
                pd.DataFrame(results)
                .sort_values("Total mass (g)")
                .reset_index(drop=True)
            )
        else:
            optimisation_df = pd.DataFrame()

    if optimisation_df.empty:
        st.warning(
            "No feasible design was found. Widen the geometry bounds, "
            "increase the grid density or relax only constraints that your "
            "engineering requirements allow."
        )

    else:
        best = optimisation_df.iloc[0]

        (
            optimisation_result_1,
            optimisation_result_2,
            optimisation_result_3,
            optimisation_result_4,
        ) = st.columns(4)

        optimisation_result_1.metric(
            "Lowest combined mass",
            f"{best['Total mass (g)']:.1f} g",
        )

        optimisation_result_2.metric(
            "Tube compliance share",
            (
                f"{best['Tube compliance at stiff (%)']:.1f}%"
            ),
        )

        optimisation_result_3.metric(
            "Blade ratio",
            f"{best['Blade ratio']:.2f}×",
        )

        optimisation_result_4.metric(
            "Feasible concepts",
            f"{len(optimisation_df)}",
        )

        mass_figure = px.scatter(
            optimisation_df,
            x="Tube compliance at stiff (%)",
            y="Total mass (g)",
            color="Blade ratio",
            hover_data=[
                "Tube OD (mm)",
                "Tube wall (mm)",
                "Root width (mm)",
                "Blade thickness (mm)",
            ],
        )

        mass_figure.update_layout(height=420)

        st.plotly_chart(
            mass_figure,
            width="stretch",
        )

        rank = st.number_input(
            "Candidate rank to inspect",
            min_value=1,
            max_value=len(optimisation_df),
            value=1,
            step=1,
        )

        chosen = optimisation_df.iloc[int(rank) - 1]

        st.dataframe(
            optimisation_df.head(25).style.format("{:.3f}"),
            width="stretch",
            height=420,
        )

        if st.button(
            "Apply selected candidate to manual sizing tabs",
            type="primary",
        ):
            chosen_factor = float(
                chosen["Tube k (N·m/deg)"] / kmax
            )

            st.session_state._candidate_to_apply = {
                "od_mm": float(chosen["Tube OD (mm)"]),
                "wall_mm": float(chosen["Tube wall (mm)"]),
                "root_w_mm": float(
                    chosen["Root width (mm)"]
                ),
                "tip_w_mm": float(
                    chosen["Tip width (mm)"]
                ),
                "thick_mm": float(
                    chosen["Blade thickness (mm)"]
                ),
                "selected_factor": chosen_factor,
                "sweep_min": min(
                    float(
                        st.session_state.get(
                            "sweep_min",
                            1.10,
                        )
                    ),
                    max(
                        1.01,
                        chosen_factor - 0.05,
                    ),
                ),
                "sweep_max": max(
                    float(
                        st.session_state.get(
                            "sweep_max",
                            4.0,
                        )
                    ),
                    chosen_factor + 0.10,
                ),
                "tube_len_mm": float(opt_tube_length),
                "blade_len_mm": float(opt_blade_length),
                "G_gpa": float(opt_G),
                "shear_yield_mpa": float(
                    opt_tube_yield
                ),
                "fatigue_allow_mpa": float(
                    opt_tube_fatigue
                ),
                "E_gpa": float(opt_E),
                "blade_allow_mpa": float(
                    opt_blade_allow
                ),
            }

            st.rerun()

        st.caption(
            "Reported mass covers the uniform tube and tapered rectangular "
            "blade portions only. Add inserts, adapters, welds, threaded "
            "features and local blade-root material before making the final "
            "selection."
        )


with tabs[4]:
    st.subheader("Hollow torsion-tube sizing")

    (
        tube_column_1,
        tube_column_2,
        tube_column_3,
        tube_column_4,
    ) = st.columns(4)

    od_mm = tube_column_1.number_input(
        "Outer diameter (mm)",
        min_value=0.1,
        value=15.875 if axle == "Rear" else 12.700,
        step=0.1,
        key="od_mm",
    )

    wall_mm = tube_column_2.number_input(
        "Wall thickness (mm)",
        min_value=0.05,
        value=0.711 if axle == "Rear" else 1.473,
        step=0.05,
        key="wall_mm",
    )

    tube_len_mm = tube_column_3.number_input(
        "Effective compliant length (mm)",
        min_value=1.0,
        value=500.0,
        step=5.0,
        key="tube_len_mm",
    )

    G_gpa = tube_column_4.number_input(
        "Shear modulus (GPa)",
        min_value=0.1,
        value=79.3,
        step=0.1,
        key="G_gpa",
    )

    strength_column_1, strength_column_2 = st.columns(2)

    shear_yield_mpa = strength_column_1.number_input(
        "Shear yield strength (MPa)",
        min_value=1.0,
        value=460.0,
        step=10.0,
        key="shear_yield_mpa",
    )

    fatigue_allow_mpa = strength_column_2.number_input(
        "Corrected shear fatigue allowable (MPa)",
        min_value=1.0,
        value=250.0,
        step=10.0,
        key="fatigue_allow_mpa",
    )

    outside_diameter = od_mm / 1000
    wall = wall_mm / 1000
    inside_diameter = outside_diameter - 2 * wall

    if inside_diameter <= 0:
        st.error(
            "Wall thickness must be less than half the OD."
        )
        st.stop()

    actual_kt = tube_stiffness_nm_per_deg(
        outside_diameter,
        inside_diameter,
        tube_len_mm / 1000,
        G_gpa * 1e9,
    )

    tube_stress_pa = tube_shear_stress_pa(
        torque_stiff,
        outside_diameter,
        inside_diameter,
    )

    tube_results = st.columns(5)

    tube_values = [
        ("Tube ID", inside_diameter * 1000, "mm"),
        (
            "Calculated tube k",
            actual_kt,
            "N·m/deg",
        ),
        (
            "Target tube k",
            selected_kt,
            "N·m/deg",
        ),
        (
            "Peak shear",
            tube_stress_pa / 1e6,
            "MPa",
        ),
        (
            "Yield FoS",
            shear_yield_mpa
            / (tube_stress_pa / 1e6),
            "",
        ),
    ]

    for column, (label, value, unit) in zip(
        tube_results,
        tube_values,
    ):
        column.metric(
            label,
            fnum(value, f" {unit}", 2),
        )

    tube_error = (
        100
        * (actual_kt - selected_kt)
        / selected_kt
    )

    st.progress(
        min(abs(tube_error) / 50, 1.0),
        text=(
            f"Tube stiffness error: {tube_error:+.1f}%"
        ),
    )

    if tube_stress_pa / 1e6 <= fatigue_allow_mpa:
        st.success(
            "Peak nominal shear is below the entered fatigue "
            f"allowable (margin "
            f"{fatigue_allow_mpa / (tube_stress_pa / 1e6):.2f})."
        )
    else:
        st.error(
            "Peak nominal shear exceeds the entered fatigue allowable."
        )

    st.caption(
        "Nominal pure-torsion result only. Apply stress-concentration and "
        "weld/insert effects separately."
    )


with tabs[5]:
    st.subheader("Adjustable tapered-blade model")

    (
        blade_column_1,
        blade_column_2,
        blade_column_3,
        blade_column_4,
    ) = st.columns(4)

    blade_len_mm = blade_column_1.number_input(
        "Effective blade length (mm)",
        min_value=1.0,
        value=max(100.0, blade_len_geom * 1000),
        step=1.0,
        key="blade_len_mm",
    )

    root_w_mm = blade_column_2.number_input(
        "Root width (mm)",
        min_value=0.1,
        value=22.0,
        step=0.5,
        key="root_w_mm",
    )

    tip_w_mm = blade_column_3.number_input(
        "Tip width (mm)",
        min_value=0.1,
        value=9.0,
        step=0.5,
        key="tip_w_mm",
    )

    thick_mm = blade_column_4.number_input(
        "Blade thickness (mm)",
        min_value=0.1,
        value=4.0,
        step=0.1,
        key="thick_mm",
    )

    (
        blade_data_column_1,
        blade_data_column_2,
        blade_data_column_3,
    ) = st.columns(3)

    E_gpa = blade_data_column_1.number_input(
        "Blade Young's modulus (GPa)",
        min_value=0.1,
        value=205.0,
        step=1.0,
        key="E_gpa",
    )

    blade_allow_mpa = blade_data_column_2.number_input(
        "Blade allowable stress (MPa)",
        min_value=1.0,
        value=1000.0,
        step=25.0,
        key="blade_allow_mpa",
    )

    angle_steps = int(
        blade_data_column_3.number_input(
            "Adjustment positions",
            min_value=2,
            max_value=91,
            value=7,
            key="angle_steps",
        )
    )

    blade_length_m = blade_len_mm / 1000

    blade_force = blade_tip_force(
        torque_stiff,
        blade_length_m,
    )

    angles = np.linspace(
        0,
        90,
        angle_steps,
    )

    blade_rows = []

    for angle in angles:
        blade_result = tapered_blade_analysis(
            blade_force,
            blade_length_m,
            root_w_mm / 1000,
            tip_w_mm / 1000,
            thick_mm / 1000,
            E_gpa * 1e9,
            angle,
        )

        achieved_stiffness = series_stiffness(
            actual_kt,
            blade_result.pair_rot_stiffness_nm_per_deg,
        )

        blade_rows.append(
            {
                "Angle (deg)": angle,
                "Blade pair k": (
                    blade_result
                    .pair_rot_stiffness_nm_per_deg
                ),
                "Total ARB k": achieved_stiffness,
                "Tip deflection (mm)": (
                    blade_result.tip_deflection_m
                    * 1000
                ),
                "Max nominal stress (MPa)": (
                    blade_result.max_stress_pa
                    / 1e6
                ),
                "Stress FoS": (
                    blade_allow_mpa
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

    blade_results = st.columns(5)

    blade_values = [
        ("Blade force", blade_force, "N"),
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
        blade_results,
        blade_values,
    ):
        column.metric(
            label,
            fnum(value, f" {unit}", 2),
        )

    soft_error = (
        100
        * (
            soft_result["Total ARB k"]
            - kmin
        )
        / kmin
    )

    stiff_error = (
        100
        * (
            stiff_result["Total ARB k"]
            - kmax
        )
        / kmax
    )

    st.write(
        f"Endpoint errors: soft **{soft_error:+.1f}%**, "
        f"stiff **{stiff_error:+.1f}%**"
    )

    st.dataframe(
        blade_df.style.format("{:.3f}"),
        width="stretch",
    )

    st.caption(
        "Beam-model stress is nominal. Root features, holes, transitions "
        "and contact require FEA. Blade rotation changes both stiffness "
        "and the extreme-fibre approximation."
    )


with tabs[6]:
    st.subheader("Drop-link axial yield and buckling")

    (
        drop_link_column_1,
        drop_link_column_2,
        drop_link_column_3,
        drop_link_column_4,
    ) = st.columns(4)

    dl_len = drop_link_column_1.number_input(
        "Pin-to-pin length (mm)",
        min_value=1.0,
        value=max(100.0, drop_len_geom * 1000),
        step=1.0,
        key="dl_len",
    )

    dl_d = drop_link_column_2.number_input(
        "Minimum/net diameter (mm)",
        min_value=0.1,
        value=5.0,
        step=0.1,
        key="dl_d",
    )

    dl_E = drop_link_column_3.number_input(
        "Young's modulus (GPa)",
        min_value=0.1,
        value=71.7,
        step=1.0,
        key="dl_E",
    )

    dl_yield = drop_link_column_4.number_input(
        "Yield strength (MPa)",
        min_value=1.0,
        value=500.0,
        step=10.0,
        key="dl_yield",
    )

    drop_link_results = drop_link_checks(
        blade_force,
        dl_len / 1000,
        dl_d / 1000,
        dl_E * 1e9,
        dl_yield * 1e6,
    )

    drop_link_result_columns = st.columns(4)

    drop_link_values = [
        (
            "Axial stress",
            (
                drop_link_results[
                    "axial_stress_pa"
                ]
                / 1e6
            ),
            "MPa",
        ),
        (
            "Yield FoS",
            drop_link_results["yield_fos"],
            "",
        ),
        (
            "Pinned Euler load",
            drop_link_results["euler_buckling_n"],
            "N",
        ),
        (
            "Buckling FoS",
            drop_link_results["buckling_fos"],
            "",
        ),
    ]

    for column, (label, value, unit) in zip(
        drop_link_result_columns,
        drop_link_values,
    ):
        column.metric(
            label,
            fnum(value, f" {unit}", 2),
        )

    st.caption(
        "Use the minimum threaded/root section. Euler buckling assumes an "
        "ideal straight pinned-pinned member and excludes rod-end "
        "limitations and eccentricity."
    )


with tabs[7]:
    st.subheader(
        "Manufacturing sensitivity and Monte Carlo tolerance"
    )

    st.caption(
        "Propagates OD, wall and blade-section tolerances through the "
        "analytical model."
    )

    (
        tolerance_column_1,
        tolerance_column_2,
        tolerance_column_3,
        tolerance_column_4,
    ) = st.columns(4)

    od_tol = tolerance_column_1.number_input(
        "Tube OD tolerance ± (mm)",
        min_value=0.0,
        value=0.05,
        step=0.01,
        key="od_tol",
    )

    wall_tol = tolerance_column_2.number_input(
        "Wall tolerance ± (mm)",
        min_value=0.0,
        value=0.05,
        step=0.01,
        key="wall_tol",
    )

    width_tol = tolerance_column_3.number_input(
        "Blade width tolerance ± (mm)",
        min_value=0.0,
        value=0.05,
        step=0.01,
        key="width_tol",
    )

    thick_tol = tolerance_column_4.number_input(
        "Blade thickness tolerance ± (mm)",
        min_value=0.0,
        value=0.025,
        step=0.005,
        key="thick_tol",
    )

    samples = 3000
    random_generator = np.random.default_rng(27)
    simulations = []

    for _ in range(samples):
        sampled_od = (
            od_mm
            + random_generator.uniform(
                -od_tol,
                od_tol,
            )
        ) / 1000

        sampled_wall = (
            wall_mm
            + random_generator.uniform(
                -wall_tol,
                wall_tol,
            )
        ) / 1000

        sampled_id = (
            sampled_od - 2 * sampled_wall
        )

        sampled_root = (
            root_w_mm
            + random_generator.uniform(
                -width_tol,
                width_tol,
            )
        ) / 1000

        sampled_tip = (
            tip_w_mm
            + random_generator.uniform(
                -width_tol,
                width_tol,
            )
        ) / 1000

        sampled_thickness = (
            thick_mm
            + random_generator.uniform(
                -thick_tol,
                thick_tol,
            )
        ) / 1000

        if (
            sampled_id <= 0
            or min(
                sampled_root,
                sampled_tip,
                sampled_thickness,
            )
            <= 0
        ):
            continue

        sampled_tube_stiffness = (
            tube_stiffness_nm_per_deg(
                sampled_od,
                sampled_id,
                tube_len_mm / 1000,
                G_gpa * 1e9,
            )
        )

        sampled_blade_result = tapered_blade_analysis(
            blade_force,
            blade_length_m,
            sampled_root,
            sampled_tip,
            sampled_thickness,
            E_gpa * 1e9,
            90,
        )

        simulations.append(
            series_stiffness(
                sampled_tube_stiffness,
                sampled_blade_result
                .pair_rot_stiffness_nm_per_deg,
            )
        )

    simulations = np.array(simulations)

    tolerance_figure = px.histogram(
        x=simulations,
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

    tolerance_results = st.columns(4)

    tolerance_values = [
        (
            "Mean",
            simulations.mean(),
            "N·m/deg",
        ),
        (
            "5th percentile",
            np.percentile(simulations, 5),
            "N·m/deg",
        ),
        (
            "95th percentile",
            np.percentile(simulations, 95),
            "N·m/deg",
        ),
        (
            "Spread",
            np.ptp(
                np.percentile(
                    simulations,
                    [5, 95],
                )
            ),
            "N·m/deg",
        ),
    ]

    for column, (label, value, unit) in zip(
        tolerance_results,
        tolerance_values,
    ):
        column.metric(
            label,
            fnum(value, f" {unit}", 2),
        )


with tabs[8]:
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
                    actual_kt,
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
        correlation[f"{source} difference (%)"] = (
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
        correlation.to_csv(index=False).encode(),
        f"{axle.lower()}_arb_correlation.csv",
        "text/csv",
    )

    st.caption(
        "A 5–10% stiffness agreement is a useful initial correlation goal. "
        "Investigate fixture compliance, contact, friction and boundary "
        "conditions when disagreement is larger."
    )


with tabs[9]:
    st.subheader("On-car roll-gradient correlation")

    st.write(
        "Upload a CSV containing lateral acceleration in g and chassis roll "
        "in degrees, or use the demo data."
    )

    uploaded_telemetry = st.file_uploader(
        "Telemetry CSV",
        type="csv",
    )

    if uploaded_telemetry is not None:
        telemetry = pd.read_csv(
            uploaded_telemetry
        )
    else:
        demo_lateral_acceleration = np.linspace(
            -2,
            2,
            201,
        )

        telemetry_random_generator = (
            np.random.default_rng(7)
        )

        telemetry = pd.DataFrame(
            {
                "LatAcc_g": demo_lateral_acceleration,
                "Roll_deg": (
                    0.62
                    * demo_lateral_acceleration
                    + 0.03
                    + telemetry_random_generator.normal(
                        0,
                        0.035,
                        len(
                            demo_lateral_acceleration
                        ),
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
        (
            telemetry_column_1,
            telemetry_column_2,
            telemetry_column_3,
        ) = st.columns(3)

        lateral_acceleration_column = (
            telemetry_column_1.selectbox(
                "Lateral acceleration column",
                numeric_columns,
                index=0,
            )
        )

        roll_column = (
            telemetry_column_2.selectbox(
                "Roll angle column",
                numeric_columns,
                index=min(
                    1,
                    len(numeric_columns) - 1,
                ),
            )
        )

        minimum_lateral_acceleration = (
            telemetry_column_3.number_input(
                "Minimum |ay| included (g)",
                min_value=0.0,
                value=0.3,
                step=0.1,
            )
        )

        telemetry_mask = (
            telemetry[
                lateral_acceleration_column
            ].abs()
            >= minimum_lateral_acceleration
        )

        try:
            slope, intercept, r_squared = (
                linear_regression(
                    telemetry.loc[
                        telemetry_mask,
                        lateral_acceleration_column,
                    ],
                    telemetry.loc[
                        telemetry_mask,
                        roll_column,
                    ],
                )
            )

            telemetry_figure = px.scatter(
                telemetry.loc[
                    telemetry_mask
                ],
                x=lateral_acceleration_column,
                y=roll_column,
                opacity=0.45,
            )

            x_values = np.array(
                [
                    telemetry.loc[
                        telemetry_mask,
                        lateral_acceleration_column,
                    ].min(),
                    telemetry.loc[
                        telemetry_mask,
                        lateral_acceleration_column,
                    ].max(),
                ]
            )

            telemetry_figure.add_trace(
                go.Scatter(
                    x=x_values,
                    y=slope * x_values + intercept,
                    name="Fit",
                    line=dict(width=3),
                )
            )

            telemetry_figure.update_layout(
                height=430,
            )

            st.plotly_chart(
                telemetry_figure,
                width="stretch",
            )

            telemetry_results = st.columns(3)

            telemetry_results[0].metric(
                "Measured roll gradient",
                fnum(slope, " deg/g", 3),
            )

            telemetry_results[1].metric(
                "Intercept",
                fnum(intercept, " deg", 3),
            )

            telemetry_results[2].metric(
                "R²",
                fnum(r_squared, "", 4),
            )

        except ValueError as exc:
            st.warning(str(exc))


with tabs[10]:
    st.subheader("Save, restore and export")

    summary = {
        "project": {
            "axle": axle,
            "motion_ratio_roll_per_twist": mr,
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
            "blade_length_mm": blade_len_mm,
            "drop_link_length_mm": dl_len,
        },
        "tube": {
            "od_mm": od_mm,
            "wall_mm": wall_mm,
            "length_mm": tube_len_mm,
            "calculated_k_nm_per_deg": actual_kt,
            "peak_shear_mpa": tube_stress_pa / 1e6,
        },
        "blade": {
            "root_width_mm": root_w_mm,
            "tip_width_mm": tip_w_mm,
            "thickness_mm": thick_mm,
            "force_n": blade_force,
            "predicted_soft_rate_nm_per_deg": float(
                soft_result["Total ARB k"]
            ),
            "predicted_stiff_rate_nm_per_deg": float(
                stiff_result["Total ARB k"]
            ),
            "worst_nominal_stress_mpa": float(
                blade_df[
                    "Max nominal stress (MPa)"
                ].max()
            ),
        },
        "drop_link": drop_link_results,
    }

    if not optimisation_df.empty:
        summary["mass_optimisation_best"] = {
            key: float(value)
            for key, value in (
                optimisation_df.iloc[0].items()
            )
        }

    input_values = {
        key: st.session_state[key]
        for key in INPUT_KEYS
        if key in st.session_state
    }

    design_file = {
        "format_version": 1,
        "inputs": input_values,
        "results": summary,
    }

    st.markdown("**Design persistence**")

    (
        save_column,
        download_column,
        clear_column,
    ) = st.columns(3)

    if save_column.button(
        "Save values to this Codespace",
        type="primary",
        width="stretch",
    ):
        try:
            SAVE_PATH.write_text(
                json.dumps(
                    design_file,
                    indent=2,
                ),
                encoding="utf-8",
            )

            st.success(
                "Saved. These values will restore automatically "
                "in this Codespace."
            )

        except OSError as exc:
            st.error(
                f"Could not save the design: {exc}"
            )

    download_column.download_button(
        "Download restorable design JSON",
        json.dumps(
            design_file,
            indent=2,
        ).encode(),
        f"{axle.lower()}_arb_design.json",
        "application/json",
        width="stretch",
    )

    if clear_column.button(
        "Clear Codespace save",
        width="stretch",
    ):
        try:
            SAVE_PATH.unlink(
                missing_ok=True
            )

            st.success(
                "The Codespace save was cleared. Current "
                "on-screen values are unchanged."
            )

        except OSError as exc:
            st.error(
                f"Could not clear the saved design: {exc}"
            )

    st.caption(
        "The Codespace save survives app restarts while that Codespace "
        "exists. Keep the downloaded JSON as the durable backup for rebuilt "
        "or deleted Codespaces."
    )

    st.markdown(
        "**Calculated design snapshot**"
    )

    st.json(summary)

    export_column_1, export_column_2 = (
        st.columns(2)
    )

    export_column_1.download_button(
        "Download stiffness sweep CSV",
        sweep_df.to_csv(
            index=False
        ).encode(),
        f"{axle.lower()}_stiffness_sweep.csv",
        "text/csv",
        width="stretch",
    )

    export_column_2.download_button(
        "Download blade settings CSV",
        blade_df.to_csv(
            index=False
        ).encode(),
        f"{axle.lower()}_blade_settings.csv",
        "text/csv",
        width="stretch",
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
                abs(soft_error) <= 10,
                abs(stiff_error) <= 10,
                (
                    shear_yield_mpa
                    / (tube_stress_pa / 1e6)
                    >= 1.5
                ),
                (
                    fatigue_allow_mpa
                    / (tube_stress_pa / 1e6)
                    >= 1.0
                ),
                (
                    blade_df[
                        "Stress FoS"
                    ].min()
                    >= 1.0
                ),
                (
                    drop_link_results[
                        "yield_fos"
                    ]
                    >= 1.5
                ),
                (
                    drop_link_results[
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
        "Passing this preliminary checklist does not replace detailed joint, "
        "weld, fatigue, contact, bearing, mount or chassis-interface analysis."
    )