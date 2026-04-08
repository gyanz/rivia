"""Tests for rivia.model.unsteady_flow — UnsteadyFlow and UnsteadyFlow.

Fixtures (real unsteady flow files from HEC-RAS 6.6 example projects):
  baxter_1d.u01       — 2 flow hydrographs, 1 friction slope, 2 lateral inflows
                        (from 1D Steady Flow / Baxter RAS Mapper example)
  baldeagle_1d.u02    — 1 flow hydrograph, 1 gate boundary, 1 rating-curve BC
                        (from 1D Unsteady / Balde Eagle Creek example)
  dambrk.u01          — 6-field Boundary Location (pre-v5 format), 1 flow hyd,
                        1 gate, 4 lateral inflows, 1 friction slope
                        (from 1D Unsteady / Dam Breaching example)
  dambrk_dss.u02      — same geometry, Flow Hydrograph= 0 (all data from DSS)
  inline_3gates.u01   — "Version=" (pre-v4), 1 flow hyd, 1 gate boundary with
                        3 named gates inside
                        (from 1D Unsteady / Inline Structure with Gated Spillways)
"""

import shutil
from pathlib import Path

import pytest

from rivia.model.unsteady_flow import (
    FrictionSlope,
    GateBoundary,
    LateralInflow,
    FlowHydrograph,
    RatingCurve,
    UnsteadyFlow,
)

FIXTURES = Path(__file__).parent / "fixtures"
BAXTER = FIXTURES / "baxter_1d.u01"
BALDEAGLE = FIXTURES / "baldeagle_1d.u02"
DAMBRK = FIXTURES / "dambrk.u01"
DAMBRK_DSS = FIXTURES / "dambrk_dss.u02"
INLINE_3G = FIXTURES / "inline_3gates.u01"


@pytest.fixture()
def tmp_copy(tmp_path: Path):
    """Factory: copy a fixture to tmp_path and return the copy path."""

    def _copy(src: Path) -> Path:
        dst = tmp_path / src.name
        shutil.copy(src, dst)
        return dst

    return _copy


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            UnsteadyFlow(tmp_path / "missing.u01")

    def test_accepts_str_path(self):
        f = UnsteadyFlow(str(BAXTER))
        assert f.flow_title is not None


# ---------------------------------------------------------------------------
# Scalar properties — UnsteadyFlow
# ---------------------------------------------------------------------------


class TestVerbatimProperties:
    def test_flow_title_baxter(self):
        assert UnsteadyFlow(BAXTER).flow_title == "Flood Event"

    def test_program_version_baxter(self):
        assert UnsteadyFlow(BAXTER).program_version == "6.30"

    def test_restart_default(self):
        assert UnsteadyFlow(BAXTER).restart == (0, None)

    def test_flow_title_inline_3gates(self):
        assert UnsteadyFlow(INLINE_3G).flow_title == "Unsteady Flow Hydrograph"

    def test_program_version_old_format_is_none(self):
        # inline_3gates uses "Version=" not "Program Version="
        assert UnsteadyFlow(INLINE_3G).program_version is None

    def test_set_flow_title(self, tmp_copy):
        dst = tmp_copy(BAXTER)
        f = UnsteadyFlow(dst)
        f.flow_title = "Modified Title"
        f.save()
        assert UnsteadyFlow(dst).flow_title == "Modified Title"

    def test_set_restart_flag_true(self, tmp_copy):
        dst = tmp_copy(BAXTER)
        f = UnsteadyFlow(dst)
        f.restart = True
        f.save()
        assert UnsteadyFlow(dst).restart == (1, None)

    def test_set_restart_flag_nonzero_int(self, tmp_copy):
        dst = tmp_copy(BAXTER)
        f = UnsteadyFlow(dst)
        f.restart = -1
        f.save()
        assert UnsteadyFlow(dst).restart == (1, None)

    def test_set_restart_filename(self, tmp_copy):
        dst = tmp_copy(BAXTER)
        f = UnsteadyFlow(dst)
        f.restart = "Baxter.p01.01JAN2000 0000.rst"
        f.save()
        flag, filename = UnsteadyFlow(dst).restart
        assert flag == 1
        assert filename == "Baxter.p01.01JAN2000 0000.rst"

    def test_set_restart_none(self, tmp_copy):
        dst = tmp_copy(BAXTER)
        f = UnsteadyFlow(dst)
        f.restart = None
        f.save()
        assert UnsteadyFlow(dst).restart == (0, None)


# ---------------------------------------------------------------------------
# get_flow_hydrograph — UnsteadyFlow
# ---------------------------------------------------------------------------


class TestVerbatimGetFlowHydrograph:
    def test_returns_correct_count(self):
        f = UnsteadyFlow(BAXTER)
        vals = f.get_flow_hydrograph("Baxter River", "Upper Reach", "84816.")
        assert len(vals) == 100

    def test_first_value(self):
        f = UnsteadyFlow(BAXTER)
        vals = f.get_flow_hydrograph("Baxter River", "Upper Reach", "84816.")
        assert vals[0] == pytest.approx(2000.0)

    def test_second_boundary(self):
        f = UnsteadyFlow(BAXTER)
        vals = f.get_flow_hydrograph("Tule Creek", "Tributary", "10982.")
        assert vals[0] == pytest.approx(2012.33)

    def test_missing_boundary_returns_none(self):
        f = UnsteadyFlow(BAXTER)
        assert f.get_flow_hydrograph("No River", "No Reach", "0") is None

    def test_wrong_bc_type_returns_none(self):
        # Lower Reach / 1192. is a friction slope, not a flow hydrograph
        f = UnsteadyFlow(BAXTER)
        assert f.get_flow_hydrograph("Baxter River", "Lower Reach", "1192.") is None

    def test_dss_zero_count_returns_empty_list(self):
        # dambrk_dss has Flow Hydrograph= 0
        f = UnsteadyFlow(DAMBRK_DSS)
        vals = f.get_flow_hydrograph("Bald Eagle Cr.", "Lock Haven", "137520")
        assert vals == []


# ---------------------------------------------------------------------------
# set_flow_hydrograph — UnsteadyFlow
# ---------------------------------------------------------------------------


class TestVerbatimSetFlowHydrograph:
    def test_set_list_same_count(self, tmp_copy):
        dst = tmp_copy(BAXTER)
        f = UnsteadyFlow(dst)
        orig = f.get_flow_hydrograph("Baxter River", "Upper Reach", "84816.")
        new_vals = [v * 2.0 for v in orig]
        f.set_flow_hydrograph_at("Baxter River", "Upper Reach", "84816.", new_vals)
        f.save()
        result = UnsteadyFlow(dst).get_flow_hydrograph(
            "Baxter River", "Upper Reach", "84816."
        )
        assert result == pytest.approx(new_vals, rel=1e-5)

    def test_set_list_different_count(self, tmp_copy):
        dst = tmp_copy(BAXTER)
        f = UnsteadyFlow(dst)
        new_vals = [500.0, 600.0, 700.0]  # 3 instead of 100
        f.set_flow_hydrograph_at("Baxter River", "Upper Reach", "84816.", new_vals)
        f.save()
        result = UnsteadyFlow(dst).get_flow_hydrograph(
            "Baxter River", "Upper Reach", "84816."
        )
        assert result == pytest.approx(new_vals, rel=1e-5)

    def test_set_scalar_broadcasts_to_original_length(self, tmp_copy):
        dst = tmp_copy(BAXTER)
        f = UnsteadyFlow(dst)
        f.set_flow_hydrograph_at("Baxter River", "Upper Reach", "84816.", 999.0)
        f.save()
        result = UnsteadyFlow(dst).get_flow_hydrograph(
            "Baxter River", "Upper Reach", "84816."
        )
        assert len(result) == 100
        assert all(v == pytest.approx(999.0) for v in result)

    def test_set_integer_scalar(self, tmp_copy):
        dst = tmp_copy(BAXTER)
        f = UnsteadyFlow(dst)
        f.set_flow_hydrograph_at("Tule Creek", "Tributary", "10982.", 1234)
        f.save()
        result = UnsteadyFlow(dst).get_flow_hydrograph(
            "Tule Creek", "Tributary", "10982."
        )
        assert len(result) == 100
        assert result[0] == pytest.approx(1234.0)

    def test_set_does_not_affect_other_boundary(self, tmp_copy):
        dst = tmp_copy(BAXTER)
        f = UnsteadyFlow(dst)
        before_tule = f.get_flow_hydrograph("Tule Creek", "Tributary", "10982.")
        f.set_flow_hydrograph_at("Baxter River", "Upper Reach", "84816.", 0.0)
        f.save()
        after_tule = UnsteadyFlow(dst).get_flow_hydrograph(
            "Tule Creek", "Tributary", "10982."
        )
        assert before_tule == pytest.approx(after_tule, rel=1e-5)

    def test_missing_boundary_raises_key_error(self, tmp_copy):
        dst = tmp_copy(BAXTER)
        f = UnsteadyFlow(dst)
        with pytest.raises(KeyError):
            f.set_flow_hydrograph_at("No River", "No Reach", "0", [1.0, 2.0])

    def test_wrong_bc_type_raises_key_error(self, tmp_copy):
        dst = tmp_copy(BAXTER)
        f = UnsteadyFlow(dst)
        with pytest.raises(KeyError):
            f.set_flow_hydrograph_at("Baxter River", "Lower Reach", "1192.", [1.0])


# ---------------------------------------------------------------------------
# get/set lateral inflow — UnsteadyFlow
# ---------------------------------------------------------------------------


class TestVerbatimLateralInflow:
    def test_get_returns_correct_count(self):
        # dambrk has 4 lateral inflows
        f = UnsteadyFlow(DAMBRK)
        vals = f.get_lateral_inflow("Bald Eagle Cr.", "Lock Haven", "28519")
        assert len(vals) == 100

    def test_get_first_value(self):
        f = UnsteadyFlow(DAMBRK)
        vals = f.get_lateral_inflow("Bald Eagle Cr.", "Lock Haven", "28519")
        assert vals[0] == pytest.approx(600.0)

    def test_set_list(self, tmp_copy):
        dst = tmp_copy(DAMBRK)
        f = UnsteadyFlow(dst)
        new_vals = [750.0] * 100
        f.set_lateral_inflow_at("Bald Eagle Cr.", "Lock Haven", "28519", new_vals)
        f.save()
        result = UnsteadyFlow(dst).get_lateral_inflow(
            "Bald Eagle Cr.", "Lock Haven", "28519"
        )
        assert result == pytest.approx(new_vals, rel=1e-5)

    def test_set_scalar_broadcasts(self, tmp_copy):
        dst = tmp_copy(DAMBRK)
        f = UnsteadyFlow(dst)
        f.set_lateral_inflow_at("Bald Eagle Cr.", "Lock Haven", "28519", 500.0)
        f.save()
        result = UnsteadyFlow(dst).get_lateral_inflow(
            "Bald Eagle Cr.", "Lock Haven", "28519"
        )
        assert len(result) == 100
        assert all(v == pytest.approx(500.0) for v in result)


# ---------------------------------------------------------------------------
# get/set gate openings — UnsteadyFlow
# ---------------------------------------------------------------------------


class TestVerbatimGateOpenings:
    def test_get_returns_correct_count(self):
        f = UnsteadyFlow(INLINE_3G)
        vals = f.get_gate_openings("Nittany River", "Weir Reach", "41.75", "Left Group")
        assert len(vals) == 100

    def test_get_first_value(self):
        f = UnsteadyFlow(INLINE_3G)
        vals = f.get_gate_openings("Nittany River", "Weir Reach", "41.75", "Left Group")
        assert vals[0] == pytest.approx(3.0)

    def test_get_second_gate(self):
        f = UnsteadyFlow(INLINE_3G)
        vals = f.get_gate_openings(
            "Nittany River", "Weir Reach", "41.75", "Middle Group"
        )
        assert vals[0] == pytest.approx(5.0)

    def test_set_list(self, tmp_copy):
        dst = tmp_copy(INLINE_3G)
        f = UnsteadyFlow(dst)
        new_vals = [8.5] * 100
        f.set_gate_opening_at(
            "Nittany River", "Weir Reach", "41.75", "Left Group", new_vals
        )
        f.save()
        result = UnsteadyFlow(dst).get_gate_openings(
            "Nittany River", "Weir Reach", "41.75", "Left Group"
        )
        assert result == pytest.approx(new_vals, rel=1e-5)

    def test_set_scalar_broadcasts(self, tmp_copy):
        dst = tmp_copy(INLINE_3G)
        f = UnsteadyFlow(dst)
        f.set_gate_opening_at("Nittany River", "Weir Reach", "41.75", "Middle Group", 7.0)
        f.save()
        result = UnsteadyFlow(dst).get_gate_openings(
            "Nittany River", "Weir Reach", "41.75", "Middle Group"
        )
        assert len(result) == 100
        assert all(v == pytest.approx(7.0) for v in result)

    def test_set_does_not_affect_other_gate(self, tmp_copy):
        dst = tmp_copy(INLINE_3G)
        f = UnsteadyFlow(dst)
        before = f.get_gate_openings(
            "Nittany River", "Weir Reach", "41.75", "Middle Group"
        )
        f.set_gate_opening_at("Nittany River", "Weir Reach", "41.75", "Left Group", 0.0)
        f.save()
        after = UnsteadyFlow(dst).get_gate_openings(
            "Nittany River", "Weir Reach", "41.75", "Middle Group"
        )
        assert before == pytest.approx(after, rel=1e-5)

    def test_missing_gate_name_raises(self, tmp_copy):
        dst = tmp_copy(INLINE_3G)
        f = UnsteadyFlow(dst)
        with pytest.raises(KeyError):
            f.set_gate_opening_at(
                "Nittany River", "Weir Reach", "41.75", "No Gate", [1.0]
            )


# ---------------------------------------------------------------------------
# Initial conditions — UnsteadyFlow
# ---------------------------------------------------------------------------


class TestVerbatimInitialConditions:
    def test_get_initial_flow(self):
        f = UnsteadyFlow(BAXTER)
        assert f.get_initial_flow(
            "Baxter River", "Upper Reach", "84816."
        ) == pytest.approx(2000.0)

    def test_get_initial_flow_second(self):
        f = UnsteadyFlow(BAXTER)
        assert f.get_initial_flow(
            "Baxter River", "Lower Reach", "47694."
        ) == pytest.approx(4000.0)

    def test_get_initial_flow_missing_returns_none(self):
        f = UnsteadyFlow(BAXTER)
        assert f.get_initial_flow("No River", "No Reach", "0") is None

    def test_set_initial_flow(self, tmp_copy):
        dst = tmp_copy(BAXTER)
        f = UnsteadyFlow(dst)
        f.set_initial_flow_at("Baxter River", "Upper Reach", "84816.", 9999.0)
        f.save()
        assert UnsteadyFlow(dst).get_initial_flow(
            "Baxter River", "Upper Reach", "84816."
        ) == pytest.approx(9999.0)

    def test_set_initial_flow_missing_raises(self, tmp_copy):
        dst = tmp_copy(BAXTER)
        f = UnsteadyFlow(dst)
        with pytest.raises(KeyError):
            f.set_initial_flow_at("No River", "No Reach", "0", 100.0)


# ---------------------------------------------------------------------------
# UnsteadyFlow — parsing
# ---------------------------------------------------------------------------


class TestEditorParsing:
    def test_flow_title(self):
        assert UnsteadyFlow(BAXTER).flow_title == "Flood Event"

    def test_program_version(self):
        assert UnsteadyFlow(BAXTER).program_version == "6.30"

    def test_restart(self):
        assert UnsteadyFlow(BAXTER).restart == (0, None)

    def test_program_version_old_format_is_none(self):
        assert UnsteadyFlow(INLINE_3G).program_version is None

    def test_flow_hydrograph_count_baxter(self):
        ed = UnsteadyFlow(BAXTER)
        assert len(ed.flow_hydrographs) == 2

    def test_friction_slope_count_baxter(self):
        ed = UnsteadyFlow(BAXTER)
        assert len(ed.friction_slopes) == 1

    def test_friction_slope_value(self):
        ed = UnsteadyFlow(BAXTER)
        assert ed.friction_slopes[0].slope == pytest.approx(0.001)

    def test_initial_flow_locs_baxter(self):
        ed = UnsteadyFlow(BAXTER)
        assert len(ed.initial_flow_locs) == 3

    def test_initial_flow_value(self):
        ed = UnsteadyFlow(BAXTER)
        loc = ed.initial_flow_locs[0]
        assert loc.flow == pytest.approx(2000.0)

    def test_initial_storage_elevs(self):
        ed = UnsteadyFlow(BAXTER)
        assert len(ed.initial_storage_elevs) == 2
        assert ed.initial_storage_elevs[0].elevation == pytest.approx(63.34)

    def test_flow_hydrograph_values(self):
        ed = UnsteadyFlow(BAXTER)
        fh = ed.flow_hydrographs[0]
        assert len(fh.values) == 100
        assert fh.values[0] == pytest.approx(2000.0)

    def test_gate_boundary_count_inline(self):
        ed = UnsteadyFlow(INLINE_3G)
        assert len(ed.gate_boundaries) == 1

    def test_gate_count_per_boundary(self):
        ed = UnsteadyFlow(INLINE_3G)
        assert len(ed.gate_boundaries[0].gates) == 3

    def test_gate_names(self):
        ed = UnsteadyFlow(INLINE_3G)
        gates = ed.gate_boundaries[0].gates
        assert gates[0].gate_name.strip() == "Left Group"
        assert gates[1].gate_name.strip() == "Middle Group"
        assert gates[2].gate_name.strip() == "Right Group"

    def test_gate_values_first(self):
        ed = UnsteadyFlow(INLINE_3G)
        assert ed.gate_boundaries[0].gates[0].values[0] == pytest.approx(3.0)
        assert ed.gate_boundaries[0].gates[1].values[0] == pytest.approx(5.0)

    def test_rating_curve_parsed(self):
        ed = UnsteadyFlow(BALDEAGLE)
        rc = [b for b in ed.boundaries if isinstance(b, RatingCurve)]
        assert len(rc) == 1
        assert len(rc[0].pairs) == 7
        # First pair from file: 529.2, 0
        assert rc[0].pairs[0][0] == pytest.approx(529.2)
        assert rc[0].pairs[0][1] == pytest.approx(0.0)

    def test_dss_zero_count(self):
        ed = UnsteadyFlow(DAMBRK_DSS)
        fh = ed.flow_hydrographs[0]
        assert fh.values == []
        assert fh.use_dss is True

    def test_lateral_inflows_dambrk(self):
        ed = UnsteadyFlow(DAMBRK)
        assert len(ed.lateral_inflows) == 4

    def test_lateral_inflow_counts(self):
        # dambrk: laterals at 28519 (100), 1 (200), 76865 (100), 67130 (100)
        ed = UnsteadyFlow(DAMBRK)
        counts = [len(li.values) for li in ed.lateral_inflows]
        assert counts == [100, 200, 100, 100]


# ---------------------------------------------------------------------------
# UnsteadyFlow — sorting
# ---------------------------------------------------------------------------


class TestEditorSorting:
    def test_sort_lateral_inflows_ascending(self):
        ed = UnsteadyFlow(DAMBRK)
        ed.sort_lateral_inflows(descending=False)
        rs = [float(li.river_station) for li in ed.lateral_inflows]
        assert rs == sorted(rs)

    def test_sort_lateral_inflows_descending(self):
        ed = UnsteadyFlow(DAMBRK)
        ed.sort_lateral_inflows(descending=True)
        rs = [float(li.river_station) for li in ed.lateral_inflows]
        assert rs == sorted(rs, reverse=True)

    def test_sort_preserves_total_boundary_count(self):
        ed = UnsteadyFlow(DAMBRK)
        n_before = len(ed.boundaries)
        ed.sort_lateral_inflows(descending=False)
        assert len(ed.boundaries) == n_before

    def test_sort_preserves_non_lateral_positions(self):
        """Flow hydrographs and friction slopes must stay at their original
        positions in the flat boundary list after sorting laterals."""
        ed = UnsteadyFlow(DAMBRK)
        before_types = [type(b) for b in ed.boundaries]
        before_flow_indices = [
            i for i, b in enumerate(ed.boundaries) if isinstance(b, FlowHydrograph)
        ]
        before_fs_indices = [
            i for i, b in enumerate(ed.boundaries) if isinstance(b, FrictionSlope)
        ]

        ed.sort_lateral_inflows(descending=False)

        after_flow_indices = [
            i for i, b in enumerate(ed.boundaries) if isinstance(b, FlowHydrograph)
        ]
        after_fs_indices = [
            i for i, b in enumerate(ed.boundaries) if isinstance(b, FrictionSlope)
        ]

        assert before_flow_indices == after_flow_indices
        assert before_fs_indices == after_fs_indices

    def test_sort_ascending_then_descending_restores_desc_order(self):
        ed = UnsteadyFlow(DAMBRK)
        ed.sort_lateral_inflows(descending=True)
        rs_desc = [float(li.river_station) for li in ed.lateral_inflows]
        ed.sort_lateral_inflows(descending=False)
        ed.sort_lateral_inflows(descending=True)
        rs_again = [float(li.river_station) for li in ed.lateral_inflows]
        assert rs_desc == rs_again

    def test_sort_respects_river_reach_group_order(self):
        """Group (River, Reach) order must follow first appearance, not be changed.

        Reach B appears first in the list; Reach A appears after.
        After ascending sort the order must be:
            (B, 200), (A, 100), (A, 300)   — group order B then A is preserved,
                                              RS sorted within each group.
        NOT (A, 100), (A, 300), (B, 200)   — alphabetical group ordering is wrong.
        NOT (B, 200), (A, 300), (A, 100)   — RS not sorted within group.
        """
        li_b200 = LateralInflow(river="River", reach="Reach B", river_station="200")
        li_a300 = LateralInflow(river="River", reach="Reach A", river_station="300")
        li_a100 = LateralInflow(river="River", reach="Reach A", river_station="100")
        ed = UnsteadyFlow.__new__(UnsteadyFlow)
        ed.boundaries = [li_b200, li_a300, li_a100]
        ed._header_lines = []
        ed._path = None

        ed.sort_lateral_inflows(descending=False)

        result = [(li.reach, float(li.river_station)) for li in ed.lateral_inflows]
        assert result == [("Reach B", 200.0), ("Reach A", 100.0), ("Reach A", 300.0)]


# ---------------------------------------------------------------------------
# UnsteadyFlow — set by index
# ---------------------------------------------------------------------------


class TestEditorSetByIndex:
    def test_set_flow_hydrograph_list(self):
        ed = UnsteadyFlow(BAXTER)
        new_vals = [1.0] * 100
        ed.set_flow_hydrograph(0, new_vals)
        assert ed.flow_hydrographs[0].values == pytest.approx(new_vals)

    def test_set_flow_hydrograph_scalar(self):
        ed = UnsteadyFlow(BAXTER)
        ed.set_flow_hydrograph(0, 500.0)
        vals = ed.flow_hydrographs[0].values
        assert len(vals) == 100
        assert all(v == pytest.approx(500.0) for v in vals)

    def test_set_flow_hydrograph_integer_scalar(self):
        ed = UnsteadyFlow(BAXTER)
        ed.set_flow_hydrograph(1, 300)
        vals = ed.flow_hydrographs[1].values
        assert len(vals) == 100
        assert vals[0] == pytest.approx(300.0)

    def test_set_gate_opening_list(self):
        ed = UnsteadyFlow(INLINE_3G)
        new_vals = [4.0] * 100
        ed.set_gate_opening(0, new_vals, gate_index=0)
        assert ed.gate_boundaries[0].gates[0].values == pytest.approx(new_vals)

    def test_set_gate_opening_scalar(self):
        ed = UnsteadyFlow(INLINE_3G)
        ed.set_gate_opening(0, 6.0, gate_index=1)
        vals = ed.gate_boundaries[0].gates[1].values
        assert len(vals) == 100
        assert all(v == pytest.approx(6.0) for v in vals)

    def test_set_lateral_inflow_list(self):
        ed = UnsteadyFlow(DAMBRK)
        new_vals = [800.0] * 100
        ed.set_lateral_inflow(0, new_vals)
        assert ed.lateral_inflows[0].values == pytest.approx(new_vals)

    def test_set_lateral_inflow_scalar(self):
        ed = UnsteadyFlow(DAMBRK)
        ed.set_lateral_inflow(2, 250.0)
        vals = ed.lateral_inflows[2].values
        assert len(vals) == 100
        assert all(v == pytest.approx(250.0) for v in vals)

    def test_set_after_sort_targets_correct_boundary(self):
        """After sorting, index 0 should address the lowest-RS boundary."""
        ed = UnsteadyFlow(DAMBRK)
        ed.sort_lateral_inflows(descending=False)
        lowest_rs = float(ed.lateral_inflows[0].river_station)
        ed.set_lateral_inflow(0, 111.0)

        # Verify we changed the right boundary
        assert all(v == pytest.approx(111.0) for v in ed.lateral_inflows[0].values)
        assert float(ed.lateral_inflows[0].river_station) == lowest_rs

    # set_all_lateral_inflows ------------------------------------------------

    def test_set_all_lateral_inflows_scalar_per_boundary(self):
        """One scalar per lateral inflow broadcasts to the full series length."""
        ed = UnsteadyFlow(DAMBRK)
        n = len(ed.lateral_inflows)
        scalars = [float(i * 10) for i in range(n)]
        ed.set_all_lateral_inflows(scalars)
        for i, bc in enumerate(ed.lateral_inflows):
            assert all(v == pytest.approx(scalars[i]) for v in bc.values)

    def test_set_all_lateral_inflows_list_per_boundary(self):
        """One list[float] per lateral inflow sets exact values."""
        ed = UnsteadyFlow(DAMBRK)
        series = [[float(i)] * 100 for i in range(len(ed.lateral_inflows))]
        ed.set_all_lateral_inflows(series)
        for i, bc in enumerate(ed.lateral_inflows):
            assert bc.values == pytest.approx(series[i])

    def test_set_all_lateral_inflows_partial_leaves_rest_unchanged(self):
        """If values is shorter than lateral inflow count, tail is untouched."""
        ed = UnsteadyFlow(DAMBRK)
        original_last = list(ed.lateral_inflows[-1].values)
        # Only update first boundary
        ed.set_all_lateral_inflows([999.0])
        assert all(v == pytest.approx(999.0) for v in ed.lateral_inflows[0].values)
        assert ed.lateral_inflows[-1].values == pytest.approx(original_last)

    # set_all_gate_openings --------------------------------------------------

    def test_set_all_gate_openings_scalar_per_gate(self):
        """One scalar per gate broadcasts to the full series length."""
        ed = UnsteadyFlow(INLINE_3G)
        gates = ed.gate_boundaries[0].gates
        scalars = [float(i + 1) for i in range(len(gates))]
        ed.set_all_gate_openings(scalars)
        for i, gate in enumerate(gates):
            assert all(v == pytest.approx(scalars[i]) for v in gate.values)

    def test_set_all_gate_openings_list_per_gate(self):
        """One list[float] per gate sets exact values."""
        ed = UnsteadyFlow(INLINE_3G)
        gates = ed.gate_boundaries[0].gates
        series = [[float(i)] * 100 for i in range(len(gates))]
        ed.set_all_gate_openings(series)
        for i, gate in enumerate(gates):
            assert gate.values == pytest.approx(series[i])

    def test_set_all_gate_openings_partial_leaves_rest_unchanged(self):
        """If values is shorter than gate count, remaining gates are untouched."""
        ed = UnsteadyFlow(INLINE_3G)
        gates = ed.gate_boundaries[0].gates
        original_last = list(gates[-1].values)
        # INLINE_3G has 3 gates; only update first one
        ed.set_all_gate_openings([7.0])
        assert all(v == pytest.approx(7.0) for v in gates[0].values)
        assert gates[-1].values == pytest.approx(original_last)


# ---------------------------------------------------------------------------
# UnsteadyFlow — set by location
# ---------------------------------------------------------------------------


class TestEditorSetAtLocation:
    def test_set_flow_hydrograph_at(self):
        ed = UnsteadyFlow(BAXTER)
        ed.set_flow_hydrograph_at("Baxter River", "Upper Reach", "84816.", 300.0)
        vals = ed.flow_hydrographs[0].values
        assert all(v == pytest.approx(300.0) for v in vals)

    def test_set_flow_hydrograph_at_wrong_type_raises(self):
        ed = UnsteadyFlow(BAXTER)
        with pytest.raises(KeyError):
            ed.set_flow_hydrograph_at("Baxter River", "Lower Reach", "1192.", [1.0])

    def test_set_lateral_inflow_at(self):
        ed = UnsteadyFlow(DAMBRK)
        ed.set_lateral_inflow_at("Bald Eagle Cr.", "Lock Haven", "28519", 777.0)
        filt = [b for b in ed.lateral_inflows if b.river_station.strip() == "28519"]
        assert filt
        assert all(v == pytest.approx(777.0) for v in filt[0].values)

    def test_set_gate_opening_at(self):
        ed = UnsteadyFlow(INLINE_3G)
        new_vals = [2.5] * 100
        ed.set_gate_opening_at(
            "Nittany River", "Weir Reach", "41.75", "Left Group", new_vals
        )
        gate = ed.gate_boundaries[0].gates[0]
        assert gate.values == pytest.approx(new_vals)

    def test_set_gate_opening_at_scalar(self):
        ed = UnsteadyFlow(INLINE_3G)
        ed.set_gate_opening_at(
            "Nittany River", "Weir Reach", "41.75", "Middle Group", 9.0
        )
        gate = ed.gate_boundaries[0].gates[1]
        assert all(v == pytest.approx(9.0) for v in gate.values)

    def test_set_gate_opening_at_missing_gate_raises(self):
        ed = UnsteadyFlow(INLINE_3G)
        with pytest.raises(KeyError):
            ed.set_gate_opening_at(
                "Nittany River", "Weir Reach", "41.75", "No Such Gate", [1.0]
            )


# ---------------------------------------------------------------------------
# UnsteadyFlow — semantic roundtrip (parse → save → re-parse)
# ---------------------------------------------------------------------------


class TestEditorSemanticRoundtrip:
    @pytest.mark.parametrize(
        "fixture",
        [
            BAXTER,
            DAMBRK,
            INLINE_3G,
        ],
    )
    def test_values_preserved_after_save(self, fixture, tmp_copy, tmp_path):
        ed1 = UnsteadyFlow(fixture)
        out = tmp_path / (fixture.stem + "_rt" + fixture.suffix)
        ed1.save(out)
        ed2 = UnsteadyFlow(out)

        assert ed1.flow_title == ed2.flow_title
        assert len(ed1.boundaries) == len(ed2.boundaries)

        for fh1, fh2 in zip(ed1.flow_hydrographs, ed2.flow_hydrographs):
            assert fh1.values == pytest.approx(fh2.values, rel=1e-5)

        for li1, li2 in zip(ed1.lateral_inflows, ed2.lateral_inflows):
            assert li1.values == pytest.approx(li2.values, rel=1e-5)

        for gb1, gb2 in zip(ed1.gate_boundaries, ed2.gate_boundaries):
            for g1, g2 in zip(gb1.gates, gb2.gates):
                assert g1.values == pytest.approx(g2.values, rel=1e-5)

    def test_initial_conditions_preserved(self, tmp_copy, tmp_path):
        ed1 = UnsteadyFlow(BAXTER)
        out = tmp_path / "baxter_rt.u01"
        ed1.save(out)
        ed2 = UnsteadyFlow(out)

        assert len(ed1.initial_flow_locs) == len(ed2.initial_flow_locs)
        for loc1, loc2 in zip(ed1.initial_flow_locs, ed2.initial_flow_locs):
            assert loc1.flow == pytest.approx(loc2.flow)

        assert len(ed1.initial_storage_elevs) == len(ed2.initial_storage_elevs)
        for se1, se2 in zip(ed1.initial_storage_elevs, ed2.initial_storage_elevs):
            assert se1.elevation == pytest.approx(se2.elevation)

    def test_sort_then_save_then_re_parse_preserves_order(self, tmp_path):
        """After sort + save, the re-parsed file should have the same order."""
        ed1 = UnsteadyFlow(DAMBRK)
        ed1.sort_lateral_inflows(descending=False)
        rs_before = [float(li.river_station) for li in ed1.lateral_inflows]

        out = tmp_path / "dambrk_sorted.u01"
        ed1.save(out)

        ed2 = UnsteadyFlow(out)
        rs_after = [float(li.river_station) for li in ed2.lateral_inflows]
        assert rs_before == rs_after
