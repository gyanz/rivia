"""Tests for raspy.model.plan.PlanFile.

Fixtures are real plan files copied from HEC-RAS 6.6 example projects:
  - baxter_steady.p01         Steady flow plan (Program Version=5.00)
  - baldeagle_unsteady1d.p01  1D unsteady plan (Program Version=5.00)
  - muncie_unsteady2d.p01     2D unsteady plan with breaches (Program Version=5.00)
  - muncie_unsteady2d_v510.p03 2D unsteady plan with repeated UNET D2 blocks (v5.10)
"""

import shutil
from pathlib import Path

import pytest

from raspy.model.plan import PlanFile

FIXTURES = Path(__file__).parent / "fixtures"
BAXTER = FIXTURES / "baxter_steady.p01"
BALDEAGLE = FIXTURES / "baldeagle_unsteady1d.p01"
MUNCIE_P01 = FIXTURES / "muncie_unsteady2d.p01"
MUNCIE_P03 = FIXTURES / "muncie_unsteady2d_v510.p03"
BALDEAGLE_P02 = FIXTURES / "baldeagle_unsteady1d.p02"  # no Mapping Interval line


@pytest.fixture()
def tmp_plan(tmp_path: Path):
    """Return a factory that copies a fixture to a temp dir for mutation tests."""

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
            PlanFile(tmp_path / "nonexistent.p01")

    def test_accepts_str_path(self):
        pf = PlanFile(str(BAXTER))
        assert pf.plan_title is not None


# ---------------------------------------------------------------------------
# Round-trip fidelity
# ---------------------------------------------------------------------------


class TestRoundTrip:
    @pytest.mark.parametrize("fixture", [BAXTER, BALDEAGLE, MUNCIE_P01, MUNCIE_P03])
    def test_save_produces_identical_file(self, fixture, tmp_plan):
        original = fixture.read_bytes()
        dst = tmp_plan(fixture)
        pf = PlanFile(dst)
        pf.save()
        assert dst.read_bytes() == original


# ---------------------------------------------------------------------------
# Identity / metadata — Baxter (steady)
# ---------------------------------------------------------------------------


class TestIdentityBaxter:
    def setup_method(self):
        self.pf = PlanFile(BAXTER)

    def test_plan_title(self):
        assert self.pf.plan_title == "Steady Flows"

    def test_short_id_stripped(self):
        # Raw line has heavy trailing padding; getter must strip it.
        assert self.pf.short_id == "Steady Flow"

    def test_program_version(self):
        assert self.pf.program_version == "5.00"

    def test_geom_file(self):
        assert self.pf.geom_file == "g02"

    def test_flow_file(self):
        assert self.pf.flow_file == "f01"


# ---------------------------------------------------------------------------
# Simulation date — present in all plan types
# ---------------------------------------------------------------------------


class TestSimulationDate:
    def test_steady_plan(self):
        pf = PlanFile(BAXTER)
        start, end = pf.simulation_date
        assert start == "01jan2005,0100"
        assert end == "04jan2005,2400"

    def test_unsteady_1d_plan(self):
        pf = PlanFile(BALDEAGLE)
        start, end = pf.simulation_date
        assert start == "18FEB1999,0000"
        assert end == "24FEB1999,0500"

    def test_2d_plan(self):
        pf = PlanFile(MUNCIE_P01)
        start, end = pf.simulation_date
        assert start == "02JAN1900,0000"
        assert end == "02JAN1900,2400"


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------


class TestIntervals:
    def test_computation_interval_baxter(self):
        assert PlanFile(BAXTER).computation_interval == "10MIN"

    def test_computation_interval_baldeagle(self):
        assert PlanFile(BALDEAGLE).computation_interval == "2MIN"

    def test_computation_interval_muncie(self):
        assert PlanFile(MUNCIE_P01).computation_interval == "15SEC"

    def test_output_interval_baxter(self):
        assert PlanFile(BAXTER).output_interval == "1HOUR"

    def test_instantaneous_interval_baxter(self):
        assert PlanFile(BAXTER).instantaneous_interval == "1HOUR"

    def test_mapping_interval_muncie(self):
        assert PlanFile(MUNCIE_P01).mapping_interval == "5MIN"

    def test_mapping_interval_baldeagle(self):
        assert PlanFile(BALDEAGLE).mapping_interval == "1HOUR"

    def test_mapping_interval_absent_returns_none(self):
        # BaldEagle.p02 has no Mapping Interval line.
        assert PlanFile(BALDEAGLE_P02).mapping_interval is None


# ---------------------------------------------------------------------------
# Plan type detection
# ---------------------------------------------------------------------------


class TestPlanType:
    def test_baxter_is_steady(self):
        pf = PlanFile(BAXTER)
        assert pf.is_steady is True
        assert pf.is_unsteady is False

    def test_baldeagle_is_unsteady(self):
        pf = PlanFile(BALDEAGLE)
        assert pf.is_unsteady is True
        assert pf.is_steady is False

    def test_muncie_is_unsteady(self):
        pf = PlanFile(MUNCIE_P01)
        assert pf.is_unsteady is True
        assert pf.is_steady is False


# ---------------------------------------------------------------------------
# Run flags
# ---------------------------------------------------------------------------


class TestRunFlags:
    def setup_method(self):
        self.baxter = PlanFile(BAXTER)
        self.p03 = PlanFile(MUNCIE_P03)

    def test_run_htab_enabled(self):
        # Baxter: "Run HTab= 1 "
        assert self.baxter.run_htab is True

    def test_run_htab_minus_one_is_true(self):
        # Muncie p03: "Run HTab=-1 " — negative-one also means enabled.
        assert self.p03.run_htab is True

    def test_run_sediment_false(self):
        assert self.baxter.run_sediment is False

    def test_run_wq_false(self):
        assert self.baxter.run_wq is False

    def test_run_post_process_true(self):
        assert self.baxter.run_post_process is True

    def test_run_ras_mapper_false(self):
        assert self.baxter.run_ras_mapper is False


# ---------------------------------------------------------------------------
# 1-D hydraulics
# ---------------------------------------------------------------------------


class TestHydraulics:
    def setup_method(self):
        self.pf = PlanFile(BAXTER)

    def test_theta(self):
        assert self.pf.theta == pytest.approx(1.0)

    def test_theta_warmup(self):
        assert self.pf.theta_warmup == pytest.approx(1.0)

    def test_z_tolerance(self):
        assert self.pf.z_tolerance == pytest.approx(0.02)

    def test_max_iterations(self):
        assert self.pf.max_iterations == 20


# ---------------------------------------------------------------------------
# Empty / missing values
# ---------------------------------------------------------------------------


class TestNoneValues:
    def test_empty_value_returns_none(self):
        # "UNET QTol=" has no value in any example plan.
        pf = PlanFile(BAXTER)
        assert pf.get("UNET QTol") is None

    def test_absent_key_returns_none(self):
        pf = PlanFile(BAXTER)
        assert pf.get("Nonexistent Key") is None


# ---------------------------------------------------------------------------
# Generic get / set
# ---------------------------------------------------------------------------


class TestGenericAccess:
    def test_get_known_key(self):
        pf = PlanFile(BAXTER)
        assert pf.get("Plan Title") == "Steady Flows"

    def test_get_repeated_key_returns_first(self):
        # BaldEagle has many "Stage Flow Hydrograph=" lines; get returns first.
        pf = PlanFile(BALDEAGLE)
        val = pf.get("Stage Flow Hydrograph")
        assert val is not None
        # First occurrence points to station 138154.4
        assert "138154.4" in val

    def test_set_unknown_key_raises(self, tmp_plan):
        dst = tmp_plan(BAXTER)
        pf = PlanFile(dst)
        with pytest.raises(KeyError):
            pf.set("Nonexistent Key", "value")

    def test_get_absent_key_returns_none(self):
        pf = PlanFile(BAXTER)
        assert pf.get("Nonexistent Key") is None


# ---------------------------------------------------------------------------
# Mutation — typed properties
# ---------------------------------------------------------------------------


class TestMutation:
    def test_set_plan_title(self, tmp_plan):
        dst = tmp_plan(BAXTER)
        pf = PlanFile(dst)
        pf.plan_title = "New Title"
        pf.save()
        assert PlanFile(dst).plan_title == "New Title"

    def test_set_short_id(self, tmp_plan):
        dst = tmp_plan(BAXTER)
        pf = PlanFile(dst)
        pf.short_id = "Scenario_A"
        pf.save()
        assert PlanFile(dst).short_id == "Scenario_A"

    def test_set_simulation_date(self, tmp_plan):
        dst = tmp_plan(BALDEAGLE)
        pf = PlanFile(dst)
        pf.simulation_date = ("01JAN2020,0000", "15JAN2020,2400")
        pf.save()
        start, end = PlanFile(dst).simulation_date
        assert start == "01JAN2020,0000"
        assert end == "15JAN2020,2400"

    def test_set_computation_interval(self, tmp_plan):
        dst = tmp_plan(BALDEAGLE)
        pf = PlanFile(dst)
        pf.computation_interval = "5MIN"
        pf.save()
        assert PlanFile(dst).computation_interval == "5MIN"

    def test_set_run_flag_false(self, tmp_plan):
        dst = tmp_plan(BAXTER)
        pf = PlanFile(dst)
        pf.run_htab = False
        pf.save()
        assert PlanFile(dst).run_htab is False

    def test_set_run_flag_true(self, tmp_plan):
        dst = tmp_plan(BAXTER)
        pf = PlanFile(dst)
        pf.run_sediment = True
        pf.save()
        assert PlanFile(dst).run_sediment is True

    def test_set_z_tolerance(self, tmp_plan):
        dst = tmp_plan(BAXTER)
        pf = PlanFile(dst)
        pf.z_tolerance = 0.005
        pf.save()
        assert PlanFile(dst).z_tolerance == pytest.approx(0.005)

    def test_set_max_iterations(self, tmp_plan):
        dst = tmp_plan(BAXTER)
        pf = PlanFile(dst)
        pf.max_iterations = 30
        pf.save()
        assert PlanFile(dst).max_iterations == 30

    def test_set_does_not_affect_other_lines(self, tmp_plan):
        """Changing one field must not alter any other line in the file."""
        dst = tmp_plan(BAXTER)
        original_lines = BAXTER.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        pf = PlanFile(dst)
        pf.plan_title = "Changed Title"
        pf.save()
        new_lines = dst.read_text(encoding="utf-8", errors="replace").splitlines()
        changed = [
            (i, old, new)
            for i, (old, new) in enumerate(zip(original_lines, new_lines))
            if old != new
        ]
        assert len(changed) == 1
        assert "Plan Title" in changed[0][2]

    def test_mutation_only_first_repeated_key(self, tmp_plan):
        """set() on a repeated key (Breach Loc) only touches the first occurrence."""
        dst = tmp_plan(MUNCIE_P01)
        pf = PlanFile(dst)
        pf.set("Breach Loc", "Modified           ,Value           ,0       ,False,")
        pf.save()
        lines = dst.read_text(encoding="utf-8", errors="replace").splitlines()
        breach_lines = [l for l in lines if l.startswith("Breach Loc=")]
        assert "Modified" in breach_lines[0]
        # All subsequent occurrences must be unchanged.
        for line in breach_lines[1:]:
            assert "Modified" not in line
