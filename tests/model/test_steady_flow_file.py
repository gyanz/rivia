"""Tests for rivia.model.flow_steady — SteadyFlowFile.

Fixtures (real steady flow files from HEC-RAS 6.6 example projects):
  baxter.f01    — 3 profiles ("Big","Bigger","Biggest"), 3 River Rch & RM
                  entries, Dn Type=3 (normal depth) for Lower Reach,
                  Storage Area Elev= trailing section.
                  (from 1D Steady / Baxter RAS Mapper)
  conspan.f01   — 4 profiles, 1 flow location, Dn Type=1 (known WS) per
                  profile.
                  (from 1D Steady / ConSpan Culvert)
  mixed.f01     — 2 profiles ("PF 1","PF 2"), 1 flow location,
                  Up Type=3 + Dn Type=3 (normal depth both ends).
                  (from 1D Steady / Mixed Flow Regime Channel)
  wailupe.f01   — 3 profiles, old ``Version=`` header, multiple flow
                  locations, Up Type=3 upstream, Dn Type=3 downstream.
                  (from 1D Steady / Wailupe GeoRAS)
"""

import shutil
from pathlib import Path

import pytest

from rivia.model.flow_steady import SteadyBoundary, SteadyFlowFile

FIXTURES = Path(__file__).parent / "fixtures"
BAXTER = FIXTURES / "baxter.f01"
CONSPAN = FIXTURES / "conspan.f01"
MIXED = FIXTURES / "mixed.f01"
WAILUPE = FIXTURES / "wailupe.f01"


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
            SteadyFlowFile(tmp_path / "missing.f01")

    def test_accepts_str_path(self):
        f = SteadyFlowFile(str(BAXTER))
        assert f.flow_title is not None

    def test_accepts_path_object(self):
        f = SteadyFlowFile(BAXTER)
        assert f.flow_title is not None


# ---------------------------------------------------------------------------
# No-op roundtrip — must produce byte-identical output
# ---------------------------------------------------------------------------


class TestVerbatimRoundtrip:
    @pytest.mark.parametrize("fixture", [BAXTER, CONSPAN, MIXED, WAILUPE])
    def test_noop_save_is_byte_identical(self, fixture, tmp_copy):
        original = fixture.read_bytes()
        dst = tmp_copy(fixture)
        SteadyFlowFile(dst).save()
        assert dst.read_bytes() == original, f"Roundtrip failed for {fixture.name}"


# ---------------------------------------------------------------------------
# Scalar properties
# ---------------------------------------------------------------------------


class TestScalarProperties:
    def test_flow_title_baxter(self):
        assert SteadyFlowFile(BAXTER).flow_title == "Steady Flows"

    def test_flow_title_mixed(self):
        assert SteadyFlowFile(MIXED).flow_title == "Flow data with two profiles"

    def test_program_version_modern(self):
        assert SteadyFlowFile(BAXTER).program_version == "6.30"

    def test_program_version_legacy_version_key(self):
        # WAILUPE uses "Version=Version 3.0 Beta" not "Program Version="
        v = SteadyFlowFile(WAILUPE).program_version
        assert v is not None
        assert "3.0" in v

    def test_n_profiles_baxter(self):
        assert SteadyFlowFile(BAXTER).n_profiles == 3

    def test_n_profiles_conspan(self):
        assert SteadyFlowFile(CONSPAN).n_profiles == 4

    def test_n_profiles_mixed(self):
        assert SteadyFlowFile(MIXED).n_profiles == 2

    def test_profile_names_baxter(self):
        assert SteadyFlowFile(BAXTER).profile_names == ["Big", "Bigger", "Biggest"]

    def test_profile_names_conspan(self):
        names = SteadyFlowFile(CONSPAN).profile_names
        assert len(names) == 4
        assert names[0].strip() == "5 yr"

    def test_profile_names_mixed(self):
        assert SteadyFlowFile(MIXED).profile_names == ["PF 1", "PF 2"]

    def test_set_flow_title(self, tmp_copy):
        dst = tmp_copy(BAXTER)
        f = SteadyFlowFile(dst)
        f.flow_title = "Modified Title"
        f.save()
        assert SteadyFlowFile(dst).flow_title == "Modified Title"

    def test_set_profile_names(self, tmp_copy):
        dst = tmp_copy(BAXTER)
        f = SteadyFlowFile(dst)
        f.profile_names = ["Q2", "Q10", "Q100"]
        f.save()
        assert SteadyFlowFile(dst).profile_names == ["Q2", "Q10", "Q100"]

    def test_set_n_profiles(self, tmp_copy):
        dst = tmp_copy(BAXTER)
        f = SteadyFlowFile(dst)
        f.n_profiles = 5
        f.save()
        assert SteadyFlowFile(dst).n_profiles == 5


# ---------------------------------------------------------------------------
# get_flows
# ---------------------------------------------------------------------------


class TestGetFlows:
    def test_baxter_upper_reach_count(self):
        vals = SteadyFlowFile(BAXTER).get_flows("Baxter River", "Upper Reach", "84816.")
        assert len(vals) == 3

    def test_baxter_upper_reach_values(self):
        vals = SteadyFlowFile(BAXTER).get_flows("Baxter River", "Upper Reach", "84816.")
        assert vals[0] == pytest.approx(31500.0)
        assert vals[1] == pytest.approx(67499.99)
        assert vals[2] == pytest.approx(149400.0)

    def test_tule_creek_first_value(self):
        vals = SteadyFlowFile(BAXTER).get_flows("Tule Creek", "Tributary", "10982.")
        assert vals[0] == pytest.approx(499.9999)

    def test_conspan_four_profiles(self):
        vals = SteadyFlowFile(CONSPAN).get_flows(
            "Spring Creek", "Culvrt Reach", "20.535"
        )
        assert len(vals) == 4
        assert vals[0] == pytest.approx(250.0)
        assert vals[3] == pytest.approx(1000.0)

    def test_missing_location_returns_none(self):
        assert SteadyFlowFile(BAXTER).get_flows("No River", "No Reach", "0") is None

    def test_wailupe_multiple_locations_for_same_reach(self):
        f = SteadyFlowFile(WAILUPE)
        v1 = f.get_flows("Wailupe", "lower", "1.18")
        v2 = f.get_flows("Wailupe", "lower", "0.04")
        assert v1 is not None
        assert v2 is not None
        assert v1 != v2


# ---------------------------------------------------------------------------
# set_flows
# ---------------------------------------------------------------------------


class TestSetFlows:
    def test_set_same_count(self, tmp_copy):
        dst = tmp_copy(BAXTER)
        f = SteadyFlowFile(dst)
        new_vals = [1000.0, 2000.0, 3000.0]
        f.set_flows("Baxter River", "Upper Reach", "84816.", new_vals)
        f.save()
        result = SteadyFlowFile(dst).get_flows("Baxter River", "Upper Reach", "84816.")
        assert result == pytest.approx(new_vals, rel=1e-5)

    def test_set_does_not_affect_other_location(self, tmp_copy):
        dst = tmp_copy(BAXTER)
        f = SteadyFlowFile(dst)
        before = f.get_flows("Tule Creek", "Tributary", "10982.")
        f.set_flows("Baxter River", "Upper Reach", "84816.", [0.0, 0.0, 0.0])
        f.save()
        after = SteadyFlowFile(dst).get_flows("Tule Creek", "Tributary", "10982.")
        assert before == pytest.approx(after, rel=1e-5)

    def test_set_missing_location_raises(self, tmp_copy):
        dst = tmp_copy(BAXTER)
        f = SteadyFlowFile(dst)
        with pytest.raises(KeyError):
            f.set_flows("No River", "No Reach", "0", [100.0])

    def test_set_preserves_trailing_dss_section(self, tmp_copy):
        dst = tmp_copy(BAXTER)
        f = SteadyFlowFile(dst)
        f.set_flows("Baxter River", "Upper Reach", "84816.", [1.0, 2.0, 3.0])
        f.save()
        content = dst.read_text(encoding="utf-8")
        assert "DSS Import StartDate=" in content
        assert "Storage Area Elev=" in content

    def test_set_conspan_four_profiles(self, tmp_copy):
        dst = tmp_copy(CONSPAN)
        f = SteadyFlowFile(dst)
        new_vals = [100.0, 200.0, 300.0, 400.0]
        f.set_flows("Spring Creek", "Culvrt Reach", "20.535", new_vals)
        f.save()
        result = SteadyFlowFile(dst).get_flows("Spring Creek", "Culvrt Reach", "20.535")
        assert result == pytest.approx(new_vals, rel=1e-5)


# ---------------------------------------------------------------------------
# get_boundary
# ---------------------------------------------------------------------------


class TestGetBoundary:
    def test_baxter_upper_up_type_none(self):
        bc = SteadyFlowFile(BAXTER).get_boundary("Baxter River", "Upper Reach", 1)
        assert bc is not None
        assert bc.up_type == 0

    def test_baxter_upper_dn_type_none(self):
        bc = SteadyFlowFile(BAXTER).get_boundary("Baxter River", "Upper Reach", 1)
        assert bc.dn_type == 0

    def test_baxter_lower_dn_type_normal_depth(self):
        bc = SteadyFlowFile(BAXTER).get_boundary("Baxter River", "Lower Reach", 1)
        assert bc is not None
        assert bc.dn_type == 3

    def test_baxter_lower_dn_slope(self):
        bc = SteadyFlowFile(BAXTER).get_boundary("Baxter River", "Lower Reach", 1)
        assert bc.dn_slope == pytest.approx(0.001)

    def test_baxter_all_profiles_same_slope(self):
        f = SteadyFlowFile(BAXTER)
        for p in (1, 2, 3):
            bc = f.get_boundary("Baxter River", "Lower Reach", p)
            assert bc.dn_slope == pytest.approx(0.001)

    def test_conspan_known_ws_profile_1(self):
        bc = SteadyFlowFile(CONSPAN).get_boundary("Spring Creek", "Culvrt Reach", 1)
        assert bc is not None
        assert bc.dn_type == 1
        assert bc.dn_known_ws == pytest.approx(28.9)

    def test_conspan_known_ws_profile_4(self):
        bc = SteadyFlowFile(CONSPAN).get_boundary("Spring Creek", "Culvrt Reach", 4)
        assert bc.dn_known_ws == pytest.approx(31.5)

    def test_mixed_up_type_normal_depth(self):
        bc = SteadyFlowFile(MIXED).get_boundary("Mixed Reach", "Mixed Reach", 1)
        assert bc.up_type == 3
        assert bc.up_slope == pytest.approx(0.01)

    def test_mixed_dn_type_normal_depth(self):
        bc = SteadyFlowFile(MIXED).get_boundary("Mixed Reach", "Mixed Reach", 1)
        assert bc.dn_type == 3
        assert bc.dn_slope == pytest.approx(0.000586)

    def test_missing_boundary_returns_none(self):
        assert SteadyFlowFile(BAXTER).get_boundary("No River", "No Reach", 1) is None

    def test_missing_profile_number_returns_none(self):
        assert (
            SteadyFlowFile(BAXTER).get_boundary("Baxter River", "Lower Reach", 99)
            is None
        )

    def test_profile_field_set_correctly(self):
        bc = SteadyFlowFile(BAXTER).get_boundary("Baxter River", "Lower Reach", 2)
        assert bc.profile == 2


# ---------------------------------------------------------------------------
# get_boundaries (all profiles for a reach)
# ---------------------------------------------------------------------------


class TestGetBoundaries:
    def test_baxter_lower_returns_three_profiles(self):
        bcs = SteadyFlowFile(BAXTER).get_boundaries("Baxter River", "Lower Reach")
        assert len(bcs) == 3

    def test_sorted_by_profile(self):
        bcs = SteadyFlowFile(BAXTER).get_boundaries("Baxter River", "Lower Reach")
        assert [bc.profile for bc in bcs] == [1, 2, 3]

    def test_conspan_four_profiles(self):
        bcs = SteadyFlowFile(CONSPAN).get_boundaries("Spring Creek", "Culvrt Reach")
        assert len(bcs) == 4

    def test_missing_reach_returns_empty(self):
        assert SteadyFlowFile(BAXTER).get_boundaries("No River", "No Reach") == []


# ---------------------------------------------------------------------------
# set_boundary
# ---------------------------------------------------------------------------


class TestSetBoundary:
    def test_set_dn_slope(self, tmp_copy):
        dst = tmp_copy(BAXTER)
        f = SteadyFlowFile(dst)
        bc = f.get_boundary("Baxter River", "Lower Reach", 1)
        bc.dn_slope = 0.005
        f.set_boundary(bc)
        f.save()
        result = SteadyFlowFile(dst).get_boundary("Baxter River", "Lower Reach", 1)
        assert result.dn_slope == pytest.approx(0.005)

    def test_set_changes_only_target_profile(self, tmp_copy):
        dst = tmp_copy(BAXTER)
        f = SteadyFlowFile(dst)
        bc1 = f.get_boundary("Baxter River", "Lower Reach", 1)
        bc1.dn_slope = 0.009
        f.set_boundary(bc1)
        f.save()
        # Profile 2 and 3 should be unchanged
        bc2 = SteadyFlowFile(dst).get_boundary("Baxter River", "Lower Reach", 2)
        bc3 = SteadyFlowFile(dst).get_boundary("Baxter River", "Lower Reach", 3)
        assert bc2.dn_slope == pytest.approx(0.001)
        assert bc3.dn_slope == pytest.approx(0.001)

    def test_change_bc_type_from_known_ws_to_normal_depth(self, tmp_copy):
        dst = tmp_copy(CONSPAN)
        f = SteadyFlowFile(dst)
        bc = f.get_boundary("Spring Creek", "Culvrt Reach", 1)
        bc.dn_type = 3
        bc.dn_slope = 0.002
        bc.dn_known_ws = None
        f.set_boundary(bc)
        f.save()
        result = SteadyFlowFile(dst).get_boundary("Spring Creek", "Culvrt Reach", 1)
        assert result.dn_type == 3
        assert result.dn_slope == pytest.approx(0.002)
        assert result.dn_known_ws is None

    def test_set_does_not_affect_flows(self, tmp_copy):
        dst = tmp_copy(BAXTER)
        f = SteadyFlowFile(dst)
        before = f.get_flows("Baxter River", "Upper Reach", "84816.")
        bc = f.get_boundary("Baxter River", "Lower Reach", 1)
        bc.dn_slope = 0.005
        f.set_boundary(bc)
        f.save()
        after = SteadyFlowFile(dst).get_flows("Baxter River", "Upper Reach", "84816.")
        assert before == pytest.approx(after, rel=1e-5)

    def test_set_missing_boundary_raises(self, tmp_copy):
        dst = tmp_copy(BAXTER)
        f = SteadyFlowFile(dst)
        bc = SteadyBoundary(river="No River", reach="No Reach", profile=1)
        with pytest.raises(KeyError):
            f.set_boundary(bc)

    def test_set_preserves_other_boundaries_in_file(self, tmp_copy):
        dst = tmp_copy(BAXTER)
        f = SteadyFlowFile(dst)
        bc = f.get_boundary("Baxter River", "Lower Reach", 1)
        bc.dn_slope = 0.007
        f.set_boundary(bc)
        f.save()
        # Tule Creek boundary (dn_type=0) should be untouched
        tc = SteadyFlowFile(dst).get_boundary("Tule Creek", "Tributary", 1)
        assert tc.dn_type == 0

    def test_set_known_ws(self, tmp_copy):
        dst = tmp_copy(CONSPAN)
        f = SteadyFlowFile(dst)
        bc = f.get_boundary("Spring Creek", "Culvrt Reach", 2)
        bc.dn_known_ws = 99.9
        f.set_boundary(bc)
        f.save()
        result = SteadyFlowFile(dst).get_boundary("Spring Creek", "Culvrt Reach", 2)
        assert result.dn_known_ws == pytest.approx(99.9)


# ---------------------------------------------------------------------------
# Generic escape hatch (get / set)
# ---------------------------------------------------------------------------


class TestEscapeHatch:
    def test_get_existing_key(self):
        assert SteadyFlowFile(BAXTER).get("Flow Title") == "Steady Flows"

    def test_get_missing_key_returns_none(self):
        assert SteadyFlowFile(BAXTER).get("Nonexistent Key") is None

    def test_set_existing_key(self, tmp_copy):
        dst = tmp_copy(BAXTER)
        f = SteadyFlowFile(dst)
        f.set("Flow Title", "New Title")
        f.save()
        assert SteadyFlowFile(dst).get("Flow Title") == "New Title"

    def test_set_missing_key_raises(self, tmp_copy):
        dst = tmp_copy(BAXTER)
        f = SteadyFlowFile(dst)
        with pytest.raises(KeyError):
            f.set("Nonexistent Key", "value")
