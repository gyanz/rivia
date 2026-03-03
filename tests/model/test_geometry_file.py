"""Tests for raspy.model.geometry — GeometryFile.

Fixtures (real HEC-RAS 6.6 geometry files):

  ex1.g01         — 2 reaches (Butte Cr./Fall River), 1 junction (Sutter),
                    3 XS each reach, simple Manning's n, bank stations.
                    (from 1D Steady / Chapter 4 Example Data)

  conspan.g01     — 1 reach (Spring Creek/Culvrt Reach), 10 XS, 1 culvert
                    node (type 2), ineffective areas on approach XS,
                    XS HTab data.
                    (from 1D Steady / ConSpan Culvert)

  beaver.g01      — 1 reach (Beaver Creek/Kentwood), many XS, 1 bridge node
                    (type 3) at RS 5.4, piers.
                    (from 1D Unsteady / Bridge Hydraulics)

  nit_inline.g01  — 1 reach (Nittany River/Weir Reach), many XS, 1 inline
                    structure node (type 5) with 3 gate groups.
                    (from Applications Guide / Example 12)
"""

import shutil
from pathlib import Path

import pytest

from raspy.model.geometry import (
    CrossSection,
    GeometryFile,
    IneffArea,
    ManningEntry,
    NODE_BRIDGE,
    NODE_CULVERT,
    NODE_INLINE_STRUCTURE,
    NODE_XS,
)

FIXTURES = Path(__file__).parent / "fixtures"
EX1 = FIXTURES / "ex1.g01"
CONSPAN = FIXTURES / "conspan.g01"
BEAVER = FIXTURES / "beaver.g01"
NIT = FIXTURES / "nit_inline.g01"


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
            GeometryFile(tmp_path / "missing.g01")

    def test_accepts_str_path(self):
        g = GeometryFile(str(EX1))
        assert g.geom_title is not None

    def test_accepts_path_object(self):
        g = GeometryFile(EX1)
        assert g.geom_title is not None


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_geom_title_ex1(self):
        g = GeometryFile(EX1)
        assert g.geom_title == "Base Geometry Data"

    def test_program_version_ex1(self):
        g = GeometryFile(EX1)
        assert g.program_version == "4.00"

    def test_geom_title_conspan(self):
        g = GeometryFile(CONSPAN)
        assert g.geom_title == "ConSpan Culvert Geometry"

    def test_program_version_conspan(self):
        g = GeometryFile(CONSPAN)
        assert g.program_version == "5.00"

    def test_geom_title_setter(self, tmp_copy):
        path = tmp_copy(EX1)
        g = GeometryFile(path)
        g.geom_title = "Modified Title"
        g.save()
        g2 = GeometryFile(path)
        assert g2.geom_title == "Modified Title"


# ---------------------------------------------------------------------------
# Reaches and junctions
# ---------------------------------------------------------------------------


class TestReachesAndJunctions:
    def test_ex1_reaches(self):
        g = GeometryFile(EX1)
        reaches = g.reaches
        # EX1 has Butte Cr./Tributary and Fall River/Upper+Lower Reach
        assert len(reaches) == 3
        rivers = {r for r, _ in reaches}
        assert "Butte Cr." in rivers
        assert "Fall River" in rivers

    def test_ex1_junctions(self):
        g = GeometryFile(EX1)
        juncts = g.junctions
        assert len(juncts) == 1
        assert "Sutter" in juncts[0]

    def test_conspan_reaches(self):
        g = GeometryFile(CONSPAN)
        reaches = g.reaches
        assert len(reaches) == 1
        assert reaches[0][0] == "Spring Creek"
        assert reaches[0][1] == "Culvrt Reach"

    def test_conspan_no_junctions(self):
        g = GeometryFile(CONSPAN)
        assert g.junctions == []

    def test_nit_reaches(self):
        g = GeometryFile(NIT)
        reaches = g.reaches
        assert len(reaches) == 1
        assert reaches[0][0] == "Nittany River"


# ---------------------------------------------------------------------------
# Node inventory
# ---------------------------------------------------------------------------


class TestNodeInventory:
    def test_ex1_node_count_butte(self):
        g = GeometryFile(EX1)
        nodes = g.node_rs_list("Butte Cr.", "Tributary")
        # EX1 Butte Cr has 3 XS
        assert len(nodes) == 3
        assert all(ntype == NODE_XS for ntype, _ in nodes)

    def test_conspan_has_culvert_node(self):
        g = GeometryFile(CONSPAN)
        nodes = g.node_rs_list("Spring Creek", "Culvrt Reach")
        types = [ntype for ntype, _ in nodes]
        assert NODE_CULVERT in types

    def test_beaver_has_bridge_node(self):
        g = GeometryFile(BEAVER)
        nodes = g.node_rs_list("Beaver Creek", "Kentwood")
        types = [ntype for ntype, _ in nodes]
        assert NODE_BRIDGE in types

    def test_nit_has_inline_structure_node(self):
        g = GeometryFile(NIT)
        nodes = g.node_rs_list("Nittany River", "Weir Reach")
        types = [ntype for ntype, _ in nodes]
        assert NODE_INLINE_STRUCTURE in types

    def test_node_type_xs(self):
        g = GeometryFile(EX1)
        rs = g.node_rs_list("Butte Cr.", "Tributary")[0][1]
        assert g.node_type("Butte Cr.", "Tributary", rs) == NODE_XS

    def test_node_type_bridge(self):
        g = GeometryFile(BEAVER)
        nodes = g.node_rs_list("Beaver Creek", "Kentwood")
        bridge_rs = next(rs for ntype, rs in nodes if ntype == NODE_BRIDGE)
        assert g.node_type("Beaver Creek", "Kentwood", bridge_rs) == NODE_BRIDGE

    def test_node_type_missing_returns_none(self):
        g = GeometryFile(EX1)
        assert g.node_type("Butte Cr.", "Tributary", "99999") is None


# ---------------------------------------------------------------------------
# Cross-section parsing
# ---------------------------------------------------------------------------


class TestCrossSectionParsing:
    def test_get_cross_section_returns_object(self):
        g = GeometryFile(EX1)
        # First XS of Butte Cr. is at RS "0.2" (river mile 0.2)
        xs_list = g.cross_sections("Butte Cr.", "Tributary")
        assert len(xs_list) == 3
        xs = xs_list[0]
        assert isinstance(xs, CrossSection)

    def test_stations_and_elevations_count(self):
        g = GeometryFile(EX1)
        xs = g.cross_sections("Butte Cr.", "Tributary")[0]
        assert len(xs.stations) == len(xs.elevations)
        assert len(xs.stations) == 8  # #Sta/Elev= 8

    def test_station_values(self):
        g = GeometryFile(EX1)
        xs = g.cross_sections("Butte Cr.", "Tributary")[0]
        # From file: 210 90 220 82 260 80 265 70 270 71 275 81 300 83 310 91
        assert xs.stations[0] == pytest.approx(210.0)
        assert xs.elevations[0] == pytest.approx(90.0)
        assert xs.stations[4] == pytest.approx(270.0)
        assert xs.elevations[4] == pytest.approx(71.0)

    def test_bank_stations(self):
        g = GeometryFile(EX1)
        xs = g.cross_sections("Butte Cr.", "Tributary")[0]
        # Bank Sta=260,275
        assert xs.bank_left == pytest.approx(260.0)
        assert xs.bank_right == pytest.approx(275.0)

    def test_mannings_count(self):
        g = GeometryFile(EX1)
        xs = g.cross_sections("Butte Cr.", "Tributary")[0]
        # #Mann= 3 , 0 , 0
        assert len(xs.mann_entries) == 3

    def test_mannings_values(self):
        g = GeometryFile(EX1)
        xs = g.cross_sections("Butte Cr.", "Tributary")[0]
        # 210 .07 0    260 .04 0    275 .07 0
        assert xs.mann_entries[0].station == pytest.approx(210.0)
        assert xs.mann_entries[0].n_value == pytest.approx(0.07)
        assert xs.mann_entries[1].station == pytest.approx(260.0)
        assert xs.mann_entries[1].n_value == pytest.approx(0.04)

    def test_expansion_contraction(self):
        g = GeometryFile(EX1)
        xs = g.cross_sections("Butte Cr.", "Tributary")[0]
        # Exp/Cntr=0.3,0.1
        assert xs.expansion == pytest.approx(0.3)
        assert xs.contraction == pytest.approx(0.1)

    def test_description_parsed(self):
        g = GeometryFile(EX1)
        xs = g.cross_sections("Butte Cr.", "Tributary")[0]
        assert "Upstream Boundary" in xs.description

    def test_river_reach_rs_set(self):
        g = GeometryFile(EX1)
        xs = g.cross_sections("Butte Cr.", "Tributary")[0]
        assert xs.river == "Butte Cr."
        assert xs.reach == "Tributary"
        assert xs.rs != ""

    def test_reach_lengths(self):
        g = GeometryFile(EX1)
        xs = g.cross_sections("Butte Cr.", "Tributary")[0]
        # Type RM Length L Ch R = 1 ,0.2     ,500,500,500
        assert xs.left_length == pytest.approx(500.0)
        assert xs.channel_length == pytest.approx(500.0)
        assert xs.right_length == pytest.approx(500.0)

    def test_downstream_xs_has_zero_lengths(self):
        g = GeometryFile(EX1)
        xs_list = g.cross_sections("Butte Cr.", "Tributary")
        last_xs = xs_list[-1]
        # Last XS has length 0,0,0
        assert last_xs.left_length == pytest.approx(0.0)
        assert last_xs.channel_length == pytest.approx(0.0)

    def test_get_cross_section_by_rs(self):
        g = GeometryFile(EX1)
        xs_list = g.cross_sections("Butte Cr.", "Tributary")
        rs = xs_list[0].rs
        xs_direct = g.get_cross_section("Butte Cr.", "Tributary", rs)
        assert xs_direct is not None
        assert xs_direct.rs == xs_list[0].rs
        assert xs_direct.stations == xs_list[0].stations

    def test_get_cross_section_missing_returns_none(self):
        g = GeometryFile(EX1)
        assert g.get_cross_section("Butte Cr.", "Tributary", "99999") is None

    def test_get_cross_section_on_structure_returns_none(self):
        g = GeometryFile(CONSPAN)
        nodes = g.node_rs_list("Spring Creek", "Culvrt Reach")
        culvert_rs = next(rs for ntype, rs in nodes if ntype == NODE_CULVERT)
        result = g.get_cross_section("Spring Creek", "Culvrt Reach", culvert_rs)
        assert result is None

    def test_cross_sections_multireach_fall_upper(self):
        g = GeometryFile(EX1)
        xs_list = g.cross_sections("Fall River", "Upper Reach")
        assert len(xs_list) == 3

    def test_cross_sections_multireach_fall_lower(self):
        g = GeometryFile(EX1)
        xs_list = g.cross_sections("Fall River", "Lower Reach")
        assert len(xs_list) == 4


# ---------------------------------------------------------------------------
# Ineffective areas
# ---------------------------------------------------------------------------


class TestIneffectiveAreas:
    def test_conspan_ineff_areas_parsed(self):
        g = GeometryFile(CONSPAN)
        # The upstream approach XS (20.238) has ineffective areas
        xs_list = g.cross_sections("Spring Creek", "Culvrt Reach")
        xs_with_ineff = [xs for xs in xs_list if xs.ineff_areas]
        assert len(xs_with_ineff) >= 1

    def test_ineff_area_fields(self):
        g = GeometryFile(CONSPAN)
        xs_list = g.cross_sections("Spring Creek", "Culvrt Reach")
        xs_with_ineff = next(xs for xs in xs_list if xs.ineff_areas)
        area = xs_with_ineff.ineff_areas[0]
        assert isinstance(area, IneffArea)
        assert area.x_start < area.x_end

    def test_ineff_permanent_flag(self):
        g = GeometryFile(CONSPAN)
        xs_list = g.cross_sections("Spring Creek", "Culvrt Reach")
        xs_with_ineff = next(xs for xs in xs_list if xs.ineff_areas)
        # Permanent Ineff= with "F F" flags → both should be False
        for area in xs_with_ineff.ineff_areas:
            assert area.permanent is False


# ---------------------------------------------------------------------------
# Interpolated cross sections
# ---------------------------------------------------------------------------


class TestInterpolatedXS:
    def test_interpolated_flag(self):
        g = GeometryFile(CONSPAN)
        # ConSpan has an interpolated XS at 20.208*
        xs_list = g.cross_sections("Spring Creek", "Culvrt Reach")
        interpolated = [xs for xs in xs_list if xs.interpolated]
        assert len(interpolated) >= 1


# ---------------------------------------------------------------------------
# Raw node access
# ---------------------------------------------------------------------------


class TestRawNodeAccess:
    def test_get_node_lines_xs(self):
        g = GeometryFile(EX1)
        xs_list = g.cross_sections("Butte Cr.", "Tributary")
        rs = xs_list[0].rs
        lines = g.get_node_lines("Butte Cr.", "Tributary", rs)
        assert lines is not None
        assert lines[0].startswith("Type RM Length L Ch R")
        assert any("#Sta/Elev" in ln for ln in lines)

    def test_get_node_lines_bridge(self):
        g = GeometryFile(BEAVER)
        nodes = g.node_rs_list("Beaver Creek", "Kentwood")
        bridge_rs = next(rs for ntype, rs in nodes if ntype == NODE_BRIDGE)
        lines = g.get_node_lines("Beaver Creek", "Kentwood", bridge_rs)
        assert lines is not None
        assert any("Bridge Culvert" in ln for ln in lines)

    def test_get_node_lines_missing_returns_none(self):
        g = GeometryFile(EX1)
        assert g.get_node_lines("Butte Cr.", "Tributary", "99999") is None

    def test_get_node_lines_inline_structure(self):
        g = GeometryFile(NIT)
        nodes = g.node_rs_list("Nittany River", "Weir Reach")
        is_rs = next(rs for ntype, rs in nodes if ntype == NODE_INLINE_STRUCTURE)
        lines = g.get_node_lines("Nittany River", "Weir Reach", is_rs)
        assert lines is not None
        assert any("Inline Weir" in ln or "IW " in ln for ln in lines)


# ---------------------------------------------------------------------------
# set_mannings
# ---------------------------------------------------------------------------


class TestSetMannings:
    def test_set_mannings_changes_values(self, tmp_copy):
        path = tmp_copy(EX1)
        g = GeometryFile(path)
        xs_list = g.cross_sections("Butte Cr.", "Tributary")
        rs = xs_list[0].rs

        new_entries = [
            ManningEntry(station=210.0, n_value=0.05),
            ManningEntry(station=260.0, n_value=0.03),
            ManningEntry(station=275.0, n_value=0.06),
        ]
        g.set_mannings("Butte Cr.", "Tributary", rs, new_entries)
        g.save()

        g2 = GeometryFile(path)
        xs2 = g2.get_cross_section("Butte Cr.", "Tributary", rs)
        assert xs2 is not None
        assert xs2.mann_entries[0].n_value == pytest.approx(0.05)
        assert xs2.mann_entries[1].n_value == pytest.approx(0.03)
        assert xs2.mann_entries[2].n_value == pytest.approx(0.06)

    def test_set_mannings_preserves_type(self, tmp_copy):
        path = tmp_copy(EX1)
        g = GeometryFile(path)
        xs_list = g.cross_sections("Butte Cr.", "Tributary")
        rs = xs_list[0].rs
        original_type = xs_list[0].mann_type
        original_alt = xs_list[0].mann_alt

        new_entries = [ManningEntry(station=210.0, n_value=0.05)]
        g.set_mannings("Butte Cr.", "Tributary", rs, new_entries)
        g.save()

        g2 = GeometryFile(path)
        xs2 = g2.get_cross_section("Butte Cr.", "Tributary", rs)
        assert xs2.mann_type == original_type
        assert xs2.mann_alt == original_alt

    def test_set_mannings_different_zone_count(self, tmp_copy):
        path = tmp_copy(EX1)
        g = GeometryFile(path)
        xs_list = g.cross_sections("Butte Cr.", "Tributary")
        rs = xs_list[0].rs

        # Change from 3 zones to 2
        new_entries = [
            ManningEntry(station=210.0, n_value=0.08),
            ManningEntry(station=260.0, n_value=0.04),
        ]
        g.set_mannings("Butte Cr.", "Tributary", rs, new_entries)
        g.save()

        g2 = GeometryFile(path)
        xs2 = g2.get_cross_section("Butte Cr.", "Tributary", rs)
        assert len(xs2.mann_entries) == 2

    def test_set_mannings_other_xs_unchanged(self, tmp_copy):
        path = tmp_copy(EX1)
        g = GeometryFile(path)
        xs_list = g.cross_sections("Butte Cr.", "Tributary")
        rs0 = xs_list[0].rs
        rs1 = xs_list[1].rs
        original_n1 = xs_list[1].mann_entries[0].n_value

        g.set_mannings(
            "Butte Cr.",
            "Tributary",
            rs0,
            [
                ManningEntry(station=210.0, n_value=0.09),
                ManningEntry(station=260.0, n_value=0.03),
                ManningEntry(station=275.0, n_value=0.09),
            ],
        )
        g.save()

        g2 = GeometryFile(path)
        xs2_1 = g2.get_cross_section("Butte Cr.", "Tributary", rs1)
        assert xs2_1.mann_entries[0].n_value == pytest.approx(original_n1)

    def test_set_mannings_missing_rs_raises(self):
        g = GeometryFile(EX1)
        with pytest.raises(KeyError):
            g.set_mannings("Butte Cr.", "Tributary", "99999", [])


# ---------------------------------------------------------------------------
# set_stations
# ---------------------------------------------------------------------------


class TestSetStations:
    def test_set_stations_changes_values(self, tmp_copy):
        path = tmp_copy(EX1)
        g = GeometryFile(path)
        xs_list = g.cross_sections("Butte Cr.", "Tributary")
        rs = xs_list[0].rs
        orig_n = len(xs_list[0].stations)

        new_sta = [0.0, 50.0, 100.0, 150.0, 200.0]
        new_elev = [10.0, 5.0, 3.0, 5.0, 10.0]
        g.set_stations("Butte Cr.", "Tributary", rs, new_sta, new_elev)
        g.save()

        g2 = GeometryFile(path)
        xs2 = g2.get_cross_section("Butte Cr.", "Tributary", rs)
        assert len(xs2.stations) == 5
        assert xs2.stations[0] == pytest.approx(0.0)
        assert xs2.elevations[2] == pytest.approx(3.0)

    def test_set_stations_length_mismatch_raises(self):
        g = GeometryFile(EX1)
        xs_list = g.cross_sections("Butte Cr.", "Tributary")
        rs = xs_list[0].rs
        with pytest.raises(ValueError):
            g.set_stations("Butte Cr.", "Tributary", rs, [0.0, 100.0], [5.0])

    def test_set_stations_preserves_mannings(self, tmp_copy):
        path = tmp_copy(EX1)
        g = GeometryFile(path)
        xs_list = g.cross_sections("Butte Cr.", "Tributary")
        rs = xs_list[0].rs
        orig_n0 = xs_list[0].mann_entries[0].n_value

        g.set_stations("Butte Cr.", "Tributary", rs, [0.0, 100.0], [5.0, 5.0])
        g.save()

        g2 = GeometryFile(path)
        xs2 = g2.get_cross_section("Butte Cr.", "Tributary", rs)
        assert xs2.mann_entries[0].n_value == pytest.approx(orig_n0)

    def test_set_stations_missing_rs_raises(self):
        g = GeometryFile(EX1)
        with pytest.raises(KeyError):
            g.set_stations("Butte Cr.", "Tributary", "99999", [0.0], [0.0])


# ---------------------------------------------------------------------------
# set_bank_stations
# ---------------------------------------------------------------------------


class TestSetBankStations:
    def test_set_bank_stations(self, tmp_copy):
        path = tmp_copy(EX1)
        g = GeometryFile(path)
        xs_list = g.cross_sections("Butte Cr.", "Tributary")
        rs = xs_list[0].rs

        g.set_bank_stations("Butte Cr.", "Tributary", rs, 250.0, 280.0)
        g.save()

        g2 = GeometryFile(path)
        xs2 = g2.get_cross_section("Butte Cr.", "Tributary", rs)
        assert xs2.bank_left == pytest.approx(250.0)
        assert xs2.bank_right == pytest.approx(280.0)

    def test_set_bank_stations_missing_raises(self):
        g = GeometryFile(EX1)
        with pytest.raises(KeyError):
            g.set_bank_stations("Butte Cr.", "Tributary", "99999", 0.0, 0.0)


# ---------------------------------------------------------------------------
# set_exp_cntr
# ---------------------------------------------------------------------------


class TestSetExpCntr:
    def test_set_exp_cntr(self, tmp_copy):
        path = tmp_copy(EX1)
        g = GeometryFile(path)
        xs_list = g.cross_sections("Butte Cr.", "Tributary")
        rs = xs_list[0].rs

        g.set_exp_cntr("Butte Cr.", "Tributary", rs, 0.5, 0.2)
        g.save()

        g2 = GeometryFile(path)
        xs2 = g2.get_cross_section("Butte Cr.", "Tributary", rs)
        assert xs2.expansion == pytest.approx(0.5)
        assert xs2.contraction == pytest.approx(0.2)

    def test_set_exp_cntr_missing_raises(self):
        g = GeometryFile(EX1)
        with pytest.raises(KeyError):
            g.set_exp_cntr("Butte Cr.", "Tributary", "99999", 0.3, 0.1)


# ---------------------------------------------------------------------------
# Round-trip fidelity
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def _assert_roundtrip(self, src: Path, tmp_path: Path) -> None:
        orig = src.read_bytes()
        dst = tmp_path / src.name
        shutil.copy(src, dst)
        g = GeometryFile(dst)
        g.save()
        assert dst.read_bytes() == orig, "save() must not modify unedited content"

    def test_roundtrip_ex1(self, tmp_path):
        self._assert_roundtrip(EX1, tmp_path)

    def test_roundtrip_conspan(self, tmp_path):
        self._assert_roundtrip(CONSPAN, tmp_path)

    def test_roundtrip_beaver(self, tmp_path):
        self._assert_roundtrip(BEAVER, tmp_path)

    def test_roundtrip_nit_inline(self, tmp_path):
        self._assert_roundtrip(NIT, tmp_path)


# ---------------------------------------------------------------------------
# Case-insensitive reach/river matching
# ---------------------------------------------------------------------------


class TestCaseInsensitiveMatching:
    def test_reach_lookup_case_insensitive(self):
        g = GeometryFile(EX1)
        xs_upper = g.cross_sections("BUTTE CR.", "TRIBUTARY")
        xs_lower = g.cross_sections("butte cr.", "tributary")
        assert len(xs_upper) == len(xs_lower) > 0

    def test_get_xs_case_insensitive(self):
        g = GeometryFile(EX1)
        xs_list = g.cross_sections("Butte Cr.", "Tributary")
        rs = xs_list[0].rs
        xs_upper = g.get_cross_section("BUTTE CR.", "TRIBUTARY", rs)
        assert xs_upper is not None
        assert xs_upper.stations == xs_list[0].stations


# ---------------------------------------------------------------------------
# Conspan culvert cross sections (approach sections)
# ---------------------------------------------------------------------------


class TestConspanXS:
    def test_conspan_xs_count(self):
        g = GeometryFile(CONSPAN)
        # 10 XS nodes + 1 culvert node; cross_sections() returns only type-1 nodes
        xs_list = g.cross_sections("Spring Creek", "Culvrt Reach")
        assert len(xs_list) == 10

    def test_conspan_upstream_xs_has_htab_data(self):
        g = GeometryFile(CONSPAN)
        # Presence of XS HTab lines doesn't break parsing
        xs_list = g.cross_sections("Spring Creek", "Culvrt Reach")
        assert len(xs_list) > 0

    def test_conspan_upstream_approach_xs_mann(self):
        g = GeometryFile(CONSPAN)
        xs_list = g.cross_sections("Spring Creek", "Culvrt Reach")
        # First XS (20.535) has 3 Manning zones
        xs0 = xs_list[0]
        assert len(xs0.mann_entries) == 3


# ---------------------------------------------------------------------------
# Beaver bridge file
# ---------------------------------------------------------------------------


class TestBeaverBridge:
    def test_beaver_xs_parsed(self):
        g = GeometryFile(BEAVER)
        xs_list = g.cross_sections("Beaver Creek", "Kentwood")
        assert len(xs_list) > 5  # many XS in beaver.g01

    def test_beaver_bridge_node_lines_non_empty(self):
        g = GeometryFile(BEAVER)
        nodes = g.node_rs_list("Beaver Creek", "Kentwood")
        bridge_rs = next(rs for ntype, rs in nodes if ntype == NODE_BRIDGE)
        lines = g.get_node_lines("Beaver Creek", "Kentwood", bridge_rs)
        assert len(lines) > 5
