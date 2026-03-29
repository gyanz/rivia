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

from math import isnan

from raspy.model.geometry import (
    Bridge,
    CrossSection,
    CulvertGroup,
    GeometryFile,
    IneffArea,
    Lateral,
    ManningEntry,
    NODE_BRIDGE,
    NODE_CULVERT,
    NODE_INLINE_STRUCTURE,
    NODE_LATERAL_STRUCTURE,
    NODE_XS,
    Pier,
    Roadway,
    Weir,
)

FIXTURES = Path(__file__).parent / "fixtures"
EX1 = FIXTURES / "ex1.g01"
CONSPAN = FIXTURES / "conspan.g01"
BEAVER = FIXTURES / "beaver.g01"
NIT = FIXTURES / "nit_inline.g01"
LAT3 = FIXTURES / "3reach_lat.g01"


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


# ---------------------------------------------------------------------------
# structures property — Bridge (type 3) from beaver.g01
# ---------------------------------------------------------------------------


class TestStructuresBridge:
    def test_summary_one_bridge(self):
        g = GeometryFile(BEAVER)
        assert g.structures.summary == {"inlines": 0, "bridges": 1, "laterals": 0}

    def test_bridge_type(self):
        g = GeometryFile(BEAVER)
        _, br = g.structures.bridges.items()[0]
        assert isinstance(br, Bridge)

    def test_bridge_key(self):
        g = GeometryFile(BEAVER)
        keys = g.structures.bridges.keys()
        assert keys == ["Beaver Creek Kentwood 5.4"]

    def test_bridge_location(self):
        g = GeometryFile(BEAVER)
        br = g.structures.bridges["Beaver Creek Kentwood 5.4"]
        assert br.location == ("Beaver Creek", "Kentwood", "5.4")

    def test_bridge_adjacent_xs(self):
        g = GeometryFile(BEAVER)
        br = g.structures.bridges["Beaver Creek Kentwood 5.4"]
        assert br.upstream_node == ("Beaver Creek", "Kentwood", "5.41")
        assert br.downstream_node == ("Beaver Creek", "Kentwood", "5.39")

    def test_bridge_connection_types(self):
        g = GeometryFile(BEAVER)
        br = g.structures.bridges["Beaver Creek Kentwood 5.4"]
        assert br.upstream_type == "XS"
        assert br.downstream_type == "XS"

    def test_bridge_weir(self):
        g = GeometryFile(BEAVER)
        br = g.structures.bridges["Beaver Creek Kentwood 5.4"]
        assert isinstance(br.weir, Weir)
        assert br.weir.width == 40.0
        assert br.weir.coefficient == 2.6
        assert br.weir.shape == "Broad Crested"
        assert br.weir.max_submergence == pytest.approx(0.95)
        assert isnan(br.weir.min_elevation)
        assert br.weir.skew == 0.0
        assert isnan(br.weir.us_slope)
        assert isnan(br.weir.ds_slope)
        assert br.weir.use_water_surface is False

    def test_bridge_gate_groups_empty(self):
        g = GeometryFile(BEAVER)
        br = g.structures.bridges["Beaver Creek Kentwood 5.4"]
        assert br.gate_groups == []

    def test_bridge_cached(self):
        g = GeometryFile(BEAVER)
        assert g.structures is g.structures


# ---------------------------------------------------------------------------
# structures property — Bridge from culvert (type 2) in conspan.g01
# ---------------------------------------------------------------------------


class TestStructuresCulvert:
    def test_summary_one_bridge(self):
        g = GeometryFile(CONSPAN)
        assert g.structures.summary == {"inlines": 0, "bridges": 1, "laterals": 0}

    def test_culvert_key(self):
        g = GeometryFile(CONSPAN)
        assert g.structures.bridges.keys() == ["Spring Creek Culvrt Reach 20.237"]

    def test_culvert_location(self):
        g = GeometryFile(CONSPAN)
        br = g.structures.bridges["Spring Creek Culvrt Reach 20.237"]
        assert br.location == ("Spring Creek", "Culvrt Reach", "20.237")

    def test_culvert_adjacent_xs(self):
        g = GeometryFile(CONSPAN)
        br = g.structures.bridges["Spring Creek Culvrt Reach 20.237"]
        assert br.upstream_node == ("Spring Creek", "Culvrt Reach", "20.238")
        assert br.downstream_node == ("Spring Creek", "Culvrt Reach", "20.227")

    def test_culvert_node_type_still_culvert(self):
        g = GeometryFile(CONSPAN)
        assert g.node_type("Spring Creek", "Culvrt Reach", "20.237") == NODE_CULVERT


# ---------------------------------------------------------------------------
# structures property — multi-reach file (3reach_lat.g01)
# ---------------------------------------------------------------------------


class TestStructures3Reach:
    def test_summary(self):
        g = GeometryFile(LAT3)
        assert g.structures.summary == {"inlines": 0, "bridges": 3, "laterals": 1}

    def test_bridge_count(self):
        g = GeometryFile(LAT3)
        assert len(g.structures.bridges) == 3

    def test_bridge_keys(self):
        g = GeometryFile(LAT3)
        keys = g.structures.bridges.keys()
        assert "Butte Cr. Butte Cr. 0.22" in keys
        assert "Fall River Upper Reach 10.1" in keys
        assert "Fall River Lower Reach 9.55" in keys

    def test_bridge_description(self):
        g = GeometryFile(LAT3)
        br = g.structures.bridges["Butte Cr. Butte Cr. 0.22"]
        assert br.description == "bridge crossing"

    def test_culvert_in_bridge_index(self):
        # Type 2 (culvert) and type 3 (bridge) both go into .bridges
        g = GeometryFile(LAT3)
        br = g.structures.bridges["Fall River Upper Reach 10.1"]
        assert br.description == "Culvert crossing"
        assert g.node_type("Fall River", "Upper Reach", "10.1") == NODE_CULVERT

    def test_bridge_weir_values(self):
        g = GeometryFile(LAT3)
        br = g.structures.bridges["Butte Cr. Butte Cr. 0.22"]
        assert br.weir.width == 100.0
        assert br.weir.coefficient == pytest.approx(2.6)
        assert br.weir.max_submergence == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# structures property — Lateral (type 6) from 3reach_lat.g01
# ---------------------------------------------------------------------------


class TestStructuresLateral:
    def test_lateral_count(self):
        g = GeometryFile(LAT3)
        assert len(g.structures.laterals) == 1

    def test_lateral_type(self):
        g = GeometryFile(LAT3)
        _, lat = g.structures.laterals.items()[0]
        assert isinstance(lat, Lateral)

    def test_lateral_key(self):
        g = GeometryFile(LAT3)
        assert g.structures.laterals.keys() == ["Fall River Upper Reach 10.25"]

    def test_lateral_location(self):
        g = GeometryFile(LAT3)
        lat = g.structures.laterals["Fall River Upper Reach 10.25"]
        assert lat.location == ("Fall River", "Upper Reach", "10.25")

    def test_lateral_upstream_node(self):
        g = GeometryFile(LAT3)
        lat = g.structures.laterals["Fall River Upper Reach 10.25"]
        assert lat.upstream_node == ("Fall River", "Upper Reach", "10.3")

    def test_lateral_downstream_node(self):
        g = GeometryFile(LAT3)
        lat = g.structures.laterals["Fall River Upper Reach 10.25"]
        assert lat.downstream_node == "Butte Cr. Butte Cr."

    def test_lateral_connection_types(self):
        g = GeometryFile(LAT3)
        lat = g.structures.laterals["Fall River Upper Reach 10.25"]
        assert lat.upstream_type == "XS"
        assert lat.downstream_type == "XS"

    def test_lateral_weir(self):
        g = GeometryFile(LAT3)
        lat = g.structures.laterals["Fall River Upper Reach 10.25"]
        assert isinstance(lat.weir, Weir)
        assert lat.weir.width == 10.0
        assert lat.weir.coefficient == pytest.approx(3.0)
        assert lat.weir.shape == "Broad Crested"
        assert isnan(lat.weir.max_submergence)
        assert isnan(lat.weir.min_elevation)
        assert lat.weir.skew == 0.0
        assert isnan(lat.weir.us_slope)
        assert isnan(lat.weir.ds_slope)

    def test_lateral_use_water_surface(self):
        # WSCriteria=-1 means use water surface
        g = GeometryFile(LAT3)
        lat = g.structures.laterals["Fall River Upper Reach 10.25"]
        assert lat.weir.use_water_surface is True

    def test_lateral_gate_groups_empty(self):
        g = GeometryFile(LAT3)
        lat = g.structures.laterals["Fall River Upper Reach 10.25"]
        assert lat.gate_groups == []

    def test_lateral_node_type(self):
        g = GeometryFile(LAT3)
        assert (
            g.node_type("Fall River", "Upper Reach", "10.25")
            == NODE_LATERAL_STRUCTURE
        )


# ---------------------------------------------------------------------------
# structures property — no structures (inline-only file)
# ---------------------------------------------------------------------------


class TestStructuresNitInline:
    def test_no_bridges_or_laterals(self):
        g = GeometryFile(NIT)
        assert g.structures.summary == {"inlines": 1, "bridges": 0, "laterals": 0}

    def test_inline_still_parsed(self):
        g = GeometryFile(NIT)
        assert len(g.structures.inlines) == 1

    def _iw(self):
        from math import isnan
        g = GeometryFile(NIT)
        return g.structures.inlines[0], isnan

    def test_pilot_flow(self):
        iw, _ = self._iw()
        assert iw.pilot_flow == 0.0

    def test_crest_profile_length(self):
        iw, _ = self._iw()
        assert len(iw.crest_profile) == 6

    def test_crest_profile_values(self):
        iw, _ = self._iw()
        # fixture: 0/13.5, 57/13.5, 61/9.5, 190/9.5, 194/13.5, 1000/13.5
        assert iw.crest_profile[0] == (0.0, 13.5)
        assert iw.crest_profile[2] == (61.0, 9.5)
        assert iw.crest_profile[5] == (1000.0, 13.5)

    def test_weir_parsed(self):
        iw, isnan = self._iw()
        assert iw.weir is not None

    def test_weir_dist(self):
        iw, _ = self._iw()
        assert iw.weir.dist == 20.0

    def test_weir_width_and_coef(self):
        iw, _ = self._iw()
        assert iw.weir.width == 50.0
        assert iw.weir.coefficient == 3.95

    def test_weir_shape_ogee(self):
        iw, _ = self._iw()
        assert iw.weir.shape == "Ogee"

    def test_weir_spillway_and_design_head(self):
        iw, _ = self._iw()
        assert iw.weir.spillway_approach_height == 24.0
        assert iw.weir.design_energy_head == 3.0

    def test_weir_min_elevation_nan(self):
        from math import isnan
        iw, _ = self._iw()
        assert isnan(iw.weir.min_elevation)

    def test_gate_group_count(self):
        iw, _ = self._iw()
        assert len(iw.gate_groups) == 3

    def test_gate_group_names(self):
        iw, _ = self._iw()
        names = [gg.name for gg in iw.gate_groups]
        assert names[0] == "Left Group"
        assert names[1] == "Center Group"
        assert names[2] == "Right Group"

    def test_gate_coefficient(self):
        iw, _ = self._iw()
        # Left Group GCoef = 0.8
        assert iw.gate_groups[0].gate_coefficient == 0.8

    def test_gate_exponents(self):
        iw, _ = self._iw()
        gg = iw.gate_groups[0]
        assert gg.trunnion_exponent == 0.16
        assert gg.opening_exponent == 0.72
        assert gg.height_exponent == 0.62

    def test_gate_type_radial(self):
        iw, _ = self._iw()
        assert iw.gate_groups[0].gate_type == "radial"

    def test_gate_openings_count(self):
        iw, _ = self._iw()
        # each gate group has 5 openings
        assert len(iw.gate_groups[0].openings) == 5

    def test_gate_opening_stations(self):
        iw, _ = self._iw()
        # Left Group stations: 220, 255, 290, 325, 360
        stations = [op.station for op in iw.gate_groups[0].openings]
        assert stations == [220.0, 255.0, 290.0, 325.0, 360.0]

    def test_weir_shape_ogee(self):
        iw, _ = self._iw()
        # nit_inline radial gate groups overflow as ogee
        assert iw.gate_groups[0].weir_shape == "Ogee"

    def test_weir_shape_broad_crested(self):
        from raspy.model.geometry import GateGroup
        g = GateGroup(
            name="test", width=10.0, height=5.0, invert=0.0,
            gate_coefficient=0.6, trunnion_exponent=0.16,
            opening_exponent=0.72, height_exponent=0.62,
            gate_type="sluice", weir_coefficient=2.6,
            is_ogee=False, spillway_approach_height=0.0,
            design_energy_head=0.0, trunnion_height=0.0,
            orifice_coefficient=0.8, head_reference=0,
            radial_coefficient=0.0, is_sharp_crested=False,
            use_weir_param1=False, use_weir_param2=False,
            use_weir_param3=False,
        )
        assert g.weir_shape == "Broad Crested"

    def test_weir_shape_sharp_crested(self):
        from raspy.model.geometry import GateGroup
        g = GateGroup(
            name="test", width=10.0, height=5.0, invert=0.0,
            gate_coefficient=0.6, trunnion_exponent=0.16,
            opening_exponent=0.72, height_exponent=0.62,
            gate_type="sluice", weir_coefficient=2.6,
            is_ogee=False, spillway_approach_height=0.0,
            design_energy_head=0.0, trunnion_height=0.0,
            orifice_coefficient=0.8, head_reference=0,
            radial_coefficient=0.0, is_sharp_crested=True,
            use_weir_param1=False, use_weir_param2=False,
            use_weir_param3=False,
        )
        assert g.weir_shape == "Sharp Crested"

    def test_Cu_alias(self):
        iw, _ = self._iw()
        g = iw.gate_groups[0]
        assert g.Cu == g.gate_coefficient

    def test_Cs_alias(self):
        iw, _ = self._iw()
        g = iw.gate_groups[0]
        assert g.Cs == g.orifice_coefficient


# ---------------------------------------------------------------------------
# Round-trip: 3reach_lat fixture
# ---------------------------------------------------------------------------


class TestRoundTripLat3:
    def test_roundtrip(self, tmp_path):
        orig = LAT3.read_bytes()
        dst = tmp_path / LAT3.name
        import shutil as _shutil
        _shutil.copy(LAT3, dst)
        g = GeometryFile(dst)
        g.save()
        assert dst.read_bytes() == orig


# ---------------------------------------------------------------------------
# Roadway geometry — beaver.g01 bridge (type 3, RS 5.4)
# ---------------------------------------------------------------------------


class TestBridgeRoadway:
    """Deck geometry arrays parsed from beaver.g01.

    Deck Dist data: 30,40,2.6,0, 6, 6, , , 0.95, 0, 0,0,,
    Up/Dn stations: 0 450 450 647 647 2000
    Up/Dn hi-chord: 216.93 × 6
    Up/Dn lo-chord: 200 200 215.7 215.7 200 200
    """

    def _br(self):
        g = GeometryFile(BEAVER)
        return g.structures.bridges["Beaver Creek Kentwood 5.4"]

    def test_roadway_present(self):
        assert isinstance(self._br().roadway, Roadway)

    def test_roadway_dist(self):
        assert self._br().roadway.dist == pytest.approx(30.0)

    def test_roadway_width(self):
        assert self._br().roadway.width == pytest.approx(40.0)

    def test_roadway_weir_coefficient(self):
        assert self._br().roadway.weir_coefficient == pytest.approx(2.6)

    def test_roadway_shape(self):
        assert self._br().roadway.shape == "Broad Crested"

    def test_roadway_max_submergence(self):
        assert self._br().roadway.max_submergence == pytest.approx(0.95)

    def test_roadway_min_lo_chord_nan(self):
        assert isnan(self._br().roadway.min_lo_chord)

    def test_roadway_max_hi_chord_nan(self):
        assert isnan(self._br().roadway.max_hi_chord)

    def test_roadway_stations_up_count(self):
        assert len(self._br().roadway.stations_up) == 6

    def test_roadway_stations_up_values(self):
        assert self._br().roadway.stations_up == pytest.approx([0, 450, 450, 647, 647, 2000])

    def test_roadway_hi_chord_up(self):
        hi = self._br().roadway.hi_chord_up
        assert len(hi) == 6
        assert all(v == pytest.approx(216.93) for v in hi)

    def test_roadway_lo_chord_up(self):
        lo = self._br().roadway.lo_chord_up
        assert lo == pytest.approx([200, 200, 215.7, 215.7, 200, 200])

    def test_roadway_stations_dn_equals_up(self):
        rw = self._br().roadway
        assert rw.stations_dn == pytest.approx(rw.stations_up)

    def test_roadway_hi_chord_dn_equals_up(self):
        rw = self._br().roadway
        assert rw.hi_chord_dn == pytest.approx(rw.hi_chord_up)

    def test_htab_hw_max(self):
        assert self._br().htab_hw_max == pytest.approx(225.0)

    def test_htab_tw_max(self):
        assert self._br().htab_tw_max == pytest.approx(220.0)

    def test_htab_max_flow(self):
        assert self._br().htab_max_flow == pytest.approx(50000.0)


# ---------------------------------------------------------------------------
# Pier geometry — beaver.g01 bridge (9 pier groups)
# ---------------------------------------------------------------------------


class TestBridgePiers:
    """Pier groups from beaver.g01 bridge.

    9 piers at stations 470..630 (step 20), each with 2 up and 2 dn elements.
    Upstream widths  [1.25, 1.25], elevations [202.7, 215.7].
    Downstream same.
    """

    def _br(self):
        g = GeometryFile(BEAVER)
        return g.structures.bridges["Beaver Creek Kentwood 5.4"]

    def test_pier_count(self):
        assert len(self._br().piers) == 9

    def test_pier_type(self):
        assert isinstance(self._br().piers[0], Pier)

    def test_first_pier_upstream_station(self):
        assert self._br().piers[0].upstream_station == pytest.approx(470.0)

    def test_last_pier_upstream_station(self):
        assert self._br().piers[-1].upstream_station == pytest.approx(630.0)

    def test_pier_upstream_count(self):
        assert self._br().piers[0].upstream_count == 2

    def test_pier_downstream_count(self):
        assert self._br().piers[0].downstream_count == 2

    def test_pier_upstream_widths(self):
        assert self._br().piers[0].upstream_widths == pytest.approx([1.25, 1.25])

    def test_pier_upstream_elevations(self):
        assert self._br().piers[0].upstream_elevations == pytest.approx([202.7, 215.7])

    def test_pier_downstream_widths(self):
        assert self._br().piers[0].downstream_widths == pytest.approx([1.25, 1.25])

    def test_pier_downstream_elevations(self):
        assert self._br().piers[0].downstream_elevations == pytest.approx([202.7, 215.7])

    def test_pier_skew_zero(self):
        assert self._br().piers[0].skew == 0.0


# ---------------------------------------------------------------------------
# Culvert parsing — conspan.g01 (type 2, RS 20.237, Con Span barrel)
# ---------------------------------------------------------------------------


class TestCulvertParsing:
    """Culvert= line parsed from conspan.g01.

    Culvert=9,6,28,50,0.013,0.5,1,61,3,25.1,1000,25,1000,Culvert # 1 , 0 ,5
    Culvert Bottom n=0.03
    Culvert Bottom Depth=0.5
    """

    def _br(self):
        g = GeometryFile(CONSPAN)
        return g.structures.bridges["Spring Creek Culvrt Reach 20.237"]

    def test_culvert_count(self):
        assert len(self._br().culvert_groups) == 1

    def test_culvert_type(self):
        assert isinstance(self._br().culvert_groups[0], CulvertGroup)

    def test_culvert_shape_code(self):
        assert self._br().culvert_groups[0].shape_code == 9

    def test_culvert_shape_name(self):
        assert self._br().culvert_groups[0].shape_name == "Con Span"

    def test_culvert_name(self):
        assert self._br().culvert_groups[0].name == "Culvert # 1"

    def test_culvert_span(self):
        assert self._br().culvert_groups[0].span == pytest.approx(6.0)

    def test_culvert_rise(self):
        assert self._br().culvert_groups[0].rise == pytest.approx(28.0)

    def test_culvert_length(self):
        assert self._br().culvert_groups[0].length == pytest.approx(50.0)

    def test_culvert_n_top(self):
        assert self._br().culvert_groups[0].n_top == pytest.approx(0.013)

    def test_culvert_entrance_loss(self):
        assert self._br().culvert_groups[0].entrance_loss == pytest.approx(0.5)

    def test_culvert_us_invert(self):
        assert self._br().culvert_groups[0].upstream_invert == pytest.approx(25.1)

    def test_culvert_us_station(self):
        assert self._br().culvert_groups[0].upstream_station == pytest.approx(1000.0)

    def test_culvert_ds_invert(self):
        assert self._br().culvert_groups[0].downstream_invert == pytest.approx(25.0)

    def test_culvert_ds_station(self):
        assert self._br().culvert_groups[0].downstream_station == pytest.approx(1000.0)

    def test_culvert_chart_number(self):
        assert self._br().culvert_groups[0].chart_number == 5

    def test_culvert_n_bottom(self):
        assert self._br().culvert_groups[0].n_bottom == pytest.approx(0.03)

    def test_culvert_depth_n_bottom(self):
        assert self._br().culvert_groups[0].depth_n_bottom == pytest.approx(0.5)

    def test_roadway_stations_up(self):
        rw = self._br().roadway
        assert len(rw.stations_up) == 8
        assert rw.stations_up[0] == pytest.approx(856.0)
        assert rw.stations_up[-1] == pytest.approx(1150.0)

    def test_roadway_hi_chord_up(self):
        rw = self._br().roadway
        assert rw.hi_chord_up[0] == pytest.approx(36.1)
        assert rw.hi_chord_up[-1] == pytest.approx(37.2)

    def test_roadway_lo_chord_up_blank(self):
        # conspan lo-chord row is all spaces → parsed as empty list
        assert self._br().roadway.lo_chord_up == []

    def test_roadway_min_lo_chord(self):
        assert self._br().roadway.min_lo_chord == pytest.approx(33.7)


# ---------------------------------------------------------------------------
# Multiple barrel culvert — 3reach_lat.g01 (Fall River / Upper Reach / 10.1)
# ---------------------------------------------------------------------------


class TestMultipleBarrelCulvert:
    """Culvert= + Multiple Barrel Culv= from 3reach_lat.g01.

    Culvert=1,8,,100,...  → Circular, 1 barrel
    Multiple Barrel Culv=2,8,6,100,...  → Box, 2 barrels
    """

    def _br(self):
        g = GeometryFile(LAT3)
        return g.structures.bridges["Fall River Upper Reach 10.1"]

    def test_culvert_count(self):
        assert len(self._br().culvert_groups) == 2

    def test_first_culvert_shape(self):
        assert self._br().culvert_groups[0].shape_code == 1
        assert self._br().culvert_groups[0].shape_name == "Circular"

    def test_first_culvert_n_bottom(self):
        assert self._br().culvert_groups[0].n_bottom == pytest.approx(0.013)

    def test_second_culvert_shape(self):
        assert self._br().culvert_groups[1].shape_code == 2
        assert self._br().culvert_groups[1].shape_name == "Box"

    def test_second_culvert_num_barrels(self):
        assert self._br().culvert_groups[1].num_barrels == 2

    def test_second_culvert_n_bottom(self):
        assert self._br().culvert_groups[1].n_bottom == pytest.approx(0.013)

    def test_htab_hw_max(self):
        assert self._br().htab_hw_max == pytest.approx(90.0)

    def test_no_piers(self):
        assert self._br().piers == []


# ---------------------------------------------------------------------------
# Pier in 3reach_lat.g01 bridge (Butte Cr. / Butte Cr. / 0.22)
# ---------------------------------------------------------------------------


class TestBridgePiers3Reach:
    """Bridge with 1 pier group in 3reach_lat.g01.

    Pier at station 267, 2 upstream + 2 downstream elements.
    Up widths [2, 2], up elevations [65, 80].
    """

    def _br(self):
        g = GeometryFile(LAT3)
        return g.structures.bridges["Butte Cr. Butte Cr. 0.22"]

    def test_pier_count(self):
        assert len(self._br().piers) == 1

    def test_pier_station(self):
        assert self._br().piers[0].upstream_station == pytest.approx(267.0)
        assert self._br().piers[0].downstream_station == pytest.approx(267.0)

    def test_pier_counts(self):
        assert self._br().piers[0].upstream_count == 2
        assert self._br().piers[0].downstream_count == 2

    def test_pier_widths(self):
        assert self._br().piers[0].upstream_widths == pytest.approx([2.0, 2.0])

    def test_pier_elevations(self):
        assert self._br().piers[0].upstream_elevations == pytest.approx([65.0, 80.0])

    def test_htab_hw_max(self):
        assert self._br().htab_hw_max == pytest.approx(90.0)

    def test_htab_tw_max(self):
        assert self._br().htab_tw_max == pytest.approx(88.0)

    def test_no_culverts(self):
        assert self._br().culvert_groups == []


# ---------------------------------------------------------------------------
# Lateral extended fields — 3reach_lat.g01 (Fall River / Upper Reach / 10.25)
# ---------------------------------------------------------------------------


class TestLateralExtended:
    """New Lateral fields parsed from 3reach_lat.g01.

    Lateral Weir Pos= 0
    Lateral Weir Distance=10
    Lateral Weir Flap Gates= 0
    Lateral Weir TW Multiple XS=0
    Lateral Weir SE= 6  → (0,90),(10,90),(10,82),(100,82),(100,90),(110,90)
    """

    def _lat(self):
        g = GeometryFile(LAT3)
        return g.structures.laterals["Fall River Upper Reach 10.25"]

    def test_pos_left_bank(self):
        assert self._lat().pos == 0

    def test_distance(self):
        assert self._lat().distance == pytest.approx(10.0)

    def test_flap_gates_false(self):
        assert self._lat().flap_gates is False

    def test_tw_multiple_xs_false(self):
        assert self._lat().tw_multiple_xs is False

    def test_crest_profile_count(self):
        assert len(self._lat().crest_profile) == 6

    def test_crest_profile_first(self):
        assert self._lat().crest_profile[0] == pytest.approx((0.0, 90.0))

    def test_crest_profile_third(self):
        # station=10, elev=82 (weir dips here)
        assert self._lat().crest_profile[2] == pytest.approx((10.0, 82.0))

    def test_crest_profile_last(self):
        assert self._lat().crest_profile[-1] == pytest.approx((110.0, 90.0))
