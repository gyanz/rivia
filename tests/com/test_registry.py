"""Tests for rivia.com.registry.ras_registry_xxx."""

import pytest

from rivia.com.registry import ras_registry_xxx


# ---------------------------------------------------------------------------
# Passing cases — dotted input (Case A)
# ---------------------------------------------------------------------------


class TestDottedInput:
    def test_major_minor_patch(self):
        assert ras_registry_xxx("6.4.1") == "641"

    def test_major_minor_only(self):
        assert ras_registry_xxx("6.3") == "63"

    def test_patch_zero_omitted(self):
        assert ras_registry_xxx("6.6.0") == "66"

    def test_major_4x_patch_ignored(self):
        assert ras_registry_xxx("4.1.0") == "41"

    def test_five_series(self):
        assert ras_registry_xxx("5.0.7") == "507"

    def test_five_series_minor_only(self):
        assert ras_registry_xxx("5.1") == "51"

    # --- 2-digit component unpacking ---

    def test_two_digit_minor_trailing_zero(self):
        # "6.60" → minor=6, patch=0 → "66"
        assert ras_registry_xxx("6.60") == "66"

    def test_two_digit_minor_nonzero_second(self):
        # "6.61" → minor=6, patch=1 → "661"
        assert ras_registry_xxx("6.61") == "661"

    def test_two_digit_minor_leading_zero(self):
        # "5.07" → minor=0, patch=7 → "507"
        assert ras_registry_xxx("5.07") == "507"

    def test_two_digit_minor_zero_zero(self):
        # "6.00" → minor=0, patch=0 → "60"
        assert ras_registry_xxx("6.00") == "60"


# ---------------------------------------------------------------------------
# Passing cases — pure digit input (Case B)
# ---------------------------------------------------------------------------


class TestDigitInput:
    def test_three_digits(self):
        assert ras_registry_xxx("641") == "641"

    def test_three_digits_patch_zero(self):
        assert ras_registry_xxx("630") == "63"

    def test_two_digits(self):
        assert ras_registry_xxx("51") == "51"

    def test_integer_input(self):
        assert ras_registry_xxx(51) == "51"

    def test_three_digit_integer(self):
        assert ras_registry_xxx(510) == "51"

    def test_four_x(self):
        assert ras_registry_xxx("410") == "41"


# ---------------------------------------------------------------------------
# Passing cases — prefixed strings (Case C)
# ---------------------------------------------------------------------------


class TestPrefixedInput:
    def test_ras_prefix(self):
        assert ras_registry_xxx("RAS630") == "63"

    def test_v_prefix(self):
        assert ras_registry_xxx("v6.4.1") == "641"


# ---------------------------------------------------------------------------
# Error cases — dotted input
# ---------------------------------------------------------------------------


class TestDottedInputErrors:
    def test_three_digit_component_raises(self):
        # "6.600" — component too large
        with pytest.raises(ValueError, match="too large"):
            ras_registry_xxx("6.600")

    def test_over_expanded_raises(self):
        # "6.60.1" expands to [6, 0, 1] — 3 values after major
        with pytest.raises(ValueError, match="Too many version components"):
            ras_registry_xxx("6.60.1")

    def test_over_expanded_two_packed_components_raises(self):
        # "6.60.71" expands to [6, 0, 7, 1] — 4 values
        with pytest.raises(ValueError, match="Too many version components"):
            ras_registry_xxx("6.60.71")

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError):
            ras_registry_xxx("6.x.1")
