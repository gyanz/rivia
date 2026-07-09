"""Tests for rivia.utils.helpers — parse_interval_strict / format_interval_strict."""

import datetime as dt

import pytest

from rivia.utils import format_interval_strict, parse_interval, parse_interval_strict


class TestParseIntervalStrict:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("1SEC", dt.timedelta(seconds=1)),
            ("30SEC", dt.timedelta(seconds=30)),
            ("1SECOND", dt.timedelta(seconds=1)),
            ("30SECOND", dt.timedelta(seconds=30)),
            ("5MIN", dt.timedelta(minutes=5)),
            ("30MIN", dt.timedelta(minutes=30)),
            ("5MINUTE", dt.timedelta(minutes=5)),
            ("30MINUTE", dt.timedelta(minutes=30)),
            ("1HOUR", dt.timedelta(hours=1)),
            ("1HR", dt.timedelta(hours=1)),
            ("12HOUR", dt.timedelta(hours=12)),
            ("1DAY", dt.timedelta(days=1)),
            ("1WEEK", dt.timedelta(weeks=1)),
            ("1MONTH", dt.timedelta(days=30)),
            ("1YEAR", dt.timedelta(days=365)),
        ],
    )
    def test_allowed_values_parse(self, text, expected):
        assert parse_interval_strict(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "7HOUR",  # not in HOUR's allowed set
            "7SEC",  # not in SEC's allowed set
            "7SECOND",  # not in SECOND's allowed set
            "7MIN",  # not in MIN's allowed set
            "7MINUTE",  # not in MINUTE's allowed set
            "2DAY",  # DAY only allows 1
            "2WEEK",  # WEEK only allows 1
            "2MONTH",  # MONTH only allows 1
            "2YEAR",  # YEAR only allows 1
        ],
    )
    def test_disallowed_values_raise(self, text):
        with pytest.raises(ValueError):
            parse_interval_strict(text)

    def test_non_integer_value_raises(self):
        with pytest.raises(ValueError):
            parse_interval_strict("1.5MIN")

    def test_unrecognised_unit_raises(self):
        with pytest.raises(ValueError):
            parse_interval_strict("5FORTNIGHT")

    def test_malformed_text_raises(self):
        with pytest.raises(ValueError):
            parse_interval_strict("not-an-interval")

    def test_lenient_parse_interval_still_accepts_values_strict_rejects(self):
        # Documents the relationship between the two functions.
        assert parse_interval("7HOUR") == dt.timedelta(hours=7)
        with pytest.raises(ValueError):
            parse_interval_strict("7HOUR")


class TestFormatIntervalStrict:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (dt.timedelta(seconds=30), "30SEC"),
            (dt.timedelta(minutes=15), "15MIN"),
            (dt.timedelta(hours=2), "2HOUR"),
            (dt.timedelta(hours=1), "1HOUR"),
            (dt.timedelta(days=1), "1DAY"),
            (dt.timedelta(weeks=1), "1WEEK"),
        ],
    )
    def test_formats_allowed_durations(self, value, expected):
        assert format_interval_strict(value) == expected

    def test_bare_number_treated_as_seconds(self):
        assert format_interval_strict(3600) == "1HOUR"
        assert format_interval_strict(900.0) == "15MIN"

    def test_prefers_largest_valid_unit(self):
        # 3600s could be "60MIN" but MIN's allowed set caps at 30, so it
        # must resolve to "1HOUR" instead.
        assert format_interval_strict(dt.timedelta(hours=1)) == "1HOUR"

    def test_falls_back_to_month_when_day_unavailable(self):
        # DAY only allows 1, so a 30-day duration falls back to MONTH.
        assert format_interval_strict(dt.timedelta(days=30)) == "1MONTH"

    def test_falls_back_to_week_when_day_unavailable(self):
        assert format_interval_strict(dt.timedelta(days=7)) == "1WEEK"

    def test_unrepresentable_duration_raises(self):
        # 5 hours: not in HOUR's set, and not evenly expressible in
        # MIN/SEC's allowed sets either.
        with pytest.raises(ValueError):
            format_interval_strict(dt.timedelta(hours=5))

    def test_two_days_raises(self):
        # DAY only allows 1, and 2 days isn't otherwise representable.
        with pytest.raises(ValueError):
            format_interval_strict(dt.timedelta(days=2))

    def test_non_positive_raises(self):
        with pytest.raises(ValueError):
            format_interval_strict(dt.timedelta(0))
        with pytest.raises(ValueError):
            format_interval_strict(-5)

    def test_round_trip_through_parse_interval_strict(self):
        for value in [
            dt.timedelta(seconds=20),
            dt.timedelta(minutes=10),
            dt.timedelta(hours=8),
            dt.timedelta(days=1),
        ]:
            text = format_interval_strict(value)
            assert parse_interval_strict(text) == value
