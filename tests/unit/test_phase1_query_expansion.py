"""Phase 1 unit tests: identifier splitting and abbreviation expansion."""

from __future__ import annotations

from codexlens_search.search.expansion import _split_identifier, _split_identifiers


class TestSplitIdentifier:
    def test_camel_case(self) -> None:
        assert _split_identifier("getUserName") == ["get", "user", "name"]

    def test_snake_case(self) -> None:
        assert _split_identifier("user_auth_token") == ["user", "auth", "token"]


class TestSplitIdentifiers:
    def test_abbrev_and_identifier_terms(self) -> None:
        terms = _split_identifiers("Fix auth cfg in getUserName")
        # Abbreviations
        assert "authentication" in terms
        assert "config" in terms
        # Identifier parts
        assert "get" in terms
        assert "user" in terms
        assert "name" in terms
