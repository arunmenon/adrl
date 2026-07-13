"""Shared secret detection used by replay and the live privacy pin."""

from router.privacy import scan_content, scan_text


def test_high_confidence_secret_is_detected_without_returning_value():
    secret = "AKIAABCDEFGHIJKLMNOP"
    hits = scan_text(f"export AWS_ACCESS_KEY_ID={secret}")
    assert "aws_access_key" in hits
    assert secret not in repr(hits)


def test_placeholder_connection_string_is_not_a_secret():
    assert scan_text("postgres://user:password@localhost/app") == []


def test_nested_tool_result_content_is_scanned():
    content = [{
        "type": "tool_result",
        "content": [{"type": "text", "text": "API_SECRET=Ab9xK2mQ7vLp4TzN8w"}],
    }]
    assert scan_content(content) == ["env_assignment"]
