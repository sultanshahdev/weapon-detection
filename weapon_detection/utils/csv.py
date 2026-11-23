"""Protect spreadsheet consumers from formula interpretation."""


def spreadsheet_safe(value: object) -> object:
    if isinstance(value, str) and (
        value.startswith(("\t", "\r", "\n")) or value.lstrip().startswith(("=", "+", "-", "@"))
    ):
        return "'" + value
    return value
