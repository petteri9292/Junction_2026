"""Search preferences are interpretation hints, never source evidence."""


def normalize_focus(value):
    if not isinstance(value, str) or len(value) > 200:
        raise ValueError("search_focus must be a string of at most 200 characters")
    return value.strip()


def focused_question(question, focus=""):
    """Keep the original question and its intended context together downstream."""
    focus = normalize_focus(focus)
    return question if not focus else f"{question}\nRequested topic context: {focus}"
