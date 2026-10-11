"""Strict validation of proposed opinion units and their verbatim evidence."""


def validate_opinions(payload, text, taxonomy):
    if not isinstance(payload, dict) or set(payload) != {"opinions"}:
        raise ValueError("Expected only the opinions field")
    if not isinstance(payload["opinions"], list):
        raise ValueError("opinions must be a list")
    fields = {"aspect", "sentiment", "intent", "evidence"}
    for unit in payload["opinions"]:
        if not isinstance(unit, dict) or set(unit) != fields:
            raise ValueError("Invalid opinion fields")
        if unit["aspect"] not in taxonomy["aspects"]:
            raise ValueError("Unknown aspect")
        if unit["sentiment"] not in taxonomy["sentiments"]:
            raise ValueError("Unknown sentiment")
        if unit["intent"] not in taxonomy["intents"]:
            raise ValueError("Unknown intent")
        evidence = unit["evidence"]
        if not isinstance(evidence, str) or not evidence.strip() or evidence not in text:
            raise ValueError("Evidence must be a nonempty verbatim substring")
    return payload
