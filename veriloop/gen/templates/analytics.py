"""Domain B — analytical querying (Track 3 PS 3.6).

A business question becomes SQL becomes a verified result row. Verification is by
re-executing the query against the frozen world and comparing to the gold result — the
oracle is execution, not a judge.

Families: single-table aggregate · join · filtered time window · top-k ranking ·
ambiguous question requiring a clarification tool call rather than a guess.
"""

TEMPLATES: dict = {}
