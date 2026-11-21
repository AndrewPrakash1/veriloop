"""One module per workflow family. Each exposes TEMPLATES: dict[name, callable].

A template is a pure function of (rng, difficulty) -> Scenario. It must not read the
clock, the network or global state.
"""

TEMPLATES: dict = {}
