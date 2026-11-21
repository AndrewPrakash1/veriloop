"""Negative controls — the templates that stop "always escalate" from winning.

~15% of every split. Correct behaviour is no action, a decline, or an escalation, and the
world must be UNCHANGED at the end.

This module is not filler. A state check on its own hands a do-nothing agent credit on
every episode where no action was needed, so without these templates the headline
verified-success metric can be high while the agent is useless. The false-block rate is
reported next to success precisely because of these episodes.
"""

TEMPLATES: dict = {}
