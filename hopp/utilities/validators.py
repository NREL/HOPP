"""
This module contains validator functions for use with `attrs` class definitions.
"""

def gt_zero(instance, attribute, value):
    """Validates that an attribute's value is greater than zero."""
    if value <= 0:
        raise ValueError("system_capacity_kw must be greater than zero")
