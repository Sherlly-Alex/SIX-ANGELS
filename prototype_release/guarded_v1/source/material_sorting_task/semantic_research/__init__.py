"""Offline semantic-research modules.

These adapters never participate in robot control. Formal clients must not
import this package.
"""

from .schema import SemanticPrediction

__all__ = ["SemanticPrediction"]
