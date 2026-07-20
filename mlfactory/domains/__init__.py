"""Domain instances that plug into the generic factory.

A domain may depend on the core (config / artifacts / compute); the core never depends on a
domain. The bundled reference domain is ``saas`` (B2B SaaS account churn).
"""

from __future__ import annotations
