"""The bundled B2B SaaS reference domain.

A deterministic synthetic account-churn panel (``generate``) plus the downstream stages that
exercise the factory end-to-end on it: retention ``policy``, ``uplift``/``qini`` causal
targeting, and drift ``monitor``. Swap this package to retarget the factory at another domain.
"""

from __future__ import annotations
