# ====================================================
# SpanWM v8: v7 (multi-span pooled test + splice fixes) with a TWO-ROLE anchor
# PRF — the per-span key admits a k-subset of the role list (roles_per_anchor,
# default 2) instead of a single role, roughly doubling the density of
# eligible anchors so more drafts reach max_spans sites.
# Separate module; v3/v4/v5/v6/v7 stay untouched.
# ====================================================

from .spanwm_v8 import SpanWMV8, SpanWMV8Config
