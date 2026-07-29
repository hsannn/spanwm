# ==============================================================
# SpanWM v9: span unit = benepar CONSTITUENT (roles = NP/VP/PP labels),
# on top of v7's multi-span pooled test + splice fixes.
# v4-v7 modules untouched (v8 is developed independently).
# ==============================================================

from .spanwm_v9 import SpanWMV9, SpanWMV9Config

__all__ = ["SpanWMV9", "SpanWMV9Config"]
