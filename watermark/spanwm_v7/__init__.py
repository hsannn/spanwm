# ====================================================
# SpanWM v7: v6 (multi-span pooled test) + splice fixes —
#   (a) anchor-boundary retokenization drift: generate from the space-stripped
#       left context so the model emits the leading-space token itself;
#   (b) no space inserted before punctuation when splicing the right context.
# Separate module; v3/v4/v5/v6 stay untouched.
# ====================================================

from .spanwm_v7 import SpanWMV7, SpanWMV7Config
