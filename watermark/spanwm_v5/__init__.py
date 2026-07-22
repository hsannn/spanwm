# ====================================================
# SpanWM v5: v3 recipe (base + left-AR + fixed-K) with PER-SPAN keyed role
# selection — role = PRF(master_key, n-gram preceding the span) instead of one
# fixed role per key. Separate module so v3 stays untouched.
# ====================================================

from .spanwm_v5 import SpanWMV5, SpanWMV5Config
