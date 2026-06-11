from . import gh8_v3, gh9_v3, h8_v3, h9_v6

ALL_PRESETS = [
    *gh8_v3.PRESETS,
    *gh9_v3.PRESETS,
    *h8_v3.PRESETS,
    *h9_v6.PRESETS,
]

BOTH = [
    gh8_v3.both,
    gh9_v3.both,
    h8_v3.both,
    h9_v6.both,
]

CD = [
    gh8_v3.cd,
    gh9_v3.cd,
    h8_v3.cd,
    h9_v6.cd,
]

VG = [
    gh8_v3.vg,
    gh9_v3.vg,
    h8_v3.vg,
    h9_v6.vg,
]
