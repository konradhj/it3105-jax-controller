from plants.bathtub import reset as bathtub_reset, step as bathtub_step
# later: from plants.tank2 import reset as tank2_reset, step as tank2_step

_PLANTS = {
    "bathtub": (bathtub_reset, bathtub_step),
    # "tank2": (tank2_reset, tank2_step),
}

def get_plant(name: str):
    try:
        return _PLANTS[name]
    except KeyError:
        raise ValueError(f"Unknown plant '{name}'. Available: {list(_PLANTS.keys())}")
