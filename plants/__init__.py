from plants.bathtub import Bathtub
from plants.cournot import Cournot
from plants.logistic import LogisticPopulation

_PLANTS = {
    "bathtub": Bathtub(),
    "cournot": Cournot(),
    "logistic": LogisticPopulation(),
}

def get_plant(name: str):
    try:
        return _PLANTS[name]
    except KeyError:
        raise ValueError(f"Unknown plant '{name}'. Available: {list(_PLANTS.keys())}")
