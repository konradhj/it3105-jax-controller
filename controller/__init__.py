from controller.pid_classic import PIDClassic
from controller.pid_nn import PIDNN

_CONTROLLERS = {
    "pid_classic": PIDClassic(),
    "pid_nn": PIDNN(),
}


def get_controller(name: str):
    try:
        return _CONTROLLERS[name]
    except KeyError:
        raise ValueError(f"Unknown controller '{name}'. Available: {list(_CONTROLLERS.keys())}")
