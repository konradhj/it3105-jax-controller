import argparse
import yaml

from consys.simulate import simulate
from consys.train import train
from vizualisation.plots import plot_mse, plot_pid_params


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to YAML config")
    parser.add_argument("--plot", action="store_true", help="Plot trajectories")
    parser.add_argument("--train", action="store_true", help="Run training loop")
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    if args.train:
        # TODO: call train(), print summary, plot MSE + PID params
        raise NotImplementedError

    # TODO: call simulate(), print summary, plot trajectories
    raise NotImplementedError


if __name__ == "__main__":
    main()
