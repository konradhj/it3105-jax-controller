import argparse
import yaml
import jax

import matplotlib.pyplot as plt

from consys.simulate import simulate


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to YAML config")
    parser.add_argument("--plot", action="store_true", help="Plot trajectories")
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    out = simulate(cfg)

    print(f"Run: {cfg['run']['name']}")
    print(f"Seed: {cfg['run']['seed']}")
    print(f"Plant: {cfg['plant']['name']}")
    print(f"Target: {cfg['plant']['target']}")
    print(f"Timesteps: {cfg['train']['timesteps']}")
    print(f"MSE: {float(out['mse']):.6f}")
    print(f"Final H: {float(out['final_H']):.6f}")

    if args.plot:
        H = out["H"]
        U = out["U"]
        D = out["D"]

        plt.figure()
        plt.plot(H)
        plt.title("Bathtub height H(t)")
        plt.xlabel("timestep")
        plt.ylabel("height")
        plt.grid(True)

        plt.figure()
        plt.plot(U)
        plt.title("Control input U(t)")
        plt.xlabel("timestep")
        plt.ylabel("U")
        plt.grid(True)

        plt.figure()
        plt.plot(D)
        plt.title("Disturbance D(t)")
        plt.xlabel("timestep")
        plt.ylabel("D")
        plt.grid(True)

        plt.show()


if __name__ == "__main__":
    main()
