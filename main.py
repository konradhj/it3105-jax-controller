import yaml

import yaml

with open("config/default.yaml", "r") as f:
    cfg = yaml.safe_load(f)

print(cfg["training"]["epochs"])