import ps4
import json, pathlib, sys
import pprint
from typing import *

SETUPS_FILEPATH = pathlib.Path("./experiments.json").resolve()

def get_setups(path: pathlib.Path = SETUPS_FILEPATH) -> Dict:
    f = path.open("r")
    obj: Dict = json.load(f)
    f.close()
    return obj

def main(args: List[str]) -> int:
    experiments = get_setups()
    if experiments["configs"]["verbosity"] > 0:
        pprint.pprint(experiments)
    if experiments["configs"]["seed"] is not None:
        ps4.use_seed(experiments["configs"]["seed"])
    for index, setup in enumerate(experiments["without"]):
        print("Running simple setup {0}...".format(index + 1))
        chronology = ps4.simulation_without_antibiotic(**setup)
        mean, width = ps4.calc_95_ci(chronology, 399)
        print(
            "Expected number of bacteria at timestep 399: {0}±{1}".format(
                mean, width/2
            )
        )
    for index, setup in enumerate(experiments["with"]):
        print("Running complex setup {0}...".format(index + 1))
        t_chronology, r_chronology = ps4.simulation_with_antibiotic(**setup)
        mean, width = ps4.calc_95_ci(t_chronology, 399)
        print(
            "Expected number of bacteria at timestep 399: {0}±{1}".format(
                mean, width/2
            )
        )
        mean, width = ps4.calc_95_ci(r_chronology, 399)
        print(
            "Expected number of resistant at timestep 399: {0}±{1}".format(
                mean, width/2
            )
        )
    return 0

if __name__ == "__main__":
    exit(main(sys.argv))
