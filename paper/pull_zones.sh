#!/bin/bash

set -e

# rsync -avzP desktop:work/dphil/jaxltl/runs/ZoneEnvComplex/genz_ltl/main/eval_results_checkpoints.csv runs/ZoneEnvComplex/genz_ltl/main/eval_results_checkpoints.csv
# rsync -avzP desktop:work/dphil/jaxltl/runs/ZoneEnvComplex/struct_ltl/main/eval_results_checkpoints.csv runs/ZoneEnvComplex/struct_ltl/main/eval_results_checkpoints.csv
# rsync -avzP desktop:work/dphil/jaxltl/runs/ZoneEnvComplex/deep_ltl/main/eval_results_checkpoints.csv runs/ZoneEnvComplex/deep_ltl/main/eval_results_checkpoints.csv
# rsync -avzP desktop:work/dphil/jaxltl/runs/ZoneEnvComplex/ltl2action/main/eval_results_checkpoints.csv runs/ZoneEnvComplex/ltl2action/main/eval_results_checkpoints.csv

rsync -avzP desktop:work/dphil/jaxltl/runs/ZoneEnvComplex/genz_ltl/main/results_infinite.csv runs/ZoneEnvComplex/genz_ltl/main/results_infinite.csv
rsync -avzP desktop:work/dphil/jaxltl/runs/ZoneEnvComplex/struct_ltl/main/results_infinite.csv runs/ZoneEnvComplex/struct_ltl/main/results_infinite.csv
rsync -avzP desktop:work/dphil/jaxltl/runs/ZoneEnvComplex/deep_ltl/main/results_infinite.csv runs/ZoneEnvComplex/deep_ltl/main/results_infinite.csv
# rsync -avzP desktop:work/dphil/jaxltl/runs/WarehouseEnv/genz_ltl/main/results_infinite.csv runs/WarehouseEnv/genz_ltl/main/results_infinite.csv
# rsync -avzP desktop:work/dphil/jaxltl/runs/WarehouseEnv/struct_ltl/main/results_infinite.csv runs/WarehouseEnv/struct_ltl/main/results_infinite.csv