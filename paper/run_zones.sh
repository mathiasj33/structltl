#!/bin/bash

set -e

pixi run -e gpu python scripts/eval/eval.py experiment=struct_ltl/zones_complex formulas=zones_complex/finite eval.finite=true eval.models_per_batch=5
pixi run -e gpu python scripts/eval/eval.py experiment=ltl2action/zones_complex formulas=zones_complex/finite eval.finite=true eval.models_per_batch=5
pixi run -e gpu python scripts/eval/eval.py experiment=deep_ltl/zones_complex formulas=zones_complex/finite eval.finite=true eval.models_per_batch=5
pixi run -e gpu python scripts/eval/eval.py experiment=genz_ltl/zones_complex formulas=zones_complex/finite eval.finite=true

pixi run -e gpu python scripts/eval/eval.py experiment=struct_ltl/zones_complex formulas=zones_complex/infinite eval.finite=false
pixi run -e gpu python scripts/eval/eval.py experiment=deep_ltl/zones_complex formulas=zones_complex/infinite eval.finite=false
pixi run -e gpu python scripts/eval/eval.py experiment=genz_ltl/zones_complex formulas=zones_complex/infinite eval.finite=false
