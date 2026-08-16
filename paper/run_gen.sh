#!/bin/bash

set -e

pixi run -e gpu python scripts/eval/eval.py experiment=struct_ltl/warehouse run=main formulas=warehouse/reach-2 eval.finite=true
pixi run -e gpu python scripts/eval/eval.py experiment=struct_ltl/warehouse run=main formulas=warehouse/reach-4 eval.finite=true
pixi run -e gpu python scripts/eval/eval.py experiment=struct_ltl/warehouse run=main formulas=warehouse/reach-8 eval.finite=true

pixi run -e gpu python scripts/eval/eval.py experiment=genz_ltl/warehouse run=main formulas=warehouse/reach-2 eval.finite=true
pixi run -e gpu python scripts/eval/eval.py experiment=genz_ltl/warehouse run=main formulas=warehouse/reach-4 eval.finite=true
pixi run -e gpu python scripts/eval/eval.py experiment=genz_ltl/warehouse run=main formulas=warehouse/reach-8 eval.finite=true

pixi run -e gpu python scripts/eval/eval.py experiment=deep_ltl/warehouse run=main formulas=warehouse/reach-2 eval.finite=true
pixi run -e gpu python scripts/eval/eval.py experiment=deep_ltl/warehouse run=main formulas=warehouse/reach-4 eval.finite=true
pixi run -e gpu python scripts/eval/eval.py experiment=deep_ltl/warehouse run=main formulas=warehouse/reach-8 eval.finite=true
pixi run -e gpu python scripts/eval/eval.py experiment=deep_ltl/warehouse run=main formulas=warehouse/reach-12 eval.finite=true

pixi run -e gpu python scripts/eval/eval.py experiment=ltl2action/warehouse run=main formulas=warehouse/reach-2 eval.finite=true
pixi run -e gpu python scripts/eval/eval.py experiment=ltl2action/warehouse run=main formulas=warehouse/reach-4 eval.finite=true
pixi run -e gpu python scripts/eval/eval.py experiment=ltl2action/warehouse run=main formulas=warehouse/reach-8 eval.finite=true
pixi run -e gpu python scripts/eval/eval.py experiment=ltl2action/warehouse run=main formulas=warehouse/reach-12 eval.finite=true