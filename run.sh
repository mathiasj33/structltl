pixi run -e gpu python scripts/train.py experiment=struct_ltl/zones run=vf num_seeds=10 save_freq=200000
pixi run -e gpu python scripts/train.py experiment=genz_ltl/warehouse run=first num_seeds=10 save_freq=2000000
pixi run -e gpu python scripts/eval/compute_eval_curves.py experiment=struct_ltl/zones run=vf +formulas=zones/finite
pixi run -e gpu python scripts/eval/compute_eval_curves.py experiment=genz_ltl/warehouse run=first +formulas=warehouse/finite
