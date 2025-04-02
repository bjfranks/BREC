import os
SEED = [100]#, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
for seed in SEED:
    script_base = (
        f"python test_BREC.py --SEED={seed} --hidden_units 32 --num_layers 6 --augmentation random --random binary --loss nt_bxent_loss --loss_parameter 1.0 --name_tag test --parts Regular "
        f"--OUTPUT_DIM=16 --BATCH_SIZE=16 --LEARNING_RATE=0.0001 --WEIGHT_DECAY=1e-05  --device=0 --num_runs=1 --EPOCH=100 --added_dimensions 2"
    )
    print(script_base)
    os.system(script_base)
