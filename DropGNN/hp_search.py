import os
#for hidden in range(3, 4):
for layers in range(1, 5):
        #for lr in range(1, 4):
        #    for wd in range(1, 4):
        #        for epoch in range(1, 4):
  script_base = (
      f"python test_BREC.py --SEED=100 --hidden_units {32} --num_layers {5} --augmentation random --random binary --loss nt_bxent_loss --loss_parameter 0.5 --name_tag regulardims --parts Regular "
      f"--OUTPUT_DIM=16 --BATCH_SIZE=16 --LEARNING_RATE={0.0001} --WEIGHT_DECAY={1e-05}  --device=0 --num_runs=100 --EPOCH={int(100)} --added_dimensions {layers} "
  )
  print(script_base)
  os.system(script_base)
