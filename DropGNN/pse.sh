#!/usr/bin/bash --login


HOME_DIR=$(dirname $(realpath $0))
echo HOME_DIR=$HOME_DIR

cd $HOME_DIR

run_script="python test_BREC.py --SEED=100 --hidden_units 16 --num_layers 10 --OUTPUT_DIM 16 --BATCH_SIZE 16 --LEARNING_RATE 0.0001 --WEIGHT_DECAY 1e-05 --num_runs 1 --EPOCH 100 --root results"

mkdir -p results/slurm_history
run_script="sbatch -c 5 -o ${HOME_DIR}/results/slurm_history/slurm-%A.out -e ${HOME_DIR}/results/slurm_history/slurm-%A.err wrapper_rptu.sb ${run_script}"

launch () {
    command=$1
    echo $command  # print out the command
    eval $command  # execute the command
}

do_all () {
    launch "$1 --loss nt_bxent_loss --parts Basic --name_tag xb$2"
    launch "$1 --loss nt_bxent_loss --parts Regular --name_tag xr$2"
    launch "$1 --loss nt_bxent_loss --parts Extension --name_tag xe$2"
    launch "$1 --loss nt_bxent_loss --parts CFI --name_tag xc$2"
    launch "$1 --loss nt_bxent_loss --parts 4-Vertex_Condition --name_tag x4$2"
    launch "$1 --loss nt_bxent_loss --parts Distance_Regular --name_tag xd$2"
    launch "$1 --loss nt_bxent_loss --parts CCoHG --name_tag xh$2"
    launch "$1 --loss nt_bxent_loss --parts 3r2r --name_tag x3$2"
    launch "$1 --loss CosineEmbeddingLoss --parts Basic --name_tag cb$2"
    launch "$1 --loss CosineEmbeddingLoss --parts Regular --name_tag cr$2"
    launch "$1 --loss CosineEmbeddingLoss --parts Extension --name_tag ce$2"
    launch "$1 --loss CosineEmbeddingLoss --parts CFI --name_tag cc$2"
    launch "$1 --loss CosineEmbeddingLoss --parts 4-Vertex_Condition --name_tag c4$2"
    launch "$1 --loss CosineEmbeddingLoss --parts Distance_Regular --name_tag cd$2"
    launch "$1 --loss CosineEmbeddingLoss --parts CCoHG --name_tag ch$2"
    launch "$1 --loss CosineEmbeddingLoss --parts 3r2r --name_tag c3$2"
    launch "$1 --loss CosineEmbeddingLoss --parts Basic --loss_parameter 0.5 --name_tag clb$2"
    launch "$1 --loss CosineEmbeddingLoss --parts Regular --loss_parameter 0.5 --name_tag clr$2"
    launch "$1 --loss CosineEmbeddingLoss --parts Extension --loss_parameter 0.5 --name_tag cle$2"
    launch "$1 --loss CosineEmbeddingLoss --parts CFI --loss_parameter 0.5 --name_tag clc$2"
    launch "$1 --loss CosineEmbeddingLoss --parts 4-Vertex_Condition --loss_parameter 0.5 --name_tag cl4$2"
    launch "$1 --loss CosineEmbeddingLoss --parts Distance_Regular --loss_parameter 0.5 --name_tag cld$2"
    launch "$1 --loss CosineEmbeddingLoss --parts CCoHG --loss_parameter 0.5 --name_tag clh$2"
    launch "$1 --loss CosineEmbeddingLoss --parts 3r2r --loss_parameter 0.5 --name_tag cl3$2"
}

if [[ 0 == $1 ]]; then
    do_all "${run_script} --augmentation none" "none"
    do_all "${run_script} --augmentation PSE --pse RWSE" "RWSE"
    do_all "${run_script} --augmentation PSE --pse ElstaticPE" "ElstaticPE"
    do_all "${run_script} --augmentation PSE --pse RElstaticPE" "RElstaticPE"
    do_all "${run_script} --augmentation PSE --pse HKdiagSE" "HKdiagSE"
    do_all "${run_script} --augmentation PSE --pse RHKdiagSE" "RHKdiagSE"
    do_all "${run_script} --augmentation PSE --pse LapPE" "LapPE"
    do_all "${run_script} --augmentation PSE --pse RLapPE" "RLapPE"
    do_all "${run_script} --augmentation PSE --pse CycleSE" "CycleSE"
    do_all "${run_script} --augmentation PSE --pse SPDPE" "SPDPE"
    do_all "${run_script} --augmentation PSE --pse RDPE" "RDPE"
else
    launch "${run_script} --augmentation PSE --pse RWSE"
    launch "${run_script} --augmentation PSE --pse ElstaticPE"
    launch "${run_script} --augmentation PSE --pse HKdiagSE"
    launch "${run_script} --augmentation PSE --pse LapPE"
    launch "${run_script} --augmentation PSE --pse CycleSE"
    launch "${run_script} --augmentation PSE --pse SPDPE"
    launch "${run_script} --augmentation PSE --pse RDPE"
fi
