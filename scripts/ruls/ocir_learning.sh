
# sh ./scripts/ruls/ocir_learning.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
runs=1
data_path_name='./datasets/cmapss_dataset'
data=FD003

random_seed=1024
run_id=222

batch_size=80
W=25

for (( i=0; i<${runs}; i++ ));
do
    python -u main.py \
    --seed $random_seed \
    --data_path $data_path_name \
    --description ocir_spc \
    --id_ $run_id \
    --dataset $data \
    --task RL \
    --gpu_dev 6 \
    --window $W \
    --time_embedding True \
    --dx 14 --dz 32 --dc 2 --d_model 128 --num_heads 4 \
    --z_projection aggregation_all \
    --D_projection spc \
    --c_posterior_param soft \
    --n_epochs 40 --batch $batch_size --lr_ 0.00015 \
    --scheduler 0 --warm_up 0.2 --final_lr 0.0001 --ref_lr 0.0004 --start_lr 0.0001 --start_wd 0.001 --final_wd 0.0001
done