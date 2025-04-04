
# sh ./scripts/trj/ocir_learning_trj_FD001.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
runs=1
data_path_name='./datasets/cmapss_dataset'
data=FD001

random_seed=1024
run_id=4

batch_size=248
W=25

for (( i=0; i<${runs}; i++ ));
do
    python -u main.py \
    --seed $random_seed \
    --data_path $data_path_name \
    --description trj \
    --id_ $run_id \
    --dataset $data \
    --task data_trj \
    --gpu_dev 7 \
    --net ocir \
    --window $W \
    --hyper_lookback 2\
    --H 4 \
    --alpha 0.7 \
    --valid_split 0.2 \
    --kl_annealing 0.1 \
    --c_kl 1 \
    --time_embedding True \
    --dx 14 --dz 24 --dc 2 --d_model 168 --num_heads 4 \
    --z_projection aggregation_all \
    --D_projection aggregation \
    --c_posterior_param soft \
    --n_epochs 70 --fore_epochs 100 --batch $batch_size --lr_ 0.00015 --trj_lr 0.0004 \
    --scheduler 0 --warm_up 0.2 --final_lr 0.0002 --ref_lr 0.0002 --start_lr 0.0002 --start_wd 0.001 --final_wd 0.001
done
# --scheduler 0 --warm_up 0.2 --final_lr 0.0001 --ref_lr 0.0004 --start_lr 0.00005 --start_wd 0.01 --final_wd 0.0001