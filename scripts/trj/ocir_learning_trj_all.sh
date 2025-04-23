
# sh ./scripts/trj/ocir_learning_trj_all.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
gpu=1
runs=1
data_path_name='./datasets/cmapss_dataset'
data=FD001

random_seed=1024
run_id=4

batch_size=248
W=25
logpath=./Logss_trj2/logs_
for (( i=0; i<${runs}; i++ ));
do
    python -u main.py \
    --seed $random_seed \
    --data_path $data_path_name \
    --description trj \
    --log_path $logpath \
    --id_ $run_id \
    --dataset $data \
    --task data_trj \
    --gpu_dev $gpu \
    --net ocir \
    --window $W \
    --hyper_lookback 1\
    --H 4 \
    --alpha 0.7 \
    --valid_split 0.2 \
    --kl_annealing 0.05 \
    --c_kl 1 \
    --time_embedding True \
    --dx 14 --dz 24 --dc 2 --d_model 168 --num_heads 4 \
    --z_projection aggregation_all \
    --D_projection aggregation \
    --c_posterior_param soft \
    --n_epochs 100 --fore_epochs 100 --batch $batch_size --lr_ 0.00015 --trj_lr 0.0004 --rul_lr 0.0004 --rul_epochs 100 \
    --scheduler 0 --warm_up 0.2 --final_lr 0.0002 --ref_lr 0.0002 --start_lr 0.0002 --start_wd 0.001 --final_wd 0.001
done

runs=1
data_path_name='./datasets/cmapss_dataset'
data=FD002

random_seed=1024
run_id=1

batch_size=368
W=25

for (( i=0; i<${runs}; i++ ));
do
    python -u main.py \
    --seed $random_seed \
    --data_path $data_path_name \
    --description trj \
    --log_path $logpath \
    --id_ $run_id \
    --dataset $data \
    --task data_trj \
    --gpu_dev $gpu \
    --net ocir \
    --window $W \
    --hyper_lookback 1\
    --H 4 \
    --alpha 0.7 \
    --valid_split 0.2 \
    --kl_annealing 0.05 \
    --time_embedding True \
    --dx 14 --dz 24 --dc 6 --d_model 168 --num_heads 4 \
    --z_projection aggregation_all \
    --D_projection aggregation \
    --c_posterior_param soft \
    --n_epochs 120 --fore_epochs 120 --batch $batch_size --lr_ 0.00015 --trj_lr 0.0004 --rul_lr 0.0004 --rul_epochs 100 \
    --scheduler 0 --warm_up 0.2 --final_lr 0.0002 --ref_lr 0.0002 --start_lr 0.0002 --start_wd 0.001 --final_wd 0.001
done

runs=1
data_path_name='./datasets/cmapss_dataset'
data=FD003

random_seed=1024
run_id=2

batch_size=248
W=25


for (( i=0; i<${runs}; i++ ));
do
    python -u main.py \
    --seed $random_seed \
    --data_path $data_path_name \
    --description trj \
    --log_path $logpath \
    --id_ $run_id \
    --dataset $data \
    --task data_trj \
    --gpu_dev $gpu \
    --net ocir \
    --window $W \
    --hyper_lookback 1\
    --H 4 \
    --alpha 0.7 \
    --valid_split 0.2 \
    --kl_annealing 0.05 \
    --c_kl 1 \
    --time_embedding True \
    --dx 14 --dz 24 --dc 2 --d_model 168 --num_heads 4 \
    --z_projection aggregation_all \
    --D_projection aggregation \
    --c_posterior_param soft \
    --n_epochs 120 --fore_epochs 140 --batch $batch_size --lr_ 0.000135 --trj_lr 0.0004 --rul_lr 0.0004 --rul_epochs 100 \
    --scheduler 0 --warm_up 0.2 --final_lr 0.0002 --ref_lr 0.0002 --start_lr 0.0002 --start_wd 0.001 --final_wd 0.001
done

runs=1
data_path_name='./datasets/cmapss_dataset'
data=FD004

random_seed=1024
run_id=1

batch_size=368
W=25

for (( i=0; i<${runs}; i++ ));
do
    python -u main.py \
    --seed $random_seed \
    --data_path $data_path_name \
    --description trj \
    --log_path $logpath \
    --id_ $run_id \
    --dataset $data \
    --task data_trj \
    --gpu_dev $gpu \
    --net ocir \
    --window $W \
    --hyper_lookback 1\
    --H 4 \
    --alpha 0.7 \
    --valid_split 0.2 \
    --kl_annealing 0.05 \
    --time_embedding True \
    --dx 14 --dz 24 --dc 6 --d_model 168 --num_heads 4 \
    --z_projection aggregation_all \
    --D_projection aggregation \
    --c_posterior_param soft \
    --n_epochs 120 --fore_epochs 120 --batch $batch_size --lr_ 0.00015 --trj_lr 0.0004 --rul_lr 0.0004 --rul_epochs 100 \
    --scheduler 0 --warm_up 0.2 --final_lr 0.0002 --ref_lr 0.0002 --start_lr 0.0002 --start_wd 0.001 --final_wd 0.001
done