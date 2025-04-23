
# sh ./scripts/rul/ocir_learning_rul_all.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
gpu=0
runs=1
data_path_name='./datasets/cmapss_dataset'
data=FD001

random_seed=1024
run_id=2222

batch_size=368
W=25

for (( i=0; i<${runs}; i++ ));
do
    python -u main.py \
    --seed $random_seed \
    --data_path $data_path_name \
    --description rul \
    --id_ $run_id \
    --dataset $data \
    --task rul \
    --gpu_dev $gpu \
    --net ocir \
    --window $W \
    --alpha 0.8 \
    --kl_annealing 0.05 \
    --c_kl 1 \
    --time_embedding False \
    --dx 14 --dz 24 --dc 2 --d_model 168 --num_heads 4 \
    --z_projection aggregation_all \
    --D_projection aggregation \
    --c_posterior_param soft \
    --n_epochs 120 --rul_epochs 100 --batch $batch_size --lr_ 0.00014 --rul_lr 0.0002 \
    --scheduler 0 --warm_up 0.2 --final_lr 0.0001 --ref_lr 0.00015 --start_lr 0.0001 --start_wd 0.001 --final_wd 0.01
done

data=FD002

random_seed=1024

batch_size=512
W=25

for (( i=0; i<${runs}; i++ ));
do
    python -u main.py \
    --seed $random_seed \
    --data_path $data_path_name \
    --description rul \
    --id_ $run_id \
    --dataset $data \
    --task rul \
    --gpu_dev $gpu \
    --net ocir \
    --window $W \
    --alpha 0.7 \
    --kl_annealing 0.05 \
    --time_embedding False \
    --dx 14 --dz 24 --dc 6 --d_model 168 --num_heads 4 \
    --z_projection aggregation_all \
    --D_projection aggregation \
    --c_posterior_param soft \
    --n_epochs 100 --rul_epochs 100 --batch $batch_size --lr_ 0.0002 --rul_lr 0.0002 \
    --scheduler 0 --warm_up 0.2 --final_lr 0.0001 --ref_lr 0.0002 --start_lr 0.00005 --start_wd 0.001 --final_wd 0.001
done

data=FD003
run_id=6666
random_seed=1024

batch_size=368
W=25

for (( i=0; i<${runs}; i++ ));
do
    python -u main.py \
    --seed $random_seed \
    --data_path $data_path_name \
    --description rul \
    --id_ $run_id \
    --dataset $data \
    --task rul \
    --gpu_dev $gpu \
    --net ocir \
    --window $W \
    --alpha 0.8 \
    --kl_annealing 0.05 \
    --c_kl 1 \
    --time_embedding True \
    --dx 14 --dz 24 --dc 2 --d_model 168 --num_heads 4 \
    --z_projection aggregation_all \
    --D_projection aggregation \
    --c_posterior_param soft \
    --n_epochs 120 --rul_epochs 100 --batch $batch_size --lr_ 0.00014 --rul_lr 0.00015 \
    --scheduler 0 --warm_up 0.2 --final_lr 0.0001 --ref_lr 0.00015 --start_lr 0.0001 --start_wd 0.001 --final_wd 0.01
done

data=FD004

random_seed=1024

batch_size=512
W=25

for (( i=0; i<${runs}; i++ ));
do
    python -u main.py \
    --seed $random_seed \
    --data_path $data_path_name \
    --description rul \
    --id_ $run_id \
    --dataset $data \
    --task rul \
    --gpu_dev $gpu \
    --net ocir \
    --window $W \
    --alpha 0.7 \
    --kl_annealing 0.05 \
    --time_embedding True \
    --dx 14 --dz 24 --dc 6 --d_model 168 --num_heads 4 \
    --z_projection aggregation_all \
    --D_projection aggregation \
    --c_posterior_param soft \
    --n_epochs 120 --rul_epochs 100 --batch $batch_size --lr_ 0.00015 --rul_lr 0.0002 \
    --scheduler 0 --warm_up 0.2 --final_lr 0.0001 --ref_lr 0.0002 --start_lr 0.00005 --start_wd 0.001 --final_wd 0.001
done