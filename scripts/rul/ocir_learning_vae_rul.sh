
# sh ./scripts/rul/ocir_learning_vae_rul.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
runs=4
data_path_name='./datasets/cmapss_dataset'
datas=(FD001 FD002 FD003 FD004)
dcs=(1 6 1 6)
lrs=(0.0003 0.0003 0.0003 0.0003)
kls=(0.05 0.05 0.05 0.05)
random_seed=1024
run_id=2

batch_size=368
W=25

for (( i=0; i<${runs}; i++ ));
do
    data=${datas[$i]}
    dc=${dcs[$i]}
    lr=${lrs[$i]}
    kl=${kls[$i]}
    python -u main.py \
    --seed $random_seed \
    --data_path $data_path_name \
    --description rul_vae_cond_lp \
    --log_path ./Logs_VAE2/logs_ \
    --id_ $run_id \
    --dataset $data \
    --conditional 1 \
    --kl_annealing $kl\
    --net vae \
    --task rul \
    --gpu_dev 7 \
    --window $W \
    --time_embedding True \
    --dx 14 --dz 24 --dc $dc --d_model 168 --num_heads 4 \
    --z_projection aggregation_all \
    --c_posterior_param soft \
    --n_epochs 80 --rul_epochs 100  --batch $batch_size --lr_ $lr --rul_lr 0.0002 \
    --scheduler 0 --warm_up 0.2 --final_lr 0.00005 --ref_lr 0.0001 --start_lr 0.00005 --start_wd 0.001 --final_wd 0.001
done

dcs=(0 0 0 0)
for (( i=0; i<${runs}; i++ ));
do
    data=${datas[$i]}
    dc=${dcs[$i]}
    lr=${lrs[$i]}
    kl=${kls[$i]}
    python -u main.py \
    --seed $random_seed \
    --data_path $data_path_name \
    --description rul_vae_uncond_lp \
    --log_path ./Logs_VAE2/logs_ \
    --id_ $run_id \
    --dataset $data \
    --conditional 0 \
    --kl_annealing $kl\
    --net vae \
    --task rul \
    --gpu_dev 7 \
    --window $W \
    --time_embedding True \
    --dx 14 --dz 24 --dc $dc --d_model 168 --num_heads 4 \
    --z_projection aggregation_all \
    --c_posterior_param soft \
    --n_epochs 80 --rul_epochs 100  --batch $batch_size --lr_ $lr --rul_lr 0.0002 \
    --scheduler 0 --warm_up 0.2 --final_lr 0.00005 --ref_lr 0.0001 --start_lr 0.00005 --start_wd 0.001 --final_wd 0.001
done
