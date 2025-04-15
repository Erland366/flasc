#!/bin/bash
# The current configuration is for the experiment 1 in the new todo list
config="basic_config"
server_opts=("yogi")  # sgd adam yogi adagrad sgdm
server_schedules=("constant")  # constant cosine
client_opts=("sgd")  # sgd adam yogi adagrad
client_epochs=(1)  # 1 2 4 6
early_stopping="0.95"  # any value between 0 and 1; default can be 0.95
merging_strategies=(average fisher_merging regmean_merging ties_merging task_arithmetic)  # average fisher_merging regmean_merging ties_merging task_arithmetic
seeds=(0 1 2)
gpu=0  # GPU ID

for client_opt in "${client_opts[@]}"; do
    for client_epoch in "${client_epochs[@]}"; do
        for server_schedule in "${server_schedules[@]}"; do
            for server_opt in "${server_opts[@]}"; do
                for merging_strategy in "${merging_strategies[@]}"; do
                    for seed in "${seeds[@]}"; do
                        # Set the server learning rate
                        server_lr=1.0e-3
                        if [[ "$server_opt" == "sgd" || "$server_opt" == "sgdm" ]]; then
                            server_lr=1.0
                        fi
                        
                        echo "Running: python train_lora.py --config ${config} \
                        --client-opt ${client_opt} --early-stopping ${early_stopping} \
                        --client-epochs ${client_epoch} --server-schedule ${server_schedule} \
                        --server-opt ${server_opt} --merging-strategy ${merging_strategy} \
                        --seed ${seed}"
                        
                        python train_lora.py \
                            --config "${config}" \
                            --client-opt "${client_opt}" \
                            --early-stopping "${early_stopping}" \
                            --client-epochs "${client_epoch}" \
                            --server-lr "${server_lr}" \
                            --server-schedule "${server_schedule}" \
                            --server-opt "${server_opt}" \
                            --merging-strategy "${merging_strategy}" \
                            --seed "${seed}" \
                            --gpu ${gpu}
                    done
                done
            done
        done
    done
done
