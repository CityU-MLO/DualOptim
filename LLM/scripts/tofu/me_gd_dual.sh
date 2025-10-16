export WANDB_MODE="offline"

MASTER_PORT=$((RANDOM % 50001 + 10000))
forget_losses=(
    ME+GD
)

split_list=(forget10)


# You can specify any forget task from 1 to 10
# the standard TOFU benchmark is task 1
task_list=(1)
#task_list=(1 2 3 4 5)
# pass to python script, for continual learning setting
export TASK_LIST=$(IFS=,; echo "${task_list[*]}")

#learning_rates=(1e-5 6e-6 8e-6 1.2e-5 1.4e-5)
learning_rates=(1e-5)

mask=true
use_LoRA=false
save_root_base=results

#forget_coeff=1.0
forget_coeff=0.1
regularization_coeff=1.0

save_checkpoint=true

num_epochs=5

### evaluate only at the last epoch
save_steps=last
eval_steps=(last)

### evaluate at each unlearning epoch
# save_steps=steps_per_epoch
# eval_steps=(50 100 150 200 250)

alternate=true
optim_cfg=dual_adam
#optim_cfg=adam
retain_freq=5


#forget_lr=(1e-5 6e-6 8e-6 1.2e-5 1.4e-5)
forget_lr=(1e-5)


save_root="${save_root_base}/optim_${optim_cfg}"

echo "Running ME+GD dual optimization with optim config $optim_cfg"
for flr in ${forget_lr[@]}; do
  for split in ${split_list[@]}; do
    for forget_loss in ${forget_losses[@]}; do
      for alpha in ${alpha_lst[@]}; do
        for lr in ${learning_rates[@]}; do
          for task_id in ${task_list[@]}; do
            export TASK_LIST=$(IFS=,; echo "${task_id}") # not continual learning setting
            COMMON="use_LoRA=$use_LoRA forget_coeff=$forget_coeff regularization_coeff=$regularization_coeff lr=$lr forget_lr=$flr split=$split forget_loss=$forget_loss num_epochs=$num_epochs \
                mask=$mask fix_ref_model=$fix_ref_model save_root=$save_root save_checkpoint=$save_checkpoint alternate=$alternate optim_cfg=$optim_cfg retain_freq=$retain_freq"
            CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$MASTER_PORT \
                    forget.py \
                    --config-name=phi1-5_tofu.yaml \
                    task_id=$task_id \
                    save_steps=$save_steps \
                    $COMMON
            for step in ${eval_steps[@]}; do
                CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
                        eval.py \
                        --config-name=phi1-5_tofu.yaml \
                        task_id=$task_id \
                        eval_unlearn_step=$step \
                        $COMMON
            done
          done
        done
      done
    done
  done
done

