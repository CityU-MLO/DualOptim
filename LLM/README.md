# DualOptim for LLMs

This repository contains the **LLM-side implementation of DualOptim**, a unified optimization framework for machine unlearning in large language models (LLMs). DualOptim introduces a **dual-objective optimization mechanism** that jointly minimizes the *retention of forgotten knowledge* and maximizes *model utility preservation*, making it applicable to both **fictitious** and **real-world unlearning scenarios**.

------

## 🔧 Environment Configuration

Please follow the environment setup from the official implementation of
 **[A Closer Look at Machine Unlearning for Large Language Models (ICLR 2025)](https://github.com/sail-sg/closer-look-LLM-unlearning)**

```bash
conda create -n tofu python=3.10
conda activate tofu
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

> Our details information about package: flash-attn==2.7.4.post1, bitsandbytes==0.45.3 

------

## 🚀 Running Experiments

DualOptim follows the same experiment structure as the **TOFU Benchmark** and **Closer Look LLM Unlearning** framework.

### 1. Baseline (**Closer Look LLM Unlearning**)

#### (1) ME+GD

```bash
bash scripts/tofu/baselines.sh
```

------

### 2. DualOptim

#### (1) ME+GD

```bash
bash scripts/continual_tofu/me_gd.sh
```

#### (2) IDK+AP

```bash
bash scripts/continual_tofu/idk_ap.sh
```

### 3. Hyperparameters

| Parameter              | Description                                                  | Example                                             | Notes                                                |
| ---------------------- | ------------------------------------------------------------ | --------------------------------------------------- | ---------------------------------------------------- |
| `forget_losses`        | Specifies the unlearning objective(s) to run. Options include `ME+GD`, `IDK+GD`, `DPO+GD`, etc. | `forget_losses=(DPO+GD)`                            | Multiple objectives can be listed.                   |
| `task_list`            | Selects which forget task(s) to perform (1–10). The TOFU benchmark uses `1`. | `task_list=(1)`                                     | Each task corresponds to a different data partition. |
| `learning_rates`       | Sets the learning rate(s) for fine-tuning.                   | `learning_rates=(1e-5)`                             | You can test multiple LRs by listing several values. |
| `forget_coeff`         | Coefficient for the unlearning loss.                         | `forget_coeff=0.1`                                  | Higher values enforce stronger forgetting.           |
| `regularization_coeff` | Coefficient for the retention (regularization) loss.         | `regularization_coeff=1.0`                          | Balances utility preservation vs. forgetting.        |
| `num_epochs`           | Number of training epochs.                                   | `num_epochs=5`                                      | Increase if unlearning converges slowly.             |
| `split`                | Defines the unlearning data ratio. Options: `forget01`, `forget05`, `forget10`. | `split=forget05`                                    | Indicates 1%, 5%, or 10% forget data.                |
| `mask`                 | Whether to apply attention masking during training.          | `mask=true`                                         | Set to `false` to disable masking.                   |
| `use_LoRA`             | Whether to use parameter-efficient LoRA tuning.              | `use_LoRA=false`                                    | Set `true` to reduce GPU memory cost.                |

#### 

------

## 📘 Notes

- DualOptim modifies the optimization loop in `forget.py` and `trainer/dual_optim_trainer.py`
   to integrate **dual loss terms** for *forgetting* and *retention balancing*.
- The framework supports any **decoder-only transformer models** that are compatible with Hugging Face’s `AutoModelForCausalLM`.
- Configuration and task lists can be customized in the `configs/` and `scripts/` folders.

------

