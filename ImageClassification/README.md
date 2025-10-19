## DualOptim for Image Classification

This repository provides the implementation of **DualOptim** applied to **image classification unlearning tasks**, integrating with existing unlearning frameworks such as **SalUn**, **SFRon**, and **SCRUB**. DualOptim introduces a *dual-objective optimization* mechanism to balance **forgetting** and **retention** objectives effectively.



## 1. Get the Origin Model

Train the original model before performing any unlearning:

```
python main_train.py \
  --arch {model name} \
  --dataset {dataset name} \
  --epochs {epochs for training} \
  --lr {learning rate for training} \
  --save_dir {path to save the origin model}
```

**Example:**
 ResNet-18 on CIFAR-10

```
python main_train.py --arch resnet18 --dataset cifar10 --lr 0.1 --epochs 182 --save_dir resnet18_cifar10
```



## 2. Generate Saliency Map (for SalUn only)

Before running **SalUn**, generate a saliency mask to indicate forget regions:

```
python generate_mask.py \
  --save_dir ${saliency_map_path} \
  --model_path ${origin_model_path} \
  --num_indexes_to_replace ${forgetting data amount} \
  --unlearn_epochs 1
```



## 3. Unlearning with DualOptim

DualOptim can be integrated into multiple existing unlearning frameworks:

### **SalUn**

```
python main_random.py \
  --unlearn RL \
  --unlearn_epochs ${epochs for unlearning} \
  --unlearn_lr ${learning rate for unlearning} \
  --retain_lr ${learning rate for retaining} \
  --num_indexes_to_replace ${forgetting data amount} \
  --model_path ${origin_model_path} \
  --save_dir ${save_dir} \
  --mask_path ${saliency_map_path} \
  --optim ${sgd, adam, dual...}
```

**Example:**
 ResNet-18 on CIFAR-10 to unlearn 10% of data

```
python main_random.py \
  --unlearn RL \
  --unlearn_epochs 10 \
  --unlearn_lr 0.018 \
  --retain_lr 0.01 \
  --num_indexes_to_replace 4500 \
  --model_path ${origin_model_path} \
  --save_dir ${save_dir} \
  --mask_path mask/with_0.5.pt
```



### **SFRon**

```
python -u main_random_sfron.py \
  --save_dir ${save_dir} \
  --model_path ${origin_model_path} \
  --unlearn SFRon \
  --num_indexes_to_replace ${forgetting data amount} \
  --sfron_alpha ${alpha} \
  --sfron_iters ${epochs for unlearning} \
  --unlearn_lr ${learning rate for forgetting} \
  --retain_lr ${learning rate for retaining} \
  --optim ${sgd, adam, dual...}
```



###  **SCRUB**

```
python -u main_forget.py \
  --save_dir ${save_dir} \
  --model_path ${origin_model_path} \
  --unlearn SCRUB \
  --num_indexes_to_replace ${forgetting data amount} \
  --unlearn_epochs 10 \
  --unlearn_lr ${learning rate for forgetting} \
  --retain_lr ${learning rate for retaining} \
  --optim ${sgd, adam, dual...}
```



## Baseline Methods

Below are the baseline methods that can be reproduced with the **SFRon-version MIA** (different from the SalUn setup).

### Retrain (Full Retraining Baseline)

```
python main_forget.py \
  --save_dir ${save_dir} \
  --model_path ${origin_model_path} \
  --unlearn retrain \
  --num_indexes_to_replace ${forgetting data amount} \
  --unlearn_epochs ${epochs for unlearning} \
  --unlearn_lr ${learning rate for unlearning}
```



### Fine-tuning (FT)

```
python main_forget.py \
  --save_dir ${save_dir} \
  --model_path ${origin_model_path} \
  --unlearn FT \
  --num_indexes_to_replace ${forgetting data amount} \
  --unlearn_epochs ${epochs for unlearning} \
  --unlearn_lr ${learning rate for unlearning}
```



### Random Label (RL)

*(Only difference from SalUn is that no mask is used.)*

```
python main_random.py \
  --unlearn RL \
  --unlearn_epochs ${epochs for unlearning} \
  --unlearn_lr ${learning rate for unlearning} \
  --num_indexes_to_replace ${forgetting data amount} \
  --model_path ${origin_model_path} \
  --save_dir ${save_dir}
```



### Gradient Ascent (GA)

```
python main_forget.py \
  --save_dir ${save_dir} \
  --model_path ${origin_model_path} \
  --unlearn GA \
  --num_indexes_to_replace ${forgetting data amount} \
  --unlearn_epochs ${epochs for unlearning} \
  --unlearn_lr ${learning rate for unlearning}
```

> 💡 *Hyperparameter settings for each baseline can be found in the corresponding SalUn or SFRon papers.*



## Evaluation

1. Use the provided scripts in the `scripts/` folder to evaluate results.
    Each unlearning method is run with **5 random seeds** to ensure statistical consistency.

2. We fix:

   - `retain_lr = 0.01` for **SGD**, or
   - `retain_lr = 1e-4` for **Adam**
      Then search for the optimal `unlearn_lr` achieving the **lowest average gap**.

3. Compute the **average gap** and **standard deviation** using:

   ```
   python parse_result.py --log_paths ${path_to_log_files}
   ```