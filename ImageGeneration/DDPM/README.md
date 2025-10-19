# **S**aliency **F**orgetting in the **R**emain-preserving manifold **on**line for **DDPM**
This is the official repository for SFR-on on CIFAR-10 using DDPM. The code structure of this project is adapted from the [DDIM](https://github.com/ermongroup/ddim), [SA](https://github.com/clear-nus/selective-amnesia/tree/a7a27ab573ba3be77af9e7aae4a3095da9b136ac/ddpm), and [SalUn](https://github.com/OPTML-Group/Unlearn-Saliency/tree/master/DDPM) codebase.

# Requirements
Install the requirements using a `conda` environment:
```
conda create --name sfron-ddpm python=3.8
conda activate sfron-ddpm
pip install -r requirements.txt
```

# Preparing for Unlearning

1. First train a conditional DDPM on all 10 CIFAR-10 classes. 

   Specify GPUs using the `CUDA_VISIBLE_DEVICES` environment flag. 

   For instance, using two GPUs with IDs 0 and 1 on CIFAR10,

   ```
    CUDA_VISIBLE_DEVICES="0,1" python train.py --config cifar10_pretrain.yml --mode pretrain
   ```

   A checkpoint should be saved under `results/cifar10/pretrain/yyyy_mm_dd_hhmmss`.

   Easy to go, you can download our pre-trained model from [here](https://drive.google.com/drive/folders/1TRouxJZAWHToCj1lzMjim_UTGmlPBN1E?usp=sharing).


2. We can retrain a conditional DDPM on the other 9 CIFAR-10 classes excluding the forgetting class. 

   For instance, forgetting class 0 on CIFAR-10,

   ```
    CUDA_VISIBLE_DEVICES="0,1" python train.py --config cifar10_pretrain.yml --mode retrain --label_to_forget 0
   ```

   A checkpoint should be saved under `results/cifar10/retrain/yyyy_mm_dd_hhmmss`.

# Forgetting with SFR-on

1. First, we need to generate fisher diagonal for saliency map.

   ```
   CUDA_VISIBLE_DEVICES="0,1" python train.py --config cifar10_fisher.yml --ckpt_folder results/cifar10/pretrain/yyyy_mm_dd_hhmmss --label_to_forget 0 --mode generate_fisher
   ```

   The fisher diagonal for remaining and forgetting will save in `results/cifar10/pretrain/yyyy_mm_dd_hhmmss/mask_0`.

2. Next, we need to generate saliency map for unlearning.

   ```
   CUDA_VISIBLE_DEVICES="0,1" python generate_fisher_mask.py --ckpt_folder results/cifar10/pretrain/yyyy_mm_dd_hhmmss/mask_0 --threshold 1.0
   ```

   This will save saliency map in `results/cifar10/pretrain/yyyy_mm_dd_hhmmss/mask_0`.

3. Forgetting training with SFR-on

   ```
   CUDA_VISIBLE_DEVICES="0,1" python train.py --config cifar10_sfron.yml \
      --ckpt_folder ${ckpt_path} --label_to_forget ${cls} --mode sfron \
      --forget_alpha 10.0 --decay_forget_alpha --remain_alpha 1.0 \
      --method ron --unlearn_loss adaga --mask_path ${mask_path}   
   ```

   You can experiment with forgetting different class labels using the `--label_to_forget` flag.

4. Forgetting training with SFR-on + **DualOptim**

   ```
   CUDA_VISIBLE_DEVICES="0,1" python train.py --config cifar10_sfron.yml \
      --ckpt_folder ${ckpt_path} --label_to_forget ${cls} --mode sfron \
      --forget_alpha 1.0 --decay_forget_alpha --remain_alpha 1.0 \
      --method ron --unlearn_loss adaga --mask_path ${mask_path} --dual --forget_lr 1e-4
   ```

# Other Baselines

## Forgetting with SA 

1. First, generate class samples for calculating the FIM, and to be used sa the GR samples later.

    ```
    CUDA_VISIBLE_DEVICES="0,1" python sample.py --config cifar10_sample.yml --ckpt_folder ${ckpt_path} --mode sample_classes --n_samples_per_class 500
    ```
    This will save them in `${ckpt_path}/class_samples`, where each folder represents a class.

2. Calculate the FIM. Depending on the value `n_samples_per_class` in step 2 (500 is what is used in the paper), this step could take a while
as the ELBO of diffusion models requires a sum over 1000 timesteps PER sample.

    ```
    CUDA_VISIBLE_DEVICES="0,1" python fim.py --config cifar10_fim.yml --ckpt_folder ${ckpt_path} --n_chunks 20
    ```
    The FIM should be saved as `fisher_dict.pkl` in the same folder. If you find that you are running out of GPU memory, increase `n_chunks` parameter. This parameter chunks the 1000 timesteps in the ELBO into `n_chunks` and computes each chunk in parallel. Increasing `n_chunks` leads to each chunk being smaller, hence using less memory. However, it means calculation is slower.

3. Forgetting with SA

    You can vary the $\lambda$ weight for the FIM in `configs/cifar10_sa.yml`.
    ```
    CUDA_VISIBLE_DEVICES="0,1" python train.py --config cifar10_sa.yml --ckpt_folder ${ckpt_path} --label_to_forget 0 --mode sa
    ```
    This should create another folder in `results/cifar10/forget_0/sa`. You can experiment with forgetting different class labels using the `--label_to_forget` flag, but we will consider forgetting the 0 (airplane) class here.

## Forgetting with SalUn

1. First, we need to generate saliency map samples for unlearning.

   ```
   CUDA_VISIBLE_DEVICES="0,1" python train.py --config cifar10_saliency_unlearn.yml --ckpt_folder ${ckpt_path} --label_to_forget 0 --mode generate_mask
   ```

   This will save saliency map in `results/cifar10/mask`.

2. Forgetting with Saliency-Unlearning

   ```
   CUDA_VISIBLE_DEVICES="0,1" python train.py --config cifar10_saliency_unlearn.yml --ckpt_folder ${ckpt_path} --label_to_forget 0 --mode salun --mask_path results/cifar10/mask/{mask_name} --remain_alpha 1e-3 --unlearn_loss rl
   ```

   This should create another folder in `results/cifar10/forget_0/salun`. 

   You can experiment with forgetting different class labels using the `--label_to_forget` flag, but we will consider forgetting the 0 (airplane) class here.

   You can experiment with forgetting different unlearn loss using the `--unlearn_loss` flag, but we will consider forgetting with random label(rl) here.


# Evaluation
1. Image Metrics Evaluation on Classes to Remember

    First generate the sample images on the model trained in step 3.
    ```
    CUDA_VISIBLE_DEVICES="0,1" python sample.py --config cifar10_sample.yml --ckpt_folder ${ckpt_path} --mode sample_fid --n_samples_per_class 5000 --classes_to_generate 'x0'
    ```
    Samples will be saved in `${ckpt_path}/fid_samples_without_label_0_guidance_2.0`. We can either use `--classes_to_generate '1,2,3,4,5,6,7,8,9'` or `--classes_to_generate 'x01'` to specify that we want to generate all classes but the 0 class (as we have forgotten it).

    Next, we need samples from the reference dataset, but without the 0 class.
    ```
    python save_base_dataset.py --dataset cifar10 --label_to_forget 0
    ```
    The images should be saved in folder `./cifar10_without_label_0`.

    Now we can evaluate the image metrics
    ```
    CUDA_VISIBLE_DEVICES="0,1" python evaluator.py ${ckpt_path}/fid_samples_without_label_0_guidance_2.0 cifar10_without_label_0
    ```
    The metrics will be printed to the screen like such
    ```
    Inception Score: 8.198589324951172
    FID: 9.670457625511688
    sFID: 7.438950112110206
    Precision: 0.3907777777777778
    Recall: 0.7879333333333334
    ```

2. Classifier Evaluation

    First fine-tune a pretrained ResNet34 classifier for CIFAR10
    ```
    CUDA_VISIBLE_DEVICES="0" python train_classifier.py --dataset cifar10 
    ```
    The classifier checkpoint will be saved as `cifar10_resnet34.pth`.

    Generate samples of just the 0th class (500 is used for classifier evaluation in the paper)
    ```
    CUDA_VISIBLE_DEVICES="0,1" python sample.py --config cifar10_sample.yml --ckpt_folder ${ckpt_path} --mode sample_classes --classes_to_generate "0" --n_samples_per_class 500
    ```
    The samples are saved in the folder `${ckpt_path}/class_samples/0`.

    Finally evaluate with the trained classifier
    ```
    CUDA_VISIBLE_DEVICES="0" python classifier_evaluation.py --sample_path ${ckpt_path}/class_samples/0 --dataset cifar10 --label_of_forgotten_class 0
    ```
    The results will be printed to screen like such
    ```
    Classifier evaluation:
    Average entropy: 1.4654556959867477
    Average prob of forgotten class: 0.15628313273191452