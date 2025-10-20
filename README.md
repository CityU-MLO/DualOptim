# DualOptim: Dual-Objective Optimization for Machine Unlearning

This repository provides the **official implementation of DualOptim**, a unified optimization framework for **machine unlearning** across multiple modalities, including **Large Language Models (LLMs)**, **Image Classification**, and **Image Generation**.

## Overview

Recent advancements in **machine unlearning (MU)** have made it a vital technique for ensuring data privacy and trustworthiness in modern AI systems. However, existing approximate MU methods — which aim to forget specific data samples without full retraining — suffer from **high sensitivity to hyperparameters** and **unstable performance** across different datasets and forgetting scenarios. These issues limit their scalability and practical deployment.

**DualOptim** addresses these challenges by introducing a **dual-optimizer framework** that integrates:

* An **adaptive learning rate optimizer** (e.g., Adam) for the *forgetting objective*, and
* A **separate optimizer** (e.g., SGD) for the *retaining objective*.

By **decoupling the momentum terms** of these two objectives, DualOptim stabilizes gradient updates, reduces parameter variance, and improves both **unlearning efficacy** and **model utility preservation**.

Extensive theoretical and empirical analyses demonstrate that DualOptim:

* Enhances the stability of existing MU algorithms by reducing gradient variance;
* Boosts performance across diverse tasks, including **image classification**, **image generation**, and **large language models (LLMs)**;
* Serves as a **plug-and-play** component that can be easily integrated into existing MU frameworks such as SalUn, SFRon, and SCRUB.

DualOptim thus represents a **generic and scalable solution** to achieve **stable, effective, and efficient machine unlearning** across modalities and architectures.



## Tasks

We evaluate the effect of DualOptim in the following MU tasks:
* **[DualOptim for LLMs](./LLM/)**
  Implements DualOptim on large language models following the TOFU benchmark and unlearning baselines.

* **[DualOptim for Image Classification](./ImageClassification/)**
  Integrates DualOptim with existing unlearning frameworks such as SalUn, SFR-on, and SCRUB.

* **[DualOptim for Image Generation](./ImageGeneration/)**
  Extends DualOptim to diffusion-based models (DDPM, DiT) for generative unlearning.



## Citation

If you use this work, please cite:

```bibtex
@article{zhong2025dualoptim,
  title={DualOptim: Enhancing Efficacy and Stability in Machine Unlearning with Dual Optimizers},
  author={Xuyang Zhong and Haochen Luo and Chen Liu},
  journal={Advances in Neural Information Processing Systems},
  year={2025}
}
```



## Acknowledgments

This project builds upon and extends several existing frameworks, including
TOFU Benchmark, Closer Look at LLM Unlearning, SalUn, SFR-on, SCRUB, and open diffusion model implementations.

