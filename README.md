# Comparative Analysis of Mixture-of-Experts (MoE) Routing Strategies for CIFAR-10

This repository contains the code and analysis for a comparative study of different model architectures and Mixture-of-Experts (MoE) routing strategies for image classification on the CIFAR-10 dataset.

## Project Overview

The goal of this project is to explore the trade-offs between standard dense convolutional networks and sparser, more efficient Mixture-of-Experts models. We establish performance baselines and then dive deep into the internal mechanics of MoE routing.

The project follows three main experiments:
1.  **DenseNet121 Baseline**: A powerful, dense convolutional neural network fine-tuned for CIFAR-10 to set a high-performance benchmark.
2.  **Top-K MoE Baseline**: A standard MoE architecture using a Top-K gating mechanism, which routes each input to a fixed number of "expert" sub-networks.
3.  **[Your Novel MoE Name, e.g., SP-MoE]**: *(This is where you'd introduce your own model)* An implementation of a novel MoE routing strategy designed to improve upon the limitations of the standard Top-K approach, such as load balancing and expert specialization.

## Key Findings & Results Summary

A comprehensive evaluation was performed, measuring accuracy, computational cost (GFLOPs), inference speed, and parameter efficiency.

| Model              | Accuracy (%) | Precision (%) | F1-Score (%) | Total Params (M) | GFLOPs | Avg Inference (ms) |
| ------------------ | ------------ | ------------- | ------------ | ---------------- | ------ | ------------------ |
| DenseNet121        | 93.12%       | 93.11%        | 93.10%       | 7.00 M           | 0.29   | 39.15 ms           |
| Top-K MoE          | 92.55%       | 92.54%        | 92.53%       | 9.88 M           | 0.23   | 43.47 ms           |

### Analysis of Top-K MoE Expert Usage

A key finding from the baseline MoE model was its **poor load balancing**. The Top-K router struggled to distribute inputs evenly across the experts, leading to significant under-utilization of the model's capacity.

*Expert 0 was utilized for over 50% of the test samples, while several other experts were used less than 5% of the time, with Expert 1 not being used at all.*

This analysis highlights a critical challenge in MoE design and serves as the primary motivation for developing a more dynamic routing strategy.

## Repository Structure

```
├── models/
│   ├── densenet_baseline.py
│   └── topk_moe.py
├── utils/
│   ├── data_loader.py
│   └── augmentations.py
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── results/
│   └── (Generated plots and summaries will be saved here)
├── requirements.txt
└── README.md
```
## Setup and Usage

**1. Clone the repository:**
```bash
git clone [https://github.com/Jade2451/MoE_CIFAR10]
cd [MoE_CIFAR10]
```

**2. Create a virtual environment and install dependencies:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

**3. Train the models:**

You can train any of the models using `scripts/train.py`.
```bash
# Train the DenseNet121 baseline
python scripts/train.py --model densenet

# Train the Top-K MoE baseline
python scripts/train.py --model topk_moe
```

**4. Evaluation phase: **

After training, run the evaluation script to generate performance metrics and visualizations.
```bash
# Evaluate the DenseNet121 model
python scripts/evaluate.py --model densenet --checkpoint_path densenet121_cifar10.pth

# Evaluate and analyze the Top-K MoE model
python scripts/evaluate.py --model topk_moe --checkpoint_path topk_moe_baseline.pth
```

## Future Work
This project establishes a strong foundation for further research into MoE models. Potential next steps include:

1. I want to target improving the expert specialisation metric in particular. 

2. Incorporate more sophisticated load balancing losses, such as entropy-based losses or those that account for expert capacity.

3. Swap the DenseNet121 backbone with other architectures like a Transformer to see how it interacts with the MoE layer.

4. Scaling Up: Experiment with a larger number of experts and different datasets to analyze scalability.(That is the main advantage of MoE after all).
