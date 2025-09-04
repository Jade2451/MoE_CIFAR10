import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import time
from thop import profile
from sklearn.metrics import precision_recall_fscore_support, classification_report

# Project-specific imports
from models.densenet_baseline import DenseNet121_CIFAR10
from models.topk_moe import TopKMoEModel
from utils.data_loader import get_cifar10_loaders

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def comprehensive_evaluate(model, test_loader, model_name="Model"):
    """Runs a comprehensive evaluation of the model."""
    model.eval()
    all_predictions, all_labels, inference_times = [], [], []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            images, labels = images.to(device), labels.to(device)
            
            start_time = time.time()
            outputs = model(images)
            inference_times.append(time.time() - start_time)
            
            # Handle tuple output from MoE models
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * (np.array(all_predictions) == np.array(all_labels)).sum() / len(all_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1_score': f1 * 100,
        'avg_inference_time_ms': np.mean(inference_times[1:]) * 1000, # Skip first batch for warmup
        'predictions': all_predictions,
        'labels': all_labels,
    }

def analyze_expert_usage(model, test_loader, num_experts):
    """Analyzes expert usage for MoE models."""
    model.eval()
    expert_usage_counts = torch.zeros(num_experts, device=device)
    total_tokens = 0
    
    with torch.no_grad():
        for inputs, _ in tqdm(test_loader, desc="Analyzing Expert Usage"):
            inputs = inputs.to(device)
            _, routing_weights, _, _ = model(inputs)
            expert_usage_counts += routing_weights.sum(dim=0)
            total_tokens += routing_weights.sum()

    expert_usage_percentages = (expert_usage_counts / total_tokens * 100).cpu().numpy()
    
    # --- Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    target_usage = 100.0 / num_experts
    
    # Plot 1: Expert usage distribution
    axes[0].bar(range(num_experts), expert_usage_percentages, color='lightblue', edgecolor='navy')
    axes[0].axhline(y=target_usage, color='red', linestyle='--', label=f'Uniform Target ({target_usage:.1f}%)')
    axes[0].set_title('Expert Usage Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Expert ID')
    axes[0].set_ylabel('Usage Percentage (%)')
    axes[0].set_xticks(range(num_experts))
    axes[0].legend()

    # Plot 2: Load balancing deviation
    usage_deviation = np.abs(expert_usage_percentages - target_usage)
    axes[1].bar(range(num_experts), usage_deviation, color='lightcoral', edgecolor='darkred')
    axes[1].set_title('Deviation from Uniform Usage', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Expert ID')
    axes[1].set_ylabel('Absolute Deviation (%)')
    axes[1].set_xticks(range(num_experts))
    
    plt.tight_layout()
    plt.savefig(f"results/{model.router.__class__.__name__}_expert_usage.png")
    plt.show()

    print("\n--- Expert Usage Analysis Summary ---")
    print(f"Usage standard deviation: {np.std(expert_usage_percentages):.2f}%")
    print(f"Average deviation from uniform: {np.mean(usage_deviation):.2f}%")

def main(args):
    """Main function to orchestrate model evaluation."""
    print(f"Evaluating model: {args.model} from checkpoint: {args.checkpoint_path}")
    
    # Load data
    _, test_loader = get_cifar10_loaders(batch_size=args.batch_size, num_workers=args.num_workers)
    class_names = test_loader.dataset.classes

    # Load model architecture
    if args.model == 'densenet':
        model = DenseNet121_CIFAR10(num_classes=10)
    elif args.model == 'topk_moe':
        model = TopKMoEModel(
            feature_dim=args.feature_dim,
            num_classes=10,
            num_experts=args.num_experts,
            k=args.top_k
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    # Load trained weights
    try:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        model.to(device)
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at '{args.checkpoint_path}'")
        return
    except Exception as e:
        print(f"Error loading model state: {e}")
        return

    # --- Run Evaluations ---
    results = comprehensive_evaluate(model, test_loader, model_name=args.model.upper())
    
    total_params = sum(p.numel() for p in model.parameters())
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    gflops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    
    # --- Print Summaries ---
    summary_data = {
        'Model': [args.model.upper()],
        'Accuracy (%)': [f"{results['accuracy']:.2f}"],
        'Precision (%)': [f"{results['precision']:.2f}"],
        'F1-Score (%)': [f"{results['f1_score']:.2f}"],
        'Total Params (M)': [f"{total_params/1e6:.2f}"],
        'GFLOPs': [f"{gflops/1e9:.3f}"],
        'Avg Inference (ms)': [f"{results['avg_inference_time_ms']:.2f}"]
    }
    
    summary_df = pd.DataFrame(summary_data)
    print("\n--- Performance Summary ---")
    print(summary_df.to_string(index=False))

    print("\n--- Detailed Classification Report ---")
    print(classification_report(results['labels'], results['predictions'], target_names=class_names, digits=3))

    # --- MoE-Specific Analysis ---
    if args.model == 'topk_moe':
        analyze_expert_usage(model, test_loader, args.num_experts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CIFAR-10 Model Evaluation Script")
    
    parser.add_argument('--model', type=str, required=True, choices=['densenet', 'topk_moe'], help='Model architecture to evaluate.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the trained model .pth file.')
    
    # General parameters
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader.')
    
    # MoE-specific parameters (only needed if model is MoE)
    parser.add_argument('--num_experts', type=int, default=10, help='Number of experts for MoE models.')
    parser.add_argument('--top_k', type=int, default=2, help='K value for Top-K MoE.')
    parser.add_argument('--feature_dim', type=int, default=256, help='Feature dimension for MoE layer.')

    args = parser.parse_args()
    main(args)