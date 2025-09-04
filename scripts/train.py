
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import random
import gc

# Project-specific imports
from models.densenet_baseline import DenseNet121_CIFAR10
from models.topk_moe import TopKMoEModel
from utils.data_loader import get_cifar10_loaders
from utils.augmentations import cutmix_data, mixup_criterion

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_loop(model, train_loader, optimizer, scheduler, criterion, num_epochs, loop_name, model_type, cutmix_prob, load_balance_weight=0.01):
    """
    Generic training loop for a single phase of training.
    """
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"({loop_name}) Epoch {epoch+1}/{num_epochs}")
        
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            # Apply CutMix augmentation
            use_cutmix = random.random() < cutmix_prob
            if use_cutmix:
                inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets)

            # Forward pass
            if model_type == 'topk_moe':
                outputs, _, _, load_balance_loss = model(inputs)
            else: # densenet
                outputs = model(inputs)

            # Calculate main loss
            if use_cutmix:
                main_loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                main_loss = criterion(outputs, targets)

            # Add load balancing loss for MoE models
            total_loss = main_loss
            if model_type == 'topk_moe':
                total_loss += load_balance_weight * load_balance_loss

            # Backward pass and optimization
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Logging
            _, preds = torch.max(outputs, 1)
            correct = (preds == targets).sum().item() if not use_cutmix else 0
            accuracy = (correct / targets.size(0)) * 100.0
            
            log_dict = {'Loss': f'{total_loss.item():.4f}', 'Acc': f'{accuracy:.2f}%'}
            if model_type == 'topk_moe':
                log_dict['LB Loss'] = f'{load_balance_loss.item():.4f}'
            progress_bar.set_postfix(log_dict)
            
        scheduler.step()
        torch.cuda.empty_cache()
        gc.collect()

def main(args):
    """
    Main function to orchestrate the model training process.
    """
    print(f"Using device: {device}")
    print(f"Training model: {args.model}")

    # Load data
    train_loader, _ = get_cifar10_loaders(batch_size=args.batch_size, num_workers=args.num_workers)
    
    # Load model
    if args.model == 'densenet':
        model = DenseNet121_CIFAR10(num_classes=10)
        backbone = model.densenet.features
        head_params = model.densenet.classifier.parameters()
    elif args.model == 'topk_moe':
        model = TopKMoEModel(
            feature_dim=args.feature_dim,
            num_classes=10,
            num_experts=args.num_experts,
            k=args.top_k
        )
        backbone = model.backbone.parameters()
        head_params = list(model.router.parameters()) + list(model.experts.parameters())
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    criterion = nn.CrossEntropyLoss()

    # --- Phase 1: Head Training (Backbone Frozen) ---
    print("\n--- PHASE 1: Training head with frozen backbone ---")
    for param in backbone:
        param.requires_grad = False

    head_optimizer = optim.AdamW(head_params, lr=args.lr_head)
    head_scheduler = optim.lr_scheduler.CosineAnnealingLR(head_optimizer, T_max=args.epochs_head)
    
    train_loop(model, train_loader, head_optimizer, head_scheduler, criterion, args.epochs_head, "Head Training", args.model, args.cutmix_prob, args.load_balance_weight)

    # --- Phase 2: Full Fine-tuning ---
    print("\n--- PHASE 2: Fine-tuning all layers ---")
    for param in model.parameters():
        param.requires_grad = True

    full_optimizer = optim.AdamW(model.parameters(), lr=args.lr_full)
    full_scheduler = optim.lr_scheduler.CosineAnnealingLR(full_optimizer, T_max=args.epochs_full)

    train_loop(model, train_loader, full_optimizer, full_scheduler, criterion, args.epochs_full, "Full Fine-Tune", args.model, args.cutmix_prob, args.load_balance_weight)

    # Save the final model
    save_path = f"{args.model}_cifar10.pth" if args.model == 'densenet' else f"{args.model}_baseline.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n Training complete. Model saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CIFAR-10 Model Training Script")
    
    # Model selection
    parser.add_argument('--model', type=str, required=True, choices=['densenet', 'topk_moe'], help='Model to train.')
    
    # General training parameters
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--epochs_head', type=int, default=20, help='Number of epochs for head-only training.')
    parser.add_argument('--epochs_full', type=int, default=80, help='Number of epochs for full fine-tuning.')
    parser.add_argument('--lr_head', type=float, default=1e-3, help='Learning rate for head training.')
    parser.add_argument('--lr_full', type=float, default=5e-5, help='Learning rate for full fine-tuning.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader.')
    
    # Augmentation parameters
    parser.add_argument('--cutmix_prob', type=float, default=0.5, help='Probability of applying CutMix.')

    # MoE-specific parameters
    parser.add_argument('--num_experts', type=int, default=10, help='Number of experts for MoE models.')
    parser.add_argument('--top_k', type=int, default=2, help='Number of experts to activate for Top-K MoE.')
    parser.add_argument('--feature_dim', type=int, default=256, help='Feature dimension from backbone to MoE layer.')
    parser.add_argument('--load_balance_weight', type=float, default=0.01, help='Weight for the load balancing loss.')
    
    args = parser.parse_args()
    main(args)