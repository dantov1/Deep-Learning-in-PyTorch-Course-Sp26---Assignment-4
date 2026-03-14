"""
Module: test.py

This module provides functions for evaluating a video classification model on test data.
It includes functions to compute predictions and accuracy, generate a detailed classification
report, and compute a multilabel confusion matrix for all classes.

Functions:
    - test: Evaluates the model on a test DataLoader and returns the ground truth labels,
      predicted labels, softmax probabilities, and overall accuracy.
    - get_test_report: Generates a classification report using scikit-learn's classification_report.
    - get_confusion_matrix: Computes a multilabel confusion matrix for each class.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report, multilabel_confusion_matrix

def test(model, dataloader, device):
    """
    Evaluate the model on the test dataset and compute overall accuracy.
    
    Args:
        model (torch.nn.Module): The trained video classification model.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the test dataset.
        device (torch.device): The device (CPU or GPU) on which to perform evaluation.

    Returns:
        tuple: (targets, outputs, all_probs, accuracy)
            - targets (list): Ground truth labels for all samples.
            - outputs (list): Predicted labels for all samples.
            - all_probs (list of list): Softmax probabilities for each sample (for AUC/top-k).
            - accuracy (float): Overall accuracy.
    """
    model.eval()
    with torch.no_grad():
        total_correct_preds = 0.0
        len_dataset = len(dataloader.dataset)
        targets, outputs, all_probs = [], [], []
        for x_batch, y_batch in tqdm(dataloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)

            # Softmax probabilities for AUC-ROC, top-k, log-loss
            probs = F.softmax(output, dim=1)
            all_probs.extend(probs.detach().cpu().numpy().tolist())

            pred = output.argmax(dim=1, keepdim=True)
            correct_preds = pred.eq(y_batch.view_as(pred)).sum().item()
            total_correct_preds += correct_preds
            outputs.extend(pred.view(-1).detach().cpu().numpy().tolist())
            targets.extend(y_batch.detach().cpu().numpy().tolist())
        
    accuracy = total_correct_preds / float(len_dataset)

    return targets, outputs, all_probs, accuracy

def get_test_report(target, output, target_names):
    """
    Generate a detailed classification report based on test results.
    
    Args:
        target (list): Ground truth labels.
        output (list): Predicted labels.
        target_names (list): List of class names corresponding to the labels.

    Returns:
        dict: A classification report as a dictionary.
    """
    return classification_report(target, output, output_dict=True, target_names=target_names)

def get_confusion_matrix(targets, outputs, labels_dict, all_cats):
    """
    Compute a multilabel confusion matrix for each class.
    
    Args:
        targets (list): Ground truth numeric labels.
        outputs (list): Predicted numeric labels.
        labels_dict (dict): Dictionary mapping class names to numeric labels.
        all_cats (list): List of all class names.

    Returns:
        dict: A dictionary where keys are class names and values are the corresponding confusion matrices.
    """
    inv_labels_dict = {label: cat for cat, label in labels_dict.items()}
    target_cats = [inv_labels_dict[target] for target in targets]
    output_cats = [inv_labels_dict[output] for output in outputs]
    confusion_mat = multilabel_confusion_matrix(target_cats, output_cats, labels=all_cats)
    return {label: mat for label, mat in zip(all_cats, confusion_mat)}