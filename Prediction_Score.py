
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from skimage.metrics import structural_similarity as ssim

def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))

def iou_score(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    return np.sum(intersection) / np.sum(union)

def hausdorff_distance(y_true, y_pred):
    return max(directed_hausdorff(y_true, y_pred)[0], directed_hausdorff(y_pred, y_true)[0])

def compare_masks(ground_truth, prediction):
    # Ensure masks are binary
    gt = ground_truth.astype(bool)
    pred = prediction.astype(bool)
    
    # Calculate metrics
    dice = dice_coefficient(gt, pred)
    iou = iou_score(gt, pred)
    hausdorff = hausdorff_distance(gt, pred)
    
    # Convert to float for SSIM calculation
    gt_float = gt.astype(float)
    pred_float = pred.astype(float)
    
    # Calculate SSIM with specified data_range
    ssim_score = ssim(gt_float, pred_float, data_range=1.0)
    
    return {
        "Dice Coefficient": dice,
        "IoU Score": iou,
        "Hausdorff Distance": hausdorff,
        "SSIM": ssim_score
    }

# Example usage
ground_truth = np.load('/home/data/processed_labels/0.npy')
medsam_prediction = np.load('/home/data/predicted_labels/0_pred.npy')
results = compare_masks(ground_truth, medsam_prediction)
for metric, value in results.items():
    print(f"{metric}: {value}")
