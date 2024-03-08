import os
import torch
import numpy as np
import cv2
from albumentations import Compose, Resize, Normalize
import matplotlib.pyplot as plt
from src.network.transfomer_based.transformer_based_network import get_transformer_based_model
from src.network.conv_based.CMUNet import CMUNet
from src.network.conv_based.U_Net import U_Net
from src.network.conv_based.AttU_Net import AttU_Net
from src.network.conv_based.UNeXt import UNext
from src.network.conv_based.UNetplus import ResNet34UnetPlus
from src.network.conv_based.UNet3plus import UNet3plus
from src.network.conv_based.CMUNeXt import cmunext

def min_max_normalize(output):
    output_normalized = (output - output.min()) / (output.max() - output.min())
    return output_normalized

def load_model_by_type(model_path, model_type, img_size, num_classes):
    if model_type == 'TransUnet':
        model = get_transformer_based_model(
            parser=None,
            model_name='TransUnet',
            img_size=img_size,
            num_classes=num_classes,
            in_ch=3
        ).cuda()
    elif model_type == 'UNet':  # Added this section to initialize and load U_Net
        model = U_Net(output_ch=num_classes).cuda()  # Replace with your actual initialization
    elif model_type == 'CMUNet':  # Added this section to initialize and load U_Net
        model = CMUNet(output_ch=num_classes).cuda()  # Replace with your actual initialization
    elif model_type == 'AttU_Net':  # Added this section to initialize and load U_Net
        model = AttU_Net(output_ch=num_classes).cuda()  # Replace with your actual initialization
    elif model_type == 'UNext':  # Added this section to initialize and load U_Net
        model = UNext(output_ch=num_classes).cuda()  # Replace with your actual initialization
    elif model_type == 'UNetplus':  # Added this section to initialize and load U_Net
        model = ResNet34UnetPlus(num_class=num_classes).cuda()  # Replace with your actual initialization
    elif model_type == 'CMUNeXt':  # Added this section to initialize and load cmunext
        model = cmunext(num_classes=num_classes).cuda()  # Replace with your actual initialization
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess(image_path, img_size):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image at {image_path} could not be loaded.")
    
    transform = Compose([
        Resize(img_size, img_size),
        Normalize(),
    ])
    
    img = transform(image=img)['image']
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = torch.tensor(img, dtype=torch.float32).cuda()
    return img

def infer(model, img):
    with torch.no_grad():
        output = model(img)
        # Normalize output
        output = min_max_normalize(output)
        # Printing min and max values of the output for diagnosis
        print(f"Model {type(model).__name__} - Output min: {output.min().item()}, max: {output.max().item()}")
    return output

def postprocess(output, threshold=0.5):
    output = output.cpu().detach().numpy()
    output = np.squeeze(output)
    output = (output > threshold).astype(np.uint8)
    return output * 255

def iou_metric(pred, true, threshold=0.5):
    pred = (pred > threshold)
    intersection = np.logical_and(true, pred).sum()
    union = np.logical_or(true, pred).sum()
    return intersection / union

def accuracy_metric(pred, true, threshold=0.5):
    pred = (pred > threshold)
    correct = np.sum(pred == true)
    return correct / true.size

def evaluate_model_predictions(predictions, ground_truth):
    # Compute evaluation metrics for each model's prediction
    metrics = []
    for pred in predictions:
        iou_value = iou_metric(pred, ground_truth)
        acc_value = accuracy_metric(pred, ground_truth)
        metrics.append((iou_value, acc_value))
    return metrics

def main():
    image_folder = "/ibex/user/maz0a/Unetfamily/Medical-Image-Segmentation-Benchmarks/test/cropped_grayscaleimages/"
    img_size = 256
    num_classes = 1

    model_list = [
        ("/ibex/user/maz0a/Unetfamily/Medical-Image-Segmentation-Benchmarks/checkpoint/U_Net_model.pth", "UNet"),
        ("/ibex/user/maz0a/Unetfamily/Medical-Image-Segmentation-Benchmarks/checkpoint/AttU_Net_model.pth", "AttU_Net"),
        ("/ibex/user/maz0a/Unetfamily/Medical-Image-Segmentation-Benchmarks/checkpoint/TransUnet_model.pth", "TransUnet"),
        
    ]

    for filename in os.listdir(image_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            img = preprocess(image_path, img_size)

            for model_path, model_type in model_list:
                model = load_model_by_type(model_path, model_type, img_size, num_classes)
                output = infer(model, img)
                prediction = postprocess(output)

                # ?????????????????????
                result_folder = os.path.join(image_folder, model_type)
                os.makedirs(result_folder, exist_ok=True)
                
                # ???????????
                result_path = os.path.join(result_folder, f"{os.path.splitext(filename)[0]}_{model_type}_prediction.png")
                cv2.imwrite(result_path, prediction)

if __name__ == "__main__":
    main()