import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
import cv2
import os
import torch.nn as nn
import math

# CA BLOCK
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        identity = x

        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

# ECA Block
class ECABlock(torch.nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECABlock, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv = torch.nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

# Load models
segformer_model_path = 'final_segformer_model.pth'
efficientnet_model_path = 'final_ef_eca_ca_model.pth'

segformer_model = torch.load(segformer_model_path, map_location=torch.device('cpu'), weights_only=False)
efficientnet_model = torch.load(efficientnet_model_path, map_location=torch.device('cpu'), weights_only=False)

segformer_model.eval()
efficientnet_model.eval()

# Image transformations
segformer_transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

efficientnet_transform = T.Compose([
    T.Resize((300, 300)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# CLAHE processing
def apply_clahe_to_rgb(image_np):
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab_clahe = cv2.merge((cl, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

# Streamlit app layout
st.markdown("<h1 style='text-align: center;'>Fish Eye Freshness Classification</h1>", unsafe_allow_html=True)

# Option to upload or capture a photo
image_source = st.radio("Select Image Source", ["Upload Image", "Capture Image"])

# Define class names
class_names = ['Fresh', 'Not Fresh', 'Spoiled', 'Very Fresh']

# Image input based on user choice
if image_source == "Upload Image":
    uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

elif image_source == "Capture Image":
    captured_image = st.camera_input("Take a photo")
    if captured_image is not None:
        image = Image.open(captured_image).convert("RGB")

# If an image has been uploaded or captured, proceed
if 'image' in locals():
    st.image(image, caption="Input Image", use_column_width=True)

    # Display spinner while processing
    with st.spinner("Processing..."):
        # Progress bar initialization
        progress = st.progress(0)

        # Convert to numpy array and process
        image_np = np.array(image)
        input_image = segformer_transform(image).unsqueeze(0)
        progress.progress(10)  # 10% - image preprocessing done

        # Segmentation (SegFormer)
        with torch.no_grad():
            outputs = segformer_model(pixel_values=input_image)
            preds = outputs.logits
            preds = torch.argmax(F.interpolate(preds, size=(image.size[1], image.size[0]), mode='bilinear', align_corners=False), dim=1)
            segmentation = preds.squeeze().numpy()
        progress.progress(40)  # 40% - segmentation done

        fish_eye_mask = (segmentation == 2).astype(np.uint8)
        masked_image = image_np.copy()
        masked_image[fish_eye_mask == 0] = [0, 0, 0]

        # Extract the largest connected component (fish eye)
        num_labels, labels_im = cv2.connectedComponents(fish_eye_mask)
        if num_labels > 1:
            max_area = 0
            max_label = 1
            for i in range(1, num_labels):
                area = np.sum(labels_im == i)
                if area > max_area:
                    max_area = area
                    max_label = i

            fish_eye_cropped = masked_image[np.min(np.argwhere(labels_im == max_label), axis=0)[0]:
                                            np.max(np.argwhere(labels_im == max_label), axis=0)[0],
                                            np.min(np.argwhere(labels_im == max_label), axis=0)[1]:
                                            np.max(np.argwhere(labels_im == max_label), axis=0)[1]]

            # Apply CLAHE
            fish_eye_clahe = apply_clahe_to_rgb(fish_eye_cropped)
            st.image(fish_eye_clahe, caption="CLAHE Enhanced Fish Eye", use_column_width=True)
            progress.progress(70)  # 70% - CLAHE enhancement done

            # Classification (EfficientNetV2)
            fish_eye_clahe_pil = Image.fromarray(fish_eye_clahe)
            fish_eye_resized = fish_eye_clahe_pil.resize((300, 300))
            input_image_classification = efficientnet_transform(fish_eye_resized).unsqueeze(0)

            with torch.no_grad():
                outputs_classification = efficientnet_model(input_image_classification)
                preds_classification = torch.argmax(outputs_classification, dim=1).item()

            predicted_label = class_names[preds_classification]

            st.markdown(f"""
                <div style="text-align: center; font-size: 30px; font-weight: bold;">
                    Predicted Freshness: {predicted_label}
                </div>
            """, unsafe_allow_html=True)
            progress.progress(100)  # 100% - classification done
