import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from keras.models import load_model
from tkinter import messagebox, filedialog
import time
from PIL import Image, ImageTk
from Feature_Extraction.Deep_map_based_feature import Deep_map_based_features
from Feature_Extraction.Hybrid_Depth_based_textural_pattern import Hybrid_depth_based_textural_pattern
from Feature_Extraction.Inception_v3 import Inception_model
from Feature_Extraction.ROI_Extraction import ROI_Extraction
from Feature_Extraction.Modified_pixel_intensity_based_structural_pattern import \
    Modified_pixel_intensity_based_structural_pattern
import matplotlib.pyplot as plt
import matplotlib.cm as cm

root = tk.Tk()

# Global variables for images and models
model11 = None
model22 = None
img_final = None
pil_image = None
depth_map_feature = None
hybrid_depth = None
inception_model1 = None
pixel_intensity = None
roi1 = None

# Global PhotoImage references to prevent garbage collection
current_display_image = None


def button1_function():
    global model11
    try:
        model1 = load_model("Saved_models/DB1.h5")
        time.sleep(2)
        model11 = model1
        print("DB1 model loaded Successfully....")
        messagebox.showinfo("Success", "DB1 model loaded successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load DB1 model: {str(e)}")


def button2_function():
    global model22
    try:
        model1 = load_model("Saved_models/DB2.h5")
        time.sleep(2)
        model22 = model1
        print("DB2 model loaded Successfully....")
        messagebox.showinfo("Success", "DB2 model loaded successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load DB2 model: {str(e)}")


def button3_function():
    global img_final, pil_image, current_display_image
    file = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.png *.tif *.jpeg")],
        initialdir="Test_data"
    )
    if file:
        img = cv2.imread(file)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_final = img

            pil_img = Image.open(file).convert("RGB")
            pil_image = pil_img

            # Display original image
            pil_display = pil_img.resize((300, 300), Image.Resampling.LANCZOS)
            current_display_image = ImageTk.PhotoImage(pil_display)
            display_area.config(image=current_display_image, text="")
            status_label.config(text="Original image loaded")
        else:
            messagebox.showerror("Error", "Could not load the image file!")


def display_image_properly(numpy_img, title="Image"):
    """Convert numpy array to displayable Tkinter image with proper normalization"""
    global current_display_image

    if numpy_img.dtype != np.uint8:
        # Normalize float arrays to 0-255 uint8
        if numpy_img.max() > 1.0:
            normalized = np.clip((numpy_img - numpy_img.min()) /
                                 (numpy_img.max() - numpy_img.min()) * 255, 0, 255)
        else:
            normalized = np.clip(numpy_img * 255, 0, 255)
        numpy_img = normalized.astype(np.uint8)

    # Handle different channel dimensions
    if len(numpy_img.shape) == 2:  # Grayscale
        pil_img = Image.fromarray(numpy_img, mode='L')
    elif len(numpy_img.shape) == 3:
        if numpy_img.shape[2] == 3:  # RGB
            pil_img = Image.fromarray(numpy_img, mode='RGB')
        else:  # Convert to RGB
            pil_img = Image.fromarray(numpy_img[:, :, :3], mode='RGB')
    else:
        return

    # Resize for display
    display_size = (300, 300)
    pil_display = pil_img.resize(display_size, Image.Resampling.LANCZOS)
    current_display_image = ImageTk.PhotoImage(pil_display)
    display_area.config(image=current_display_image, text="")
    status_label.config(text=f"{title} displayed")


def depth_map_based_feature():
    global depth_map_feature
    if img_final is None:
        messagebox.showwarning("Warning", "Please load an image first!")
        return

    try:
        dep = Deep_map_based_features(img_final)
        depth_map_feature = dep

        # Display with colormap for better visualization
        norm_depth = (dep - dep.min()) / (dep.max() - dep.min())
        colormap = cm.plasma(norm_depth)[:, :, :3]  # RGB
        depth_colored = (colormap * 255).astype(np.uint8)

        display_image_properly(depth_colored, "Depth Map")
        print("Depth map features extracted successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Depth map extraction failed: {str(e)}")


def Hybrid_depth_text_pt():
    global hybrid_depth
    if pil_image is None:
        messagebox.showwarning("Warning", "Please load an image first!")
        return

    try:
        dyp = Hybrid_depth_based_textural_pattern(pil_image)
        hybrid_depth = dyp
        status_label.config(text="Hybrid depth features extracted")
        messagebox.showinfo("Success", "Hybrid depth features extracted!")
        print("Hybrid depth features extracted successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Hybrid depth extraction failed: {str(e)}")


def Inception_model1():
    global inception_model1
    if img_final is None:
        messagebox.showwarning("Warning", "Please load an image first!")
        return

    try:
        inc = Inception_model(img_final)
        inception_model1 = inc
        status_label.config(text="Inception_v3 features extracted")
        messagebox.showinfo("Success", "Inception_v3 features extracted!")
        print("Inception_v3 features extracted successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Inception_v3 extraction failed: {str(e)}")


def Modified_pixel_intensity():
    global pixel_intensity
    if img_final is None:
        messagebox.showwarning("Warning", "Please load an image first!")
        return

    try:
        pix = Modified_pixel_intensity_based_structural_pattern(img_final)
        pixel_intensity = pix
        status_label.config(text="Pixel intensity features extracted")
        messagebox.showinfo("Success", "Pixel intensity features extracted!")
        print("Pixel intensity features extracted successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Pixel intensity extraction failed: {str(e)}")


def ROI_Extraction1():
    global roi1
    if img_final is None:
        messagebox.showwarning("Warning", "Please load an image first!")
        return

    try:
        roi_img = ROI_Extraction(img_final)
        roi1 = roi_img
        display_image_properly(roi_img, "ROI")
        print("ROI extracted successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"ROI extraction failed: {str(e)}")


def model_prediction_function():
    if model11 is None and model22 is None:
        messagebox.showwarning("Warning", "Please load at least one model first!")
        return
    if img_final is None:
        messagebox.showwarning("Warning", "Please load an image first!")
        return

    messagebox.showinfo("Info", "Model prediction functionality to be implemented!")


# Create modern styled GUI
root.title("Papaya Grading GUI - Advanced Feature Extraction")
root.geometry("800x700")
root.configure(bg='#2c3e50')

# Style configuration
style = {
    'bg': '#34495e',
    'fg': 'white',
    'font_title': ('Arial', 16, 'bold'),
    'font_button': ('Arial', 12, 'bold'),
    'font_status': ('Arial', 10),
    'button_bg': '#3498db',
    'button_active': '#2980b9',
    'success_bg': '#27ae60'
}

# Title
title1 = tk.Label(root, text="🌿 Papaya Grading System",
                  font=style['font_title'], bg=style['bg'], fg='gold')
title1.pack(pady=20)

# Control Frame
control_frame = tk.Frame(root, bg=style['bg'])
control_frame.pack(pady=10)

# Model Loading Buttons
tk.Button(control_frame, text="🔄 Load DB1 Model", command=button1_function,
          bg=style['button_bg'], fg='white', font=style['font_button'],
          relief='flat', padx=20, pady=8).pack(pady=5)
tk.Button(control_frame, text="🔄 Load DB2 Model", command=button2_function,
          bg=style['button_bg'], fg='white', font=style['font_button'],
          relief='flat', padx=20, pady=8).pack(pady=5)

# Image Selection
tk.Button(control_frame, text="📁 Select Image", command=button3_function,
          bg='#f39c12', fg='white', font=style['font_button'],
          relief='flat', padx=20, pady=8).pack(pady=10)

# Feature Extraction Buttons
feature_frame = tk.Frame(root, bg=style['bg'])
feature_frame.pack(pady=10)

tk.Button(feature_frame, text="📊 Depth Map Features", command=depth_map_based_feature,
          bg='#9b59b6', fg='white', font=style['font_button'],
          relief='flat', padx=15, pady=8).pack(pady=3, fill='x', padx=20)
tk.Button(feature_frame, text="🔗 Hybrid Depth Texture", command=Hybrid_depth_text_pt,
          bg='#e74c3c', fg='white', font=style['font_button'],
          relief='flat', padx=15, pady=8).pack(pady=3, fill='x', padx=20)
tk.Button(feature_frame, text="🧠 Inception_v3", command=Inception_model1,
          bg='#1abc9c', fg='white', font=style['font_button'],
          relief='flat', padx=15, pady=8).pack(pady=3, fill='x', padx=20)
tk.Button(feature_frame, text="🎨 Pixel Intensity SP", command=Modified_pixel_intensity,
          bg='#f39c12', fg='white', font=style['font_button'],
          relief='flat', padx=15, pady=8).pack(pady=3, fill='x', padx=20)
tk.Button(feature_frame, text="✂️ ROI Extraction", command=ROI_Extraction1,
          bg='#e67e22', fg='white', font=style['font_button'],
          relief='flat', padx=15, pady=8).pack(pady=3, fill='x', padx=20)

# Model Prediction
tk.Button(root, text="🚀 Run Model Prediction", command=model_prediction_function,
          bg=style['success_bg'], fg='white', font=style['font_button'],
          relief='flat', padx=30, pady=10).pack(pady=20)

# Display Area
display_frame = tk.LabelFrame(root, text="Image Display", font=('Arial', 12, 'bold'),
                              bg=style['bg'], fg='white', padx=10, pady=10)
display_frame.pack(pady=20, padx=50, fill='both', expand=True)

display_area = tk.Label(display_frame, text="No image loaded", bg="gray",
                        width=40, height=25, relief='sunken', bd=2)
display_area.pack(pady=10)

# Status Label
status_label = tk.Label(root, text="Ready", bg=style['bg'], fg=style['success_bg'],
                        font=style['font_status'], pady=5)
status_label.pack(pady=10)

# Instructions
instr_label = tk.Label(root, text="👆 Load image first, then extract features!",
                       bg=style['bg'], fg='lightblue', font=style['font_status'])
instr_label.pack(pady=5)

root.mainloop()
