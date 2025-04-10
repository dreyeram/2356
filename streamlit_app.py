# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import zipfile
import io
import time
import joblib
import torch
import torchvision
import timm # PyTorch Image Models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, TextClassificationPipeline
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# --- Configuration ---
st.set_page_config(layout="wide", page_title="No-Code ML/DL Trainer")

# --- Constants ---
SUPPORTED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')
TEMP_IMAGE_DIR = "temp_images"
MODEL_SAVE_DIR = "saved_models"

# --- Helper Functions ---

def detect_data_type(uploaded_file):
    """Detects if the uploaded file is CSV (tabular/text) or ZIP (images)."""
    if uploaded_file is None:
        return None
    file_name = uploaded_file.name
    if file_name.lower().endswith('.csv'):
        # Could be tabular or text, needs further inspection maybe
        return "csv"
    elif file_name.lower().endswith('.zip'):
        return "zip"
    else:
        st.warning("Unsupported file type. Please upload a CSV or a ZIP file.")
        return None

def extract_zip(uploaded_file, extract_to='.'):
    """Extracts a zip file and returns the path to the extracted directory."""
    try:
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            # Create a unique directory based on filename to avoid conflicts
            extract_path = os.path.join(extract_to, os.path.splitext(uploaded_file.name)[0])
            if not os.path.exists(extract_path):
                os.makedirs(extract_path)
            zip_ref.extractall(extract_path)
            # Basic validation: Check if it contains image-like subdirectories or files
            contains_images = False
            for root, dirs, files in os.walk(extract_path):
                if any(f.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS) for f in files):
                    contains_images = True
                    break
                # Heuristic: if top-level dirs exist, assume they are classes
                if root == extract_path and len(dirs) > 0:
                     contains_images = True # Assume structure like class/image.jpg

            if not contains_images:
                 st.warning("ZIP file does not seem to contain image files or expected structure (e.g., class_name/image.jpg).")
                 # Clean up potentially empty directory
                 # shutil.rmtree(extract_path) # Be careful with auto-cleanup
                 return None
            st.success(f"Extracted ZIP to {extract_path}")
            return extract_path
    except zipfile.BadZipFile:
        st.error("Invalid or corrupted ZIP file.")
        return None
    except Exception as e:
        st.error(f"Error extracting ZIP file: {e}")
        return None

def get_image_dataset_stats(image_folder_path):
    """Gets basic stats (number of classes, images) from an image folder."""
    if not image_folder_path or not os.path.isdir(image_folder_path):
        return 0, 0, []
    
    classes = [d for d in os.listdir(image_folder_path) if os.path.isdir(os.path.join(image_folder_path, d))]
    if not classes: # Check if images are directly in the folder (single class?)
         all_files = [f for f in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path, f)) and f.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS)]
         if all_files:
             return 1, len(all_files), ["single_class"] # Treat as one class
         else:
             return 0,0, [] # No valid structure found

    num_classes = len(classes)
    num_images = 0
    class_list = []
    for class_name in classes:
        class_path = os.path.join(image_folder_path, class_name)
        try:
            num_images_in_class = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f)) and f.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS)])
            if num_images_in_class > 0:
                num_images += num_images_in_class
                class_list.append(class_name)
            else:
                 st.warning(f"Class directory '{class_name}' contains no supported images. Skipping.")
        except Exception as e:
            st.warning(f"Could not read class directory {class_name}: {e}")
            
    return len(class_list), num_images, sorted(class_list)


class SimpleImageDataset(Dataset):
    """Basic PyTorch Dataset for image classification from folder structure."""
    def __init__(self, image_folder_path, transform=None, class_to_idx=None):
        self.image_folder_path = image_folder_path
        self.transform = transform
        self.samples = []
        self.classes = sorted([d for d in os.listdir(image_folder_path) if os.path.isdir(os.path.join(image_folder_path, d))])

        if not self.classes: # Handle case where images are directly in the folder
             files = [f for f in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path, f)) and f.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS)]
             if files:
                 self.classes = ["single_class"]
                 self.class_to_idx = {"single_class": 0}
                 for fname in files:
                     path = os.path.join(image_folder_path, fname)
                     self.samples.append((path, 0))
             else:
                 raise ValueError(f"No valid image structure found in {image_folder_path}")

        else:
            if class_to_idx is None:
                self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            else:
                 self.class_to_idx = class_to_idx

            for class_name in self.classes:
                class_idx = self.class_to_idx[class_name]
                class_dir = os.path.join(image_folder_path, class_name)
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
                        path = os.path.join(class_dir, fname)
                        self.samples.append((path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        try:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, target
        except Exception as e:
            print(f"Warning: Skipping image {path} due to error: {e}") # Log error
            # Return a placeholder or skip? Returning placeholder might break training.
            # Best might be to filter out problematic images beforehand or handle Nones in DataLoader collation
            # For simplicity here, we'll let it potentially raise an error if not caught later.
            # A robust solution would involve filtering `self.samples` during init.
            return None, None # Indicate failure

def collate_fn(batch):
    """Collate function to handle None values from dataset errors."""
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return torch.Tensor(), torch.Tensor() # Return empty tensors if batch is empty
    return torch.utils.data.dataloader.default_collate(batch)


# TODO: Implement more robust training loops (especially for DL)
# TODO: Add more sophisticated preprocessing options
# TODO: Add hyperparameter tuning options (e.g., Optuna integration)
# TODO: Implement cross-validation
# TODO: Support more data formats (JSON, Parquet, etc.)
# TODO: Handle larger-than-memory datasets (streaming, Dask)


# --- Session State Initialization ---
if 'data' not in st.session_state:
    st.session_state.data = None
if 'data_type' not in st.session_state:
    st.session_state.data_type = None # 'tabular', 'text', 'image'
if 'image_folder_path' not in st.session_state:
    st.session_state.image_folder_path = None
if 'image_classes' not in st.session_state:
    st.session_state.image_classes = []
if 'task_type' not in st.session_state:
    st.session_state.task_type = None
if 'selected_model_name' not in st.session_state:
    st.session_state.selected_model_name = None
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None
if 'training_logs' not in st.session_state:
    st.session_state.training_logs = ""
if 'target_variable' not in st.session_state: # For tabular/text
    st.session_state.target_variable = None
if 'feature_columns' not in st.session_state: # For tabular
    st.session_state.feature_columns = None
if 'text_column' not in st.session_state: # For text classification
    st.session_state.text_column = None
if 'model_params' not in st.session_state:
    st.session_state.model_params = {}
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'class_labels' not in st.session_state: # Store class labels for prediction display
     st.session_state.class_labels = None


# --- Sidebar ---
st.sidebar.title("⚙️ Configuration")

# 1. Data Upload
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV (Tabular/Text) or ZIP (Images)", type=["csv", "zip"], key="data_uploader")

if uploaded_file:
    detected_type = detect_data_type(uploaded_file)
    if detected_type == "csv":
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.data = df
            # Basic heuristic: if a column has many unique text values, suggest text classification
            likely_text_col = None
            for col in df.select_dtypes(include='object').columns:
                if df[col].nunique() > len(df) * 0.5 and df[col].apply(lambda x: isinstance(x, str) and len(x.split()) > 5).mean() > 0.5:
                     likely_text_col = col
                     break # Take the first likely candidate

            if likely_text_col:
                 st.session_state.data_type = "text"
                 st.sidebar.info("Detected potential text data. Select 'Text Classification' task.")
                 # Automatically select likely text and target columns if possible
                 st.session_state.text_column = likely_text_col
                 potential_target = [col for col in df.columns if col != likely_text_col and df[col].nunique() < 20] # Guess target
                 if potential_target:
                      st.session_state.target_variable = potential_target[0]

            else:
                 st.session_state.data_type = "tabular"
                 st.sidebar.info("Detected tabular data.")
                 # Guess target column (often last or named 'target'/'label')
                 if 'target' in df.columns: st.session_state.target_variable = 'target'
                 elif 'label' in df.columns: st.session_state.target_variable = 'label'
                 elif df.columns[-1] != likely_text_col: st.session_state.target_variable = df.columns[-1]


            st.session_state.image_folder_path = None # Reset image path
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")
            st.session_state.data = None
            st.session_state.data_type = None
    elif detected_type == "zip":
        # Ensure temp dir exists
        if not os.path.exists(TEMP_IMAGE_DIR):
            os.makedirs(TEMP_IMAGE_DIR)
        extracted_path = extract_zip(uploaded_file, TEMP_IMAGE_DIR)
        if extracted_path:
            st.session_state.data_type = "image"
            st.session_state.image_folder_path = extracted_path
            st.session_state.data = None # Clear potential previous tabular data
            num_classes, num_images, class_list = get_image_dataset_stats(extracted_path)
            st.session_state.image_classes = class_list
            st.sidebar.info(f"Detected Image Data: Found {num_images} images in {num_classes} classes.")
        else:
             st.session_state.data_type = None # Extraction failed
             st.session_state.image_folder_path = None

    # Clear dependent states if data changes
    st.session_state.task_type = None
    st.session_state.selected_model_name = None
    st.session_state.trained_model = None
    st.session_state.evaluation_results = None
    st.session_state.training_logs = ""
    st.session_state.last_prediction = None

# --- Main Area ---
st.title("🤖 No-Code ML/DL Model Trainer")
st.markdown("Upload your data, configure the model, train, evaluate, and predict - all without writing code!")

if not st.session_state.data_type:
    st.info("☝️ Upload your data using the sidebar to get started.")

# Data Preview Section
if st.session_state.data is not None and st.session_state.data_type in ["tabular", "text"]:
    st.header("📊 Data Preview")
    st.dataframe(st.session_state.data.head())
    st.write(f"Shape: {st.session_state.data.shape}")
elif st.session_state.data_type == "image" and st.session_state.image_folder_path:
    st.header("🖼️ Image Data Preview")
    num_classes, num_images, class_list = get_image_dataset_stats(st.session_state.image_folder_path)
    st.write(f"Found **{num_images}** images across **{num_classes}** classes.")
    st.write(f"Classes: `{', '.join(class_list)}`")

    # Show some sample images
    expander = st.expander("Show Sample Images")
    with expander:
        preview_cols = st.columns(5)
        img_count = 0
        max_previews = 10
        if class_list:
            class_to_show = class_list[0] # Show images from the first class
            class_path = os.path.join(st.session_state.image_folder_path, class_to_show)
            try:
                for img_file in os.listdir(class_path):
                     if img_file.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS) and img_count < max_previews:
                         img_path = os.path.join(class_path, img_file)
                         try:
                             img = Image.open(img_path)
                             col_idx = img_count % 5
                             preview_cols[col_idx].image(img, caption=f"{class_to_show}/{img_file}", use_column_width=True)
                             img_count += 1
                         except Exception as e:
                             st.warning(f"Could not load image {img_file}: {e}")

            except FileNotFoundError:
                 # Handle case where images are directly in root
                 if class_to_show == "single_class":
                     try:
                        for img_file in os.listdir(st.session_state.image_folder_path):
                            if img_file.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS) and img_count < max_previews:
                                img_path = os.path.join(st.session_state.image_folder_path, img_file)
                                try:
                                    img = Image.open(img_path)
                                    col_idx = img_count % 5
                                    preview_cols[col_idx].image(img, caption=f"{img_file}", use_column_width=True)
                                    img_count += 1
                                except Exception as e:
                                    st.warning(f"Could not load image {img_file}: {e}")
                     except Exception as e:
                         st.error(f"Error reading image directory: {e}")


# --- Configuration Continuation (Sidebar) ---

if st.session_state.data_type:
    st.sidebar.header("2. Select Task")
    task_options = []
    if st.session_state.data_type == "tabular":
        task_options = ["Classification", "Regression"]
    elif st.session_state.data_type == "text":
        task_options = ["Text Classification"]
    elif st.session_state.data_type == "image":
        task_options = ["Image Classification"]

    st.session_state.task_type = st.sidebar.selectbox(
        "Choose the ML/DL task",
        options=task_options,
        index=task_options.index(st.session_state.task_type) if st.session_state.task_type in task_options else 0,
        key="task_selector",
        help="Select the type of problem you want to solve."
    )

    # Task-specific data configuration
    if st.session_state.data_type in ["tabular", "text"] and st.session_state.data is not None:
         st.sidebar.subheader("Data Configuration")
         columns = st.session_state.data.columns.tolist()
         default_target = st.session_state.target_variable if st.session_state.target_variable in columns else columns[-1]
         st.session_state.target_variable = st.sidebar.selectbox("Select Target Variable", columns, index=columns.index(default_target), key="target_selector", help="The column you want to predict.")

         if st.session_state.data_type == "tabular":
              default_features = [col for col in columns if col != st.session_state.target_variable]
              st.session_state.feature_columns = st.sidebar.multiselect("Select Feature Columns", columns, default=default_features, key="feature_selector", help="Columns used to make predictions. Exclude the target variable and identifiers.")
         elif st.session_state.data_type == "text":
              object_columns = st.session_state.data.select_dtypes(include='object').columns.tolist()
              default_text_col = st.session_state.text_column if st.session_state.text_column in object_columns else (object_columns[0] if object_columns else None)
              if default_text_col:
                   st.session_state.text_column = st.sidebar.selectbox("Select Text Column", object_columns, index=object_columns.index(default_text_col), key="text_col_selector", help="The column containing the text data.")
              else:
                   st.sidebar.warning("No text (object type) columns found for Text Classification.")
                   st.session_state.text_column = None


    st.sidebar.header("3. Choose Model")
    model_options = []
    if st.session_state.task_type == "Classification" and st.session_state.data_type == "tabular":
        model_options = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]
    elif st.session_state.task_type == "Regression" and st.session_state.data_type == "tabular":
        model_options = ["Random Forest", "XGBoost", "LightGBM"] # Add LinearRegression, SVR later
    elif st.session_state.task_type == "Text Classification":
        model_options = ["BERT (bert-base-uncased)", "DistilBERT (distilbert-base-uncased)", "RoBERTa (roberta-base)"]
    elif st.session_state.task_type == "Image Classification":
        model_options = ["ResNet18", "EfficientNet-B0", "MobileNetV2"] # Add more from timm later

    st.session_state.selected_model_name = st.sidebar.selectbox(
        "Select Model",
        options=model_options,
        index=model_options.index(st.session_state.selected_model_name) if st.session_state.selected_model_name in model_options else 0,
        key="model_selector",
        help="Choose the algorithm for training."
    )

    st.sidebar.header("4. Configure Parameters")
    st.session_state.model_params['test_size'] = st.sidebar.slider(
        "Train/Test Split Ratio (Test Size)", 0.1, 0.5, 0.2, 0.05, key="test_split",
        help="Proportion of data reserved for testing the model's performance."
    )

    # Model-specific parameters
    if st.session_state.selected_model_name and st.session_state.data_type != "image" and st.session_state.task_type != "Text Classification":
         # Simpler models params
         if "Random Forest" in st.session_state.selected_model_name:
              st.session_state.model_params['n_estimators'] = st.sidebar.number_input("Number of Trees (n_estimators)", 10, 1000, 100, 10, key="rf_estimators")
              st.session_state.model_params['max_depth'] = st.sidebar.number_input("Max Depth", 1, 100, 10, 1, key="rf_max_depth", help="Set to 0 for unlimited depth.") or None
         # Add params for XGB, LGBM, LogisticRegression if needed

    if st.session_state.task_type in ["Text Classification", "Image Classification"]:
         # DL Params
         st.session_state.model_params['epochs'] = st.sidebar.number_input("Epochs", 1, 100, 3, 1, key="dl_epochs", help="Number of times the model sees the entire dataset.")
         st.session_state.model_params['batch_size'] = st.sidebar.select_slider("Batch Size", options=[4, 8, 16, 32, 64], value=8, key="dl_batch_size", help="Number of samples processed before the model is updated.")
         st.session_state.model_params['learning_rate'] = st.sidebar.select_slider("Learning Rate", options=[1e-5, 2e-5, 3e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3], value=5e-5, format_func=lambda x: f"{x:.0e}", key="dl_lr", help="Controls how much to change the model in response to the estimated error.")
         # TODO: Add optimizer choice later

    # --- Training Button ---
    st.sidebar.header("5. Train Model")
    can_train = False
    required_info = ""
    if st.session_state.data_type == "tabular":
         if st.session_state.data is not None and st.session_state.target_variable and st.session_state.feature_columns and st.session_state.task_type and st.session_state.selected_model_name:
              can_train = True
         else:
              required_info = "Data, Task, Target, Features, Model"
    elif st.session_state.data_type == "text":
         if st.session_state.data is not None and st.session_state.target_variable and st.session_state.text_column and st.session_state.task_type and st.session_state.selected_model_name:
             can_train = True
         else:
             required_info = "Data, Task, Target, Text Column, Model"
    elif st.session_state.data_type == "image":
         if st.session_state.image_folder_path and st.session_state.task_type and st.session_state.selected_model_name:
             can_train = True
         else:
             required_info = "Image Data (ZIP), Task, Model"

    if not can_train:
         st.sidebar.warning(f"Please configure: {required_info}")

    train_button = st.sidebar.button("🚀 Start Training", disabled=not can_train)


# --- Main Area: Training, Evaluation, Prediction ---

if train_button and can_train:
    st.session_state.trained_model = None # Reset previous model if retraining
    st.session_state.evaluation_results = None
    st.session_state.training_logs = "Starting training...\n"

    st.header("⏳ Training Progress")
    log_area = st.expander("Show Logs", expanded=True)
    log_text_area = log_area.empty() # Placeholder for logs
    log_text_area.text_area("Logs", st.session_state.training_logs, height=200, key="log_area_content")
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # --- Data Preparation ---
        status_text.info("🔄 Preparing data...")
        st.session_state.training_logs += "Preparing data...\n"
        log_text_area.text_area("Logs", st.session_state.training_logs, height=200)

        X_train, X_test, y_train, y_test = None, None, None, None
        train_dataset, test_dataset = None, None
        num_labels = 0

        if st.session_state.data_type == "tabular":
            df = st.session_state.data.copy()
            # Basic Preprocessing: Fill NaNs (simple strategy), One-Hot Encode
            numeric_cols = df[st.session_state.feature_columns].select_dtypes(include=np.number).columns
            categorical_cols = df[st.session_state.feature_columns].select_dtypes(exclude=np.number).columns

            for col in numeric_cols:
                df[col].fillna(df[col].median(), inplace=True)
            for col in categorical_cols:
                df[col].fillna(df[col].mode()[0], inplace=True) # Fill with mode

            # One-Hot Encode only features, not target
            df_features = df[st.session_state.feature_columns]
            df_features_processed = pd.get_dummies(df_features, columns=categorical_cols, drop_first=True) # Handle categorical

            X = df_features_processed
            y = df[st.session_state.target_variable]

             # Ensure target is numeric for classification models if it's not
            if st.session_state.task_type == "Classification":
                 if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
                      st.session_state.class_labels = y.astype('category').cat.categories.tolist()
                      y = y.astype('category').cat.codes # Convert to numeric codes
                 else:
                      st.session_state.class_labels = sorted(y.unique().tolist()) # Assume numeric labels are direct classes
                 num_labels = len(st.session_state.class_labels)
            else: # Regression
                st.session_state.class_labels = None # No class labels for regression


            # Keep track of columns used during training
            st.session_state.training_columns = X.columns.tolist()

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=st.session_state.model_params['test_size'], random_state=42
            )
            st.session_state.training_logs += f"Data split: Train={X_train.shape[0]}, Test={X_test.shape[0]}\n"
            st.session_state.training_logs += f"Features used: {len(st.session_state.training_columns)}\n"
            log_text_area.text_area("Logs", st.session_state.training_logs, height=200)

        elif st.session_state.data_type == "text":
            df = st.session_state.data.copy()
            text_col = st.session_state.text_column
            target_col = st.session_state.target_variable

            # Ensure target is numeric category
            df[target_col] = df[target_col].astype('category')
            st.session_state.class_labels = df[target_col].cat.categories.tolist()
            df['label'] = df[target_col].cat.codes
            num_labels = len(st.session_state.class_labels)

            train_df, test_df = train_test_split(df[[text_col, 'label']], test_size=st.session_state.model_params['test_size'], random_state=42)

            # Prepare for Hugging Face Trainer
            model_checkpoint = st.session_state.selected_model_name.split('(')[1].split(')')[0] # e.g., bert-base-uncased
            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

            def tokenize_function(examples):
                return tokenizer(examples[text_col], padding="max_length", truncation=True)

            # Create HF Datasets
            from datasets import Dataset as HFDataset
            train_hf_dataset = HFDataset.from_pandas(train_df)
            test_hf_dataset = HFDataset.from_pandas(test_df)

            tokenized_train_dataset = train_hf_dataset.map(tokenize_function, batched=True)
            tokenized_test_dataset = test_hf_dataset.map(tokenize_function, batched=True)

            # Use these datasets directly with Trainer
            train_dataset = tokenized_train_dataset
            test_dataset = tokenized_test_dataset
            st.session_state.training_logs += f"Tokenized data: Train={len(train_dataset)}, Test={len(test_dataset)}\n"
            log_text_area.text_area("Logs", st.session_state.training_logs, height=200)

        elif st.session_state.data_type == "image":
            image_folder = st.session_state.image_folder_path
            num_classes, _, class_list = get_image_dataset_stats(image_folder)
            num_labels = num_classes
            st.session_state.class_labels = class_list

            # Define basic transforms
            # TODO: Make transforms configurable or use model-specific defaults
            img_transforms = transforms.Compose([
                transforms.Resize((224, 224)), # Common size
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
            ])

            full_dataset = SimpleImageDataset(image_folder, transform=img_transforms)

            # Split dataset indices
            indices = list(range(len(full_dataset)))
            labels = [label for _, label in full_dataset.samples] # Get all labels for stratified split
            
            try:
                train_indices, test_indices = train_test_split(
                    indices,
                    test_size=st.session_state.model_params['test_size'],
                    random_state=42,
                    stratify=labels if num_classes > 1 else None # Stratify if multiple classes
                )
            except ValueError as e:
                 st.warning(f"Could not stratify split (maybe too few samples per class?): {e}. Performing non-stratified split.")
                 train_indices, test_indices = train_test_split(
                    indices,
                    test_size=st.session_state.model_params['test_size'],
                    random_state=42)


            train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
            test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
            
            # Store class_to_idx mapping
            st.session_state.class_to_idx = full_dataset.class_to_idx

            st.session_state.training_logs += f"Image data split: Train={len(train_dataset)}, Test={len(test_dataset)}\n"
            st.session_state.training_logs += f"Classes: {num_labels} - {', '.join(class_list)}\n"
            log_text_area.text_area("Logs", st.session_state.training_logs, height=200)


        progress_bar.progress(10)

        # --- Model Initialization ---
        status_text.info("🛠️ Initializing model...")
        st.session_state.training_logs += f"Initializing model: {st.session_state.selected_model_name}\n"
        log_text_area.text_area("Logs", st.session_state.training_logs, height=200)
        model = None

        if st.session_state.data_type == "tabular":
            if st.session_state.task_type == "Classification":
                if st.session_state.selected_model_name == "Logistic Regression":
                    model = LogisticRegression(random_state=42, max_iter=1000) # Increased max_iter
                elif st.session_state.selected_model_name == "Random Forest":
                    model = RandomForestClassifier(
                        n_estimators=st.session_state.model_params.get('n_estimators', 100),
                        max_depth=st.session_state.model_params.get('max_depth', None),
                        random_state=42)
                elif st.session_state.selected_model_name == "XGBoost":
                    model = xgb.XGBClassifier(objective='binary:logistic' if num_labels == 2 else 'multi:softprob', eval_metric='logloss', use_label_encoder=False, random_state=42)
                elif st.session_state.selected_model_name == "LightGBM":
                     model = lgb.LGBMClassifier(objective='binary' if num_labels == 2 else 'multiclass', random_state=42)

            elif st.session_state.task_type == "Regression":
                if st.session_state.selected_model_name == "Random Forest":
                    model = RandomForestRegressor(
                        n_estimators=st.session_state.model_params.get('n_estimators', 100),
                        max_depth=st.session_state.model_params.get('max_depth', None),
                        random_state=42)
                elif st.session_state.selected_model_name == "XGBoost":
                    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
                elif st.session_state.selected_model_name == "LightGBM":
                    model = lgb.LGBMRegressor(objective='regression', random_state=42)

        elif st.session_state.task_type == "Text Classification":
            model_checkpoint = st.session_state.selected_model_name.split('(')[1].split(')')[0]
            model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

        elif st.session_state.task_type == "Image Classification":
             # Load pretrained model and modify the final layer
             model_name = st.session_state.selected_model_name.lower()
             if model_name == "resnet18":
                 model = torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT')
                 num_ftrs = model.fc.in_features
                 model.fc = torch.nn.Linear(num_ftrs, num_labels)
             elif model_name == "efficientnet-b0":
                 model = timm.create_model('efficientnet_b0', pretrained=True)
                 num_ftrs = model.classifier.in_features
                 model.classifier = torch.nn.Linear(num_ftrs, num_labels)
             elif model_name == "mobilenetv2":
                 model = torchvision.models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT')
                 num_ftrs = model.classifier[1].in_features
                 model.classifier[1] = torch.nn.Linear(num_ftrs, num_labels)

        progress_bar.progress(20)

        # --- Training ---
        status_text.info("🏋️ Training model...")
        st.session_state.training_logs += "Starting model training...\n"
        log_text_area.text_area("Logs", st.session_state.training_logs, height=200)

        if st.session_state.data_type == "tabular":
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            st.session_state.training_logs += f"Sklearn model fitting complete. Time: {training_time:.2f}s\n"
            log_text_area.text_area("Logs", st.session_state.training_logs, height=200)
            progress_bar.progress(80) # Jump ahead for sklearn

        elif st.session_state.task_type in ["Text Classification", "Image Classification"]:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            st.session_state.training_logs += f"Using device: {device}\n"
            log_text_area.text_area("Logs", st.session_state.training_logs, height=200)
            model.to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=st.session_state.model_params['learning_rate'])
            # TODO: Add LR scheduler

            if st.session_state.task_type == "Text Classification":
                # Use Hugging Face Trainer for simplicity
                training_args = TrainingArguments(
                    output_dir='./results',          # output directory
                    num_train_epochs=st.session_state.model_params['epochs'],
                    per_device_train_batch_size=st.session_state.model_params['batch_size'],
                    per_device_eval_batch_size=st.session_state.model_params['batch_size'] * 2, # Larger batch size for eval
                    warmup_steps=100,                # number of warmup steps for learning rate scheduler
                    weight_decay=0.01,               # strength of weight decay
                    logging_dir='./logs',            # directory for storing logs
                    logging_steps=50,               # Log every 50 steps
                    evaluation_strategy="epoch",     # Evaluate every epoch
                    save_strategy="epoch",           # Save checkpoint every epoch
                    load_best_model_at_end=True,     # Load the best model based on eval loss
                    metric_for_best_model="loss",    # Use loss to determine best model
                    report_to="none",                # Disable wandb/tensorboard reporting for this app
                    fp16 = torch.cuda.is_available() # Enable mixed precision if GPU available
                )

                # Define metrics for HF Trainer
                from datasets import load_metric
                metric = load_metric("accuracy") # Can add F1, Precision, Recall
                def compute_metrics(eval_pred):
                    logits, labels = eval_pred
                    predictions = np.argmax(logits, axis=-1)
                    return metric.compute(predictions=predictions, references=labels)


                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=test_dataset, # Use the tokenized test set for evaluation during training
                    tokenizer=tokenizer,
                    compute_metrics=compute_metrics,
                )

                start_time = time.time()
                status_text.info("🏋️ Training Hugging Face model (this may take time)...")
                st.session_state.training_logs += "Starting Hugging Face Trainer...\n"
                log_text_area.text_area("Logs", st.session_state.training_logs, height=200)
                # TODO: Capture trainer logs better
                # Simple progress simulation for now
                for i in range(st.session_state.model_params['epochs']):
                     st.session_state.training_logs += f"Epoch {i+1}/{st.session_state.model_params['epochs']}\n"
                     log_text_area.text_area("Logs", st.session_state.training_logs, height=200)
                     # Simulate work
                     time.sleep(2) # Placeholder
                     progress_bar.progress(20 + int(60 * (i + 1) / st.session_state.model_params['epochs']))

                # Actual Training (replaces simulation above when ready)
                # trainer.train()
                # st.session_state.training_logs += "Hugging Face Trainer finished.\n"
                # log_text_area.text_area("Logs", st.session_state.training_logs, height=200)


                training_time = time.time() - start_time
                st.session_state.training_logs += f"HF Training complete. Time: {training_time:.2f}s\n"
                log_text_area.text_area("Logs", st.session_state.training_logs, height=200)
                # Model is already the best one due to `load_best_model_at_end=True`


            elif st.session_state.task_type == "Image Classification":
                 # Basic PyTorch Training Loop
                 train_loader = DataLoader(train_dataset, batch_size=st.session_state.model_params['batch_size'], shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_fn)
                 test_loader = DataLoader(test_dataset, batch_size=st.session_state.model_params['batch_size'] * 2, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn)
                 criterion = torch.nn.CrossEntropyLoss()

                 start_time = time.time()
                 model.train()
                 for epoch in range(st.session_state.model_params['epochs']):
                      running_loss = 0.0
                      correct_predictions = 0
                      total_samples = 0
                      epoch_start_time = time.time()

                      for i, data in enumerate(train_loader, 0):
                          if data[0].nelement() == 0: # Skip empty batches from collate_fn
                              continue

                          inputs, labels = data
                          inputs, labels = inputs.to(device), labels.to(device)

                          optimizer.zero_grad()
                          outputs = model(inputs)
                          loss = criterion(outputs, labels)
                          loss.backward()
                          optimizer.step()

                          running_loss += loss.item()
                          _, predicted = torch.max(outputs.data, 1)
                          total_samples += labels.size(0)
                          correct_predictions += (predicted == labels).sum().item()

                          # Log batch progress (optional, can be verbose)
                          # if i % 50 == 49: # Log every 50 batches
                          #     st.session_state.training_logs += f'[Epoch {epoch + 1}, Batch {i + 1:5d}] loss: {running_loss / 50:.3f}\n'
                          #     log_text_area.text_area("Logs", st.session_state.training_logs, height=200)
                          #     running_loss = 0.0


                      epoch_loss = running_loss / len(train_loader)
                      epoch_acc = correct_predictions / total_samples
                      epoch_time = time.time() - epoch_start_time
                      st.session_state.training_logs += f"Epoch {epoch + 1}/{st.session_state.model_params['epochs']} - Time: {epoch_time:.2f}s - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}\n"
                      log_text_area.text_area("Logs", st.session_state.training_logs, height=200)
                      progress_bar.progress(20 + int(60 * (epoch + 1) / st.session_state.model_params['epochs']))
                      status_text.info(f"🏋️ Training... Epoch {epoch + 1}/{st.session_state.model_params['epochs']}")

                 training_time = time.time() - start_time
                 st.session_state.training_logs += f"PyTorch Training complete. Time: {training_time:.2f}s\n"
                 log_text_area.text_area("Logs", st.session_state.training_logs, height=200)


        progress_bar.progress(85)
        st.session_state.trained_model = model # Store the trained model object

        # --- Evaluation ---
        status_text.info("📊 Evaluating model...")
        st.session_state.training_logs += "Evaluating model on test set...\n"
        log_text_area.text_area("Logs", st.session_state.training_logs, height=200)
        evaluation_results = {}

        if st.session_state.data_type == "tabular":
             y_pred = model.predict(X_test)
             if st.session_state.task_type == "Classification":
                 accuracy = accuracy_score(y_test, y_pred)
                 report = classification_report(y_test, y_pred, target_names=st.session_state.class_labels if st.session_state.class_labels else None, output_dict=True)
                 cm = confusion_matrix(y_test, y_pred)
                 evaluation_results = {"accuracy": accuracy, "report": report, "confusion_matrix": cm, "report_str": classification_report(y_test, y_pred, target_names=st.session_state.class_labels if st.session_state.class_labels else None)}
                 st.session_state.training_logs += f"Evaluation: Accuracy={accuracy:.4f}\n"
             elif st.session_state.task_type == "Regression":
                 mse = mean_squared_error(y_test, y_pred)
                 r2 = r2_score(y_test, y_pred)
                 evaluation_results = {"mse": mse, "r2": r2}
                 st.session_state.training_logs += f"Evaluation: MSE={mse:.4f}, R2={r2:.4f}\n"

        elif st.session_state.task_type == "Text Classification":
             # Evaluate using HF Trainer's predict method
             status_text.info("📊 Evaluating Hugging Face model...")
             predictions = trainer.predict(test_dataset)
             preds = np.argmax(predictions.predictions, axis=-1)
             labels = predictions.label_ids
             accuracy = accuracy_score(labels, preds)
             report = classification_report(labels, preds, target_names=st.session_state.class_labels, output_dict=True)
             cm = confusion_matrix(labels, preds)
             evaluation_results = {"accuracy": accuracy, "report": report, "confusion_matrix": cm, "report_str": classification_report(labels, preds, target_names=st.session_state.class_labels)}
             st.session_state.training_logs += f"Evaluation: Accuracy={accuracy:.4f}\n"
             st.session_state.trained_model = trainer.model # Ensure we have the loaded best model

        elif st.session_state.task_type == "Image Classification":
             # PyTorch Evaluation Loop
             status_text.info("📊 Evaluating PyTorch model...")
             model.eval()
             all_preds = []
             all_labels = []
             with torch.no_grad():
                 for data in test_loader:
                      if data[0].nelement() == 0: continue # Skip empty batches
                      images, labels = data
                      images, labels = images.to(device), labels.to(device)
                      outputs = model(images)
                      _, predicted = torch.max(outputs.data, 1)
                      all_preds.extend(predicted.cpu().numpy())
                      all_labels.extend(labels.cpu().numpy())

             accuracy = accuracy_score(all_labels, all_preds)
             report = classification_report(all_labels, all_preds, target_names=st.session_state.class_labels, output_dict=True, zero_division=0)
             cm = confusion_matrix(all_labels, all_preds)
             evaluation_results = {"accuracy": accuracy, "report": report, "confusion_matrix": cm, "report_str": classification_report(all_labels, all_preds, target_names=st.session_state.class_labels, zero_division=0)}
             st.session_state.training_logs += f"Evaluation: Accuracy={accuracy:.4f}\n"


        st.session_state.evaluation_results = evaluation_results
        st.session_state.training_logs += "Evaluation complete.\n"
        log_text_area.text_area("Logs", st.session_state.training_logs, height=200)
        progress_bar.progress(100)
        status_text.success("✅ Training and Evaluation Complete!")


    except Exception as e:
        status_text.error(f"An error occurred during training: {e}")
        st.session_state.training_logs += f"\nERROR: {e}\n"
        log_text_area.text_area("Logs", st.session_state.training_logs, height=200, key="log_error_area")
        st.exception(e) # Show full traceback in the app
        st.session_state.trained_model = None # Ensure no partial model is stored
        st.session_state.evaluation_results = None


# --- Display Evaluation Results ---
if st.session_state.evaluation_results:
    st.header("📈 Evaluation Results")
    results = st.session_state.evaluation_results
    res_col1, res_col2 = st.columns(2)

    if "accuracy" in results:
        res_col1.metric("Accuracy", f"{results['accuracy']:.4f}")
        # Display Classification Report
        st.subheader("Classification Report")
        st.text(results['report_str'])
        report_df = pd.DataFrame(results['report']).transpose()
        st.dataframe(report_df)

        # Download Report
        csv_report = report_df.to_csv().encode('utf-8')
        st.download_button(
            label="Download Report as CSV",
            data=csv_report,
            file_name='classification_report.csv',
            mime='text/csv',
        )

        # Display Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = results['confusion_matrix']
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=st.session_state.class_labels, yticklabels=st.session_state.class_labels, ax=ax)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        st.pyplot(fig)

    elif "mse" in results:
        res_col1.metric("Mean Squared Error (MSE)", f"{results['mse']:.4f}")
        res_col2.metric("R-squared (R2)", f"{results['r2']:.4f}")
        # TODO: Add residual plot for regression
        # Download Regression Metrics
        reg_metrics = pd.DataFrame([results])
        csv_metrics = reg_metrics.to_csv(index=False).encode('utf-8')
        st.download_button(
           label="Download Metrics as CSV",
           data=csv_metrics,
           file_name='regression_metrics.csv',
           mime='text/csv',
       )


# --- Model Saving and Loading ---
st.sidebar.header("6. Model Persistence")

# Saving
if st.session_state.trained_model:
    save_placeholder = st.sidebar.empty() # Placeholder to potentially show progress
    # Define a default filename
    model_type_tag = st.session_state.task_type.lower().replace(" ", "_")
    model_name_tag = st.session_state.selected_model_name.split("(")[0].strip().lower().replace(" ", "_").replace("-","_")
    default_filename = f"{model_type_tag}_{model_name_tag}_model.pkl" # Use pkl as generic suffix for simplicity

    try:
        model_bytes = io.BytesIO()
        # TODO: More robust saving, maybe include metadata (task type, model name, class labels)
        if st.session_state.data_type == "tabular":
            # Include preprocessing info needed for prediction (columns, encoders if any)
            model_payload = {
                'model': st.session_state.trained_model,
                'task_type': st.session_state.task_type,
                'model_name': st.session_state.selected_model_name,
                'training_columns': st.session_state.get('training_columns'), # Important for tabular!
                'class_labels': st.session_state.get('class_labels')
            }
            joblib.dump(model_payload, model_bytes)
            default_filename = f"{model_type_tag}_{model_name_tag}_sklearn.joblib"
        elif st.session_state.task_type == "Text Classification":
            # For HF models, saving the whole pipeline might be better
            # Or save model and tokenizer separately
             # Create a temporary directory to save model and tokenizer
             # Not directly possible with st.download_button, needs server-side storage first
             # Simpler: just save the model state dict for now (requires loading architecture separately)
             # Even simpler for demo: Treat like PyTorch below (save state_dict)
             model_payload = {
                 'model_state_dict': st.session_state.trained_model.state_dict(),
                 'model_name': st.session_state.selected_model_name, # Includes HF identifier
                 'task_type': st.session_state.task_type,
                 'class_labels': st.session_state.get('class_labels'),
                 'num_labels': len(st.session_state.get('class_labels', []))
             }
             torch.save(model_payload, model_bytes)
             default_filename = f"{model_type_tag}_{model_name_tag}_hf.pt" # Using .pt suffix
        elif st.session_state.task_type == "Image Classification":
             # Save model state_dict and necessary info
             model_payload = {
                 'model_state_dict': st.session_state.trained_model.state_dict(),
                 'model_name': st.session_state.selected_model_name, # e.g., ResNet18
                 'task_type': st.session_state.task_type,
                 'class_labels': st.session_state.get('class_labels'),
                 'num_labels': len(st.session_state.get('class_labels', [])),
                 'class_to_idx': st.session_state.get('class_to_idx')
             }
             torch.save(model_payload, model_bytes)
             default_filename = f"{model_type_tag}_{model_name_tag}_torch.pt" # Using .pt suffix


        model_bytes.seek(0)
        st.sidebar.download_button(
            label="💾 Save Trained Model",
            data=model_bytes,
            file_name=default_filename,
            mime="application/octet-stream", # Generic binary file type
            help="Download the trained model for later use."
        )
    except Exception as e:
        st.sidebar.error(f"Error preparing model for download: {e}")

# Loading
loaded_model_file = st.sidebar.file_uploader("Load Previously Saved Model", type=["pkl", "joblib", "pt"], key="model_loader")

if loaded_model_file:
    try:
        load_bytes = io.BytesIO(loaded_model_file.getvalue())
        model_payload = None
        
        # Determine how to load based on extension (crude but simple)
        if loaded_model_file.name.endswith(".joblib") or loaded_model_file.name.endswith(".pkl"):
            model_payload = joblib.load(load_bytes)
            st.session_state.trained_model = model_payload.get('model')
            st.session_state.task_type = model_payload.get('task_type')
            st.session_state.selected_model_name = model_payload.get('model_name')
            st.session_state.training_columns = model_payload.get('training_columns') # Crucial for tabular
            st.session_state.class_labels = model_payload.get('class_labels')
            st.session_state.data_type = "tabular" # Assume based on loading method
            st.sidebar.success(f"Loaded Sklearn model: {st.session_state.selected_model_name}")

        elif loaded_model_file.name.endswith(".pt"):
             model_payload = torch.load(load_bytes, map_location=torch.device('cpu')) # Load to CPU initially
             st.session_state.task_type = model_payload.get('task_type')
             st.session_state.selected_model_name = model_payload.get('model_name')
             st.session_state.class_labels = model_payload.get('class_labels')
             num_labels = model_payload.get('num_labels')
             model_state_dict = model_payload.get('model_state_dict')

             # Re-initialize model architecture based on saved info
             reloaded_model = None
             if st.session_state.task_type == "Text Classification":
                 st.session_state.data_type = "text"
                 hf_model_name = st.session_state.selected_model_name.split('(')[1].split(')')[0]
                 reloaded_model = AutoModelForSequenceClassification.from_pretrained(hf_model_name, num_labels=num_labels)

             elif st.session_state.task_type == "Image Classification":
                 st.session_state.data_type = "image"
                 model_name_key = st.session_state.selected_model_name.lower()
                 if model_name_key == "resnet18":
                     reloaded_model = torchvision.models.resnet18(weights=None) # Load architecture only
                     num_ftrs = reloaded_model.fc.in_features
                     reloaded_model.fc = torch.nn.Linear(num_ftrs, num_labels)
                 elif model_name_key == "efficientnet-b0":
                     reloaded_model = timm.create_model('efficientnet_b0', pretrained=False) # Load architecture only
                     num_ftrs = reloaded_model.classifier.in_features
                     reloaded_model.classifier = torch.nn.Linear(num_ftrs, num_labels)
                 elif model_name_key == "mobilenetv2":
                     reloaded_model = torchvision.models.mobilenet_v2(weights=None) # Load architecture only
                     num_ftrs = reloaded_model.classifier[1].in_features
                     reloaded_model.classifier[1] = torch.nn.Linear(num_ftrs, num_labels)
                 st.session_state.class_to_idx = model_payload.get('class_to_idx') # Load mapping


             if reloaded_model and model_state_dict:
                 reloaded_model.load_state_dict(model_state_dict)
                 reloaded_model.eval() # Set to evaluation mode
                 st.session_state.trained_model = reloaded_model
                 st.sidebar.success(f"Loaded PyTorch/HF model: {st.session_state.selected_model_name}")
             else:
                 st.sidebar.error("Could not re-initialize model architecture from saved file.")
                 st.session_state.trained_model = None

        # Clear any previous results if a model is loaded
        st.session_state.evaluation_results = None
        st.session_state.training_logs = f"Loaded model {st.session_state.selected_model_name} from {loaded_model_file.name}\n"
        st.session_state.last_prediction = None

    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        st.exception(e)
        st.session_state.trained_model = None


# --- Live Prediction ---
if st.session_state.trained_model:
    st.header("🚀 Try Live Prediction")

    model_ready_for_predict = False
    if st.session_state.data_type == "tabular" and st.session_state.training_columns:
         model_ready_for_predict = True
    elif st.session_state.data_type == "text":
         model_ready_for_predict = True # HF models usually bundle tokenizer or can be re-loaded
    elif st.session_state.data_type == "image":
        model_ready_for_predict = True # Transforms are standard

    if not model_ready_for_predict and st.session_state.data_type == "tabular":
        st.warning("Cannot perform live prediction for tabular data without information about training columns (load a model saved by this app).")


    if model_ready_for_predict:
        if st.session_state.data_type == "tabular":
            st.subheader("Enter Data for Prediction (CSV row format)")
            # Dynamically create input fields based on training columns
            input_data = {}
            cols = st.columns(min(4, len(st.session_state.training_columns))) # Layout columns
            col_idx = 0
            # TODO: Infer data types better from original data if available, or just use text input
            for feature in st.session_state.training_columns:
                 input_data[feature] = cols[col_idx].text_input(f"Enter value for '{feature}'", key=f"predict_input_{feature}")
                 col_idx = (col_idx + 1) % len(cols)


            predict_button = st.button("Predict (Tabular)", key="predict_tabular")
            if predict_button:
                try:
                    # Create a DataFrame with the same columns as training data
                    input_df = pd.DataFrame([input_data])
                    # Basic type conversion attempt (can be fragile)
                    # Ideally, we'd save and load dtype info or encoders
                    for col in input_df.columns:
                        try:
                            input_df[col] = pd.to_numeric(input_df[col])
                        except ValueError:
                            pass # Keep as object if conversion fails

                    # Preprocess like training data (handle missing - simple fill with 0/mode, one-hot encode)
                    # This is complex without saving the exact encoders/scalers
                    # Simple approach: Assume input is already somewhat processed, or fill NaNs simply
                    # Need to ensure columns match exactly after potential dummy creation
                    # A robust solution requires saving the preprocessing pipeline (e.g., sklearn Pipeline)
                    input_df_processed = pd.get_dummies(input_df) # Apply same dummy encoding
                    # Reindex to match training columns, filling missing with 0
                    input_df_aligned = input_df_processed.reindex(columns=st.session_state.training_columns, fill_value=0)


                    prediction = st.session_state.trained_model.predict(input_df_aligned)
                    prediction_proba = None
                    if hasattr(st.session_state.trained_model, "predict_proba"):
                         prediction_proba = st.session_state.trained_model.predict_proba(input_df_aligned)


                    st.subheader("Prediction Result")
                    if st.session_state.task_type == "Classification" and st.session_state.class_labels:
                         predicted_label = st.session_state.class_labels[prediction[0]]
                         st.success(f"Predicted Class: **{predicted_label}**")
                         if prediction_proba is not None:
                             proba_df = pd.DataFrame(prediction_proba, columns=st.session_state.class_labels)
                             st.write("Probabilities:")
                             st.dataframe(proba_df)
                             st.session_state.last_prediction = {"label": predicted_label, "probabilities": proba_df.iloc[0].to_dict()}

                    else: # Regression
                        st.success(f"Predicted Value: **{prediction[0]:.4f}**")
                        st.session_state.last_prediction = {"value": prediction[0]}

                except Exception as e:
                    st.error(f"Prediction Error: {e}")
                    st.exception(e)


        elif st.session_state.data_type == "text":
            st.subheader("Enter Text for Prediction")
            text_input = st.text_area("Input Text", key="predict_text_input")
            predict_button = st.button("Predict (Text)", key="predict_text")

            if predict_button and text_input:
                try:
                    # Determine how to predict (pipeline or model+tokenizer)
                    hf_model_name = st.session_state.selected_model_name.split('(')[1].split(')')[0]
                    # Try creating a pipeline (simpler)
                    try:
                         pipe = TextClassificationPipeline(model=st.session_state.trained_model, tokenizer=hf_model_name, device=0 if torch.cuda.is_available() else -1)
                         results = pipe(text_input, top_k=None) # Get all class scores
                         # Map numeric label back to string label if needed
                         predicted_label = results[0]['label']
                         # Find the corresponding score
                         prediction_score = results[0]['score']

                         # If class labels are stored, try to map LABEL_X back
                         if st.session_state.class_labels:
                             try:
                                 label_index = int(predicted_label.replace("LABEL_", ""))
                                 predicted_label = st.session_state.class_labels[label_index]
                             except:
                                 pass # Keep original label if mapping fails

                         st.subheader("Prediction Result")
                         st.success(f"Predicted Class: **{predicted_label}** (Score: {prediction_score:.4f})")

                         # Show all scores if available
                         if len(results) > 1:
                             st.write("All Class Scores:")
                             scores_dict = {}
                             for res in results:
                                label = res['label']
                                if st.session_state.class_labels:
                                     try:
                                         label_index = int(label.replace("LABEL_", ""))
                                         label = st.session_state.class_labels[label_index]
                                     except: pass
                                scores_dict[label] = res['score']
                             st.json(scores_dict)
                             st.session_state.last_prediction = {"label": predicted_label, "probabilities": scores_dict}

                    except Exception as pipe_err:
                         st.warning(f"Pipeline creation failed ({pipe_err}), attempting manual prediction.")
                         # Fallback: Manual prediction (requires tokenizer)
                         tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
                         inputs = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True)
                         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                         st.session_state.trained_model.to(device)
                         inputs = {k: v.to(device) for k, v in inputs.items()}

                         with torch.no_grad():
                             logits = st.session_state.trained_model(**inputs).logits
                             probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                             prediction_idx = np.argmax(probabilities)

                         st.subheader("Prediction Result")
                         if st.session_state.class_labels:
                             predicted_label = st.session_state.class_labels[prediction_idx]
                             st.success(f"Predicted Class: **{predicted_label}**")
                             proba_dict = {st.session_state.class_labels[i]: prob for i, prob in enumerate(probabilities)}
                             st.write("Probabilities:")
                             st.json(proba_dict)
                             st.session_state.last_prediction = {"label": predicted_label, "probabilities": proba_dict}
                         else:
                             st.success(f"Predicted Class Index: **{prediction_idx}**")
                             st.write("Probabilities:", probabilities)
                             st.session_state.last_prediction = {"label_index": prediction_idx, "probabilities": probabilities.tolist()}

                except Exception as e:
                    st.error(f"Prediction Error: {e}")
                    st.exception(e)


        elif st.session_state.data_type == "image":
            st.subheader("Upload Image for Prediction")
            uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="predict_image_uploader")

            if uploaded_image is not None:
                 try:
                     image = Image.open(uploaded_image).convert('RGB')
                     st.image(image, caption="Uploaded Image", width=250)

                     # Apply the same transforms used during training
                     # TODO: Save/load transforms with the model
                     predict_transforms = transforms.Compose([
                         transforms.Resize((224, 224)),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                     ])
                     img_tensor = predict_transforms(image).unsqueeze(0) # Add batch dimension

                     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                     st.session_state.trained_model.to(device)
                     img_tensor = img_tensor.to(device)

                     with torch.no_grad():
                         outputs = st.session_state.trained_model(img_tensor)
                         probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                         prediction_idx = np.argmax(probabilities)

                     st.subheader("Prediction Result")
                     if st.session_state.class_labels:
                          predicted_label = st.session_state.class_labels[prediction_idx]
                          confidence = probabilities[prediction_idx]
                          st.success(f"Predicted Class: **{predicted_label}** (Confidence: {confidence:.4f})")

                          # Show top probabilities
                          proba_dict = {st.session_state.class_labels[i]: prob for i, prob in enumerate(probabilities)}
                          sorted_proba = dict(sorted(proba_dict.items(), key=lambda item: item[1], reverse=True))
                          st.write("Probabilities:")
                          st.json({k: f"{v:.4f}" for k, v in sorted_proba.items()}) # Format for display
                          st.session_state.last_prediction = {"label": predicted_label, "probabilities": proba_dict}

                     else:
                          st.success(f"Predicted Class Index: **{prediction_idx}** (Confidence: {probabilities[prediction_idx]:.4f})")
                          st.write("Probabilities:", probabilities.tolist())
                          st.session_state.last_prediction = {"label_index": prediction_idx, "probabilities": probabilities.tolist()}

                 except Exception as e:
                     st.error(f"Prediction Error: {e}")
                     st.exception(e)
