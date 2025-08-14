# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2, InceptionV3, EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import pickle
import streamlit as st
from PIL import Image

# Define paths (adjust based on your local directory structure)
base_dir = 'Tumour'  # Assuming the data folder is named 'data' containing train, test, val
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Define image parameters
img_height, img_width = 224, 224  # Standard size for pre-trained models
batch_size = 32
epochs = 50  # Adjustable

# Get class names (11 classes based on folder names)
class_names = [f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f)) and f != '.DS_Store']
num_classes = len(class_names)
print(f"Classes: {class_names}")
print(f"Number of classes: {num_classes}")

# Data Preprocessing and Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # For evaluation
)

# Function to build and train CNN from scratch
def build_cnn_from_scratch():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Function for transfer learning
def build_transfer_model(base_model_name):
    if base_model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    elif base_model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    elif base_model_name == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    elif base_model_name == 'InceptionV3':
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    elif base_model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    
    base_model.trainable = False  # Freeze base layers initially
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model, base_model

# Training function with callbacks
def train_model(model, model_name, base_model=None):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(f'models1/{model_name}_best.keras', monitor='val_accuracy', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
    ]
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    # Fine-tuning for transfer models
    if base_model:
        base_model.trainable = True
        model.compile(optimizer=Adam(learning_rate=0.0001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        history_fine = model.fit(
            train_generator,
            epochs=20,  # Additional epochs for fine-tuning
            validation_data=val_generator,
            callbacks=callbacks
        )
        history.history['loss'].extend(history_fine.history['loss'])
        history.history['accuracy'].extend(history_fine.history['accuracy'])
        history.history['val_loss'].extend(history_fine.history['val_loss'])
        history.history['val_accuracy'].extend(history_fine.history['val_accuracy'])
    
    # Save full model
    model.save(f'models1/{model_name}.keras')
    with open(f'models1/{model_name}_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    
    return model, history

# Evaluation function
def evaluate_model(model, model_name):
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"{model_name} Test Accuracy: {test_acc}")
    
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes
    
    print(classification_report(y_true, y_pred_classes, target_names=class_names))
    
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f'artifacts/{model_name}_cm.png')
    plt.close()
    
    return test_acc

# Visualize training history
def plot_history(history, model_name):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Val Accuracy')
    plt.title(f'Accuracy - {model_name}')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'Loss - {model_name}')
    plt.legend()
    
    plt.savefig(f'artifacts/{model_name}_history.png')
    plt.close()

# Main training loop
models_to_train =  ['CNN_Scratch', 'VGG16','ResNet50', 'MobileNetV2', 'InceptionV3', 'EfficientNetB0']
results = {}

os.makedirs('models', exist_ok=True)
os.makedirs('artifacts', exist_ok=True)

for model_name in models_to_train:
    if model_name == 'CNN_Scratch':
        model = build_cnn_from_scratch()
        base_model = None
    else:
        model, base_model = build_transfer_model(model_name)
    
    trained_model, history = train_model(model, model_name, base_model)
    plot_history(history.history, model_name)
    acc = evaluate_model(trained_model, model_name)
    results[model_name] = acc

# Find best model
best_model_name = max(results, key=results.get)
print(f"Best model: {best_model_name} with accuracy {results[best_model_name]}")

# Comparison Report
print("\nModel Comparison:")
for name, acc in results.items():
    print(f"{name}: {acc}")

# Save comparison to file
with open('artifacts/comparison_report.txt', 'w') as f:
    f.write("Model Comparison:\n")
    for name, acc in results.items():
        f.write(f"{name}: {acc}\n")
