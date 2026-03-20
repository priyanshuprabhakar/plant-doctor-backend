import os
# --- FIX FOR MEMORY ERROR ---
# This disables Intel's oneDNN optimizations which cause "could not create a memory object" on some CPUs.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras import models, layers
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.layers import Dense, Conv2D

# --- CONFIGURATION ---
BATCH_SIZE = 32
IMAGE_SIZE = 256
CHANNELS = 3
EPOCHS = 25
MODEL_PATH = "plant_disease_model.keras"
DATASET_DIR = "dataset"

# --- KNOWLEDGE BASE (AI SOLUTIONS) ---
# Ensure your dataset folder names match these keys EXACTLY.
DISEASE_KNOWLEDGE_BASE = {
    # --- POTATO CLASSES ---
    "Potato___Early_blight": {
        "description": "Early blight is a common fungal disease characterized by dark, concentric rings on older leaves.",
        "treatment": [
            "Apply copper-based fungicides (mancozeb, chlorothalonil).",
            "Remove infected leaves immediately.",
            "Improve air circulation."
        ],
        "prevention": [
            "Rotate crops every 2-3 years.",
            "Use drip irrigation to keep foliage dry.",
            "Plant resistant varieties."
        ]
    },
    "Potato___Late_blight": {
        "description": "Late blight is a serious water mold disease causing dark lesions and white fungal growth.",
        "treatment": [
            "Apply systemic fungicides like metalaxyl.",
            "Destroy infected plants immediately.",
            "Harvest tubers only after vines die."
        ],
        "prevention": [
            "Eliminate cull piles.",
            "Use certified disease-free seeds.",
            "Monitor cool, wet weather."
        ]
    },
    "Potato___Healthy": {
        "description": "The plant appears healthy.",
        "treatment": ["No treatment needed."],
        "prevention": ["Maintain consistent watering.", "Keep area weed-free."]
    },

    # --- TOMATO CLASSES ---
    "Tomato_Bacterial_spot": {
        "description": "Bacterial spot causes small, water-soaked spots on leaves that turn brown/black.",
        "treatment": [
            "Apply copper sprays or streptomycin.",
            "Remove infected plant debris."
        ],
        "prevention": [
            "Use disease-free seeds.",
            "Avoid overhead irrigation.",
            "Rotate with non-solanaceous crops."
        ]
    },
    "Tomato_Early_blight": {
        "description": "Fungal disease causing 'bullseye' pattern spots on lower leaves.",
        "treatment": [
            "Apply fungicides (Chlorothalonil or Copper).",
            "Stake plants to improve airflow."
        ],
        "prevention": [
            "Mulch soil to prevent splash-back.",
            "Rotate crops yearly."
        ]
    },
    "Tomato_Late_blight": {
        "description": "A destructive disease causing large, dark, greasy-looking blotches on leaves and stems.",
        "treatment": [
            "Apply fungicides immediately (chlorothalonil).",
            "Remove and destroy plants if severe."
        ],
        "prevention": [
            "Plant resistant varieties.",
            "Keep foliage dry."
        ]
    },
    "Tomato_Leaf_Mold": {
        "description": "Fungal disease causing pale green/yellow spots on upper leaves and gray mold underneath.",
        "treatment": [
            "Apply fungicides.",
            "Increase spacing for ventilation."
        ],
        "prevention": [
            "Avoid wetting leaves.",
            "Sanitize greenhouse tools."
        ]
    },
    "Tomato_Septoria_leaf_spot": {
        "description": "Causes numerous small, circular spots with dark borders and light centers.",
        "treatment": [
            "Remove lower infected leaves.",
            "Apply fungicide."
        ],
        "prevention": [
            "Remove crop debris.",
            "Mulch around base of plants."
        ]
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "description": "Tiny pests that cause yellow stippling on leaves and fine webbing.",
        "treatment": [
            "Apply insecticidal soap or neem oil.",
            "Introduce predatory mites."
        ],
        "prevention": [
            "Keep plants well-watered (mites love dry heat).",
            "Dust off leaves regularly."
        ]
    },
    "Tomato__Target_Spot": {
        "description": "Fungal disease causing brown, necrotic lesions with concentric rings.",
        "treatment": [
            "Apply fungicides (azoxystrobin).",
            "Remove infected leaves."
        ],
        "prevention": [
            "Ensure good airflow.",
            "Practice crop rotation."
        ]
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "description": "Viral disease transmitted by whiteflies causing upward curling and yellowing leaves.",
        "treatment": [
            "No cure for infected plants; remove them.",
            "Control whitefly populations."
        ],
        "prevention": [
            "Use reflective mulch.",
            "Use virus-resistant varieties."
        ]
    },
    "Tomato__Tomato_mosaic_virus": {
        "description": "Viral disease causing mottled, mosaic patterns on leaves.",
        "treatment": [
            "Remove and destroy infected plants.",
            "Disinfect tools (virus is highly contagious)."
        ],
        "prevention": [
            "Wash hands before handling plants.",
            "Avoid using tobacco products near plants."
        ]
    },
    "Tomato_healthy": {
        "description": "The tomato plant appears healthy.",
        "treatment": ["No treatment needed."],
        "prevention": ["Maintain regular care routine."]
    },

    # Fallback
    "default": {
        "description": "Disease detected, but specific advice is missing for this class.",
        "treatment": ["Consult a local agricultural expert."],
        "prevention": ["Practice general field sanitation."]
    }
}

# --- MODEL ARCHITECTURE ---
def build_model(num_classes):
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
    ])

    input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)

    model = models.Sequential([
        layers.Input(shape=input_shape),
        data_augmentation,
        layers.Rescaling(1./255),
        layers.Conv2D(32, kernel_size=(3,3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5), 
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    return model

# --- TRAINING PIPELINE ---
def train_model():
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory '{DATASET_DIR}' not found.")
        print("Please structure data as: dataset/ClassName/image.jpg")
        return

    print("Loading dataset...")
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_DIR,
        shuffle=True,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE
    )

    class_names = dataset.class_names
    print(f"Classes found: {class_names}")

    def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
        ds_size = len(ds)
        if shuffle:
            ds = ds.shuffle(shuffle_size, seed=12)
        train_size = int(train_split * ds_size)
        val_size = int(val_split * ds_size)
        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size).take(val_size)
        test_ds = ds.skip(train_size).skip(val_size)
        return train_ds, val_ds, test_ds

    train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    print("Building model...")
    model = build_model(len(class_names))
    
    print("Starting training...")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)

    print(f"Saving model to {MODEL_PATH}...")
    model.save(MODEL_PATH)
    
    with open("class_names.txt", "w") as f:
        f.write("\n".join(class_names))
    print("Training complete!")

# --- FASTAPI BACKEND ---
app = FastAPI(title="Plant Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
LOADED_CLASSES = []

def load_inference_model():
    global MODEL, LOADED_CLASSES
    if os.path.exists(MODEL_PATH) and os.path.exists("class_names.txt"):
        print("Loading trained model...")
        MODEL = models.load_model(MODEL_PATH)
        with open("class_names.txt", "r") as f:
            LOADED_CLASSES = [line.strip() for line in f.readlines()]
        print(f"Model loaded. Classes: {LOADED_CLASSES}")
    else:
        print("WARNING: Model not found. API will run in MOCK mode.")

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)).convert("RGB"))
    return image

@app.on_event("startup")
async def startup_event():
    load_inference_model()

@app.get("/")
async def ping():
    return "Plant Disease Detection API is running"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    image = read_file_as_image(await file.read())
    
    if MODEL:
        img_tensor = tf.convert_to_tensor(image)
        img_resized = tf.image.resize(img_tensor, [IMAGE_SIZE, IMAGE_SIZE])
        img_batch = tf.expand_dims(img_resized, 0)
        predictions = MODEL.predict(img_batch)
        predicted_class = LOADED_CLASSES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        status = "Real Model Prediction"
    else:
        import random
        mock_classes = ["Tomato___Early_blight", "Potato___Early_blight", "Tomato___Healthy"]
        predicted_class = random.choice(mock_classes)
        confidence = random.uniform(0.7, 0.99)
        status = "MOCK_RESPONSE (Train model to get real results)"

    print(f"\n--- DEBUG INFO ---")
    print(f"Predicted Class: '{predicted_class}'")
    
    if predicted_class in DISEASE_KNOWLEDGE_BASE:
        print(f"Match found in Knowledge Base!")
        solutions = DISEASE_KNOWLEDGE_BASE[predicted_class]
    else:
        print(f"WARNING: '{predicted_class}' NOT found in keys.")
        solutions = DISEASE_KNOWLEDGE_BASE["default"]
    print(f"------------------\n")

    return {
        "class": predicted_class,
        "confidence": confidence,
        "status": status,
        "solutions": solutions
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='serve', choices=['train', 'serve'])
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_model()
    else:
        uvicorn.run(app, host="localhost", port=8000)