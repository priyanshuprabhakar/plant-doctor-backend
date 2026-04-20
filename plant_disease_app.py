import os
# --- FIX FOR MEMORY ERROR ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import numpy as np
from io import BytesIO
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras import models, layers
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware

# --- CONFIGURATION ---
BATCH_SIZE = 16 # Lowered to prevent Windows I/O crash
IMAGE_SIZE = 256
CHANNELS = 3
EPOCHS = 20
MODEL_PATH = "plant_disease_model.keras"
DATASET_DIR = "dataset"
CONFIDENCE_THRESHOLD = 0.70 # User defined strict threshold

# --- KNOWLEDGE BASE (AI SOLUTIONS) ---
DISEASE_KNOWLEDGE_BASE = {
    "Potato___Early_blight": {
        "description": "Early blight is a common fungal disease characterized by dark, concentric rings on older leaves.",
        "treatment": ["Apply copper-based fungicides.", "Remove infected leaves immediately."],
        "prevention": ["Rotate crops every 2-3 years.", "Use drip irrigation."]
    },
    "Potato___Late_blight": {
        "description": "Late blight is a serious water mold disease causing dark lesions and white fungal growth.",
        "treatment": ["Apply systemic fungicides like metalaxyl.", "Destroy infected plants immediately."],
        "prevention": ["Eliminate cull piles.", "Use certified disease-free seeds."]
    },
    "Potato___Healthy": {
        "description": "The plant appears healthy.",
        "treatment": ["No treatment needed."],
        "prevention": ["Maintain consistent watering.", "Keep area weed-free."]
    },
    "Tomato_Bacterial_spot": {
        "description": "Bacterial spot causes small, water-soaked spots on leaves that turn brown/black.",
        "treatment": ["Apply copper sprays or streptomycin.", "Remove infected plant debris."],
        "prevention": ["Use disease-free seeds.", "Avoid overhead irrigation."]
    },
    "Tomato_Early_blight": {
        "description": "Fungal disease causing 'bullseye' pattern spots on lower leaves.",
        "treatment": ["Apply fungicides (Chlorothalonil or Copper).", "Stake plants to improve airflow."],
        "prevention": ["Mulch soil to prevent splash-back.", "Rotate crops yearly."]
    },
    "Tomato_Late_blight": {
        "description": "A destructive disease causing large, dark, greasy-looking blotches on leaves and stems.",
        "treatment": ["Apply fungicides immediately.", "Remove and destroy plants if severe."],
        "prevention": ["Plant resistant varieties.", "Keep foliage dry."]
    },
    "Tomato_Leaf_Mold": {
        "description": "Fungal disease causing pale green/yellow spots on upper leaves and gray mold underneath.",
        "treatment": ["Apply fungicides.", "Increase spacing for ventilation."],
        "prevention": ["Avoid wetting leaves.", "Sanitize greenhouse tools."]
    },
    "Tomato_Septoria_leaf_spot": {
        "description": "Causes numerous small, circular spots with dark borders and light centers.",
        "treatment": ["Remove lower infected leaves.", "Apply fungicide."],
        "prevention": ["Remove crop debris.", "Mulch around base of plants."]
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "description": "Tiny pests that cause yellow stippling on leaves and fine webbing.",
        "treatment": ["Apply insecticidal soap or neem oil.", "Introduce predatory mites."],
        "prevention": ["Keep plants well-watered.", "Dust off leaves regularly."]
    },
    "Tomato__Target_Spot": {
        "description": "Fungal disease causing brown, necrotic lesions with concentric rings.",
        "treatment": ["Apply fungicides (azoxystrobin).", "Remove infected leaves."],
        "prevention": ["Ensure good airflow.", "Practice crop rotation."]
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "description": "Viral disease transmitted by whiteflies causing upward curling and yellowing leaves.",
        "treatment": ["No cure for infected plants; remove them.", "Control whitefly populations."],
        "prevention": ["Use reflective mulch.", "Use virus-resistant varieties."]
    },
    "Tomato__Tomato_mosaic_virus": {
        "description": "Viral disease causing mottled, mosaic patterns on leaves.",
        "treatment": ["Remove and destroy infected plants.", "Disinfect tools."],
        "prevention": ["Wash hands before handling plants.", "Avoid using tobacco products near plants."]
    },
    "Tomato_healthy": {
        "description": "The tomato plant appears healthy.",
        "treatment": ["No treatment needed."],
        "prevention": ["Maintain regular care routine."]
    },
    "default": {
        "description": "Disease detected, but specific advice is missing for this class.",
        "treatment": ["Consult a local agricultural expert."],
        "prevention": ["Practice general field sanitation."]
    }
}

# --- SMART MATCHER ---
def get_solution(class_name):
    def normalize(text):
        return "".join(e for e in text if e.isalnum()).lower()
    target = normalize(class_name)
    for key, val in DISEASE_KNOWLEDGE_BASE.items():
        if target in normalize(key) or normalize(key) in target:
            return val
    return DISEASE_KNOWLEDGE_BASE["default"]

# --- REAL WORLD IMAGE PREPROCESSING ---
def process_wild_image(image: Image.Image) -> np.ndarray:
    image = ImageOps.autocontrast(image, cutoff=2)
    # Smart Square Crop & Resize
    image = ImageOps.fit(image, (IMAGE_SIZE, IMAGE_SIZE), method=Image.Resampling.LANCZOS)
    return np.array(image)

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    return process_wild_image(image)

# --- ADVANCED MODEL ARCHITECTURE ---
def build_model(num_classes):
    print("Building Advanced EfficientNetB0 Model...")
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.3), 
        layers.RandomZoom(0.3),     
        layers.RandomContrast(0.2),
        layers.RandomTranslation(height_factor=0.2, width_factor=0.2), 
    ])
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
    base_model = tf.keras.applications.EfficientNetB0(input_shape=input_shape, include_top=False, weights='imagenet')
    
    # FINE-TUNING
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model = models.Sequential([
        layers.Input(shape=input_shape),
        data_augmentation,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'), 
        layers.Dropout(0.5), 
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    return model

# --- TRAINING PIPELINE ---
def train_model():
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory '{DATASET_DIR}' not found.")
        return

    print("Loading dataset...")
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_DIR, shuffle=True, image_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE
    )
    class_names = dataset.class_names
    print(f"Classes found: {class_names}")

    def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
        ds_size = len(ds)
        if shuffle:
            ds = ds.shuffle(shuffle_size, seed=12)
        train_size = int(train_split * ds_size)
        val_size = int(val_split * ds_size)
        return ds.take(train_size), ds.skip(train_size).take(val_size), ds.skip(train_size).skip(val_size)

    train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)
    
    # DYNAMIC BUFFER with SPEED LIMITS for Windows
    train_ds = train_ds.shuffle(buffer_size=len(train_ds)).prefetch(buffer_size=2)
    val_ds = val_ds.prefetch(buffer_size=2)
    test_ds = test_ds.prefetch(buffer_size=2)

    print("Building model...")
    model = build_model(len(class_names))
    print("Starting training...")
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1, callbacks=[early_stopping])

    print(f"Saving model to {MODEL_PATH}...")
    model.save(MODEL_PATH)
    with open("class_names.txt", "w") as f:
        f.write("\n".join(class_names))
    print("Training complete!")

# --- FASTAPI BACKEND ---
app = FastAPI(title="Plant Doctor AI API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

MODEL = None
LOADED_CLASSES = []

def load_inference_model():
    global MODEL, LOADED_CLASSES
    if os.path.exists(MODEL_PATH) and os.path.exists("class_names.txt"):
        print("Loading trained model...")
        MODEL = models.load_model(MODEL_PATH)
        with open("class_names.txt", "r") as f:
            LOADED_CLASSES = [line.strip() for line in f.readlines()]
        print("Model loaded.")
    else:
        print("WARNING: Model not found.")

@app.on_event("startup")
async def startup_event():
    load_inference_model()

@app.get("/")
async def ping():
    return "Plant Doctor AI is running"

@app.post("/predict")
async def predict(file: UploadFile = File(...), plant_type: str = Form("auto")):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    image = read_file_as_image(await file.read())
    
    if MODEL:
        img_tensor = tf.convert_to_tensor(image)
        img_batch = tf.expand_dims(img_tensor, 0)
        
        # Get the raw array of probabilities for all 13 classes
        predictions = MODEL.predict(img_batch)[0]
        
        # --- THE MAGIC FIX: PROBABILITY MASKING ---
        if plant_type != "auto":
            # Loop through all classes. If it doesn't match the user's dropdown, set it to 0%
            for i, class_name in enumerate(LOADED_CLASSES):
                if plant_type.lower() not in class_name.lower():
                    predictions[i] = 0.0 
            
            # Re-normalize the math so the remaining options add back up to 100%
            total_prob = np.sum(predictions)
            if total_prob > 0:
                predictions = predictions / total_prob
        # ------------------------------------------

        # Now pick the highest probability from the filtered list
        predicted_class = LOADED_CLASSES[np.argmax(predictions)]
        confidence = float(np.max(predictions))
        status = "Real Model Prediction"
    else:
        import random
        predicted_class = "Potato___Early_blight"
        confidence = 0.99
        status = "MOCK_RESPONSE"

    # Strict Confidence Check
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "class": "Unknown / Unclear Image",
            "confidence": confidence,
            "status": "Low Confidence",
            "solutions": {
                "description": "The AI is not confident enough. Please get closer and ensure the leaf is in focus.",
                "treatment": ["Take a new photo closer to the leaf."],
                "prevention": ["Avoid blurry or heavily shadowed photos."]
            }
        }

    solutions = get_solution(predicted_class)

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
        port = int(os.environ.get("PORT", 8000))
        uvicorn.run(app, host="0.0.0.0", port=port)