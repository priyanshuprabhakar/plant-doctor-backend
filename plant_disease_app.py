import os
# --- FIX FOR MEMORY ERROR ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import random
import argparse
import numpy as np
from io import BytesIO
from contextlib import asynccontextmanager
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras import models, layers
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware

# --- CONFIGURATION ---
BATCH_SIZE = 16  # Lowered to prevent Windows I/O crash
IMAGE_SIZE = 256
CHANNELS = 3
EPOCHS = 20
MODEL_PATH = os.environ.get("MODEL_PATH", "plant_disease_model.keras")
DATASET_DIR = os.environ.get("DATASET_DIR", "dataset")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.70"))
MISMATCH_THRESHOLD = float(os.environ.get("MISMATCH_THRESHOLD", "0.88"))  # FIX: was 0.50, too aggressive

# Valid plant types accepted by the API
VALID_PLANT_TYPES = {"potato", "tomato"}

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
    # Smart square crop & resize
    image = ImageOps.fit(image, (IMAGE_SIZE, IMAGE_SIZE), method=Image.Resampling.LANCZOS)
    return np.array(image)

def read_file_as_image(data) -> np.ndarray:
    # FIX: wrapped in try/except to handle corrupt or non-image files gracefully
    try:
        image = Image.open(BytesIO(data)).convert("RGB")
        return process_wild_image(image)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file. Please upload a valid image (JPG, PNG, etc).")

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

    # Fine-tuning: freeze all but last 20 layers
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

    # Dynamic buffer with speed limits for Windows
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

# FIX: Use modern lifespan context manager instead of deprecated @app.on_event("startup")
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_inference_model()
    yield

app = FastAPI(title="Plant Doctor AI API", lifespan=lifespan)

# FIX: Restrict CORS origins in production — replace "*" with your frontend domain when deploying
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Change to your domain before going live, e.g. ["https://yourapp.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# FIX: Use app.state instead of bare globals for model storage
def load_inference_model():
    if os.path.exists(MODEL_PATH) and os.path.exists("class_names.txt"):
        print("Loading trained model...")
        app.state.model = models.load_model(MODEL_PATH)
        with open("class_names.txt", "r") as f:
            app.state.class_names = [line.strip() for line in f.readlines()]
        print("Model loaded successfully.")
    else:
        print("WARNING: Model not found. Run with --mode train first.")
        app.state.model = None
        app.state.class_names = []

@app.get("/")
async def ping():
    return {"status": "Plant Doctor AI is running", "model_loaded": app.state.model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...), plant_type: str = Form(...)):
    # FIX: Validate plant_type against allowed values
    if plant_type.lower() not in VALID_PLANT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid plant_type '{plant_type}'. Must be one of: {', '.join(VALID_PLANT_TYPES)}"
        )

    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    # FIX: read_file_as_image now raises HTTPException on corrupt files
    image = read_file_as_image(await file.read())

    # FIX: Raise a proper 503 error if model isn't loaded instead of returning a silent mock
    if app.state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please train the model first using --mode train."
        )

    img_tensor = tf.convert_to_tensor(image)
    img_batch = tf.expand_dims(img_tensor, 0)

    # Get raw probabilities from the model
    raw_predictions = app.state.model.predict(img_batch)[0]

    # Determine the AI's top choice before any masking
    raw_top_index = np.argmax(raw_predictions)
    raw_top_class = app.state.class_names[raw_top_index]
    raw_confidence = float(raw_predictions[raw_top_index])

    print(f"USER SELECTED: {plant_type}")
    print(f"AI RAW GUESS: {raw_top_class} ({raw_confidence * 100:.1f}%)")

    # --- HYBRID MASKING LOGIC ---
    if plant_type.lower() != "auto":
        # FIX: Raised threshold from 0.50 → 0.88
        # The old 0.50 threshold fired almost every time because softmax top predictions
        # are nearly always above 50%, causing potato leaves to be flagged as tomato and vice versa.
        # Now we only reject if the model is very confident (88%+) it's a different plant.
        if plant_type.lower() not in raw_top_class.lower() and raw_confidence > MISMATCH_THRESHOLD:
            return {
                "class": "Plant Mismatch Detected",
                "confidence": raw_confidence,
                "status": "Error",
                "solutions": {
                    "description": f"You selected {plant_type.capitalize()}, but the AI is {(raw_confidence * 100):.0f}% sure this is a {raw_top_class.split('_')[0]} leaf.",
                    "treatment": ["Please check your dropdown selection or ensure you are scanning the correct plant."],
                    "prevention": ["Use 'Auto-Detect' if you are unsure of the plant species."]
                }
            }

        # Apply plant-type mask: zero out classes that don't match
        predictions = np.copy(raw_predictions)
        for i, class_name in enumerate(app.state.class_names):
            if plant_type.lower() not in class_name.lower():
                predictions[i] = 0.0

        total_prob = np.sum(predictions)
        if total_prob > 0:
            # Renormalize so probabilities sum to 1
            predictions = predictions / total_prob
        # If total_prob == 0, all predictions are zero → will fall through as low-confidence unknown
    else:
        predictions = raw_predictions

    # Pick the highest probability class from the (possibly filtered) predictions
    predicted_class = app.state.class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))
    status = "Real Model Prediction"

    # Strict confidence check — return a helpful message if the model isn't sure enough
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