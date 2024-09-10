LISTENING_PORT=5000
TEXTURES_BUCKET = 'textures.catverse.app'
MODELS_BUCKET = 'datasets.catverse.app'
ASSETS_BUCKET = 'assets.catverse.app'
GOOGLE_CLOUD_PROJECT_NUMBER = "catverse-poc" #780404922374

TEMP_SOUND_FILE_PATH = "./tempSoundFiles"
SOUND_MODEL_SAVE_PATH = "./models/sound_classifier.pt"
SOUND_MODEL_LABELS_SAVE_PATH = ".models/sound_classifier.json"
AUDIO_PATH=""
LABEL_PATH=""
TARGET_SAMPLE_RATE=16_000
NUM_SAMPLES=int(16_000*0.975) #to align with the 0.975s window size used by YAMNet
WINDOW_SLIDE_STEPS= int(NUM_SAMPLES/2)
N_MELS=128
BATCH_SIZE=64
EPOCHS=1000