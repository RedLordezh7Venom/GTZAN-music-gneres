import joblib
import numpy as np
from scripts.utilities import predict_audio_class,convert_mp3_to_wav
from tensorflow.keras.models import load_model 

model = load_model('models/preprocessing/audio_extractor_model.keras')
converter = joblib.load('models/preprocessing/label_encoder.joblib')

music_file = input("Enter path to music file")
convert_mp3_to_wav(music_file,'temp/music.wav')
predaud = predict_audio_class(model, 'temp/music.wav')

predicted_label = converter.inverse_transform([np.argmax(predaud)])
print("Predicted Genre : ", predicted_label)