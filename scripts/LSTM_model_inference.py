from tensorflow.keras.models import load_model
import numpy as np
from scripts.utilities import predict_genre_LSTM,genres

# Load the LSTM model from checkpoint
model = load_model('models/ckpt_ResLSTM_Reg (3).keras')

# Get MP3 file path from user
mp3_file = input("Enter path to MP3 file: ")



result = predict_genre_LSTM(mp3_file, model)
# Predict genre
predicted_index = np.argmax(result, axis=1)[0]

print(f"You're listening to {genres[predicted_index]}")