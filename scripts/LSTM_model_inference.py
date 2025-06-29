from tensorflow.keras.models import load_model
from scripts.utilities import predict_genre_LSTM

# Load the LSTM model from checkpoint
model = load_model('models/ckpt_ResLSTM_Reg (3).keras')

# Get MP3 file path from user
mp3_file = input("Enter path to MP3 file: ")

# Predict genre
result = predict_genre_LSTM(mp3_file, model)
print(result)