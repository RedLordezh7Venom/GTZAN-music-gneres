import gradio as gr
from transformers import pipeline
import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model

from scripts.utilities import features_extractor, predict_genre_LSTM, predict_audio_class, convert_mp3_to_wav

# Load the transformer model
model_id = "provetgrizzner/distilhubert-finetuned-gtzan"
pipe = pipeline("audio-classification", model=model_id)

# Load ResNetLSTM model
resnet_lstm_model = load_model('models/ckpt_ResLSTM_Reg (3).keras')

# Load Audio Extractor model and label converter
audio_extractor_model_keras = load_model('models/audio_extractor_model.keras')
audio_extractor_converter = joblib.load('models/label_encoder.joblib')

def classify_audio_transformer(filepath):
    if filepath is None:
        return {"Error": "No audio file uploaded for Transformer model."}
    preds = pipe(filepath)
    outputs = {}
    for p in preds:
        outputs[p["label"]] = p["score"]
    return outputs

def resnetLSTM_model(filepath):
    if filepath is None:
        return {"Error": "No audio file uploaded for ResNetLSTM model."}
    
    prediction_string = predict_genre_LSTM(filepath, resnet_lstm_model)
    # Extract the genre from the string "You're listening to [genre]"
    genre = prediction_string.replace("You're listening to ", "").strip()
    return {genre: 1.0} # Return with a confidence score for Gradio Label
 
def audio_extractor_model(filepath):
    if filepath is None:
        return {"Error": "No audio file uploaded for Audio Extractor model."}
    
    # Create a temporary directory if it doesn't exist
    os.makedirs('temp', exist_ok=True)
    wav_file_path = 'temp/music.wav'
    
    # Convert the uploaded audio to WAV
    convert_mp3_to_wav(filepath, wav_file_path)
    
    # Predict using the audio extractor model
    predaud = predict_audio_class(audio_extractor_model_keras, wav_file_path)
    predicted_label = audio_extractor_converter.inverse_transform([np.argmax(predaud)])
    
    # Clean up the temporary WAV file
    os.remove(wav_file_path)
    
    return {predicted_label[0]: 1.0} # Return with a confidence score for Gradio Label

def predict_genre(audio_file, model_choice):
    if model_choice == "Transformer Model":
        return classify_audio_transformer(audio_file)
    elif model_choice == "ResNetLSTM Model":
        return resnetLSTM_model(audio_file)
    elif model_choice == "No Audio Upload":
        return no_audio_upload()
    elif model_choice == "Audio Extractor Model":
        return audio_extractor_model(audio_file)
    else:
        return {"Error": "Invalid model choice."}

# Gradio Interface
demo = gr.Interface(
    fn=predict_genre,
    inputs=[
        gr.Audio(type="filepath", label="Upload Audio"),
        gr.Radio(["Transformer Model", "ResNetLSTM Model", "No Audio Upload", "Audio Extractor Model"], label="Choose Model", value="Transformer Model")
    ],
    outputs=gr.Label(),
    title="Music Genre Classification",
    description="Upload an audio file and select a model to classify its genre."
)

if __name__ == "__main__":
    demo.launch(debug=True)
