import gradio as gr
from transformers import pipeline
import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model

from scripts.utilities import features_extractor, predict_genre_LSTM, predict_audio_class, convert_mp3_to_wav

#Load Models

## Load the transformer model
model_id = "provetgrizzner/distilhubert-finetuned-gtzan"
pipe = pipeline("audio-classification", model=model_id)

## Load ResNetLSTM model
resnet_lstm_model = load_model('models/ckpt_ResLSTM_Reg (3).keras')

## Load Audio Extractor model and label converter
audio_extractor_model_keras = load_model('models/audio_extractor_model.keras')
audio_extractor_converter = joblib.load('models/preprocessing/label_encoder.joblib') # Corrected path

# Define genres for LSTM model (copied from scripts/utilities.py)
lstm_genres = {
    0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop',
    5: 'metal', 6: 'pop', 7: 'reggae', 8: 'rock', 9: 'jazz'
}

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
    
    predictions = predict_genre_LSTM(filepath, resnet_lstm_model)
    outputs = {}
    # Assuming predictions is a 2D array like [[prob1, prob2, ...]]
    for i, score in enumerate(predictions[0]):
        genre_label = lstm_genres.get(i, f"unknown_genre_{i}")
        outputs[genre_label] = float(score) # Ensure score is float
    return outputs
 
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
    
    outputs = {}
    all_labels = audio_extractor_converter.inverse_transform(np.arange(predaud.shape[1]))
    
    for i, score in enumerate(predaud[0]):
        outputs[all_labels[i]] = float(score) # Ensure score is float
    return outputs

def predict_genre(audio_file, model_choice):
    model_dispatch = {
    "Transformer Model": classify_audio_transformer,
    "ResNetLSTM Model": resnetLSTM_model,
    "Audio Extractor Model": audio_extractor_model
    }
    model_function = model_dispatch.get(model_choice)
    if model_function:
        return model_function(audio_file)
    else:
        return {"Error": "Invalid model choice."}


# Gradio Interface

# Model Descriptions from README.md
model_descriptions = {
    "Transformer Model": """
    **Finetuned Hubert Transformer Model**
    - **Description:** A state-of-the-art transformer-based model (`provetgrizzner/distilhubert-finetuned-gtzan`) finetuned for audio classification. These models are highly effective for complex audio tasks due to their ability to capture intricate patterns in raw audio.
    - **Accuracy:** 90%+
    - **Inference Speed:** Slow
    """,
    "ResNetLSTM Model": """
    **ResNetLSTM Model**
    - **Description:** A custom deep learning model combining Residual Networks (ResNet) for feature extraction from audio and Long Short-Term Memory (LSTM) networks for sequence modeling. This architecture is designed to handle temporal dependencies in audio features.
    - **Accuracy:** ~50%
    - **Inference Speed:** Very fast
    """,
    "Audio Extractor Model": """
    **Audio Feature-Based Classifier**
    - **Description:** A traditional machine learning model (likely a Dense Neural Network as seen in `gtzan_data.ipynb`) that classifies genres based on hand-crafted audio features such as MFCCs, RMS, spectral centroid, zero-crossing rate, etc.
    - **Accuracy:** ~80%
    - **Inference Speed:** Very slow
    """
}

with gr.Blocks(theme=gr.themes.Soft(), title="Music Genre Classification") as demo:
    gr.Markdown("# Music Genre Classification")
    gr.Markdown("Upload an audio file and select a model to classify its genre. Click on a model name for more details.")

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(type="filepath", label="Upload Audio")
            model_choice_radio = gr.Radio(
                ["Transformer Model", "ResNetLSTM Model", "Audio Extractor Model"],
                label="Choose Model",
                value="Transformer Model"
            )
            classify_button = gr.Button("Classify Genre")
        with gr.Column(scale=2):
            output_label = gr.Label(label="Predicted Genre")
            model_description_output = gr.Markdown(
                model_descriptions["Transformer Model"],
                label="Model Details"
            )

    def update_model_description(model_name):
        return model_descriptions.get(model_name, "No description available.")

    model_choice_radio.change(
        fn=update_model_description,
        inputs=model_choice_radio,
        outputs=model_description_output
    )

    classify_button.click(
        fn=predict_genre,
        inputs=[audio_input, model_choice_radio],
        outputs=output_label
    )

if __name__ == "__main__":
    demo.launch(debug=True)
