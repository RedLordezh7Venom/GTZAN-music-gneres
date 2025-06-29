import gradio as gr
from transformers import pipeline
import os

from scripts.utilities import features_extractor

# Load the transformer model
model_id = "provetgrizzner/distilhubert-finetuned-gtzan"
pipe = pipeline("audio-classification", model=model_id)

def classify_audio_transformer(filepath):
    if filepath is None:
        return {"Error": "No audio file uploaded for Transformer model."}
    preds = pipe(filepath)
    outputs = {}
    for p in preds:
        outputs[p["label"]] = p["score"]
    return outputs

def resnetLSTM_model(filepath):
    # Placeholder for ResNetLSTM model
    return {"ResNetLSTM": "Model not implemented yet. Uploaded file: " + (filepath if filepath else "None")}

def no_audio_upload():
    # Placeholder for "no audio upload" option
    return {"Info": "This option does not require audio upload."}

def audio_extractor_model(filepath):
    # Placeholder for Audio Extractor model
    return {"Audio Extractor": "Model not implemented yet. Uploaded file: " + (filepath if filepath else "None")}

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
