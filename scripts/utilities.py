import librosa
import joblib
import numpy as np

def features_extractor(file):
  audio,sample_rate = librosa.load(file,duration=2)
  mfccs_features = librosa.feature.mfcc(y=audio,sr=sample_rate,n_mfcc=40)
  mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
  return mfccs_scaled_features
 

def extract_audio_features(y, sr):
    features = {}

    # RMS Energy
    rms = librosa.feature.rms(y=y)
    features['rms_mean'] = np.mean(rms)
    features['rms_var'] = np.var(rms)

    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['rolloff_mean'] = np.mean(rolloff)
    features['rolloff_var'] = np.var(rolloff)

    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spectral_centroid_mean'] = np.mean(spectral_centroid)
    features['spectral_centroid_var'] = np.var(spectral_centroid)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zero_crossing_rate_mean'] = np.mean(zcr)
    features['zero_crossing_rate_var'] = np.var(zcr)

    # Harmony
    harmony = librosa.effects.harmonic(y)
    features['harmony_mean'] = np.mean(harmony)
    features['harmony_var'] = np.var(harmony)

    # Tempo
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    features['tempo'] = tempo

    return features

fit = joblib.load('models/scaler.pkl')

def predict_audio_class(model, audio_path):
    """
    Predicts the genre for a wav audio file
    """
    
    # Load the audio
    y, sr = librosa.load(audio_path, sr=None)

    # Extract features
    features = extract_audio_features(y, sr)

    # Convert features to a numpy array (in correct order)
    feature_array = np.array([
        features['rms_mean'],
        features['rms_var'],
        features['rolloff_mean'],
        features['spectral_centroid_mean'],
        features['spectral_centroid_var'],
        features['rolloff_var'],
        features['zero_crossing_rate_mean'],
        features['zero_crossing_rate_var'],
        features['harmony_mean'],
        features['harmony_var'],
        features['tempo'][0]
    ]).reshape(1, -1)  # Reshape to 2D array for model input

    # Predict with the model
    z = fit.transform(feature_array)
    print(z)
    prediction = model.predict(z)
    return prediction

from pydub import AudioSegment
def convert_mp3_to_wav(mp3_file, wav_file):
    # Load the MP3 file
    audio = AudioSegment.from_mp3(mp3_file)
    # Export as WAV
    audio.export(wav_file, format='wav')




####LSTM Model Functions
genres = {
    0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop',
    5: 'metal', 6: 'pop', 7: 'reggae', 8: 'rock', 9: 'jazz'
}

def predict_genre_LSTM(mp3_file,model):
    # Convert mp3 to wav
    wav_file = "temp/temp_converted_audio.wav"
    convert_mp3_to_wav(mp3_file, wav_file)
    # X = np.array(combined_df['feature'].tolist())
# y = np.array(combined_df['class'].tolist())
# labelencoder = LabelEncoder()
# y = to_categorical(labelencoder.fit_transform(y))
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
# X_train = np.expand_dims(X_train, axis=-1)  # One timestep per sample
# X_test = np.expand_dims(X_test, axis=-1)

    # Extract features and make prediction
    features = features_extractor(wav_file)
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=-1)
    prediction = model.predict(features)
    predicted_index = np.argmax(prediction, axis=1)[0]

    return f"You're listening to {genres[predicted_index]}"