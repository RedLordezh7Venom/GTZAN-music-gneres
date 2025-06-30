# Music Genre Classification Project

## Table of Contents
- [About the Project](#about-the-project)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Models Used](#models-used)
- [Data Processing and Model Training Notebook (`gtzan_data.ipynb`)](#data-processing-and-model-training-notebook-gtzan_dataipynb)
- [Custom Data Scraping and Dataset Creation](#custom-data-scraping-and-dataset-creation)
- [Use Cases and Project Extras](#use-cases-and-project-extras)

## About the Project
This project implements a Gradio-based web application for classifying music genres using various machine learning models. It is an audio classification deep learning project. The dataset used was GTZAN-Music Genres. It includes data scraping, feature extraction, model training, and a user-friendly interface for real-time predictions.

## Project Structure

```
.
├── app.py                          # Main Gradio application script
├── requirements.txt                # Python dependencies
├── custom_data/                     # Contains CSV files related to data
│   ├── songs.csv
│   └── top_200_songs_by_genre.csv
├── models/                         # Model checkpoints and preprocessing
│   ├── audio_extractor_model.keras
│   ├── ckpt_ResLSTM_Reg (3).keras
│   └── preprocessing/
│       ├── label_encoder.joblib
│       └── scaler.pkl
├── scripts/                        # Utility scripts for data processing
│   ├── audio_extractor_model.py
│   ├── genres.py
│   ├── LSTM_model_inference.py
│   ├── rename.py
│   ├── scrapemusic.py
│   └── utilities.py
├── gtzan_data.ipynb                # Main notebook for analysis and training
├── archived_notebooks/             # Contains older and misc notebooks
├── temp/             
├── .gitignore                      # Specifies intentionally untracked files to ignore
└── README.md                       # Project documentation
```

## Setup Instructions

Follow these steps to set up and run the Gradio application:

### 1. Clone the Repository (if you haven't already)

```bash
git clone https://github.com/your-repo/GTZAN-music-gneres.git
cd GTZAN-music-gneres
```

### 2. Install Python Dependencies

It is recommended to use a virtual environment.

```bash
python -m venv venv
.\venv\Scripts\activate   # On Windows
source venv/bin/activate  # On macOS/Linux
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Install FFmpeg

The Transformer model and audio processing utilities require FFmpeg to handle various audio file formats. Please install FFmpeg and add it to your system's PATH.

**For Windows:**

1.  Go to the official FFmpeg website: [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
2.  Download a build for Windows (e.g., from `gyan.dev` or `btbn.fi`).
3.  Extract the downloaded zip file to a directory (e.g., `C:\ffmpeg`).
4.  Add the `bin` directory inside the extracted folder (e.g., `C:\ffmpeg\bin`) to your system's PATH environment variable.
    *   Search for "Environment Variables" in the Windows search bar.
    *   Click "Edit the system environment variables".
    *   In the System Properties window, click "Environment Variables...".
    *   Under "System variables", find and select the "Path" variable, then click "Edit...".
    *   Click "New" and add the path to your `ffmpeg\bin` directory.
    *   Click "OK" on all windows to close them.
5.  Open a **new** command prompt or PowerShell window and verify the installation by typing `ffmpeg -version`.

**For macOS/Linux:**

You can usually install FFmpeg using your system's package manager:

*   **macOS (using Homebrew):**
    ```bash
    brew install ffmpeg
    ```
*   **Ubuntu/Debian:**
    ```bash
    sudo apt update
    sudo apt install ffmpeg
    ```
*   **Fedora:**
    ```bash
    sudo dnf install ffmpeg
    ```

### 4. Run the Gradio Application

Once all dependencies and FFmpeg are installed, you can run the Gradio application:

```bash
python app.py
```

The application will start a local server, and you will see a URL in your terminal (e.g., `http://127.0.0.1:7860`). Open this URL in your web browser to access the Gradio interface.

## Gradio Interface (`app.py`)

The main application is built using Gradio, providing an interactive web interface to upload audio files and classify their genre using different models.

The interface allows you to:
-   Upload an audio file (MP3 or WAV).
-   Select one of the three available models for classification.
-   View the predicted genre(s) and their confidence scores.

## Models Used

This project utilizes three distinct machine learning models for music genre classification:

### a. Finetuned Hubert Transformer Model
-   **Description:** A state-of-the-art transformer-based model (`provetgrizzner/distilhubert-finetuned-gtzan`) finetuned for audio classification. These models are highly effective for complex audio tasks due to their ability to capture intricate patterns in raw audio.
-   **Accuracy:** 90%+

### b. ResNetLSTM Model
-   **Description:** A custom deep learning model combining Residual Networks (ResNet) for feature extraction from audio and Long Short-Term Memory (LSTM) networks for sequence modeling. This architecture is designed to handle temporal dependencies in audio features.
-   **Accuracy:** ~50%

### c. Audio Feature-Based Classifier
-   **Description:** A traditional machine learning model (likely a Dense Neural Network as seen in `gtzan_data.ipynb`) that classifies genres based on hand-crafted audio features such as MFCCs, RMS, spectral centroid, zero-crossing rate, etc.
-   **Accuracy:** ~80%

## Data Processing and Model Training Notebook (`gtzan_data.ipynb`)

The `gtzan_data.ipynb` Jupyter notebook in the root directory details the entire process of data acquisition, preprocessing, feature engineering, and model training.

**Key Steps in the Notebook:**
-   **Dataset Download:** Downloads the GTZAN dataset from Kaggle.
-   **Exploratory Data Analysis (EDA):** Visualizes audio waveforms and spectrograms.
-   **Feature Engineering:** Extracts various audio features (MFCCs, RMS, spectral properties, tempo, etc.) using `librosa`. The `features_extractor` function is a key component here.
-   **Data Augmentation:** Applies techniques like time stretching and frequency masking to increase the diversity and size of the training dataset, improving model robustness.
-   **Model Definition and Training:**
    -   Trains a simple Dense Neural Network (Audio Feature-Based Classifier).
    -   Trains a ResNet-LSTM model, incorporating L2 regularization, dropout, and learning rate scheduling for improved performance and generalization.
    -   Fine-tunes a pre-trained Hubert Transformer model for audio classification.
-   **Model Saving:** Saves the trained models and preprocessing components (like `scaler.pkl` and `label_encoder.joblib`) to the `models/` directory for later use in `app.py`.

## Custom Data Scraping and Dataset Creation

This project includes scripts to create a custom dataset based on top songs by genre, complementing the GTZAN dataset.

### Data Sources
-   `data/songs.csv`: (If applicable, this would list songs used for custom dataset creation).
-   `data/top_200_songs_by_genre.csv`: This CSV file is generated by the scraping process and contains a list of top songs categorized by genre.

### Procedure for Custom Dataset Creation (using scripts in `scripts/` folder)

The `scripts/` directory contains utilities for data acquisition and preparation:

-   **`scripts/genres.py`**:
    -   **Purpose:** Scrapes song titles for various music genres from `last.fm` (or a similar music website).
    -   **Process:** Iterates through a predefined list of genres, sends requests to the website, parses HTML content using `BeautifulSoup` to extract song titles, and saves the top 200 songs per genre into `top_200_songs_by_genre.csv`. Includes delays to respect website scraping policies.
    -   **How to Run:** `python scripts/genres.py`

-   **`scripts/scrapemusic.py`**:
    -   **Purpose:** Downloads audio snippets (first 30 seconds) of songs listed in `top_200_songs_by_genre.csv` from YouTube.
    -   **Process:** Reads the CSV, uses `yt-dlp` (a command-line program for downloading videos from YouTube and other video sites) to search for and download each song, extracting audio and converting it to WAV format. Downloads are saved to a `downloads/` directory.
    -   **How to Run:** `python scripts/scrapemusic.py`

-   **`scripts/rename.py`**:
    -   **Purpose:** Cleans and renames `.wav` audio files in a specified directory.
    -   **Process:** Removes special characters and replaces spaces with underscores in filenames to ensure compatibility and consistency.
    -   **How to Run:** `python scripts/rename.py` (requires user input for directory path)

-   **`scripts/utilities.py`**:
    -   **Purpose:** Contains helper functions for audio feature extraction and model prediction.
    -   **Key Functions:**
        -   `features_extractor`: Extracts MFCC features from audio files.
        -   `extract_audio_features`: Extracts a broader set of audio features (RMS, spectral rolloff, centroid, ZCR, harmony, tempo).
        -   `predict_audio_class`: Uses a trained model to predict genre based on extracted features.
        -   `convert_mp3_to_wav`: Converts MP3 audio files to WAV format using `pydub`.
        -   `predict_genre_LSTM`: Performs genre prediction specifically for the LSTM model, including MP3 to WAV conversion and feature extraction.

## Use Cases and Project Extras

### Use Cases
-   **Music Recommendation Systems:** The genre classification can be a foundational component for recommending music to users.
-   **Audio Content Organization:** Automatically categorize large audio libraries for easier management and search.
-   **Music Production Tools:** Assist producers in identifying the genre characteristics of their tracks.
-   **Educational Tool:** Demonstrate different machine learning approaches to audio classification.

### Project Extras
-   **Modular Design:** The project is structured with separate scripts for data handling, feature extraction, and application logic, promoting reusability and maintainability.
-   **Multiple Model Support:** Integration of three distinct models allows for comparison of performance and exploration of different classification approaches.
-   **Gradio Interface:** Provides an easy-to-use web interface, making the models accessible without requiring coding knowledge.
-   **Data Augmentation:** The notebook demonstrates techniques to enhance dataset size and diversity, crucial for robust model training.
