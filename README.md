# GTZAN Music Genre Classification

This project provides a Gradio application for classifying music genres using different models, including a pre-trained Hugging Face Transformer model.

## Project Structure

-   `app.py`: The main Gradio application script.
-   `requirements.txt`: Lists the Python dependencies required to run the application.
-   `data/`: Contains CSV files related to song data.
-   `models/`: Contains pre-trained model checkpoints (e.g., `ckpt_ResLSTM_Reg (3).keras`).
-   `scripts/`: Utility scripts for data processing or other tasks.
-   `gtzan_data.ipynb`: Jupyter notebook for data analysis or model training.
-   `old_notebooks/`: Contains older versions of notebooks.

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

The Transformer model requires `ffmpeg` to process audio files. Please install `ffmpeg` and add it to your system's PATH.

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

You can usually install `ffmpeg` using your package manager:

-   **macOS (using Homebrew):**
    ```bash
    brew install ffmpeg
    ```
-   **Ubuntu/Debian:**
    ```bash
    sudo apt update
    sudo apt install ffmpeg
    ```
-   **Fedora:**
    ```bash
    sudo dnf install ffmpeg
    ```

### 4. Run the Gradio Application

Once all dependencies and `ffmpeg` are installed, you can run the Gradio application:

```bash
python app.py
```

The application will start a local server, and you will see a URL in your terminal (e.g., `http://127.0.0.1:7860`). Open this URL in your web browser to access the Gradio interface.

## Gradio App Features

The Gradio application provides options for different music genre classification models:

-   **Transformer Model:** Utilizes a pre-trained `provetgrizzner/distilhubert-finetuned-gtzan/` model from Hugging Face for audio classification.
-   **ResNetLSTM Model:** A placeholder for a ResNetLSTM-based model. (Implementation pending)
-   **No Audio Upload:** A placeholder option that does not require audio input. (Implementation pending)
-   **Audio Extractor Model:** A placeholder for an audio feature extraction model. (Implementation pending)

You can upload an audio file and select your desired model to get genre predictions.
