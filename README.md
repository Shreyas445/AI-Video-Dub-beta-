# AI Movie Dub Maker 🎬🤖

`beta-version (low vram mode)`

An automated pipeline to **extract**, **translate**, and **dub** movie clips from Japanese (or other languages) to English using AI.

## Features ✨
*   **Vocal Separation**: Uses `audio-separator` (Kim Vocal 2) to cleanly isolate speech from background music.
*   **Transcription & Diarization**: Uses `WhisperX` to transcribe text and identify speakers (e.g., SPEAKER_00, SPEAKER_01).
*   **Translation**: Automatically translates Japanese to English.
*   **AI Dubbing**: Generates high-quality Neural English speech using `Edge-TTS` (Free & Unlimited).
*   **Auto Mixing**: Stitches the new voice track back onto the original background music (with auto-ducking).
*   **Web UI**: Simple Gradio interface to drag-and-drop videos.

## Requirements 🛠️
*   **Windows / Linux**
*   **GPU**: NVIDIA RTX 3050 (6GB VRAM) or better recommended.(but most of the case it uses 2.5GB to 4GB VRAM)
*   **FFmpeg**: Must be installed and added to PATH.

## Installation 📦

1.  **Clone the repo**
    `
    just download the zip file of the code
    `

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This installs PyTorch with CUDA 11.8 support by default.*

3.  **Install FFmpeg**
    *   Download from [ffmpeg.org](https://ffmpeg.org/download.html).
    *   Extract and add `bin/` folder to your System PATH.

## Usage 🚀

1.  **Run the Web UI**
    ```bash
    python main.py
    ```

2.  **Open Browser**
    Go to `http://127.0.0.1:7860`

3.  **Start Dubbing**
    *   **Upload Video**: Select your target MP4/MKV file.
    *   **HF Token**: Enter your [Hugging Face Access Token](https://huggingface.co/settings/tokens) (Required for Speaker Diarization models).
    *   **Translate Checkbox**: Ensure "Translate to English" is checked.
    *   **Click Start**: The process will take 2-5 minutes depending on video length.

## Models Used 🧠
*   **Separation**: `Kim_Vocal_2` & `UVR-MDX-NET-Inst_HQ_2`
*   **ASR**: `WhisperX` (large-v2)
*   **Diarization**: `pyannote/speaker-diarization-3.1`
*   **TTS**: `Edge-TTS` (en-US-ChristopherNeural, en-US-AriaNeural)

## License 📄
MIT License.

