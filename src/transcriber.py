import whisperx
import torch
import gc
from src.utils import clear_gpu_memory

class Transcriber:
    def __init__(self, device="cuda", compute_type="int8", hf_token=None):
        self.device = device
        self.compute_type = compute_type
        self.hf_token = hf_token
        self.model = None
        
    def transcribe_and_diarize(self, audio_path, num_speakers=None, task="transcribe"):
        """
        Transcribes audio and diarizes speakers using WhisperX.
        Returns a list of segments with speaker labels.
        task: "transcribe" (original lang) or "translate" (to English).
        """
        print(f"Loading WhisperX model (Device: {self.device}, Type: {self.compute_type})...")
        
        # 1. Transcribe
        try:
            self.model = whisperx.load_model(
                "large-v2", 
                self.device, 
                compute_type=self.compute_type
            )
        except Exception as e:
            print(f"Failed to load WhisperX model: {e}")
            # Fallback to CPU if CUDA fails
            if self.device == "cuda":
                print("Falling back to CPU...")
                self.device = "cpu"
                self.compute_type = "int8"
                self.model = whisperx.load_model("large-v2", self.device, compute_type="int8")

        audio = whisperx.load_audio(audio_path)
        # Reduced batch size for 6GB VRAM safety
        # Pass task="translate" here to auto-translate to English
        result = self.model.transcribe(audio, batch_size=4, task=task)
        
        # Align (improves timestamps)
        try:
            print("Loading Alignment model...")
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, device=self.device, return_char_alignments=False)
            del model_a
            clear_gpu_memory()
        except Exception as e:
            print(f"Alignment failed (Security block or network issue): {e}")
            print("Using original timestamps (Slightly less precise).")
            # Clear if partially loaded
            clear_gpu_memory()

        # 2. Diarize
        print("Loading Diarization model...")
        diarize_model = None
        try:
            from whisperx.diarize import DiarizationPipeline
            diarize_model = DiarizationPipeline(use_auth_token=self.hf_token, device=self.device)
            diar_segments = diarize_model(audio, min_speakers=num_speakers, max_speakers=num_speakers)
            result = whisperx.assign_word_speakers(diar_segments, result)
            print("Diarization complete.")
        except Exception as e:
            print(f"Diarization failed: {e}")
            print("Proceeding with transcription only (No Speaker IDs).")
        
        # Cleanup
        del self.model
        if diarize_model:
            del diarize_model
        self.model = None
        clear_gpu_memory()
        
        return result["segments"]
