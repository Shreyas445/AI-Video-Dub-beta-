import os
import subprocess
from audio_separator.separator import Separator
from src.utils import clear_gpu_memory

class AudioProcessor:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def extract_audio(self, video_path):
        """
        Extracts audio from video using ffmpeg.
        """
        audio_output = os.path.join(self.output_dir, "input_audio.wav")
        # Overwrite if exists
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "2",
            audio_output
        ]
        print(f"Extracting audio from {video_path}...")
        subprocess.run(cmd, check=True)
        return audio_output

    def separate_audio(self, audio_path):
        """
        Separates audio into Vocals (Kim Vocal 2) and Instrumental (Inst HQ 2).
        Returns paths to the separated files.
        """
        
        import onnxruntime as ort
        print(f"DEBUG: Available ORT Providers: {ort.get_available_providers()}")
        print("Starting audio separation...")
        
        # Phase 1: Vocals using Kim Vocal 2
        import logging
        print("Loading Kim_Vocal_2 for vocal extraction...")
        separator = Separator(
            log_level=logging.INFO, # Change to DEBUG if needed, INFO should show device
            output_dir=self.output_dir,
            output_format="wav",
            model_file_dir=os.path.join(self.output_dir, "models")
        )
        
        # Note: audio-separator typically downloads models automatically.
        # We assume standard usage.
        # Kim Vocal 2 is often used for vocals. 
        # Inst HQ 2 is for instrumentals.
        
        # Run separation for Vocals
        separator.load_model(model_filename="Kim_Vocal_2.onnx")
        output_files_vocals = separator.separate(audio_path)
        print(f"Kim Vocal 2 Output: {output_files_vocals}")

        if not output_files_vocals:
            raise Exception("Separation failed: No output files produced for Vocals.")
        
        # Find the file with "(Vocals)" in the name
        vocals_path = None
        for f in output_files_vocals:
            if "(Vocals)" in f:
                vocals_path = os.path.join(self.output_dir, f)
                break
        if not vocals_path:
             # Fallback if naming convention differs, though Kim Vocal 2 usually adheres to this
             vocals_path = os.path.join(self.output_dir, output_files_vocals[0])

        clear_gpu_memory()

        # Phase 2: Background using Inst HQ 2
        print("Loading UVR-MDX-NET-Inst_HQ_2 for instrumental extraction...")
        separator.load_model(model_filename="UVR-MDX-NET-Inst_HQ_2.onnx")
        output_files_inst = separator.separate(audio_path)
        print(f"Inst HQ 2 Output: {output_files_inst}")

        if not output_files_inst:
            raise Exception("Separation failed: No output files produced for Instrumental.")
        
        # Find the file with "(Instrumental)" in the name
        inst_path = None
        for f in output_files_inst:
            if "(Instrumental)" in f:
                inst_path = os.path.join(self.output_dir, f)
                break
        if not inst_path:
             inst_path = os.path.join(self.output_dir, output_files_inst[0])

        clear_gpu_memory()
        
        return vocals_path, inst_path

if __name__ == "__main__":
    # Test stub
    processor = AudioProcessor()
    # processor.extract_audio("test_video.mp4")
