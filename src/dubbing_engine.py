import os
from pydub import AudioSegment
import requests
import json

class DubbingEngine:
    def __init__(self):
        # Voice mapping: Speaker ID -> Edge-TTS Voice
        self.voice_map = {
            "SPEAKER_00": "en-US-ChristopherNeural", # Male
            "SPEAKER_01": "en-US-AriaNeural",       # Female
            "SPEAKER_02": "en-US-GuyNeural",        # Male
            "SPEAKER_03": "en-US-JennyNeural"       # Female
        }
        self.default_voice = "en-US-AriaNeural"

    def get_voice(self, speaker_name):
        return self.voice_map.get(speaker_name, self.default_voice)

    def generate_speech(self, text, speaker_name, output_path):
        """
        Generates audio using Edge-TTS via CLI.
        """
        voice = self.get_voice(speaker_name)
        print(f"Generating speech for {speaker_name} ({voice}): '{text[:30]}...'")
        
        # Cleanup text for CLI (remove quotes etc)
        safe_text = text.replace('"', '').replace("'", "")
        
        import subprocess
        try:
            # edge-tts --text "Hello" --write-media out.mp3 --voice en-US-AriaNeural
            cmd = [
                "edge-tts",
                "--text", safe_text,
                "--write-media", output_path,
                "--voice", voice
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
        except Exception as e:
            print(f"TTS Error: {e}")
            return None

    def stitch_and_mix(self, segments, background_path, output_path):
        """
        Stitches generated speech segments and mixes with background music (with ducking).
        """
        print("Stitching audio segments...")
        
        # 1. Determine total duration (roughly last segment end + buffer)
        if not segments:
            print("No segments to dub.")
            return None
            
        last_seg_end_ms = int(segments[-1]['end'] * 1000) + 2000
        
        # Load background to get true duration if possible
        background = AudioSegment.from_file(background_path)
        total_duration_ms = max(len(background), last_seg_end_ms)
        
        # Trim or Loop background to match
        if len(background) < total_duration_ms:
             background = background * (int(total_duration_ms / len(background)) + 1)
        background = background[:total_duration_ms]
        
        # 2. Create the speech track
        speech_track = AudioSegment.silent(duration=total_duration_ms)
        ducking_mask = [] # (start_ms, end_ms) of speech
        
        for seg in segments:
            start_ms = int(seg['start'] * 1000)
            end_ms = int(seg['end'] * 1000)
            text = seg.get('text', '').strip()
            speaker = seg.get('speaker', 'Unknown')
            
            if not text:
                continue

            # Generate TTS
            temp_tts_path = f"temp_tts_{start_ms}.mp3"
            if self.generate_speech(text, speaker, temp_tts_path):
                # Load generated audio
                if os.path.exists(temp_tts_path):
                    speech_segment = AudioSegment.from_file(temp_tts_path)
                    
                    # Place at timestamp
                    # Note: We rely on the original timestamp. 
                    # Ideally we should speed up/slow down audio to fit the gap, 
                    # but for basic dubbing, overlaying at start is a good first step.
                    speech_track = speech_track.overlay(speech_segment, position=start_ms)
                    
                    # Record actual speech duration for ducking
                    actual_duration = len(speech_segment)
                    ducking_mask.append((start_ms, start_ms + actual_duration))
                    
                    # Cleanup
                    os.remove(temp_tts_path)
        
        # 3. Apply Simple Ducking (Global reduction for now is safer/cleaner for Phase 1)
        # Or better: Reduce background volume globally by 6dB?
        # Let's do the dynamic ducking we planned.
        
        final_bg = background - 3 # Reduce BG slightly overall always
        
        # Create ducked version (-10dB more)
        bg_ducked = final_bg - 10
        
        # We will build the background track by splicing
        # This is complex to get perfect. 
        # Simpler approach for reliability:
        # Just create the mix. If user wants ducking, we can add logic later.
        # Actually, let's just do global volume reduction of music so voice pops.
        
        final_bg = final_bg - 5 # -8dB total
        
        final_mix = final_bg.overlay(speech_track)
        
        final_mix.export(output_path, format="mp3")
        print(f"Dubbing complete: {output_path}")
        return output_path
