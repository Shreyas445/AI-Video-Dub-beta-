import gradio as gr
import os
from src.audio_processor import AudioProcessor
from src.transcriber import Transcriber
from src.dubbing_engine import DubbingEngine
from src.utils import clear_gpu_memory

def process_video(video_file, hf_token, translate_to_english=False):
    if video_file is None:
        return None, None, None, None
    
    print(f"Processing video: {video_file} (Type: {type(video_file)})")
    task_mode = "translate" if translate_to_english else "transcribe"
    status_log = f"Starting Process... (Mode: {task_mode})\n"
    yield status_log, None, None, None
    
    output_dir = "output_temp"
    processor = AudioProcessor(output_dir=output_dir)
    # Pass the user-provided token to Transcriber
    transcriber = Transcriber(device="cuda", compute_type="int8", hf_token=hf_token) 
    dubbing = DubbingEngine() # Add API key here if needed
    
    # 1. Extract Audio
    status_log += "Step 1: Extracting Audio...\n"
    yield status_log, None, None, None
    try:
        audio_path = processor.extract_audio(video_file)
    except Exception as e:
        status_log += f"Error: {str(e)}\n"
        yield status_log, None, None, None
        return

    # 2. Separate Audio
    status_log += "Step 2: Separating Vocals (Kim Vocal 2)....\n"
    yield status_log, None, None, None
    try:
        vocals_path, inst_path = processor.separate_audio(audio_path)
    except Exception as e:
        status_log += f"Error Separation: {str(e)}\n"
        yield status_log, None, None, None
        return
        
    # 3. Transcribe & Diarize
    status_log += f"Step 3: Transcribing Vocals (WhisperX, Task={task_mode})...\n"
    yield status_log, vocals_path, inst_path, None
    try:
        # We use the VOCALS track for transcription
        if not hf_token:
            status_log += "Warning: No HF Token provided. Skipping Diarization (Speaker ID).\n"
        
        segments = transcriber.transcribe_and_diarize(vocals_path, task=task_mode)
        status_log += f"Transcription Complete. Found {len(segments)} segments.\n"
        
        # Preview first few segments
        for i, seg in enumerate(segments[:3]):
             spk = seg.get('speaker', 'Unknown')
             txt = seg.get('text', '')[:50]
             status_log += f"Sample: [{spk}] {txt}...\n"

    except Exception as e:
        status_log += f"Error Transcription: {str(e)}\n"
        yield status_log, vocals_path, inst_path, None
        return

    # 4. Generate Dubs & Mix
    status_log += "Step 4: Generating Dub and Mixing (Edge-TTS)...\n"
    yield status_log, vocals_path, inst_path, None
    
    try:
        final_output_path = os.path.join(output_dir, "final_dubbed_audio.mp3")
        dubbing.stitch_and_mix(segments, inst_path, final_output_path)
        status_log += f"Process Complete! File saved: {final_output_path}\n"
    except Exception as e:
        status_log += f"Error Mixing: {str(e)}\n"
        yield status_log, vocals_path, inst_path, None
        return

    status_log += "Pipeline Complete! Audio Dubbed & Mixed.\n"
    yield status_log, vocals_path, inst_path, final_output_path

def create_ui():
    with gr.Blocks(title="AI Movie Dub Maker") as app:
        gr.Markdown("# AI Movie Dub Maker")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Source Input")
                video_input = gr.Video(label="Upload Video (For Previews)", sources=["upload"])
                file_input = gr.File(label="Alternative: Upload Video/Audio File (Best for MKV/Unsupported Formats)", file_types=["video", "audio", ".mkv", ".mp4", ".avi"], type="filepath")
                
                # Add Token Input
                token_input = gr.Textbox(label="Hugging Face Token (Required for Speaker ID)", placeholder="hf_...", type="password")
                
                translate_chk = gr.Checkbox(label="Translate to English (WhisperX Built-in)", value=True)
                
                process_btn = gr.Button("Start Full Pipeline", variant="primary")
            
            with gr.Column():
                status_output = gr.Textbox(label="Status", lines=15)
                vocals_output = gr.Audio(label="Extracted Vocals (Reference)", type="filepath")
                inst_output = gr.Audio(label="Extracted Background", type="filepath")
                final_output = gr.Audio(label="Final AI Dubbed Audio", type="filepath")
        
        def router(video, file, token, translate):
            # Prefer file input if provided, otherwise video input
            target_input = file if file else video
            yield from process_video(target_input, token, translate)

        process_btn.click(
            fn=router,
            inputs=[video_input, file_input, token_input, translate_chk],
            outputs=[status_output, vocals_output, inst_output, final_output]
        )
        
    return app

if __name__ == "__main__": 
    app = create_ui()
    app.launch(share=True)
