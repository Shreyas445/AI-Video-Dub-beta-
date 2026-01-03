from src import gpu_setup
gpu_setup.setup_gpu_environment()

from src.web_ui import create_ui

if __name__ == "__main__":
    print("Launching AI Movie Dub Maker...")
    app = create_ui()
    app.launch(inbrowser=True, share=True)
