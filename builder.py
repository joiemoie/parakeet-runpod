# builder.py
import nemo.collections.asr as nemo_asr

def download_model():
    print("Downloading Parakeet TDT 1.1b model...")
    # This triggers the download and caching of the model
    nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="nvidia/parakeet-tdt-1.1b")
    print("Model downloaded successfully.")

if __name__ == "__main__":
    download_model()
