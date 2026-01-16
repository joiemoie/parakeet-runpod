# Parakeet-TDT-1.1B RunPod Serverless

This is a [RunPod](https://www.runpod.io/) serverless worker for the NVIDIA Parakeet TDT 1.1B ASR model.

## Prerequisites

*   **RunPod Network Volume**: This template expects the model file to be present in a Network Volume.
    *   Mount path: `/workspace`
    *   Model file path: `/workspace/models/parakeet-tdt-1.1b.nemo`

You must download the `.nemo` file and place it in your network volume before starting the endpoint.

## Usage

### 1. Build and Deploy

Build the Docker image:

```bash
docker build -t <your-repo>/parakeet-runpod:latest .
docker push <your-repo>/parakeet-runpod:latest
```

Deploy on RunPod Serverless using this image. Ensure you attach your Network Volume to `/workspace`.

### 2. Inference

Send a POST request to your endpoint.

**Input Format:**

You can provide either `audio_url` (publicly accessible URL) or `audio_b64` (base64 encoded audio file).

```json
{
  "input": {
    "audio_url": "https://example.com/sample.mp3"
  }
}
```

OR

```json
{
  "input": {
    "audio_b64": "<base64_string>"
  }
}
```

**Output Format:**

```json
{
  "text": "transcribed text content",
  "model_path": "/workspace/models/parakeet-tdt-1.1b.nemo",
  "model": "parakeet-tdt-1.1b"
}
```