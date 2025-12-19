import os
import time
import subprocess
import requests
import runpod
import sys

# Configuration
VLLM_PORT = 8000
MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "magnum-v4-72b-awq")
BASE_URL = f"http://localhost:{VLLM_PORT}/v1"

def wait_for_port(port, timeout=300):
    """Wait for the vLLM server to start listening."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://localhost:{port}/health")
            if response.status_code == 200:
                print(f"Service running on port {port}")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(5)
        print(f"Waiting for vLLM to start... ({int(time.time() - start_time)}s)")
    return False

def start_vllm():
    """Start the vLLM API server as a subprocess."""
    # We use the environment variables already set in Dockerfile for Model path
    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "/model/magnum-v4-72b-awq",
        "--served-model-name", MODEL_NAME,
        "--port", str(VLLM_PORT),
        "--trust-remote-code",
        "--max-model-len", "16384", # Safety limit for A100 80GB just in case, or remove if confident
        "--gpu-memory-utilization", "0.95"
    ]
    
    print(f"Starting vLLM: {' '.join(cmd)}")
    process = subprocess.Popen(cmd)
    return process

# --- Start Server ---
vllm_process = start_vllm()

if not wait_for_port(VLLM_PORT):
    print("Failed to start vLLM server!")
    vllm_process.terminate()
    sys.exit(1)

print("vLLM Server Ready! Starting RunPod Handler...")

# --- RunPod Handler ---
def handler(job):
    """Forward RunPod job input to local vLLM OpenAI API."""
    job_input = job.get("input", {})
    
    # Check if this is a standard OpenAI Chat Completion payload
    # RunPod input usually looks like: { "method": "POST", "body": { ... } } or just the body
    # We assume the input IS the body (openai payload)
    
    endpoint = f"{BASE_URL}/chat/completions"
    
    try:
        print(f"Processing job: {job.get('id')}")
        response = requests.post(
            endpoint,
            headers={"Content-Type": "application/json"},
            json=job_input,
            timeout=300
        )
        
        if response.status_code != 200:
            return {"error": f"vLLM Error {response.status_code}: {response.text}"}
            
        return response.json()
        
    except Exception as e:
        return {"error": str(e)}

# Start the worker loop
runpod.serverless.start({"handler": handler})
