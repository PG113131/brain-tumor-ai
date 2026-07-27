import os
import sys
import time
import socket
import signal
import webbrowser
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

FASTAPI_PORT = 8000
STREAMLIT_PORT = 8501


def print_banner():
    print("=" * 70)
    print("🧠 Brain Tumor Multi-Modal AI Diagnostic Suite")
    print("=" * 70)


def create_directories():
    directories = [
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "data" / "uploads",
        PROJECT_ROOT / "data" / "heatmaps",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def check_env():
    env_file = PROJECT_ROOT / ".env"

    if env_file.exists():
        print("✅ .env file found")
    else:
        print("⚠️  .env file not found.")
        print("Create one from .env.example before using the AI report feature.\n")


def download_model():
    print("\nChecking model weights...\n")

    subprocess.run(
        [sys.executable, "scripts/download_weights.py"],
        check=True,
    )


def wait_for_server(host="127.0.0.1", port=8000, timeout=60):
    start = time.time()

    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            time.sleep(1)

    return False


def start_backend():
    print("\nStarting FastAPI Backend...\n")

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(FASTAPI_PORT),
        ]
    )

    return process


def start_streamlit():
    print("\nStarting Streamlit Frontend...\n")

    process = subprocess.Popen(
        [
            "streamlit",
            "run",
            "frontend/app.py",
            "--server.port",
            str(STREAMLIT_PORT),
        ]
    )

    return process


def open_browser():
    print("\nOpening browser...\n")

    webbrowser.open(f"http://localhost:{STREAMLIT_PORT}")
    webbrowser.open(f"http://localhost:{FASTAPI_PORT}/docs")


def shutdown(backend, frontend):
    print("\nShutting down application...")

    backend.terminate()
    frontend.terminate()

    backend.wait()
    frontend.wait()

    print("Application stopped.")


def main():
    print_banner()

    create_directories()

    check_env()

    download_model()

    backend = start_backend()

    print("Waiting for backend...")

    if not wait_for_server(port=FASTAPI_PORT):
        print("Failed to start FastAPI.")
        backend.kill()
        return

    frontend = start_streamlit()

    time.sleep(4)

    open_browser()

    print("\nApplication is running.")
    print(f"FastAPI    : http://localhost:{FASTAPI_PORT}")
    print(f"Swagger    : http://localhost:{FASTAPI_PORT}/docs")
    print(f"Streamlit  : http://localhost:{STREAMLIT_PORT}")
    print("\nPress Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown(backend, frontend)


if __name__ == "__main__":
    main()