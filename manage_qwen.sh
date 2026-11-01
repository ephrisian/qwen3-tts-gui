#!/bin/bash

ENV_NAME="qwen3-tts"
SCRIPT_NAME="launch.py"

# Resolve the directory this script lives in so the python script is found
# regardless of the current working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$SCRIPT_NAME"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH."
    echo "Please install Miniconda or Anaconda first."
    exit 1
fi

# Initialize conda for bash (handles non-interactive shells)
eval "$(conda shell.bash hook)"

# Get the conda environments directory (may be on a different drive)
CONDA_ENVS_DIR=$(conda info --base 2>/dev/null)/envs
if [ ! -d "$CONDA_ENVS_DIR" ]; then
    # Fallback: check CONDA_ENVS_PATH or common locations
    CONDA_ENVS_DIR="${CONDA_ENVS_PATH:-$HOME/.conda/envs}"
fi

# Check if the environment already exists
env_exists() {
    conda env list | grep -q "^${ENV_NAME}[[:space:]]"
}

# Checkpoint file to track completed install steps (supports repair/resume)
CHECKPOINT_FILE="$CONDA_ENVS_DIR/$ENV_NAME/.install_checkpoints"

checkpoint_done() {
    [ -f "$CHECKPOINT_FILE" ] && grep -qx "$1" "$CHECKPOINT_FILE"
}

mark_checkpoint() {
    echo "$1" >> "$CHECKPOINT_FILE"
}

# Set TMPDIR to a location inside the project folder so nothing is stored
# outside of it. This avoids "Invalid cross-device link" errors when pip
# builds wheels across mount points and keeps all temp files in-project.
ENV_TMPDIR="$SCRIPT_DIR/.tmp"
mkdir -p "$ENV_TMPDIR"
export TMPDIR="$ENV_TMPDIR"

# Keep Hugging Face cache inside the project folder so no model files are
# stored on the system outside of this project.
export HF_HOME="${HF_HOME:-$SCRIPT_DIR/.hf_cache}"

# Cleanup: remove temporary files, pip cache, and reset checkpoints from the
# conda environment so it can be reinstalled cleanly.
cleanup_env() {
    if ! env_exists; then
        echo "Conda environment '$ENV_NAME' does not exist. Nothing to clean up."
        exit 0
    fi

    echo "Activating environment for cleanup..."
    conda activate $ENV_NAME
    if [ $? -ne 0 ]; then
        echo "Error: Failed to activate conda environment '$ENV_NAME'."
        exit 1
    fi

    echo "Cleaning pip cache..."
    pip cache purge 2>/dev/null || true

    echo "Removing temporary files..."
    rm -rf "$ENV_TMPDIR"/* 2>/dev/null || true

    echo "Resetting install checkpoints..."
    rm -f "$CHECKPOINT_FILE" 2>/dev/null || true

    echo "Cleanup complete. Run './manage_qwen.sh install' to reinstall from scratch."
}

# Purge: completely remove the conda environment and all project-generated
# files so nothing is left behind on the system.
purge_all() {

    # Remove project temp directory
    PROJECT_TMPDIR="$SCRIPT_DIR/.tmp"
    if [ -d "$PROJECT_TMPDIR" ]; then
        echo "Removing project temp directory: $PROJECT_TMPDIR..."
        rm -rf "$PROJECT_TMPDIR"
        echo "Project temp directory removed."
    fi

    # Remove conda environment if it exists
    if env_exists; then
        echo "Removing Conda environment: $ENV_NAME..."
        conda env remove -n $ENV_NAME -y
        if [ $? -ne 0 ]; then
            echo "Warning: Failed to remove conda environment '$ENV_NAME'."
        else
            echo "Conda environment removed."
        fi
    else
        echo "Conda environment '$ENV_NAME' does not exist. Skipping."
    fi

    # Remove Hugging Face model cache from project folder
    HF_CACHE="$SCRIPT_DIR/.hf_cache"
    if [ -d "$HF_CACHE" ]; then
        echo "Removing Hugging Face cache: $HF_CACHE..."
        rm -rf "$HF_CACHE"
        echo "Hugging Face cache removed."
    fi

    # Remove output directory
    OUTPUT_DIR="$SCRIPT_DIR/output"
    if [ -d "$OUTPUT_DIR" ]; then
        echo "Removing output directory: $OUTPUT_DIR..."
        rm -rf "$OUTPUT_DIR"
        echo "Output directory removed."
    fi

    # Remove any leftover conda envs directory entry
    ENV_DIR="$CONDA_ENVS_DIR/$ENV_NAME"
    if [ -d "$ENV_DIR" ]; then
        echo "Removing leftover environment directory: $ENV_DIR..."
        rm -rf "$ENV_DIR"
        echo "Environment directory removed."
    fi

    echo ""
    echo "Purge complete. All Qwen3-TTS files have been removed from the system."
}

case "$1" in
    install)
        if env_exists; then
            echo "Conda environment '$ENV_NAME' already exists."
        else
            echo "Creating Conda environment: $ENV_NAME..."
            conda create -n $ENV_NAME python=3.10 -y
            if [ $? -ne 0 ]; then
                echo "Error: Failed to create conda environment."
                exit 1
            fi
            mark_checkpoint "env_created"
        fi

        echo "Activating environment..."
        conda activate $ENV_NAME
        if [ $? -ne 0 ]; then
            echo "Error: Failed to activate conda environment '$ENV_NAME'."
            echo "Note: Conda environments may be on a different drive. Ensure '$CONDA_ENVS_DIR' is accessible."
            exit 1
        fi

        # Checkpoint: torch stack
        if checkpoint_done "torch_installed"; then
            echo "[skip] torch/torchvision/torchaudio already installed."
        else
            echo "Installing torch/torchvision/torchaudio (cu121)..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
            if [ $? -ne 0 ]; then
                echo "Error: Failed to install torch stack."
                exit 1
            fi
            mark_checkpoint "torch_installed"
        fi

        # Checkpoint: qwen-tts
        if checkpoint_done "qwen_tts_installed"; then
            echo "[skip] qwen-tts already installed."
        else
            echo "Installing qwen-tts..."
            pip install -U qwen-tts
            if [ $? -ne 0 ]; then
                echo "Error: Failed to install qwen-tts."
                exit 1
            fi
            mark_checkpoint "qwen_tts_installed"
        fi

        # Checkpoint: flash-attn
        if checkpoint_done "flash_attn_installed"; then
            echo "[skip] flash-attn already installed."
        else
            echo "Installing flash-attn..."
            pip install -U flash-attn --no-build-isolation
            if [ $? -ne 0 ]; then
                echo "Error: Failed to install flash-attn."
                exit 1
            fi
            mark_checkpoint "flash_attn_installed"
        fi

        # Checkpoint: pygame (for audio playback in GUI)
        if checkpoint_done "pygame_installed"; then
            echo "[skip] pygame already installed."
        else
            echo "Installing pygame..."
            pip install -U pygame
            if [ $? -ne 0 ]; then
                echo "Error: Failed to install pygame."
                exit 1
            fi
            mark_checkpoint "pygame_installed"
        fi

        # Checkpoint: PySide6 (Qt GUI)
        if checkpoint_done "pyside6_installed"; then
            echo "[skip] PySide6 already installed."
        else
            echo "Installing PySide6..."
            pip install -U PySide6-Essentials
            if [ $? -ne 0 ]; then
                echo "Error: Failed to install PySide6."
                exit 1
            fi
            mark_checkpoint "pyside6_installed"
        fi

        echo "Installation complete."
        ;;
    run)
        if ! env_exists; then
            echo "Error: Conda environment '$ENV_NAME' does not exist."
            echo "Run './manage_qwen.sh install' first."
            exit 1
        fi

        echo "Starting Qwen3-TTS..."
        conda activate $ENV_NAME
        if [ $? -ne 0 ]; then
            echo "Error: Failed to activate conda environment '$ENV_NAME'."
            echo "Note: Conda environments may be on a different drive. Ensure '$CONDA_ENVS_DIR' is accessible."
            exit 1
        fi
        if [ ! -f "$SCRIPT_PATH" ]; then
            echo "Error: Python script not found at '$SCRIPT_PATH'."
            echo "Make sure launch.py sits next to manage_qwen.sh."
            exit 1
        fi
        cd "$SCRIPT_DIR" && python "$SCRIPT_PATH" "${@:2}"
        ;;
    remove)
        if ! env_exists; then
            echo "Conda environment '$ENV_NAME' does not exist. Nothing to remove."
            exit 0
        fi

        echo "Removing Conda environment: $ENV_NAME..."
        conda env remove -n $ENV_NAME -y
        echo "Environment removed."
        ;;
    cleanup)
        cleanup_env
        ;;
    purge)
        purge_all
        ;;
    *)
        echo "Usage: ./manage_qwen.sh {install|run|remove|cleanup|purge}"
        exit 1
esac