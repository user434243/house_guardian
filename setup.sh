#!/usr/bin/env bash
# setup.sh — Run once on a fresh Raspberry Pi 5
# Usage: chmod +x setup.sh && ./setup.sh

set -e

echo "=========================================="
echo "  House Guardian — Setup"
echo "=========================================="

# 1. System packages
sudo apt-get update
sudo apt-get install -y \
    python3-pip python3-venv \
    libopencv-dev libatlas-base-dev \
    libjpeg-dev libpng-dev \
    ffmpeg \
    git curl

# 2. Python deps
pip3 install --break-system-packages -r requirements.txt

# 3. Install and start Ollama
echo "Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh
sleep 3

echo "Pulling LLM model (llama3.2:3b — fast, fits in 4GB RAM)..."
ollama pull llama3.2:3b

echo ""
echo "Optional: pull a larger model for better reasoning:"
echo "  ollama pull llama3.1:8b     (needs 8GB RAM)"
echo "  ollama pull phi3:mini       (very fast, 3.8B)"
echo ""

# 4. Create data directories
mkdir -p data/{clips,snapshots,logs,known_faces}

# 5. Config reminder
if [ ! -f config/config.yaml.local ]; then
    echo "⚠️  Don't forget to fill in config/config.yaml with your email credentials!"
fi

# 6. Google Drive setup reminder
echo ""
echo "=========================================="
echo "  NEXT STEPS"
echo "=========================================="
echo "1. Fill in config/config.yaml with your SMTP credentials"
echo "2. Set up Google Drive API:"
echo "   - Go to console.cloud.google.com"
echo "   - Create a service account → download JSON → save as config/drive_credentials.json"
echo "   - Share your HouseGuardian Drive folder with the service account email"
echo "3. Add known faces: put folders in data/known_faces/<person_name>/ with 3-5 photos"
echo "4. Run: python3 main.py"
echo ""
echo "✅ Setup complete!"
