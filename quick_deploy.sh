#!/bin/bash
# Quick one-command deployment to mimoria server

SERVER="mimoria@100.126.32.26"

echo "🚀 Quick Deploy to Mimoria Server"
echo ""

# Copy and run deployment
scp deploy_server.sh $SERVER:~/ 2>/dev/null || echo "Note: deploy_server.sh not found locally, will clone from git"

ssh $SERVER bash << 'ENDSSH'
set -e

echo "=== Starting Deployment ==="

# Install system dependencies if needed
if ! command -v python3.14 &> /dev/null; then
    echo "Installing Python 3.14..."
    sudo apt update
    sudo apt install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install -y python3.14 python3.14-venv python3-pip git curl nginx
fi

# Clone or update repository
APP_DIR=$HOME/llm-analysis-quiz
if [ -d "$APP_DIR" ]; then
    echo "Updating existing installation..."
    cd "$APP_DIR"
    git pull || echo "Git pull failed, using existing code"
else
    echo "Cloning repository..."
    git clone https://github.com/manavkdubey/llm-analysis-quiz.git "$APP_DIR"
    cd "$APP_DIR"
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3.14 -m venv venv
fi

source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install Playwright browsers
echo "Installing Playwright browsers..."
playwright install chromium
playwright install-deps chromium || true

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOF
EMAIL=23f2001947@ds.study.iitm.ac.in
SECRET=manavkumardubey
OPENROUTER_API_KEY=
HOST=0.0.0.0
PORT=8000
EOF
    echo "⚠️  Please edit .env and add your OPENROUTER_API_KEY"
fi

# Create systemd service
echo "Setting up systemd service..."
CURRENT_USER=$(whoami)
sudo tee /etc/systemd/system/llm-quiz.service > /dev/null << EOF
[Unit]
Description=LLM Analysis Quiz Solver
After=network.target

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/venv/bin"
ExecStart=$APP_DIR/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable llm-quiz.service

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "⚠️  IMPORTANT: Edit .env file and add your OPENROUTER_API_KEY:"
echo "   nano $APP_DIR/.env"
echo ""
echo "Then start the service:"
echo "   sudo systemctl start llm-quiz"
echo "   sudo systemctl status llm-quiz"
echo ""
echo "Your API will be at: http://65.109.236.39:8000/quiz"
echo ""
echo "For HTTPS, run: cd $APP_DIR && ./setup_https.sh"
ENDSSH

echo ""
echo "✅ Deployment script executed on server"
echo "SSH to server to complete setup: ssh $SERVER"

