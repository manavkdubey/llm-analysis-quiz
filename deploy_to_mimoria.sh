#!/bin/bash
# Complete deployment script for mimoria server
# This script deploys the LLM Analysis Quiz Solver to the server

set -e

SERVER="mimoria@100.126.32.26"
APP_DIR="llm-analysis-quiz"
DOMAIN=""

echo "=== LLM Analysis Quiz - Complete Deployment ==="
echo ""

# Get domain name for HTTPS
read -p "Enter your domain name for HTTPS (or press Enter to skip HTTPS): " DOMAIN

# Step 1: Copy deployment files to server
echo "📦 Step 1: Copying deployment files to server..."
scp deploy_server.sh setup_https.sh nginx_setup.sh $SERVER:~/

# Step 2: Run deployment on server
echo ""
echo "🚀 Step 2: Running deployment on server..."
ssh $SERVER << 'ENDSSH'
set -e

# Make scripts executable
chmod +x ~/deploy_server.sh ~/setup_https.sh ~/nginx_setup.sh

# Check if app directory exists
if [ -d ~/llm-analysis-quiz ]; then
    echo "App directory exists, updating..."
    cd ~/llm-analysis-quiz
    git pull || echo "Git pull failed, continuing..."
else
    echo "Cloning repository..."
    cd ~
    git clone https://github.com/manavkdubey/llm-analysis-quiz.git
    cd llm-analysis-quiz
fi

# Run deployment script
echo "Running deployment script..."
~/deploy_server.sh

echo "✅ Deployment script completed on server"
ENDSSH

# Step 3: Setup HTTPS if domain provided
if [ ! -z "$DOMAIN" ]; then
    echo ""
    echo "🔒 Step 3: Setting up HTTPS with domain: $DOMAIN"
    echo "⚠️  Make sure $DOMAIN points to 65.109.236.39"
    read -p "Press Enter when DNS is configured, or Ctrl+C to cancel..."
    
    ssh $SERVER << ENDSSH
set -e
cd ~/llm-analysis-quiz
./setup_https.sh << EOF
$DOMAIN
EOF
ENDSSH
    
    echo ""
    echo "✅ HTTPS setup complete!"
    echo "Your API endpoint: https://$DOMAIN/quiz"
else
    echo ""
    echo "⚠️  HTTPS skipped. Your API is available at:"
    echo "   http://65.109.236.39:8000/quiz"
    echo ""
    echo "To setup HTTPS later, run on server:"
    echo "   cd ~/llm-analysis-quiz && ./setup_https.sh"
fi

# Step 4: Final instructions
echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Next steps:"
echo "1. SSH to server: ssh $SERVER"
echo "2. Edit .env file: nano ~/$APP_DIR/.env"
echo "3. Add your OPENROUTER_API_KEY"
echo "4. Start service: sudo systemctl start llm-quiz"
echo "5. Check status: sudo systemctl status llm-quiz"
echo "6. View logs: sudo journalctl -u llm-quiz -f"
echo ""

if [ ! -z "$DOMAIN" ]; then
    echo "✅ Your HTTPS endpoint: https://$DOMAIN/quiz"
else
    echo "✅ Your HTTP endpoint: http://65.109.236.39:8000/quiz"
fi

