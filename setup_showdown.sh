#!/bin/bash
echo "Setting up Pokemon Showdown..."

# Check for node
if ! command -v node &> /dev/null; then
    echo "Error: node is not in the PATH. Please ensure node is installed and accessible."
    exit 1
fi

# Clone if not exists (I already cloned it, but good to be safe)
if [ ! -d "pokemon-showdown" ]; then
    git clone https://github.com/smogon/pokemon-showdown.git
fi

cd pokemon-showdown

# Install dependencies
echo "Cleaning up old modules..."
rm -rf node_modules package-lock.json

echo "Installing dependencies (Fresh)..."
npm install
echo "Installing pg explicitly..."
npm install pg --save
npm install sqlite3 --save

# Verify pg
if [ ! -d "node_modules/pg" ]; then
    echo "ERROR: pg module still missing in node_modules!"
    ls -F node_modules/ | grep pg
    exit 1
fi

# Create config
cp config/config-example.js config/config.js

echo "Starting server on port 8000..."
echo "You can stop this server with Ctrl+C, but please keep it running for the agent to use."
node pokemon-showdown 8000
