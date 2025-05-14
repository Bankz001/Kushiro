# Create empty __init__.py files
touch src/__init__.py
touch tests/__init__.py

# Create directory structure
mkdir -p models
mkdir -p data/images
mkdir -p data/videos
mkdir -p output/images
mkdir -p output/videos

# Create empty placeholder files
touch models/.gitkeep
touch data/images/.gitkeep
touch data/videos/.gitkeep
touch output/images/.gitkeep
touch output/videos/.gitkeep

# Install requirements
pip install -r requirements.txt

# Download YOLO model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

echo "Setup complete! Project structure created and dependencies installed."