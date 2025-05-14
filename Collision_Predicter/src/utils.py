import os
import requests
import torch
import cv2
import numpy as np

def download_model(model_name):
    """
    Download a YOLO model if not already present.
    
    Args:
        model_name: Name of the model to download
        
    Returns:
        Path to the downloaded model
    """
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, model_name)
    
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}")
        return model_path
    
    # Simple models are available via the ultralytics pip package
    try:
        from ultralytics import YOLO
        model = YOLO(model_name)
        print(f"Model loaded from ultralytics package: {model_name}")
        return model_name
    except Exception as e:
        print(f"Could not load model from ultralytics package: {str(e)}")
        print("Trying to download from YOLO GitHub...")
    
    # For custom models, try downloading from YOLO GitHub releases
    try:
        url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_name}"
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(model_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"Downloaded model to {model_path}")
            return model_path
        else:
            print(f"Failed to download model: HTTP status code {response.status_code}")
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
    
    return model_name  # Return the model name as a fallback


def calculate_depth_map(frame):
    """
    Calculate a simple depth map based on gradients in the image.
    This is a naive approach and not as accurate as stereo vision or dedicated depth sensors.
    
    Args:
        frame: Input frame
        
    Returns:
        depth_map: Simple depth map
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Calculate gradients using Sobel
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize to 0-255
    gradient_mag = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Invert (edges are typically at depth discontinuities)
    depth_map = 255 - gradient_mag
    
    # Apply colormap for visualization
    depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
    
    return depth_colored


def estimate_distance(obj_height_pixels, known_height_meters, focal_length_pixels):
    """
    Estimate distance to an object using its apparent height in the image.
    
    Args:
        obj_height_pixels: Height of the object in pixels
        known_height_meters: Actual height of the object in meters
        focal_length_pixels: Focal length of the camera in pixels
        
    Returns:
        distance_meters: Estimated distance to the object in meters
    """
    # Distance = (Known Height * Focal Length) / Apparent Height
    distance_meters = (known_height_meters * focal_length_pixels) / obj_height_pixels
    return distance_meters


def estimate_time_to_collision(distance_meters, relative_velocity_mps):
    """
    Estimate time to collision based on distance and relative velocity.
    
    Args:
        distance_meters: Distance to the object in meters
        relative_velocity_mps: Relative velocity in meters per second
        
    Returns:
        ttc_seconds: Time to collision in seconds, or float('inf') if no collision expected
    """
    if relative_velocity_mps <= 0:
        return float('inf')  # No collision if not approaching
    
    ttc_seconds = distance_meters / relative_velocity_mps
    return ttc_seconds


def filter_detection_by_region(detection, frame_shape, region='center', center_ratio=0.5):
    """
    Filter a detection based on its position in the frame.
    
    Args:
        detection: Detection (x1, y1, x2, y2)
        frame_shape: Shape of the frame (height, width)
        region: Which region to filter for ('center', 'left', 'right', 'top', 'bottom')
        center_ratio: Size of the central region as a ratio of frame size
        
    Returns:
        in_region: Boolean indicating if the detection is in the specified region
    """
    x1, y1, x2, y2 = detection[:4]
    h, w = frame_shape[:2]
    
    # Calculate center of the detection
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Define regions
    center_left = w * (0.5 - center_ratio/2)
    center_right = w * (0.5 + center_ratio/2)
    center_top = h * (0.5 - center_ratio/2)
    center_bottom = h * (0.5 + center_ratio/2)
    
    if region == 'center':
        return (center_left <= center_x <= center_right and 
                center_top <= center_y <= center_bottom)
    elif region == 'left':
        return center_x < center_left
    elif region == 'right':
        return center_x > center_right
    elif region == 'top':
        return center_y < center_top
    elif region == 'bottom':
        return center_y > center_bottom
    else:
        return True  # No filtering


def get_object_dimensions(class_id):
    """
    Get typical dimensions for common object classes.
    These are rough averages for estimation purposes.
    
    Args:
        class_id: Class ID from COCO dataset
        
    Returns:
        dimensions: Dictionary with height, width, and depth in meters
    """
    # Default dimensions
    default = {'height': 1.7, 'width': 0.5, 'depth': 0.5}
    
    # Dimensions for common classes (in meters)
    dimensions = {
        0: {'height': 1.7, 'width': 0.5, 'depth': 0.3},      # person
        1: {'height': 1.0, 'width': 1.7, 'depth': 0.5},      # bicycle
        2: {'height': 1.5, 'width': 1.8, 'depth': 4.5},      # car
        3: {'height': 1.2, 'width': 0.8, 'depth': 2.0},      # motorcycle
        5: {'height': 3.0, 'width': 2.5, 'depth': 12.0},     # bus
        6: {'height': 3.6, 'width': 3.0, 'depth': 20.0},     # train
        7: {'height': 3.0, 'width': 2.5, 'depth': 8.0},      # truck
        9: {'height': 0.9, 'width': 0.3, 'depth': 0.3},      # traffic light
        10: {'height': 0.9, 'width': 0.5, 'depth': 0.5},     # fire hydrant
        11: {'height': 0.9, 'width': 0.9, 'depth': 0.1},     # stop sign
    }
    
    return dimensions.get(class_id, default)


def create_empty_files():
    """Create empty files for the project structure."""
    # Define folders to create
    folders = [
        'models',
        'data/images',
        'data/videos',
        'output/images',
        'output/videos',
        'src',
        'tests'
    ]
    
    # Define empty files to create
    empty_files = [
        'models/.gitkeep',
        'data/images/.gitkeep',
        'data/videos/.gitkeep',
        'output/images/.gitkeep',
        'output/videos/.gitkeep',
        'src/__init__.py',
        'tests/__init__.py',
    ]
    
    # Get the base directory (assuming this script is in the project root)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Create folders
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f"Created folder: {folder_path}")
    
    # Create empty files
    for file in empty_files:
        file_path = os.path.join(base_dir, file)
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                pass  # Create empty file
            print(f"Created file: {file_path}")


if __name__ == "__main__":
    # If this script is run directly, create empty files for project structure
    create_empty_files()