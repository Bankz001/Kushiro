# Collision Probability Prediction System
# collision_prediction.py - Main script

import argparse
import os
import sys

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from collision_predictor import CollisionProbabilityPredictor
import utils
import visualization

def process_video(video_path=None, output_path=None):
    """
    Process a video file or webcam feed and predict collision probabilities.
    
    Args:
        video_path: Path to video file, or None for webcam
        output_path: Path to save output video, or None to not save
    """
    import cv2
    
    # Initialize video capture
    if video_path is None:
        cap = cv2.VideoCapture(0)  # Use webcam
    else:
        cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer if output path is provided
    out = None
    if output_path is not None:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Initialize collision probability predictor
    predictor = CollisionProbabilityPredictor()
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Predict collision probability
            collision_prob, annotated_frame, object_risks = predictor.predict_collision(frame)
            
            # Display frame
            cv2.imshow('Collision Probability Prediction', annotated_frame)
            
            # Write frame to output video if applicable
            if out is not None:
                out.write(annotated_frame)
            
            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Clean up
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()


def process_image(image_path, output_path=None):
    """
    Process a single image and predict collision probability.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save output image, or None to not save
    
    Returns:
        collision_prob: Calculated collision probability
        annotated_image: Image with visualization
    """
    import cv2
    
    # Read image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image from {image_path}")
        return None, None
    
    # Initialize collision probability predictor
    predictor = CollisionProbabilityPredictor()
    
    # Predict collision probability
    collision_prob, annotated_frame, object_risks = predictor.predict_collision(frame)
    
    # Save output image if requested
    if output_path is not None:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, annotated_frame)
    
    # Display result
    print(f"Collision probability: {collision_prob:.2f}")
    for risk, label, *_ in object_risks:
        print(f"  - {label}: risk score {risk:.2f}")
    
    return collision_prob, annotated_frame


def main():
    """Main function to parse arguments and call the appropriate processing function."""
    parser = argparse.ArgumentParser(description='Collision Probability Prediction System')
    parser.add_argument('--input', type=str, help='Path to input image or video file (default: use webcam)')
    parser.add_argument('--output', type=str, help='Path to save output')
    parser.add_argument('--model', type=str, default='yolov8n.pt', 
                        help='Path to YOLO model file (default: yolov8n.pt)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold for object detection (default: 0.25)')
    parser.add_argument('--mode', type=str, default='auto', choices=['image', 'video', 'auto'],
                        help='Processing mode (default: auto-detect from file extension)')
    args = parser.parse_args()
    
    # Download model if not exists
    if not os.path.exists(args.model) and not os.path.exists(f"models/{args.model}"):
        try:
            model_path = utils.download_model(args.model)
            args.model = model_path
        except Exception as e:
            print(f"Error downloading model: {str(e)}")
            print("Using default model instead.")
    
    # Auto-detect mode from file extension if not specified
    if args.mode == 'auto' and args.input is not None:
        if args.input.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            args.mode = 'image'
        else:
            args.mode = 'video'
    
    # Set default output paths if not provided
    if args.input is not None and args.output is None:
        if args.mode == 'image':
            input_name = os.path.splitext(os.path.basename(args.input))[0]
            args.output = f"output/images/{input_name}_output.jpg"
        elif args.mode == 'video':
            input_name = os.path.splitext(os.path.basename(args.input))[0]
            args.output = f"output/videos/{input_name}_output.mp4"
    
    # Process according to mode
    if args.mode == 'image':
        if args.input is None:
            print("Error: Input image path must be provided for image mode")
        else:
            collision_prob, annotated_image = process_image(args.input, args.output)
            if annotated_image is not None:
                cv2.imshow('Collision Probability Prediction', annotated_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    else:  # video mode
        process_video(args.input, args.output)


if __name__ == "__main__":
    main()