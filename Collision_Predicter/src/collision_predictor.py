import cv2
import numpy as np
import torch
import time
import math
from ultralytics import YOLO

class CollisionProbabilityPredictor:
    """
    A system that predicts the probability of collision based on forward-facing camera input.
    Uses YOLO for object detection and estimates collision probability based on object position,
    size, and movement patterns.
    """
    
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.25, collision_threshold=0.7):
        """
        Initialize the collision probability predictor.
        
        Args:
            model_path: Path to the YOLO model (default: 'yolov8n.pt')
            confidence_threshold: Confidence threshold for object detection
            collision_threshold: Probability threshold to consider an object as collision risk
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.collision_threshold = collision_threshold
        self.previous_detections = {}  # Store previous detections for tracking
        self.previous_time = time.time()
        
        # Track classes that are considered collision risks (based on COCO dataset)
        self.collision_risk_classes = {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus",
            6: "train", 7: "truck", 9: "traffic light", 10: "fire hydrant",
            11: "stop sign", 13: "bench", 14: "bird", 15: "cat", 16: "dog", 
            17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 22: "bear"
        }
        
        # Weight factors for different aspects of collision risk
        self.weight_size = 0.3        # How much of the frame the object takes up
        self.weight_center = 0.4      # How close the object is to the center of the frame
        self.weight_growth = 0.3      # How fast the object is growing (approaching)
    
    def _calculate_object_risk(self, detection, frame_shape, prev_detection=None):
        """
        Calculate the collision risk for a single detected object.
        
        Args:
            detection: The detection data from YOLO (x1, y1, x2, y2, confidence, class_id)
            frame_shape: Shape of the input frame (height, width)
            prev_detection: Previous detection of the same object, if available
            
        Returns:
            Float value representing the collision probability (0.0 to 1.0)
        """
        # Unpack detection data
        x1, y1, x2, y2 = detection[:4]
        class_id = int(detection[5])
        
        if class_id not in self.collision_risk_classes:
            return 0.0  # Not a collision risk class
        
        # Calculate relative size (area covered)
        frame_height, frame_width = frame_shape[:2]
        obj_width = x2 - x1
        obj_height = y2 - y1
        relative_size = (obj_width * obj_height) / (frame_width * frame_height)
        size_score = min(relative_size * 10, 1.0)  # Normalize to 0-1 range
        
        # Calculate center proximity (how close to center of frame)
        obj_center_x = (x1 + x2) / 2
        obj_center_y = (y1 + y2) / 2
        frame_center_x = frame_width / 2
        frame_center_y = frame_height / 2
        
        # Normalized distance from center (0 = center, 1 = edge/corner)
        distance_from_center = math.sqrt(
            ((obj_center_x - frame_center_x) / frame_width) ** 2 + 
            ((obj_center_y - frame_center_y) / frame_height) ** 2
        ) * 2  # Scale to make max distance to corner = 1.0
        
        center_score = 1.0 - min(distance_from_center, 1.0)  # Closer to center = higher risk
        
        # Calculate size growth rate if previous detection exists
        growth_score = 0.0
        if prev_detection is not None:
            prev_x1, prev_y1, prev_x2, prev_y2 = prev_detection[:4]
            prev_width = prev_x2 - prev_x1
            prev_height = prev_y2 - prev_y1
            prev_area = prev_width * prev_height
            current_area = obj_width * obj_height
            
            if prev_area > 0:
                # Calculate growth ratio
                growth_ratio = (current_area / prev_area) - 1.0
                growth_score = min(max(growth_ratio, 0.0), 1.0)
        
        # Calculate weighted risk score
        risk_score = (
            self.weight_size * size_score +
            self.weight_center * center_score +
            self.weight_growth * growth_score
        )
        
        return min(risk_score, 1.0)  # Ensure it's between 0 and 1
    
    def predict_collision(self, frame):
        """
        Process a frame and predict collision probability.
        
        Args:
            frame: Input image frame from the camera
            
        Returns:
            collision_prob: Overall collision probability
            annotated_frame: Frame with visualized detection and risk assessment
            detected_objects: List of detected objects with their risk scores
        """
        from visualization import visualize_detections
        
        # Run object detection
        results = self.model(frame, conf=self.confidence_threshold)[0]
        detections = results.boxes.data.cpu().numpy()
        
        # Current time for calculating time between frames
        current_time = time.time()
        time_delta = current_time - self.previous_time
        self.previous_time = current_time
        
        # Track objects and calculate individual risks
        object_risks = []
        current_detections = {}
        
        # Process each detection
        for detection in detections:
            # Get detection data
            x1, y1, x2, y2 = detection[:4].astype(int)
            confidence = detection[4]
            class_id = int(detection[5])
            
            # Skip if not in our risk classes
            if class_id not in self.collision_risk_classes:
                continue
                
            # Generate a simple object ID based on position (for tracking)
            obj_id = f"{class_id}_{x1}_{y1}"
            current_detections[obj_id] = detection
            
            # Find the closest previous detection of this object (simple tracking)
            prev_detection = None
            if len(self.previous_detections) > 0:
                # Look for previous detections of the same class
                class_detections = [det for k, det in self.previous_detections.items() if k.startswith(f"{class_id}_")]
                if class_detections:
                    # Find the closest detection by center point
                    obj_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                    min_dist = float('inf')
                    for det in class_detections:
                        prev_x1, prev_y1, prev_x2, prev_y2 = det[:4].astype(int)
                        prev_center = np.array([(prev_x1 + prev_x2) / 2, (prev_y1 + prev_y2) / 2])
                        dist = np.linalg.norm(obj_center - prev_center)
                        if dist < min_dist:
                            min_dist = dist
                            prev_detection = det
            
            # Calculate risk for this object
            risk_score = self._calculate_object_risk(detection, frame.shape, prev_detection)
            label = self.collision_risk_classes[class_id]
            object_risks.append((risk_score, label, x1, y1, x2, y2))
        
        # Update previous detections
        self.previous_detections = current_detections
        
        # Calculate overall collision probability
        if object_risks:
            # Use the maximum risk as baseline
            max_risk = max([risk for risk, *_ in object_risks])
            
            # Add additional risk based on number of risky objects
            high_risk_count = sum(1 for risk, *_ in object_risks if risk > 0.5)
            collision_prob = min(max_risk + (high_risk_count * 0.1), 1.0)
        else:
            collision_prob = 0.0
        
        # Visualize the results
        annotated_frame = visualize_detections(frame, object_risks, collision_prob, self.collision_threshold)
        
        return collision_prob, annotated_frame, object_risks