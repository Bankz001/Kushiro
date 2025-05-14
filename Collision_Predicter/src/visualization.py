import cv2

def visualize_detections(frame, object_risks, collision_prob, collision_threshold):
    """
    Visualize detected objects and collision risks on the frame.
    
    Args:
        frame: Input frame
        object_risks: List of tuples (risk_score, label, x1, y1, x2, y2)
        collision_prob: Overall collision probability
        collision_threshold: Threshold for collision warning
        
    Returns:
        annotated_frame: Frame with visualization
    """
    # Make a copy of the frame to draw on
    annotated_frame = frame.copy()
    
    # Draw each detected object with its risk score
    for risk_score, label, x1, y1, x2, y2 in object_risks:
        # Color based on risk score (green to red)
        color = get_risk_color(risk_score)
        
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label and risk score
        risk_text = f"{label}: {risk_score:.2f}"
        cv2.putText(annotated_frame, risk_text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add overall collision probability to the frame
    status_color = get_risk_color(collision_prob)
    status_text = f"Collision Probability: {collision_prob:.2f}"
    cv2.putText(annotated_frame, status_text, (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
    
    # Visual indicator for collision warning
    if collision_prob > collision_threshold:
        # Add warning border
        border_thickness = 20
        h, w = annotated_frame.shape[:2]
        cv2.rectangle(annotated_frame, (0, 0), (w, h), (0, 0, 255), border_thickness)
        
        # Add warning text
        cv2.putText(annotated_frame, "COLLISION WARNING!", (w//2 - 150, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    return annotated_frame


def get_risk_color(risk_score):
    """
    Get color based on risk score (green to red).
    
    Args:
        risk_score: Risk score from 0.0 to 1.0
        
    Returns:
        BGR color tuple
    """
    # Green (low risk) to Red (high risk)
    green = int(255 * (1 - risk_score))
    red = int(255 * risk_score)
    return (0, green, red)


def draw_trajectory_prediction(frame, object_risks, previous_positions, max_history=10):
    """
    Draw predicted trajectories for tracked objects.
    
    Args:
        frame: Input frame
        object_risks: List of current object risks
        previous_positions: Dictionary mapping object IDs to lists of previous positions
        max_history: Maximum number of positions to keep in history
        
    Returns:
        frame: Frame with trajectories drawn
    """
    # Make a copy of the frame
    annotated_frame = frame.copy()
    
    # For each object, draw its trajectory
    for obj_id, positions in previous_positions.items():
        if len(positions) < 2:
            continue
        
        # Draw trajectory line
        for i in range(1, len(positions)):
            prev_pos = positions[i-1]
            curr_pos = positions[i]
            
            # Color gets more intense as we reach the current position
            intensity = int(255 * (i / len(positions)))
            color = (0, intensity, 255 - intensity)
            
            cv2.line(annotated_frame, prev_pos, curr_pos, color, 2)
        
        # Draw predicted future position if we have enough history
        if len(positions) >= 3:
            # Simple linear extrapolation
            last_pos = positions[-1]
            prev_pos = positions[-2]
            
            # Calculate movement vector
            dx = last_pos[0] - prev_pos[0]
            dy = last_pos[1] - prev_pos[1]
            
            # Predict future position
            future_x = int(last_pos[0] + dx)
            future_y = int(last_pos[1] + dy)
            
            # Draw predicted position
            cv2.circle(annotated_frame, (future_x, future_y), 5, (0, 0, 255), -1)
            cv2.line(annotated_frame, last_pos, (future_x, future_y), (0, 0, 255), 2, cv2.LINE_DASHED)
    
    return annotated_frame


def create_bird_eye_view(object_risks, frame_width, frame_height, scale=100):
    """
    Create a bird's eye view visualization showing object positions from above.
    
    Args:
        object_risks: List of tuples (risk_score, label, x1, y1, x2, y2)
        frame_width: Width of the original frame
        frame_height: Height of the original frame
        scale: Scale factor for the bird's eye view
        
    Returns:
        bird_view: Bird's eye view image
    """
    # Create blank image for bird's eye view
    bird_view = np.zeros((scale, scale, 3), dtype=np.uint8)
    
    # Add grid lines
    grid_spacing = scale // 10
    for i in range(0, scale, grid_spacing):
        cv2.line(bird_view, (0, i), (scale, i), (50, 50, 50), 1)
        cv2.line(bird_view, (i, 0), (i, scale), (50, 50, 50), 1)
    
    # Draw camera position at the bottom center
    camera_pos = (scale // 2, scale - 10)
    cv2.circle(bird_view, camera_pos, 5, (0, 255, 255), -1)
    
    # Draw field of view
    fov_points = [
        (camera_pos[0], camera_pos[1]),
        (0, 0),
        (scale, 0)
    ]
    cv2.fillPoly(bird_view, [np.array(fov_points)], (20, 20, 60))
    
    # Add objects to the bird's eye view
    for risk_score, label, x1, y1, x2, y2 in object_risks:
        # Calculate center and size
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        obj_width = x2 - x1
        obj_height = y2 - y1
        
        # Normalize to frame size
        norm_x = center_x / frame_width
        
        # Use object size as a proxy for distance
        norm_y = 1.0 - (obj_height / frame_height)
        
        # Map to bird's eye view coordinates
        bev_x = int(norm_x * scale)
        bev_y = int(norm_y * scale * 0.8)  # 0.8 to leave space at top
        
        # Size based on object size
        size = max(2, int(5 * (1 - norm_y)))  # Larger when closer
        
        # Color based on risk
        color = get_risk_color(risk_score)
        
        # Draw object
        cv2.circle(bird_view, (bev_x, bev_y), size, color, -1)
    
    return bird_view


import numpy as np

def overlay_bird_eye_view(frame, bird_view, position=(50, 50), size=(150, 150)):
    """
    Overlay bird's eye view on the main frame.
    
    Args:
        frame: Main frame
        bird_view: Bird's eye view image
        position: Top-left position for overlay
        size: Size of the overlay
        
    Returns:
        frame: Frame with bird's eye view overlay
    """
    # Make a copy of the frame
    result = frame.copy()
    
    # Resize bird's eye view to the desired size
    bird_view_resized = cv2.resize(bird_view, size)
    
    # Create region of interest
    roi = result[position[1]:position[1]+size[1], position[0]:position[0]+size[0]]
    
    # Add semi-transparent overlay
    cv2.addWeighted(bird_view_resized, 0.7, roi, 0.3, 0, roi)
    
    # Place ROI back into the frame
    result[position[1]:position[1]+size[1], position[0]:position[0]+size[0]] = roi
    
    # Add border
    cv2.rectangle(result, position, (position[0]+size[0], position[1]+size[1]), (255, 255, 255), 2)
    
    # Add title
    cv2.putText(result, "Bird's Eye View", (position[0], position[1]-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return result