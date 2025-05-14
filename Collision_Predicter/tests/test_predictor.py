import unittest
import os
import sys
import cv2
import numpy as np

# Ensure src directory is in path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from collision_predictor import CollisionProbabilityPredictor
import utils

class TestCollisionPredictor(unittest.TestCase):
    """Test cases for collision probability prediction system."""
    
    def setUp(self):
        """Set up test case."""
        # Initialize predictor with a lower confidence threshold for testing
        self.predictor = CollisionProbabilityPredictor(confidence_threshold=0.2)
        
        # Create a test image with a simple object
        self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a rectangle in the center (simulating an object)
        cv2.rectangle(self.test_image, (270, 200), (370, 300), (0, 255, 0), -1)
    
    def test_initialization(self):
        """Test that the predictor initializes correctly."""
        self.assertIsNotNone(self.predictor.model, "Model should be loaded")
        self.assertEqual(self.predictor.confidence_threshold, 0.2, "Confidence threshold should match")
    
    def test_calculate_object_risk(self):
        """Test the object risk calculation."""
        # Create a mock detection
        detection = np.array([300, 200, 340, 280, 0.9, 0])  # Simulating a person in the center
        
        # Calculate risk
        risk_score = self.predictor._calculate_object_risk(detection, (480, 640))
        
        # The risk should be > 0 since it's a person in the center
        self.assertGreater(risk_score, 0, "Risk score should be greater than zero")
        self.assertLessEqual(risk_score, 1.0, "Risk score should not exceed 1.0")
    
    def test_collision_risk_classes(self):
        """Test that collision risk classes are properly defined."""
        # Check that important classes are included
        self.assertIn(0, self.predictor.collision_risk_classes, "Person should be a collision risk")
        self.assertIn(2, self.predictor.collision_risk_classes, "Car should be a collision risk")
    
    def test_weights_sum_to_one(self):
        """Test that risk factor weights sum to approximately 1."""
        total_weight = (self.predictor.weight_size + 
                       self.predictor.weight_center + 
                       self.predictor.weight_growth)
        self.assertAlmostEqual(total_weight, 1.0, delta=0.01, 
                              msg="Risk factor weights should sum to approximately 1")
    
    def test_predict_collision_returns_correct_types(self):
        """Test that predict_collision returns the expected data types."""
        # Load a test image
        test_image_path = os.path.join(os.path.dirname(__file__), "test_image.jpg")
        
        # If test image doesn't exist, create and save one
        if not os.path.exists(test_image_path):
            cv2.imwrite(test_image_path, self.test_image)
        
        # Load the test image
        test_image = cv2.imread(test_image_path)
        if test_image is None:
            test_image = self.test_image  # Use generated test image if file can't be loaded
        
        # Predict collision
        collision_prob, annotated_frame, object_risks = self.predictor.predict_collision(test_image)
        
        # Check return types
        self.assertIsInstance(collision_prob, float, "Collision probability should be a float")
        self.assertGreaterEqual(collision_prob, 0.0, "Collision probability should be >= 0")
        self.assertLessEqual(collision_prob, 1.0, "Collision probability should be <= 1")
        
        self.assertIsInstance(annotated_frame, np.ndarray, "Annotated frame should be a numpy array")
        self.assertEqual(annotated_frame.shape, test_image.shape, "Annotated frame should have same shape as input")
        
        self.assertIsInstance(object_risks, list, "Object risks should be a list")
    
    def test_utils_download_model(self):
        """Test the model download functionality."""
        # Try to download the smallest YOLO model
        model_path = utils.download_model("yolov8n.pt")
        self.assertIsNotNone(model_path, "Model path should not be None")
    
    def test_utils_filter_detection_by_region(self):
        """Test the region filtering functionality."""
        # Create a detection in the center
        center_detection = np.array([300, 200, 340, 280])
        
        # Test center region
        in_center = utils.filter_detection_by_region(center_detection, (480, 640), region='center')
        self.assertTrue(in_center, "Detection should be in center region")
        
        # Create a detection on the left
        left_detection = np.array([50, 200, 100, 280])
        
        # Test left region
        in_left = utils.filter_detection_by_region(left_detection, (480, 640), region='left')
        self.assertTrue(in_left, "Detection should be in left region")
        
        # Test that it's not in center
        not_in_center = utils.filter_detection_by_region(left_detection, (480, 640), region='center')
        self.assertFalse(not_in_center, "Left detection should not be in center region")


if __name__ == '__main__':
    unittest.main()