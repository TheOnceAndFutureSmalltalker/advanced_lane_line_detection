import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import lanelines as ll
import unittest

class TestCameraCalibrator(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def test_init(self):
        cc = ll.CameraCalibrator(9, 6, 'camera_cal/calibration*.jpg')
        self.assertEqual(cc.nx, 9, 'nx not initialized')
        self.assertEqual(cc.ny, 6, 'ny not initialized')
        self.assertEqual(cc.file_template, \
                         'camera_cal/calibration*.jpg', \
                         'file_template not initialized')
        self.assertIsNone(cc.mtx, 'mtx should initialize to None')
        self.assertIsNone(cc.dist, 'dist should initialize to None')
        
    def test_calibrate(self):
        cc = ll.CameraCalibrator(9, 6, 'camera_cal/calibration*.jpg')
        cc.calibrate()
        self.assertIsNotNone(cc.mtx, 'mtx should not be None')
        self.assertIsNotNone(cc.dist, 'dist should not be None')
        
    def test_undistort(self):
        cc = ll.CameraCalibrator(9, 6, 'camera_cal/calibration*.jpg')
        img = cv2.imread('test_images/test1.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        with self.assertRaises(AssertionError):
            undistorted = cc.undistort([])
        undistorted = cc.undistort(gray)
        self.assertTrue(type(undistorted) is np.ndarray, \
                        'undistorted image is not a numpy array')
        self.assertEqual(gray.shape, undistorted.shape, \
                         'undistored image is not same dimensions as original')
        

unittest.main()


