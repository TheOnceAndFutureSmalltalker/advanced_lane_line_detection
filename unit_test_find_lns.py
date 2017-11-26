import numpy as np
import cv2
import scipy.misc
import matplotlib.pyplot as plt
import lanelines as ll
import unittest

class TestLaneFinder(unittest.TestCase):
    
    def setUp(self):
        self.test_img = scipy.misc.imread('output_images/test6_ppl_warped_bin.jpg')
    
    def test_find_lane_lines(self):
        llf = ll.LaneLineFinder()
        ploty, left_fitx, right_fity = llf.find_lane_lines(self.test_img, True)
        self.assertIsInstance(ploty, np.ndarray)
        self.assertIsInstance(left_fitx, np.ndarray)
        self.assertIsInstance(right_fity, np.ndarray)
        self.assertTrue(llf.left_rad > 500, 'lane radius should be at least 500m')
        self.assertTrue(llf.right_rad > 500, 'lane radius should be at least 500m')
        self.assertIsInstance(llf.out_img, np.ndarray)
        plt.imshow(llf.out_img)
        
        # second time through 
        ploty, left_fitx, right_fity = llf.find_lane_lines(self.test_img, False)
        self.assertIsInstance(ploty, np.ndarray)
        self.assertIsInstance(left_fitx, np.ndarray)
        self.assertIsInstance(right_fity, np.ndarray) 
        self.assertIsNone(llf.out_img)
        self.assertTrue(llf.left_rad > 500, 'lane radius should be at least 500m')
        self.assertTrue(llf.right_rad > 500, 'lane radius should be at least 500m')
        print(llf.left_rad)
        print(llf.right_rad)
        print(llf.center_offset)
        

        


        
        
        
        
unittest.main()