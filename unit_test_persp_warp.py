import numpy as np
import lanelines as ll
import unittest

class TestPerspectiveWarper(unittest.TestCase):
    
    def setUp(self):
        self.src = np.float32([ [0, 2], [2, 0], [4, 0], [6, 2] ])
        self.dst = np.float32([ [0, 2], [0, 0], [6, 0], [6, 2] ])        
          
    def test_init(self):
        bad_pts = np.float32([ [0, 0],  [0, 0] ])
        with self.assertRaises(AssertionError):
            ll.PerspectiveWarper(self.src, None)
        with self.assertRaises(AssertionError):
            ll.PerspectiveWarper(self.src, bad_pts)
        with self.assertRaises(AssertionError):
            ll.PerspectiveWarper(None, self.dst)
        with self.assertRaises(AssertionError):
            ll.PerspectiveWarper(bad_pts, self.dst)
            
    def test_warp(self):
        good_img = np.array([[1,2,3],[4,5,6]])
        bad_img = np.array([1,2,3])
        pw = ll.PerspectiveWarper(self.src, self.dst)
        with self.assertRaises(AssertionError):
            pw.warp(bad_img)
        warped = pw.warp(good_img)
        self.assertIsNotNone(warped)
 
    def test_unwarp(self):
        good_img = np.array([[1,2,3],[4,5,6]])
        bad_img = np.array([1,2,3])
        pw = ll.PerspectiveWarper(self.src, self.dst)
        with self.assertRaises(AssertionError):
            pw.unwarp(bad_img)
        unwarped = pw.unwarp(good_img)
        self.assertIsNotNone(unwarped)        


unittest.main()