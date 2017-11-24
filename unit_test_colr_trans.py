import numpy as np
import cv2
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import lanelines as ll
import unittest


class TestColorTransformer(unittest.TestCase):
    
    def setUp(self):
        self.test_img = scipy.misc.imread('test_images/signs_vehicles_xygrad.png')
        
    def test_init(self):
        ct = ll.ColorTransformer()
        self.assertEqual(ct.color_channel, 'RGB', \
                         'color_channel did not default to RGB')
        ct = ll.ColorTransformer('BGR')
        self.assertEqual(ct.color_channel, 'BGR', \
                         'color_channel did not init to BGR')
        with self.assertRaises(AssertionError):
            ct = ll.ColorTransformer('XYZ')

    def test_gray_thresh(self):
        ct = ll.ColorTransformer()
        gray_thresh = ct.gray_thresh(self.test_img, (180,255))
        self.assertEqual(gray_thresh.shape, self.test_img.shape[0:2], \
                         'return image not same dimension as original')
        self.assertTrue(((gray_thresh==0) | (gray_thresh==1)).all(), \
                        'return image not all 0s and 1s')
        
    def test_s_thresh(self):
        ct = ll.ColorTransformer()
        s_thresh = ct.s_thresh(self.test_img, (180,255))
        self.assertEqual(s_thresh.shape, self.test_img.shape[0:2], \
                         'return image not same dimension as original')
        self.assertTrue(((s_thresh==0) | (s_thresh==1)).all(), \
                        'return image not all 0s and 1s')
        
    def test_r_thresh(self):
        ct = ll.ColorTransformer()
        r_thresh = ct.r_thresh(self.test_img, (180,255))
        self.assertEqual(r_thresh.shape, self.test_img.shape[0:2], \
                         'return image not same dimension as original')
        self.assertTrue(((r_thresh==0) | (r_thresh==1)).all(), \
                        'return image not all 0s and 1s')
        
        
unittest.main()