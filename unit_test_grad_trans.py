import numpy as np
import cv2
import scipy.misc
import lanelines as ll
import unittest

class TestGradientTransformer(unittest.TestCase):
    
    def setUp(self):
        self.test_img = scipy.misc.imread('test_images/signs_vehicles_xygrad.png')
    
    def test_init(self):
        gt = ll.GradientTransformer()
        self.assertEqual(gt.kernel, 3, 'kernel did not default to 3')
        self.assertEqual(gt.color_convert, cv2.COLOR_RGB2GRAY, \
                         'color_convert did not default to cv2.COLOR_RGB2GRAY')
        gt = ll.GradientTransformer(15, cv2.COLOR_BGR2GRAY)
        self.assertEqual(gt.kernel, 15, 'kernel did not init to 15')
        self.assertEqual(gt.color_convert, cv2.COLOR_BGR2GRAY, \
                         'color_convert did not init to cv2.COLOR_BGR2GRAY')
        
    def test_abs_thresh(self):
        gt = ll.GradientTransformer();
        abs_bin = gt.abs_thresh(self.test_img, orient='x', thresh=(50, 150))
        self.assertEqual(abs_bin.shape, self.test_img.shape[0:2], \
                         'return image not same dimension as original')
        self.assertTrue(((abs_bin==0) | (abs_bin==1)).all(), \
                        'return image not all 0s and 1s')

    def test_mag_thresh(self):
        gt = ll.GradientTransformer();
        abs_bin = gt.mag_thresh(self.test_img, thresh=(50, 150))
        self.assertEqual(abs_bin.shape, self.test_img.shape[0:2], \
                         'return image not same dimension as original')
        self.assertTrue(((abs_bin==0) | (abs_bin==1)).all(), \
                        'return image not all 0s and 1s')
        
    def test_dir_thresh(self):
        gt = ll.GradientTransformer();
        abs_bin = gt.dir_thresh(self.test_img, thresh=(0.7, 1.3))
        self.assertEqual(abs_bin.shape, self.test_img.shape[0:2], \
                         'return image not same dimension as original')
        self.assertTrue(((abs_bin==0) | (abs_bin==1)).all(), \
                        'return image not all 0s and 1s')
    
unittest.main()