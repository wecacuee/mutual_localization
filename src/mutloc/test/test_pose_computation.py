import unittest
import transformations as tf
import numpy as np


import mutloc.core as corr
import mutloc.utils as utils

# test case succeeds if all floats in the matrix are within TOL tolerance
TOL = 1e-6
# Run the test case N times with different random transforms
N = 10

class MutualLocalizationTest(unittest.TestCase):

    def test_mutual_localization(self):
        for i in range(N):
            self._test_one_mutual_localization()
            self._test_colinear_mutual_localization()
            print("Ran %dth test" % i)

    def _test_colinear_mutual_localization(self):
        """ Test when the markers are colinear """
        # Set the absolute position of markers
        # M_1, M_2 are fixed on robot L
        markersL = [np.array(p) for p in [(-.1, -.0, .0), (.1, -.0, .0)]]
        # M_3, M_4 are fixed on robot R
        markersR = [np.array(p) for p in [(-.1, -.0, .0), (.1, -.0, .0)]]

        # Choose a random rotation
        quaternion = np.random.random(4)
        mag = np.linalg.norm(quaternion)
        quaternion /= mag

        # Random translation with minimum more than twise the euclidean
        # distance of the markers.
        trans_min = 0.7
        trans_max = 10
        translation = np.random.rand(3) * (trans_max - trans_min) + trans_min
        self._test_mutual_localization(quaternion,translation, markersL, markersR)

    def _test_one_mutual_localization(self):
        # Set the absolute position of markers
        # M_1, M_2 are fixed on robot L
        markersL = [np.array(p) for p in [(-.1, -.1, .3), (.1, -.1, .3)]]
        # M_3, M_4 are fixed on robot R
        markersR = [np.array(p) for p in [(-.1, -.1, .3), (.1, -.1, .3)]]

        # Choose a random rotation
        quaternion = np.random.random(4)
        mag = np.linalg.norm(quaternion)
        quaternion /= mag

        # Random translation with minimum more than twise the euclidean
        # distance of the markers.
        trans_min = 0.7
        trans_max = 10
        translation = np.random.rand(3) * (trans_max - trans_min) + trans_min
        self._test_mutual_localization(quaternion,translation, markersL, markersR)

    def _test_mutual_localization(self, quaternion, translation, markersL, markersR):
        # Convert to a transform matrix
        T = tf.quaternion_matrix(quaternion)
        T[:3, 3] = translation

        # compute the marker positions in the other coordinate frame
        # M_3 and M_4 in coordinate frame L
        Tinv = utils.transform_inv(T)
        frame1scaled = [(utils.apply_transform(Tinv, pR), pR)
                        for pR in markersR]

        # M_1 and M_2 in coordinate frame R
        frame2scaled = [(pL, utils.apply_transform(T, pL))
                        for pL in markersL]

        # Normalize computed coordinates to unit vector in order to
        # simulate perspective projection
        frame1scaled = [(p1 / np.linalg.norm(p1), p2) for p1, p2 in frame1scaled]
        frame2scaled = [(p1, p2 / np.linalg.norm(p2)) for p1, p2 in frame2scaled]
        Tgot_roots = corr.solve_mutual_localization(frame1scaled, frame2scaled, tol=TOL)
        self.assertTrue(
            np.allclose(Tgot_roots[0], T),
            "Got {0} expected {1}".format(Tgot_roots[0], T))

if __name__ == '__main__':
    unittest.main()
