import threading
import unittest
import cv2
import numpy as np

from src.Config import NUM_LANDMARKS, SAMPLE_PICTURE_PATH
from src.PipelineModules.Extractor.FeatureExtractor import ExtractorBase
from src.Utility.Dataclasses import FeaturePackage


class LmCapturerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.stop_event = threading.Event()
        cls.feature_extractor = ExtractorBase(stop_event=cls.stop_event)
        cls.sample_picture = cv2.cvtColor(cv2.imread(SAMPLE_PICTURE_PATH), cv2.COLOR_BGR2RGB)

    def test_feature_package_nullcheck(self):
        feature_package: FeaturePackage = self.feature_extractor.extract(self.sample_picture)

        self.assertIsNotNone(feature_package)
        self.assertIsNotNone(feature_package.lm_coordinates)

    def test_extract_dimensions(self):
        feature_package: FeaturePackage = self.feature_extractor.extract(self.sample_picture)

        landmark_shape = feature_package.lm_coordinates.shape
        self.assertEqual(landmark_shape[0], NUM_LANDMARKS,
                         "Number of rows should equal NUM_LANDMARKS (Config.py)")
        self.assertEqual(landmark_shape[1], NUM_LANDMARKS,
                         "Number of rows should equal NUM_LANDMARKS (Config.py)")

    def test_hand_detected_field(self):
        feature_package: FeaturePackage = self.feature_extractor.extract(self.sample_picture)
        self.assertEqual(feature_package.hand_detected, 1.0,
                         "hand_detected did not indicate a detected hand")  # 1.0 = hand detected

    def test_calcRelativeVector_dimensions(self):
        feature_package: FeaturePackage = self.feature_extractor.extract(self.sample_picture)
        landmark_coordinates = feature_package.lm_coordinates

        self.assertEqual(landmark_coordinates.shape[0], NUM_LANDMARKS,
                         "Number of rows should equal NUM_LANDMARKS (Config.py)")
        self.assertEqual(landmark_coordinates.shape[1], NUM_LANDMARKS,
                         "Number of rows should equal NUM_LANDMARKS (Config.py)")

    def test_calcRelativeVector_first_frame(self):
        self.feature_extractor.lastFrame = None #empty last frame to simulate the first iteration
        feature_package: FeaturePackage = self.feature_extractor.extract(self.sample_picture)
        lm_coordinates = feature_package.lm_coordinates.cpu() #tensor has to be copied to cpu to allow a comparison
        np.testing.assert_array_equal(
            lm_coordinates,
            np.zeros((21, 3)),
            err_msg="First frame not only zeros."
        )

    def test_calcRelativeVector_consecutive_frames(self):  # counts for all frames after the first one
        np.random.seed(42)
        rel_vector_1 = np.random.randint(0, 9, size=(21, 3))
        # print(str(vector_1))
        rel_vector_2 = np.random.randint(0, 9, size=(21, 3))
        # print(str(vector_2))
        rel_vector_3 = np.random.randint(0, 9, size=(21, 3))
        # print(str(vector_3))

        # starting with two because first frame is only 0
        rel_comp_vector_2 = np.array([[-2, -2, -3],
                                      [3, 2, 6],
                                      [-6, 1, 2],
                                      [5, 0, -7],
                                      [5, 2, -2],
                                      [-1, 0, -3],
                                      [1, -4, 4],
                                      [1, 0, 6],
                                      [6, 1, -2],
                                      [-8, 4, 2],
                                      [5, -2, -2],
                                      [-1, -1, 1],
                                      [-3, -6, 3],
                                      [-6, -4, 3],
                                      [3, 0, 1],
                                      [0, 6, -3],
                                      [2, -4, 1],
                                      [5, 1, -2],
                                      [3, -3, 4],
                                      [0, 5, -3],
                                      [-1, -5, -1]])

        rel_comp_vector_3 = np.array([[4, 5, -4],
                                      [-7, 0, 0],
                                      [3, 0, -4],
                                      [-2, -2, 7],
                                      [1, -3, -2],
                                      [2, 0, 3],
                                      [5, 8, -1],
                                      [-6, -8, -3],
                                      [-2, -6, 1],
                                      [0, -2, -6],
                                      [0, -4, -2],
                                      [-6, -4, 3],
                                      [6, 2, -4],
                                      [-2, 2, -3],
                                      [-2, -1, -2],
                                      [1, 0, 7],
                                      [-3, 2, 3],
                                      [-2, -3, -2],
                                      [-1, 3, -5],
                                      [7, -3, -2],
                                      [-2, 0, -4]]
                                     )

        result_vector = self.feature_extractor.calcRelativeVector(rel_vector_1)
        result_vector = self.feature_extractor.calcRelativeVector(rel_vector_2)
        np.testing.assert_array_equal(
            result_vector,
            rel_comp_vector_2,
            err_msg="real_vector_2 is not correct. (not equal to rel_comp_vector_2)"
        )
        result_vector = self.feature_extractor.calcRelativeVector(rel_vector_3)
        np.testing.assert_array_equal(
            result_vector,
            rel_comp_vector_3,
            err_msg="real_vector_3 is not correct. (not equal to rel_comp_vector_3)"
        )

        if __name__ == '__main__':
            unittest.main()
