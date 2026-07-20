import unittest
from io import BytesIO

from PIL import Image
from cscience.features.api.utils.measure_time import measure_time

from cscience.features.net_connector.clip_connector.clip_connector import get_service_info, get_feature_info, \
    initialize_once


class ClipConnectorTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # runs once before any tests in this class
        initialize_once()
        pass

    @classmethod
    def tearDownClass(cls):
        # runs once after all tests in this class
        pass

    def setUp(self):
        # runs before each test method
        pass

    def tearDown(self):
        # runs after each test method
        pass


    def test_service_info(self):
        info = get_service_info()
        self.assertTrue(isinstance(info, str))


    def test_feature_info(self):
        info = get_feature_info()
        self.assertTrue(isinstance(info, str))


    @measure_time(times=10, ignore_first=True)
    def test_text(self):
        embed_text("a photo of a cat")

    @measure_time(times=10, ignore_first=True)
    def test_image(self):
        with Image.open("./fixtures/flickr-dog-1.jpg") as img:
            v = embed_image(img)




if __name__ == '__main__':
    unittest.main()
