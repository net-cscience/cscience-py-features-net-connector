import io
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image
from cscience.features.api.utils.measure_time import measure_time

from cscience.features.net_connector.clip_spatial_connector.clip_spatial_connector import get_service_info, \
    get_feature_info, \
    initialize_once, image_regions, score_regions

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
class ClipSpatialConnectorTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # runs once before any tests in this class
        initialize_once(str(FIXTURE_ROOT/"config"), False)
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
    def test_image_regions(self):
        image = Image.open("./fixtures/flickr-dog-1.jpg")
        v = image_regions(image.tobytes())
        pass

    @measure_time(times=10, ignore_first=True)
    def test_score_regions(self):

        image = Image.open("./fixtures/flickr-dog-1.jpg")
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=95)

        v= image_regions(buffer.getvalue())
        s = score_regions(["a photo of a dog", "a photo of a cat"], v)
        pass

if __name__ == '__main__':
    unittest.main()
