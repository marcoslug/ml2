import json
import unittest
from unittest.mock import patch, MagicMock

with patch("logging.getLogger") as m_logger:
    from ml2.predict import load_model, predict_genre


class TestPredict(unittest.TestCase):
    """This class contains unitary tests for the module ml2/predict.py"""

    @patch("os.path.exists")
    def test_load_model_not_existing(self, m_exists):
        # given
        m_exists.return_value = False
        # when
        loaded_model = load_model()
        # then
        self.assertTrue(loaded_model is None)

    @patch("os.path.exists")
    def test_predict_genre_no_model(self, m_exists):
        # given
        m_exists.return_value = False
        title = "a title"
        description = "a description"
        # when
        response = predict_genre(title, description)
        # then
        expected_dict = {"title": title, "description": description, "genre": ""}
        self.assertEqual(json.dumps(expected_dict, indent=4), response)

    @patch("ml2.predict.load")
    @patch("os.path.exists")
    def test_predict_genre_with_model(self, m_exists, m_load):
        # given
        m_exists.return_value = True
        comedy_model = MagicMock()
        comedy_model.predict.return_value = ["Comedy"]
        m_load.return_value = comedy_model
        title = "a title"
        description = "a description"
        # when
        response = predict_genre(title, description)
        # then
        expected_dict = {"title": title, "description": description, "genre": "Comedy"}
        self.assertEqual(json.dumps(expected_dict, indent=4), response)


if __name__ == '__main__':
    unittest.main()