import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

with patch("logging.getLogger") as m_logger:
    from ml2.train import check_and_get_df, compute_unique_genres


class TestTrain(unittest.TestCase):
    """This class contains unitary tests for the module ml2/train.py"""

    @patch("os.path.exists")
    def test_check_and_get_df_not_existing(self, m_exists):
        # given
        m_exists.return_value = False
        # when
        flag, df = check_and_get_df("any_path")
        # then
        self.assertFalse(flag)
        self.assertTrue(df.empty)

    @patch("pandas.read_csv")
    @patch("os.path.exists")
    def test_check_and_get_df_bad_format(self, m_exists, m_readcsv):
        # given
        m_exists.return_value = True
        input_df = pd.DataFrame(data={
            "Title": ["t1", "t2"],
            "Description": ["d1", "d2"],
            "AnotherField": ["a1", "a2"]
        })
        m_readcsv.return_value = input_df
        # when
        flag, df = check_and_get_df("any_path")
        # then
        self.assertFalse(flag)
        self.assertTrue(df.empty)

    @patch("pandas.read_csv")
    @patch("os.path.exists")
    def test_check_and_get_df_good_format(self, m_exists, m_readcsv):
        # given
        m_exists.return_value = True
        input_df = pd.DataFrame(data={
            "Title": ["t1", "t2"],
            "Description": ["d1", "d2"],
            "Genre": ["g1", "g2"]
        })
        m_readcsv.return_value = input_df
        # when
        flag, df = check_and_get_df("any_path")
        # then
        self.assertTrue(flag)
        self.assertTrue(df.equals(input_df))

    def test_compute_unique_genres_prioritization(self):
        # given
        input_df = pd.DataFrame(data={
            "Genre": ["Animation,History", "Crime,Action"]
        })
        # when
        df_train = compute_unique_genres(input_df)
        # then
        computed_genres = np.array(df_train["UniqueGenre"])
        expected_genres = np.array(["Animation", "Action"])
        self.assertTrue(all(expected_genres == computed_genres))

    def test_compute_unique_genres_mapping(self):
        # given
        input_df = pd.DataFrame(data={
            "Genre": ["Crime", "Fantasy"]
        })
        # when
        df_train = compute_unique_genres(input_df)
        # then
        computed_genres = np.array(df_train["UniqueGenre"])
        expected_genres = np.array(["Thriller", "Adventure"])
        self.assertTrue(all(expected_genres == computed_genres))


if __name__ == '__main__':
    unittest.main()