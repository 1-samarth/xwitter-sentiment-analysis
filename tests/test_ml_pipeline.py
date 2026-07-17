import io
import unittest

from ml_pipeline import load_data


class LoadDataTest(unittest.TestCase):
    def test_preserves_utf8_text_and_emoji(self):
        content = "text,sentiment\nCafé 🚀,positive\nMañana,negative\n"

        dataframe = load_data(io.BytesIO(content.encode("utf-8")))

        self.assertEqual(set(dataframe["text"]), {"Café 🚀", "Mañana"})
        self.assertEqual(set(dataframe["polarity"]), {0, 1})

    def test_accepts_float_labels_created_by_missing_numeric_values(self):
        content = "text,polarity\npositive row,4\nmissing row,\nnegative row,0\n"

        dataframe = load_data(io.BytesIO(content.encode("utf-8")))

        labels = dict(zip(dataframe["text"], dataframe["polarity"]))
        self.assertEqual(labels, {"positive row": 1, "negative row": 0})

    def test_falls_back_to_latin1_for_legacy_datasets(self):
        content = b"text,sentiment\ncaf\xe9,positive\n"

        dataframe = load_data(io.BytesIO(content))

        self.assertEqual(dataframe.iloc[0]["text"], "café")
        self.assertEqual(dataframe.iloc[0]["polarity"], 1)


if __name__ == "__main__":
    unittest.main()
