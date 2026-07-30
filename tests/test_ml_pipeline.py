import io
import unittest
from unittest import mock

import ml_pipeline
from ml_pipeline import load_data


class LoadDataTest(unittest.TestCase):
    def test_preserves_utf8_text_and_emoji(self):
        content = "text,sentiment\nCafé 🚀,positive\nMañana,negative\n"

        dataframe = load_data(io.BytesIO(content.encode("utf-8-sig")))

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

    def test_inspects_only_the_header_before_streaming_sentiment140(self):
        content = (
            "0,1,date,query,user,bad day\n4,2,date,query,user,great day\n"
        ).encode("latin-1")

        with mock.patch.object(
            ml_pipeline.pd,
            "read_csv",
            wraps=ml_pipeline.pd.read_csv,
        ) as read_csv:
            dataframe = load_data(io.BytesIO(content), sample_size=2)

        self.assertEqual(set(dataframe["text"]), {"bad day", "great day"})
        self.assertEqual(set(dataframe["polarity"]), {0, 1})
        self.assertEqual(read_csv.call_args_list[0].kwargs["nrows"], 0)
        self.assertEqual(read_csv.call_args_list[1].kwargs["chunksize"], 50000)

    def test_rejects_missing_labels_before_loading_export_rows(self):
        content = b"text,author\nhello,example\n"

        with (
            mock.patch.object(
                ml_pipeline.pd,
                "read_csv",
                wraps=ml_pipeline.pd.read_csv,
            ) as read_csv,
            self.assertRaisesRegex(ValueError, "needs a sentiment"),
        ):
            load_data(io.BytesIO(content))

        self.assertEqual(read_csv.call_count, 1)
        self.assertEqual(read_csv.call_args.kwargs["nrows"], 0)


if __name__ == "__main__":
    unittest.main()
