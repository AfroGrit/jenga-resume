import os
import pandas as pd

import minsearch


DATA_PATH = os.getenv("DATA_PATH", "../data/_pmg_sample_clean.parquet.brotli")


def load_index(data_path=DATA_PATH):
    df = pd.read_csv(data_path)

    df['responder'] = df['responder'].str.replace('to ask the ', '', regex=False)
    # Convert 'date' column to the desired string format 'YYYY-MM-DD'
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    # Confirm that the 'date' column is now of type 'object'
    df['date'] = df['date'].astype('object')

    documents = df.to_dict(orient="records")

    index = minsearch.Index(
        text_fields=['date', 'id', 'mp', 'responder', 'question', 'answer'],
        keyword_fields=["id"],
    )

    index.fit(documents)
    return 
