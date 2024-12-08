import os

import pandas as pd

from bitchute import download, search_all
from pipeline import load_model, transcribe
from seed import keywords

model = load_model()

dfs = []
for word in keywords:
    df = search_all(word)
    df["word"] = word
    dfs.append(df)

maindf = pd.concat(dfs, ignore_index=True)

# maindf["path"] = maindf["url"].apply(download)
# maindf["text"] = maindf["path"].apply(transcribe)
# maindf.to_csv("dataset.csv")


def process_video(url, model):
    try:
        video_path = download(url)
        transcription = transcribe(video_path, model)

        return (url, transcription)

    except Exception as e:
        print(f"Error processing video {url}: {e}")
        return (url, None)

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
            print(f"Deleted audio file: {video_path}")
