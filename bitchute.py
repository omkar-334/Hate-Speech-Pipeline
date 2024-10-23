from typing import Literal

import pandas as pd
import requests
import yt_dlp


def search(
    query: str,
    sensitivity: Literal["normal", "nsfw", "nsfl"],
    num: int = 100,
    minutes: int = 5,
):
    url = "https://api.bitchute.com/api/beta/search/videos"
    payload = {
        "offset": 0,
        "limit": num,
        "query": query,
        "sensitivity_id": sensitivity,
        "sort": "new",
    }

    response = requests.post(url, json=payload)
    videos = response.json()["videos"]
    df = pd.DataFrame(videos)[["video_name", "duration", "video_id"]]
    df["duration"] = df["duration"].apply(normalize_duration)
    df["duration"] = pd.to_timedelta(df["duration"])
    df = df[df["duration"] < pd.Timedelta(minutes=minutes + 1)]
    df["url"] = df["video_id"].apply(get_url)
    df = df.dropna(subset=["url"])
    df = df.reset_index(drop=True)
    return df


def search_all(query: str, num: int = 100, minutes: int = 5, nsfl=False):
    dfs = [
        search(query, "normal", num, minutes),
        search(query, "nsfw", num, minutes),
    ]
    if nsfl:
        dfs.append(search(query, "nsfl", num, minutes))

    return pd.concat(dfs, ignore_index=True)


def download(url):
    id = url.split("/")[-1].split(".")[0]
    path = f"/content/videos/{id}.mp3"
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
            }
        ],
        "outtmpl": path,
    }
    print(f"Downloading video from {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(url)
        return path


def normalize_duration(duration):
    parts = duration.split(":")
    if len(parts) == 2:
        return f"00:{duration}"
    return duration


def get_url(video_id: str):
    url = "https://api.bitchute.com/api/beta/video/media"
    data = {"video_id": video_id}
    headers = {
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Access-Control-Request-Headers": "content-type",
        "Access-Control-Request-Method": "POST",
        "Origin": "https://www.bitchute.com",
        "Referer": "https://www.bitchute.com/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
        "Content-Type": "application/json",
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        try:
            return response.json().get("media_url", None)
        except:
            return None
    return None
