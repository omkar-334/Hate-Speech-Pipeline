import whisperx
from openai import OpenAI

client = OpenAI()


def load_model(lang="en"):
    asr_options = {
        "temperatures": 0,
        "beam_size": 1,
        "without_timestamps": True,
    }

    model = whisperx.load_model(
        "base",  # small
        device="cuda",
        compute_type="float32",  # 16
        asr_options=asr_options,
        language=lang,
    )
    return model


def transcribe(video_path):
    audio = whisperx.load_audio(video_path)
    output = model.transcribe(audio, batch_size=16)
    output = "".join([segment["text"] for segment in output["segments"]])
    return output


def detect(text):
    response = client.moderations.create(
        model="omni-moderation-latest",
        input=text,
    )
    return response
