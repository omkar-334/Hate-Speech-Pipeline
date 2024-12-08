import asyncio
import os

import pandas as pd
import whisperx
from dotenv import load_dotenv
from openai import OpenAI
from strictjson import strict_json_async

load_dotenv()


def load_model(lang="en"):
    asr_options = {
        "temperatures": 0,
        "beam_size": 1,
        "without_timestamps": True,
    }

    model = whisperx.load_model(
        "large",
        device="cuda",
        compute_type="float16",
        asr_options=asr_options,
        language=lang,
    )
    return model


client = OpenAI()
model = load_model()


def transcribe(video_path):
    audio = whisperx.load_audio(video_path)
    output = model.transcribe(audio, batch_size=32)
    output = "".join([segment["text"] for segment in output["segments"]])
    return output


def detect(text):
    response = client.moderations.create(
        model="omni-moderation-latest",
        input=text,
    )
    return response


async def llm(system_prompt: str, user_prompt: str) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    chat_completion = await client.chat.completions.create(
        messages=messages,
        model="gpt-4o-mini",
        temperature=0.5,
        max_tokens=360,
        stop=None,
        stream=False,
    )

    return chat_completion.choices[0].message.content


async def call_agent(user_prompt):
    system_prompt = """
    Determine if the given text contains hate speech. Follow these rules:

    ### Return True if:
    The text contains hate speech, either explicit or implicit, directed at a person or group.
    The text is in English.
    
    ### Return False if:
    The text does not contain hate speech.
    It describes violence or hate speech but is not directed at a person or group.
    It mentions a politician or famous person (e.g., Kamala Harris, Donald Trump).
    The text is not in English.
    
    ### Examples:
    "Kamala Harris is a terrible leader." → False (mentions a politician).
    "He was killed by smashing the head." → False (describes violence, not directed).
    "Go back to where you came from!" → True (explicit hate speech directed at a group).
    """

    result = await strict_json_async(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_format={
            "accept": "Evaluate based on the given rules, type: bool",
        },
        llm=llm,
    )
    return result
