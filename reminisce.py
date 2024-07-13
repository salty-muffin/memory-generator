from typing import Literal, IO

import os
import click
from dotenv import dotenv_values
import cv2
import time
import base64
from openai import OpenAI
from elevenlabs import VoiceSettings, play
from elevenlabs.client import ElevenLabs
from io import BytesIO

import json


def encode_frame(path: str):
    # read frame
    frame = cv2.imread(path)

    # convert to jpeg
    _, buffer = cv2.imencode(".jpg", frame)

    # convert to base64 string
    base64_string = base64.b64encode(buffer).decode("utf-8")

    return base64_string


def generate_new_line(
    base64_image: bytes, prompt: str, detail: Literal["high", "low", "auto"]
) -> dict[str, any]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": detail,
                    },
                },
            ],
        },
    ]


def analyze_image(
    client: OpenAI,
    model: str,
    base64_image: bytes,
    system_prompt: str,
    image_prompt: str,
    image_detail: Literal["high", "low", "auto"],
    script: dict[str, any],
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
        ]
        + script
        + generate_new_line(base64_image, image_prompt, image_detail),
        max_tokens=500,
    )
    response_text = response.choices[0].message.content
    return response_text


def text_to_speech_stream(client: ElevenLabs, voice_id: str, text: str) -> IO[bytes]:
    # Perform the text-to-speech conversion
    response = client.text_to_speech.convert(
        voice_id=voice_id,
        optimize_streaming_latency="0",
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_turbo_v2",
        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=0.75,
            style=0.0,
            use_speaker_boost=True,
        ),
    )

    # Create a BytesIO object to hold the audio data in memory
    audio_stream = BytesIO()

    # Write each chunk of audio data to the stream
    for chunk in response:
        if chunk:
            audio_stream.write(chunk)

    # Reset stream position to the beginning
    audio_stream.seek(0)

    # Return the stream for further use
    return audio_stream


@click.command()
@click.argument("directory", type=click.Path(exists=False))
def reminisce(directory: str) -> None:
    # get data from .env file
    config = dotenv_values(".env")
    filename = "frame.jpg"

    # setup apis for openai (text) and elevenlabs (text to speech)
    openai = OpenAI(api_key=config["OPENAI_API_KEY"])
    elevenlabs = ElevenLabs(api_key=config["ELEVENLABS_API_KEY"])

    script = []
    system_prompt = """
                You are the person in the image. Narrate the picture as if it is a memory from the perspective of the person in the picture.
                Speak in past tense. Speak as if you remember this moment from the future. Don't repeat yourself. Make it short.
                Speak less about your surroundings and more about the things going through your head. You can make these up.
                """
    image_prompt = "Describe this image"
    while os.path.isfile(frame_path := os.path.join(directory, filename)):
        frame = encode_frame(frame_path)

        analysis = analyze_image(
            client=openai,
            model="gpt-4o",
            base64_image=frame,
            system_prompt=system_prompt,
            image_prompt=image_prompt,
            image_detail="low",
            script=[],
        )

        script = script + [{"role": "assistant", "content": analysis}]

        print(f"answer: {analysis}")

        audio_stream = text_to_speech_stream(
            elevenlabs, config["ELEVENLABS_VOICE_ID"], analysis
        )

        play(audio_stream)

        # time.sleep(5)
        break


if __name__ == "__main__":
    reminisce()
