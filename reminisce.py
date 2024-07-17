from typing import Literal, IO

import os
import click
from dotenv import dotenv_values
import errno
import time
import base64
from io import BytesIO
from operator import itemgetter
from openai import OpenAI
from elevenlabs import VoiceSettings, play
from elevenlabs.client import ElevenLabs
import yaml


def encode_frame(path: str):
    while True:
        try:
            # read frame & to base64 string
            with open(path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except IOError as e:
            if e.errno != errno.EACCES:
                # not a "file in use" error, re-raise
                raise
            # file is being written to, wait a bit and retry
            time.sleep(0.1)


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
    system_prompt: str,
    script: list[dict[str, any]],
    new_line: dict[str, any],
    max_tokens=500,
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
        + new_line,
        max_tokens=max_tokens,
    )
    response_text = response.choices[0].message.content
    return response_text


def text_to_speech_stream(
    client: ElevenLabs, voice_id: str, voice_settings: VoiceSettings, text: str
) -> IO[bytes]:
    # Perform the text-to-speech conversion
    response = client.text_to_speech.convert(
        voice_id=voice_id,
        optimize_streaming_latency="0",
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_turbo_v2",
        voice_settings=voice_settings,
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


# fmt: off
@click.command()
@click.option("--stability", type=click.FloatRange(0, 1),  default=0.5,  help="the stability setting for the elevenlabs voice")
@click.option("--similarity", type=click.FloatRange(0, 1), default=0.75, help="the similarity setting for the elevenlabs voice")
@click.option("--style", type=click.FloatRange(0, 1),      default=0.0,  help="the style setting for the elevenlabs voice")
@click.option("--boost", is_flag=True, show_default=True,  default=True, help="the 'use speaker boost' setting for the elevenlabs voice to boost similarity")
@click.option("--interval", type=int,                      default=5,    help="the interval between checking analysing frames")
@click.option("--prompts", type=click.File(mode="r", encoding="utf-8"),  help="a .yml file with the 'system' prompt & a list of 'user' prompts", required=True)
@click.argument("directory", type=click.Path(exists=False))
# fmt: on
def reminisce(
    stability: float,
    similarity: float,
    style: float,
    boost: bool,
    interval: int,
    prompts: IO[str],
    directory: str,
) -> None:
    # get data from .env file
    config = dotenv_values(".env")
    filename = "frame.jpg"
    frame_path = os.path.join(directory, filename)

    # setup apis for openai (text) and elevenlabs (text to speech)
    openai = OpenAI(api_key=config["OPENAI_API_KEY"])
    elevenlabs = ElevenLabs(api_key=config["ELEVENLABS_API_KEY"])

    # parse prompts file
    system_prompt, user_prompts = itemgetter("system", "user")(yaml.safe_load(prompts))

    script = []
    for image_prompt in user_prompts:
        if not os.path.isfile(frame_path := os.path.join(directory, filename)):
            raise RuntimeError(
                f"'{frame_path}' was not found. is 'capture.py' running?"
            )

        print("analysing frame")

        frame = encode_frame(frame_path)

        analysis = analyze_image(
            client=openai,
            model="gpt-4o",
            system_prompt=system_prompt,
            script=script,
            new_line=generate_new_line(frame, image_prompt, "low"),
            max_tokens=300,
        )

        script = script + [{"role": "assistant", "content": analysis}]

        print(f"answer: {analysis}")

        audio_stream = text_to_speech_stream(
            client=elevenlabs,
            voice_id=config["ELEVENLABS_VOICE_ID"],
            voice_settings=VoiceSettings(
                stability=stability,
                similarity_boost=similarity,
                style=style,
                use_speaker_boost=boost,
            ),
            text=analysis,
        )

        play(audio_stream)

        print(f"waiting for {interval} seconds...")

        time.sleep(interval)


if __name__ == "__main__":
    reminisce()
