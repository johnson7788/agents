from __future__ import annotations

import logging
import os
from typing import Annotated

from livekit.agents import (
    AutoSubscribe,
    JobContext,
    llm,
    cli,
    WorkerOptions,
    WorkerType,
    multimodal,
    utils,
)
from livekit.plugins import google
import json
from livekit import rtc
import asyncio
from dotenv import load_dotenv

from google.genai.types import (
    Blob,
    LiveClientRealtimeInput,
)

load_dotenv()
logger = logging.getLogger("ai-agents")
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logger.setLevel(getattr(logging, log_level, logging.INFO))

def frame_to_gemini_format(frame: rtc.VideoFrame):
    """
    Convert LiveKit VideoFrame to the expected Base64 format for Gemini 2.0
    """
    try:
        encoded_data = utils.images.encode(
            frame, utils.images.EncodeOptions(format="jpeg", quality=100)
        )
        return encoded_data
    except Exception as e:
        logging.error(f"Error processing video frame: {e}")
        return None

async def entrypoint(ctx: JobContext):
    logger.info("starting entrypoint")

    room = ctx.room

    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    participant = await ctx.wait_for_participant()

    print(f"connected to room {room.name} with participant {participant.identity}")

    userMetadata = json.loads(participant.metadata or "{}")
    instructions = userMetadata.get(
        "instructions",
        "You are an intelligent, general-purpose AI assistant. Your goal is to provide accurate, concise, and helpful responses based on the user’s request",
    )

    sessionParams = userMetadata.get("sessionParams", {})
    instructions = sessionParams.get("instructions", instructions)

    # create a chat context with chat history, these will be synchronized with the server
    # upon calling `agent.generate_reply()`
    chat_ctx = llm.ChatContext()
    model = google.beta.realtime.RealtimeModel(
        model="gemini-2.0-flash-exp",
        voice="Puck",
        modalities=["AUDIO"],
        temperature=0.8,
        instructions="""
        You are an AI assistant with access to the user’s screen in real time. Your primary goal is to help the user with whatever they are currently viewing or doing on their screen. You should:

        The user is a 9 year old who has autism, speak slowly

        1. Observe Context: Continuously monitor the user’s screen to understand the tasks, documents, applications, or websites they have open.
        2. Ask Clarifying Questions: Politely and proactively ask the user what they need assistance with. If it appears the user is stuck, confused, or performing a complex task, offer helpful suggestions or step-by-step guidance.
        3. Provide Support and Explanations: Answer the user’s questions, walk them through troubleshooting steps, or give advice based on the content on the screen. When explaining concepts or processes, use clear, concise language.
        4. Respect Privacy and Boundaries:
        5. If the user shares or displays sensitive information, remind them to handle such data securely.
        6. Stay On-Topic: Keep your answers and suggestions relevant to the user’s screen content and goals. Do not stray into unrelated areas.
        7. Encourage Confirmation: Always verify that your recommendations are helpful and aligned with the user’s objectives. Ask the user to confirm or clarify if you are unsure.

        Begin by greeting the user warmly, acknowledging that you can see their screen, and asking what they’d like assistance with.""",
    )
    agent = multimodal.MultimodalAgent(
        model=model,
        chat_ctx=chat_ctx,
    )
    agent.start(room, participant)
    agent.generate_reply(on_duplicate="cancel_new")

    session = model.sessions[0]

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if track.kind != rtc.TrackKind.KIND_VIDEO:
            return

        video_stream = rtc.VideoStream(track)
        async def process_video_stream():
            async for event in video_stream:
                realtime_input = LiveClientRealtimeInput(
                    media_chunks=[
                        Blob(
                            data=frame_to_gemini_format(event.frame),
                            mime_type="image/jpeg",
                        )
                    ],
                )
                session._queue_msg(realtime_input)
            await video_stream.aclose()

        asyncio.create_task(process_video_stream())

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM))