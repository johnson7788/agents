#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/2/11 09:59
# @File  : gemini2.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :

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
import aiohttp
from dotenv import load_dotenv

from google.genai.types import (
    Blob,
    LiveClientRealtimeInput,
)

load_dotenv()
logger = logging.getLogger("ai-agents")
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logger.setLevel(getattr(logging, log_level, logging.INFO))

async def entrypoint(ctx: JobContext):
    logger.info("starting entrypoint")

    fnc_ctx = llm.FunctionContext()

    @fnc_ctx.ai_callable()
    async def get_weather(
        location: Annotated[
            str, llm.TypeInfo(description="The location to get the weather for")
        ],
    ):
        """Called when the user asks about the weather. This function will return the weather for the given location."""
        logger.info(f"getting weather for {location}")
        url = f"https://wttr.in/{location}?format=%C+%t"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    weather_data = await response.text()
                    # # response from the function call is returned to the LLM
                    return f"The weather in {location} is {weather_data}."
                else:
                    raise Exception(
                        f"Failed to get weather data, status code: {response.status}"
                    )

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()

    print(f"connected to room {ctx.room.name} with participant {participant.identity}")

    # create a chat context with chat history, these will be synchronized with the server
    # upon calling `agent.generate_reply()`
    chat_ctx = llm.ChatContext()
    model = google.beta.realtime.RealtimeModel(
        model="gemini-2.0-flash-exp",
        voice="Puck",
        modalities=["AUDIO"],
        temperature=0.8,
        instructions="You are an intelligent, general-purpose AI assistant. Your goal is to provide accurate, concise, and helpful responses based on the user’s request",
    )
    agent = multimodal.MultimodalAgent(
        model=model,
        chat_ctx=chat_ctx,
        fnc_ctx=fnc_ctx,
    )
    agent.start(ctx.room, participant)
    agent.generate_reply(on_duplicate="cancel_new")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM))