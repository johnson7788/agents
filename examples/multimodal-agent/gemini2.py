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

    room = ctx.room

    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    participant = await ctx.wait_for_participant()

    print(f"connected to room {room.name} with participant {participant.identity}")

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
    )
    agent.start(room, participant)
    agent.generate_reply(on_duplicate="cancel_new")
    session = model.sessions[0]

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM))