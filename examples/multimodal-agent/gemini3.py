#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/2/11 12:01
# @File  : gemini3.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :

import logging
import os
import aiohttp
import datetime

log_path = "logs"
if not os.path.exists(log_path):
    os.makedirs(log_path)
logfile = os.path.join(log_path, "agent_api.log")
# 日志的格式
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler(logfile, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
import time
import json
import asyncio
import copy
import aiofiles
from typing import Annotated
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import cartesia, deepgram, openai, silero,azure
from openai import AsyncClient
from livekit.agents.multimodal import MultimodalAgent
from livekit import rtc
from livekit.agents.utils import AudioBuffer
from livekit.plugins import google
from agent_utils import InterviewUtils

load_dotenv(dotenv_path=".env")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    logging.info(f"开始建立一个新的房间的聊天，房间名为： {ctx.room.name}")
    room_name = ctx.room.name
        # https://ai.google.dev/gemini-api/docs/models/gemini?authuser=1&_gl=1*cksse8*_up*MQ..*_ga*NDMyNDI2MDEwLjE3MzY3NDQ2MTc.*_ga_P1DBVKWT6V*MTczODkyMDMyOC41LjEuMTczODkyMDM1MS4zNy4wLjUyODA3MjQxMA..#gemini-2.0-flash
    logging.info("starting GeminiRealFunction，Gemini Realtime API, 支持模型: gemini-2.0-flash-001, gemini-2.0-flash-lite-preview-02-05,LLM根据函数调用获取下一个问题，使用Realtime多模态模式")
    await gemini_multimodal_function(ctx)
async def gemini_multimodal_function(ctx: JobContext):
    """
    Gemini的Realtime接口，通过函数获取下一道题的内容
    Args:
        ctx:
    Returns:
    """
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()
    participant_identity = participant.identity
    logging.info(f"starting voice assistant for participant {participant_identity}")
    # 获取音频流
    user_audio_stream = rtc.AudioStream.from_participant(
        participant=participant,
        track_source=rtc.TrackSource.SOURCE_MICROPHONE,
    )
    metadata = json.loads(participant.metadata)
    language = metadata.get("language", "English")
    logging.info(f"starting Openai MultimodalAgent with config: {metadata}, 语言是: {language}")

    outline_info = metadata.get("outline_info")

    fnc_ctx = llm.FunctionContext()

    @fnc_ctx.ai_callable()
    async def get_weather(
            location: Annotated[
                str, llm.TypeInfo(description="The location to get the weather for")
            ],
    ):
        """Called when the user asks about the weather. This function will return the weather for the given location."""
        logging.info(f"getting weather for {location}")
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

    instructions = f"""Your name is Ava, An Interviewer, Now, we're about to begin a friendly chat centered around the project of {outline_info['name']}. 
The background of this conversation is {outline_info['topic']}. When you think the user has finished answering the current question, you can use a function to get the next interview question. do not return any text while calling the function."""
    first_user_msg = f"Hello there, what's your name? What will be the content of today's interview?"
    logging.info(f"instructions: {instructions}")

    model = google.beta.realtime.RealtimeModel(
        instructions=instructions,
        model="gemini-2.0-flash-exp",  #gemini-2.0-flash 兼容性问题
        api_key=os.environ["GOOGLE_API_KEY"],
        voice="Puck",
        modalities=["AUDIO"],  # 不能加["Text"]，加了会报错invalid_argument: Error in program Instantiation for language
        temperature=0.8,
    )
    chat_ctx = llm.ChatContext()
    # chat_ctx.append(text=first_user_msg, role="user") # Gemini不能加这个，否则不断自动循环调用function
    agent = MultimodalAgent(model=model, fnc_ctx=fnc_ctx, chat_ctx=chat_ctx)
    # agent._model._capabilities.supports_truncate = False # 这里不能改成True，否则agent_speech_committed永远不会被调用
    agent.start(ctx.room, participant)
    # 等待agent的初始化完成，否则没有microphone， 不能根据agent._started的状态，不知道为啥
    agent.generate_reply()

    audio_buffer: AudioBuffer = []
    audio_lock = asyncio.Lock()  # 用于确保缓冲区的线程安全
    @agent.on("user_started_speaking")
    def _on_user_started_speaking():
        logging.info("user started speaking, 开始存储audio_buffer")  # 4) 用户开始说话
        # chat_ctx = agent.chat_ctx_copy() # 对于multimodal，需要先复制一份chat_ctx
        # logging.info(f"user_started_speaking messages: {chat_ctx.messages[-1]}")
        async def handle_audio_stream():
            async for audio_event in user_audio_stream:
                async with audio_lock:
                    audio_buffer.append(audio_event.frame)
        # start_time = time.time()
        asyncio.create_task(handle_audio_stream())  # 使用 asyncio.create_task 调度异步任务
        # logging.info(f"user_started_speaking time: {time.time() - start_time} 秒")
    @agent.on("user_stopped_speaking")
    def _on_user_stopped_speaking():
        logging.info("user stopped speaking, 开始保存audio_buffer为wav文件")
        # chat_ctx = agent.chat_ctx_copy() # 对于multimodal，需要先复制一份chat_ctx
        # logging.info(f"user_stopped_speaking messages: {chat_ctx.messages[-1]}")
        async def save_audio_buffer():
            async with audio_lock:
                if not audio_buffer:
                    logging.warning("audio_buffer is empty, skipping save.")
                    return
                # 假设 combine_audio_frames 和 to_wav_bytes 是同步操作
                wav_data = rtc.combine_audio_frames(audio_buffer).to_wav_bytes()
                await fnc_ctx.save_audio_data(audio_data=wav_data)
                audio_buffer.clear()  # 清空缓冲区，准备下一次使用
            # 异步保存 WAV 文件
        # start_time = time.time()
        asyncio.create_task(save_audio_buffer())  # 使用 asyncio.create_task 调度异步任务
        # logging.info(f"保存User语音的耗时:user_stopped_speaking , {time.time()-start_time}秒")

    @agent.on("user_speech_committed")
    def _on_user_speech_committed(user_msg):
        logging.info(f"user said: {user_msg}")  # 6）用户的语音识别的内容提交
        # chat_ctx = agent.chat_ctx_copy() # 对于multimodal，需要先复制一份chat_ctx
        # logging.info(f"user_speech_committed messages: {chat_ctx.messages[-1]}")
        # 保存消息
        # start_time = time.time()
        asyncio.create_task(fnc_ctx.save_user_msg(user_msg))
        # logging.info(f"保存User消息耗时:user_speech_committed , {time.time()-start_time}秒")
    @agent.on("agent_started_speaking")  #1)Agent先说话
    def _on_agent_started_speaking():
        logging.info("agent started speaking")
        # chat_ctx = agent.chat_ctx_copy() # 对于multimodal，需要先复制一份chat_ctx
        # logging.info(f"agent_started_speaking messages: {chat_ctx.messages[-1]}")
    @agent.on("agent_stopped_speaking")  #2)Agent停止说话
    def _on_agent_stopped_speaking():
        logging.info("agent stopped speaking")
        # chat_ctx = agent.chat_ctx_copy() # 对于multimodal，需要先复制一份chat_ctx
        # logging.info(f"agent_stopped_speaking messages: {chat_ctx.messages[-1]}")
        if fnc_ctx.communication_records.get("is_end"):
            asyncio.create_task(fnc_ctx.end_the_interview())
    @agent.on("agent_speech_committed")
    def _on_agent_speech_committed(msg):
        logging.info(f"agent said: {msg}") #3) Agent说的内容提交
        # chat_ctx = agent.chat_ctx_copy() # 对于multimodal，需要先复制一份chat_ctx
        # logging.info(f"agent_speech_committed messages: {chat_ctx.messages[-1]}")
        # 保存消息
        # start_time = time.time()
        asyncio.create_task(fnc_ctx.save_agent_msg(msg))
        # logging.info(f"保存Agent消息耗时，agent_speech_committed time: {time.time()-start_time}秒")

    @agent.on("function_calls_collected")
    def _on_function_calls_collected(function_calls):
        logging.info(f"function calls collected: {function_calls}")

    @agent.on("function_calls_finished")
    def _on_function_calls_finished(called_fncs):
        logging.info(f"function calls finished: {called_fncs}")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
