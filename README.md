"""
title: MoEA
author: Just-S-Rod
version: 2.0
required_open_webui_version: 0.3.9
"""

import logging
import json
import asyncio
from typing import List, Dict, Optional, AsyncGenerator, Callable, Awaitable, Union, Any
from pydantic import BaseModel, Field
from open_webui.constants import TASKS
from open_webui.utils.chat import generate_chat_completion
from open_webui.models.users import User
from fastapi import Request

# Constants and Setup
name = "MoEA"


def setup_logger():
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.set_name(name)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


logger = setup_logger()


class Pipe:
    __current_event_emitter__: Callable[[dict], Awaitable[None]]
    __user__: User
    __model__: str
    __request__: Any

    class Valves(BaseModel):
        """Define the default values for each valve."""

        MODELS: List[str] = Field(
            default=["llama3", "mistral", "gemma"],
            description="List of models to use in the MoEA architecture.",
        )
        NUM_LAYERS: int = Field(default=1, description="Number of MoEA layers.")
        TEMPERATURE: float = Field(
            default=0.7, description="Temperature for model generation"
        )
        TOP_K: int = Field(default=50, description="Top-k sampling parameter")
        TOP_P: float = Field(default=0.9, description="Top-p sampling parameter")
        MAX_RETRIES: int = Field(
            default=3, description="Maximum number of retry attempts"
        )
        USE_STREAMING: bool = Field(
            default=True, description="Whether to use streaming for responses"
        )
        SHOW_AGENT_THOUGHTS: bool = Field(
            default=False,
            description="Include agent thoughts in final response (for debugging)",
        )

    def __init__(self):
        """Initialize the Pipe object."""
        self.type = "manifold"
        self.valves = self.Valves()
        self.current_output = ""

    def pipes(self) -> list[dict[str, str]]:
        """Define the available pipes."""
        return [{"id": f"{name}-pipe", "name": f"{name} Pipe"}]

    async def get_streaming_completion(
        self,
        messages,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        model: str = "",
    ):
        """Get streaming completion from a model."""
        try:
            form_data = {
                "model": model,
                "messages": messages,
                "stream": True,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            }
            response = await generate_chat_completion(
                self.__request__,
                form_data,
                user=self.__user__,
            )
            if not hasattr(response, "body_iterator"):
                raise ValueError("Response does not support streaming")
            async for chunk in response.body_iterator:
                for part in self.get_chunk_content(chunk):
                    yield part
        except Exception as e:
            raise RuntimeError(f"Streaming completion failed: {e}")

    def get_chunk_content(self, chunk):
        """Extract content from streaming chunks."""
        if isinstance(chunk, bytes):
            chunk_str = chunk.decode("utf-8")
        else:
            chunk_str = chunk

        if chunk_str.startswith("data: "):
            chunk_str = chunk_str[6:]

        chunk_str = chunk_str.strip()

        if chunk_str == "[DONE]" or not chunk_str:
            return

        try:
            chunk_data = json.loads(chunk_str)
            if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                delta = chunk_data["choices"][0].get("delta", {})
                if "content" in delta:
                    yield delta["content"]
        except json.JSONDecodeError:
            logger.error(f'ChunkDecodeError: unable to parse "{chunk_str[:100]}"')

    async def get_completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        model: str = "",
    ) -> str:
        """Get non-streaming completion from a model."""
        logger.debug(f"LLM Query Prompt to {model}: {prompt[:100]}...")
        model_to_use = model if model else self.__model__
        try:
            response = await generate_chat_completion(
                self.__request__,
                {
                    "model": model_to_use,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                },
                user=self.__user__,
            )
            response_content = response["choices"][0]["message"]["content"]
            logger.debug(f"LLM Response from {model}: {response_content[:100]}...")
            return response_content
        except Exception as e:
            logger.error(f"LLM Call Error with model {model}: {e}")
            return f"Error querying model {model}: {str(e)}"

    async def query_with_retry(
        self,
        model: str,
        prompt: str,
        temperature: float = None,
        top_k: int = None,
        top_p: float = None,
        max_retries: int = None,
    ) -> str:
        """Query a model with retry support."""
        temperature = (
            temperature if temperature is not None else self.valves.TEMPERATURE
        )
        top_k = top_k if top_k is not None else self.valves.TOP_K
        top_p = top_p if top_p is not None else self.valves.TOP_P
        max_retries = (
            max_retries if max_retries is not None else self.valves.MAX_RETRIES
        )

        for attempt in range(max_retries):
            try:
                return await self.get_completion(
                    prompt=prompt,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    model=model,
                )
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Retrying model {model} after error: {e}")
                    await asyncio.sleep(2**attempt)
                    continue
                else:
                    logger.error(
                        f"Failed after {max_retries} attempts with model {model}: {e}"
                    )
                    return f"Error after {max_retries} attempts with model {model}: {str(e)}"

    def agent_prompt(self, original_prompt: str, previous_responses: List[str]) -> str:
        """Create a prompt for the agents and the aggregator."""
        thoughts = "\n\n".join(
            [f"Expert {i+1}: {resp}" for i, resp in enumerate(previous_responses)]
        )
        return f"""*Expert Insights:*
{thoughts}

*Original Question:*
{original_prompt}

Your task is to provide an answer based on these expert insights. Do not mention that you're using expert insights or that this is coming from multiple sources. Just provide a comprehensive answer.
"""

    async def moa_process(self, prompt: str) -> str:
        """Apply the Multi-Agent architecture to process a prompt."""
        layer_outputs = []

        if not self.valves.MODELS:
            return "Error: No models configured for MoEA."

        await self.emit_status("info", "Consulting expert models...", False)

        for layer in range(self.valves.NUM_LAYERS):
            current_layer_outputs = []
            layer_agents = self.valves.MODELS

            for i, agent in enumerate(layer_agents):
                await self.emit_status(
                    "info",
                    f"Consulting expert {i+1}/{len(layer_agents)} (model: {agent})...",
                    False,
                )

                if layer == 0:
                    instruct_prompt = prompt
                else:
                    instruct_prompt = self.agent_prompt(prompt, layer_outputs[-1])

                response = await self.query_with_retry(agent, instruct_prompt)
                current_layer_outputs.append(response)
                await self.emit_status(
                    "info", f"Expert {i+1} ({agent}) has responded.", False
                )

            layer_outputs.append(current_layer_outputs)
            await self.emit_status(
                "info", f"Layer {layer+1}/{self.valves.NUM_LAYERS} complete.", False
            )

        # Merge all expert responses
        merged_responses = []
        for layer_responses in layer_outputs:
            merged_responses.extend(layer_responses)

        # Create final prompt for synthesis
        await self.emit_status("info", "Synthesizing expert responses...", False)

        final_prompt = """You are an expert at synthesizing information from multiple sources.

I've consulted several experts about a question, and I need you to create a comprehensive answer based on their insights.

Guidelines:
1. Integrate all relevant information from the expert responses into a cohesive answer
2. Resolve any contradictions by providing balanced perspectives
3. DO NOT mention the experts or that this response is based on multiple sources
4. Present the information as your own comprehensive analysis
5. Format your response clearly with appropriate headings and structure
6. Be thorough but concise

Here are the expert responses:
"""

        for i, response in enumerate(merged_responses):
            final_prompt += f"\n\nEXPERT {i+1}:\n{response}"

        final_prompt += f"\n\nOriginal question: {prompt}\n\nYour synthesized answer:"

        # Get synthesis from a model (using the first model as the synthesizer)
        synthesizer_model = self.valves.MODELS[0]
        synthesis = await self.query_with_retry(
            synthesizer_model,
            final_prompt,
            temperature=0.5,  # Lower temperature for more coherent synthesis
            top_p=0.8,
        )

        # Include expert thoughts for debugging if enabled
        if self.valves.SHOW_AGENT_THOUGHTS:
            debug_info = "\n\n--- DEBUG: EXPERT THOUGHTS ---\n"
            for i, response in enumerate(merged_responses):
                debug_info += f"\n\nEXPERT {i+1} ({self.valves.MODELS[i % len(self.valves.MODELS)]}):\n{response}"
            synthesis += debug_info

        await self.emit_status("success", "Expert synthesis complete", True)
        return synthesis

    async def emit_message(self, message: str):
        """Emit a message."""
        await self.__current_event_emitter__(
            {"type": "message", "data": {"content": message}}
        )

    async def emit_replace(self, message: str):
        """Replace the current message."""
        await self.__current_event_emitter__(
            {"type": "replace", "data": {"content": message}}
        )

    async def emit_status(self, level: str, message: str, done: bool):
        """Emit a status update."""
        await self.__current_event_emitter__(
            {
                "type": "status",
                "data": {
                    "status": "complete" if done else "in_progress",
                    "level": level,
                    "description": message,
                    "done": done,
                },
            }
        )

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__=None,
        __task__=None,
        __model__=None,
        __request__=None,
    ) -> str:
        """Process the input through the MoEA architecture."""
        self.__user__ = User(**__user__)
        self.__request__ = __request__
        self.__current_event_emitter__ = __event_emitter__
        self.__model__ = __model__ if __model__ else self.valves.MODELS[0]

        # Handle non-streaming tasks
        if __task__ and __task__ != TASKS.DEFAULT:
            model_to_use = self.valves.MODELS[0]  # Use first model for non-streaming
            response = await generate_chat_completion(
                self.__request__,
                {
                    "model": model_to_use,
                    "messages": body.get("messages"),
                    "stream": False,
                },
                user=self.__user__,
            )
            return f"{name}: {response['choices'][0]['message']['content']}"

        logger.debug(f"Pipe {name} received: {body}")

        # Get the user's query
        messages = body.get("messages", [])
        if not messages:
            await self.emit_status("error", "No messages found in request", True)
            return "Error: No messages found in request"

        query = messages[-1].get("content", "").strip()
        if not query:
            await self.emit_status("error", "Empty query", True)
            return "Error: Empty query"

        # Process with MoEA
        await self.emit_status(
            "info", f"Processing with MoEA ({len(self.valves.MODELS)} models)...", False
        )
        result = await self.moa_process(query)

        # Emit the final result
        await self.emit_message(result)
        return ""

