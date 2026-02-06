"""ElevenLabs Conversational AI handler for Reachy Mini.

Implements the fastrtc AsyncStreamHandler interface using ElevenLabs Voice Agent APIs.
"""

import json
import random
import asyncio
import logging
from typing import Any, Tuple, Optional

import numpy as np
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item
from numpy.typing import NDArray
from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import ClientTools, Conversation

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.elevenlabs_audio import (
    ELEVENLABS_SAMPLE_RATE,
    ReachyAudioInterface,
    convert_audio_for_elevenlabs,
)
from reachy_mini_conversation_app.tools.core_tools import (
    ToolDependencies,
    get_tool_specs,
    dispatch_tool_call,
)
from reachy_mini_conversation_app.tools.openclaw_relay import set_pending_response


logger = logging.getLogger(__name__)

IDLE_TIMEOUT_S = 15.0  # Seconds of inactivity before triggering idle behavior


def get_elevenlabs_agent_id(profile: str | None) -> str:
    """Get ElevenLabs agent ID for a profile.

    Checks for agent_id.txt in the profile directory, falls back to default.

    Args:
        profile: Profile name or None for default

    Returns:
        ElevenLabs agent ID string

    """
    from pathlib import Path

    profiles_dir = Path(__file__).parent / "profiles"

    if profile:
        agent_file = profiles_dir / profile / "agent_id.txt"
        if agent_file.exists():
            agent_id = agent_file.read_text(encoding="utf-8").strip()
            if agent_id:
                logger.info("Using agent ID from profile '%s': %s", profile, agent_id[:8] + "...")
                return agent_id

    default_agent_id = config.ELEVENLABS_DEFAULT_AGENT_ID
    if default_agent_id:
        logger.info("Using default ElevenLabs agent ID: %s", default_agent_id[:8] + "...")
        return default_agent_id

    logger.error("No ElevenLabs agent ID configured. Set ELEVENLABS_DEFAULT_AGENT_ID or create agent_id.txt")
    raise ValueError("No ElevenLabs agent ID configured")


class ElevenLabsRealtimeHandler(AsyncStreamHandler):
    """ElevenLabs Conversational AI handler implementing fastrtc AsyncStreamHandler."""

    def __init__(
        self,
        deps: ToolDependencies,
        gradio_mode: bool = False,
        instance_path: Optional[str] = None,
    ):
        """Initialize the handler.

        Args:
            deps: Tool dependencies (robot, movement manager, etc.)
            gradio_mode: Whether running in Gradio UI mode
            instance_path: Path to instance directory for config persistence

        """
        super().__init__(
            expected_layout="mono",
            output_sample_rate=ELEVENLABS_SAMPLE_RATE,
            input_sample_rate=ELEVENLABS_SAMPLE_RATE,
        )

        self.deps = deps
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path

        self._input_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.output_queue: asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs] = asyncio.Queue()

        self._conversation: Conversation | None = None
        self._audio_interface: ReachyAudioInterface | None = None
        self._client: ElevenLabs | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._shutdown_requested: bool = False

        self.last_activity_time = 0.0
        self.start_time = 0.0
        self.is_idle_tool_call = False

    def copy(self) -> "ElevenLabsRealtimeHandler":
        """Create a copy of the handler for stream instantiation."""
        return ElevenLabsRealtimeHandler(self.deps, self.gradio_mode, self.instance_path)

    async def start_up(self) -> None:
        """Initialize ElevenLabs client and start conversation session."""
        self._loop = asyncio.get_event_loop()
        self.start_time = self._loop.time()
        self.last_activity_time = self.start_time

        api_key = config.ELEVENLABS_API_KEY
        if not api_key:
            logger.error("ELEVENLABS_API_KEY not configured")
            raise ValueError("ELEVENLABS_API_KEY required")

        agent_id = get_elevenlabs_agent_id(config.REACHY_MINI_CUSTOM_PROFILE)

        self._client = ElevenLabs(api_key=api_key)

        # Create audio interface with HeadWobbler for speech-reactive head movement
        self._audio_interface = ReachyAudioInterface(
            sample_rate=ELEVENLABS_SAMPLE_RATE,
            head_wobbler=self.deps.head_wobbler,
        )
        self._audio_interface.set_queues(self._input_queue, self.output_queue, self._loop)

        client_tools = ClientTools()
        self._register_tools(client_tools)

        self._conversation = Conversation(
            client=self._client,
            agent_id=agent_id,
            requires_auth=True,
            audio_interface=self._audio_interface,
            client_tools=client_tools,
            callback_agent_response=self._on_agent_response,
            callback_agent_response_correction=self._on_agent_response_correction,
            callback_user_transcript=self._on_user_transcript,
            callback_latency_measurement=self._on_latency_measurement,
        )

        self._conversation.start_session()
        logger.info("ElevenLabs conversation session started with agent: %s", agent_id[:8] + "...")

    async def receive(self, frame: Tuple[int, NDArray[Any]]) -> None:
        """Receive audio frame from microphone and queue for ElevenLabs.

        Args:
            frame: Tuple of (sample_rate, audio_data)

        """
        if self._shutdown_requested:
            return

        input_sample_rate, audio_frame = frame

        audio_bytes = convert_audio_for_elevenlabs(
            audio_frame,
            input_sample_rate,
            ELEVENLABS_SAMPLE_RATE,
        )

        await self._input_queue.put(audio_bytes)

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit audio frame or transcript to be processed by the stream.

        Returns:
            Audio tuple, AdditionalOutputs with transcript, or None on timeout

        """
        if self._shutdown_requested:
            return None

        # Check for idle behavior trigger
        if self._loop:
            idle_duration = self._loop.time() - self.last_activity_time
            if idle_duration > IDLE_TIMEOUT_S and self.deps.movement_manager.is_idle():
                await self._trigger_idle_behavior()
                self.last_activity_time = self._loop.time()

        return await wait_for_item(self.output_queue)

    def shutdown(self) -> None:
        """Shutdown the handler and end conversation session."""
        self._shutdown_requested = True

        if self._conversation:
            try:
                self._conversation.end_session()
            except Exception as e:
                logger.debug("Error ending conversation session: %s", e)
            finally:
                self._conversation = None

        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        logger.info("ElevenLabs handler shutdown complete")

    async def apply_personality(self, profile: str | None) -> str:
        """Apply a new personality by switching to a different ElevenLabs agent.

        Args:
            profile: Profile name to switch to, or None for default

        Returns:
            Status message

        """
        from reachy_mini_conversation_app.config import set_custom_profile

        set_custom_profile(profile)

        try:
            new_agent_id = get_elevenlabs_agent_id(profile)
        except ValueError as e:
            return f"Failed to apply personality: {e}"

        if self._conversation:
            try:
                self._conversation.end_session()
            except Exception as e:
                logger.warning("Error ending previous session: %s", e)
            self._conversation = None

        if not self._client or not self._loop:
            return "Personality applied. Will take effect on next connection."

        self._audio_interface = ReachyAudioInterface(
            sample_rate=ELEVENLABS_SAMPLE_RATE,
            head_wobbler=self.deps.head_wobbler,
        )
        self._audio_interface.set_queues(self._input_queue, self.output_queue, self._loop)

        client_tools = ClientTools()
        self._register_tools(client_tools)

        self._conversation = Conversation(
            client=self._client,
            agent_id=new_agent_id,
            requires_auth=True,
            audio_interface=self._audio_interface,
            client_tools=client_tools,
            callback_agent_response=self._on_agent_response,
            callback_agent_response_correction=self._on_agent_response_correction,
            callback_user_transcript=self._on_user_transcript,
            callback_latency_measurement=self._on_latency_measurement,
        )

        self._conversation.start_session()
        logger.info("Switched to ElevenLabs agent: %s", new_agent_id[:8] + "...")

        return f"Applied personality '{profile or 'default'}' with new agent."

    def _register_tools(self, client_tools: ClientTools) -> None:
        """Register tool handlers with ElevenLabs ClientTools.

        Args:
            client_tools: ElevenLabs ClientTools instance

        """
        for tool_spec in get_tool_specs():
            tool_name = tool_spec["name"]

            def create_handler(name: str) -> Any:
                async def handler(params: dict[str, Any]) -> Any:
                    args_json = json.dumps(params)
                    try:
                        result = await dispatch_tool_call(name, args_json, self.deps)
                        logger.info("Tool '%s' executed successfully", name)

                        self._emit_tool_result(name, result)

                        if self.deps.head_wobbler is not None:
                            self.deps.head_wobbler.reset()

                        return result
                    except Exception as e:
                        logger.error("Tool '%s' failed: %s", name, e)
                        return {"error": str(e)}

                return handler

            client_tools.register(tool_name, create_handler(tool_name), is_async=True)
            logger.debug("Registered tool: %s", tool_name)

    def _emit_tool_result(self, tool_name: str, result: dict[str, Any]) -> None:
        """Emit tool result to output queue.

        Args:
            tool_name: Name of the tool that was called
            result: Tool execution result

        """
        if not self._loop:
            return

        def put_result() -> None:
            try:
                self.output_queue.put_nowait(
                    AdditionalOutputs(
                        {
                            "role": "assistant",
                            "content": json.dumps(result),
                            "metadata": {"title": f"Used tool {tool_name}", "status": "done"},
                        }
                    )
                )
            except asyncio.QueueFull:
                logger.warning("Output queue full, dropping tool result")

        self._loop.call_soon_threadsafe(put_result)

    def _on_agent_response(self, response: str) -> None:
        """Handle agent text response callback.

        Args:
            response: Agent's text response

        """
        if not self._loop:
            return

        self.last_activity_time = self._loop.time()

        # Clear listening state when agent responds
        if self.deps.movement_manager is not None:
            self.deps.movement_manager.set_listening(False)

        def put_response() -> None:
            try:
                self.output_queue.put_nowait(AdditionalOutputs({"role": "assistant", "content": response}))
            except asyncio.QueueFull:
                logger.warning("Output queue full, dropping agent response")

        self._loop.call_soon_threadsafe(put_response)
        logger.debug("Agent response: %s", response[:100] if len(response) > 100 else response)

    def _on_agent_response_correction(self, original: str, corrected: str) -> None:
        """Handle agent response correction callback.

        Args:
            original: Original response
            corrected: Corrected response

        """
        logger.debug("Agent response correction: %s -> %s", original[:50], corrected[:50])

    def _on_user_transcript(self, transcript: str) -> None:
        """Handle user transcript callback.

        When OpenClaw is enabled, pre-fetches the response in parallel so
        the ask_clawson tool can return immediately.

        Args:
            transcript: User's transcribed speech

        """
        if not self._loop:
            return

        self.last_activity_time = self._loop.time()

        # Reset head wobbler when user speaks (sync with user input)
        if self.deps.head_wobbler is not None:
            self.deps.head_wobbler.reset()

        # Set listening state on movement manager
        if self.deps.movement_manager is not None:
            self.deps.movement_manager.set_listening(True)

        # Pre-fetch OpenClaw response when bridge is available
        if self.deps.openclaw_bridge is not None and transcript.strip():
            self._prefetch_openclaw(transcript)

        def put_transcript() -> None:
            try:
                self.output_queue.put_nowait(AdditionalOutputs({"role": "user", "content": transcript}))
            except asyncio.QueueFull:
                logger.warning("Output queue full, dropping user transcript")

        self._loop.call_soon_threadsafe(put_transcript)
        logger.debug("User transcript: %s", transcript)

    def _prefetch_openclaw(self, transcript: str) -> None:
        """Fire an async OpenClaw request and store the future for the relay tool.

        Args:
            transcript: User's transcribed speech to send to OpenClaw

        """
        from reachy_mini_conversation_app.tools.openclaw_relay import ROBOT_SYSTEM_CONTEXT

        bridge = self.deps.openclaw_bridge

        async def _do_prefetch() -> Any:
            return await bridge.chat(
                message=transcript,
                system_context=ROBOT_SYSTEM_CONTEXT,
            )

        future = asyncio.run_coroutine_threadsafe(_do_prefetch(), self._loop)
        set_pending_response(future)
        logger.debug("OpenClaw pre-fetch started for: %s", transcript[:80])

    def _on_latency_measurement(self, latency: float) -> None:
        """Handle latency measurement callback.

        Args:
            latency: Latency in milliseconds

        """
        logger.debug("ElevenLabs latency: %.0fms", latency)

    async def _trigger_idle_behavior(self) -> None:
        """Trigger idle behavior by directly executing a dance or emotion tool.

        Unlike OpenAI Realtime which can inject text prompts to the agent,
        ElevenLabs agents are dashboard-configured without mid-conversation
        injection. We directly invoke tools to keep the robot animated.
        """
        self.is_idle_tool_call = True

        # Randomly choose between dance and emotion
        idle_tools = ["dance", "play_emotion"]
        chosen_tool = random.choice(idle_tools)

        try:
            if chosen_tool == "dance":
                args = {"move": "random"}
            else:
                # Get random emotion from available list
                try:
                    from reachy_mini_conversation_app.tools.play_emotion import (
                        RECORDED_MOVES,
                        EMOTION_AVAILABLE,
                    )

                    if EMOTION_AVAILABLE and RECORDED_MOVES is not None:
                        emotions = RECORDED_MOVES.list_moves()
                        args = {"emotion": random.choice(emotions)}
                    else:
                        # Fallback to dance if emotions unavailable
                        chosen_tool = "dance"
                        args = {"move": "random"}
                except ImportError:
                    chosen_tool = "dance"
                    args = {"move": "random"}

            logger.info("Idle behavior: triggering %s with %s", chosen_tool, args)
            result = await dispatch_tool_call(chosen_tool, json.dumps(args), self.deps)
            logger.debug("Idle behavior result: %s", result)

            # Reset head wobbler after tool execution
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.reset()

        except Exception as e:
            logger.warning("Idle behavior failed: %s", e)
        finally:
            self.is_idle_tool_call = False
