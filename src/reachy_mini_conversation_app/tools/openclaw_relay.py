"""OpenClaw relay tool — routes messages through OpenClaw gateway.

Acts as a client tool for ElevenLabs Voice Agent: receives the user's
transcribed speech, sends it to OpenClaw for intelligence, parses the
response for robot action keywords, dispatches robot tools (dance, emotion,
head movement), and returns cleaned text for ElevenLabs to speak.
"""

import json
import base64
import asyncio
import logging
import concurrent.futures
from typing import Any, Dict, Optional

import cv2

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# System context sent to OpenClaw with each request
ROBOT_SYSTEM_CONTEXT = (
    "You are speaking through a Reachy Mini robot body. "
    "You can express emotions through movement. If you want to dance, say 'dance' naturally. "
    "If you want to express emotions, use words like 'excited', 'curious', 'thinking'. "
    "If you want to look somewhere, say 'look left', 'look right', 'look up', or 'look down'. "
    "Keep responses conversational and concise — they will be spoken aloud."
)

# Vision-related keywords that trigger camera capture
VISION_KEYWORDS = ("look", "see", "camera", "photo", "picture", "show", "what's in front", "what do you see")


def _capture_frame_b64(deps: ToolDependencies) -> Optional[str]:
    """Capture a frame from the camera worker and encode as base64 JPEG.

    Args:
        deps: Tool dependencies with camera_worker

    Returns:
        Base64-encoded JPEG string, or None if unavailable

    """
    if deps.camera_worker is None:
        return None

    frame = deps.camera_worker.get_latest_frame()
    if frame is None:
        logger.warning("No frame available from camera worker")
        return None

    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        logger.error("Failed to encode frame as JPEG")
        return None

    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def _has_vision_keyword(message: str) -> bool:
    """Check if the message contains vision-related keywords."""
    msg_lower = message.lower()
    return any(kw in msg_lower for kw in VISION_KEYWORDS)


async def _execute_robot_actions(response_text: str, deps: ToolDependencies) -> None:
    """Parse OpenClaw response for robot action keywords and dispatch tools.

    Args:
        response_text: The response text from OpenClaw
        deps: Tool dependencies for dispatching robot tools

    """
    from reachy_mini_conversation_app.tools.core_tools import dispatch_tool_call

    response_lower = response_text.lower()

    # Head direction actions
    if any(phrase in response_lower for phrase in ("look left", "looking left", "turn left")):
        await dispatch_tool_call("move_head", json.dumps({"direction": "left"}), deps)
    elif any(phrase in response_lower for phrase in ("look right", "looking right", "turn right")):
        await dispatch_tool_call("move_head", json.dumps({"direction": "right"}), deps)
    elif any(phrase in response_lower for phrase in ("look up", "looking up")):
        await dispatch_tool_call("move_head", json.dumps({"direction": "up"}), deps)
    elif any(phrase in response_lower for phrase in ("look down", "looking down")):
        await dispatch_tool_call("move_head", json.dumps({"direction": "down"}), deps)

    # Dance / emotion actions
    if any(word in response_lower for word in ("dance", "dancing", "celebrate")):
        await dispatch_tool_call("dance", json.dumps({"move": "random"}), deps)
    elif any(word in response_lower for word in ("excited", "exciting")):
        await dispatch_tool_call("play_emotion", json.dumps({"emotion": "excited"}), deps)
    elif any(word in response_lower for word in ("thinking", "let me think", "hmm")):
        await dispatch_tool_call("play_emotion", json.dumps({"emotion": "thinking"}), deps)
    elif any(word in response_lower for word in ("curious", "interesting")):
        await dispatch_tool_call("play_emotion", json.dumps({"emotion": "curious"}), deps)


# Pre-fetched response storage (set by ElevenLabs handler on transcript callback)
_pending_response: Optional[concurrent.futures.Future[Any]] = None


def set_pending_response(future: concurrent.futures.Future[Any]) -> None:
    """Store a pre-fetched OpenClaw response future for the relay tool to await.

    Called by the ElevenLabs handler when a user transcript arrives, so the
    OpenClaw request starts before ElevenLabs decides to call ask_clawson.

    Args:
        future: Future that will resolve to an OpenClawResponse

    """
    global _pending_response
    _pending_response = future


def take_pending_response() -> Optional[concurrent.futures.Future[Any]]:
    """Take and clear the pending pre-fetched response future.

    Returns:
        The pending future if one exists, otherwise None

    """
    global _pending_response
    future = _pending_response
    _pending_response = None
    return future


class OpenclawRelay(Tool):
    """Route user messages through OpenClaw gateway for AI responses."""

    name = "ask_clawson"
    description = (
        "Send the user's message to the Clawson AI assistant and get a response. "
        "Always call this tool with the user's exact message."
    )
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The user's message to send to Clawson",
            },
        },
        "required": ["message"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Route message to OpenClaw, parse robot actions, return response text.

        Args:
            deps: Tool dependencies (must include openclaw_bridge)
            **kwargs: Must include 'message' parameter

        Returns:
            Dict with 'response' text or 'error'

        """
        message = (kwargs.get("message") or "").strip()
        if not message:
            return {"error": "message must be a non-empty string"}

        bridge = deps.openclaw_bridge
        if bridge is None:
            return {"error": "OpenClaw bridge not configured"}

        logger.info("OpenClaw relay: message=%s", message[:120])

        # Check for pre-fetched response first
        pending = take_pending_response()
        if pending is not None and not pending.cancelled():
            try:
                # run_coroutine_threadsafe returns concurrent.futures.Future;
                # wrap it so asyncio can await it
                async_future = asyncio.wrap_future(pending)
                response = await asyncio.wait_for(async_future, timeout=30.0)
                logger.info("Using pre-fetched OpenClaw response")
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception) as e:
                logger.warning("Pre-fetched response unavailable (%s), making fresh request", e)
                pending = None

        # Fresh request if no pre-fetch available
        if pending is None:
            # Capture camera frame if vision keywords detected
            image_b64: Optional[str] = None
            if _has_vision_keyword(message):
                image_b64 = _capture_frame_b64(deps)
                if image_b64:
                    logger.info("Captured camera frame for vision query")

            response = await bridge.chat(
                message=message,
                image_b64=image_b64,
                system_context=ROBOT_SYSTEM_CONTEXT,
            )

        if response.error:
            logger.error("OpenClaw error: %s", response.error)
            return {"error": f"OpenClaw error: {response.error}"}

        if not response.content:
            return {"error": "Empty response from OpenClaw"}

        # Parse and dispatch robot actions in the background
        try:
            await _execute_robot_actions(response.content, deps)
        except Exception as e:
            logger.warning("Robot action dispatch failed: %s", e)

        # Return the response text for ElevenLabs to speak
        logger.info("OpenClaw response: %s", response.content[:120])
        return {"response": response.content}
