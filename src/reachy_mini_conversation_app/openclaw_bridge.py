"""OpenClaw Gateway bridge for AI responses.

Provides HTTP client integration with the OpenClaw gateway using the
OpenAI-compatible Chat Completions API. Supports text and multimodal
(text + image) messages, streaming, and cross-channel session context.
"""

import json
import logging
from typing import Any, Optional, AsyncIterator
from dataclasses import dataclass

import httpx
from httpx_sse import aconnect_sse


logger = logging.getLogger(__name__)


@dataclass
class OpenClawResponse:
    """Response from OpenClaw gateway."""

    content: str
    error: Optional[str] = None


class OpenClawBridge:
    """Bridge to OpenClaw Gateway using HTTP Chat Completions API.

    Sends user messages to OpenClaw and receives AI responses.
    The gateway maintains conversation context and can include images.

    Example:
        bridge = OpenClawBridge(gateway_url="http://localhost:18789", agent_id="main")
        await bridge.connect()

        response = await bridge.chat("Hello!")
        print(response.content)

        # With image
        response = await bridge.chat("What do you see?", image_b64="...")

    """

    def __init__(
        self,
        gateway_url: str = "http://localhost:18789",
        gateway_token: str = "",
        agent_id: str = "main",
        session_key: str = "main",
        timeout: float = 120.0,
    ) -> None:
        """Initialize the OpenClaw bridge.

        Args:
            gateway_url: URL of the OpenClaw gateway
            gateway_token: Authentication token for the gateway
            agent_id: OpenClaw agent ID to use
            session_key: Session key for cross-channel context sharing
            timeout: Request timeout in seconds

        """
        self.gateway_url = gateway_url
        self.gateway_token = gateway_token
        self.agent_id = agent_id
        self.session_key = session_key
        self.timeout = timeout
        self._connected = False

    async def connect(self) -> bool:
        """Test connection to the OpenClaw gateway.

        Returns:
            True if connection successful, False otherwise

        """
        logger.info(
            "Attempting to connect to OpenClaw at %s (token: %s)",
            self.gateway_url,
            "set" if self.gateway_token else "not set",
        )
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                url = f"{self.gateway_url}/v1/chat/completions"
                logger.info("Testing endpoint: %s", url)
                response = await client.post(
                    url,
                    json={
                        "model": f"openclaw:{self.agent_id}",
                        "messages": [{"role": "user", "content": "ping"}],
                    },
                    headers=self._get_headers(),
                )
                logger.info("Response status: %d", response.status_code)
                if response.status_code == 200:
                    self._connected = True
                    logger.info("Connected to OpenClaw gateway at %s", self.gateway_url)
                    return True
                else:
                    logger.warning(
                        "OpenClaw gateway returned %d: %s",
                        response.status_code,
                        response.text[:100],
                    )
                    self._connected = False
                    return False
        except Exception as e:
            logger.error("Failed to connect to OpenClaw gateway: %s (type: %s)", e, type(e).__name__)
            self._connected = False
            return False

    def _get_headers(self) -> dict[str, str]:
        """Get headers for OpenClaw API requests."""
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "x-openclaw-session-key": f"agent:{self.agent_id}:{self.session_key}",
        }
        if self.gateway_token:
            headers["Authorization"] = f"Bearer {self.gateway_token}"
        return headers

    async def chat(
        self,
        message: str,
        image_b64: Optional[str] = None,
        system_context: Optional[str] = None,
    ) -> OpenClawResponse:
        """Send a message to OpenClaw and get a response.

        OpenClaw maintains conversation memory on its end, so it will be aware
        of conversations from other channels (WhatsApp, web, etc.). We only send
        the current message and let OpenClaw handle the context.

        Args:
            message: The user's message (transcribed speech)
            image_b64: Optional base64-encoded image from robot camera
            system_context: Optional additional system context

        Returns:
            OpenClawResponse with the AI's response

        """
        content: str | list[dict[str, Any]]
        if image_b64:
            content = [
                {"type": "text", "text": message},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ]
        else:
            content = message

        request_messages: list[dict[str, Any]] = []
        if system_context:
            request_messages.append({"role": "system", "content": system_context})
        request_messages.append({"role": "user", "content": content})

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout)) as client:
                response = await client.post(
                    f"{self.gateway_url}/v1/chat/completions",
                    json={
                        "model": f"openclaw:{self.agent_id}",
                        "messages": request_messages,
                        "stream": False,
                    },
                    headers=self._get_headers(),
                )
                response.raise_for_status()

                data = response.json()
                choices = data.get("choices", [])
                if choices:
                    assistant_content = choices[0].get("message", {}).get("content", "")
                    return OpenClawResponse(content=assistant_content)
                return OpenClawResponse(content="", error="No response from OpenClaw")

        except httpx.HTTPStatusError as e:
            logger.error("OpenClaw HTTP error: %d - %s", e.response.status_code, e.response.text[:200])
            return OpenClawResponse(content="", error=f"HTTP {e.response.status_code}")
        except Exception as e:
            logger.error("OpenClaw chat error: %s", e)
            return OpenClawResponse(content="", error=str(e))

    async def stream_chat(
        self,
        message: str,
        image_b64: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream a response from OpenClaw.

        OpenClaw maintains conversation memory on its end, so it will be aware
        of conversations from other channels (WhatsApp, web, etc.).

        Args:
            message: The user's message
            image_b64: Optional base64-encoded image

        Yields:
            String chunks of the response as they arrive

        """
        content: str | list[dict[str, Any]]
        if image_b64:
            content = [
                {"type": "text", "text": message},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ]
        else:
            content = message

        request_messages: list[dict[str, Any]] = [{"role": "user", "content": content}]

        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout)) as client:
            try:
                async with aconnect_sse(
                    client,
                    "POST",
                    f"{self.gateway_url}/v1/chat/completions",
                    json={
                        "model": f"openclaw:{self.agent_id}",
                        "messages": request_messages,
                        "stream": True,
                    },
                    headers=self._get_headers(),
                ) as event_source:
                    event_source.response.raise_for_status()

                    async for sse in event_source.aiter_sse():
                        if sse.data == "[DONE]":
                            break

                        try:
                            data = json.loads(sse.data)
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                chunk = delta.get("content", "")
                                if chunk:
                                    yield chunk
                        except json.JSONDecodeError:
                            continue

            except httpx.HTTPStatusError as e:
                logger.error("OpenClaw streaming error: %d", e.response.status_code)
                yield f"[Error: HTTP {e.response.status_code}]"
            except Exception as e:
                logger.error("OpenClaw streaming error: %s", e)
                yield f"[Error: {e}]"

    @property
    def is_connected(self) -> bool:
        """Check if bridge is connected to gateway."""
        return self._connected
