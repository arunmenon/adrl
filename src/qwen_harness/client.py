"""The session orchestrator. Port of core/client.ts (GeminiClient).

This is the heart of the harness. One user prompt becomes a *chain* of
model calls:

  send_message_stream(prompt)
      │  pre-send: compression gate (services/compression)
      │  new Turn -> stream events
      │       per event: always-on loop safeties, then opt-in heuristics
      │  Turn ends.
      ├─ pending tool calls?   -> the DRIVER (cli.py) schedules them and
      │                           calls send_message_stream(tool_responses)
      │                           — a continuation, bounded_turns - 1
      └─ no tool calls?        -> next-speaker check; verdict 'model' means
                                  recurse with the literal text
                                  'Please continue.'

The recursion is bounded by MAX_TURNS = 100 (exhaustion is silent —
returning without an event is upstream behavior), and a per-session turn
cap can end the conversation with MAX_SESSION_TURNS. History starts with
an environment-context user message; the system prompt gets the QWEN.md
memory suffix and (upstream) a frozen git snapshot.
"""

from __future__ import annotations

import itertools
import uuid
from typing import AsyncIterator

from .chat import GeminiChat
from .config import Config
from .content_generator import ContentGenerator
from .environment import get_environment_context, wrap_system_reminder
from .memory import load_hierarchical_memory
from .prompts import get_core_system_prompt
from .services.compression import ChatCompressionService, estimate_tokens
from .services.loop_detection import LoopDetectionService
from .services.next_speaker import CONTINUE_MESSAGE, check_next_speaker
from .token_limits import token_limit
from .tools.registry import ToolRegistry
from .turn import Turn
from .types import Content, GeminiEvent, GeminiEventType, Part

MAX_TURNS = 100


class GeminiClient:
    def __init__(self, config: Config, generator: ContentGenerator | None = None,
                 registry: ToolRegistry | None = None):
        self.config = config
        self.generator = generator or ContentGenerator(config.generator)
        self.registry = registry or ToolRegistry()
        self.loop_detector = LoopDetectionService(config.max_tool_calls_per_turn)
        self.compression = ChatCompressionService()
        self.session_turn_count = 0
        self.chat: GeminiChat | None = None
        self._prompt_counter = itertools.count()

    # -- session setup (startChat) --------------------------------------------

    def start_chat(self, extra_history: list[Content] | None = None) -> GeminiChat:
        user_memory, _count = load_hierarchical_memory(self.config.target_dir)
        system_instruction = get_core_system_prompt(
            user_memory=user_memory, model=self.config.model,
            cwd=self.config.target_dir)

        # history[0]: environment context. Current qwen-code sends it as a
        # single <system-reminder>-wrapped user message; classic qwen-code /
        # gemini-cli instead used the pair
        #   [user(env), model('Got it. Thanks for the context!')].
        env = wrap_system_reminder(get_environment_context(self.config.target_dir))
        history: list[Content] = [Content(role="user", parts=[Part(text=env)])]
        history.extend(extra_history or [])

        self.chat = GeminiChat(
            generator=self.generator, model=self.config.model,
            system_instruction=system_instruction, history=history,
            tools=self.registry.get_function_declarations())
        self.chat.repair_orphaned_tool_calls()
        return self.chat

    def get_chat(self) -> GeminiChat:
        if self.chat is None:
            self.start_chat()
        return self.chat

    def new_prompt_id(self) -> str:
        return f"{uuid.uuid4().hex[:8]}#{next(self._prompt_counter)}"

    # -- the loop (sendMessageStream) -------------------------------------------

    async def send_message_stream(self, request_parts: list[Part], prompt_id: str,
                                  turns: int = MAX_TURNS,
                                  is_continuation: bool = False
                                  ) -> AsyncIterator[GeminiEvent]:
        """Yields events; the final event of a healthy chain is FINISHED.
        Tool execution is the caller's job (see cli.py): when the returned
        turn left pending_tool_calls, the caller schedules them and calls
        this again with the functionResponse parts."""
        chat = self.get_chat()

        # loop-detector state spans tool-result continuations; only a fresh
        # top-level user prompt resets it
        if not is_continuation:
            self.loop_detector.reset(prompt_id)
            self.session_turn_count += 1
            if (self.config.max_session_turns > 0
                    and self.session_turn_count > self.config.max_session_turns):
                yield GeminiEvent(GeminiEventType.MAX_SESSION_TURNS)
                return

        bounded_turns = min(turns, MAX_TURNS)
        if bounded_turns <= 0:
            return  # recursion exhaustion is silent (upstream behavior)

        # pre-send compression gate
        window = token_limit(self.config.model, "input")
        estimated = (chat.last_prompt_token_count + chat.last_output_token_count
                     or estimate_tokens(chat.get_history(curated=True)))
        if self.compression.should_compress(estimated, window):
            info = await self.compression.compress(chat, self.generator,
                                                   self.config.model)
            yield GeminiEvent(GeminiEventType.CHAT_COMPRESSED, info)

        turn = Turn(chat, prompt_id)
        self._last_turn = turn
        async for event in turn.run(request_parts):
            if self.loop_detector.check_always_on_safeties(event):
                turn.pending_tool_calls.clear()
                yield GeminiEvent(GeminiEventType.LOOP_DETECTED,
                                  self.loop_detector.last_loop_type)
                return
            if not self.config.skip_loop_detection and \
                    self.loop_detector.add_and_check_heuristic_loops(event):
                yield GeminiEvent(GeminiEventType.LOOP_DETECTED,
                                  self.loop_detector.last_loop_type)
                return
            yield event
            if event.type == GeminiEventType.ERROR:
                return

        # next-speaker continuation: only when the model stopped without
        # requesting tools
        if not turn.pending_tool_calls and not self.config.skip_next_speaker_check:
            verdict = await check_next_speaker(chat, self.generator, self.config.model)
            if verdict == "model":
                async for event in self.send_message_stream(
                        [Part(text=CONTINUE_MESSAGE)], prompt_id,
                        turns=bounded_turns - 1, is_continuation=True):
                    yield event

    @property
    def pending_tool_calls(self):
        turn = getattr(self, "_last_turn", None)
        return list(turn.pending_tool_calls) if turn else []
