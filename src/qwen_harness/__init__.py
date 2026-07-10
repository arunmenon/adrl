"""qwen_harness — a study re-implementation of the qwen-code agent harness.

This package ports the core architecture of qwen-code (QwenLM/qwen-code,
Apache-2.0, itself a fork of google-gemini/gemini-cli) to readable Python,
so the harness mechanics can be studied and instrumented alongside the rest
of this repo's routing work.

Pinned upstream: qwen-code v0.19.8, commit 7ad2a5f (2026-07-10).

Layer map (Python module -> upstream TypeScript source):

    types.py              packages/core/src/core/turn.ts (Content/Part shapes,
                          ServerGeminiEventType) + @google/genai types
    config.py             packages/core/src/config/config.ts (ApprovalMode, Config)
    token_limits.py       packages/core/src/core/tokenLimits.ts
    environment.py        packages/core/src/utils/environmentContext.ts,
                          getFolderStructure.ts
    memory.py             packages/core/src/utils/memoryDiscovery.ts
    prompts.py            packages/core/src/core/prompts.ts
    content_generator.py  packages/core/src/core/contentGenerator.ts +
                          openaiContentGenerator/ (Gemini<->OpenAI conversion,
                          streaming, retry)
    chat.py               packages/core/src/core/geminiChat.ts
    turn.py               packages/core/src/core/turn.ts
    client.py             packages/core/src/core/client.ts (sendMessageStream
                          recursion, compression trigger, next-speaker)
    tools/base.py         packages/core/src/tools/tools.ts
    tools/registry.py     packages/core/src/tools/tool-registry.ts
    scheduler.py          packages/core/src/core/coreToolScheduler.ts
    tools/*.py            packages/core/src/tools/<tool>.ts (one file per tool)
    services/loop_detection.py   packages/core/src/services/loopDetectionService.ts
    services/next_speaker.py     packages/core/src/utils/nextSpeakerChecker.ts
    services/compression.py      packages/core/src/services/chatCompressionService.ts
    cli.py                packages/cli/src/nonInteractiveCli.ts + a minimal REPL

See docs/qwen-code-harness-study.md for the guided tour.
"""

__version__ = "0.1.0"
UPSTREAM = "qwen-code v0.19.8 @ 7ad2a5f"
