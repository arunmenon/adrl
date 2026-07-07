# B5 — tool-call ID cross-provider experiment

Model: claude-haiku-4-5. Each case is a completed tool round-trip with IDs varied.

| Case | Result | Tests |
|---|---|---|
| `anthropic_native_consistent` | ACCEPT | baseline: normal IDs, matched |
| `foreign_format_consistent` | ACCEPT | OpenAI-style IDs, internally matched — does format alone matter? |
| `arbitrary_format_consistent` | ACCEPT | made-up local IDs, internally matched — provenance vs consistency |
| `mismatched_ids` | REJECT(400) | well-formed but tool_result points at a different id — internal consistency broken |

## Verdict

**IDs need only be internally consistent, not provider-minted.** Foreign-format and arbitrary IDs are accepted as long as tool_use.id == tool_result.tool_use_id; a mismatch is rejected.

**Design impact (§5.5):** the escalation rebuild does NOT need to re-mint IDs into a provider's namespace. It only needs to preserve internal pairing — which the harness's own transcript already does. The persistent ID-map machinery the design carried can be dropped; the internal review's suspicion is confirmed. Thinking-block signatures and encrypted reasoning remain the real cross-provider blockers (unchanged).
