# qzhRL Guarded prototype freeze

This archive freezes the 2026-08-27 post-Task3-fix RL prototype.

- Primary formal controller: the deterministic CompetitionController state
  machine. It owns task order, stage transitions, carry state, recovery and
  referee synchronization.
- Optional scheduling layer: V2 Heuristic may rank already-safe macro-action
  candidates, but it never owns the task state machine or low-level motion.
- Guarded status: optional research prototype above the same candidate layer;
  it is explicitly enabled and is never the default mode.
- Model SHA256: `5340c47b1fbcfaf799667e1b36a2474e7809817abca78e38875f690a222fb785`.
- Approval SHA256: `0f92ad4a1a0039c9dbefc54d3710aeba38910b0aaf259a443db7dd9af9a95f0a`.
- Approval benchmark: 500 blind seeds, Heuristic 241/500, RL 303/500,
  RL inference p95 3.698 ms, recovery count improved by 28.39%.
- Shadow acceptance: five sessions, 2,132 suggestions, zero runtime fallback,
  inference p95 1.900 ms, zero actual takeover as required by Shadow mode.
- Guarded canaries: seeds 20260917, 20260918 and 20260919 all scored 160.
  Seeds 20260918 and 20260919 each traced all 8 applied candidates to RL.

The deterministic state machine, hard action mask, safety checks and Heuristic
fallback remain authoritative. Disabling the scheduling extension must leave
the state-machine execution path intact. Guarded must be enabled explicitly.
