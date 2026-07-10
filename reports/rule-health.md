# rule health - do the hand-heuristics earn their keep?

- db: `data/router-memory.db`
- source pool: **organic**
- closed turns analysed: 754
- base hard-rate: 16.3% (123/754 turns went hard)

Lift = P(hard | rule fired) - base rate. Easy-leaning rules want negative lift; hard-leaning want positive.

| rule | leaning | fired | fire% | hard-rate | lift | verdict |
|------|---------|------:|------:|----------:|-----:|---------|
| verb:trivial | easy | 23 | 3.1% | 22% | +5% | DEMOTE-CANDIDATE |
| context:big | hard | 574 | 76.1% | 18% | +2% | WEAK-SIGNAL |
| scope:narrow | easy | 75 | 9.9% | 15% | -2% | WEAK-SIGNAL |
| verb:unknown | neutral | 655 | 86.9% | 16% | -0% | OK |
| verb:explain | easy | 47 | 6.2% | 6% | -10% | OK |
| verb:fix | neutral | 17 | 2.3% | 53% | +37% | INSUFFICIENT |
| retry_signal | hard | 17 | 2.3% | 12% | -5% | INSUFFICIENT |
| terse_continuation | neutral | 11 | 1.5% | 27% | +11% | INSUFFICIENT |
| verb:write | neutral | 8 | 1.1% | 12% | -4% | INSUFFICIENT |
| scope:broad | hard | 8 | 1.1% | 12% | -4% | INSUFFICIENT |
| traj:recent_errors | hard | 5 | 0.7% | 0% | -16% | INSUFFICIENT |
| verb:small_edit | neutral | 3 | 0.4% | 33% | +17% | INSUFFICIENT |
| verb:hard | hard | 1 | 0.1% | 0% | -16% | INSUFFICIENT |
| traj:edit_failures | hard | 1 | 0.1% | 0% | -16% | INSUFFICIENT |

## demote candidates (human review - no auto-demote in v1)
- **verb:trivial** - easy-leaning but hard-rate 22% exceeds base 16% (lift +5%) - anti-signal

_Insufficient sample (< 20 fires), no verdict yet: verb:fix, retry_signal, terse_continuation, verb:write, scope:broad, traj:recent_errors, verb:small_edit, verb:hard, traj:edit_failures._
