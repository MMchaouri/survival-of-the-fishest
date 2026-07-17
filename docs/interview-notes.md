# Survival of the Fishest — how I'd explain this in an interview

## The pitch (30 seconds)

A browser aquarium where a shark and a school of fish are both controlled by
neural networks, and neither network is hand-programmed — they evolve
through genetic algorithms (neuroevolution) instead of being trained with
labeled data. The interesting part isn't the game, it's the six-stage
sensory ladder: the fish start with zero inputs (pure random movement) and
each stage adds one more sense (walls, then shark distance, then shark
angle, then nearby fish, then predictive velocity). You can watch, stage by
stage, how much a network's behavior depends on what it's even allowed to
perceive — the same tiny brain goes from flailing randomly to fleeing with
foresight, purely because its input vector got richer.

## Why this is a good interview topic

- It's a self-contained ML systems project you can demo live, not just describe.
- It touches four skill areas at once: ML fundamentals (neuroevolution,
  fitness design, coevolution), software architecture (clean module
  boundaries), front-end/graphics (canvas rendering, live data viz), and
  disciplined engineering process (spec-first, TDD, structured review).
- The design decisions have real trade-offs you can defend, not just
  "it worked."

## The process, and why it matters

I didn't start writing code. The sequence was:

1. **Brainstorm** — turned a one-paragraph idea ("shark eats fish, fish learn
   to see the shark") into a fully specified design through back-and-forth:
   what controls what, how learning happens, how stages progress, what the
   visualization shows. Several real decisions got made here, not left
   implicit:
   - Neuroevolution over reinforcement learning — simpler to reason about
     live in a browser, fitness = survival time / eaten count, no reward
     shaping headaches.
   - Both shark and fish get NN brains (coevolution), not just the fish —
     makes it an arms race, not a static obstacle course.
   - Plain JS arrays for the network math instead of TensorFlow.js — the
     networks are tiny (a handful of inputs, 1-2 hidden layers), and the GA
     needs direct read/write access to raw weight arrays for
     clone/mutate/crossover. Pulling in a tensor library would have added a
     dependency and a conversion layer for zero benefit at this scale. This
     is the kind of "the obvious tech choice isn't always the right one for
     this specific size of problem" call that's worth explaining out loud.
2. **Write a design spec** — a real markdown doc capturing every decision
   above, committed before any implementation, so there's a reference of
   record for *why*, not just *what*.
3. **Write an implementation plan** — broke the spec into 9 tasks, each with
   exact file paths, exact function signatures, and (for the logic-bearing
   modules) complete test code written test-first. The plan is the contract:
   an engineer picking up any single task doesn't need the rest of the
   conversation to know what to build.
4. **Execute task-by-task with review gates** — each task went to a fresh
   implementer with no memory of the rest of the project (just its task
   brief), then a fresh reviewer checked it for spec compliance and code
   quality before the next task started. This mirrors how you'd want a real
   team to work: small reviewable diffs, nothing merges without a second set
   of eyes, and the reviewer explicitly does NOT trust the implementer's own
   claims — it re-derives correctness from the diff.

That process is itself the answer to "how do you approach ambiguous
requirements" — you'll have a real, recent example instead of a hypothetical.

## Architecture, in one paragraph

Six pure-logic modules with zero UI coupling (`nn.js`, `genetic.js`,
`stages.js`, `sim.js`) are unit tested with Node's built-in test runner —
no framework, no mocks, real assertions on real math. Three
rendering/interaction modules (`render.js`, `nnviz.js`, `ui.js`) own the
canvas and DOM and are verified visually, because there's nothing meaningful
to unit-test in a `ctx.fillRect` call. `main.js` is the only file that knows
about all of them — it's the composition root. That split matters: you can
explain the genetic algorithm to someone without them needing to know a
single thing about canvas gradients, and vice versa.

## The core ML idea, precisely

- **Genome = a network's weights.** `cloneNN`/`mutateNN`/`crossoverNN` in
  `nn.js` operate directly on those arrays.
- **Generic GA engine** (`genetic.js`) doesn't know what a "fish" or "shark"
  is — it takes a population, a fitness array, and injected
  clone/crossover/mutate functions. The same 30-line engine evolves both
  populations. This is a real reuse decision, not just DRY for its own sake:
  it means the fish and shark's evolution can never accidentally diverge in
  behavior because of copy-pasted-then-drifted logic.
- **Fitness functions are asymmetric on purpose.** Fish fitness rewards
  survival time plus distance from the shark (evasion). Shark fitness
  rewards eating, plus a small bonus for *closing* distance even without a
  kill — otherwise a shark that never catches anything in an episode gets
  zero signal and evolution stalls (a classic sparse-reward problem, solved
  here without RL machinery).
- **Stage transitions preserve partial knowledge.** When a fish population
  moves to a stage with more inputs, `resizeInputLayer` keeps the weights
  for inputs that still exist and Xavier-initializes only the new ones —
  instead of starting evolution over from scratch every stage. That's the
  detail that makes the "gradual sense ladder" premise actually work instead
  of just being 6 disconnected demos.

## Strengths a reviewer independently called out (not my own claims)

Each implementation task went through an independent review pass that
verified the code by reading it, not by trusting the report. A few things
worth repeating because someone else checked them, not because I said so:

- `mutateNN`, `crossoverNN`, and `resizeInputLayer` were verified to be
  genuinely pure — the original network is provably untouched after each
  call, confirmed by reading the actual assignment lines, not just by the
  tests passing.
- The episode loop's collision handling was checked for a subtle class of
  bug (double-counting a fish's death, or continuing to award it fitness
  after it's already dead) and confirmed single-pass-safe.
- The generic GA engine's edge cases — all-zero fitness, `eliteCount`
  larger than the population — were traced by hand and confirmed to degrade
  safely instead of crashing or infinite-looping.
- Two honest limitations were surfaced by review, not hidden: wall
  "bouncing" flips velocity but not the steering angle a fish's brain is
  actually driving from, so a fish commanding forward-thrust into a wall
  will pin against it rather than visibly redirect; and a fish that's eaten
  mid-tick still banks that tick's survival fitness before freezing. Neither
  breaks anything, but being able to name your own system's rough edges is
  exactly what a good engineer should be able to do in an interview.

## What I'd say if asked "what would you improve"

- Wall bounce should redirect `angle`, not just clamp `vx`/`vy`, so evasive
  fish visibly turn off walls instead of pinning against them.
- Reinforcement learning was considered and deliberately deferred — it's a
  legitimate alternative training path (reward-driven rather than
  population-driven) that could be added later as a second mode without
  touching the neuroevolution path, since the sim/render layers don't care
  how a network's weights got set.
- No persistence yet — trained genomes vanish on page reload. Straightforward
  to add (serialize weights to JSON, localStorage) but wasn't in scope.
