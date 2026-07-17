# Shark Tank NN — Design

## Overview

Browser-based aquarium simulation. A shark hunts a population of fish. Both are
controlled by neural networks evolved via genetic algorithms (neuroevolution),
not hand-scripted AI. The simulation runs in 6 stages — each stage gives the
fish NN more sensory inputs (walls → shark distance → shark angle → schooling
→ predictive), visibly changing behavior from random flailing to coordinated
evasion. The shark's NN co-evolves against the fish the whole time.

Purpose: visual/educational demo of how sensory inputs and network depth
change emergent behavior in neuroevolution, with a live view into the network
itself (not just the agents).

## Stack

- Plain HTML/JS + Canvas 2D for rendering
- Plain JS arrays/matrices for NN forward passes (no TF.js) — networks are
  tiny (few inputs, 1-2 hidden layers), and GA needs direct weight-array
  access for clone/mutate/crossover. Zero runtime dependency, trivially
  unit-testable with Node's built-in test runner.
- No build step required (single-page app, script tags), unless dependency
  management calls for a minimal bundler later — start without one

## Modules

- `nn.js` — feedforward NN using plain JS arrays (no tensor library).
  Configurable input size per stage. Exposes weight get/set for GA
  operations (clone, mutate, crossover).
- `genetic.js` — generic GA engine: fitness-weighted selection, crossover,
  gaussian mutation. Used for both the fish population and the shark
  population.
- `sim.js` — episode loop: agent positions/velocities, wall collision, shark
  eats fish on contact, per-tick fitness accumulation.
- `stages.js` — stage definitions (1–6), each an input schema for the fish
  NN. Stage switch rebuilds the fish input layer, keeps old weights for
  inputs that still exist in the new schema (Xavier-inits the new ones),
  and resets the fish population + generation counter for the new stage.
- `render.js` — aquarium canvas: procedural fish/shark art (bezier body +
  fins/tail, gradient shading, tail-wag/fin-flap driven by velocity
  magnitude, per-fish hue jitter for variety). Shark rendered larger, darker,
  more angular fins.
- `nnviz.js` — node-link diagram canvas for the currently-selected agent
  (click a fish or the shark to select). Nodes/edges colored or pulsed by
  live activation value, redrawn every frame.
- `ui.js` — "Next Stage" button (manual advance, user-controlled pacing),
  speed control (1x / 4x / max, max skips rendering for fast generations),
  HUD: stage #, generation #, alive count, best fitness this generation,
  running/paused state.

## Agents & Neuroevolution

### Fish

- Population: 30 fish, all alive simultaneously in one aquarium per episode.
- NN: feedforward, 1 hidden layer (2 layers from stage 6), output =
  [turn, thrust].
- Fitness: ticks survived + small bonus for time-integrated distance from
  shark.
- GA: elitism top 3 unchanged; rest bred via fitness-weighted roulette
  crossover (2 parents) + gaussian mutation (~10% rate, small sigma).

### Shark

- Population: 8 genomes, evolved in the background — one genome is "active"
  (rendered, hunting) per episode; the pool rotates/evolves once per
  generation using the same elitism/crossover/mutation scheme as fish.
- Senses: full vision from stage 1 onward — nearest-fish distance + angle
  (adds nearest-fish velocity vector at stage 6, matching fish's predictive
  stage so the arms race stays fair). Shark senses do NOT ramp with stage;
  only fish senses ramp.
- Fitness: fish-eaten count + bonus for closing distance over the episode
  (avoids sparse-reward stalls when the shark starves an entire episode).

## Stage Ladder (fish NN input schema)

| Stage | Inputs added | Resulting behavior |
|-------|---------------|---------------------|
| 1 | none (or noise) | fully random movement |
| 2 | distance to 4 walls | avoids walls/corners |
| 3 | + distance to shark | some avoidance, no direction |
| 4 | + angle to shark | directional fleeing |
| 5 | + avg position/heading of N nearest fish | schooling/clustering emerges |
| 6 | + shark velocity vector, 2nd hidden layer | predictive evasion |

Stage advance is manual (user clicks "Next Stage"). On advance: fish input
layer is rebuilt for the new schema, matching old weights are kept, new
weights are Xavier-initialized, fish population and generation counter reset
for the new stage. Shark population and its NN size are unaffected by fish
stage changes (shark schema only changes at stage 6 to add the matching
velocity input).

## Episode Mechanics

- One episode = one generation. Fixed max duration (~20s sim time) or until
  all fish are eaten, whichever comes first.
- Eaten fish are removed from rendering; their fitness is frozen at the tick
  of death. No mid-episode respawn.
- Speed control lets the user run at 1x (real time), 4x, or max (skip
  rendering, computed as fast as possible) to fast-forward through
  generations.

## Visualization

- **Aquarium canvas**: procedural fish/shark art as described above (no
  external image-generation tool available in this environment — confirmed
  via tool search, only a design-system sync tool exists and doesn't apply
  here).
- **NN diagram canvas**: live node-link diagram of the selected agent's
  brain, activations redrawn every frame.
- **HUD**: stage, generation, alive count, best fitness, running/paused,
  speed — essential for using the manual stage-advance flow, folded into the
  main UI rather than treated as a separate feature.

## Out of scope (v1)

- Reinforcement learning path (neuroevolution only, per decision — may
  revisit as an alternate training mode later)
- Multiple simultaneous sharks in one episode
- Persisting/exporting trained genomes across browser sessions
