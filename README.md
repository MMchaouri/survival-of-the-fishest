# Survival of the Fishest

A shark and a school of fish in a browser aquarium. Both are controlled by neural networks, and neither network is hand programmed. They evolve through a genetic algorithm, generation after generation, while you watch.

## Why I built this

I wanted a hands on way to see something that's usually just a slide in an ML talk: that a network's behavior is bounded by what it can actually perceive, not just how big it is. So the fish start with a tiny brain and zero senses. They flail around at random. Then, stage by stage, I hand them one more input: distance to the walls, then distance to the shark, then the angle to the shark, then awareness of nearby fish, then the shark's velocity so they can anticipate where it's heading. Nothing about the network's learning algorithm changes between stages. Only what it's allowed to see changes. Watching the exact same tiny brain go from swimming in circles to fleeing with something that looks like foresight, purely because its input vector got richer, is the whole point of this project.

The shark evolves too, in parallel, hunting whatever strategy the fish population currently has. So it's not a static obstacle course, it's an actual arms race.

## How it works

Every fish and the shark run on a small feedforward network (an input layer, one or two hidden layers, and two outputs: turn and thrust). None of them are trained with labeled data or backpropagation. Instead I use neuroevolution: a population of networks, a fitness score per episode, and a genetic algorithm that keeps the best performers, breeds the rest through crossover, and mutates the offspring. Repeat every ~20 seconds (or until every fish is eaten) and the population gets a little better at whatever its fitness function rewards.

Fish fitness rewards staying alive and staying far from the shark. Shark fitness rewards eating fish, plus a smaller reward just for closing the distance, so a shark that goes a whole episode without catching anything still gets a signal to learn from instead of flatlining at zero.

The shark has its own population of 8 genomes evolving in the background. Only one is active in the tank at a time; the pool sweeps through all 8 before evolving, so there's real selection pressure across the population instead of just one lineage randomly drifting.

When you click "Next Stage," the fish's input layer gets rebuilt for the new sense, but the weights connecting the senses it already had carry over. Only the brand new input gets randomly initialized. So a fish that partially learned to avoid the shark by distance doesn't have to relearn that from scratch the moment it also gets an angle to work with.

## Technical choices, and why

**Plain JavaScript and Canvas, no framework.** This runs as a single page with no build step. I wanted anyone to be able to clone it and open index.html, not npm install their way into a working demo.

**Plain arrays for the neural network, not TensorFlow.js.** These networks are small: a handful of inputs, one or two hidden layers of 8 neurons, two outputs. Pulling in a tensor library would have added a dependency and a conversion layer between weight arrays (which the genetic algorithm needs direct access to for cloning, mutating, and crossover) and tensors, for zero real benefit at this scale. Plain arrays of numbers are also trivial to unit test with nothing but Node's built in test runner, no mocking required.

**Neuroevolution instead of reinforcement learning.** RL is the more "proper" way to train an agent against a reward signal, but it comes with a lot of machinery (replay buffers, reward shaping, hyperparameter sensitivity) that would have buried the actual point of this project, which is about sensory inputs, not training algorithms. A genetic algorithm is simple enough to explain in one sentence and still produces visibly improving behavior over generations.

**Coevolution instead of a scripted predator.** I could have made the shark a simple "chase the nearest fish" script and only evolved the fish. I didn't, because a fixed predator is a fixed target: the fish population would eventually just solve it and plateau. Having the shark evolve too keeps both sides honest.

## Project layout

- `src/nn.js`: the network itself. Create one, run a forward pass, clone it, mutate it, breed two together, or resize its input layer when a stage changes.
- `src/genetic.js`: the genetic algorithm. It knows nothing about neural networks specifically, it just takes a population, a fitness array, and functions for cloning/breeding/mutating whatever the population is made of. That's how the same code evolves both the fish and the shark.
- `src/stages.js`: the six stage definitions and the functions that turn "a fish, the shark, the walls" into the actual numeric input vector for a given stage.
- `src/sim.js`: the physics tick. Movement, wall avoidance, collisions, fitness accumulation.
- `src/render.js` and `src/nnviz.js`: the drawing code, fish and shark art, and the live node diagram of whichever agent's brain you've clicked on.
- `src/ui.js` and `src/main.js`: the controls (next/previous stage, pause, speed, mode toggle, speed sliders) and the code that wires everything above into a running loop.

## Running it

No build step. From the project folder:

```
python3 -m http.server 8080
```

Then open `http://localhost:8080` in a browser.

## Running the tests

The pure logic (the network math, the genetic algorithm, the stage input builders, the simulation physics) is unit tested with Node's built in test runner, no dependencies needed:

```
npm test
```

The rendering and UI code isn't unit tested since there's nothing meaningful to assert about a canvas drawing call. That part I just checked by eye in the browser.

## What's still rough

Wall handling steers an agent away before it reaches the edge, but if it still ends up right at the boundary the position gets hard clamped as a last resort, which can look like a small stutter in rare cases.

There's no persistence. Close the tab and the evolved genomes are gone. Adding a save/load to localStorage would be a small, obvious next step if I keep working on this.

Reinforcement learning as an alternate training mode is something I thought about and deliberately left out for now. The simulation and rendering layers don't care how a network's weights got set, so it could be added later as a second mode without touching anything else.
