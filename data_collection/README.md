# Data Collection 

## Goal
To learn a feature extractor for the state space that maps from (S,A) -> pref.

We're doing this in a supervised fashion, meaning that we need to collect labels for different trajectories. For each trajectory, we can label a particular play style. These play styles need to be sufficiently differentiated in order for us to capture the relationship between playstyles accurately.

## Format
use pickle for serializing/deserializing the following dictionary. All arrays are numpy.
`N` := `# of rollouts X rollout_lengths (total number of states seen by playing)`

```
starpilot.pkl

{
    "state": [ N x |S| ],
    "action": [ N x |A| ],
    ... etc,
    "label": [ N x {0, 1} (passive or aggressive, correspondingly) ]
}
```

## Output
    - (State, action, reward, terminal) sequence for each rollout.
    - Label for user preference (e.g. `"aggressive"`, or `"passive"`)

## Assignments
    - Brian : StarPilot (procgen) -> ./data/starpilot
    - Ashwin: Plunder (procgen) -> ./data/plunder
    - Korn: Berzerk (gym[atari]) -> ./data/berzerk
