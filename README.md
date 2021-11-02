# GymSlug
This is a registered custom gym environment for reinforcement learning toward bio-inspired, explainable control of Aplysia californica feeding via motor neuron control. Please refer to [pending] for details.

# usage.ipynb provides an example of a complete training routine using DQN

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CMU-BORG/GymSlug/blob/main/usage.ipynb)

# Basic usage below:
# Installation
1. unzip GymSlug.zip
2. cd GymSlug
3. pip install -e .
# Obtain expert performance under unbreakable seaweed scenario (as benchmark/goal for the reinforcement agent)
```
from aplysia_feeding_ub import AplysiaFeedingUB
from datetime import date

suffix = str(date.today())
xlimits = [0,60]
aplysia = AplysiaFeedingUB() # change to aplysia = AplysiaFeedingB() for breakable seaweed scenario
aplysia.SetSensoryStates('swallow')
aplysia.RunSimulation()
aplysia.GeneratePlots('Swallow_'+suffix,xlimits)
```
# Create a new instance of the GymSlug environment
```
env = gym.make("gym_slug:slug-v0") # for unbreakable seaweed
env = gym.make("gym_slug:slug-v1") # for breakable seaweed (variable seaweed strength)
```
