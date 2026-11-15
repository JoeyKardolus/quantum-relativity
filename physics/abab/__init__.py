"""ABAB layer: the owner's foundation (bible, physics/quantum-relativity, "The foundation").

An irregular structure of energy volumes (A, pull) and joints (B, push); every A
touches only B's and every B only A's. The physics keeps only kind, energy and size
per volume and who touches whom; positions exist only for drawing (new_state hands
them back beside the State). Open, not wrapped.

Module map:
  state.py            State: kind, energy, size, rate, contacts, counts, start sizes
  settings.py         Settings: rule strengths, all_on() defaults
  new_state.py        new_state: scattered points, half A half B, nearest other-kind contacts
  step.py             step: pull, openness, swing, leak, collapse, split
  stats.py            stats: counts, size against the start, energy, lumps
  pull_potential.py   pull spread and fading through the contacts (rule 7)
  joint_squeeze.py    the squeeze on each joint from its energy volumes' pull
  joint_rest_size.py  the size a joint's energy wants
  joint_energy.py     the energy a joint needs to want a size (its inverse)
  joint_balance.py    the size where a joint's push meets its squeeze (the start)
  joint_openness.py   how much a joint lets through, 0 past the seal point
  joint_push.py       a joint's push toward its rest size, steep as it thins (rule 8)
  swing_joints.py     one undamped Verlet step of every joint's swing (rule 5)
  passages.py         every A -> B -> A passage
  leak.py             energy moves A -> B -> A by pull (rule 6)
  grow_by_energy.py   an energy volume's size follows its energy
  collapse_joints.py  joints past their limit vanish, their energy volumes fall into one
  split_volumes.py    an energy volume grown too big splits around a new joint
  lumps.py            packed groups of energy volumes and whether they sealed
"""
from physics.abab.state import KIND_A, KIND_B, State
from physics.abab.settings import Settings
from physics.abab.joint_rest_size import joint_rest_size
from physics.abab.joint_energy import joint_energy
from physics.abab.pull_potential import pull_potential
from physics.abab.joint_squeeze import joint_squeeze
from physics.abab.joint_balance import joint_balance
from physics.abab.joint_openness import joint_openness
from physics.abab.passages import passages
from physics.abab.leak import leak
from physics.abab.grow_by_energy import grow_by_energy
from physics.abab.collapse_joints import collapse_joints
from physics.abab.split_volumes import split_volumes
from physics.abab.lumps import lumps
from physics.abab.new_state import new_state
from physics.abab.step import step
from physics.abab.stats import stats

__all__ = ["KIND_A", "KIND_B", "State", "Settings", "joint_rest_size", "joint_energy", "pull_potential", "joint_squeeze",
           "joint_balance", "joint_openness", "passages", "leak", "grow_by_energy", "collapse_joints",
           "split_volumes", "lumps", "new_state", "step", "stats"]
