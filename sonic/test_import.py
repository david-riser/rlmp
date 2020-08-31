""" 

Test importing the retro package
and loading the sonic environment. 

"""

import gym
import retro 


if __name__ == "__main__":

    env = retro.make(
        game='SonicTheHedgehog-Genesis',
        state='LabyrinthZone.Act1'
    )


    print("Successfully loaded Sonic The Hedgehog!")
    print("Action Space: ", env.action_space.n)
    print("Observation Space: ", env.observation_space.shape)

    state = env.reset()
    print("First state: ", state)
