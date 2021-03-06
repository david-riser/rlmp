"""

Use human demonstrations to create a dataset
for training imitation models.  The replay 
files are saved in a bk2 format that can be 
loaded with the retro package.  Output format
is a .csv file containing image names and actions
and a folder with images for each replay.

"""

import numpy as np
import os
import pandas as pd
import retro

if __name__ == "__main__":

    # Path to the replay data and the files contained
    # therein.  
    data_path = os.path.abspath("./data/human/")
    data_files = os.listdir(data_path)
    

    # Start processing the human replays
    for data_file in data_files:
        level = data_file.split("-")[-2]
        print("Processing level: {}".format(level))

        # Load the movie from the replay file
        movie = retro.Movie(os.path.join(data_path, data_file))
        movie.step()

        # Setup an environment for the agent to play in and
        # get rewards out from. 
        env = retro.make(
            game=movie.get_game(),
            state=retro.State.NONE,
            use_restricted_actions=retro.Actions.ALL
        )
        env.initial_state = movie.get_state()
        state = env.reset()

        # Initialize an empty set of places to store the data
        # and begin stepping through the movie frames.
        states, actions, next_states, rewards, dones = [], [], [], [], []
        while movie.step():

            # Get the actions
            keys = []
            for i in range(len(env.buttons)):
                keys.append(movie.get_key(i,0))
            action = np.array(keys, dtype=np.int8)

            # Get the update from the environment and
            # add them to the current state description. 
            next_state, reward, done, info = env.step(action)
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            
            state = next_state

        env.close()

        print("Size of actions: ", len(actions))
        
        # Output these suckers into some kind of file.
        if not os.path.exists("./data/{}".format(level)):
            os.mkdir("./data/{}".format(level))

        indices = []
        for index in range(len(dones)):
            indices.append(index)
            np.save("./data/{}/state_{}.npy".format(level, index), states[index])
            np.save("./data/{}/action_{}.npy".format(level, index), actions[index])

        dataframe = pd.DataFrame({
            "index":indices,
            "reward":rewards,
            "done":dones
        })
        dataframe.to_csv("./data/{}/meta.csv".format(level), index=False)
