Steps to train DQN agent in Haoqin's folder:

1. Enter haoqin_code folder. Copy the DQN folder and Env folder to your own directory(e.g. lihan_code)

2. Enter the DQN folder in your own directory (e.g. lihan_code/DQN):
        change main_dqn.py line 9: 
            sys.path.insert(0, r'C:\Users\haoqi\OneDrive\Desktop\shooter-squad\haoqin_code')
        to have the absolute path of your own directory (e.g.'/home/Desktop/shooter_squad/lihan_code') on your own computer.

        change utils.py line 8:
            sys.path.insert(0, r'C:\Users\haoqi\OneDrive\Desktop\shooter-squad\haoqin_code')
        to have the absolute path of your own directory (e.g.'/home/Desktop/shooter_squad/lihan_code') on your own computer.

3. run xxx_code/DQN/main_dqn.py file (xxx is the name of the corresponding user's directory).

4. Hyperparameters of DQN agent can be found in xxx_code/DQN/main_dqn.py, line 19-31 (xxx is the name of the corresponding user's directory)

5. Under no circumstance should a user modify or run the content inside other users' folder. Just copy out what you want to your own folder and modify/run here.

_________________________________________________UPDATED instruction on 10/4/2021__________________________________________________
1. in xxx_code/DQN/utils.py, line 55, change
        for i in range(repeat):
   into:
        for i in range(1):
   The reason is because in the original implementation, an action is repeated for four times to avoid frame skipping. However, our envrionment does not have frameskipping, so we do not want to repeat an action four times.
   If we do repeat the actions, we clearly see that the agent fires bullets consequtively (three times) even though it means to fire it only once.


