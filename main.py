__author__ = 'Caleytown'

import gzip
import numpy as np
from PIL import Image
from RobotClass import Robot
from GameClass import Game
from GreedyNavigator import GreedyNavigator
from InformedNavigator import InformedNavigator
from networkFolder.functionList import Map, WorldEstimatingNetwork, DigitClassificationNetwork
import time

# Create a Map Class Object
map = Map()

# Get the current map from the Map Class
data = map.map
# map.getNewMap()

# Print the number of the current map
print('map number:', map.number)

# Create a Robot that starts at (0,0)
# The Robot Class stores the current position of the robot
# and provides ways to move the robot 
robot = Robot(0, 0)

# ***********************************************************************
# *********** CHANGE NAVIGATOR FROM GREEDY TO INFORMED HERE *************
# ***********************************************************************
# Choose which navigator to use
which_navigator = 1  # 0 = greedy_navigator; 1 = informed_navigator
plot_trajectories = False

if which_navigator == 0:
    navigator = GreedyNavigator()
    plot_title = 'Greedy Navigator trajectory'
else:
    navigator = InformedNavigator()
    plot_title = 'Informed Navigator trajectory'

# Create a Game object, providing it with the map data, the goal location of the map, the navigator, and the robot
game = Game(data, map.number, navigator, robot)

# This loop runs the game for 1000 ticks, stopping if a goal is found.
all_scores = []
all_times = []
for trial in range(10):
    start_time = time.time()
    map.getNewMap()
    data = map.map
    navigator.reset()
    robot.resetRobot()
    game = Game(data, map.number, navigator, robot)
    for x in range(0, 1000):
        found_goal = game.tick()
        print(f"{game.getIteration()}: Robot at: {robot.getLoc()}, Score = {game.getScore()}")
        if found_goal:
            print(f"Found goal at time step: {game.getIteration()}!")
            if plot_trajectories:
                game.plot_path(plot_title)
            break
    print(f"Final Score: {game.score}")
    end_time = time.time()
    all_scores.append(game.score)
    all_times.append(end_time - start_time)
print(f"Average Score greedy navigator: {np.mean(np.array(all_scores))}")
print(f"Average Running time: {np.mean(np.array(all_times))}")

