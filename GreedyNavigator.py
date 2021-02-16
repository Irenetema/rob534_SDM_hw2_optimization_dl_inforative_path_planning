__author__ = 'Caleytown'

import numpy as np
from random import randint
import random
from networkFolder.functionList import Map, WorldEstimatingNetwork, DigitClassificationNetwork

# Create the world estimating network
uNet = WorldEstimatingNetwork()

# Create the digit classification network
classNet = DigitClassificationNetwork()


def get_goal(digit):
    """
        Returns a tuple containing
        - the goal location based on the digit
    """
    goals = [(0, 27), (27, 27), (27, 0)]
    if digit in range(0, 3):
        goal = goals.pop(0)
    elif digit in range(3, 6):
        goal = goals.pop(1)
    elif digit in range(6, 10):
        goal = goals.pop(2)
    else:
        raise ValueError("Bad digit input: " + str(digit))
    return goal


def compute_distance(pos1, pos2):
    which_dist = 'manhattan'

    if which_dist == 'manhattan':
        # Manhattan distance
        dist = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    else:
        # Euclidean distance
        squared_dist = (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2
        dist = np.sqrt(squared_dist)
    return dist


def entropy(p):
    # Compute the entropy of a probability distribution p
    log_p = np.log2(p)
    return - np.dot(p, log_p)


def softmax(p):
    # p: probability distribution
    return np.exp(p) / sum(np.exp(p))


def get_adjacent_states(position, image=None):
    """
    returns the adjacent states to the current position
    Args:
        image: map
        position: current position of the robot

    Returns: dictionary of coordinates (values) of adjacent states and direction leading to those states (keys)

    """
    neighbors = {}
    pos_left = [position[0] - 1, position[1]]
    pos_right = [position[0] + 1, position[1]]
    pos_up = [position[0], position[1] - 1]
    pos_down = [position[0], position[1] + 1]
    for direction, coordinates in zip(['left', 'right', 'down', 'up'], [pos_left, pos_right, pos_down, pos_up]):
        if coordinates[0] < 0 or coordinates[0] > 27 or coordinates[1] < 0 or coordinates[1] > 27:
            continue
        else:
            neighbors[direction] = coordinates
    return neighbors


def get_neighboring_pixels0(image, position):
    """
    retrieve the value of the surrounding pixels at location 'position' given the map
    """
    pixel_values = -10000 * np.ones((1, 4)).ravel()  # if the position is out of the map, return -1

    pos_left = [position[0] - 1, position[1]]
    pos_right = [position[0] + 1, position[1]]
    pos_up = [position[0], position[1] - 1]
    pos_down = [position[0], position[1] + 1]
    for i, pos in enumerate([pos_left, pos_right, pos_down, pos_up]):
        if pos[0] < 0 or pos[0] > 27 or pos[1] < 0 or pos[1] > 27:
            continue
        else:
            pixel_values[i] = image[pos[0], pos[1]]
    return pixel_values, [pos_left, pos_right, pos_down, pos_up]


def get_neighboring_pixels(image, neighbors_position):
    """
    retrieve the value of the surrounding pixels at location 'position' given the map
    """
    pixel_values = {}
    for action in neighbors_position:
        pos = neighbors_position[action]
        # print('pos:', neighbors_position)
        pixel_values[action] = image[pos[0], pos[1]]
    return pixel_values


class GreedyNavigator:
    def __init__(self):
        # The random navigator doesn't have any data members
        # But a more complex navigator may need to keep track of things
        # so you can create data members in this constructor
        # self.my_variable = 0

        # initialiaze the entropy to one to signal maximum uncertainty in the beginning
        self.better_goal_loc = None
        self.visited_locations = set()
        self.visited_locations.add((0, 0))
        self.current_entropy = 1  # np.ones((1, 4))
        self.directions = ['left', 'right', 'down', 'up']
        self.path = []
        pass

    def getAction(self, robot, map):
        """ Randomly selects a valid direction for the robot to travel

            The RandomNavigator completely ignores the incoming map of what has been seen so far.
            Maybe a smarter agent would take this additional info into account...
        """

        # This loop shows how you can create a mask, an grid of 0s and 1s
        # where 0s represent unexplored areas and 1s represent explored areas
        # This mask is used by the world estimating network
        mask = np.zeros((28, 28))
        for col in range(0, 28):
            for row in range(0, 28):
                if map[col, row] != 128:
                    mask[col, row] = 1

        # Creates an estimate of what the world looks like
        image = uNet.runNetwork(map, mask)

        # Use the classification network on the estimated image
        # to get a guess of what "world" we are in (e.g., what the MNIST digit of the world)
        char = classNet.runNetwork(image).ravel()
        output_dist = softmax(char)

        self.current_entropy = entropy(output_dist)
        robot_loc = robot.getLoc()
        neighbors = get_adjacent_states(robot_loc)
        # neighbors_pixel, neighbors_position = get_neighboring_pixels(image, robot_loc)
        neighbors_pixel = get_neighboring_pixels(image, neighbors)
        # info_gain = np.zeros((1, 4))

        self.path.append(robot_loc)

        goal_loc = get_goal(np.argmax(output_dist))
        # print(f'predicted number: {np.argmax(output_dist)} -- goal state returned: {goal_loc}')

        direction = None

        # print('max probability:', max(output_dist))

        if max(output_dist) <= 0.40:
            info_qual = 0
            for action in neighbors_pixel.keys():
                if abs(image[robot_loc[0], robot_loc[1]] - neighbors_pixel[action]) >= info_qual:
                    info_qual = abs(image[robot_loc[0], robot_loc[1]] - neighbors_pixel[action])
                    direction = action
                    new_pos = neighbors[action]
                else:
                    continue

                # If it is not a valid move, reset
                if not robot.checkValidLoc(new_pos[0], new_pos[1]) or tuple(new_pos) in self.visited_locations:
                    direction = None

            if direction is None:
                potential_actions = list(neighbors.keys())
                direction = random.choice(potential_actions)

        else:
            if len(neighbors) == 0:
                raise ValueError('No neighbor found!')

            if self.better_goal_loc is not None:
                goal_loc = self.better_goal_loc
                # print('************* Better location: ', goal_loc)

            min_distance = np.inf
            for action in neighbors.keys():
                distance_from_next_to_goal = compute_distance(goal_loc, neighbors[action])
                if distance_from_next_to_goal <= min_distance:
                    direction = action
                    min_distance = distance_from_next_to_goal
                else:
                    continue
        self.visited_locations.add(tuple(neighbors[direction]))

        return direction

    def reset(self):
        self.better_goal_loc = None
        self.visited_locations = set()
        self.visited_locations.add((0, 0))
        self.current_entropy = 1  # np.ones((1, 4))
        self.directions = ['left', 'right', 'down', 'up']
        self.path = []
