The objective of this assignment is for the robot, which starts at (0,0) to navigate to a specific corner of the map. The goal depends on what world you occupy. The maps are constructed from the MNIST dataset, a hand drawn number dataset consisting of numbers 0-9. The world is 28x28 grid.

If the MNIST digit is: 

digit: 0-2 - goal: (0,27)
digit: 3-5 - goal: (27,27)
digit: 6-9 - goal: (27,0)

The robot can travel one space at a time and can observe the value of the spaces it has visited. It costs -1 to move and -400 if it travels to the wrong goal. Reaching the goal has a reward of 100. Try get the maximum score by reaching the goal in the fewest number of steps. Two trained neural networks are provided to help. One network provides an estimate of what the world looks like given what you have currently observed. The other network takes this estimate and provides an estimate of what digit the world belongs to.


This code was written in python 3.6.5
To execute this code, you will need Python 3.5+ (I recommend downloading Anaconda) and pytorch
Anaconda Package Manager can be found at https://docs.anaconda.com/anaconda/install/
pytorch can be downloaded at https://pytorch.org

Pytorch install instructions:
1. Go to https://pytorch.org/get-started/locally/
2. Select the appropriate settings (I would recommend: Stable, OS, Conda, python 3.6, None)
3. Follow the instructions to install Python (if needed) and Pytorch
4. Run the main script with `python main.py` to test.

`networkFolder/FunctionList.py` is the main file you need to concern yourself with. It provides 3 classes:

Class `Map`:

  - Class Variables:
    - `map` - saves a numpy array of an MNIST digit
    - `number` - the number represented by the MNIST image
  - Functions:
    - `getNewMap()` - sets map to the next numpy array from the MNIST dataset


Class `WorldEstimatingNetwork`:

  - Class Variables:
    - `network` - This is the network that will provide an estimate of what the world looks like

  - Functions:
    - `runNetwork(map, exploredArea)`
	  - requires two inputs. The first is the map at its current level of exploration. Second is a binary mask the same size of the map, with 1's for area's explored and 0's for unexplored regions

      - This function returns a numpy array of the current estimate of what the world looks like

- Class `DigitClassificationNetwork`:

  - Class Variables:
    - `network` - This is the network that will provide an estimate of what Digit the world is

  - Functions:
    - `runNetwork(map)`
	  - Single input, the estimated world provided by WorldEstimatingNetwork

      - This function returns an array of size 10, with each index representing the different digits the world could be. The largest number represents the networks estimate of what class the world is.

We also provide a `Robot` class in `RobotClass.py` and  a Game Class in `GameClass.py`. Look at `main.py` for an example of how they are used.

Class `Robot`:
  - Class Variables:
    - `xLoc` - x position of robot
    - `yLoc` - y position of robot
  	- `xLim` - tuple (min x coordinate, max x coordinate)
  	- `yLim` - tuple (min y coordinate, max y coordiante)
  - Functions:
    - `getLoc()`
	  - Outputs the x,y location of the robot
	- `move(direction)`
	  - direction is a string - "left", "right", "up", "down"
	  - Updates the robot location based on the direction provided

Class `Game`:
  - Class Variables:
    - `truthMap` - The ground truth map used in the game world
	  - `navigator` - A class that has a `getAction(robot, map)` function that generates the next move the robot should take given the input map. (See `RandomNavigator.py` for an example)
	  - `robot` - A robot object that tracks robot location
	  - `score` - The current score 
  - Functions:
	- `tick()` - Runs one iteration of the game
	- `getScore()` - Returns the current score
	- `getIteration()` - Returns the number of times `tick()` has been run

Finally, we provide an example of a navigator in `RandomNavigator.py` that interfaces with the Game class. In order to create more sophisticated agents, you will want to create different Navigators.

Class `RandomNavigator`:
  - Class Variables: None
  - Functions:
    - getAction(robot, map)
	  - Takes in the robot and the currently explored map
	  - Randomly chooses a valid action for the robot to take, completely ignoring the currently explored map. A smarter agent would take advantage of this info...

Gotchas:
- To access position `(x,y)` on the numpy array `image`, you should call `image[y][x]`. This is because you are looking at the `y`-th row, `x`-th column of the array.
- Your navigator classes should not use the ground truth map of the Game class in any way, since that would be peeking at the solution.