from RobotClass import Robot

# Top Left Corner
r = Robot(0,0)
assert r.checkValidMove("up") == False
assert r.checkValidMove("left") == False
assert r.checkValidMove("down") == True
assert r.checkValidMove("right") == True

# Bottom Right Corner
r._setLoc(27,27)
assert r.checkValidMove("up") == True
assert r.checkValidMove("left") == True
assert r.checkValidMove("down") == False
assert r.checkValidMove("right") == False

# Bottom Left Corner
r._setLoc(0,27)
assert r.checkValidMove("up") == True
assert r.checkValidMove("left") == False
assert r.checkValidMove("down") == False
assert r.checkValidMove("right") == True

# Top Right Corner
r._setLoc(27,0)
assert r.checkValidMove("up") == False
assert r.checkValidMove("left") == True
assert r.checkValidMove("down") == True
assert r.checkValidMove("right") == False

# Top Right Corner
r._setLoc(10,10)
assert r.checkValidMove("up") == True
assert r.checkValidMove("left") == True
assert r.checkValidMove("down") == True
assert r.checkValidMove("right") == True

# Top Edge
r._setLoc(10,0)
assert r.checkValidMove("up") == False
assert r.checkValidMove("left") == True
assert r.checkValidMove("down") == True
assert r.checkValidMove("right") == True

# Bottom Edge
r._setLoc(10,27)
assert r.checkValidMove("up") == True
assert r.checkValidMove("left") == True
assert r.checkValidMove("down") == False
assert r.checkValidMove("right") == True

# Left Edge
r._setLoc(0,10)
assert r.checkValidMove("up") == True
assert r.checkValidMove("left") == False
assert r.checkValidMove("down") == True
assert r.checkValidMove("right") == True

# Left Edge
r._setLoc(27,10)
assert r.checkValidMove("up") == True
assert r.checkValidMove("left") == True
assert r.checkValidMove("down") == True
assert r.checkValidMove("right") == False

# Move Up/Down/Left Right
r = Robot(0,0)
r.move("right")
assert r.getLoc()==(1,0)
r.move("down")
assert r.getLoc()==(1,1)
r.move("left")
assert r.getLoc()==(0,1)
r.move("up")
assert r.getLoc()==(0,0)

r._setLoc(100, 0)