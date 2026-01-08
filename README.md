# CS460/CS560 MP3 — RRT (2D) Planner

Implementation of a **Rapidly-exploring Random Tree (RRT)** planner for a **translating convex polygon robot** (no rotation) in a **10×10** 2D workspace with polygonal obstacles.

## What’s in this repo
- `rrt.py` — RRT implementation + visualization

## Requirements
- Python 3.7–3.11
- `numpy`, `matplotlib`

Install:
```bash
pip install numpy matplotlib
```

## Input format
Environment files are plain text:
- **Line 1:** robot polygon (clockwise vertices)
- **Line 2+:** obstacle polygons (clockwise vertices), one polygon per line  
Vertices are written as `x,y` pairs separated by `;`.

Example:
```txt
0,0;0.2,0;0.1,0.1
2,2;3,2;3,3;2,3
```

## Run
```bash
python rrt.py <env-file> <x_start> <y_start> <x_goal> <y_goal>
```

Example:
```bash
python rrt.py robot_env_01.txt 1.0 2.0 8.5 7.0
```

## What it does when you run it
1. Parses the robot + obstacles from the env file
2. Shows the environment (start robot in **green**, goal robot in **red**, obstacles in **gray**)
3. Builds and visualizes a sample RRT from provided points (tree in **black**, path in **orange**)
4. Runs the full RRT planner with random sampling + collision checking and visualizes the final tree/path

## Implemented functions (in `rrt.py`)
- `growSimpleRRT(points)`  
  Builds an RRT rooted at (5,5) using provided sample points; supports nearest-point projection onto edges.
- `basicSearch(tree, start, goal)`  
  BFS to recover a path in the adjacency-list tree.
- `displayRRTandPath(points, tree, path, robotStart=None, robotGoal=None, polygons=None)`  
  Draws the tree (black) and the path (orange); optionally overlays the environment.
- `isCollisionFree(robot, point, obstacles)`  
  Polygon collision test against workspace boundary + obstacles.
- `RRT(robot, obstacles, startPoint, goalPoint)`  
  Full RRT planner: random sampling, edge checking, connect-to-goal, then visualizes and returns `(points, tree, path)`.

## Notes
- Workspace bounds are fixed: **0 ≤ x ≤ 10**, **0 ≤ y ≤ 10**
- Robot is treated as a convex polygon (up to 6 vertices), **translation only**
- Only standard Python + `numpy`/`matplotlib` are used
