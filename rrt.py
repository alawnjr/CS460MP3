import sys

import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np

'''
Set up matplotlib to create a plot with an empty square
'''
def setupPlot():
    fig = plt.figure(num=None, figsize=(5, 5), dpi=120, facecolor='w', edgecolor='k')
    plt.autoscale(False)
    plt.axis('off')
    ax = fig.add_subplot(1,1,1)
    ax.set_axis_off()
    ax.add_patch(patches.Rectangle(
        (0,0),   # (x,y)
        1,          # width
        1,          # height
        fill=False
        ))
    return fig, ax

'''
Make a patch for a single polygon 
'''
def createPolygonPatch(polygon, color):
    verts = []
    codes= []
    for v in range(0, len(polygon)):
        xy = polygon[v]
        verts.append((xy[0]/10., xy[1]/10.))
        if v == 0:
            codes.append(Path.MOVETO)
        else:
            codes.append(Path.LINETO)
    verts.append(verts[0])
    codes.append(Path.CLOSEPOLY)
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor=color, lw=1)

    return patch
    

'''
Render the problem  
'''
def drawProblem(robotStart, robotGoal, polygons):
    fig, ax = setupPlot()
    patch = createPolygonPatch(robotStart, 'green')
    ax.add_patch(patch)    
    patch = createPolygonPatch(robotGoal, 'red')
    ax.add_patch(patch)    
    for p in range(0, len(polygons)):
        patch = createPolygonPatch(polygons[p], 'gray')
        ax.add_patch(patch)    
    plt.show()

'''
Grow a simple RRT 
'''
def growSimpleRRT(points):
    # Copy the input points so we can add new vertices created by edge splits
    newPoints = dict(points)
    adjListMap = dict()

    # Identify (or create) the root at (5, 5)
    root_coord = (5, 5)
    root_id = None
    for pid, coord in newPoints.items():
        if abs(coord[0] - root_coord[0]) < 1e-9 and abs(coord[1] - root_coord[1]) < 1e-9:
            root_id = pid
            break
    if root_id is None:
        root_id = max(newPoints.keys(), default=0) + 1
        newPoints[root_id] = root_coord

    adjListMap[root_id] = []
    in_tree = {root_id}

    # Preserve the original sampling order
    sample_ids = sorted(points.keys())
    next_id = max(newPoints.keys()) + 1

    def point_dist(a, b):
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def closest_on_segment(p, a, b):
        """
        Return (closest_point(x,y), t, distance) where t is the clamped
        projection factor on segment a->b (0 <= t <= 1).
        """
        a_vec = np.array(a, dtype=float)
        b_vec = np.array(b, dtype=float)
        p_vec = np.array(p, dtype=float)
        ab = b_vec - a_vec
        denom = float(np.dot(ab, ab))
        if denom == 0.0:
            return (a[0], a[1]), 0.0, point_dist(p, a)
        t = float(np.dot(p_vec - a_vec, ab) / denom)
        t_clamped = min(1.0, max(0.0, t))
        closest = a_vec + t_clamped * ab
        dist = float(np.linalg.norm(p_vec - closest))
        return (closest[0], closest[1]), t_clamped, dist

    for pid in sample_ids:
        # Skip the root if it appears in the samples
        if pid == root_id:
            continue
        sample_pt = points[pid]

        # Find the nearest point on the existing tree (vertex or along an edge)
        best_dist = float("inf")
        best_info = None  # (closest_point, (u, v), t)

        # If the tree only has the root, we still want to consider it
        for u in in_tree:
            u_pt = newPoints[u]
            dist_to_u = point_dist(sample_pt, u_pt)
            if dist_to_u < best_dist:
                best_dist = dist_to_u
                best_info = (u_pt, (u, u), 0.0)

            for v in adjListMap.get(u, []):
                # Avoid processing the same undirected edge twice
                if u >= v:
                    continue
                v_pt = newPoints[v]
                closest, t, dist = closest_on_segment(sample_pt, u_pt, v_pt)
                if dist < best_dist:
                    best_dist = dist
                    best_info = (closest, (u, v), t)

        closest_pt, (u, v), t = best_info
        on_vertex = u == v or t <= 1e-9 or t >= 1 - 1e-9

        # Ensure the sample point exists in the master list
        if pid not in newPoints:
            newPoints[pid] = sample_pt
        adjListMap.setdefault(pid, [])

        if on_vertex:
            # Attach directly to the nearest existing vertex
            attach_id = u if t <= 0.5 else v
            adjListMap.setdefault(attach_id, [])
            if pid not in adjListMap[attach_id]:
                adjListMap[attach_id].append(pid)
            if attach_id not in adjListMap[pid]:
                adjListMap[pid].append(attach_id)
            in_tree.add(pid)
        else:
            # Split the edge (u, v) at the projection point
            proj_id = next_id
            next_id += 1
            newPoints[proj_id] = closest_pt
            adjListMap.setdefault(u, [])
            adjListMap.setdefault(v, [])
            adjListMap[proj_id] = []

            # Remove the original edge
            if v in adjListMap[u]:
                adjListMap[u].remove(v)
            if u in adjListMap[v]:
                adjListMap[v].remove(u)

            # Add the two new edges created by the split
            adjListMap[u].append(proj_id)
            adjListMap[proj_id].append(u)
            adjListMap[v].append(proj_id)
            adjListMap[proj_id].append(v)

            # Connect the sample to the projection point
            adjListMap[pid].append(proj_id)
            adjListMap[proj_id].append(pid)

            in_tree.update({pid, proj_id})

    return newPoints, adjListMap

'''
Perform basic search 
'''
def basicSearch(tree, start, goal):
    # Simple breadth-first search on an adjacency-list tree/graph.
    if start not in tree or goal not in tree:
        return []

    from collections import deque

    q = deque([start])
    parent = {start: None}

    while q:
        node = q.popleft()
        if node == goal:
            break
        for nbr in tree.get(node, []):
            if nbr not in parent:
                parent[nbr] = node
                q.append(nbr)

    if goal not in parent:
        return []

    # Reconstruct path from goal back to start
    rev_path = []
    cur = goal
    while cur is not None:
        rev_path.append(cur)
        cur = parent[cur]
    rev_path.reverse()
    return rev_path

'''
Display the RRT and Path
'''
def displayRRTandPath(points, tree, path, robotStart = None, robotGoal = None, polygons = None, original_ids = None, grown_obstacles = None):
    fig, ax = setupPlot()

    # Draw grown obstacles first (for debugging) in light blue
    if grown_obstacles is not None:
        for grown_obs in grown_obstacles:
            ax.add_patch(createPolygonPatch(grown_obs, 'lightblue'))

    # Draw environment if provided
    if robotStart is not None and robotGoal is not None and polygons is not None:
        ax.add_patch(createPolygonPatch(robotStart, 'green'))
        ax.add_patch(createPolygonPatch(robotGoal, 'red'))
        for obs in polygons:
            ax.add_patch(createPolygonPatch(obs, 'gray'))

    # Draw the RRT edges in black
    drawn = set()
    for u, nbrs in tree.items():
        for v in nbrs:
            if (v, u) in drawn:
                continue
            drawn.add((u, v))
            pu = points[u]
            pv = points[v]
            ax.plot([pu[0]/10., pv[0]/10.], [pu[1]/10., pv[1]/10.], color='black', linewidth=1)

    # Draw vertices: original in black, added in red
    if original_ids is None:
        xs = [p[0]/10. for p in points.values()]
        ys = [p[1]/10. for p in points.values()]
        ax.scatter(xs, ys, color='black', s=8)
    else:
        orig_x, orig_y, added_x, added_y = [], [], [], []
        for pid, (x, y) in points.items():
            if pid in original_ids:
                orig_x.append(x/10.)
                orig_y.append(y/10.)
            else:
                added_x.append(x/10.)
                added_y.append(y/10.)
        if orig_x:
            ax.scatter(orig_x, orig_y, color='black', s=8)
        if added_x:
            ax.scatter(added_x, added_y, color='red', s=12)

    # Draw the path in orange if non-empty
    if path:
        for i in range(len(path) - 1):
            a = points[path[i]]
            b = points[path[i+1]]
            ax.plot([a[0]/10., b[0]/10.], [a[1]/10., b[1]/10.], color='orange', linewidth=2.5)
        # Highlight path nodes
        px = [points[n][0]/10. for n in path]
        py = [points[n][1]/10. for n in path]
        ax.scatter(px, py, color='orange', s=16)

    plt.show()
    return 

'''
Collision checking
'''
def isCollisionFree(robot, point, obstacles):
    # Translate robot to world coordinates using the origin "point"
    qx, qy = point
    placed_robot = [(qx + x, qy + y) for (x, y) in robot]

    eps = 1e-9

    # Boundary of workspace is [0,10] x [0,10]
    for (x, y) in placed_robot:
        if x < -eps or x > 10 + eps or y < -eps or y > 10 + eps:
            return False

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a, b, p):
        return (
            min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps and
            min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps and
            abs(cross(a, b, p)) <= eps
        )

    def segments_intersect(p1, p2, q1, q2):
        d1 = cross(p1, p2, q1)
        d2 = cross(p1, p2, q2)
        d3 = cross(q1, q2, p1)
        d4 = cross(q1, q2, p2)

        if (d1 * d2 < -eps) and (d3 * d4 < -eps):
            return True

        # Colinear / touching cases
        if abs(d1) <= eps and on_segment(p1, p2, q1):
            return True
        if abs(d2) <= eps and on_segment(p1, p2, q2):
            return True
        if abs(d3) <= eps and on_segment(q1, q2, p1):
            return True
        if abs(d4) <= eps and on_segment(q1, q2, p2):
            return True
        return False

    def polygon_contains(poly, p):
        # Ray casting algorithm for point-in-polygon
        cnt = 0
        n = len(poly)
        for i in range(n):
            a = poly[i]
            b = poly[(i + 1) % n]
            if on_segment(a, b, p):
                return True  # On boundary counts as collision
            if ((a[1] > p[1]) != (b[1] > p[1])):
                x_int = (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1] + eps) + a[0]
                if p[0] < x_int:
                    cnt += 1
        return cnt % 2 == 1

    def polygons_intersect(poly1, poly2):
        # Edge intersection
        for i in range(len(poly1)):
            a1 = poly1[i]
            a2 = poly1[(i + 1) % len(poly1)]
            for j in range(len(poly2)):
                b1 = poly2[j]
                b2 = poly2[(j + 1) % len(poly2)]
                if segments_intersect(a1, a2, b1, b2):
                    return True
        # Containment
        if polygon_contains(poly1, poly2[0]):
            return True
        if polygon_contains(poly2, poly1[0]):
            return True
        return False

    for obs in obstacles:
        if polygons_intersect(placed_robot, obs):
            return False

    return True

'''
The full RRT algorithm - SIMPLE VERSION with toggleable collision detection
'''
def RRT(robot, obstacles, startPoint, goalPoint):
    # ========== TOGGLE COLLISION DETECTION METHOD ==========
    USE_MINKOWSKI = False  # Set to False to use sampling-based collision detection
    # =======================================================
    
    points = {1: startPoint, 2: goalPoint}
    tree = {1: [], 2: []}
    start_id, goal_id = 1, 2
    original_ids = {start_id, goal_id}
    
    # Initialize grown_obstacles to None (will be set if using Minkowski)
    grown_obstacles = None
    
    if USE_MINKOWSKI:
        # Pre-compute Minkowski sums for collision detection
        print("Computing Minkowski sums for non-convex obstacles...")
        
        def convex_hull(pts):
            pts = sorted(set(pts))
            if len(pts) <= 2:
                return pts
            def ccw(o, a, b):
                return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
            lower = []
            for p in pts:
                while len(lower) >= 2 and ccw(lower[-2], lower[-1], p) <= 0:
                    lower.pop()
                lower.append(p)
            upper = []
            for p in reversed(pts):
                while len(upper) >= 2 and ccw(upper[-2], upper[-1], p) <= 0:
                    upper.pop()
                upper.append(p)
            return lower[:-1] + upper[:-1]
        
        def poly_area(p):
            return 0.5 * sum(p[i][0]*p[(i+1)%len(p)][1] - p[(i+1)%len(p)][0]*p[i][1] for i in range(len(p)))
        
        def ensure_ccw(p):
            return p if poly_area(p) > 0 else list(reversed(p))
        
        def point_in_triangle(tri, p):
            def cross(o, a, b):
                return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
            a, b, c = tri
            c1 = cross(a, b, p)
            c2 = cross(b, c, p)
            c3 = cross(c, a, p)
            has_neg = (c1 < 0) or (c2 < 0) or (c3 < 0)
            has_pos = (c1 > 0) or (c2 > 0) or (c3 > 0)
            return not (has_neg and has_pos)
        
        def triangulate(poly):
            # Ear clipping triangulation
            p = list(poly)
            if len(p) < 3:
                return []
            if poly_area(p) < 0:
                p.reverse()
            
            def cross(o, a, b):
                return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
            
            triangles = []
            idxs = list(range(len(p)))
            
            def is_ear(i):
                pos = idxs.index(i)
                prev_i = idxs[(pos-1) % len(idxs)]
                next_i = idxs[(pos+1) % len(idxs)]
                a, b, c = p[prev_i], p[i], p[next_i]
                if cross(a, b, c) <= 0:
                    return False
                tri = (a, b, c)
                for j in idxs:
                    if j in (prev_i, i, next_i):
                        continue
                    if point_in_triangle(tri, p[j]):
                        return False
                return True
            
            while len(idxs) > 3:
                ear_found = False
                for i in list(idxs):
                    if is_ear(i):
                        pos = idxs.index(i)
                        prev_i = idxs[(pos-1) % len(idxs)]
                        next_i = idxs[(pos+1) % len(idxs)]
                        triangles.append([p[prev_i], p[i], p[next_i]])
                        idxs.remove(i)
                        ear_found = True
                        break
                if not ear_found:
                    # Degenerate case, use convex hull as fallback
                    return [convex_hull(p)]
            
            if len(idxs) == 3:
                triangles.append([p[idxs[0]], p[idxs[1]], p[idxs[2]]])
            
            return [ensure_ccw(t) for t in triangles]
        
        def convex_minkowski_sum(poly_a, poly_b):
            # Both must be convex and CCW
            a = ensure_ccw(poly_a)
            b = ensure_ccw(poly_b)
            
            def lowest_idx(poly):
                return min(range(len(poly)), key=lambda i: (poly[i][1], poly[i][0]))
            
            def cross(o, a, b):
                return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
            
            ia = lowest_idx(a)
            ib = lowest_idx(b)
            res = []
            i, j = ia, ib
            n, m = len(a), len(b)
            
            for _ in range(n + m):
                res.append((a[i][0] + b[j][0], a[i][1] + b[j][1]))
                ni = (i + 1) % n
                nj = (j + 1) % m
                
                edge_a = (a[ni][0] - a[i][0], a[ni][1] - a[i][1])
                edge_b = (b[nj][0] - b[j][0], b[nj][1] - b[j][1])
                cross_val = cross((0,0), edge_a, edge_b)
                
                if cross_val >= 0:
                    i = ni
                if cross_val <= 0:
                    j = nj
            
            return ensure_ccw(res)
        
        # Negate robot for Minkowski sum (obstacle ⊕ -robot)
        neg_robot = [(-x, -y) for (x, y) in robot]
        
        # Compute grown obstacles by triangulating non-convex obstacles
        grown_obstacles = []
        for obs in obstacles:
            # Triangulate obstacle
            triangles = triangulate(obs)
            # Compute Minkowski sum for each triangle
            for tri in triangles:
                grown = convex_minkowski_sum(tri, neg_robot)
                grown_obstacles.append(grown)
        
        # Also grow the workspace boundaries (treated as thin obstacles)
        # Create boundary "obstacles" - thin rectangles along each edge
        boundary_thickness = 0.01
        boundary_obstacles = [
            # Left boundary (x=0)
            [(-boundary_thickness, -boundary_thickness), (-boundary_thickness, 10+boundary_thickness), 
             (0, 10+boundary_thickness), (0, -boundary_thickness)],
            # Right boundary (x=10)
            [(10, -boundary_thickness), (10, 10+boundary_thickness),
             (10+boundary_thickness, 10+boundary_thickness), (10+boundary_thickness, -boundary_thickness)],
            # Bottom boundary (y=0)
            [(-boundary_thickness, -boundary_thickness), (10+boundary_thickness, -boundary_thickness),
             (10+boundary_thickness, 0), (-boundary_thickness, 0)],
            # Top boundary (y=10)
            [(-boundary_thickness, 10), (10+boundary_thickness, 10),
             (10+boundary_thickness, 10+boundary_thickness), (-boundary_thickness, 10+boundary_thickness)]
        ]
        
        for boundary_obs in boundary_obstacles:
            grown = convex_minkowski_sum(boundary_obs, neg_robot)
            grown_obstacles.append(grown)
        
        print(f"Grown obstacles: {len(grown_obstacles)} polygons from {len(obstacles)} obstacles + 4 boundaries")
        
        # New collision checker using point-in-polygon on grown obstacles
        def point_in_polygon(poly, pt):
            x, y = pt
            inside = False
            n = len(poly)
            p1x, p1y = poly[0]
            for i in range(1, n + 1):
                p2x, p2y = poly[i % n]
                if y > min(p1y, p2y):
                    if y <= max(p1y, p2y):
                        if x <= max(p1x, p2x):
                            if p1y != p2y:
                                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xinters:
                                inside = not inside
                p1x, p1y = p2x, p2y
            return inside
        
        def point_collides(pt):
            # Check if point is inside any grown obstacle (includes grown boundaries)
            for grown_obs in grown_obstacles:
                if point_in_polygon(grown_obs, pt):
                    return True
            return False
        
        def segment_crosses_grown_obstacle(p1, p2):
            # Check if segment crosses any grown obstacle edge
            eps = 1e-9
            
            def cross(o, a, b):
                return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
            
            def on_seg(a, b, p):
                return (min(a[0],b[0])-eps <= p[0] <= max(a[0],b[0])+eps and
                        min(a[1],b[1])-eps <= p[1] <= max(a[1],b[1])+eps and
                        abs(cross(a,b,p)) <= eps)
            
            def seg_intersect(a1, a2, b1, b2):
                d1 = cross(a1, a2, b1)
                d2 = cross(a1, a2, b2)
                d3 = cross(b1, b2, a1)
                d4 = cross(b1, b2, a2)
                if (d1*d2 < -eps) and (d3*d4 < -eps):
                    return True
                if abs(d1)<=eps and on_seg(a1,a2,b1): return True
                if abs(d2)<=eps and on_seg(a1,a2,b2): return True
                if abs(d3)<=eps and on_seg(b1,b2,a1): return True
                if abs(d4)<=eps and on_seg(b1,b2,a2): return True
                return False
            
            for grown_obs in grown_obstacles:
                for i in range(len(grown_obs)):
                    if seg_intersect(p1, p2, grown_obs[i], grown_obs[(i+1)%len(grown_obs)]):
                        return True
            return False
        
        def edge_free(p1, p2):
            # Check if either endpoint is in collision (includes grown boundaries now)
            if point_collides(p1) or point_collides(p2):
                return False
            # Check if segment crosses grown obstacles (includes grown boundaries)
            if segment_crosses_grown_obstacle(p1, p2):
                return False
            return True
    
    else:  # USE_MINKOWSKI == False: Use sampling-based collision detection
        print("Using sampling-based collision detection...")
        
        def point_collides(pt):
            # Use isCollisionFree to check if robot at this point is collision-free
            return not isCollisionFree(robot, pt, obstacles)
        
        def edge_free(p1, p2, step=0.05, min_steps=10, max_steps=2000):
            # Sample points along the edge and check each one
            dist = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
            steps = max(int(dist / step), min_steps)
            steps = min(steps, max_steps)
            for i in range(steps + 1):
                t = i / steps
                px = p1[0] + t * (p2[0] - p1[0])
                py = p1[1] + t * (p2[1] - p1[1])
                if not isCollisionFree(robot, (px, py), obstacles):
                    return False
            return True
    
    # Check validity of start and goal
    if point_collides(startPoint):
        print("Start in collision!")
        robotStart = [(x + startPoint[0], y + startPoint[1]) for (x, y) in robot]
        robotGoal = [(x + goalPoint[0], y + goalPoint[1]) for (x, y) in robot]
        displayRRTandPath(points, tree, [], robotStart, robotGoal, obstacles, original_ids, grown_obstacles)
        return points, tree, []
    
    if point_collides(goalPoint):
        print("Goal in collision!")
        robotStart = [(x + startPoint[0], y + startPoint[1]) for (x, y) in robot]
        robotGoal = [(x + goalPoint[0], y + goalPoint[1]) for (x, y) in robot]
        displayRRTandPath(points, tree, [], robotStart, robotGoal, obstacles, original_ids, grown_obstacles)
        return points, tree, []
    
    def nearest(pt):
        best, best_d = start_id, float("inf")
        for nid in points:
            if nid == goal_id:
                continue
            d = np.hypot(points[nid][0] - pt[0], points[nid][1] - pt[1])
            if d < best_d:
                best, best_d = nid, d
        return best
    
    next_id = 3
    max_iter = 10000
    
    print(f"RRT: {startPoint} → {goalPoint}")
    
    for it in range(max_iter):
        # Sample random point in workspace
        sample = (np.random.rand()*10, np.random.rand()*10)
        
        # Find nearest node in tree
        nid = nearest(sample)
        npt = points[nid]
        
        # Try to connect directly to sample (no steering)
        # Collision check at sample point
        if point_collides(sample):
            continue
        
        # Collision check along edge
        if not edge_free(npt, sample):
            continue
        
        # Add new node
        tree[next_id] = [nid]
        tree[nid].append(next_id)
        points[next_id] = sample
        next_id += 1
        
        # Try to connect to goal
        if edge_free(sample, goalPoint):
            tree[next_id-1].append(goal_id)
            tree[goal_id] = [next_id-1]
            print(f"Goal reached at iteration {it}, tree has {len(points)} nodes")
            break
        
        if it % 2000 == 0 and it > 0:
            print(f"  Iteration {it}: {len(points)} nodes")
    
    path = basicSearch(tree, start_id, goal_id)
    print(f"Path: {len(path)} nodes" if path else "No path")
    
    robotStart = [(x + startPoint[0], y + startPoint[1]) for (x, y) in robot]
    robotGoal = [(x + goalPoint[0], y + goalPoint[1]) for (x, y) in robot]
    displayRRTandPath(points, tree, path, robotStart, robotGoal, obstacles, original_ids, grown_obstacles)
    return points, tree, path

if __name__ == "__main__":
    
    # Retrive file name for input data
    if(len(sys.argv) < 6):
        print ("Five arguments required: python rrt.py [env-file] [x1] [y1] [x2] [y2]")
        exit()
    
    filename = sys.argv[1]
    x1 = float(sys.argv[2])
    y1 = float(sys.argv[3])
    x2 = float(sys.argv[4])
    y2 = float(sys.argv[5])

    # Read data and parse polygons
    lines = [line.rstrip('\n').strip() for line in open(filename)]
    # Filter out empty lines
    lines = [line for line in lines if line]
    
    robot = []
    obstacles = []
    for line_idx in range(0, len(lines)):
        xys = lines[line_idx].split(';')
        polygon = []
        for p in range(0, len(xys)):
            xy_str = xys[p].strip()
            if not xy_str:  # Skip empty strings
                continue
            xy = xy_str.split(',')
            if len(xy) >= 2:
                polygon.append((float(xy[0].strip()), float(xy[1].strip())))
        if line_idx == 0:
            robot = polygon
        else:
            obstacles.append(polygon)

    # Print out the data
    print ("Robot:")
    print (str(robot))
    print ("Pologonal obstacles:")
    for p in range(0, len(obstacles)):
        print (str(obstacles[p]))
    print ("")

    # Visualize
    robotStart = []
    robotGoal = []

    for i in range(0, len(robot)):
        robotStart.append((robot[i][0] + x1, robot[i][1] + y1))
        robotGoal.append((robot[i][0] + x2, robot[i][1] + y2))
    drawProblem(robotStart, robotGoal, obstacles)

    # Example points for calling growSimpleRRT
    # You should expect many mroe points, e.g., 200-500
    points = dict()
    points[1] = (5, 5)
    points[2] = (7, 8.2)
    points[3] = (6.5, 5.2)
    points[4] = (0.3, 4)
    points[5] = (6, 3.7)
    points[6] = (9.7, 6.4)
    points[7] = (4.4, 2.8)
    points[8] = (9.1, 3.1)
    points[9] = (8.1, 6.5)
    points[10] = (0.7, 5.4)
    points[11] = (5.1, 3.9)
    points[12] = (2, 6)
    points[13] = (0.5, 6.7)
    points[14] = (8.3, 2.1)
    points[15] = (7.7, 6.3)
    points[16] = (7.9, 5)
    points[17] = (4.8, 6.1)
    points[18] = (3.2, 9.3)
    points[19] = (7.3, 5.8)
    points[20] = (9, 0.6)

    # Printing the points
    print ("")
    print ("The input points are:")
    print (str(points))
    print ("")
    
    original_ids = set(points.keys())
    points, adjListMap = growSimpleRRT(points)

    # Search for a solution  
    path = basicSearch(adjListMap, 1, 2)    

    # Your visualization code 
    displayRRTandPath(points, adjListMap, path, None, None, None, original_ids, None) 

    # Solve a real RRT problem (visualized inside the call)
    RRT(robot, obstacles, (x1, y1), (x2, y2))



