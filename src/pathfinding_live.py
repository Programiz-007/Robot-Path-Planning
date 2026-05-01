import tkinter as tk
from tkinter import ttk
from collections import deque
import heapq
import math
import random
import time


GRID_ROWS = 25
GRID_COLS = 25
CELL_SIZE = 26
MARGIN_TOP = 100
MARGIN_LEFT = 20
MARGIN_RIGHT = 220
MARGIN_BOTTOM = 20

START = (1, 1)
GOAL  = (23, 23)

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Colors
C_BG        = '#1A202C'
C_FREE      = '#EDF2F7'
C_OBSTACLE  = '#2D3748'
C_EXPLORED  = '#90CDF4'
C_FRONTIER  = '#63B3ED'
C_PATH      = '#F6AD55'
C_START     = '#48BB78'
C_GOAL      = '#FC8181'
C_GRID      = '#A0AEC0'
C_CURRENT   = '#E53E3E'
C_TEXT      = '#E2E8F0'
C_ACCENT    = '#63B3ED'


def generate_obstacles():
    obs = set()
    for c in range(4, 18):   obs.add((5, c))
    for r in range(2, 14):   obs.add((r, 20))
    for c in range(8, 23):   obs.add((12, c))
    for r in range(14, 23):  obs.add((r, 8))
    for r in range(16, 21):
        for c in range(14, 17): obs.add((r, c))
    for c in range(1, 7):    obs.add((19, c))
    for r in range(18, 25):  obs.add((r, 21))
    for r in range(7, 11):   obs.add((r, 4))
    for c in range(10, 14):  obs.add((9, c))
    obs.discard(START)
    obs.discard(GOAL)
    return obs

# Each algorithm is a generator that yields (action, data) tuples:
#   ("explore", (r, c))       — mark cell as explored
#   ("frontier", (r, c))      — mark cell as frontier/queued
#   ("current", (r, c))       — highlight current cell being processed
#   ("path", [(r,c), ...])    — final path found
#   ("done", stats_dict)      — algorithm finished

def bfs_gen(grid, start, goal):
    visited = {start}
    parent = {start: None}
    queue = deque([start])
    explored_count = 0

    while queue:
        node = queue.popleft()
        yield ("current", node)
        yield ("explore", node)
        explored_count += 1

        if node == goal:
            path = _reconstruct(parent, goal)
            yield ("path", path)
            yield ("done", {"explored": explored_count, "path_len": len(path)})
            return

        for dr, dc in DIRECTIONS:
            nr, nc = node[0] + dr, node[1] + dc
            if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS and (nr, nc) not in visited and grid[nr][nc] == 0:
                visited.add((nr, nc))
                parent[(nr, nc)] = node
                queue.append((nr, nc))
                yield ("frontier", (nr, nc))

    yield ("done", {"explored": explored_count, "path_len": 0})


def dfs_gen(grid, start, goal):
    visited = {start}
    parent = {start: None}
    stack = [start]
    explored_count = 0

    while stack:
        node = stack.pop()
        yield ("current", node)
        yield ("explore", node)
        explored_count += 1

        if node == goal:
            path = _reconstruct(parent, goal)
            yield ("path", path)
            yield ("done", {"explored": explored_count, "path_len": len(path)})
            return

        for dr, dc in DIRECTIONS:
            nr, nc = node[0] + dr, node[1] + dc
            if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS and (nr, nc) not in visited and grid[nr][nc] == 0:
                visited.add((nr, nc))
                parent[(nr, nc)] = node
                stack.append((nr, nc))
                yield ("frontier", (nr, nc))

    yield ("done", {"explored": explored_count, "path_len": 0})


def dijkstra_gen(grid, start, goal):
    dist = {start: 0}
    parent = {start: None}
    heap = [(0, start[0], start[1])]
    closed = set()
    explored_count = 0

    while heap:
        d, r, c = heapq.heappop(heap)
        node = (r, c)
        if node in closed:
            continue
        closed.add(node)
        yield ("current", node)
        yield ("explore", node)
        explored_count += 1

        if node == goal:
            path = _reconstruct(parent, goal)
            yield ("path", path)
            yield ("done", {"explored": explored_count, "path_len": len(path)})
            return

        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS and (nr, nc) not in closed and grid[nr][nc] == 0:
                nd = d + 1
                if nd < dist.get((nr, nc), float('inf')):
                    dist[(nr, nc)] = nd
                    parent[(nr, nc)] = node
                    heapq.heappush(heap, (nd, nr, nc))
                    yield ("frontier", (nr, nc))

    yield ("done", {"explored": explored_count, "path_len": 0})


def astar_gen(grid, start, goal, heuristic='manhattan'):
    def h(p):
        if heuristic == 'manhattan':
            return abs(p[0] - goal[0]) + abs(p[1] - goal[1])
        else:
            return math.sqrt((p[0] - goal[0])**2 + (p[1] - goal[1])**2)

    g_score = {start: 0}
    parent = {start: None}
    heap = [(h(start), start[0], start[1])]
    closed = set()
    explored_count = 0

    while heap:
        f, r, c = heapq.heappop(heap)
        node = (r, c)
        if node in closed:
            continue
        closed.add(node)
        yield ("current", node)
        yield ("explore", node)
        explored_count += 1

        if node == goal:
            path = _reconstruct(parent, goal)
            yield ("path", path)
            yield ("done", {"explored": explored_count, "path_len": len(path)})
            return

        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS and (nr, nc) not in closed and grid[nr][nc] == 0:
                ng = g_score[node] + 1
                if ng < g_score.get((nr, nc), float('inf')):
                    g_score[(nr, nc)] = ng
                    parent[(nr, nc)] = node
                    heapq.heappush(heap, (ng + h((nr, nc)), nr, nc))
                    yield ("frontier", (nr, nc))

    yield ("done", {"explored": explored_count, "path_len": 0})


def rrt_gen(grid, start, goal, max_iter=5000):
    rng = random.Random()
    tree = {start: None}
    nodes = [start]
    explored_count = 0

    for _ in range(max_iter):
        if rng.random() < 0.10:
            rand_pt = goal
        else:
            rand_pt = (rng.randint(0, GRID_ROWS - 1), rng.randint(0, GRID_COLS - 1))

        nearest = min(nodes, key=lambda n: abs(n[0] - rand_pt[0]) + abs(n[1] - rand_pt[1]))

        dr = rand_pt[0] - nearest[0]
        dc = rand_pt[1] - nearest[1]
        if dr == 0 and dc == 0:
            continue
        if abs(dr) >= abs(dc):
            sr, sc = (1 if dr > 0 else -1), 0
        else:
            sr, sc = 0, (1 if dc > 0 else -1)

        new_node = (nearest[0] + sr, nearest[1] + sc)

        if 0 <= new_node[0] < GRID_ROWS and 0 <= new_node[1] < GRID_COLS and \
           grid[new_node[0]][new_node[1]] == 0 and new_node not in tree:
            tree[new_node] = nearest
            nodes.append(new_node)
            explored_count += 1
            yield ("current", nearest)
            yield ("explore", new_node)

            if new_node == goal:
                path = _reconstruct(tree, goal)
                yield ("path", path)
                yield ("done", {"explored": explored_count, "path_len": len(path)})
                return

            if abs(new_node[0] - goal[0]) + abs(new_node[1] - goal[1]) == 1 and \
               grid[goal[0]][goal[1]] == 0 and goal not in tree:
                tree[goal] = new_node
                nodes.append(goal)
                explored_count += 1
                yield ("explore", goal)
                path = _reconstruct(tree, goal)
                yield ("path", path)
                yield ("done", {"explored": explored_count, "path_len": len(path)})
                return

    yield ("done", {"explored": explored_count, "path_len": 0})


def _reconstruct(parent, end):
    if end not in parent:
        return []
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path


class PathfindingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Robot Pathfinding — Live Visualizer")
        self.root.configure(bg=C_BG)

        canvas_w = MARGIN_LEFT + GRID_COLS * CELL_SIZE + MARGIN_RIGHT
        canvas_h = MARGIN_TOP + GRID_ROWS * CELL_SIZE + MARGIN_BOTTOM
        self.root.geometry(f"{canvas_w}x{canvas_h}")
        self.root.resizable(False, False)

        self.canvas = tk.Canvas(root, width=canvas_w, height=canvas_h,
                                bg=C_BG, highlightthickness=0)
        self.canvas.pack()

        # State
        self.obstacles_on = False
        self.obstacles = generate_obstacles()
        self.grid = [[0]*GRID_COLS for _ in range(GRID_ROWS)]
        self.cell_rects = {}
        self.cell_state = {}  # (r,c) -> 'free'|'obstacle'|'explored'|'frontier'|'path'|'current'
        self.generator = None
        self.running = False
        self.speed = 5  # ms delay
        self.steps_per_frame = 2
        self.algo_name = "BFS"
        self.path_line_ids = []
        self.stats = {"explored": 0, "path_len": 0}

        self._build_ui()
        self._draw_grid()
        self._update_info()

    def _build_ui(self):
        cx = MARGIN_LEFT + GRID_COLS * CELL_SIZE + 15
        cy = MARGIN_TOP

        # Title
        self.canvas.create_text(MARGIN_LEFT + GRID_COLS * CELL_SIZE // 2, 15,
                                text="ROBOT PATHFINDING", font=("Helvetica", 16, "bold"),
                                fill=C_TEXT, anchor='n')
        self.canvas.create_text(MARGIN_LEFT + GRID_COLS * CELL_SIZE // 2, 38,
                                text="Live Step-by-Step Visualizer", font=("Helvetica", 10),
                                fill=C_ACCENT, anchor='n')

        # Algorithm selector
        self.canvas.create_text(cx, cy, text="Algorithm", font=("Helvetica", 10, "bold"),
                                fill=C_TEXT, anchor='nw')
        algo_frame = tk.Frame(self.root, bg=C_BG)
        self.algo_var = tk.StringVar(value="BFS")
        algos = ["BFS", "DFS", "Dijkstra", "A* (Manhattan)", "A* (Euclidean)", "RRT"]
        self.algo_menu = ttk.Combobox(algo_frame, textvariable=self.algo_var,
                                       values=algos, state='readonly', width=18)
        self.algo_menu.pack()
        self.canvas.create_window(cx, cy + 20, window=algo_frame, anchor='nw')

        # Obstacles toggle
        self.obs_var = tk.BooleanVar(value=False)
        obs_frame = tk.Frame(self.root, bg=C_BG)
        self.obs_check = tk.Checkbutton(obs_frame, text="Enable Obstacles (Phase 2)",
                                         variable=self.obs_var, command=self._toggle_obstacles,
                                         bg=C_BG, fg=C_TEXT, selectcolor=C_BG,
                                         activebackground=C_BG, activeforeground=C_ACCENT,
                                         font=("Helvetica", 9))
        self.obs_check.pack(anchor='w')
        self.canvas.create_window(cx, cy + 60, window=obs_frame, anchor='nw')

        # Speed slider
        self.canvas.create_text(cx, cy + 95, text="Speed", font=("Helvetica", 10, "bold"),
                                fill=C_TEXT, anchor='nw')
        speed_frame = tk.Frame(self.root, bg=C_BG)
        self.speed_scale = tk.Scale(speed_frame, from_=1, to=50, orient='horizontal',
                                     length=170, bg=C_BG, fg=C_TEXT, troughcolor='#4A5568',
                                     highlightthickness=0, command=self._update_speed,
                                     font=("Helvetica", 8))
        self.speed_scale.set(5)
        self.canvas.create_text(cx, cy + 145,
                                font=("Helvetica", 8), fill='#718096', anchor='nw')
        self.speed_scale.pack()
        self.canvas.create_window(cx, cy + 115, window=speed_frame, anchor='nw')

        # Buttons
        btn_frame = tk.Frame(self.root, bg=C_BG)
        self.start_btn = tk.Button(btn_frame, text="▶  START", command=self._start,
                                    bg='#48BB78', fg='white', font=("Helvetica", 11, "bold"),
                                    width=10, relief='flat', cursor='hand2')
        self.start_btn.pack(pady=3)
        self.reset_btn = tk.Button(btn_frame, text="↺  RESET", command=self._reset,
                                    bg='#E53E3E', fg='white', font=("Helvetica", 11, "bold"),
                                    width=10, relief='flat', cursor='hand2')
        self.reset_btn.pack(pady=3)
        self.canvas.create_window(cx, cy + 175, window=btn_frame, anchor='nw')

        # Stats display
        self.stats_text_id = self.canvas.create_text(cx, cy + 280,
            text="", font=("Courier", 9), fill=C_TEXT, anchor='nw', width=190)

        # Legend
        legend_y = cy + 380
        self.canvas.create_text(cx, legend_y, text="Legend", font=("Helvetica", 10, "bold"),
                                fill=C_TEXT, anchor='nw')
        legend_items = [
            (C_START, "Start (Robot)"),
            (C_GOAL, "Goal"),
            (C_CURRENT, "Current Node"),
            (C_FRONTIER, "Frontier/Queued"),
            (C_EXPLORED, "Explored"),
            (C_PATH, "Final Path"),
            (C_OBSTACLE, "Obstacle"),
        ]
        for i, (color, label) in enumerate(legend_items):
            y = legend_y + 22 + i * 22
            self.canvas.create_rectangle(cx, y, cx + 16, y + 14, fill=color, outline='#4A5568')
            self.canvas.create_text(cx + 22, y + 7, text=label, font=("Helvetica", 9),
                                    fill=C_TEXT, anchor='w')

    def _draw_grid(self):
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                x1 = MARGIN_LEFT + c * CELL_SIZE
                y1 = MARGIN_TOP + r * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE
                color = C_FREE
                if self.obstacles_on and (r, c) in self.obstacles:
                    color = C_OBSTACLE
                if (r, c) == START:
                    color = C_START
                elif (r, c) == GOAL:
                    color = C_GOAL
                rect_id = self.canvas.create_rectangle(x1, y1, x2, y2,
                    fill=color, outline='#CBD5E0', width=0.5)
                self.cell_rects[(r, c)] = rect_id
                self.cell_state[(r, c)] = 'free'

        # Start/Goal labels
        sx = MARGIN_LEFT + START[1] * CELL_SIZE + CELL_SIZE // 2
        sy = MARGIN_TOP + START[0] * CELL_SIZE - 8
        self.canvas.create_text(sx, sy, text="S", font=("Helvetica", 9, "bold"), fill=C_START)

        gx = MARGIN_LEFT + GOAL[1] * CELL_SIZE + CELL_SIZE // 2
        gy = MARGIN_TOP + GOAL[0] * CELL_SIZE + CELL_SIZE + 8
        self.canvas.create_text(gx, gy, text="G", font=("Helvetica", 9, "bold"), fill=C_GOAL)

    def _toggle_obstacles(self):
        self.obstacles_on = self.obs_var.get()
        self._reset()

    def _update_speed(self, val):
        v = int(val)
        self.speed = max(1, 51 - v)  # Invert: higher slider = faster
        self.steps_per_frame = max(1, v // 5)

    def _build_grid_array(self):
        self.grid = [[0]*GRID_COLS for _ in range(GRID_ROWS)]
        if self.obstacles_on:
            for (r, c) in self.obstacles:
                if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS:
                    self.grid[r][c] = 1

    def _start(self):
        if self.running:
            return
        self._reset_visuals()
        self._build_grid_array()
        self.algo_name = self.algo_var.get()

        algo_map = {
            "BFS": lambda: bfs_gen(self.grid, START, GOAL),
            "DFS": lambda: dfs_gen(self.grid, START, GOAL),
            "Dijkstra": lambda: dijkstra_gen(self.grid, START, GOAL),
            "A* (Manhattan)": lambda: astar_gen(self.grid, START, GOAL, 'manhattan'),
            "A* (Euclidean)": lambda: astar_gen(self.grid, START, GOAL, 'euclidean'),
            "RRT": lambda: rrt_gen(self.grid, START, GOAL),
        }

        self.generator = algo_map[self.algo_name]()
        self.running = True
        self.start_btn.config(state='disabled')
        self.t_start = time.time()
        self._step()

    def _step(self):
        if not self.running or self.generator is None:
            return

        try:
            for _ in range(self.steps_per_frame):
                action, data = next(self.generator)

                if action == "explore":
                    r, c = data
                    if (r, c) != START and (r, c) != GOAL:
                        self.canvas.itemconfig(self.cell_rects[(r, c)], fill=C_EXPLORED)
                        self.cell_state[(r, c)] = 'explored'

                elif action == "frontier":
                    r, c = data
                    if (r, c) != START and (r, c) != GOAL:
                        self.canvas.itemconfig(self.cell_rects[(r, c)], fill=C_FRONTIER)
                        self.cell_state[(r, c)] = 'frontier'

                elif action == "current":
                    r, c = data
                    if (r, c) != START and (r, c) != GOAL:
                        self.canvas.itemconfig(self.cell_rects[(r, c)], fill=C_CURRENT)
                        # Revert after brief moment handled by next explore

                elif action == "path":
                    path = data
                    for (r, c) in path:
                        if (r, c) != START and (r, c) != GOAL:
                            self.canvas.itemconfig(self.cell_rects[(r, c)], fill=C_PATH)
                    # Draw path line
                    if len(path) > 1:
                        coords = []
                        for (r, c) in path:
                            x = MARGIN_LEFT + c * CELL_SIZE + CELL_SIZE // 2
                            y = MARGIN_TOP + r * CELL_SIZE + CELL_SIZE // 2
                            coords.extend([x, y])
                        line_id = self.canvas.create_line(*coords, fill='#DD6B20',
                                                          width=3, smooth=True)
                        self.path_line_ids.append(line_id)

                elif action == "done":
                    self.stats = data
                    self.stats['time'] = (time.time() - self.t_start) * 1000
                    self.running = False
                    self.start_btn.config(state='normal')
                    self._update_info()
                    return

            self._update_info()
            self.root.after(self.speed, self._step)

        except StopIteration:
            self.running = False
            self.start_btn.config(state='normal')

    def _reset(self):
        self.running = False
        self.generator = None
        self._reset_visuals()
        self.stats = {"explored": 0, "path_len": 0}
        self.start_btn.config(state='normal')
        self._update_info()

    def _reset_visuals(self):
        for line_id in self.path_line_ids:
            self.canvas.delete(line_id)
        self.path_line_ids = []

        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if (r, c) == START:
                    color = C_START
                elif (r, c) == GOAL:
                    color = C_GOAL
                elif self.obstacles_on and (r, c) in self.obstacles:
                    color = C_OBSTACLE
                else:
                    color = C_FREE
                self.canvas.itemconfig(self.cell_rects[(r, c)], fill=color)
                self.cell_state[(r, c)] = 'free'

    def _update_info(self):
        phase = "Phase 2 (Obstacles)" if self.obstacles_on else "Phase 1 (No Obstacles)"
        status = "RUNNING..." if self.running else "READY"
        if self.stats.get('path_len', 0) > 0 and not self.running:
            status = "COMPLETE ✓"
        elif self.stats.get('explored', 0) > 0 and self.stats.get('path_len', 0) == 0 and not self.running:
            status = "NO PATH FOUND"

        info = (
            f"Algorithm: {self.algo_name}\n"
            f"{phase}\n"
            f"Status: {status}\n"
            f"─────────────────\n"
            f"Explored:  {self.stats.get('explored', 0)}\n"
            f"Path Len:  {self.stats.get('path_len', 0)}\n"
        )
        if 'time' in self.stats:
            info += f"Time:      {self.stats['time']:.1f} ms\n"

        self.canvas.itemconfig(self.stats_text_id, text=info)

if __name__ == "__main__":
    root = tk.Tk()
    app = PathfindingApp(root)
    root.mainloop()