"""
Generate animated GIF previews of each pathfinding algorithm.
Shows the step-by-step exploration and final path discovery.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import deque
import heapq
import math
import random
import io

GRID_ROWS = 25
GRID_COLS = 25
CELL = 22
PAD = 30
START = (1, 1)
GOAL = (23, 23)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Colors (RGB)
C_BG       = (26, 32, 44)
C_FREE     = (237, 242, 247)
C_OBS      = (45, 55, 72)
C_EXPLORED = (144, 205, 244)
C_FRONTIER = (99, 179, 237)
C_PATH     = (246, 173, 85)
C_START    = (72, 187, 120)
C_GOAL     = (252, 129, 129)
C_CURRENT  = (229, 62, 62)
C_PATHLINE = (221, 107, 32)
C_TEXT     = (226, 232, 240)
C_DIM      = (113, 128, 150)

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
    obs.discard(START); obs.discard(GOAL)
    return obs

def _recon(parent, end):
    if end not in parent: return []
    path = []; node = end
    while node is not None:
        path.append(node); node = parent[node]
    path.reverse(); return path

# ── Generators ──

def bfs_gen(grid, s, g):
    visited = {s}; parent = {s: None}; q = deque([s])
    while q:
        n = q.popleft()
        yield ('e', n)
        if n == g: yield ('p', _recon(parent, g)); return
        for dr, dc in DIRECTIONS:
            nr, nc = n[0]+dr, n[1]+dc
            if 0<=nr<GRID_ROWS and 0<=nc<GRID_COLS and (nr,nc) not in visited and grid[nr][nc]==0:
                visited.add((nr,nc)); parent[(nr,nc)] = n; q.append((nr,nc))
                yield ('f', (nr,nc))

def dfs_gen(grid, s, g):
    visited = {s}; parent = {s: None}; st = [s]
    while st:
        n = st.pop()
        yield ('e', n)
        if n == g: yield ('p', _recon(parent, g)); return
        for dr, dc in DIRECTIONS:
            nr, nc = n[0]+dr, n[1]+dc
            if 0<=nr<GRID_ROWS and 0<=nc<GRID_COLS and (nr,nc) not in visited and grid[nr][nc]==0:
                visited.add((nr,nc)); parent[(nr,nc)] = n; st.append((nr,nc))
                yield ('f', (nr,nc))

def dijkstra_gen(grid, s, g):
    dist = {s: 0}; parent = {s: None}; heap = [(0,s[0],s[1])]; closed = set()
    while heap:
        d,r,c = heapq.heappop(heap); n=(r,c)
        if n in closed: continue
        closed.add(n); yield ('e', n)
        if n == g: yield ('p', _recon(parent, g)); return
        for dr, dc in DIRECTIONS:
            nr, nc = r+dr, c+dc
            if 0<=nr<GRID_ROWS and 0<=nc<GRID_COLS and (nr,nc) not in closed and grid[nr][nc]==0:
                nd = d+1
                if nd < dist.get((nr,nc), float('inf')):
                    dist[(nr,nc)] = nd; parent[(nr,nc)] = n
                    heapq.heappush(heap, (nd,nr,nc)); yield ('f', (nr,nc))

def astar_gen(grid, s, g, manhattan=True):
    h = (lambda p: abs(p[0]-g[0])+abs(p[1]-g[1])) if manhattan else \
        (lambda p: math.sqrt((p[0]-g[0])**2+(p[1]-g[1])**2))
    gs = {s: 0}; parent = {s: None}; heap = [(h(s),s[0],s[1])]; closed = set()
    while heap:
        f,r,c = heapq.heappop(heap); n=(r,c)
        if n in closed: continue
        closed.add(n); yield ('e', n)
        if n == g: yield ('p', _recon(parent, g)); return
        for dr, dc in DIRECTIONS:
            nr, nc = r+dr, c+dc
            if 0<=nr<GRID_ROWS and 0<=nc<GRID_COLS and (nr,nc) not in closed and grid[nr][nc]==0:
                ng = gs[n]+1
                if ng < gs.get((nr,nc), float('inf')):
                    gs[(nr,nc)] = ng; parent[(nr,nc)] = n
                    heapq.heappush(heap, (ng+h((nr,nc)),nr,nc)); yield ('f', (nr,nc))

def rrt_gen(grid, s, g):
    rng = random.Random(42)
    tree = {s: None}; nodes = [s]
    for _ in range(5000):
        rp = g if rng.random()<0.10 else (rng.randint(0,GRID_ROWS-1), rng.randint(0,GRID_COLS-1))
        nearest = min(nodes, key=lambda n: abs(n[0]-rp[0])+abs(n[1]-rp[1]))
        dr, dc = rp[0]-nearest[0], rp[1]-nearest[1]
        if dr==0 and dc==0: continue
        sr = (1 if dr>0 else -1) if abs(dr)>=abs(dc) else 0
        sc = (1 if dc>0 else -1) if abs(dc)>abs(dr) else 0
        if sr==0 and sc==0: sr = (1 if dr>0 else -1) if dr!=0 else 0; sc = (1 if dc>0 else -1) if dc!=0 else 0
        nn = (nearest[0]+sr, nearest[1]+sc)
        if 0<=nn[0]<GRID_ROWS and 0<=nn[1]<GRID_COLS and grid[nn[0]][nn[1]]==0 and nn not in tree:
            tree[nn] = nearest; nodes.append(nn); yield ('e', nn)
            if nn == g: yield ('p', _recon(tree, g)); return
            if abs(nn[0]-g[0])+abs(nn[1]-g[1])==1 and grid[g[0]][g[1]]==0 and g not in tree:
                tree[g] = nn; yield ('e', g); yield ('p', _recon(tree, g)); return

# ── Frame renderer ──

def render_frame(grid, explored, frontier, current, path, algo_name, phase_name, step_num):
    w = CELL * GRID_COLS + 2 * PAD
    h = CELL * GRID_ROWS + PAD + 50
    img = Image.new('RGB', (w, h), C_BG)
    draw = ImageDraw.Draw(img)

    # Title
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        font = ImageFont.load_default()
        font_sm = font

    title = f"{algo_name}  |  {phase_name}  |  Step: {step_num}  |  Explored: {len(explored)}"
    draw.text((PAD, 8), title, fill=C_TEXT, font=font)

    if path:
        draw.text((PAD, 26), f"PATH FOUND! Length: {len(path)}", fill=C_PATH, font=font_sm)
    
    oy = 45

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            x1 = PAD + c * CELL
            y1 = oy + r * CELL
            col = C_FREE
            if grid[r][c] == 1:
                col = C_OBS
            elif (r,c) in path:
                col = C_PATH
            elif (r,c) == current:
                col = C_CURRENT
            elif (r,c) in explored:
                col = C_EXPLORED
            elif (r,c) in frontier:
                col = C_FRONTIER

            if (r,c) == START: col = C_START
            elif (r,c) == GOAL: col = C_GOAL

            draw.rectangle([x1, y1, x1+CELL-1, y1+CELL-1], fill=col, outline=(160,174,192))

    # Path line
    if len(path) > 1:
        coords = [(PAD + c*CELL + CELL//2, oy + r*CELL + CELL//2) for r,c in path]
        draw.line(coords, fill=C_PATHLINE, width=3)

    # Start/Goal markers
    sx, sy = PAD + START[1]*CELL + CELL//2, oy + START[0]*CELL + CELL//2
    draw.ellipse([sx-6, sy-6, sx+6, sy+6], fill=C_START, outline='white')
    gx, gy = PAD + GOAL[1]*CELL + CELL//2, oy + GOAL[0]*CELL + CELL//2
    draw.ellipse([gx-6, gy-6, gx+6, gy+6], fill=C_GOAL, outline='white')

    return img


def generate_gif(algo_name, gen_func, grid, phase_name, filename, frame_skip=3):
    """Generate an animated GIF from algorithm generator."""
    print(f"  Generating {algo_name} ({phase_name})...")
    
    explored_set = set()
    frontier_set = set()
    current = None
    path_set = set()
    path_list = []
    frames = []
    step = 0

    gen = gen_func(grid, START, GOAL) if 'astar' not in str(gen_func) else gen_func

    for action, data in gen:
        if action == 'e':
            explored_set.add(data)
            frontier_set.discard(data)
            current = data
        elif action == 'f':
            frontier_set.add(data)
        elif action == 'p':
            path_list = data
            path_set = set(data)

        step += 1
        if step % frame_skip == 0 or action == 'p':
            frame = render_frame(grid, explored_set, frontier_set, current,
                               path_set, algo_name, phase_name, step)
            frames.append(frame)

    # Hold final frame longer
    if frames:
        for _ in range(15):
            frames.append(frames[-1])

    if frames:
        frames[0].save(filename, save_all=True, append_images=frames[1:],
                       duration=50, loop=0, optimize=True)
        print(f"    -> {filename} ({len(frames)} frames)")
    return len(frames)


# ═══════════════════════════ MAIN ═════════════════════════════════════

def main():
    print("=" * 60)
    print("  Generating Animated GIF Previews")
    print("=" * 60)

    obstacles = generate_obstacles()

    for phase_idx, (phase_name, use_obs) in enumerate([("Phase 1 - No Obstacles", False),
                                                         ("Phase 2 - Obstacles", True)]):
        grid = [[0]*GRID_COLS for _ in range(GRID_ROWS)]
        if use_obs:
            for r, c in obstacles:
                if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS:
                    grid[r][c] = 1

        print(f"\n{phase_name}:")
        
        algos = [
            ("BFS", lambda g, s, go: bfs_gen(g, s, go)),
            ("DFS", lambda g, s, go: dfs_gen(g, s, go)),
            ("Dijkstra", lambda g, s, go: dijkstra_gen(g, s, go)),
            ("A* Manhattan", lambda g, s, go: astar_gen(g, s, go, True)),
            ("A* Euclidean", lambda g, s, go: astar_gen(g, s, go, False)),
            ("RRT", lambda g, s, go: rrt_gen(g, s, go)),
        ]

        for algo_name, gen_factory in algos:
            gen = gen_factory(grid, START, GOAL)
            safe_name = algo_name.replace(" ", "_").replace("*", "star")
            fname = f"anim_p{phase_idx+1}_{safe_name}.gif"

            explored_set = set()
            frontier_set = set()
            current = None
            path_set = set()
            frames = []
            step = 0
            skip = 4 if algo_name != "RRT" else 2

            for action, data in gen:
                if action == 'e':
                    explored_set.add(data)
                    frontier_set.discard(data)
                    current = data
                elif action == 'f':
                    frontier_set.add(data)
                elif action == 'p':
                    path_set = set(data)

                step += 1
                if step % skip == 0 or action == 'p':
                    frame = render_frame(grid, explored_set, frontier_set, current,
                                       path_set, algo_name, phase_name, step)
                    frames.append(frame)

            if frames:
                for _ in range(15):
                    frames.append(frames[-1])
                frames[0].save(fname, save_all=True, append_images=frames[1:],
                               duration=50, loop=0, optimize=True)
                print(f"  {algo_name}: {fname} ({len(frames)} frames)")

    # Also generate a combined "best of" GIF with BFS and A* side by side for Phase 2
    print("\nGenerating combined showcase GIF (A* Manhattan Phase 2)...")
    grid = [[0]*GRID_COLS for _ in range(GRID_ROWS)]
    for r, c in obstacles:
        if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS:
            grid[r][c] = 1

    gen = astar_gen(grid, START, GOAL, True)
    explored_set = set(); frontier_set = set(); current = None; path_set = set()
    frames = []; step = 0
    for action, data in gen:
        if action == 'e':
            explored_set.add(data); frontier_set.discard(data); current = data
        elif action == 'f': frontier_set.add(data)
        elif action == 'p': path_set = set(data)
        step += 1
        if step % 2 == 0 or action == 'p':
            frame = render_frame(grid, explored_set, frontier_set, current,
                               path_set, "A* Manhattan", "Phase 2 - Obstacles", step)
            frames.append(frame)
    for _ in range(20): frames.append(frames[-1])
    fname = "showcase_astar.gif"
    frames[0].save(fname, save_all=True, append_images=frames[1:],
                   duration=40, loop=0, optimize=True)
    print(f"  Showcase: {fname} ({len(frames)} frames)")

    print("\nDone! All GIFs generated.")


if __name__ == "__main__":
    main()