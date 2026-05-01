import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from collections import deque
import heapq
import random
import math
import time
import os
import glob

GRID_ROWS = 25
GRID_COLS = 25
START = (1, 1)
END = (23, 23)


def generate_obstacles():
    """Generate a set of obstacle cells for Phase 2."""
    obstacles = set()
    
    # Wall 1 – horizontal
    for c in range(4, 18):
        obstacles.add((5, c))
    # Wall 2 – vertical
    for r in range(2, 14):
        obstacles.add((r, 20))
    # Wall 3 – horizontal
    for c in range(8, 23):
        obstacles.add((12, c))
    # Wall 4 – vertical
    for r in range(14, 23):
        obstacles.add((r, 8))
    # Wall 5 – diagonal-ish block
    for r in range(16, 21):
        for c in range(14, 17):
            obstacles.add((r, c))
    # Wall 6 – small horizontal
    for c in range(1, 7):
        obstacles.add((19, c))
    # Wall 7 – vertical near end
    for r in range(18, 25):
        obstacles.add((r, 21))
    # Scattered blocks
    for r in range(7, 11):
        obstacles.add((r, 4))
    for c in range(10, 14):
        obstacles.add((9, c))
    
    # Make sure start and end are never obstacles
    obstacles.discard(START)
    obstacles.discard(END)
    return obstacles


DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

def bfs(grid, start, end):
    """Breadth-First Search explores level by level."""
    rows, cols = grid.shape
    visited = set()
    visited.add(start)
    parent = {start: None}
    queue = deque([start])
    exploration_order = []

    while queue:
        node = queue.popleft()
        exploration_order.append(node)
        if node == end:
            break
        for dr, dc in DIRECTIONS:
            nr, nc = node[0] + dr, node[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and grid[nr][nc] == 0:
                visited.add((nr, nc))
                parent[(nr, nc)] = node
                queue.append((nr, nc))

    path = reconstruct_path(parent, end)
    return path, exploration_order


def dfs(grid, start, end):
    """Depth-First Search explores as deep as possible first."""
    rows, cols = grid.shape
    visited = set()
    visited.add(start)
    parent = {start: None}
    stack = [start]
    exploration_order = []

    while stack:
        node = stack.pop()
        exploration_order.append(node)
        if node == end:
            break
        for dr, dc in DIRECTIONS:
            nr, nc = node[0] + dr, node[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and grid[nr][nc] == 0:
                visited.add((nr, nc))
                parent[(nr, nc)] = node
                stack.append((nr, nc))

    path = reconstruct_path(parent, end)
    return path, exploration_order


def dijkstra(grid, start, end):
    """Dijkstra's Algorithm uniform cost (weight=1 on grid)."""
    rows, cols = grid.shape
    dist = {start: 0}
    parent = {start: None}
    heap = [(0, start)]
    visited = set()
    exploration_order = []

    while heap:
        d, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        exploration_order.append(node)
        if node == end:
            break
        for dr, dc in DIRECTIONS:
            nr, nc = node[0] + dr, node[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and grid[nr][nc] == 0:
                new_dist = d + 1
                if new_dist < dist.get((nr, nc), float('inf')):
                    dist[(nr, nc)] = new_dist
                    parent[(nr, nc)] = node
                    heapq.heappush(heap, (new_dist, (nr, nc)))

    path = reconstruct_path(parent, end)
    return path, exploration_order


def astar_manhattan(grid, start, end):
    """A* with Manhattan distance heuristic."""
    return _astar(grid, start, end, heuristic='manhattan')


def astar_euclidean(grid, start, end):
    """A* with Euclidean distance heuristic."""
    return _astar(grid, start, end, heuristic='euclidean')


def _astar(grid, start, end, heuristic='manhattan'):
    """Generic A* implementation."""
    rows, cols = grid.shape

    def h(node):
        if heuristic == 'manhattan':
            return abs(node[0] - end[0]) + abs(node[1] - end[1])
        else:
            return math.sqrt((node[0] - end[0]) ** 2 + (node[1] - end[1]) ** 2)

    g_score = {start: 0}
    f_score = {start: h(start)}
    parent = {start: None}
    open_set = [(f_score[start], start)]
    closed_set = set()
    exploration_order = []

    while open_set:
        _, node = heapq.heappop(open_set)
        if node in closed_set:
            continue
        closed_set.add(node)
        exploration_order.append(node)
        if node == end:
            break
        for dr, dc in DIRECTIONS:
            nr, nc = node[0] + dr, node[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in closed_set and grid[nr][nc] == 0:
                tentative_g = g_score[node] + 1
                if tentative_g < g_score.get((nr, nc), float('inf')):
                    g_score[(nr, nc)] = tentative_g
                    f_score[(nr, nc)] = tentative_g + h((nr, nc))
                    parent[(nr, nc)] = node
                    heapq.heappush(open_set, (f_score[(nr, nc)], (nr, nc)))

    path = reconstruct_path(parent, end)
    return path, exploration_order


def rrt(grid, start, end, max_iter=5000, step_size=1):
    """Rapidly-exploring Random Tree (grid-adapted)."""
    rows, cols = grid.shape
    tree = {start: None}
    nodes = [start]
    exploration_order = []

    for _ in range(max_iter):
        # Bias towards goal 10% of the time
        if random.random() < 0.10:
            rand_point = end
        else:
            rand_point = (random.randint(0, rows - 1), random.randint(0, cols - 1))

        # Find nearest node in tree
        nearest = min(nodes, key=lambda n: abs(n[0] - rand_point[0]) + abs(n[1] - rand_point[1]))

        # Step towards random point
        dr = rand_point[0] - nearest[0]
        dc = rand_point[1] - nearest[1]
        dist = max(abs(dr), abs(dc))
        if dist == 0:
            continue
        # Normalize step
        sr = int(np.sign(dr)) if abs(dr) >= abs(dc) else 0
        sc = int(np.sign(dc)) if abs(dc) > abs(dr) else 0
        if sr == 0 and sc == 0:
            sr = int(np.sign(dr)) if dr != 0 else 0
            sc = int(np.sign(dc)) if dc != 0 else 0

        new_node = (nearest[0] + sr, nearest[1] + sc)

        if 0 <= new_node[0] < rows and 0 <= new_node[1] < cols and grid[new_node[0]][new_node[1]] == 0:
            if new_node not in tree:
                tree[new_node] = nearest
                nodes.append(new_node)
                exploration_order.append(new_node)

                if new_node == end:
                    break
                # Also check if we're adjacent to the end
                if abs(new_node[0] - end[0]) + abs(new_node[1] - end[1]) == 1 and grid[end[0]][end[1]] == 0:
                    if end not in tree:
                        tree[end] = new_node
                        nodes.append(end)
                        exploration_order.append(end)
                    break

    path = reconstruct_path(tree, end)
    return path, exploration_order


def reconstruct_path(parent, end):
    """Trace back the path from end to start using the parent map."""
    if end not in parent:
        return []
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path

def build_grid(obstacles=None):
    """Build a 2D numpy grid. 0 = free, 1 = obstacle."""
    grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    if obstacles:
        for (r, c) in obstacles:
            if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS:
                grid[r][c] = 1
    return grid

# Color palette
C_FREE       = '#F0F4F8'   # Light gray-blue  (free cell)
C_OBSTACLE   = '#2D3748'   # Dark slate       (obstacle)
C_EXPLORED   = '#BEE3F8'   # Light blue       (explored)
C_PATH       = '#F6AD55'   # Orange           (final path)
C_START      = '#48BB78'   # Green            (start)
C_END        = '#FC8181'   # Red              (end)
C_GRID_LINE  = '#CBD5E0'   # Gray             (grid lines)


def plot_single(ax, grid, path, explored, title, start, end):
    """Draw a single algorithm result on a matplotlib axes."""
    rows, cols = grid.shape

    # Build color matrix
    display = np.zeros((rows, cols, 3))
    c_free = np.array(mcolors.to_rgb(C_FREE))
    c_obs  = np.array(mcolors.to_rgb(C_OBSTACLE))
    c_exp  = np.array(mcolors.to_rgb(C_EXPLORED))
    c_path = np.array(mcolors.to_rgb(C_PATH))
    c_start = np.array(mcolors.to_rgb(C_START))
    c_end  = np.array(mcolors.to_rgb(C_END))

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                display[r][c] = c_obs
            else:
                display[r][c] = c_free

    # Color explored cells
    explored_set = set(explored)
    for (r, c) in explored_set:
        if grid[r][c] == 0:
            display[r][c] = c_exp

    # Color path cells
    path_set = set(path)
    for (r, c) in path_set:
        display[r][c] = c_path

    # Start and end always on top
    display[start[0]][start[1]] = c_start
    display[end[0]][end[1]] = c_end

    ax.imshow(display, origin='upper', interpolation='nearest')

    # Draw grid lines
    for r in range(rows + 1):
        ax.axhline(r - 0.5, color=C_GRID_LINE, linewidth=0.3)
    for c in range(cols + 1):
        ax.axvline(c - 0.5, color=C_GRID_LINE, linewidth=0.3)

    # Draw path line
    if path:
        path_cols = [p[1] for p in path]
        path_rows = [p[0] for p in path]
        ax.plot(path_cols, path_rows, color='#DD6B20', linewidth=2.0, alpha=0.9, zorder=5)

    # Robot at start
    ax.plot(start[1], start[0], marker='o', color=C_START, markersize=10,
            markeredgecolor='white', markeredgewidth=1.5, zorder=10)
    ax.text(start[1], start[0] - 1.3, 'ROBOT START', ha='center', va='center',
            fontsize=6, fontweight='bold', color='#276749')

    # Flag at end
    ax.plot(end[1], end[0], marker='*', color=C_END, markersize=14,
            markeredgecolor='white', markeredgewidth=1.0, zorder=10)
    ax.text(end[1], end[0] + 1.5, 'GOAL', ha='center', va='center',
            fontsize=6, fontweight='bold', color='#C53030')

    # Stats
    path_len = len(path) if path else 0
    explored_count = len(explored)
    status = f"Path: {path_len} cells  |  Explored: {explored_count} cells"
    if not path:
        status = "NO PATH FOUND!"
    ax.set_title(f"{title}\n{status}", fontsize=9, fontweight='bold', pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)


def create_legend_ax(ax):
    """Draw a legend on a given axes."""
    ax.axis('off')
    legend_items = [
        (C_START, 'Start (Robot)'),
        (C_END, 'Goal'),
        (C_EXPLORED, 'Explored Cells'),
        (C_PATH, 'Final Path'),
        (C_OBSTACLE, 'Obstacle'),
        (C_FREE, 'Free Cell'),
    ]
    patches = [mpatches.Patch(facecolor=color, edgecolor='#718096', label=label, linewidth=0.5)
               for color, label in legend_items]
    ax.legend(handles=patches, loc='center', fontsize=8, frameon=True,
              fancybox=True, shadow=True, ncol=3, handlelength=1.5,
              edgecolor='#A0AEC0', facecolor='#FFFFFF')


def run_and_visualize(phase_name, grid, obstacles_present):
    """Run all algorithms on a grid and produce a visualization figure."""
    algorithms = [
        ("BFS", bfs),
        ("DFS", dfs),
        ("Dijkstra", dijkstra),
        ("A* (Manhattan)", astar_manhattan),
        ("A* (Euclidean)", astar_euclidean),
        ("RRT", rrt),
    ]

    random.seed()  # Reproducibility for RRT

    results = {}
    for name, algo in algorithms:
        t0 = time.time()
        path, explored = algo(grid, START, END)
        elapsed = time.time() - t0
        results[name] = {
            'path': path,
            'explored': explored,
            'time': elapsed,
            'path_len': len(path),
            'explored_count': len(explored),
        }

    # ── Create figure ──
    fig = plt.figure(figsize=(22, 16), facecolor='#FFFFFF')
    fig.suptitle(
        f"ROBOT PATHFINDING VISUALIZER  —  {phase_name}",
        fontsize=18, fontweight='bold', color='#1A202C', y=0.97
    )
    subtitle = (f"Grid: {GRID_ROWS}×{GRID_COLS}  |  "
                f"Start: {START}  →  End: {END}  |  "
                f"Obstacles: {'YES' if obstacles_present else 'NONE'}")
    fig.text(0.5, 0.94, subtitle, ha='center', fontsize=11, color='#4A5568')

    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.25,
                  top=0.90, bottom=0.12, left=0.04, right=0.96)

    for idx, (name, _) in enumerate(algorithms):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])
        res = results[name]
        plot_single(ax, grid, res['path'], res['explored'], name, START, END)

    # ── Legend row ──
    ax_legend = fig.add_axes([0.05, 0.03, 0.9, 0.05])
    create_legend_ax(ax_legend)

    # ── Comparison stats table ──
    table_data = []
    for name, _ in algorithms:
        r = results[name]
        table_data.append([name, r['path_len'], r['explored_count'], f"{r['time']*1000:.1f} ms"])
    
    ax_table = fig.add_axes([0.15, 0.005, 0.7, 0.025])
    ax_table.axis('off')
    header = "  |  ".join([f"{n}: path={r['path_len']}, explored={r['explored_count']}"
                           for n, r in zip([a[0] for a in algorithms], [results[a[0]] for a in algorithms])])
    ax_table.text(0.5, 0.5, header, ha='center', va='center', fontsize=7, color='#4A5568',
                  family='monospace')

    return fig, results

def create_comparison_chart(results_phase1, results_phase2):
    """Create a bar chart comparing all algorithms across both phases."""
    algo_names = list(results_phase1.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), facecolor='#FFFFFF')
    fig.suptitle("ALGORITHM COMPARISON  —  Phase 1 vs Phase 2",
                 fontsize=16, fontweight='bold', color='#1A202C', y=1.02)

    x = np.arange(len(algo_names))
    width = 0.35

    # ── Path Length ──
    ax = axes[0]
    p1_vals = [results_phase1[n]['path_len'] for n in algo_names]
    p2_vals = [results_phase2[n]['path_len'] for n in algo_names]
    bars1 = ax.bar(x - width/2, p1_vals, width, label='Phase 1 (No Obstacles)',
                   color='#63B3ED', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, p2_vals, width, label='Phase 2 (Obstacles)',
                   color='#FC8181', edgecolor='white', linewidth=0.5)
    ax.set_title('Path Length (cells)', fontweight='bold', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(algo_names, rotation=30, ha='right', fontsize=8)
    ax.legend(fontsize=8)
    ax.set_ylabel('Cells')
    ax.grid(axis='y', alpha=0.3)
    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=7)

    # ── Explored Cells ──
    ax = axes[1]
    p1_vals = [results_phase1[n]['explored_count'] for n in algo_names]
    p2_vals = [results_phase2[n]['explored_count'] for n in algo_names]
    bars1 = ax.bar(x - width/2, p1_vals, width, label='Phase 1',
                   color='#63B3ED', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, p2_vals, width, label='Phase 2',
                   color='#FC8181', edgecolor='white', linewidth=0.5)
    ax.set_title('Explored Cells (search effort)', fontweight='bold', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(algo_names, rotation=30, ha='right', fontsize=8)
    ax.legend(fontsize=8)
    ax.set_ylabel('Cells')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=7)

    # ── Execution Time ──
    ax = axes[2]
    p1_vals = [results_phase1[n]['time'] * 1000 for n in algo_names]
    p2_vals = [results_phase2[n]['time'] * 1000 for n in algo_names]
    bars1 = ax.bar(x - width/2, p1_vals, width, label='Phase 1',
                   color='#63B3ED', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, p2_vals, width, label='Phase 2',
                   color='#FC8181', edgecolor='white', linewidth=0.5)
    ax.set_title('Execution Time', fontweight='bold', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(algo_names, rotation=30, ha='right', fontsize=8)
    ax.legend(fontsize=8)
    ax.set_ylabel('ms')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    return fig

def main():
    output_files = [
        'phase1_no_obstacles.png',
        'phase2_with_obstacles.png',
        'comparison_chart.png',
    ]
    deleted = 0
    for f in output_files:
        if os.path.exists(f):
            os.remove(f)
            deleted += 1
    if deleted:
        print(f"Cleaned up {deleted} image(s) from previous run.\n")

    print("=" * 70)
    print("  ROBOT PATHFINDING VISUALIZER")
    print("  Phase 1: No Obstacles  |  Phase 2: With Obstacles")
    print("  Algorithms: BFS, DFS, Dijkstra, A*(Manhattan), A*(Euclidean), RRT")
    print("=" * 70)

    # ── Phase 1: No Obstacles ──
    print("\n▶ Phase 1: Running algorithms on obstacle-free grid...")
    grid_phase1 = build_grid(obstacles=None)
    fig1, results1 = run_and_visualize("PHASE 1  —  No Obstacles", grid_phase1, obstacles_present=False)
    fig1.savefig('phase1_no_obstacles.png', dpi=150, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print("  ✅ Phase 1 saved → phase1_no_obstacles.png")

    # ── Phase 2: With Obstacles ──
    print("\n▶ Phase 2: Running algorithms with obstacles...")
    obstacles = generate_obstacles()
    grid_phase2 = build_grid(obstacles=obstacles)
    fig2, results2 = run_and_visualize("PHASE 2  —  With Obstacles", grid_phase2, obstacles_present=True)
    fig2.savefig('phase2_with_obstacles.png', dpi=150, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print("  ✅ Phase 2 saved → phase2_with_obstacles.png")

    # ── Comparison Chart ──
    print("\n▶ Generating comparison chart...")
    fig3 = create_comparison_chart(results1, results2)
    fig3.savefig('comparison_chart.png', dpi=150, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print("  ✅ Comparison chart saved → comparison_chart.png")

    # ── Print summary table ──
    print("\n" + "─" * 70)
    print(f"  {'Algorithm':<20} {'Phase':>8} {'Path Len':>10} {'Explored':>10} {'Time (ms)':>10}")
    print("─" * 70)
    for phase_label, results in [("Phase 1", results1), ("Phase 2", results2)]:
        for name, r in results.items():
            print(f"  {name:<20} {phase_label:>8} {r['path_len']:>10} {r['explored_count']:>10} {r['time']*1000:>10.2f}")
        print("─" * 70)

    print("\n✅ All done! 3 images generated.")
    plt.close('all')


if __name__ == "__main__":
    main()