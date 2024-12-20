import random
from collections import deque

# Set a random seed for reproducibility (optional)
# Uncomment the following line and set it to any integer to get repeatable results
# random.seed(42)

def initialize_grid(rows, cols):
    """
    Initializes a grid filled with empty spaces 'E', surrounded by walls 'W'.
    
    Parameters:
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
    
    Returns:
        list of list: The initialized grid.
    """
    grid = [['W' for _ in range(cols)] for _ in range(rows)]
    for x in range(1, rows-1):
        for y in range(1, cols-1):
            grid[x][y] = 'E'  # 'E' represents an empty space
    return grid

def place_start_goal(grid, start_pos, goal_pos):
    """
    Places the start 'S' and goal 'G' positions on the grid.
    
    Parameters:
        grid (list of list): The grid.
        start_pos (tuple): Coordinates of the start position (x, y).
        goal_pos (tuple): Coordinates of the goal position (x, y).
    """
    sx, sy = start_pos
    gx, gy = goal_pos
    grid[sx][sy] = 'S'
    grid[gx][gy] = 'G'

def generate_random_path_dfs(grid, start, goal):
    """
    Generates a path from the start to the goal using Depth-First Search (DFS).
    
    Parameters:
        grid (list of list): The grid.
        start (tuple): Coordinates of the start position (x, y).
        goal (tuple): Coordinates of the goal position (x, y).
    
    Returns:
        list of tuple: The path as a list of coordinates. Returns an empty list if no path is found.
    """
    rows = len(grid)
    cols = len(grid[0])
    stack = [start]
    came_from = {start: None}
    
    while stack:
        current = stack.pop()
        if current == goal:
            break
        x, y = current
        # Shuffle directions to ensure randomness
        directions = [(-1,0),(1,0),(0,-1),(0,1)]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)
            if 1 <= nx < rows-1 and 1 <= ny < cols-1:
                if grid[nx][ny] in ['E', 'G'] and neighbor not in came_from:
                    stack.append(neighbor)
                    came_from[neighbor] = current
    
    # If no path is found
    if goal not in came_from:
        return []
    
    # Reconstruct the path
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    
    print("Generated path:", path)  # Debug information
    return path

def add_key_and_door(grid, path):
    """
    Adds a key 'K' and a door 'D' to the path. The door can be adjacent or non-adjacent to the key.
    
    Parameters:
        grid (list of list): The grid.
        path (list of tuple): The main path as a list of coordinates.
    
    Returns:
        tuple: (success flag, key position, door position)
    """
    if len(path) < 4:
        # Path too short to place both key and door with non-adjacent options
        return False, None, None

    # Randomly choose the key position, not at the start or end
    key_index = random.randint(1, len(path) - 3)
    key_pos = path[key_index]
    grid[key_pos[0]][key_pos[1]] = 'K'
    print(f"Key position: {key_pos}")  # Debug information

    # Randomly choose the door position after the key, can be adjacent or not
    door_index = random.randint(key_index + 1, len(path) - 2)
    door_pos = path[door_index]
    grid[door_pos[0]][door_pos[1]] = 'D'
    print(f"Door position: {door_pos}")  # Debug information

    return True, key_pos, door_pos

def add_random_walls(grid, path, wall_prob=0.3):
    """
    Randomly adds walls 'W' to the grid, avoiding the main path, key, and door positions.
    
    Parameters:
        grid (list of list): The grid.
        path (list of tuple): The main path as a list of coordinates.
        wall_prob (float): Probability of adding a wall to an empty space.
    """
    path_set = set(path)
    rows = len(grid)
    cols = len(grid[0])
    for x in range(1, rows-1):
        for y in range(1, cols-1):
            if (x, y) in path_set:
                continue  # Do not add walls on the main path
            if grid[x][y] in ['S', 'G', 'K', 'D']:
                continue  # Do not add walls on key, door, start, or goal positions
            if grid[x][y] == 'E' and random.random() < wall_prob:
                grid[x][y] = 'W'  # 'W' represents a wall

def is_reachable(grid, start, goal, has_key=False):
    """
    Checks if the goal is reachable from the start using BFS. Considers the door and key.
    
    Parameters:
        grid (list of list): The grid.
        start (tuple): Coordinates of the start position (x, y).
        goal (tuple): Coordinates of the goal position (x, y).
        has_key (bool): Whether the key has already been obtained.
    
    Returns:
        bool: True if reachable, False otherwise.
    """
    rows = len(grid)
    cols = len(grid[0])
    queue = deque([(start, has_key)])
    visited = set()
    visited.add((start, has_key))
    
    while queue:
        (x, y), key = queue.popleft()
        if (x, y) == goal:
            return True
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:  # Up, Down, Left, Right
            nx, ny = x + dx, y + dy
            if 1 <= nx < rows-1 and 1 <= ny < cols-1:
                cell = grid[nx][ny]
                new_key = key
                if cell == 'W':
                    continue  # Cannot pass through walls
                if cell == 'D':
                    if not key:
                        continue  # Need key to pass through the door
                if cell == 'K':
                    new_key = True  # Acquire key
                state = ((nx, ny), new_key)
                if state not in visited:
                    visited.add(state)
                    queue.append((state[0], new_key))
    return False

def generate_valid_minigrid_with_key_door(rows, cols, start, goal, wall_prob=0.3, max_attempts=1000):
    """
    Generates a valid MiniGrid environment ensuring that the start can reach the goal by picking up a key and opening a door.
    
    Parameters:
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
        start (tuple): Coordinates of the start position (x, y).
        goal (tuple): Coordinates of the goal position (x, y).
        wall_prob (float): Probability of adding a wall to an empty space.
        max_attempts (int): Maximum number of attempts to generate a valid grid.
    
    Returns:
        list of list: The generated grid.
    
    Raises:
        ValueError: If a valid grid cannot be generated within the maximum number of attempts.
    """
    for attempt in range(max_attempts):
        grid = initialize_grid(rows, cols)
        place_start_goal(grid, start, goal)
        path = generate_random_path_dfs(grid, start, goal)
        if not path:
            print(f"Attempt {attempt+1}: Path generation failed.")
            continue  # Path generation failed, retry
        success, key_pos, door_pos = add_key_and_door(grid, path)
        if not success:
            print(f"Attempt {attempt+1}: Failed to add key and door.")
            continue  # Failed to add key and door, retry
        add_random_walls(grid, path, wall_prob)
        
        # Check connectivity from S to K
        if key_pos:
            reachable_s_k = is_reachable(grid, start, key_pos, has_key=False)
            if not reachable_s_k:
                print(f"Attempt {attempt+1}: Start to Key is not reachable.")
                continue  # S to K is not reachable, retry
            # Check connectivity from K to G
            reachable_k_g = is_reachable(grid, key_pos, goal, has_key=True)
            if not reachable_k_g:
                print(f"Attempt {attempt+1}: Key to Goal is not reachable.")
                continue  # K to G is not reachable, retry
        else:
            print(f"Attempt {attempt+1}: No key position found.")
            continue  # No key, retry

        # If all checks pass, return the grid
        print(f"Successfully generated environment on attempt {attempt+1}.")
        return grid
    raise ValueError("Failed to generate a valid environment within the maximum number of attempts. Please try increasing the grid size or adjusting the wall probability.")

def print_grid(grid):
    """
    Prints the grid in a readable format.
    
    Parameters:
        grid (list of list): The grid to print.
    """
    for row in grid:
        print(' '.join(row))
    print()

# Example usage
if __name__ == "__main__":
    # Optional: Set a random seed for reproducibility
    # Uncomment the following line to set the seed
    # random.seed(42)
    
    # Get grid dimensions from the user
    try:
        rows = int(input("Enter the number of rows for the grid (e.g., 10): "))
        cols = int(input("Enter the number of columns for the grid (e.g., 10): "))
    except ValueError:
        print("Please enter valid integers for the number of rows and columns.")
        exit(1)
    
    # Ensure the grid is large enough to accommodate walls, start, and goal
    if rows < 4 or cols < 4:
        print("Grid size too small. Please enter at least 4 rows and 4 columns to accommodate walls, start, and goal.")
        exit(1)
    
    # Define start and goal positions
    start_pos = (1, 1)
    goal_pos = (rows-2, cols-2)
    
    try:
        grid = generate_valid_minigrid_with_key_door(rows, cols, start_pos, goal_pos, wall_prob=0.3)
        print("Generated MiniGrid Environment:")
        print_grid(grid)
    except ValueError as e:
        print(e)
