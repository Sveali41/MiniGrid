"""
CrafterSymbolicEnv (Tensor + Render)
-----------------------------------
Outputs symbolic (H, W, 2) grid plus Crafter's RGB render.
- C0 = object ID
- C1 = direction ID (0–4)
"""

import numpy as np
import gym
from gym import spaces


# ---------------------------------------------------------------------
# 1. Object / Tile IDs
# ---------------------------------------------------------------------

TILE_ID = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4,
    5: 5, 6: 6, 7: 7, 8: 8,
}

ENTITY_ID = {
    "player": 10,
    "cow": 11,
    "zombie": 12,
    "skeleton": 13,
    "arrow": 14,
    "plant": 15,
    "fence": 16,
}

DIR_TO_ID = {
    (0, -1): 1,   # up
    (0,  1): 2,   # down
    (-1, 0): 3,   # left
    (1,  0): 4,   # right
}


# ---------------------------------------------------------------------
# 2. Adapter for old/new Crafter versions
# ---------------------------------------------------------------------

def get_engine(env):
    inner = getattr(env, "_env", env)
    if hasattr(inner, "engine"):
        return inner.engine
    if hasattr(inner, "_world"):
        class LegacyAdapter:
            def __init__(self, env):
                self._env = env
                self._world = env._world
            @property
            def tile_map(self): return self._world._mat_map
            @property
            def entities(self): return self._world.objects
            @property
            def player(self): return self._env._player
            @property
            def world_shape(self): return self._world._mat_map.shape
        return LegacyAdapter(inner)
    raise AttributeError("Unsupported Crafter API version.")


# ---------------------------------------------------------------------
# 3. Extract (H, W, 2) symbolic tensor
# ---------------------------------------------------------------------

def extract_tensor_grid(env):
    engine = get_engine(env)
    H, W = engine.world_shape
    grid = np.zeros((H, W, 2), dtype=np.int32)

    mat_map = engine.tile_map
    for y in range(H):
        for x in range(W):
            tile_val = int(mat_map[x, y]) if x < mat_map.shape[0] and y < mat_map.shape[1] else 0
            grid[y, x, 0] = TILE_ID.get(tile_val, 0)

    for ent in getattr(engine, "entities", []):
        ex, ey = int(ent.pos[0]), int(ent.pos[1])
        if not (0 <= ex < W and 0 <= ey < H):
            continue
        kind = type(ent).__name__.lower()
        grid[ey, ex, 0] = ENTITY_ID.get(kind, 0)
        if hasattr(ent, "facing"):
            dir_id = DIR_TO_ID.get(tuple(int(v) for v in ent.facing), 0)
            grid[ey, ex, 1] = dir_id

    px, py = map(int, engine.player.pos)
    grid[py, px, 0] = ENTITY_ID["player"]
    if hasattr(engine.player, "facing"):
        dir_id = DIR_TO_ID.get(tuple(int(v) for v in engine.player.facing), 0)
        grid[py, px, 1] = dir_id

    return grid


# ---------------------------------------------------------------------
# 4. CrafterSymbolicEnv
# ---------------------------------------------------------------------

class CrafterSymbolicEnv(gym.Env):
    """Crafter environment returning both symbolic tensor and RGB render."""

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, reward=False, seed=0):
        super().__init__()
        import crafter
        try:
            self.env = crafter.Env(reward=reward, seed=seed, render_mode="rgb_array")
        except TypeError:
            self.env = crafter.Env(reward=reward, seed=seed)

        engine = get_engine(self.env)
        H, W = engine.world_shape

        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=20, shape=(H, W, 2), dtype=np.int32),
            "rgb": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            "info": spaces.Dict({
                "health": spaces.Box(0, 10, shape=(), dtype=np.float32),
                "food": spaces.Box(0, 10, shape=(), dtype=np.float32),
                "drink": spaces.Box(0, 10, shape=(), dtype=np.float32),
                "energy": spaces.Box(0, 10, shape=(), dtype=np.float32),
            }),
        })
        self.action_space = self.env.action_space

    # -----------------------------------------------------------
    # Observation extraction
    # -----------------------------------------------------------
    def _extract_obs(self):
        grid = extract_tensor_grid(self.env)
        try:
            rgb = self.env.render(mode="rgb_array")
        except TypeError:
            rgb = self.env.render()
        engine = get_engine(self.env)
        player = engine.player
        info = {
            "health": float(player.health),
            "food": float(player.inventory.get("food", 0)),
            "drink": float(player.inventory.get("drink", 0)),
            "energy": float(player.inventory.get("energy", 0)),
        }
        return {"grid": grid, "rgb": rgb, "info": info}

    def reset(self, **kwargs):
        self.env.reset()
        return self._extract_obs()

    def step(self, action):
        _, reward, done, info_env = self.env.step(action)
        obs = self._extract_obs()
        info_env.update(obs["info"])
        return obs, reward, done, info_env

    def render(self, mode="human"):
        try:
            return self.env.render(mode=mode)
        except TypeError:
            return self.env.render()

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()


import numpy as np
import gym
from gym import spaces
import crafter
from crafter.engine import World
from crafter import constants, objects


# ------------------------------------------------------------
# 1. Character → Material / Entity mapping
# ------------------------------------------------------------
CHAR_TO_TILE = {
    '.': 'grass',
    'G': 'grass',
    'W': 'water',
    'T': 'tree',
    'R': 'stone',
    'C': 'coal',
    'I': 'iron',
    'O': 'diamond',  # use diamond as placeholder for gold
    'L': 'lava',
    'P': 'path',
    'S': 'sand',
}

CHAR_TO_ENTITY = {
    'A': objects.Player,
    'M': objects.Cow,
    'Z': objects.Zombie,
    'K': objects.Skeleton,
    't': objects.Plant,
    'F': objects.Fence,
}


# ------------------------------------------------------------
# 2. Build Crafter world from character grid (safe version)
# ------------------------------------------------------------
def make_world_from_chars(char_grid, seed=0):
    """Safe version: Create a Crafter world from a character grid, ensuring material ID alignment."""
    H, W = char_grid.shape

    # ---- 1. Create a temporary world to read official material mappings ----
    # Crafter automatically adds [None] + materials to align ID=0 as empty
    from crafter.engine import World
    from crafter import constants
    dummy_world = World(area=(1, 1), materials=list(constants.materials), chunk_size=(12, 12))

    # Official internal mapping (includes None)
    MATERIAL_NAME_TO_ID = dummy_world._mat_ids
    # e.g. {None: 0, 'water': 1, 'grass': 2, 'stone': 3, ...}

    # ---- 2. Create the actual world with the same structure ----
    world = World(area=(H, W), materials=list(constants.materials), chunk_size=(12, 12))
    world.daylight = 1.0

    # ---- 3. Fill the material map from character layout ----
    for y in range(H):
        for x in range(W):
            ch = char_grid[y, x]
            mat_name = CHAR_TO_TILE.get(ch, 'grass')  # Default to grass if unknown
            mat_id = MATERIAL_NAME_TO_ID.get(mat_name, MATERIAL_NAME_TO_ID['grass'])
            world._mat_map[x, y] = mat_id


    # ---- 4. Place the player ----
    player = None
    for y in range(H):
        for x in range(W):
            if char_grid[y, x] == 'A':
                player = objects.Player(world, (x, y))
                world.add(player)
                break
        if player:
            break
    if player is None:
        raise ValueError("No player 'A' found in the layout.")

    # ---- 5. Place other entities (cow, zombie, etc.) ----
    for y in range(H):
        for x in range(W):
            ch = char_grid[y, x]
            if ch in CHAR_TO_ENTITY and ch != 'A':
                cls = CHAR_TO_ENTITY[ch]
                # Zombies and Skeletons need a player reference
                if cls.__name__ in ["Zombie", "Skeleton"]:
                    obj = cls(world, (x, y), player)
                else:
                    obj = cls(world, (x, y))
                world.add(obj)

    # ---- 6. Debug output (recommended to verify material correctness) ----
    # print(">>> Unique mat IDs:", np.unique(world._mat_map))
    # print(">>> Material table:", world._mat_ids)

    return world, player


# ------------------------------------------------------------
# 3. Custom Crafter Environment (string map + native rendering)
# ------------------------------------------------------------
class CustomCrafterEnv(gym.Env):
    """Crafter environment with string-defined maps and native renderer."""

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, layout_str, seed=0):
        super().__init__()
        import crafter.worldgen
        # Disable default random world generation
        crafter.worldgen.generate_world = lambda world, player: None

        self.seed = seed
        self.char_grid = np.array(
            [list(line.strip()) for line in layout_str.strip().split("\n") if line.strip()]
        )

        # Initialize native Crafter environment
        self.env = crafter.Env(reward=False, seed=seed)

        # Inject custom world
        world, player = make_world_from_chars(self.char_grid, seed)
        self.env._world = world
        self.env._player = player

        # Define action/observation space
        self.action_space = self.env.action_space
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        self.ai_enabled = False

    # --------------------------------------------------------
    # Reset / Step / Render
    # --------------------------------------------------------
    def reset(self, **kwargs):
        self.env.reset()

        # Build custom map
        world, player = make_world_from_chars(self.char_grid, seed=self.seed)
        self.env._world = world
        self.env._player = player

        # Critical fix: reload textures and rebuild view pipeline
        from crafter import engine, constants
        self.env._textures = engine.Textures(constants.root / "assets")
        view_h, view_w = self.env._view
        item_rows = int(np.ceil(len(constants.items) / view_h))
        self.env._local_view = engine.LocalView(
            self.env._world, self.env._textures, [view_h, view_w - item_rows]
        )
        self.env._item_view = engine.ItemView(
            self.env._textures, [view_h, item_rows]
        )

        obs = self.env._obs()
        return obs, {}

    def step(self, action):
        """
        Deterministic or semi-deterministic step function for Crafter.
        - Only updates the player by default (for World Model training).
        - Optionally updates other entities (cow, zombie, etc.) if AI is enabled.
        """

        from crafter import constants

        # --- 1. Apply player action ---
        # Map the discrete action ID to the Crafter action constant
        # and update the player’s state accordingly.
        self.env._player.action = constants.actions[action]
        self.env._player.update()

        # --- 2. Optionally update other entities (if AI is enabled) ---
        # This allows switching between deterministic and full simulation modes.
        if self.ai_enabled:
            for obj in self.env._world.objects:
                if obj is not self.env._player:
                    obj.update()

        # --- 3. Update environment time/daylight cycle ---
        # Keep the world visually consistent even if other entities are frozen.
        if hasattr(self.env, "_update_time"):
            self.env._update_time()

        # --- 4. Get the new observation ---
        obs = self.env._obs()

        # --- 5. Return standard Gym-style outputs ---
        # Reward and done are dummy values since Crafter’s reward system
        # is not used for World Model training.
        reward = 0.0
        done = False
        info = {}

        return obs, reward, done, info


    def render(self, mode="rgb_array"):
        return self.env.render(size=(128, 128))

    def close(self):
        self.env.close()


# ------------------------------------------------------------
# 4. Example test run
# ------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from crafter import constants

    # actually the string shape need to be square

    layout_str = """
    GGIIGGG
    GGAGGGW
    GKGGGTG
    RGGPGGI
    GWGGGGG
    GKGGGTG
    GWGGGGG
    """

    # --- 1. Create a custom environment from the layout string ---
    base_env = CustomCrafterEnv(layout_str, seed=0)
    base_env.ai_enabled = False  # keep deterministic (disable Cow/Zombie auto-movement)

    # --- 2. Reset the environment ---
    obs, _ = base_env.reset()

    # --- Handle action mapping (compatible with both list and dict) ---
    # Available actions:
    # 0: noop
    # 1: move_left
    # 2: move_right
    # 3: move_up
    # 4: move_down
    # 5: do
    # 6: sleep
    # 7: place_stone
    # 8: place_table
    # 9: place_furnace
    # 10: place_plant
    # 11: make_wood_pickaxe
    # 12: make_stone_pickaxe
    # 13: make_iron_pickaxe
    # 14: make_wood_sword
    # 15: make_stone_sword
    # 16: make_iron_sword

    if isinstance(constants.actions, dict):
        action_names = list(constants.actions.keys())
    else:
        action_names = list(constants.actions)

    plt.ion()
    for i in range(5):
        # --- Execute a random action ---
        action_id = base_env.action_space.sample()
        action_name = action_names[action_id] if action_id < len(action_names) else str(action_id)

        obs, reward, done, info = base_env.step(action_id)

        # --- Extract symbolic grid representation ---
        symbolic_obs = extract_tensor_grid(base_env.env)
        rgb = base_env.render(mode="rgb_array")

        # --- Print debug info ---
        print(f"\n=== Step {i} ===")
        print(f"Action ID: {action_id}  →  {action_name}")
        print("Object layer:")
        print(symbolic_obs[..., 0])
        print("Direction layer:")
        print(symbolic_obs[..., 1])
        print("Symbolic obs shape:", symbolic_obs.shape)

        # --- Visualize RGB frame ---
        plt.imshow(rgb)
        plt.title(f"Step {i} - Action: {action_name}")
        plt.axis("off")
        plt.pause(0.5)

    plt.ioff()
    plt.show()
