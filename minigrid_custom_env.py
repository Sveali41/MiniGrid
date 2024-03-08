from typing import Optional
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import *
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from path import *
from PIL import Image, ImageDraw

ACTION_NAMES = {
    0: 'Turn Left',
    1: 'Turn Right',
    2: 'Move Forward',
    3: 'Pick Up',
    4: 'Drop',
    5: 'Toggle',
    # 6: 'Done'
}


class CustomEnvFromFile(MiniGridEnv):
    """
    A custom MiniGrid environment that loads its layout and object properties from a text file.

    Attributes:
        txt_file_path (str): Path to the text file containing the environment layout.
        layout_size (int): The size of the environment, either specified or determined from the file.
        agent_start_pos (tuple[int, int]): Starting position of the agent.
        agent_start_dir (int): Initial direction the agent is facing. None for random direction
        mission (str): Custom mission description.
    """

    def __init__(
            self,
            txt_file_path: str,
            size: Optional[int] = None,
            agent_start_pos: Optional[tuple[int, int]] = None,  # Allow None for random initialization
            agent_start_dir: Optional[int] = None,  # Allow None for random initialization
            custom_mission: str = "Explore and interact with objects.",
            max_steps: Optional[int] = None,
            **kwargs,
    ) -> None:
        """
        Initializes the custom environment.

        If 'size' is not specified, it determines the size based on the content of the given text file.
        """
        self.txt_file_path = txt_file_path
        # Determine the size of the environment if not provided
        self.layout_size = self.determine_layout_size() if size is None else size
        # Initialize the MiniGrid environment with the determined size
        super().__init__(
            mission_space=MissionSpace(mission_func=lambda: custom_mission),
            grid_size=self.layout_size,
            see_through_walls=False,
            max_steps=max_steps or 4 * self.layout_size ** 2,
            **kwargs,
        )
        self.rand_agent_start_pos = agent_start_pos is None
        self.agent_start_pos = agent_start_pos
        self.rand_agent_start_dir = agent_start_dir is None
        self.agent_start_dir = agent_start_dir
        self.mission = custom_mission

    # def reset(self, **kwargs):
    #     """
    #     Resets the environment for a new episode. If agent_start_dir was initially None,
    #     it randomizes the agent's starting direction again.
    #     """
    #     # Proceed with the standard reset process
    #     return super().reset(**kwargs)

    def determine_layout_size(self) -> int:
        """
        Reads the layout from the file to determine the environment's size based on its width and height.

        Returns:
            int: The larger value between the height and width of the layout to ensure a square grid.
        """
        with open(self.txt_file_path, 'r') as file:
            sections = file.read().split('\n\n')
            layout_lines = sections[0].strip().split('\n')
            # Set the environment's width and height based on the layout
            self.height = len(layout_lines)
            self.width = max(len(line) for line in layout_lines)
            return max(self.height, self.width)

    def _gen_grid(self, width: int, height: int) -> None:
        """
        Generates the grid for the environment based on the layout specified in the text file.
        """
        self.grid = Grid(width, height)
        self.read_layout_from_file()
        # Randomize the starting direction again if it was initialized as random
        if self.rand_agent_start_dir:
            self.agent_start_dir = np.random.randint(0, 4)
        if self.rand_agent_start_pos:
            self.agent_start_pos = self.find_empty_position()
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

    def read_layout_from_file(self) -> None:
        """
        Parses the text file specified by 'txt_file_path' to set the objects in the environment's grid.
        """
        with open(self.txt_file_path, 'r') as file:
            sections = file.read().split('\n\n')
            if len(sections) != 2:
                raise ValueError("File must contain exactly two sections separated by one empty line.")

            layout_lines, color_lines = sections[0].strip().split('\n'), sections[1].strip().split('\n')

            if len(layout_lines) != len(color_lines) or any(
                    len(layout) != len(color) for layout, color in zip(layout_lines, color_lines)):
                raise ValueError("Object and color matrices must have the same size.")

            for y, (layout_line, color_line) in enumerate(zip(layout_lines, color_lines)):
                for x, (char, color_char) in enumerate(zip(layout_line, color_line)):
                    color = self.char_to_color(color_char)
                    obj = self.char_to_object(char, color)
                    if obj:
                        self.grid.set(x, y, obj)  # Place the object on the grid

    def find_empty_position(self) -> tuple[int, int]:
        """
        Finds an empty position on the grid where there is no object.

        Returns:
            tuple[int, int]: The coordinates of an empty position.
        """
        empty_positions = [(x, y) for x in range(self.width) for y in range(self.height)
                           if self.grid.get(x, y) is None]
        if not empty_positions:
            raise ValueError("No empty position available on the grid.")
        # Use random.randint to select an index and then retrieve the position
        index = np.random.randint(0, len(empty_positions))
        return empty_positions[index]

    def char_to_color(self, char: str) -> Optional[str]:
        """
        Maps a single character to a color name supported by MiniGrid objects.

        Args:
            char (str): A character representing a color.

        Returns:
            Optional[str]: The name of the color, or None if the character is not recognized.
        """
        color_map = {'R': 'red', 'G': 'green', 'B': 'blue', 'Y': 'yellow', 'M': 'magenta', 'C': 'cyan'}
        return color_map.get(char.upper(), None)

    def char_to_object(self, char: str, color: str) -> Optional[WorldObj]:
        """
        Maps a character (and its associated color) to a MiniGrid object.

        Args:
            char (str): A character representing an object type.
            color (str): The color of the object.

        Returns:
            Optional[WorldObj]: The MiniGrid object corresponding to the character and color, or None if unrecognized.
        """
        obj_map = {
            'W': lambda: Wall(), 'F': lambda: Floor(), 'B': lambda: Ball(color),
            'K': lambda: Key(color), 'X': lambda: Box(color), 'D': lambda: Door(color, is_locked=True),
            'G': lambda: Goal(), 'L': lambda: Lava(),
        }
        constructor = obj_map.get(char, None)
        return constructor() if constructor else None

    def create_minigrid_image(self, cell_size=100):
        # Define the mapping from characters to colors
        color_map = {
            'W': 'gray',  # Wall
            'K': 'yellow',  # Key
            'G': 'green'  # Goal
        }
        with open(self.txt_file_path, 'r') as file:
            sections = file.read().split('\n\n')
            env_string = sections[0].split('\n')
        # Calculate the width of the image
        height = cell_size * len(env_string)
        width = cell_size * max(len(line) for line in env_string)

        # Create a new image with a white background
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)

        # Draw each cell
        for i, char in enumerate(env_string):
            color = color_map.get(char, 'white')  # Default to white if the character is not found
            # Determine the boundaries of the cell
            left = i * cell_size
            top = 0
            right = left + cell_size
            bottom = top + cell_size
            # Draw the cell
            draw.rectangle((left, top, right, bottom), fill=color)
            img.show()

if __name__ == "__main__":
    # Example usage of the CustomEnvFromFile class
    path = Paths()
    env = CustomEnvFromFile(txt_file_path=path.LEVEL_FILE, custom_mission="Find the key and open the door.",
                            render_mode="human")
    CustomEnvFromFile(txt_file_path=path.LEVEL_FILE, custom_mission="Find the key and open the door.",
                      render_mode="human").create_minigrid_image()
    manual_control = ManualControl(env)  # Allows manual control for testing and visualization
    manual_control.start()  # Start the manual control interface
