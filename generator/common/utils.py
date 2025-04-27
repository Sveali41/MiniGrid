import torch
import sys  
sys.path.append('/home/siyao/project/rlPractice/MiniGrid')
from generator.gen import GAN
import random
import re

def load_gen(cfg):
    hparams = cfg
    if hparams.training_generator.generator == "deconv":
        import sys
        sys.path.append('/home/siyao/project/rlPractice/MiniGrid/generator')
        from generator.deconv_gen import Generator, Discriminator
        model = GAN(generator=Generator(hparams.training_generator.z_shape, len(hparams.training_generator.map_element)), 
            discriminator=Discriminator(input_channels = len(hparams.training_generator.map_element)), 
            z_size=hparams.training_generator.z_shape, lr=hparams.training_generator.lr, wd=hparams.training_generator.wd)
        
    elif cfg.training_generator.model == "basic":
        from generator.basic_gen import Generator, Discriminator
        model = GAN(generator=Generator(hparams.basic.z_shape, hparams.basic.dropout), discriminator=Discriminator(hparams.basic.input_channels, hparams.basic.dropout), z_size=hparams.basic.z_shape, lr=0.0002, wd=0.0)

    checkpoint = torch.load(hparams.training_generator.validation_path)
    
    import io
    state_dict = checkpoint['state_dict']
    # make the state_dict a buffer
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)

    model.load_state_dict(torch.load(buffer))
    model.eval()
    return model

def generate_obj_map(layout, map_dict):
    reverse_map_dict = {v: k for k, v in map_dict.items()}
    layout_strings = []
    for i in range(layout.shape[1]): 
        row = layout[0][i]  
        layout_line = ''.join([reverse_map_dict.get(num.item()) for num in row])
        layout_strings.append(layout_line)
    return '\n'.join(layout_strings)

def generate_color_map(layout_strings):
    '''
    generate color map from layout strings
    '''
    # define the mapping from object to color
    object_to_color_map = {
        'W': 'W',  # Wall → W
        'E': 'E',  # Floor → E
        'G': 'G',  # Goal → G
        'S': 'E',  # Start → E
        'K': 'Y',  # Key → Y
        'D': 'Y'   # Door → Y
    }

    table = str.maketrans(object_to_color_map)
    return layout_strings.translate(table)

def layout_to_string(layout):
    """
    Convert a 2D list of characters into a single string with newline separators.
    
    Example:
    [['W', 'W', 'E'],
    ['S', 'E', 'G']]
    → 'WWE\nSEG'
    """
    return '\n'.join(''.join(row) for row in layout)


def clean_and_place_goal(layout_string):
    # Step 0: Replace 'K', 'D', 'S' with 'E'
    layout_string = re.sub(r'[KDS]', 'E', layout_string)

    # Step 1: Convert to list of rows
    layout_rows = layout_string.strip().split('\n')

    # Step 2: Check if G exists
    if any('G' in row for row in layout_rows):
        return '\n'.join(layout_rows)  # Already has a goal

    # Step 3: Find all E positions
    e_positions = []
    for row_idx, row in enumerate(layout_rows):
        for col_idx, char in enumerate(row):
            if char == 'E':
                e_positions.append((row_idx, col_idx))

    if not e_positions:
        raise ValueError("No empty spaces 'E' available to place a goal.")

    # Step 4: Pick one 'E' and turn it into 'G'
    y, x = random.choice(e_positions)
    row_chars = list(layout_rows[y])
    row_chars[x] = 'G'
    layout_rows[y] = ''.join(row_chars)

    # Step 5: Join back to string
    return '\n'.join(layout_rows)

    
def combine_maps(layout: str, color: str, file_path) -> str:
    combine_maps = layout.strip() + "\n\n" + color.strip()
    with open(file_path, 'w') as f:
        f.write(combine_maps)
    print(f"Layouts saved to {file_path}")

# for VAE generator based on classification
def map_value_to_index(batch, class_values):
    """
    Map values in the batch to the class of the model.
    Args:
        batch (torch.Tensor): The input tensor.
        value_to_idx (dict): A dictionary mapping values to indices."""
    value_to_idx = {v: i for i, v in enumerate(class_values)}
    batch_mapped = batch.clone()
    for val, idx in value_to_idx.items():
        batch_mapped[batch == val] = idx
    return batch_mapped.float()

def map_index_to_value(pred_classes, class_values):
    """
    Map indices in the predicted classes to their corresponding values.
    Args:
        pred_classes (torch.Tensor): The predicted classes.
        idx_to_value (dict): A dictionary mapping indices to values.
    """
    idx_to_value = {i: v for i, v in enumerate(class_values)}  # 类别索引到数值的反向映射
    pred_map = pred_classes.clone()
    for idx, val in idx_to_value.items():
        pred_map[pred_classes == idx] = val
    return pred_map


