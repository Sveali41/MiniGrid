import torch
import sys
sys.path.append('/home/siyao/project/rlPractice/MiniGrid')
sys.path.append('/home/siyao/project/rlPractice/MiniGrid/generator')
from modelBased.common.utils import PROJECT_ROOT, get_env
from omegaconf import DictConfig
from generator.gen import GAN
from modelBased.common.utils import GENERATOR_PATH
import hydra
import json

# 1. load the generator 
@hydra.main(version_base=None, config_path=str(GENERATOR_PATH / "conf"), config_name="config")
def load_generator(cfg: DictConfig):
    hparams = cfg
    if hparams.training_generator.generator == "deconv":
        from generator.deconv_gen import Generator, Discriminator
        model = GAN(generator=Generator(hparams.deconv.z_shape, len(hparams.training_generator.map_element)), 
                    discriminator=Discriminator(dropout = hparams.deconv.dropout), 
                    z_size=hparams.deconv.z_shape, lr=hparams.training_generator.lr, wd=hparams.training_generator.wd)
        
    elif cfg.training_generator.model == "basic":
        from generator.basic_gen import Generator, Discriminator
        model = GAN(generator=Generator(hparams.basic.z_shape, hparams.basic.dropout), discriminator=Discriminator(hparams.basic.input_channels, hparams.basic.dropout), z_size=hparams.basic.z_shape, lr=0.0002, wd=0.0)

    # Load state_dict into the model
    ## **************test 2024 09 07 *******************
    # moel.load can't directly read checkpoint['state_dict'],do as follows:
    # model.load_from_checkpoint(checkpoint['state_dict'])
    import io
    checkpoint = torch.load(hparams.training_generator.validation_path)
    state_dict = checkpoint['state_dict']
    # make the state_dict a buffer
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)

    model.load_state_dict(torch.load(buffer))
    ## **************************************************
    # Set the model to evaluation mode (optional, depends on use case)
    model.eval()
    batch_size = 1
    z = torch.randn(batch_size,hparams.deconv.z_shape)
    with torch.no_grad():  
        generated_map = model(z)
        generated_map = torch.argmax(generated_map, dim=1)
        print(generated_map)
        # Assuming the rest of your code is already set up as provided
    # 2. add the color map
    char_to_int = hparams.training_generator.map_element
    map = env_map(char_to_int, generated_map)
    output_file = hparams.training_generator.env_path
    with open(output_file, 'w') as file:
        file.write(map)
    print(f"Level map saved to {output_file}")


def env_map(char_to_int, generated_map):
    int_to_char = {v: k for k, v in char_to_int.items()}
    mapped_obj = map_numbers_to_chars(generated_map, int_to_char)
    # add color map
    color_map = []
    for row in mapped_obj:
        color_row = []
        for char in row:
            if char in ['K', 'D', 'B']:
                color_row.append('Y')  
            elif char == 'O':
                color_row.append('B')  
            else:
                color_row.append(char)  
        color_map.append(color_row)
    map = format_maps(mapped_obj, color_map)
    return map


def map_numbers_to_chars(tensor, int_to_char):
    tensor = tensor.squeeze(0)
    tensor_np = tensor.numpy()
    mapped_chars = [''.join([int_to_char[num] for num in row]) for row in tensor_np]
    return mapped_chars

def format_maps(object_map, color_map):
    # Convert object_map and color_map into string format
    object_map_str = '\n'.join([''.join(row) for row in object_map])
    color_map_str = '\n'.join([''.join(row) for row in color_map])
    
    # Combine the two maps with a double newline separating them
    return f"{object_map_str}\n\n{color_map_str}"

if __name__ == "__main__":
    map = load_generator()
    pass
