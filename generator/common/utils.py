import torch
import sys  
sys.path.append('/home/siyao/project/rlPractice/MiniGrid')
from generator.gen import GAN

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


    # color_strings = []
    # for line in layout_strings:
    #     color_line = ''
    #     for char in line:
    #         color_char = object_to_color_map.get(char.upper(), 'E')
    #         color_line += color_char
    #     color_strings.append(color_line)

    # return color_strings


    
    s