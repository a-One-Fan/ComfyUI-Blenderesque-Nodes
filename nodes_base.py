import torch
import torch.nn.functional as functional

DEFAULT_CANVAS = (64, 64)

class BlenderData:
    image: torch.Tensor | None
    canvas: tuple[int, int] | None
    value: tuple | float | int | None
    def __init__(self, any, paramname: str | torch.Tensor | None = None, no_colortransform = False):
        """Create from sRGB Image tensor (1, 3 or 4 channels), 2 tensors (3 channels + 1 channel), int, float or 3/4-wide tuple of int/float"""

        def color_transform(te):
            if no_colortransform:
                return te
            return torch.pow(te, 2.2)

        self.image = None
        self.canvas = None
        self.value = None

        if type(any) is BlenderData:
            self.image = any.image
            self.canvas = any.canvas
            self.value = any.value

        elif type(any) is torch.Tensor:
            if type(paramname) is torch.Tensor:
                self.image = join_rgba(any, paramname)
            else:
                self.image = any
                if self.image.size()[3] in [3, 4]:
                    if self.image.size()[3] == 3:
                        self.image = color_transform(self.image)
                    else:
                        size_cut = list(self.image.size())[:-1]
                        mask = torch.cat(torch.full(size_cut + [3], 1.0), 
                                         torch.full(size_cut + [1], 0.0),
                                         dim = -1)
                        self.image = color_transform(any) * mask + any * (1.0 - mask)
            self.canvas = (self.image.size()[1], self.image.size()[2])

        elif type(any) == dict:
            if type(paramname) != str:
                raise Exception(f"Paramname of non-string type: {type(paramname)} {paramname}")
            if type(any.get(paramname)) is torch.Tensor:
                self.image = color_transform(any[paramname])
            if type(any.get(paramname)) is BlenderData:
                other = any[paramname]
                self.image = other.image
                self.canvas = other.canvas
                self.value = other.value
            elif any.get(paramname + "A", None) != None:
                self.value = (any[paramname + "R"], any[paramname + "G"], any[paramname + "B"], any[paramname + "A"])
            elif any.get(paramname + "R", None) != None:
                self.value = (any[paramname + "R"], any[paramname + "G"], any[paramname + "B"])
            elif any.get(paramname + "F", None) != None:
                self.value = any[paramname + "F"]
            elif any.get(paramname + "X", None) != None:
                self.value = (any[paramname + "X"], any[paramname + "Y"], any[paramname + "Z"])
            else:
                raise KeyError(f"No Blender-like parameters to initialize by {paramname} in {any}")
        
        elif paramname != None:
            raise Exception(f"Paramname of incorrect type: {type(paramname)} {paramname}")
            
        elif type(any) in [tuple, list, float, int]:
            if type(any) in [tuple, list]:
                if not len(any) in [3, 4]:
                    raise Exception(F"Can not convert {len(any)}-elemnt tuple/list to Blender data!")
                
            self.value = any
        
        else:
            raise Exception(f"Can not convert {type(any)} to Blender data: {any}")

    def set_canvas(self, canvas: tuple[int, int]):
        self.canvas = canvas
        if self.image != None and self.image.size()[1:-1] != canvas:
            mode = 'bilinear'
            if self.image.size()[1] < canvas[0] and self.image.size()[2] < canvas[1]:
                mode = 'bicubic'
            print(f"Interp: {mode}, {self.image.size()}, {canvas}")
            self.image = functional.interpolate(self.image, [self.image.size()[0], canvas[0], canvas[1], self.image.size()[3]], mode=mode)

    def as_rgba(self, batch=1) -> torch.Tensor:
        """Interpret as a [batch, canvas x, canvas y, 4] tensor"""
        size = (batch, self.canvas[0], self.canvas[1])

        if self.image == None:
            if type(self.value) == tuple: 
                padded = list(self.value)
                if len(padded) < 4:
                    padded += [1.0]
                return torch.stack([torch.full(size, val) for val in padded], dim=-1)
            else:
                return torch.cat(torch.full((*size, 3), self.value), torch.ones((*size, 1), dim=-1))
        
        if self.image.size()[3] == 1:
            return torch.cat((self.image, self.image, self.image, torch.ones(*size, 1)), dim=-1)
        if self.image.size()[3] == 3:
            return torch.cat((self.image, torch.ones((*size, 1))), dim=-1)
        if self.image.size()[3] == 4:
            return self.image
        
        raise Exception(f"Failed to interpret tensor of size {self.image.size()} as rgba!")
    
    def as_out(self, batch=1) -> torch.Tensor:
        """Interpret as standard ComfyUI image tensor (RGBA)"""

        if self.canvas == None:
            self.canvas = DEFAULT_CANVAS
            res = self.as_rgba(batch)
            self.canvas = None
        else:
            res = self.as_rgba(batch)
        
        return torch.pow(res, 1/2.2) # sRGB

    def as_rgb_a(self, batch=1) -> tuple[torch.Tensor, torch.Tensor]:
        """Interpret as ([batch, canvas x, canvas y, 3], [batch, canvas x, canvas y, 1]) tensors"""
        rgba = self.as_rgba(batch)
        return rgba.split([3, 1], dim=-1)
    
    def as_rgb(self, batch=1) -> torch.Tensor:
        """Interpret as a [batch, canvas x, canvas y, 3] tensor"""
        rgb, a = self.as_rgb_a(batch)
        return rgb
    
    def as_float(self, batch=1, use_alpha=False) -> torch.Tensor:
        """Interpret as a [batch, canvas x, canvas y, 1] tensor"""
        if self.image == None:
            if type(self.value) == tuple: 
                avg = sum(self.value) / len(self.value)
                return torch.full((batch, self.canvas[0], self.canvas[1], 1), avg)
            else:
                return torch.full((batch, self.canvas[0], self.canvas[1], 1), self.value)
        
        if self.image.size()[3] == 1:
            return self.image
        if self.image.size()[3] in [3, 4]:
            im_alt = self.image
            if not use_alpha and self.image.size()[3] == 4:
                im_alt = torch.cat(im_alt.split(1, dim=-1)[:-1], dim=-1)
            return torch.unsqueeze(torch.mean(im_alt, dim=-1), dim=-1)
        
        raise Exception(f"Failed to interpret tensor of size {self.image.size()} as float!")
        return None

    def is_value(self):
        return self.value != None

def split_rgba(tensor, rgb_chunk=True):
    """rgb_chunk: return rgb, a instead of r, g, b, a"""
    if rgb_chunk:
        if tensor.size()[3] == 3:
            return tensor, torch.full(list(tensor.size())[:-1] + [1], 1.0)
        return tensor.split([3, 1], dim=-1)
    
    if tensor.size()[3] == 3:
        r, g, b = tensor.split(1, dim=-1)
        a = torch.ones_like(r)
    else:
        r, g, b, a = tensor.split(1, dim=-1)
    return r, g, b, a

def join_rgba(rgb, a):
    return torch.cat((rgb, a), dim=-1)

def col_to_float(tensor):
    channels_first = tensor.permute(3, 1, 2, 0)
    avg = (channels_first[0] + channels_first[1] + channels_first[2]) / 3.0
    stacked = torch.stack((avg, avg, avg))
    normal_order = stacked.permute(3, 1, 2, 0)
    return normal_order

def guess_canvas(*blens: BlenderData, default=DEFAULT_CANVAS):
    res = None
    for b in blens:
        if type(b.image) is torch.Tensor:
            res = b.image.size()[1:-1]

    if not res:
        res = default
    
    for b in blens:
        b.set_canvas(res)

    return res

COLMAX = 1.0 
# Blender has soft min and max. 
# All colors have a soft max of 1.0 can't be dragged further, but can be typed in higher. 
# ComfyUI does not seem to have soft min/max?
COLSTEP = 0.005

def COLOR_INPUT(name, default=1.0, alpha=False, step=COLSTEP, max=COLMAX, astep=0.005, hidden_default=False):
    if hidden_default:
        return {name: ("*", {})}
    base = {name: ("*", {}),
            name+"R": ("FLOAT", {"default": default, "min": 0.0, "max": max, "step": step}),
            name+"G": ("FLOAT", {"default": default, "min": 0.0, "max": max, "step": step}),
            name+"B": ("FLOAT", {"default": default, "min": 0.0, "max": max, "step": step}),}
    if alpha:
        base[name+"A"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": astep})
    
    return base

def FLOAT_INPUT(name, default=0.0, min=0.0, max=1.0, step=COLSTEP, hidden_default=False):
    if hidden_default:
        return {name: ("*", {})}
    return {name: ("*", {}),
            name+"F": ("FLOAT", {"default": default, "min": min, "max": max, "step": step})}

def VECTOR_INPUT(name, default=0.0, min=-100000.0, max=100000.0, step=0.01, hidden_default=False):
    if hidden_default:
        return {name: ("*", {})}
    return {name: ("*", {}),
            name+"X": ("FLOAT", {"default": default, "min": min, "max": max, "step": step}),
            name+"Y": ("FLOAT", {"default": default, "min": min, "max": max, "step": step}),
            name+"Z": ("FLOAT", {"default": default, "min": min, "max": max, "step": step}),}

def BLENDER_OUTPUT(single=False):
    if single:
        return (("BLENDER", ))
    return ("BLENDER", "IMAGE", )

def BLEND_VALID_INPUTS(input_types, ref):
    #input_types is a dict of {"Name": "TYPE_NAME"}
    return True