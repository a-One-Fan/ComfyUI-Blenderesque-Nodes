import torch
import torch.nn.functional as functional
from .util import *
from .cl_wrapper import transform
from math import floor, ceil

DEFAULT_CANVAS = (64, 64)
DEBUG = False

def maybe_convert_to_list(thing):
    if type(thing) in [int, float]:
        return [thing, ]
    if type(thing) == tuple:
        return list(thing)
    return thing

class BlenderData:
    image: torch.Tensor | None
    canvas: tuple[int, int] | None # Canvas is Height x Width
    value: list | None
    is_color: bool
    def __init__(self, any, paramname: str | torch.Tensor | None = None, 
                 colortransform_if_converting: bool = True, colortransform_force: bool | None = None,
                 widget_override=None, default_notfound=None):
        """
        - Tensor (gets colortransformed if 3 or 4 channels)
        - Tensor (RGB), Tensor (A) -> RGBA
        - kwargs, parameter name -> Guess automatically
        - BlenderData (copies by reference)
        - number
        - 2/3/4-dim tuple of numbers e.g. (1.0, 1.0, 0.0)\n
        widget_override: Instead of looking for a widget value, use this value directly\n
        default_notfound: Value to use when no widgets or inputs are found
        """

        default_notfound = maybe_convert_to_list(default_notfound)
        widget_override = maybe_convert_to_list(widget_override)
        any = maybe_convert_to_list(any)

        if colortransform_force != None:
            colortransform = colortransform_force
        else:
            colortransform = False
            if type(paramname) == str:
                if any.get(paramname + "R", None) != None:
                    colortransform = colortransform_if_converting # No syntactic diabetes!
                if any.get(paramname, None) != None:
                    t = any[paramname]
                    if type(t) is torch.Tensor and t.size()[-1] in [3, 4]: # TODO: pad masks?
                        colortransform = colortransform_if_converting

        self.is_color = colortransform

        def color_transform(te):
            if not colortransform:
                return te
            return srgb_to_rgb(te)

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
                if self.image.size()[3] in [2, 3, 4]:
                    if self.image.size()[3] == 3:
                        self.image = color_transform(self.image)
                    elif self.image.size()[3] == 4:
                        rgb, a = self.image.split([3, 1], dim=-1)
                        rgb = color_transform(rgb)
                        self.image = torch.cat((rgb, a), dim=-1)
            self.canvas = (self.image.size()[1], self.image.size()[2])

        elif type(any) == dict:
            if type(paramname) != str:
                raise Exception(f"Paramname of non-string type: {type(paramname)} {paramname}")
            param = any.get(paramname)
            if type(param) is torch.Tensor:
                if len(param.size()) == 4:
                    self.image = color_transform(param)
                else:
                    self.image = param.unsqueeze(-1)
                self.canvas = (self.image.size()[1], self.image.size()[2])
            elif type(param) is BlenderData:
                other = any[paramname]
                self.image = other.image
                self.canvas = other.canvas
                self.value = other.value
                self.is_color = other.is_color
            elif type(param) is dict:
                if param.get("samples", None) != None:
                    self.image = param["samples"].permute((0, 2, 3, 1))
                    self.canvas = (self.image.size()[1], self.image.size()[2])
                    self.is_color = False
                else:
                    print(param)
                    raise KeyError("Can't convert unknown data (see above)")
            elif not widget_override is None:
                self.value = widget_override
            elif any.get(paramname + "A", None) != None:
                self.value = [any[paramname + "R"], any[paramname + "G"], any[paramname + "B"], any[paramname + "A"]]
            elif any.get(paramname + "R", None) != None:
                self.value = [any[paramname + "R"], any[paramname + "G"], any[paramname + "B"]]
            elif any.get(paramname + "F", None) != None:
                self.value = [any[paramname + "F"], ]
            elif any.get(paramname + "X", None) != None:
                self.value = [any[paramname + "X"], any[paramname + "Y"], any[paramname + "Z"]]
            else:
                if default_notfound != None:
                    self.value = default_notfound
                else:
                    raise KeyError(f"No Blender-like parameters to initialize by {paramname} in {any}")
        
        elif paramname != None:
            raise Exception(f"Paramname of incorrect type: {type(paramname)} {paramname}")
            
        elif type(any) == list:
            self.value = any
        
        else:
            raise Exception(f"Can not convert {type(any)} to Blender data: {any}")
        
        if DEBUG:
            print(f"blender data init, paramname ", end="")
            if type(paramname) is torch.Tensor:
                print(f"tensor {paramname.size()}", end="")
            else:
                print(paramname, end="")
            print(" with:")

            if type(any) == dict:
                s = ""
                for k in any.keys():
                    v = any[k]
                    if type(v) is torch.Tensor:
                        s += f"\"{k}\": Tensor {v.size()}, "
                    else:
                        s += f"\"{k}\": {v}, "
                print(s)
            else:
                if type(any) is torch.Tensor:
                    print(f"tensor {any.size()}")
                else:
                    print(any)
            print(colortransform_if_converting, colortransform_force, widget_override, default_notfound)
            print(self)

    def __str__(self):
        if self.image is None:
            return f"Primitive {self.value}, canvas {self.canvas}"
        return f"Tensor {self.image.size()}, canvas {self.canvas}"

    def set_canvas(self, canvas: tuple[int, int], scale=True, off=(0, 0)):
        if self.image != None and self.image.size()[1:-1] != canvas:
            interp = 1 #'bilinear'
            oldcanvas = (self.image.size()[1], self.image.size()[2])
            if oldcanvas[0] < canvas[0] and oldcanvas[1] < canvas[1]:
                interp = 2 #'bicubic'

            if scale:
                sx = canvas[1] / oldcanvas[1] # Tensors are Height x Width
                sy = canvas[0] / oldcanvas[0] # Width is 1, Height is 0 
            else:
                sx = 1
                sy = 1

            size_123 = (self.image.size()[0], canvas[0], canvas[1])
            locx = torch.full((*size_123, 1), off[0])
            locy = torch.full((*size_123, 1), off[1])
            rot = torch.zeros((*size_123, 1))
            scale_x = torch.full((*size_123, 1), sx)
            scale_y = torch.full((*size_123, 1), sy)
            locrotscale = torch.cat((locx, locy, rot, scale_x, scale_y, ), dim=-1)

            self.image = transform(self.image, canvas, locrotscale, interp, 0)
                    
        self.canvas = canvas

    def as_2wide(self, batch=1) -> torch.Tensor:
        """
        Interpret as a [batch, canvas x, canvas y, 2] tensor
        """

        size = (batch, self.canvas[0], self.canvas[1])

        if not self.image:
            if len(self.value) > 1:
                return torch.stack((torch.full(size, self.value[0]), torch.full(size, self.value[1])), dim=-1)
            return torch.full((*size, 2), self.value[0])
        
        if self.image.size()[3] == 2:
            return self.image
        
        if self.image.size()[3] == 1:
            return torch.cat((self.image, self.image), dim=-1)
        
        return self.image.split([2, self.image.size()[3]-2], dim=-1)[0]


    def as_rgba(self, batch=1) -> torch.Tensor:
        """Interpret as a [batch, canvas x, canvas y, 4] tensor"""
        size = (batch, self.canvas[0], self.canvas[1])

        if self.image == None:
            if self.is_color:
                padded = self.value
                if len(padded) < 4:
                    padded = padded + [1.0]
            else:
                if DEBUG:
                    print(f"Self value: {self.value}")
                padded = resize_channels(self.value, 4)
            
            if DEBUG:
                print(f"Padded: {padded}")
            return torch.stack([torch.full(size, val) for val in padded], dim=-1)
        
        if self.image.size()[3] == 1:
            return torch.cat((self.image, self.image, self.image, torch.ones(*size, 1)), dim=-1)
        if self.image.size()[3] == 2:
            return torch.cat((self.image, torch.zeros((*size, 1)), torch.ones((*size, 1))), dim=-1)
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
        
        return rgb_to_srgb(res)

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
            avg = sum(self.value) / len(self.value)
            return torch.full((batch, self.canvas[0], self.canvas[1], 1), avg)
        
        if self.image.size()[3] == 1:
            return self.image
        if self.image.size()[3] in [3, 4]:
            im_alt = self.image
            if not use_alpha and self.image.size()[3] == 4:
                im_alt = torch.cat(im_alt.split(1, dim=-1)[:-1], dim=-1)
            return torch.unsqueeze(torch.mean(im_alt, dim=-1), dim=-1)
        
        raise Exception(f"Failed to interpret tensor of size {self.image.size()} as float!")
        return None

    def as_vector(self, batch=1, channels=None, mode_shrink = 0, mode_extend = 0) -> torch.Tensor:
        """Interpret as a [batch, canvas x, canvas y, 3?] tensor"""
        if self.image is None:
            size = (batch, self.canvas[0], self.canvas[1], 1)
            if channels is None:
                val_channels = self.value
            else:
                val_channels = resize_channels(self.value, channels, mode_shrink, mode_extend)
            return torch.cat([torch.full(size, val) for val in self.value], dim=-1)

        if channels is None:
            return self.image
        return resize_channels(self.image, channels)

    def is_value(self):
        return not self.value is None
    
    def as_primitive_float(self) -> float:
        if self.image:
            if self.is_color and self.image.size()[3] == 4:
                im_to_mean, _ = self.image.split((3, 1), dim=-1)
            else:
                im_to_mean = self.image
            
            if self.is_color:
                return rgb_to_srgb(im_to_mean.mean()).item()
            return im_to_mean.mean().item()
        
        sum = 0
        if self.is_color and len(self.value) == 4:
            items_to_sum = self.value[:-1]
        else:
            items_to_sum = self.value

        for e in items_to_sum:
            sum += e
        sum /= len(items_to_sum)
        if self.is_color:
            return rgb_to_srgb(torch.Tensor((sum,))).item()
        return sum

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
            break

    if not res:
        res = default
    
    for b in blens:
        b.set_canvas(res)

    return res

def extend_channels(te: torch.Tensor, desired_count: int, mode: int=0):
    """
    - 0 = Pad with 0\n
    - 1 = Repeat, cutting off what doesn't fit\n
    - 2 = Repeat last channel\n
    - 3 = Pad with 1\n
    """
    assert te.size()[-1] < desired_count, f"{te.size()} has too many channels to extend to {desired_count}"
    if mode == 0:
        return torch.cat((te, torch.zeros(list(te.size())[:-1] + [desired_count - te.size()[-1]])), dim=-1)
    elif mode == 3:
        return torch.cat((te, torch.ones(list(te.size())[:-1] + [desired_count - te.size()[-1]])), dim=-1)
    elif mode == 1:
        chopcount = te.size()[-1] - desired_count % te.size()[-1]
        if chopcount == te.size()[-1]:
            chopcount = 0
        repeatcount = ceil(desired_count / te.size()[-1])
        repeated = te.repeat((1, 1, 1, repeatcount))
        if chopcount > 0:
            chopped = repeated.split((desired_count, chopcount), dim=3)
        else:
            chopped = repeated
        return chopped
    elif mode == 2:
        first, last = te.split((te.size()[-1]-1, 1), dim=-1)
        return torch.cat((first, last.repeat((1, 1, 1, desired_count - te.size()[-1]))), dim=-1)
    
    raise Exception(f"Unrecognized channel extension mode {mode}")

def shrink_channels(te: torch.Tensor, desired_count: int, mode: int=0):
    """
    - 0 = Chop off\n
    - 1 = Pair downscale
    """
    assert te.size()[-1] > desired_count, f"{te.size()} has too few channels to shrink to {desired_count}"
    if mode == 0:
        first, _ = te.split((desired_count, te.size()[-1] - desired_count), dim=-1)
        return first
    elif mode == 1:
        splits = te.split(ceil(te.size()[-1] / desired_count), dim=3)
        avg = [s.mean(dim=-1).unsqueeze(-1) for s in splits]
        return torch.cat(avg, dim=-1)
    
    raise Exception(f"Unrecognized channel extension mode {mode}")

def resize_channels(te: torch.Tensor | list, desired_count: int, mode_shrink: int=0, mode_extend: int=0):
    """
    Shrink modes:\n
    - 0 = Chop off\n
    - 1 = Pair downscale\n
    Extend modes:\n
    - 0 = Pad with 0\n
    - 1 = Repeat, cutting off what doesn't fit\n
    - 2 = Repeat last channel\n
    - 3 = Pad with 1\n
    """
    islist = type(te) == list
    if islist:
        te = torch.Tensor(te)

    if te.size()[-1] == desired_count:
        res = te
    if te.size()[-1] > desired_count:
        res = shrink_channels(te, desired_count, mode_shrink)
    if te.size()[-1] < desired_count:
        res = extend_channels(te, desired_count, mode_extend)
    
    if islist:
        return res.tolist()
    return res
    
def ensure_samesize_channels(*blens: BlenderData, force=None):
    first_te = None
    for b in blens:
        if not b.image is None:
            first_te = b.image
            break

    if force is None:
        if not first_te is None:
            desired_channels = first_te.size()[3]
        else:
            if type(blens[0].value) in [int, float]:
                desired_channels = 1
            else:
                desired_channels = len(blens[0].value)
    else:
        desired_channels = force
    
    for b in blens:
        if not b.image is None:
            b.image = resize_channels(b.image, desired_channels)
        else:
            if type(b.value) in [int, float]:
                as_te = torch.Tensor((b.value, ))
            else:
                as_te = torch.Tensor(b.value)
            as_te = as_te.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            as_te = resize_channels(as_te, desired_channels)
            b.value = as_te[0][0][0].tolist()
    return blens

    

# Returns [1] x [Height] x [Width] x [3] UV-map in this format:
# (0.0, 0.999) --- (0.999, 0.999)
#      |                  |
# (0.0,   0.0) --- (0.999,   0.0)
def make_uv(width, height, w1 = True, device="cpu"):
    u_1d1c = torch.linspace(0.0, 1.0, width+1)[:-1]
    u_1d3c = torch.stack((u_1d1c, torch.zeros_like(u_1d1c), torch.zeros_like(u_1d1c)), dim=-1)
    u_2d = u_1d3c.repeat((height, 1, 1)).to(device)

    v_1d1c = torch.linspace(1.0, 0.0, height+1)[1:]
    v_1d3c = torch.stack((torch.zeros_like(v_1d1c), v_1d1c, torch.zeros_like(v_1d1c)), dim=-1)
    v_2d = v_1d3c.repeat((width, 1, 1)).permute(1, 0, 2).to(device)

    wh = (width, height)
    w = torch.stack((torch.zeros(wh), torch.zeros(wh), torch.ones(wh)), dim=-1)
    if not w1: w = w * 0.0

    return (u_2d+v_2d+w).unsqueeze(0)

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

def VECTOR_INPUT(name, default: float | tuple = 0.0, min=-100000.0, max=100000.0, step=0.01, hidden_default=False):
    if type(default) in [float, int]:
        default = (default, default, default)
    if hidden_default:
        return {name: ("*", {})}
    return {name: ("*", {}),
            name+"X": ("FLOAT", {"default": default[0], "min": min, "max": max, "step": step}),
            name+"Y": ("FLOAT", {"default": default[1], "min": min, "max": max, "step": step}),
            name+"Z": ("FLOAT", {"default": default[2], "min": min, "max": max, "step": step}),}

__DIMENSIONS = ["1D", "2D", "3D", "4D"]
DIMENSIONS_INPUT = {"Dimensions": (__DIMENSIONS, {"default": "3D"})}

def get_kwargs_dim(kwargs):
    dims = kwargs["Dimensions"]
    return __DIMENSIONS.index(dims) + 1

def BLENDER_OUTPUT(single=False):
    if single:
        return (("BLENDER", ))
    return ("BLENDER", "IMAGE", )

def BLENDER_OUTPUT_FLOAT(single=False):
    if single:
        return (("BLENDER_FLOAT", ))
    return ("BLENDER_FLOAT", "IMAGE", )

def BLENDER_OUTPUT_COLOR(single=False):
    if single:
        return (("BLENDER_RGB", ))
    return ("BLENDER_RGB", "IMAGE", )

def BLENDER_OUTPUT_VECTOR(single=False):
    if single:
        return (("BLENDER_VEC", ))
    return ("BLENDER_VEC", "IMAGE", )

def BLENDER_OUTPUT_WITHFAC(single=False):
    if single:
        return ("BLENDER_RGB", "BLENDER_FLOAT", )
    return ("BLENDER_RGB", "BLENDER_FLOAT", "IMAGE", "IMAGE", )

# Types: RGB / COLOR / RGBA, FLOAT / VALUE, VEC / VECTOR
def BLENDER_OUTPUT_BYLIST(typelist: list[str], single=False):
    l_up = [t.upper() for t in typelist]
    l_conv = []
    for t in l_up:
        if t in ["RGB", "RGBA", "COLOR", "COLOUR"]:
            l_conv.append("BLENDER_RGB")
        elif t in ["FLOAT", "VALUE"]:
            l_conv.append("BLENDER_FLOAT")
        elif t in ["VEC", "VECTOR"]:
            l_conv.append("BLENDER_VEC")
        else:
            raise Exception(f"Unknown Blender type: {t}")
    
    if single:
        return tuple(l_conv)
    return tuple(l_conv + ["IMAGE"] * len(l_conv))


def BLEND_VALID_INPUTS(input_types, ref):
    #input_types is a dict of {"Name": "TYPE_NAME"}
    return True

FILTERS = ["Nearest", "Bilinear", "Bicubic"]
EXTENSIONS = ["Clip", "Repeat", "Extend", "Mirror"]

def FILTER_AND_EXTENSION():
    return {"Filter": (FILTERS, ), 
            "Extension": (EXTENSIONS, ), }

def get_filter_extension(kwargs):
    f = FILTERS.index(kwargs["Filter"])
    e = EXTENSIONS.index(kwargs["Extension"])
    return f, e