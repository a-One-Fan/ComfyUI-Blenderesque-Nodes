import torch
import torch.nn.functional as functional

# Image: batch, x, y, channels

IMAGE_NONE = None
inf = 1000000

def image_prim(r, g, b, a=1.0):
    return torch.stack((torch.full((1, 1, 1), r), torch.full((1, 1, 1), g), 
                        torch.full((1, 1, 1), b), torch.full((1, 1, 1), a)), dim=-1)

def is_image_prim(te):
    return te.size()[:-1] == (1, 1, 1)

def interp(te, shape):
    if te.size() == shape:
        return te
    if is_image_prim(te):
        tostack = []
        for i in range(shape[-1]):
            tostack.append(torch.full(shape[:-1], te[0][0][0][i]))
        return torch.stack(tostack, dim=-1)

    mode = 'bilinear'
    if te.size()[1] < shape[1] and te.size()[2] < shape[2]:
        mode = 'bicubic'
    print(f"Interp: {mode}, {te.size()}, {shape}")
    return functional.interpolate(te, shape, mode=mode)

def get_common_shape(*tensors):
    biggest_size = (1, 1)
    for t in tensors:
        if (t.size()[1] * t.size()[2]) > (biggest_size[0] * biggest_size[1]):
            biggest_size = t.size()[1:-1]
    return biggest_size

def make_common_shape(*tensors):
    shape = get_common_shape(*tensors)
    res = []
    for t in tensors:
        res.append(interp(t, (t.size()[0], shape[0], shape[1], t.size()[3])))
    return res

def col_to_float(tensor):
    channels_first = tensor.permute(3, 1, 2, 0)
    avg = (channels_first[0] + channels_first[1] + channels_first[2]) / 3.0
    stacked = torch.stack((avg, avg, avg))
    normal_order = stacked.permute(3, 1, 2, 0)
    return normal_order

def split_rgba(tensor):
    if tensor.size()[3] == 3:
        return tensor, torch.full(list(tensor.size())[:-1] + [1], 1.0)
    return tensor.split([3, 1], dim=-1)

def join_rgba(rgb, a):
    return torch.cat((rgb, a), dim=-1)

COLMAX = 1.0 
# Blender has soft min and max. 
# All colors have a soft max of 1.0 can't be dragged further, but can be typed in higher. 
# ComfyUI does not seem to have soft min/max?
COLSTEP = 0.005

INPUT_RGB_WHITE = {"ColorR": ("FLOAT", {"default": 1.0, "min": 0, "max": COLMAX, "step": COLSTEP}),
                   "ColorG": ("FLOAT", {"default": 1.0, "min": 0, "max": COLMAX, "step": COLSTEP}),
                   "ColorB": ("FLOAT", {"default": 1.0, "min": 0, "max": COLMAX, "step": COLSTEP}),}

INPUT_RGB_WHITE08 = {"ColorR": ("FLOAT", {"default": 0.8, "min": 0, "max": COLMAX, "step": COLSTEP}),
                     "ColorG": ("FLOAT", {"default": 0.8, "min": 0, "max": COLMAX, "step": COLSTEP}),
                     "ColorB": ("FLOAT", {"default": 0.8, "min": 0, "max": COLMAX, "step": COLSTEP}),}

INPUT_RGB_BLACK = {"ColorR": ("FLOAT", {"default": 0.0, "min": 0, "max": COLMAX, "step": COLSTEP}),
                   "ColorG": ("FLOAT", {"default": 0.0, "min": 0, "max": COLMAX, "step": COLSTEP}),
                   "ColorB": ("FLOAT", {"default": 0.0, "min": 0, "max": COLMAX, "step": COLSTEP}),}

class BlenderValue:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Value": ("FLOAT", {"default": 0.0, "step": 0.01, "min": -inf, "max": inf}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "get_value"
    CATEGORY = "Blender/Input" 
    
    def get_value(self, Value):
        return (image_prim(Value, Value, Value), )

class BlenderBrightnessContrast:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "Color": ("IMAGE",),
                **INPUT_RGB_WHITE,
                "Bright": ("IMAGE",),
                "BrightF": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "Contrast": ("IMAGE",),
                "ContrastF": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "brightness_contrast"
    CATEGORY = "Blender/Color" 

    def brightness_contrast(self, ColorR, ColorG, ColorB, BrightF, ContrastF, Color: torch.Tensor = IMAGE_NONE, Bright: torch.Tensor = IMAGE_NONE, Contrast: torch.Tensor = IMAGE_NONE):
        if Color == IMAGE_NONE:
            Color = image_prim(ColorR, ColorG, ColorB)
        if Bright == IMAGE_NONE:
            Bright = image_prim(BrightF, BrightF, BrightF)
        if Contrast == IMAGE_NONE:
            Contrast = image_prim(ContrastF, ContrastF, ContrastF)
        
        Bright = col_to_float(Bright)
        Contrast = col_to_float(Contrast)
        Color, Color_Alpha = split_rgba(Color)
        Color, Color_Alpha, Bright, Contrast = make_common_shape(Color, Color_Alpha, Bright, Contrast)

        Color = torch.pow(Color, 2.2) # TODO: Blender uses sRGB by default. How should colorspace be handled?

        # (bright + color * (contrast + val(1.0f)) - contrast * val(0.5f)).max(val(0.0f))
        res = (Bright + Color * (Contrast + 1.0) - Contrast * 0.5).maximum(torch.zeros_like(Color))
        
        res = torch.pow(res, 1/2.2)

        return (join_rgba(res, Color_Alpha), )
    
class BlenderGamma:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "Color": ("IMAGE",),
                **INPUT_RGB_WHITE,
                "Gamma": ("IMAGE",),
                "GammaF": ("FLOAT", {"default": 1.0, "min": 0.0001, "max": 100000, "step": 0.00001}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "gamma"
    CATEGORY = "Blender/Color"

    def gamma(self, ColorR: float, ColorG: float, ColorB: float, GammaF: float, Color: torch.Tensor = IMAGE_NONE, Gamma: torch.Tensor = IMAGE_NONE):
        if Color == IMAGE_NONE:
            Color = image_prim(ColorR, ColorG, ColorB)
        if Gamma == IMAGE_NONE:
            Gamma = image_prim(GammaF, GammaF, GammaF)
        Gamma = col_to_float(Gamma)
        Color, Color_Alpha = split_rgba(Color)
        Color, Color_Alpha, Gamma = make_common_shape(Color, Color_Alpha, Gamma)

        res = torch.pow(Color, Gamma).maximum(torch.zeros_like(Color))
        return (join_rgba(res, Color_Alpha), )

class BlenderInvertColor:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "Fac": ("IMAGE", ),
                "FacF": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "Color": ("IMAGE",),
                **INPUT_RGB_BLACK,
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "invert"
    CATEGORY = "Blender/Color"

    def invert(self, FacF, ColorR, ColorG, ColorB, Color: torch.Tensor = IMAGE_NONE, Fac: torch.Tensor = IMAGE_NONE):
        if Color == IMAGE_NONE:
            Color = image_prim(ColorR, ColorG, ColorB)
        if Fac == IMAGE_NONE:
            Fac = image_prim(FacF, FacF, FacF)
        
        Fac = col_to_float(Fac)
        Color, Color_Alpha = split_rgba(Color)
        Color, Color_Alpha, Fac = make_common_shape(Color, Color_Alpha, Fac)

        Color = torch.pow(Color, 2.2) # TODO: Blender uses sRGB by default. How should colorspace be handled?

        # (bright + color * (contrast + val(1.0f)) - contrast * val(0.5f)).max(val(0.0f))
        res = ((1.0-Color)*Fac + Color*(1.0-Fac)).maximum(torch.zeros_like(Color))
        
        res = torch.pow(res, 1/2.2)
        return (join_rgba(res, Color_Alpha), )