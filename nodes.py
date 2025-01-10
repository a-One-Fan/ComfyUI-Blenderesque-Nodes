import torch
import torch.nn.functional as functional

# Image: batch, x, y, channels

IMAGE_NONE = None
inf = 1000000

def image_prim(r, g, b):
    return torch.stack((torch.full((1, 1, 1), r), torch.full((1, 1, 1), g), torch.full((1, 1, 1), b)), dim=-1)

def is_image_prim(te):
    return te.size() == (1, 1, 1, 3)

def interp(te, shape):
    if te.size() == shape:
        return te
    if is_image_prim(te):
        r = torch.full(shape[:-1], te[0][0][0][0])
        g = torch.full(shape[:-1], te[0][0][0][0])
        b = torch.full(shape[:-1], te[0][0][0][0])
        return torch.stack((r, g, b), dim=-1)

    mode = 'bilinear'
    if te.size()[1] < shape[1] and te.size()[2] < shape[2]:
        mode = 'bicubic'
    print(f"Interp: {mode}, {te.size()}, {shape}")
    return functional.interpolate(te, shape, mode=mode)

def get_common_shape(*tensors):
    biggest_size = (1, 1, 1, 3)
    for t in tensors:
        if (t.size()[1] * t.size()[2]) > (biggest_size[1] * biggest_size[2]):
            biggest_size = t.size()
    return biggest_size

def make_common_shape(*tensors):
    shape = get_common_shape(*tensors)
    res = []
    for t in tensors:
        res.append(interp(t, shape))
    return res

def col_to_float(tensor):
    channels_first = tensor.permute(3, 1, 2, 0)
    avg = (channels_first[0] + channels_first[1] + channels_first[2]) / 3.0
    stacked = torch.stack((avg, avg, avg))
    normal_order = stacked.permute(3, 1, 2, 0)
    return normal_order

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
        
        Color, Bright, Contrast = make_common_shape(Color, Bright, Contrast)
        Bright = col_to_float(Bright)
        Contrast = col_to_float(Contrast)

        Color = torch.pow(Color, 2.2) # TODO: Blender uses sRGB by default. How should colorspace be handled?

        # (bright + color * (contrast + val(1.0f)) - contrast * val(0.5f)).max(val(0.0f))
        res = (Bright + Color * (Contrast + 1.0) - Contrast * 0.5).maximum(torch.zeros_like(Color))
        
        res = torch.pow(res, 1/2.2)

        return (res, )
    
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
        Color, Gamma = make_common_shape(Color, Gamma)

        res = torch.pow(Color, Gamma).maximum(torch.zeros_like(Color))
        return (res, )

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
        
        Color, Fac = make_common_shape(Color, Fac)
        Fac = col_to_float(Fac)

        Color = torch.pow(Color, 2.2) # TODO: Blender uses sRGB by default. How should colorspace be handled?

        # (bright + color * (contrast + val(1.0f)) - contrast * val(0.5f)).max(val(0.0f))
        res = ((1.0-Color)*Fac + Color*(1.0-Fac)).maximum(torch.zeros_like(Color))
        
        res = torch.pow(res, 1/2.2)
        return (res, )