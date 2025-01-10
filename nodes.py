import torch
import torch.nn.functional as functional

from .nodes_base import *
from .util import *

# Image: batch, x, y, channels

inf = 1000000

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
    
    RETURN_TYPES = (*BLENDER_OUTPUT(), "FLOAT", )
    FUNCTION = "get_value"
    CATEGORY = "Blender/Input" 
    
    def get_value(self, Value):
        bd = BlenderData(Value)
        return (bd, bd.as_out(), Value)
    
class BlenderRGB:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Mode": (["RGB", "HSV"], ),
                "Red/Hue": ("FLOAT", {"default": 1.0, "step": 0.01, "min": 0.0, "max": 1.0}),
                "Green/Saturation": ("FLOAT", {"default": 1.0, "step": 0.01, "min": 0.0, "max": 1.0}),
                "Blue/Value": ("FLOAT", {"default": 1.0, "step": 0.01, "min": 0.0, "max": 1.0}),
                "Alpha": ("FLOAT", {"default": 1.0, "step": 0.01, "min": 0.0, "max": 1.0}),
            },
            # TODO: Can dynamic optional inputs be done based on Channels selected?
        }
    
    RETURN_TYPES = (*BLENDER_OUTPUT(),)
    RETURN_NAMES = ("Color", "Image")
    FUNCTION = "get_rgb"
    CATEGORY = "Blender/Input" 
    
    def get_value(self, **kwargs):
        mode = kwargs["Mode"]
        r = kwargs["Red/Hue"]
        g = kwargs["Green/Saturation"]
        b = kwargs["Blue/Value"]
        a = kwargs["Alpha"]
        
        if mode == "RGB":
            bd = BlenderData((r, g, b, a))
        
        if mode == "HSV":
            r, g, b = hsv_to_rgb_primitive(r, g, b)
            bd = BlenderData((r, g, b, a))
        
        return (bd, bd.as_out())

class BlenderBrightnessContrast:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                **COLOR_INPUT("Color"),
                **FLOAT_INPUT("Bright", 0.0, -100.0, 100.0, 0.01),
                **FLOAT_INPUT("Contrast", 0.0, -100.0, 100.0, 0.01),
            },
        }
    
        
    @classmethod
    def VALIDATE_INPUTS(self, input_types):
        return BLEND_VALID_INPUTS(input_types, self.INPUT_TYPES())

    RETURN_TYPES = (*BLENDER_OUTPUT(),)
    RETURN_NAMES = ("Color", "Image")
    FUNCTION = "brightness_contrast"
    CATEGORY = "Blender/Color" 

    def brightness_contrast(self, **kwargs):
        b_color = BlenderData(kwargs, "Color")
        b_bright = BlenderData(kwargs, "Bright")
        b_contrast = BlenderData(kwargs, "Contrast")
        guess_canvas(b_color, b_bright, b_contrast)
        
        color, alpha = b_color.as_rgb_a()
        bright = b_bright.as_float()
        contrast = b_contrast.as_float()

        # (bright + color * (contrast + val(1.0f)) - contrast * val(0.5f)).max(val(0.0f))
        res = (bright + color * (contrast + 1.0) - contrast * 0.5).maximum(torch.zeros_like(color))

        b_res = BlenderData(res, alpha)

        return (b_res, b_res.as_out(), )
    
class BlenderGamma:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                **COLOR_INPUT("Color"),
                **FLOAT_INPUT("Gamma", 1.0, 0.0001, 1000, 0.001),
            },
        }
    
    @classmethod
    def VALIDATE_INPUTS(self, input_types):
        return BLEND_VALID_INPUTS(input_types, self.INPUT_TYPES())

    RETURN_TYPES = (*BLENDER_OUTPUT(),)
    RETURN_NAMES = ("Color", "Image")
    FUNCTION = "gamma"
    CATEGORY = "Blender/Color"

    def gamma(self, **kwargs):
        b_color = BlenderData(kwargs, "Color")
        b_gamma = BlenderData(kwargs, "Gamma")
        guess_canvas(b_color, b_gamma)

        color, alpha = b_color.as_rgb_a()
        gamma = b_gamma.as_float()

        res = torch.pow(color, gamma).maximum(torch.zeros_like(color))
        b_res = BlenderData(res, alpha)
        return (b_res, b_res.as_out())

class BlenderInvertColor:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                **FLOAT_INPUT("Fac", 1.0),
                **COLOR_INPUT("Color", 0.0),
            },
        }
    
    @classmethod
    def VALIDATE_INPUTS(self, input_types):
        return BLEND_VALID_INPUTS(input_types, self.INPUT_TYPES())

    RETURN_TYPES = (*BLENDER_OUTPUT(), )
    RETURN_NAMES = ("Color", "Image")
    FUNCTION = "invert"
    CATEGORY = "Blender/Color"

    def invert(self, **kwargs):
        b_color = BlenderData(kwargs, "Color")
        b_fac = BlenderData(kwargs, "Fac")
        guess_canvas(b_color, b_fac)

        color, alpha = b_color.as_rgb_a()
        fac = b_fac.as_float()

        res = ((1.0-color)*fac + color*(1.0-fac)).maximum(torch.zeros_like(color))
        
        b_res = BlenderData(res, alpha)
        return (b_res, b_res.as_out())
    
class BlenderSeparateColor:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "Mode": (["RGB", "HSV", "HSL"], ), #TODO: YUV, YCbCr
                **COLOR_INPUT("Color", 1.0, True),
            }
            # TODO: Can dynamic optional inputs be done based on Mode selected?
        }
    
    @classmethod
    def VALIDATE_INPUTS(self, input_types):
        return BLEND_VALID_INPUTS(input_types, self.INPUT_TYPES())
    
    RETURN_TYPES = (*BLENDER_OUTPUT(), *BLENDER_OUTPUT(), *BLENDER_OUTPUT(), *BLENDER_OUTPUT(),)
    RETURN_NAMES = ("R/H", "R/H", "G/S", "G/S", "B/V", "B/V", "A", "A")
    FUNCTION = "separate_color"
    CATEGORY = "Blender/Converter"
    
    def separate_color(self, **kwargs):
        mode = kwargs["Mode"]
        b_col = BlenderData(kwargs, "Color")
        guess_canvas(b_col)
        
        rgb, a = b_col.as_rgb_a()

        if mode == "RGB":
            pass

        if mode == "HSV":
            rgb = rgb_to_hsv(rgb)

        if mode == "HSL":
            rgb = rgb_to_hsl(rgb)

        r, g, b = rgb.split(1, dim=-1)
        b_r, b_g, b_b, b_a = BlenderData(r), BlenderData(g), BlenderData(b), BlenderData(a)
        
        return (b_r, b_r.as_out(), b_g, b_g.as_out(), b_b, b_b.as_out(), b_a, b_a.as_out())
