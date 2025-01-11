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
    
    def get_rgb(self, **kwargs):
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
                "Mode": (["RGB", "HSV", "HSL", "YUV", "YCbCr"], ),
                **COLOR_INPUT("Color", 1.0, True),
            }
            # TODO: Can dynamic optional inputs be done based on Mode selected?
        }
    
    @classmethod
    def VALIDATE_INPUTS(self, input_types):
        return BLEND_VALID_INPUTS(input_types, self.INPUT_TYPES())
    
    RETURN_TYPES = (*BLENDER_OUTPUT(), *BLENDER_OUTPUT(), *BLENDER_OUTPUT(), *BLENDER_OUTPUT(),)
    RETURN_NAMES = ("R", "R", "G", "G", "B", "B", "A", "A")
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

        if mode == "YUV":
            rgb = rgb_to_yuv(rgb)

        if mode == "YCbCr":
            rgb = rgb_to_ycc(rgb)

        r, g, b = rgb.split(1, dim=-1)
        b_r, b_g, b_b, b_a = BlenderData(r), BlenderData(g), BlenderData(b), BlenderData(a)
        
        return (b_r, b_r.as_out(), b_g, b_g.as_out(), b_b, b_b.as_out(), b_a, b_a.as_out())
    

class BlenderMapRange:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "Data Type": (["Float", "Vector"], ),
                "Interpolation Type": (["Linear", "Stepped Linear", "Smooth Step", "Smoother Step"], ),
                "Clamp": ("BOOLEAN", {"default": True}),
                **FLOAT_INPUT("Value", 1.0, -inf, inf),
                **FLOAT_INPUT("From Min Float", 0.0, -inf, inf),
                **FLOAT_INPUT("From Max Float", 1.0, -inf, inf),
                **FLOAT_INPUT("To Min Float", 0.0, -inf, inf),
                **FLOAT_INPUT("To Max Float", 1.0, -inf, inf),
                **VECTOR_INPUT("Vector", 0.0, hidden_default=True),
                **VECTOR_INPUT("From Min Vector", 0.0),
                **VECTOR_INPUT("From Max Vector", 1.0),
                **VECTOR_INPUT("To Min Vector", 0.0),
                **VECTOR_INPUT("To Max Vector", 1.0),
                **FLOAT_INPUT("Steps", 4.0, 0.00001, 1000000.0),
            }
            # TODO: Can dynamic optional inputs be done based on Mode selected?
        }
    
    @classmethod
    def VALIDATE_INPUTS(self, input_types):
        return BLEND_VALID_INPUTS(input_types, self.INPUT_TYPES())
    
    RETURN_TYPES = (*BLENDER_OUTPUT(), )
    RETURN_NAMES = ("Result", "Result")
    FUNCTION = "map_range"
    CATEGORY = "Blender/Converter"
    
    def map_range(self, **kwargs):
        dtype = kwargs["Data Type"]
        mode = kwargs["Interpolation Type"]

        b_steps = BlenderData(kwargs, "Steps")

        if dtype == "Float":
            b_val = BlenderData(kwargs, "Value")
            b_frommin = BlenderData(kwargs, "From Min Float")
            b_frommax = BlenderData(kwargs, "From Max Float")
            b_tomin = BlenderData(kwargs, "To Min Float")
            b_tomax = BlenderData(kwargs, "To Max Float")
            
            guess_canvas(b_val, b_frommin, b_frommax, b_tomin, b_tomax, b_steps)

            val = b_val.as_float()
            frommin = b_frommin.as_float()
            frommax = b_frommax.as_float()
            tomin = b_tomin.as_float()
            tomax = b_tomax.as_float()
        elif dtype == "Vector":
            b_val = BlenderData(kwargs, "Vector")
            b_frommin = BlenderData(kwargs, "From Min Vector")
            b_frommax = BlenderData(kwargs, "From Max Vector")
            b_tomin = BlenderData(kwargs, "To Min Vector")
            b_tomax = BlenderData(kwargs, "To Max Vector")
            
            guess_canvas(b_val, b_frommin, b_frommax, b_tomin, b_tomax, b_steps)

            val = b_val.as_rgb()
            frommin = b_frommin.as_rgb()
            frommax = b_frommax.as_rgb()
            tomin = b_tomin.as_rgb()
            tomax = b_tomax.as_rgb()

        steps = b_steps.as_float()

        # tomin - fac * tomin + fac * tomax = val
        # (tomax - tomin)*fac = val - tomin
        # fac = (val - tomin) / (tomax - tomin)
        fac = (val - frommin) / (frommax - frommin)
        if mode == "Stepped Linear":
            fac = torch.floor(fac * (steps + 1.0)) / steps
        elif mode == "Smooth Step":
            fac = (3.0 - 2.0 * fac) * (fac * fac)
        elif mode == "Smoother Step":
            fac = fac * fac * fac * (fac * (fac * 6.0 - 15.0) + 10.0)

        res = (1.0 - fac) * tomin + fac * tomax
        
        if kwargs["Clamp"]:
            res = torch.clamp(res, tomin, tomax)

        b_r = BlenderData(res, no_colortransform=True)
        
        return (b_r, b_r.as_out(), )
    
class BlenderBlackbody:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                **FLOAT_INPUT("Temperature", 1500.0, 800.0, 12000.0),
            },
        }
    
    @classmethod
    def VALIDATE_INPUTS(self, input_types):
        return BLEND_VALID_INPUTS(input_types, self.INPUT_TYPES())

    RETURN_TYPES = (*BLENDER_OUTPUT(), )
    RETURN_NAMES = ("Color", "Image")
    FUNCTION = "blackbody"
    CATEGORY = "Blender/Converter"

    def blackbody(self, **kwargs):
        b_fac = BlenderData(kwargs, "Temperature")
        guess_canvas(b_fac)

        fac = b_fac.as_float()

        res = blackbody_temperature_to_rec709(fac) # TODO: rec709 -> linear?
        
        b_res = BlenderData(res, no_colortransform=True)
        return (b_res, b_res.as_out())
    
class BlenderRGBtoBW:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                **COLOR_INPUT("Color", default=0.5),
            },
        }
    
    @classmethod
    def VALIDATE_INPUTS(self, input_types):
        return BLEND_VALID_INPUTS(input_types, self.INPUT_TYPES())

    RETURN_TYPES = (*BLENDER_OUTPUT(), )
    RETURN_NAMES = ("Val", "Image")
    FUNCTION = "rgb_to_bw"
    CATEGORY = "Blender/Converter"

    def rgb_to_bw(self, **kwargs):
        b_col = BlenderData(kwargs, "Color")
        guess_canvas(b_col)
        
        b_res = BlenderData(b_col.as_float())
        return (b_res, b_res.as_out())
    
class BlenderSeparateXYZ:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                **VECTOR_INPUT("Vector"),
            },
        }
    
    @classmethod
    def VALIDATE_INPUTS(self, input_types):
        return BLEND_VALID_INPUTS(input_types, self.INPUT_TYPES())

    RETURN_TYPES = (*BLENDER_OUTPUT(), *BLENDER_OUTPUT(), *BLENDER_OUTPUT(), )
    RETURN_NAMES = ("X", "X", "Y", "Y", "Z", "Z")
    FUNCTION = "separate_xyz"
    CATEGORY = "Blender/Converter"

    def separate_xyz(self, **kwargs):
        b_vec = BlenderData(kwargs, "Vector", no_colortransform=True)
        guess_canvas(b_vec)
        
        xyz = b_vec.as_rgb()
        x, y, z = xyz.split(1, dim=-1)
        b_x, b_y, b_z = BlenderData(x), BlenderData(y), BlenderData(z)
        return (b_x, b_x.as_out(), b_y, b_y.as_out(), b_z, b_z.as_out())
    
class BlenderCombineXYZ:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                **FLOAT_INPUT("X", default=0.0),
                **FLOAT_INPUT("Y", default=0.0),
                **FLOAT_INPUT("Z", default=0.0),
            },
        }
    
    @classmethod
    def VALIDATE_INPUTS(self, input_types):
        return BLEND_VALID_INPUTS(input_types, self.INPUT_TYPES())

    RETURN_TYPES = (*BLENDER_OUTPUT(), )
    RETURN_NAMES = ("Vector", "Image")
    FUNCTION = "combine_xyz"
    CATEGORY = "Blender/Converter"

    def combine_xyz(self, **kwargs):
        b_x = BlenderData(kwargs, "X")
        b_y = BlenderData(kwargs, "Y")
        b_z = BlenderData(kwargs, "Z")
        guess_canvas(b_x, b_y, b_z)

        x, y, z = b_x.as_float(), b_y.as_float(), b_z.as_float()
        
        b_res = BlenderData(torch.cat((x, y, z), dim=-1), no_colortransform=True)
        return (b_res, b_res.as_out())
    
class BlenderClamp:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "Clamp Type": (["Min Max", "Range"], ),
                **FLOAT_INPUT("Value", 1.0, -inf, inf),
                **FLOAT_INPUT("Min", 0.0, -inf, inf),
                **FLOAT_INPUT("Max", 1.0, -inf, inf),
            },
        }
    
    @classmethod
    def VALIDATE_INPUTS(self, input_types):
        return BLEND_VALID_INPUTS(input_types, self.INPUT_TYPES())

    RETURN_TYPES = (*BLENDER_OUTPUT(), )
    RETURN_NAMES = ("Result", "Result")
    FUNCTION = "clamp"
    CATEGORY = "Blender/Converter"

    def clamp(self, **kwargs):
        b_val = BlenderData(kwargs, "Value")
        b_min = BlenderData(kwargs, "Min")
        b_max = BlenderData(kwargs, "Max")
        guess_canvas(b_val, b_min, b_max)

        mode = kwargs["Clamp Type"]

        val, mi, mx = b_val.as_float(), b_min.as_float(), b_max.as_float()

        if mode == "Range":
            miswap = tmix(mi, mx, mi > mx)
            mxswap = tmix(mx, mi, mi > mx)
            mi = miswap
            mx = mxswap

        val = tmix(tmix(val, mi, val < mi), mx, val > mx)
        
        b_res = BlenderData(val)
        return (b_res, b_res.as_out())

class BlenderHueSaturationValue:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                **FLOAT_INPUT("Hue", 0.5),
                **FLOAT_INPUT("Saturation", 1.0, 0.0, 5.0),
                **FLOAT_INPUT("Value", 1.0, 0.0, 5.0),
                **FLOAT_INPUT("Fac", 1.0, 0.0, 1.0),
                **COLOR_INPUT("Color"),
            },
        }
    
    @classmethod
    def VALIDATE_INPUTS(self, input_types):
        return BLEND_VALID_INPUTS(input_types, self.INPUT_TYPES())

    RETURN_TYPES = (*BLENDER_OUTPUT(), )
    RETURN_NAMES = ("Result", "Result")
    FUNCTION = "do_hsv"
    CATEGORY = "Blender/Color"

    def do_hsv(self, **kwargs):
        b_hue = BlenderData(kwargs, "Hue")
        b_sat = BlenderData(kwargs, "Saturation")
        b_val = BlenderData(kwargs, "Value")
        b_fac = BlenderData(kwargs, "Fac")
        b_col = BlenderData(kwargs, "Color")
        guess_canvas(b_hue, b_sat, b_val, b_fac, b_col)

        hue, sat, val, fac = b_hue.as_float(), b_sat.as_float(), b_val.as_float(), b_fac.as_float()
        col, alpha = b_col.as_rgb_a()

        hsv = rgb_to_hsv(col)
        h, s, v = hsv.split(1, dim=-1)
        h += hue - 0.5
        h = torch.remainder(h, torch.ones_like(h))
        s *= sat
        s = torch.clamp(s, 0.0, 1.0)
        v *= val
        v = torch.clamp(v, 0.0, 1.0)

        rgb = hsv_to_rgb(torch.cat((h, s, v), dim=-1))
        rgb = tmix(col, rgb, fac)

        b_res = BlenderData(rgb, alpha)
        return (b_res, b_res.as_out())

class BlenderCombineColor:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "Mode": (["RGB", "HSV", "HSL", "YUV", "YCbCr"], ),
                **FLOAT_INPUT("R"),
                **FLOAT_INPUT("G"),
                **FLOAT_INPUT("B"),
                **FLOAT_INPUT("A"),
            }
            # TODO: Can dynamic optional inputs be done based on Mode selected?
        }
    
    @classmethod
    def VALIDATE_INPUTS(self, input_types):
        return BLEND_VALID_INPUTS(input_types, self.INPUT_TYPES())
    
    RETURN_TYPES = (*BLENDER_OUTPUT(), )
    RETURN_NAMES = ("Color", "Image", )
    FUNCTION = "combine_color"
    CATEGORY = "Blender/Converter"
    
    def combine_color(self, **kwargs):
        mode = kwargs["Mode"]
        b_r = BlenderData(kwargs, "R")
        b_g = BlenderData(kwargs, "G")
        b_b = BlenderData(kwargs, "B")
        b_a = BlenderData(kwargs, "A")
        guess_canvas(b_r, b_g, b_b, b_a)
        
        r, g, b, a = b_r.as_float(), b_g.as_float(), b_b.as_float(), b_a.as_float()
        rgb = torch.cat((r, g, b), dim=-1)

        if mode == "RGB":
            pass

        if mode == "HSV":
            rgb = hsv_to_rgb(rgb)

        if mode == "HSL":
            rgb = hsl_to_rgb(rgb)

        if mode == "YUV":
            rgb = yuv_to_rgb(rgb)

        if mode == "YCbCr":
            rgb = ycc_to_rgb(rgb)

        b_res = BlenderData(rgb, a)
        
        return (b_res, b_res.as_out(), )