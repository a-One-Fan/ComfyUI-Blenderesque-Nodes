import torch
import torch.nn.functional as functional

from .nodes_base import *
from .util import *
from .cl_wrapper import *

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
                #"Test": ("FLOAT", {"default": 0.0, "step": 0.01, "min": -inf, "max": inf}),
            }
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

        if dtype == "Float":
            b_val = BlenderData(kwargs, "Value")
            b_frommin = BlenderData(kwargs, "From Min Float")
            b_frommax = BlenderData(kwargs, "From Max Float")
            b_tomin = BlenderData(kwargs, "To Min Float")
            b_tomax = BlenderData(kwargs, "To Max Float")
            
            guess_canvas(b_val, b_frommin, b_frommax, b_tomin, b_tomax)

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
            
            guess_canvas(b_val, b_frommin, b_frommax, b_tomin, b_tomax)

            val = b_val.as_rgb()
            frommin = b_frommin.as_rgb()
            frommax = b_frommax.as_rgb()
            tomin = b_tomin.as_rgb()
            tomax = b_tomax.as_rgb()

        if(mode == "Stepped Linear"):
            b_steps = BlenderData(kwargs, "Steps")
            guess_canvas(b_val, b_steps)
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

        b_r = BlenderData(res)
        
        return (b_r, b_r.as_out(), )
    
class BlenderBlackbody:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                **FLOAT_INPUT("Temperature", 1500.0, 800.0, 12000.0, 1.0),
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
        
        b_res = BlenderData(res)
        return (b_res, b_res.as_out())
    
class BlenderWavelength:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                **FLOAT_INPUT("Wavelength", 500.0, 380.0, 780.0),
            },
        }
    
    @classmethod
    def VALIDATE_INPUTS(self, input_types):
        return BLEND_VALID_INPUTS(input_types, self.INPUT_TYPES())

    RETURN_TYPES = (*BLENDER_OUTPUT(), )
    RETURN_NAMES = ("Color", "Image")
    FUNCTION = "wavelength"
    CATEGORY = "Blender/Converter"

    def wavelength(self, **kwargs):
        b_fac = BlenderData(kwargs, "Wavelength")
        guess_canvas(b_fac)

        fac = b_fac.as_float()

        res = ciexyz_to_rgb(wavelength_to_xyz(fac))
        
        b_res = BlenderData(res)
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
        b_vec = BlenderData(kwargs, "Vector")
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
        
        b_res = BlenderData(torch.cat((x, y, z), dim=-1))
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
    
class BlenderSetAlpha:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "Mode": (["Apply Mask", "Replace Alpha"], ),
                **COLOR_INPUT("Color", 1.0),
                **FLOAT_INPUT("Alpha", 1.0, 0.0, 1.0),
            },
        }
    
    @classmethod
    def VALIDATE_INPUTS(self, input_types):
        return BLEND_VALID_INPUTS(input_types, self.INPUT_TYPES())

    RETURN_TYPES = (*BLENDER_OUTPUT(), )
    RETURN_NAMES = ("Result", "Result")
    FUNCTION = "set_alpha"
    CATEGORY = "Blender/Color"

    def set_alpha(self, **kwargs):
        b_col = BlenderData(kwargs, "Color")
        b_alpha = BlenderData(kwargs, "Alpha")
        guess_canvas(b_col, b_alpha)

        mode = kwargs["Mode"]

        col, orig_alpha = b_col.as_rgb_a()
        new_alpha = b_alpha.as_float()

        if mode == "Apply Mask":
            new_alpha *= orig_alpha
        
        b_res = BlenderData(col, new_alpha)
        return (b_res, b_res.as_out())
    
FILTERS = ["Nearest", "Bilinear", "Bicubic"]
EXTENSIONS = ["Clip", "Repeat", "Extend", "Mirror"]

class BlenderRotate:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "Filter": (FILTERS, ),
                "Extension": (EXTENSIONS, ),
                **FLOAT_INPUT("Rotation", 0.0, -360.0, 360.0, 2.0),
                **COLOR_INPUT("Color", 1.0, True),
            }
        }
    
    @classmethod
    def VALIDATE_INPUTS(self, input_types):
        return BLEND_VALID_INPUTS(input_types, self.INPUT_TYPES())
    
    RETURN_TYPES = (*BLENDER_OUTPUT(), )
    RETURN_NAMES = ("Image", "Image", )
    FUNCTION = "rotate"
    CATEGORY = "Blender/Transform"
    
    def rotate(self, **kwargs):
        filter = kwargs["Filter"]
        ext = kwargs["Extension"]

        b_loc = BlenderData((0.0, 0.0))
        if not kwargs.get("Rotation"):
            kwargs["RotationF"] *= PI/180
        b_rot = BlenderData(kwargs, "Rotation")
        b_scale = BlenderData((1.0, 1.0))
        b_col = BlenderData(kwargs, "Color")
        guess_canvas(b_col, b_loc, b_rot, b_scale)
        
        loc = b_loc.as_2wide()
        rot = b_rot.as_float()
        scale = b_scale.as_2wide()

        col = b_col.as_rgba()

        locrotscale = torch.cat((loc, rot, scale), dim=-1)

        res = transform(col, (col.size()[1], col.size()[2]), locrotscale, FILTERS.index(filter), EXTENSIONS.index(ext))
        b_r = BlenderData(res)
        
        return (b_r, b_r.as_out(), )
    
class BlenderScale:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "Filter": (FILTERS, ),
                "Extension": (EXTENSIONS, ),
                **COLOR_INPUT("Color", 1.0, True),
                **FLOAT_INPUT("X", 1.0, -inf, inf),
                **FLOAT_INPUT("Y", 1.0, -inf, inf),
            }
        }
    
    @classmethod
    def VALIDATE_INPUTS(self, input_types):
        return BLEND_VALID_INPUTS(input_types, self.INPUT_TYPES())
    
    RETURN_TYPES = (*BLENDER_OUTPUT(), )
    RETURN_NAMES = ("Image", "Image", )
    FUNCTION = "scale"
    CATEGORY = "Blender/Transform"
    
    def scale(self, **kwargs):
        filter = kwargs["Filter"]
        ext = kwargs["Extension"]

        b_loc = BlenderData((0.0, 0.0))
        b_rot = BlenderData(0.0)
        b_scale_x = BlenderData(kwargs, "X")
        b_scale_y = BlenderData(kwargs, "Y")
        b_col = BlenderData(kwargs, "Color")
        guess_canvas(b_col, b_loc, b_rot, b_scale_x, b_scale_y)
        
        loc = b_loc.as_2wide()
        rot = b_rot.as_float()
        scale_x = b_scale_x.as_float()
        scale_y = b_scale_y.as_float()

        col = b_col.as_rgba()

        locrotscale = torch.cat((loc, rot, scale_x, scale_y), dim=-1)

        res = transform(col, (col.size()[1], col.size()[2]), locrotscale, FILTERS.index(filter), EXTENSIONS.index(ext))
        b_r = BlenderData(res)
        
        return (b_r, b_r.as_out(), )
    
class BlenderTranslate:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "Filter": (FILTERS, ),
                "Extension": (EXTENSIONS, ),
                "Relative": ("BOOLEAN", {"default": False}),
                **COLOR_INPUT("Color", 1.0, True),
                **FLOAT_INPUT("X", 1.0, -inf, inf),
                **FLOAT_INPUT("Y", 1.0, -inf, inf),
            }
        }
    
    @classmethod
    def VALIDATE_INPUTS(self, input_types):
        return BLEND_VALID_INPUTS(input_types, self.INPUT_TYPES())
    
    RETURN_TYPES = (*BLENDER_OUTPUT(), )
    RETURN_NAMES = ("Image", "Image", )
    FUNCTION = "translate"
    CATEGORY = "Blender/Transform"
    
    def translate(self, **kwargs):
        filter = kwargs["Filter"]
        ext = kwargs["Extension"]

        b_loc_x = BlenderData(kwargs, "X")
        b_loc_y = BlenderData(kwargs, "Y")
        b_rot = BlenderData(0.0)
        b_scale = BlenderData((1.0, 1.0))
        b_col = BlenderData(kwargs, "Color")
        guess_canvas(b_col, b_loc_x, b_loc_y, b_rot, b_scale)
        
        loc_x = b_loc_x.as_float()
        loc_y = b_loc_y.as_float()
        rot = b_rot.as_float()
        scale = b_scale.as_2wide()

        col = b_col.as_rgba()

        locrotscale = torch.cat((loc_x, loc_y, rot, scale), dim=-1)

        res = transform(col, (col.size()[1], col.size()[2]), locrotscale, FILTERS.index(filter), EXTENSIONS.index(ext))
        b_r = BlenderData(res)
        
        return (b_r, b_r.as_out(), )

class BlenderTransform:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "Filter": (FILTERS, ),
                "Extension": (EXTENSIONS, ),
                **COLOR_INPUT("Color", 1.0, True),
                **FLOAT_INPUT("X", 0.0, -inf, inf),
                **FLOAT_INPUT("Y", 0.0, -inf, inf),
                **FLOAT_INPUT("Rotation", 0.0, -360.0, 360.0, 2.0),
                **FLOAT_INPUT("Scale", 1.0, -inf, inf),
            }
        }
    
    @classmethod
    def VALIDATE_INPUTS(self, input_types):
        return BLEND_VALID_INPUTS(input_types, self.INPUT_TYPES())
    
    RETURN_TYPES = (*BLENDER_OUTPUT(), )
    RETURN_NAMES = ("Image", "Image", )
    FUNCTION = "transform"
    CATEGORY = "Blender/Transform"
    
    def transform(self, **kwargs):
        filter = kwargs["Filter"]
        ext = kwargs["Extension"]

        b_loc_x = BlenderData(kwargs, "X")
        b_loc_y = BlenderData(kwargs, "Y")
        if not kwargs.get("Rotation"):
            kwargs["RotationF"] *= PI/180
        b_rot = BlenderData(kwargs, "Rotation")
        b_scale = BlenderData(kwargs, "Scale")
        b_col = BlenderData(kwargs, "Color")
        guess_canvas(b_col, b_loc_x, b_loc_y, b_rot, b_scale)
        
        loc_x = b_loc_x.as_float()
        loc_y = b_loc_y.as_float()
        rot = b_rot.as_float()
        scale = b_scale.as_2wide()

        col = b_col.as_rgba()

        locrotscale = torch.cat((loc_x, loc_y, rot, scale), dim=-1)

        res = transform(col, (col.size()[1], col.size()[2]), locrotscale, FILTERS.index(filter), EXTENSIONS.index(ext))
        b_r = BlenderData(res)
        
        return (b_r, b_r.as_out(), )
    
class BlenderMix:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "Data Type": (["Float", "Vector", "Color"], ),
                "Factor Mode": (["Uniform", "Non-Uniform"], ),
                "Blending Mode": (["Mix", "Darken", "Multiply", "Color Burn", "Lighten", "Screen", "Color Dodge", "Add", 
                                   "Overlay", "Soft Light", "Linear Light", "Difference", "Exclusion", "Subtract", "Divide",
                                   "Hue", "Saturation", "Color", "Value"], ),
                "Clamp Result": ("BOOLEAN", {"default": False}),
                "Use Alpha": ("BOOLEAN", {"default": False}),
                "Clamp Factor": ("BOOLEAN", {"default": True}),
                **FLOAT_INPUT("Factor", 0.5, 0.0, 1.0),
                **COLOR_INPUT("A", 1.0),
                **COLOR_INPUT("B", 1.0),
            }
        }
    
    @classmethod
    def VALIDATE_INPUTS(self, input_types):
        return BLEND_VALID_INPUTS(input_types, self.INPUT_TYPES())
    
    RETURN_TYPES = (*BLENDER_OUTPUT(), )
    RETURN_NAMES = ("Image", "Image", )
    FUNCTION = "mix"
    CATEGORY = "Blender/Converter"

    def mix(self, **kwargs):
        dtype = kwargs["Data Type"]
        clamp_fac = kwargs["Clamp Factor"]
        clamp_res = kwargs["Clamp Result"]

        b_fac = BlenderData(kwargs, "Factor")
        b_a = BlenderData(kwargs, "A", colortransform_if_converting=dtype=="Color", widget_override=kwargs["AR"] if dtype=="Float" else None)
        b_b = BlenderData(kwargs, "B", colortransform_if_converting=dtype=="Color", widget_override=kwargs["BR"] if dtype=="Float" else None)
        guess_canvas(b_fac, b_a, b_b)


        fac = b_fac.as_float()
        if clamp_fac:
            fac = torch.clamp(fac, torch.zeros_like(fac), torch.ones_like(fac))

        if dtype == "Float":
            a_f = b_a.as_float()
            b_f = b_b.as_float()

            res = a_f * (1.0-fac) + b_f * fac
            b_res = BlenderData(res)
        
        elif dtype == "Vector":
            a_v = b_a.as_rgb()
            b_v = b_b.as_rgb()

            res = a_v * (1.0-fac) + b_v * fac
            b_res = BlenderData(res, colortransform_force=False)

        elif dtype == "Color":
            a_color, a_alpha = b_a.as_rgb_a()
            b_color, b_alpha = b_b.as_rgb_a()

            mode = kwargs["Blending Mode"]
            # No idea how use_alpha factors in yet

            postmix = True

            # A = "background", B = "foreground"
            if mode == "Mix":
                mixed = b_color
            elif mode == "Darken":
                mixed = torch.minimum(a_color, b_color)
            elif mode == "Multiply":
                mixed = a_color * b_color
            elif mode == "Color Burn":
                mixed = 1.0 - (1.0-a_color) / b_color
            elif mode == "Lighten":
                mixed = torch.maximum(a_color, b_color)
            elif mode == "Screen":
                mixed = 1.0 - (1.0-a_color)*(1.0-b_color)
            elif mode == "Color Dodge":
                mixed = a_color / (1.0 - b_color)
            elif mode == "Add":
                mixed = a_color + b_color
            elif mode == "Overlay": # blender/source/blender/blenkernel/intern/material.cc
                less_than_half = (1.0 - fac + 2.0 * fac * b_color) * a_color
                more_than_half = 1.0 - (1.0 - fac + 2.0 * fac * (1.0 - b_color)) * (1.0 - a_color)

                mixed = tmix(less_than_half, more_than_half, a_color >= 0.5)
                postmix = False
            elif mode == "Soft Light":
                scr = 1.0 - (1.0-b_color)*(1.0-a_color)
                mixed = (1.0-a_color) * b_color * a_color + a_color * scr
            elif mode == "Linear Light":
                mixed = a_color + fac * (2.0 * (b_color-0.5))
                postmix = False
            elif mode == "Difference":
                mixed = torch.abs(a_color-b_color)
            elif mode == "Exclusion":
                mixed = a_color + b_color - (a_color*b_color*2.0)
            elif mode == "Subtract":
                mixed = a_color - b_color
            elif mode == "Divide":
                mixed = tmix(a_color / b_color, 0.0, torch.isclose(b_color, torch.zeros_like(b_color)))
            elif mode in ["Hue", "Saturation", "Color", "Value"]:
                hsv_a = rgb_to_hsv(a_color)
                hsv_b = rgb_to_hsv(b_color)

                ha, sa, va = torch.split(hsv_a, 1, dim=-1)
                hb, sb, vb = torch.split(hsv_b, 1, dim=-1)

                if mode == "Hue":
                    combo = torch.cat((hb, sa, va), dim=-1)
                elif mode == "Saturation":
                    combo = torch.cat((ha, sb, va), dim=-1)
                elif mode == "Color":
                    combo = torch.cat((hb, sb, va), dim=-1)
                elif mode == "Value":
                    combo = torch.cat((ha, sa, vb), dim=-1)
                
                mixed = hsv_to_rgb(combo)
            
            if postmix:
                mixed = tmix(a_color, mixed, fac)

            if clamp_res:
                mixed = torch.clamp(mixed, torch.zeros_like(mixed), torch.ones_like(mixed))
            
            b_res = BlenderData(mixed, a_alpha)

        return (b_res, b_res.as_out(), )
    
class BlenderMath:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "Operation": (["Add", "Subtract", "Multiply", "Divide", "Multiply Add", 
                               "Power", "Logarithm", "Square Root", "Inverse Square Root", "Absolute", "Exponent",
                               "Minimum", "Maximum", "Less Than", "Greater Than", "Sign", "Compare", "Smooth Minimum", "Smooth Maximum",
                               "Round", "Floor", "Ceil", "Truncate", 
                               "Fraction", "Truncated Modulo", "Floored Modulo", "Wrap", "Snap", "Ping-Pong", 
                               "Sine", "Cosine", "Tangent", "Arcsine", "Arccosine", "Arctangent", "Arctan2", 
                               "Hyperbolic Sine", "Hyperbolic Cosine", "Hyperbolic Tangent",
                               "To Radians", "To Degrees"], ),
                "Clamp": ("BOOLEAN", {"default": False}),
                **FLOAT_INPUT("A", 0.0, -inf, inf),
                **FLOAT_INPUT("B", 0.0, -inf, inf),
                **FLOAT_INPUT("C", 0.0, -inf, inf),
            }
        }
    
    @classmethod
    def VALIDATE_INPUTS(self, input_types):
        return BLEND_VALID_INPUTS(input_types, self.INPUT_TYPES())
    
    RETURN_TYPES = (*BLENDER_OUTPUT(), )
    RETURN_NAMES = ("Image", "Image", )
    FUNCTION = "math"
    CATEGORY = "Blender/Converter"

    def math(self, **kwargs):
        clamp = kwargs["Clamp"]
        op = kwargs["Operation"]

        b_a = BlenderData(kwargs, "A")
        b_b = BlenderData(kwargs, "B")
        b_c = BlenderData(kwargs, "C")
        guess_canvas(b_a, b_b, b_c)

        a, b, c = b_a.as_float(), b_b.as_float(), b_c.as_float()

        if op == "Add":
            res = a + b
        elif op == "Subtract":
            res = a - b
        elif op == "Multiply":
            res = a * b
        elif op == "Divide":
            res = tmix(a / b, 0.0, torch.isclose(b, torch.zeros_like(b)))
        elif op == "Multiply Add":
            res = a * b + c
        
        elif op == "Power":
            res = torch.pow(a, b)
        elif op == "Logarithm":
            res = torch.log(a) / torch.log(b)
        elif op == "Square Root":
            res = torch.sqrt(a)
        elif op == "Inverse Square Root":
            res = 1.0/torch.sqrt(a) # TODO: more precise variant?
        elif op == "Absolute":
            res = torch.abs(a)
        elif op == "Exponent":
            res = torch.exp(a)
        
        elif op == "Minimum":
            res = torch.minimum(a, b)
        elif op == "Maximum":
            res = torch.maximum(a, b)
        elif op == "Less Than":
            res = (a < b).to(torch.float32)
        elif op == "Greater Than":
            res = (a > b).to(torch.float32)
        elif op == "Sign":
            res = tmix(tmix(-1.0, 0.0, a == 0.0), 1.0, a > 0.0)
        elif op == "Compare":
            res = (torch.abs(a-b) < c).to(torch.float32)
        elif op in ["Smooth Minimum", "Smooth Maximum"]: #blender/source/blender/nodes/shader/nodes/node_shader_math.cc
            if op == "Smooth Minimum":
                sign = 1.0
            else:
                sign = -1.0
                a, b, c = -a, -b, -c
            h = (c - (a - b).abs()).maximum(torch.zeros_like(a)) / c
            if_branch = a.minimum(b) - h * h * h * c * (1.0 / 6.0)
            res = tmix(if_branch, a.minimum(b), c == 0.0) * sign
        
        elif op == "Round":
            res = torch.round(a)
        elif op == "Floor":
            res = torch.floor(a)
        elif op == "Ceil":
            res = torch.ceil(a)
        elif op == "Truncate":
            res = torch.trunc(a)
        elif op == "Fraction":
            res = a - torch.floor(a)
        elif op == "Truncated Modulo":
            res = a - torch.trunc(a / b) * b
        elif op == "Floored Modulo":
            res = a % b
        elif op == "Wrap": # blender/source/blender/nodes/shader/nodes/node_shader_math.cc -> line 312
            range = b - c
            if_branch = a - (range * ((a - c) / range).floor())
            res = tmix(if_branch, c, range == 0.0)
        elif op == "Snap":
            res = torch.floor(a / b) * b
        elif op == "Ping-Pong":
            res = torch.abs((a + b) % (b * 2) - b)
        
        elif op == "Sine":
            res = torch.sin(a)
        elif op == "Cosine":
            res = torch.cos(a)
        elif op == "Tangent":
            res = torch.tan(a)
        elif op == "Arcsine":
            res = torch.arcsin(a)
        elif op == "Arccosine":
            res = torch.arccos(a)
        elif op == "Arctangent":
            res = torch.arctan(a)
        elif op == "Arctan2":
            res = torch.arctan2(a, b)
        elif op == "Hyperbolic Sine":
            res = torch.sinh(a)
        elif op == "Hyperbolic Cosine":
            res = torch.cosh(a)
        elif op == "Hyperbolic Tangent":
            res = torch.tanh(a)
        
        elif op == "To Radians":
            res = a * (PI / 180)
        elif op == "To Degrees":
            res = a * (180 / PI)

        if clamp:
            res = torch.clamp(res, 0.0, 1.0)

        b_res = BlenderData(res)

        return (b_res, b_res.as_out(), )
