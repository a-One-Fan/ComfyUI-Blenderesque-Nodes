const EXTENSION_NAME = "blenderesque_dynamic";

const TypeSlot = {
    Input: 1,
    Output: 2,
};

const TypeSlotEvent = {
    Connect: true,
    Disconnect: false,
};

const COLSTEP = 0.005;
const inf = 100000000;

const CONVERTED_TYPE = "converted-widget";
const GET_CONFIG = Symbol();

const COLOR_RGB_CONNECTED = "#C7C729"
const COLOR_RGB_DISCONNECTED = COLOR_RGB_CONNECTED
const COLOR_VEC_CONNECTED = "#6363C7"
const COLOR_VEC_DISCONNECTED = COLOR_VEC_CONNECTED
const COLOR_FLOAT_CONNECTED = "#A1A1A1"
const COLOR_FLOAT_DISCONNECTED = COLOR_FLOAT_CONNECTED
const COLOR_BOOL_CONNECTED = "#CCA6D6"
const COLOR_BOOL_DISCONNECTED = COLOR_BOOL_CONNECTED
const COLOR_IMAGE_CONNECTED = "#633863"
const COLOR_IMAGE_DISCONNECTED = COLOR_IMAGE_CONNECTED

const COLOR_DISABLED = "#00000088"
const COLOR_FACTOR = "#4772B3FF"
const COLOR_FACTOR_DARK = "#20203BFF"
const COLOR_TEXT = "#FFFFFFFF"
const COLOR_WIDGET = "#545454FF"
const COLOR_OUTLINE = "#3D3D3DFF"

const HEIGHT_OUTPUT = 24;
const HEIGHT_HEADER_OFFSET = 6;
const HEIGHT_INPUT = 10.5;
const HEIGHT_WIDGET = 24;

const WIDGET_SUFFIXES = ["F", "R", "G", "B", "A", "X", "Y", "Z"];
const SOLO_WIDGET = "solo_widget";
const SOLO_INPUT = "solo_input";
const INPUT_DELETED_NAME = "X";

const RELABEL_MAP = {
    "F": "",
    "R": " Red",
    "G": " Green",
    "B": " Blue",
    "A": " Alpha",
    "X": " X",
    "Y": " Y",
    "Z": " Z",
}

const NODES_INPUT = ["BlenderValue", "BlenderRGB", "BlenderUV"];
const NODES_COLOR = ["BlenderBrightnessContrast", "BlenderGamma", "BlenderHueSaturationValue", "BlenderInvertColor", 
    "BlenderExposure", "BlenderTonemap", "BlenderAlphaOver", "BlenderZCombine", "BlenderAlphaConvert", "BlenderConvertColorspace",
    "BlenderSetAlpha",
];
const NODES_CONVERTER = ["BlenderBlackbody", "BlenderClamp", "BlenderCombineColor", "BlenderCombineXYZ", "BlenderMapRange", 
    "BlenderMath", "BlenderMix", "BlenderRGBtoBW", "BlenderSeparateColor", "BlenderSeparateXYZ", "BlenderVectorMath", "BlenderWavelength",
];
const NODES_TRANSFORM = ["BlenderRotate", "BlenderScale", "BlenderTransform", "BlenderTranslate", "BlenderCornerPin", "BlenderCrop",
    "BlenderDisplace", "BlenderFlip", "BlenderMapUV", "BlenderLensDistortion", "BlenderMovieDistortion",
];
const NODES_TEXTURE = ["BlenderBrickTexture", "BlenderCheckerTexture", "BlenderGaborTexture", "BlenderGradientTexture", "BlenderMagicTexture",
    "BlenderNoiseTexture", "BlenderVoronoiTexture", "BlenderWaveTexture", "BlenderWhiteNoiseTexture",
];
const NODES_FILTER = ["BlenderBlur", "BlenderAntiAliasing", "BlenderDespeckle", "BlenderDilateErode", "BlenderFilter", "BlenderGlare", 
    "BlenderKuwahara", "BlenderPixelate", "BlenderPosterize", "BlenderSunBeams"
];

const NODES_ALL = [
    {nodes: NODES_INPUT, color: "#83314A"}, // 0
    {nodes: NODES_COLOR, color: "#6E6E1D"}, // 1
    {nodes: NODES_CONVERTER, color: "#246283"}, // 2
    {nodes: NODES_TRANSFORM, color: "#3B5959"}, // 3
    {nodes: NODES_TEXTURE, color: "#79461D"}, // 4
    {nodes: NODES_FILTER, color: "#3F2750"}, // 5
    {nodes: 0, color: "#3C3C83"}, // 6 Vector nodes?
];

const MATH_NAMEMAP = {
    "Multiply Add": ["Value", "Multiplier", "Addend"],
    "Power": ["Base", "Exponent"],
    "Logarithm": ["Value", "Base"],
    "Square Root": ["Value"],
    "Inverse Square Root": ["Value"],
    "Absolute": ["Value"],
    "Exponent": ["Value"],
    "Less Than": ["Value", "Threshold"],
    "Greater Than": ["Value", "Threshold"],
    "Sign": ["Value"],
    "Compare": ["Value", "Value", "Epsilon"],
    "Smooth Minimum": ["Value", "Value", "Distance"],
    "Smooth Maximum": ["Value", "Value", "Distance"],
    "Round": ["Value"],
    "Floor": ["Value"],
    "Ceil": ["Value"],
    "Truncate": ["Value"],
    "Fraction": ["Value"],
    "Wrap": ["Value", "Min", "Max"],
    "Snap": ["Value", "Increment"],
    "Ping-Pong": ["Value", "Scale"],
    "Sine": ["Value"],
    "Cosine": ["Value"],
    "Tangent": ["Value"],
    "Arcsine": ["Value"],
    "Arccosine": ["Value"],
    "Arctangent": ["Value"],
    "Hyperbolic Sine": ["Value"],
    "Hyperbolic Cosine": ["Value"],
    "Hyperbolic Tangent": ["Value"],
    "To Radians": ["Degrees"],
    "To Degrees": ["Radians"],
}

export {

    EXTENSION_NAME,


    TypeSlot, TypeSlotEvent, COLSTEP, inf, CONVERTED_TYPE, GET_CONFIG, 

    COLOR_RGB_CONNECTED, COLOR_RGB_DISCONNECTED, COLOR_VEC_CONNECTED, COLOR_VEC_DISCONNECTED, 
    COLOR_FLOAT_CONNECTED, COLOR_FLOAT_DISCONNECTED, COLOR_BOOL_CONNECTED, COLOR_BOOL_DISCONNECTED,
    COLOR_IMAGE_CONNECTED, COLOR_IMAGE_DISCONNECTED,
    COLOR_DISABLED, COLOR_FACTOR, COLOR_FACTOR_DARK, COLOR_TEXT, COLOR_WIDGET, COLOR_OUTLINE,

    HEIGHT_INPUT, HEIGHT_OUTPUT, HEIGHT_HEADER_OFFSET, HEIGHT_WIDGET,

    WIDGET_SUFFIXES, SOLO_INPUT, SOLO_WIDGET, INPUT_DELETED_NAME,

    RELABEL_MAP,

    NODES_ALL, MATH_NAMEMAP
}