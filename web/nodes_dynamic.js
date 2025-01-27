import { app } from "../../../scripts/app.js"

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

// TODO: Import these?
function hideWidget(node, widget, suffix = "") {
    if (widget.type?.startsWith(CONVERTED_TYPE)) return;
    widget.origType = widget.type;
    widget.origComputeSize = widget.computeSize;
    widget.origSerializeValue = widget.serializeValue;
    widget.computeSize = () => [0, -4];
    widget.type = CONVERTED_TYPE + suffix;
    widget.serializeValue = () => {
      if (!node.inputs) {
        return void 0;
      }
      let node_input = node.inputs.find((i) => i.widget?.name === widget.name);
      if (!node_input || !node_input.link) {
        return void 0;
      }
      return widget.origSerializeValue ? widget.origSerializeValue() : widget.value;
    };
    if (widget.linkedWidgets) {
      for (const w of widget.linkedWidgets) {
        hideWidget(node, w, ":" + widget.name);
      }
    }
}

function showWidget(widget) {
    widget.type = widget.origType;
    widget.computeSize = widget.origComputeSize;
    widget.serializeValue = widget.origSerializeValue;
    delete widget.origType;
    delete widget.origComputeSize;
    delete widget.origSerializeValue;
    if (widget.linkedWidgets) {
        for (const w of widget.linkedWidgets) {
            showWidget(w);
        }
    }
}

function recalculateHeight(node) {
    let totalHeight = 2;
    for(let i=0; i<node.widgets.length; i++){
        let size = 24;
        size *= node.widgets[i].type != CONVERTED_TYPE
        totalHeight += size;
    } 
    for(let i=0; i<node.inputs.length; i++){
        totalHeight += 21;
    }
    node._size[1] = totalHeight;
}

// TODO: Better names?
function __REMOVE_INPUT(obj, name){
    let ii = obj.inputs.findIndex((i) => i.name == name);
    let i = obj.inputs[ii];
    // Do something
}

function __ADD_INPUT(obj, name, type=0){
    let ii = obj.inputs.findIndex((i) => i.name == name);
    let i = null;
    if (ii == -1){
        i = obj.addInput(name, type);
    }else{
        i = obj.inputs[ii];
    }
}

function __REMOVE_WIDGET(obj, name){
    let w = obj.widgets.find((i) => i.name == name);
    if(w == null){
        return;
    }
    hideWidget(obj, w);
}

function __ADD_WIDGET(obj, type, name, label, def, min, max, step, callback = () => {}, properties = {}){
    if(type == "FLOAT"){
        type = "number";
    }
    if(["BOOLEAN", "BOOL"].includes(type)){
        type = "toggle";
    }
    let wi = obj.widgets.findIndex((w) => w.name == name);
    let w = null;
    if (wi == -1){
        w = obj.addWidget(type, name, def, callback, properties, {min: min, max: max, step: step});
        w.label = label;
    }else{
        w = obj.widgets[wi];
        showWidget(w);
        w.type = type;
        w.label = label;
    }
}

function COLOR_INPUT(obj, name, def=1.0, alpha=false, step=COLSTEP, max=COLMAX, astep=0.005, hidden_default=false){
    __ADD_INPUT(obj, name, 0);
    if (hidden_default){
        return;
    }
    __ADD_WIDGET(obj, "FLOAT", name + "R", name + " Red", def, min, max, step);
    __ADD_WIDGET(obj, "FLOAT", name + "G", name + " Green", def, min, max, step);
    __ADD_WIDGET(obj, "FLOAT", name + "B", name + " Blue", def, min, max, step);
    if(alpha){
        __ADD_WIDGET(obj, "FLOAT", name + "A", name + " Alpha", def);
    }
}

function REMOVE_COLOR_INPUT(obj, name, alpha=false){
    __REMOVE_INPUT(obj, name);
    __REMOVE_WIDGET(obj, name+"R");
    __REMOVE_WIDGET(obj, name+"G");
    __REMOVE_WIDGET(obj, name+"B");
    if(alpha){
        __REMOVE_WIDGET(obj, name+"A");
    }
}

function FLOAT_INPUT(obj, name, def=0.0, min=0.0, max=1.0, step=COLSTEP, hidden_default=false){
    __ADD_INPUT(obj, name, 0);
    if (hidden_default){
        return;
    }
    __ADD_WIDGET(obj, "FLOAT", name+"F", name, def, min, max, step);
}

function REMOVE_FLOAT_INPUT(obj, name){
    __REMOVE_INPUT(obj, name);
    __REMOVE_WIDGET(obj, name+"F");
}

function VECTOR_INPUT(obj, name, def=0.0, step=COLSTEP, min=-inf, max=inf, hidden_default=false){
    __ADD_INPUT(obj, name, 0);
    if (hidden_default){
        return;
    }
    __ADD_WIDGET(obj, "FLOAT", name+"X", name + " X", def, min, max, step);
    __ADD_WIDGET(obj, "FLOAT", name+"Y", name + " Y", def, min, max, step);
    __ADD_WIDGET(obj, "FLOAT", name+"Z", name + " Z", def, min, max, step);
}

function REMOVE_VECTOR_INPUT(obj, name){
    __REMOVE_INPUT(obj, name);
    __REMOVE_WIDGET(obj, name+"X");
    __REMOVE_WIDGET(obj, name+"Y");
    __REMOVE_WIDGET(obj, name+"Z");
}

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

function relabel_widgets(node){
    for(let i=0; i<node.widgets.length; i++){
        let w = node.widgets[i];

        let last = w.name[w.name.length-1];
        let mapped = RELABEL_MAP[last];
        if(mapped != undefined){
            w.label = w.name.substr(0, w.name.length-1) + mapped;
        }
    }
}

const EXTENSION_NAME = "blenderesque_dynamic";

const MAPRANGE = "BlenderMapRange";

const NODES_INPUT = ["BlenderValue", "BlenderRGB"];
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

function find_blender_node(name){
    for(let i=0; i<NODES_ALL.length; i++){
        for(let j=0; j<NODES_ALL[i].nodes.length; j++){
            if (name == NODES_ALL[i].nodes[j]){
                return [i, j];
            }
        }
    }

    return null;
}

function blender_node_color(name){
    let indices = find_blender_node(name);
    return NODES_ALL[indices[0]].color;
}

function get_vec_widgets(node, name){
    let wx = node.widgets.find((w) => w.name == (name+"X"));
    let wy = node.widgets.find((w) => w.name == (name+"Y"));
    let wz = node.widgets.find((w) => w.name == (name+"Z"));
    return [wx, wy, wz];
}

function register_map_range(nodeType, nodeData){
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = async function () {
        const me = await onNodeCreated?.apply(this);

        this.widgets.find((w) => w.name == "Data Type").callback = 
            (widgetval) => {
                let iv = this.inputs.find((i) => i.name == "Value");

                let wfmin = get_vec_widgets(this, "From Min");
                let wfmax = get_vec_widgets(this, "From Max");
                let wtmin = get_vec_widgets(this, "To Min");
                let wtmax = get_vec_widgets(this, "To Max");

                const SUFFIXES = [" X", " Y", " Z"];

                let all_wids = [wfmin, wfmax, wtmin, wtmax];

                if(widgetval == "Float"){
                    for(let i=0; i<all_wids.length; i++){
                        for(let j=1; j<3; j++){
                            __REMOVE_WIDGET(this, all_wids[i][j].name);
                        }
                        all_wids[i][0].label = all_wids[i][0].name.substr(0, all_wids[i][0].name.length-1);
                    }

                    __ADD_WIDGET(this, "FLOAT", "ValueX", "Value");
                    __REMOVE_WIDGET(this, "ValueY");
                    __REMOVE_WIDGET(this, "ValueZ");
                    iv.label = "Value";
                }else{
                    for(let i=0; i<all_wids.length; i++){
                        for(let j=0; j<3; j++){
                            let name = all_wids[i][j].name;
                            __ADD_WIDGET(this, "FLOAT", name, name.substr(0, name.length-1) + SUFFIXES[j]);
                        }
                    }

                    __ADD_WIDGET(this, "FLOAT", "ValueX", "Value X");
                    __ADD_WIDGET(this, "FLOAT", "ValueY", "Value Y");
                    __ADD_WIDGET(this, "FLOAT", "ValueZ", "Value Z");
                    iv.label = "Vector";
                }
                this.graph.setDirtyCanvas(true);
                recalculateHeight(this);
            }
        
        let w = this.widgets.find((w) => w.name == "Data Type");
        w.callback(w.value);

        this.widgets.find((w) => w.name == "Interpolation Type").callback = 
            (widgetval) => {
                if(["Linear", "Stepped Linear"].includes(widgetval)){
                    __ADD_WIDGET(this, "BOOLEAN", "Clamp", "Clamp", true);
                }else{
                    __REMOVE_WIDGET(this, "Clamp");
                }

                if(widgetval == "Stepped Linear"){
                    FLOAT_INPUT(this, "Steps", 4.0, 0.0);
                }else{
                    REMOVE_FLOAT_INPUT(this, "Steps");
                }

                this.graph.setDirtyCanvas(true);
                recalculateHeight(this);
            }
            
        w = this.widgets.find((w) => w.name == "Interpolation Type");
        w.callback(w.value);

        return me;
    }
}

function register_mix(nodeType, nodeData){
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = async function () {
        const me = await onNodeCreated?.apply(this);

        let w_dtype = this.widgets.find((w) => w.name == "Data Type");
        let w_blendmode = this.widgets.find((w) => w.name == "Blending Mode");

        w_blendmode.callback = (widgetval) => {
            this.title = widgetval;
        }

        if (w_dtype == "Color"){
            w_blendmode.callback(w_blendmode.value);
        }

        w_dtype.callback = 
            (widgetval) => {
                if(widgetval == "Float") {
                    __REMOVE_WIDGET(this, "Factor Mode");
                    __REMOVE_WIDGET(this, "Clamp Result");
                    __REMOVE_WIDGET(this, "Blending Mode");
                    this.color = NODES_ALL[2].color; // Converter color
                    this.title = "Mix";
                } else if (widgetval == "Vector") {
                    __ADD_WIDGET(this, "combo", "Factor Mode");
                    __REMOVE_WIDGET(this, "Clamp Result");
                    __REMOVE_WIDGET(this, "Blending Mode");
                    this.color = NODES_ALL[6].color; // Vector color
                    this.title = "Mix";
                } else if (widgetval == "Color") {
                    __ADD_WIDGET(this, "combo", "Blending Mode");
                    __ADD_WIDGET(this, "BOOLEAN", "Clamp Result");
                    __REMOVE_WIDGET(this, "Factor Mode");
                    this.color = NODES_ALL[1].color; // Color color
                    this.title = w_blendmode.value;
                }

                let war = this.widgets.find((w) => w.name == "AR");
                let wag = this.widgets.find((w) => w.name == "AG");
                let wab = this.widgets.find((w) => w.name == "AB");

                let wbr = this.widgets.find((w) => w.name == "BR");
                let wbg = this.widgets.find((w) => w.name == "BG");
                let wbb = this.widgets.find((w) => w.name == "BB");

                if(widgetval == "Float") {
                    __REMOVE_WIDGET(this, "AG");
                    __REMOVE_WIDGET(this, "AB");

                    __REMOVE_WIDGET(this, "BG");
                    __REMOVE_WIDGET(this, "BB");

                    war.label = "A";
                    wbr.label = "B";
                }else{
                    __ADD_WIDGET(this, "FLOAT", "AG");
                    __ADD_WIDGET(this, "FLOAT", "AB");

                    __ADD_WIDGET(this, "FLOAT", "BG");
                    __ADD_WIDGET(this, "FLOAT", "BB");

                    if(widgetval == "Vector") {
                        war.label = "A X";
                        wag.label = "A Y";
                        wab.label = "A Z";

                        wbr.label = "B X";
                        wbg.label = "B Y";
                        wbb.label = "B Z";
                    }else{
                        war.label = "A Red";
                        wag.label = "A Green";
                        wab.label = "A Blue";

                        wbr.label = "B Red";
                        wbg.label = "B Green";
                        wbb.label = "B Blue";
                    }
                }

                if(widgetval == "Vector") {

                }
                
                this.graph.setDirtyCanvas(true);
                recalculateHeight(this);
            }
        
        w_dtype.callback(w_dtype.value);

        return me;
    }
}

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

function register_math(nodeType, nodeData){
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = async function () {
        const me = await onNodeCreated?.apply(this);

        let w_operation = this.widgets.find((w) => w.name == "Operation");

        w_operation.callback = (widgetval) => {
            this.title = widgetval;
            let ins = MATH_NAMEMAP[widgetval];
            if(!ins){
                ins = ["Value", "Value"];
            }
            let wa = this.widgets.find((w) => w.name == "AF");
            let wb = this.widgets.find((w) => w.name == "BF");
            let wc = this.widgets.find((w) => w.name == "CF");
            let wabc = [wa, wb, wc];
            if(ins.length == 1){
                REMOVE_FLOAT_INPUT(this, "B");
                REMOVE_FLOAT_INPUT(this, "C");
            }
            if(ins.length == 2){
                FLOAT_INPUT(this, "B");
                REMOVE_FLOAT_INPUT(this, "C");
            }
            if(ins.length == 3){
                FLOAT_INPUT(this, "B");
                FLOAT_INPUT(this, "C");
            }

            for(let i=0; i<ins.length; i++){
                wabc[i].label = ins[i];
            }

            this.graph.setDirtyCanvas(true);
            recalculateHeight(this);
        }
        
        w_operation.callback(w_operation.value);

        return me;
    }
}

const REGISTER_MAP = {
    "BlenderMapRange": register_map_range,
    "BlenderMix": register_mix,
    "BlenderMath": register_math,
}

app.registerExtension({
	name: EXTENSION_NAME,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {

        if (find_blender_node(nodeData.name) != null){
            const onNodeCreatedPrev = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = async function () {
                const me = await onNodeCreatedPrev?.apply(this);
                this.color = blender_node_color(nodeData.name);
                this.bgcolor = "#303030";
                this.boxcolor = "#DDDDDD";
                this.constructor.title_text_color = "#E8E8E8";
                relabel_widgets(this);
                return me;
            }
        }

        let reg = REGISTER_MAP[nodeData.name];
        if(reg){
            reg(nodeType, nodeData);
        }

        return nodeType;
    },

})