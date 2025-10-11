import { app } from "../../../scripts/app.js"

import { 
    EXTENSION_NAME,
    NODES_ALL,
    CONVERTED_TYPE, RELABEL_MAP, 
    WIDGET_SUFFIXES, SOLO_INPUT, PAIRED_INPUT, SOLO_WIDGET, INPUT_DELETED_NAME,
    HEIGHT_INPUT, HEIGHT_OUTPUT, HEIGHT_WIDGET, HEIGHT_HEADER_OFFSET, HEIGHT_BOTTOM_MARGIN,

    COLOR_FLOAT_CONNECTED, COLOR_FLOAT_DISCONNECTED, COLOR_RGB_CONNECTED, COLOR_RGB_DISCONNECTED,
    COLOR_VEC_CONNECTED, COLOR_VEC_DISCONNECTED, COLOR_BOOL_CONNECTED, COLOR_BOOL_DISCONNECTED,
    COLOR_IMAGE_CONNECTED, COLOR_IMAGE_DISCONNECTED, COLOR_DISABLED,
    COLOR_OUTPUT_GENERIC_DISCONNECTED, COLOR_OUTPUT_GENERIC_CONNECTED,

    BLENDER_OUTPUT_TYPE, BLENDER_COLOR_MAP,

    MATH_NAMEMAP,

    COLSTEP, inf, 

} from "./consts.js"

import { NumberWidgetBlender } from "./widgets.js"

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

// List of:
// - (input, SOLO_INPUT) - an input with no corresponding widget/s
// - (input, PAIRED_INPUT) - an input with corresponding widget/s that should be disregarded
// - (widget, SOLO_WIDGET) - a widget with no corresponding input/s
// - (input, [widgets], int(widget count)) - real input-widgets pair
function get_inputs_widgets_as_pairlist(node, preferred_order = []) {
    let pairs = [];
    for(let i=0; i<node.inputs.length; i++) {
        let inp = node.inputs[i];
        let found_widgets = [];
        let real_widgets = 0;
        for(let j=0; j<node.widgets.length; j++) {
            let w = node.widgets[j];
            for(let k=0; k<WIDGET_SUFFIXES.length; k++) {
                if(w.name == inp.name+WIDGET_SUFFIXES[k]) {
                    if(w.type != CONVERTED_TYPE) {
                        real_widgets++;
                    }
                    found_widgets.push(w);
                    break;
                }
            }
        }
        if(found_widgets.length > 0) {
            pairs.push([inp, found_widgets, real_widgets]);
        }else{
            if(node.widgets.find((w) => w.name == inp.name)){
                pairs.push([inp, PAIRED_INPUT]);
            }else{
                pairs.push([inp, SOLO_INPUT]);
            }
        }
    }

    let inputless_widgets = [];
    for(let i=0; i<node.widgets.length; i++) {
        if(node.widgets[i].type == CONVERTED_TYPE) {
            continue;
        }
        let found=false;
        for(let j=0; j<pairs.length && !found; j++) {
            if(pairs[j][1] != SOLO_INPUT) {
                for(let k=0; k<pairs[j][1].length; k++) {
                    if(pairs[j][1][k].name == node.widgets[i].name) {
                        found = true;
                        break;
                    }
                }
            }
        }
        if(!found) {
            inputless_widgets.push([node.widgets[i], SOLO_WIDGET]);
        }
    }

    pairs = inputless_widgets.concat(pairs);

    let ordered = [];
    for(let i=0; i<preferred_order.length; i++) {
        for(let j=0; j<pairs.length; j++) {
            if(pairs[j][0].name == preferred_order[i]) {
                ordered.push(pairs[j]);
                break;
            }
        }
    }

    ordered = ordered.concat(pairs);
    return ordered;
}

function convert_widgets(node) { // TODO: don't mix camelcase and snake case
    for(let i=0; i<node.widgets.length; i++){
        node.widgets[i] = convertWidget(node.widgets[i]);
    }
}

function rearrange_inputs_and_widgets(node, preferred_order = []) {
    let ordered = get_inputs_widgets_as_pairlist(node, preferred_order);

    for(let i=0; i<ordered.length; i++) {
        if([SOLO_INPUT, SOLO_WIDGET].includes(ordered[i][1])) {
            continue;
        }

        if(ordered[i][0].link != undefined && ordered[i][2] > 0) { // Must hide widgets
            let real_wids = [];
            for(let j=0;j<ordered[i][1].length;j++) {
                let w = ordered[i][1][j];
                if(w.type != CONVERTED_TYPE) {
                    real_wids.push(w);
                }
            }
            ordered[i][0].origVisibleWidgets = real_wids;
            for(let j=0; j<real_wids.length; j++) {
                hideWidget(node, real_wids[j]);
            }
            ordered[i][2] = 0;
        }

        if(ordered[i][0].link == undefined && ordered[i][0].origVisibleWidgets) { // Must reveal hidden widgets
            let real_wids = ordered[i][0].origVisibleWidgets;
            for(let j=0; j<real_wids.length; j++) {
                showWidget(real_wids[j]);
            }
            ordered[i][2] += ordered[i][0].origVisibleWidgets.length;
            ordered[i][0].origVisibleWidgets = undefined;
        }
    }

    let currentHeight = node.outputs.length * HEIGHT_OUTPUT + HEIGHT_HEADER_OFFSET;
    node.widgets_start_y = currentHeight;
    for(let i=0; i<ordered.length; i++) {
        if(ordered[i][1] == SOLO_WIDGET) {
            ordered[i][0].force_y = currentHeight;
            ordered[i][0].y = currentHeight;
        } else if (ordered[i][1] == PAIRED_INPUT){
            currentHeight -= HEIGHT_WIDGET;
        } else if (ordered[i][1] == SOLO_INPUT || ordered[i][2] == 0) {
            ordered[i][0].pos = [0, currentHeight + HEIGHT_INPUT];
            let desired_label = ordered[i][0].name;
            if(ordered[i][0].backupLabel) {
                desired_label = ordered[i][0].backupLabel;
                ordered[i][0].shape = 0;
                ordered[i][0].color_on = COLOR_DISABLED;
                ordered[i][0].color_off = COLOR_DISABLED;
            }else{
                ordered[i][0].shape = 0;
            }
            ordered[i][0].label = desired_label;
        } else {
            ordered[i][0].label = "   ";
            ordered[i][0].pos = [0, currentHeight + HEIGHT_INPUT];

            let wids=ordered[i][1];
            for(let j=0; j<wids.length; j++) {
                if(wids[j].type == CONVERTED_TYPE) {
                    continue;
                }
                wids[j].force_y = currentHeight;
                wids[j].y = currentHeight;
                currentHeight += HEIGHT_WIDGET;
            }
            if(ordered[i][0].color_hint){
                ordered[i][0].color_on = ordered[i][0].color_hint;
                ordered[i][0].color_off = ordered[i][0].color_hint;
            }else{
                if(wids[0].name.endsWith("R")){
                    ordered[i][0].color_on = COLOR_RGB_CONNECTED;
                    ordered[i][0].color_off = COLOR_RGB_DISCONNECTED;
                }
                if(wids[0].name.endsWith("X")){
                    ordered[i][0].color_on = COLOR_VEC_CONNECTED;
                    ordered[i][0].color_off = COLOR_VEC_DISCONNECTED;
                }
                if(wids[0].name.endsWith("F")){
                    ordered[i][0].color_on = COLOR_FLOAT_CONNECTED;
                    ordered[i][0].color_off = COLOR_FLOAT_DISCONNECTED;
                }
            }
            ordered[i][0].shape = 0;
            currentHeight -= (ordered[i][2] > 0) * HEIGHT_WIDGET;
        }
        currentHeight += HEIGHT_WIDGET;
    }
    node._size[1] = currentHeight + HEIGHT_BOTTOM_MARGIN;
}

function recalculateHeight(node) {
    let totalHeight = 2;
    for(let i=0; i<node.widgets.length; i++) {
        let size = 24;
        size *= node.widgets[i].type != CONVERTED_TYPE
        totalHeight += size;
    } 
    for(let i=0; i<node.inputs.length; i++) {
        totalHeight += 21;
    }
    node._size[1] = totalHeight;
}

function convertWidget(wid) {
    let newwid = wid;

    if (wid.type == "number"){
        newwid = new NumberWidgetBlender(wid);
    }

    return newwid;
}

// TODO: Better names?
function __REMOVE_INPUT(obj, name) {
    let ii = obj.inputs.findIndex((i) => i.name == name);
    let i = obj.inputs[ii];
    // Do something
    i.backupLabel = INPUT_DELETED_NAME;
}

function __ADD_INPUT(obj, name, type=0) {
    let ii = obj.inputs.findIndex((i) => i.name == name);
    let i = null;
    if (ii == -1) {
        i = obj.addInput(name, type);
    }else{
        i = obj.inputs[ii];
    }
    i.backupLabel = undefined;
}

function __REMOVE_WIDGET(obj, name) {
    let w = obj.widgets.find((i) => i.name == name);
    if(w == null) {
        return;
    }
    hideWidget(obj, w);
}

function __ADD_WIDGET(obj, type, name, label, def, min, max, step, callback = () => {}, properties = {}) {
    if(type == "FLOAT") {
        type = "number";
    }
    if(["BOOLEAN", "BOOL"].includes(type)) {
        type = "toggle";
    }
    let wi = obj.widgets.findIndex((w) => w.name == name);
    let w = null;
    if (wi == -1) {
        w = obj.addWidget(type, name, def, callback, properties, {min: min, max: max, step: step});
        w.label = label;
        wi = obj.widgets.findIndex((w) => w.name == name);
    }else{
        w = obj.widgets[wi];
        showWidget(w);
        w.type = type;
        w.label = label;
    }

    obj.widgets[wi] = convertWidget(obj.widgets[wi]);
}

function COLOR_INPUT(obj, name, def=1.0, alpha=false, step=COLSTEP, max=COLMAX, astep=0.005, hidden_default=false) {
    __ADD_INPUT(obj, name, 0);
    if (hidden_default) {
        return;
    }
    __ADD_WIDGET(obj, "FLOAT", name + "R", name + " Red", def, min, max, step);
    __ADD_WIDGET(obj, "FLOAT", name + "G", name + " Green", def, min, max, step);
    __ADD_WIDGET(obj, "FLOAT", name + "B", name + " Blue", def, min, max, step);
    if(alpha) {
        __ADD_WIDGET(obj, "FLOAT", name + "A", name + " Alpha", def);
    }
}

function REMOVE_COLOR_INPUT(obj, name, alpha=false) {
    __REMOVE_INPUT(obj, name);
    __REMOVE_WIDGET(obj, name+"R");
    __REMOVE_WIDGET(obj, name+"G");
    __REMOVE_WIDGET(obj, name+"B");
    if(alpha) {
        __REMOVE_WIDGET(obj, name+"A");
    }
}

function FLOAT_INPUT(obj, name, def=0.0, min=0.0, max=1.0, step=COLSTEP, hidden_default=false) {
    __ADD_INPUT(obj, name, 0);
    if (hidden_default) {
        return;
    }
    __ADD_WIDGET(obj, "FLOAT", name+"F", name, def, min, max, step);
}

function REMOVE_FLOAT_INPUT(obj, name) {
    __REMOVE_INPUT(obj, name);
    __REMOVE_WIDGET(obj, name+"F");
}

function VECTOR_INPUT(obj, name, def=0.0, step=COLSTEP, min=-inf, max=inf, hidden_default=false) {
    __ADD_INPUT(obj, name, 0);
    if (hidden_default) {
        return;
    }
    __ADD_WIDGET(obj, "FLOAT", name+"X", name + " X", def, min, max, step);
    __ADD_WIDGET(obj, "FLOAT", name+"Y", name + " Y", def, min, max, step);
    __ADD_WIDGET(obj, "FLOAT", name+"Z", name + " Z", def, min, max, step);
}

function REMOVE_VECTOR_INPUT(obj, name) {
    __REMOVE_INPUT(obj, name);
    __REMOVE_WIDGET(obj, name+"X");
    __REMOVE_WIDGET(obj, name+"Y");
    __REMOVE_WIDGET(obj, name+"Z");
}

function relabel_widgets(node) {
    for(let i=0; i<node.widgets.length; i++) {
        let w = node.widgets[i];

        let last = w.name[w.name.length-1];
        let mapped = RELABEL_MAP[last];
        if(mapped != undefined) {
            w.label = w.name.substr(0, w.name.length-1) + mapped;
        }
    }
}

function get_color_by_type(type) {
    if (!type.startsWith(BLENDER_OUTPUT_TYPE)){
        return false;
    }
    const type_rest = type.substr(BLENDER_OUTPUT_TYPE.length+1);
    const cols = BLENDER_COLOR_MAP[type_rest];

    if (cols){
        return cols;
    }

    return [COLOR_OUTPUT_GENERIC_CONNECTED, COLOR_OUTPUT_GENERIC_DISCONNECTED];
}

function recolor_outputs(node) {
    for(let i=0; i<node.outputs.length; i++){
        const o = node.outputs[i];
        const cols = get_color_by_type(o.type);
        if (cols){
            o.color_on = cols[0];
            o.color_off = cols[1];
        }
    }
}

function find_blender_node(name) {
    for(let i=0; i<NODES_ALL.length; i++) {
        for(let j=0; j<NODES_ALL[i].nodes.length; j++) {
            if (name == NODES_ALL[i].nodes[j]) {
                return [i, j];
            }
        }
    }

    return null;
}

function blender_node_color(name) {
    let indices = find_blender_node(name);
    return NODES_ALL[indices[0]].color;
}

function get_vec_widgets(node, name) {
    let wx = node.widgets.find((w) => w.name == (name+"X"));
    let wy = node.widgets.find((w) => w.name == (name+"Y"));
    let wz = node.widgets.find((w) => w.name == (name+"Z"));
    return [wx, wy, wz];
}

function register_map_range(nodeType, nodeData) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = async function () {
        const me = await onNodeCreated?.apply(this);

        this.widgets.find((w) => w.name == "Data Type").callback = 
            (widgetval) => {
                let iv = this.inputs.find((i) => i.name == "Value");

                let ifmin = this.inputs.find((i) => i.name == "From Min");
                let ifmax = this.inputs.find((i) => i.name == "From Max");
                let itmin = this.inputs.find((i) => i.name == "To Min");
                let itmax = this.inputs.find((i) => i.name == "To Max");

                let all_ins = [iv, ifmin, ifmax, itmin, itmax];

                let wfmin = get_vec_widgets(this, "From Min");
                let wfmax = get_vec_widgets(this, "From Max");
                let wtmin = get_vec_widgets(this, "To Min");
                let wtmax = get_vec_widgets(this, "To Max");

                const SUFFIXES = [" X", " Y", " Z"];

                let all_wids = [wfmin, wfmax, wtmin, wtmax];

                if(widgetval == "Float") {
                    for(let i=0; i<all_wids.length; i++) {
                        for(let j=1; j<3; j++) {
                            __REMOVE_WIDGET(this, all_wids[i][j].name);
                        }
                        all_wids[i][0].label = all_wids[i][0].name.substr(0, all_wids[i][0].name.length-1);
                    }

                    __ADD_WIDGET(this, "FLOAT", "ValueX", "Value");
                    __REMOVE_WIDGET(this, "ValueY");
                    __REMOVE_WIDGET(this, "ValueZ");
                    iv.label = "Value";

                    for(let i=0; i<all_ins.length; i++) {
                        all_ins[i].color_hint = COLOR_FLOAT_CONNECTED;
                    }
                }else{
                    for(let i=0; i<all_wids.length; i++) {
                        for(let j=0; j<3; j++) {
                            let name = all_wids[i][j].name;
                            __ADD_WIDGET(this, "FLOAT", name, name.substr(0, name.length-1) + SUFFIXES[j]);
                        }
                    }

                    __ADD_WIDGET(this, "FLOAT", "ValueX", "Value X");
                    __ADD_WIDGET(this, "FLOAT", "ValueY", "Value Y");
                    __ADD_WIDGET(this, "FLOAT", "ValueZ", "Value Z");
                    iv.label = "Vector";

                    for(let i=0; i<all_ins.length; i++) {
                        all_ins[i].color_hint = COLOR_VEC_CONNECTED;
                    }
                }
                this.graph.setDirtyCanvas(true);
                rearrange_inputs_and_widgets(this);
            }
        
        let w = this.widgets.find((w) => w.name == "Data Type");
        w.callback(w.value);

        this.widgets.find((w) => w.name == "Interpolation Type").callback = 
            (widgetval) => {
                if(["Linear", "Stepped Linear"].includes(widgetval)) {
                    __ADD_WIDGET(this, "BOOLEAN", "Clamp", "Clamp", true);
                }else{
                    __REMOVE_WIDGET(this, "Clamp");
                }

                if(widgetval == "Stepped Linear") {
                    FLOAT_INPUT(this, "Steps", 4.0, 0.0);
                }else{
                    REMOVE_FLOAT_INPUT(this, "Steps");
                }

                this.graph.setDirtyCanvas(true);
                rearrange_inputs_and_widgets(this);
            }
            
        w = this.widgets.find((w) => w.name == "Interpolation Type");
        w.callback(w.value);

        return me;
    }
}

function register_mix(nodeType, nodeData) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = async function () {
        const me = await onNodeCreated?.apply(this);

        let w_dtype = this.widgets.find((w) => w.name == "Data Type");
        let w_blendmode = this.widgets.find((w) => w.name == "Blending Mode");

        w_blendmode.callback = (widgetval) => {
            this.title = widgetval;
        }

        if (w_dtype == "Color") {
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

                let ia = this.inputs.find((i) => i.name == "A");
                let ib = this.inputs.find((i) => i.name == "B");

                if(widgetval == "Float") {
                    __REMOVE_WIDGET(this, "AG");
                    __REMOVE_WIDGET(this, "AB");

                    __REMOVE_WIDGET(this, "BG");
                    __REMOVE_WIDGET(this, "BB");

                    war.label = "A";
                    wbr.label = "B";
                    
                    if(ia.origVisibleWidgets) {
                        ia.origVisibleWidgets = [war]
                    }
                    if(ib.origVisibleWidgets) {
                        ib.origVisibleWidgets = [wbr]
                    }
                }else{
                    if(ia.link) {
                        ia.origVisibleWidgets = [war, wag, wab]
                    }else{
                        __ADD_WIDGET(this, "FLOAT", "AG");
                        __ADD_WIDGET(this, "FLOAT", "AB");
                    }
                    if(ib.link) {
                        ib.origVisibleWidgets = [wbr, wbg, wbb]
                    }else{
                        __ADD_WIDGET(this, "FLOAT", "BG");
                        __ADD_WIDGET(this, "FLOAT", "BB");
                    }

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

                if(widgetval == "Vector") { // TODO Non-uniform factor

                }
                
                this.graph.setDirtyCanvas(true);
                rearrange_inputs_and_widgets(this);
                relabel_widgets(this);
            }
        
        w_dtype.callback(w_dtype.value);

        return me;
    }
}

function register_math(nodeType, nodeData) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = async function () {
        const me = await onNodeCreated?.apply(this);

        let w_operation = this.widgets.find((w) => w.name == "Operation");

        w_operation.callback = (widgetval) => {
            this.title = widgetval;
            let ins = MATH_NAMEMAP[widgetval];
            if(!ins) {
                ins = ["Value", "Value"];
            }
            let wa = this.widgets.find((w) => w.name == "AF");
            let wb = this.widgets.find((w) => w.name == "BF");
            let wc = this.widgets.find((w) => w.name == "CF");
            let wabc = [wa, wb, wc];
            if(ins.length == 1) {
                REMOVE_FLOAT_INPUT(this, "B");
                REMOVE_FLOAT_INPUT(this, "C");
            }
            if(ins.length == 2) {
                FLOAT_INPUT(this, "B");
                REMOVE_FLOAT_INPUT(this, "C");
            }
            if(ins.length == 3) {
                FLOAT_INPUT(this, "B");
                FLOAT_INPUT(this, "C");
            }

            for(let i=0; i<ins.length; i++) {
                wabc[i].label = ins[i];
            }

            this.graph.setDirtyCanvas(true);
            rearrange_inputs_and_widgets(this);
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

        if (find_blender_node(nodeData.name) != null) {
            const onNodeCreatedPrev = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = async function () {
                const me = await onNodeCreatedPrev?.apply(this);
                this.color = blender_node_color(nodeData.name);
                this.bgcolor = "#303030";
                this.boxcolor = "#DDDDDD";
                this.constructor.title_text_color = "#E8E8E8";
                convert_widgets(this);
                relabel_widgets(this);
                rearrange_inputs_and_widgets(this);
                recolor_outputs(this);

                const onConnectionsChangePrev = this.onConnectionsChange;
                this.onConnectionsChange = async function (eventtype, slotid, is_connect, link_info, input) {
                    if(onConnectionsChangePrev) {
                        await onConnectionsChangePrev(eventtype, slotid, is_connect, link_info, input);
                    }
                    rearrange_inputs_and_widgets(this);
                }
                return me;
            }
        }

        let reg = REGISTER_MAP[nodeData.name];
        if(reg) {
            reg(nodeType, nodeData);
        }

        return nodeType;
    },

})