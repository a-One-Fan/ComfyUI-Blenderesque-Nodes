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

// TODO: Better names?
function __REMOVE_INPUT(obj, name){
    obj.removeInput(obj.inputs.findIndex((i) => i.name === name));
}

function __REMOVE_WIDGET(obj, name){
    let w = obj.widgets.find((i) => i.name == name);
    if(w == null){
        console.error("Could not find widget named ", name, obj);
    }
    hideWidget(obj, w);
}

function __ADD_WIDGET(obj, type, name, def, min, max, step, callback = () => {}, properties = {}){
    if(type == "FLOAT"){
        type = "number";
    }
    if(type == "BOOLEAN"){
        type = "toggle";
    }
    let wi = obj.widgets.findIndex((w) => w.name == name);
    let w = null;
    if (wi == -1){
        w = obj.addWidget(type, name, def, callback, properties, {min: min, max: max, step: step});
    }else{
        w = obj.widgets[wi];
        showWidget(w);
    }
}

function COLOR_INPUT(obj, name, def=1.0, alpha=false, step=COLSTEP, max=COLMAX, astep=0.005, hidden_default=false){
    obj.addInput(name, 0);
    if (hidden_default){
        return;
    }
    __ADD_WIDGET(obj, "FLOAT", name + "R", def, min, max, step);
    __ADD_WIDGET(obj, "FLOAT", name + "G", def, min, max, step);
    __ADD_WIDGET(obj, "FLOAT", name + "B", def, min, max, step);
    if(alpha){
        __ADD_WIDGET(obj, "FLOAT", name + "A", def);
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
    obj.addInput(name, 0);
    if (hidden_default){
        return;
    }
    __ADD_WIDGET(obj, "FLOAT", name+"F", def, min, max, step);
}

function REMOVE_FLOAT_INPUT(obj, name){
    __REMOVE_INPUT(obj, name);
    __REMOVE_WIDGET(obj, name+"F");
}

function VECTOR_INPUT(obj, name, def=0.0, step=COLSTEP, min=-inf, max=inf, hidden_default=false){
    obj.addInput(name, 0);
    if (hidden_default){
        return;
    }
    __ADD_WIDGET(obj, "FLOAT", name+"X", def, min, max, step);
    __ADD_WIDGET(obj, "FLOAT", name+"Y", def, min, max, step);
    __ADD_WIDGET(obj, "FLOAT", name+"Z", def, min, max, step);
}

function REMOVE_VECTOR_INPUT(obj, name){
    __REMOVE_INPUT(obj, name);
    __REMOVE_WIDGET(obj, name+"X");
    __REMOVE_WIDGET(obj, name+"Y");
    __REMOVE_WIDGET(obj, name+"Z");
}

const PREFIX = 'blenderesque_dynamic_inputs.'

const MAPRANGE = "BlenderMapRange";

app.registerExtension({
	name: PREFIX + MAPRANGE,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== MAPRANGE) {
            return
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = await onNodeCreated?.apply(this);

            REMOVE_FLOAT_INPUT(this, "Value");
            REMOVE_FLOAT_INPUT(this, "From Min Float");
            REMOVE_FLOAT_INPUT(this, "From Max Float");
            REMOVE_FLOAT_INPUT(this, "To Min Float");
            REMOVE_FLOAT_INPUT(this, "To Max Float");
            
            REMOVE_VECTOR_INPUT(this, "Vector");
            REMOVE_VECTOR_INPUT(this, "From Min Vector");
            REMOVE_VECTOR_INPUT(this, "From Max Vector");
            REMOVE_VECTOR_INPUT(this, "To Min Vector");
            REMOVE_VECTOR_INPUT(this, "To Max Vector");

            REMOVE_FLOAT_INPUT(this, "Steps");

            let last_widgetval_dtype = "Float";
            this.widgets.find((w) => w.name == "Data Type").callback = 
                (widgetval) => {
                    if(last_widgetval_dtype == widgetval){
                        return;
                    }
                    last_widgetval_dtype = widgetval;
                    if(widgetval == "Float"){
                        REMOVE_VECTOR_INPUT(this, "Vector");
                        REMOVE_VECTOR_INPUT(this, "From Min Vector");
                        REMOVE_VECTOR_INPUT(this, "From Max Vector");
                        REMOVE_VECTOR_INPUT(this, "To Min Vector");
                        REMOVE_VECTOR_INPUT(this, "To Max Vector");

                        FLOAT_INPUT(this, "Value");
                        FLOAT_INPUT(this, "From Min Float");
                        FLOAT_INPUT(this, "From Max Float");
                        FLOAT_INPUT(this, "To Min Float");
                        FLOAT_INPUT(this, "To Max Float");
                    }else{
                        REMOVE_FLOAT_INPUT(this, "Value");
                        REMOVE_FLOAT_INPUT(this, "From Min Float");
                        REMOVE_FLOAT_INPUT(this, "From Max Float");
                        REMOVE_FLOAT_INPUT(this, "To Min Float");
                        REMOVE_FLOAT_INPUT(this, "To Max Float");

                        VECTOR_INPUT(this, "Vector");
                        VECTOR_INPUT(this, "From Min Vector");
                        VECTOR_INPUT(this, "From Max Vector");
                        VECTOR_INPUT(this, "To Min Vector");
                        VECTOR_INPUT(this, "To Max Vector");
                    }
                }

            const interptypes = ["Linear", "Stepped Linear", "Smooth Step", "Smoother Step"];
            const interptypes_clamp = ["Linear", "Stepped Linear"];
            let last_widgetval_interptype = 0;
            this.widgets.find((w) => w.name == "Interpolation Type").callback = 
                (widgetval) => {
                    if(interptypes_clamp.includes(last_widgetval_interptype) != interptypes_clamp.includes(widgetval)){
                        if(interptypes_clamp.includes(widgetval)){
                            __ADD_WIDGET(this, "BOOLEAN", "Clamp", true);
                        }else{
                            __REMOVE_WIDGET(this, "Clamp");
                        }
                    }

                    if(last_widgetval_interptype != widgetval){
                        if(widgetval == 1){
                            __ADD_WIDGET(this, "BOOLEAN", "Clamp", true);
                        }else{
                            if(last_widgetval_interptype == 1){
                                __REMOVE_WIDGET(this, "Clamp");
                            }
                        }
                    }
                    last_widgetval_interptype = widgetval;
                }
            
            FLOAT_INPUT(this, "Value");
            FLOAT_INPUT(this, "From Min Float");
            FLOAT_INPUT(this, "From Max Float");
            FLOAT_INPUT(this, "To Min Float");
            FLOAT_INPUT(this, "To Max Float");

            return me;
        }

        return nodeType;
    },

})