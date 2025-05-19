import { CONVERTED_TYPE } from "./consts.js"

var __defProp2 = Object.defineProperty;
var __name = (target, value) => __defProp2(target, "name", { value, configurable: true });

function getWidgetStep(options2) {
    return options2.step2 || (options2.step || 10) * 0.1;
}

class BaseWidget {
    static {
      __name(this, "BaseWidget");
    }
    /** From node edge to widget edge */
    static margin = 15;
    /** From widget edge to tip of arrow button */
    static arrowMargin = 6;
    /** Arrow button width */
    static arrowWidth = 10;
    /** Absolute minimum display width of widget values */
    static minValueWidth = 42;
    linkedWidgets;
    name;
    options;
    label;
    type;
    value;
    y = 0;
    last_y;
    width;
    disabled;
    computedDisabled;
    hidden;
    advanced;
    tooltip;
    element;
    constructor(widget) {
      Object.assign(this, widget);
      this.name = widget.name;
      this.options = widget.options;
      this.type = widget.type;
      this.label = widget.label;
      this.backupLabel = widget.backupLabel;
    }
    get outline_color() {
      return this.advanced ? LiteGraph.WIDGET_ADVANCED_OUTLINE_COLOR : LiteGraph.WIDGET_OUTLINE_COLOR;
    }
    get background_color() {
      return LiteGraph.WIDGET_BGCOLOR;
    }
    get height() {
      return LiteGraph.NODE_WIDGET_HEIGHT;
    }
    get text_color() {
      return LiteGraph.WIDGET_TEXT_COLOR;
    }
    get secondary_text_color() {
      return LiteGraph.WIDGET_SECONDARY_TEXT_COLOR;
    }
    get disabledTextColor() {
      return LiteGraph.WIDGET_DISABLED_TEXT_COLOR;
    }
    /**
     * Sets the value of the widget
     * @param value The value to set
     * @param options The options for setting the value
     */
    setValue(value, { e: e2, node: node2, canvas: canvas2 }) {
        const oldValue = this.value;
        if (value === this.value) return;
        const v2 = this.type === "number_blender" ? Number(value) : value;
        this.value = v2;
        if (this.options?.property && node2.properties[this.options.property] !== void 0) {
            node2.setProperty(this.options.property, v2);
        }
        const pos = canvas2.graph_mouse;
        this.callback?.(this.value, canvas2, node2, pos, e2);
        node2.onWidgetChanged?.(this.name ?? "", v2, oldValue, this);
        if (node2.graph) node2.graph._version++;
    }
}

class BaseSteppedWidget extends BaseWidget {
    static {
      __name(this, "BaseSteppedWidget");
    }
    /**
     * Draw the arrow buttons for the widget
     * @param ctx The canvas rendering context
     * @param margin The margin of the widget
     * @param y The y position of the widget
     * @param width The width of the widget
     */
    drawArrowButtons(ctx, margin, y2, width2) {
        const { height, text_color, disabledTextColor } = this;
        const { arrowMargin, arrowWidth } = BaseWidget;
        const arrowTipX = margin + arrowMargin;
        const arrowInnerX = arrowTipX + arrowWidth;
        ctx.fillStyle = this.canDecrement() ? text_color : disabledTextColor;
        ctx.beginPath();
        ctx.moveTo(arrowInnerX, y2 + 5);
        ctx.lineTo(arrowTipX, y2 + height * 0.5);
        ctx.lineTo(arrowInnerX, y2 + height - 5);
        ctx.fill();
        ctx.fillStyle = this.canIncrement() ? text_color : disabledTextColor;
        ctx.beginPath();
        ctx.moveTo(width2 - arrowInnerX, y2 + 5);
        ctx.lineTo(width2 - arrowTipX, y2 + height * 0.5);
        ctx.lineTo(width2 - arrowInnerX, y2 + height - 5);
        ctx.fill();
    }
}

class NumberWidgetBlender extends BaseSteppedWidget {
    static {
        __name(this, "NumberWidgetBlender");
    }
    constructor(widget) {
        super(widget);
        this.type = "number_blender";
        this.value = widget.value;
    }
    canIncrement() {
        const { max } = this.options;
        return max == null || this.value < max;
    }
    canDecrement() {
        const { min } = this.options;
        return min == null || this.value > min;
    }
    incrementValue(options2) {
        this.setValue(this.value + getWidgetStep(this.options), options2);
    }
    decrementValue(options2) {
        this.setValue(this.value - getWidgetStep(this.options), options2);
    }
    setValue(value, options2) {
        let newValue2 = value;
        if (this.options.min != null && newValue2 < this.options.min) {
            newValue2 = this.options.min;
        }
        if (this.options.max != null && newValue2 > this.options.max) {
            newValue2 = this.options.max;
        }
        super.setValue(newValue2, options2);
    }
    
    /**
     * Draws the widget
     * @param ctx The canvas context
     * @param options The options for drawing the widget
     */
    drawWidget(ctx, {
        y: y2,
        width: width2,
        show_text = true,
        margin = BaseWidget.margin
    }) {
        if(this.type.startsWith(CONVERTED_TYPE)){
            return;
        }
        y2 = this.force_y || y2;
        const originalTextAlign = ctx.textAlign;
        const originalStrokeStyle = ctx.strokeStyle;
        const originalFillStyle = ctx.fillStyle;
        const { height } = this;
        ctx.textAlign = "left";
        ctx.strokeStyle = this.outline_color;
        ctx.fillStyle = this.background_color;
        ctx.beginPath();
        if (show_text){
            ctx.roundRect(margin, y2, width2 - margin * 2, height, [height * 0.5]);
        } else {
            ctx.rect(margin, y2, width2 - margin * 2, height);
        }
        ctx.fill();
        if (show_text) {
            if (!this.computedDisabled) {
                ctx.stroke();
                this.drawArrowButtons(ctx, margin, y2, width2);
            }
            ctx.fillStyle = this.secondary_text_color;
            const label = this.label || this.name;
            if (label != null) {
                ctx.fillText(label, margin * 2 + 5, y2 + height * 0.7);
            }
            ctx.fillStyle = this.text_color;
            ctx.textAlign = "right";
            ctx.fillText(
                Number(this.value).toFixed(
                    this.options.precision !== void 0 ? this.options.precision : 3
                ),
                width2 - margin * 2 - 20,
                y2 + height * 0.7
            );
        }
        ctx.textAlign = originalTextAlign;
        ctx.strokeStyle = originalStrokeStyle;
        ctx.fillStyle = originalFillStyle;
    }

    draw(ctx, node, widget_width, y2, H2, lowQuality) {
        this.drawWidget(ctx, {y: y2, width: widget_width, show_text: true, margin: BaseWidget.margin})
    }

    onClick({ e, node, canvas }) {
        const x = e.canvasX - node.pos[0];
        const width = this.width || node.size[0];
        const delta = x < 40 ? -1 : x > width - 40 ? 1 : 0;
        if (delta) {
            this.setValue(this.value + delta * getWidgetStep(this.options), { e, node, canvas });
            return;
        }
        const promptname = this.label || "Value";
        canvas.prompt(promptname, this.value, (v) => {
            if (/^[\d\s()*+/-]+|\d+\.\d+$/.test(v)) {
                try {
                    v = eval(v);
                } catch {
                }
            }
            const newValue = Number(v);
            if (!isNaN(newValue)) {
                this.setValue(newValue, { e, node, canvas });
            }
        }, e);
    }
    /**
     * Handles drag events for the number widget
     * @param options The options for handling the drag event
     */
    onDrag({ e: e2, node: node2, canvas: canvas2 }) {
        const width2 = this.width || node2.width;
        const x2 = e2.canvasX - node2.pos[0];
        const delta2 = x2 < 40 ? -1 : x2 > width2 - 40 ? 1 : 0;
        if (delta2 && (x2 > -3 && x2 < width2 + 3)){
            return;
        } 
        if (e2.deltaX){
             this.hasDragged = true;
        }
        this.setValue(this.value + (e2.deltaX ?? 0) * getWidgetStep(this.options), { e: e2, node: node2, canvas: canvas2 });
    }

    // Pressure = 0.5 -> mouse down
    // Pressure = 0 -> mouse up
    mouse(e2, co, node2){
        if(e2.pressure == 0){
            if(!this.hasDragged){
                this.onClick({e: e2, node: node2, canvas: this.currentCanvas });
            }
            this.currentCanvas = undefined;
        }else{
            this.onDrag({e: e2, node: node2, canvas: {graph_mouse: co}});
        }
    }
    onPointerDown(pointer, node2, canvas){
        this.currentCanvas = canvas;
        this.hasDragged = false;
    }
}

export { NumberWidgetBlender }