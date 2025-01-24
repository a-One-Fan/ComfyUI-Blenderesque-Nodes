# ComfyUI-Blenderesque-Nodes
Blender-like nodes for ComfyUI.<br>

<h1>Work In Progress<br>
Nodes might change and you will need to redo your workflow</h1>

<img src=mary_combo.png style="width:50%;height:50%">

Install:
```
cd ./custom_nodes
git clone
pip install -r ./ComfyUI-Blendersque-Nodes/requirements.txt
```

Usage:<br><br>
Nodes will accept most inputs, akin to Blender's implicit conversion between vectors, floats, colors, and image types thereof.<br>
Nodes will output Blender-like data, and an image for general Comfy usage.<br>
Differently sized images are automatically cropped/padded like Blender's compositor.
You should prefer to use the Blender-like data output between these nodes, to reduce loss in precision from excessive colorspace transforms, or avoid very small images inteded for a simple preview, among other possible issues.<br>

Currently implemented nodes:<br>
<span style="color:LightGreen">Green with ✔</span> = Implemented<br>
<span style="color:IndianRed">Red with ❌</span> = Not implemented<br>
<span style="color:DimGrey">Dark grey with -</span> = Probably won't be implemented?<br>
<SPAN STYLE="color:GoldenRod">Yellow with O</span> = Partially implemented, useable<br>
Hopefully I'll figure out a way to do colorramps at some point.<br>
Blender lacks some compositor-applicable shader nodes, so this list will also include some shader nodes.<br>
Currently the list is incomplete.<br>

<details>
<summary>Meta/Bugs/TODO</summary>
<ul>
<li><span style="color:IndianRed">Dynamic inputs ❌</span></li>
<li><span style="color:GoldenRod">Dynamic widgets O</span></li>
<li><span style="color:IndianRed">UV Input Node ❌ (For mapping textures)</span></li>
<li><span style="color:IndianRed">Resize Canvas Node ❌</span></li>
<li><span style="color:IndianRed">Extract Data Node ❌ (Get image, mask, canvas xy, float, etc.)</span></li>
<li><span style="color:IndianRed">Merged input sockets and default values ("widgets") ❌</span></li>
<li><span style="color:IndianRed">Merged output blender and image sockets ❌</span></li>
<li><span style="color:IndianRed">Low precision on image transforms, teethy edges ❌</span></li>
</ul>
</details>

<details>
<summary>Input</summary>
<ul>
<li><span style="color:LightGreen">Value ✔</span></li>
<li><span style="color:LightGreen">RGB ✔</span></li>
<li><span style="color:IndianRed">Bokeh Image ❌</span></li>
Most other input nodes seem redundant or not applicable.
</ul>
</details>

<details>
<summary>Color</summary>
<ul>
<li><span style="color:LightGreen">Brightness/Contrast ✔</span></li>
<li><span style="color:LightGreen">Gamma ✔</span></li>
<li><span style="color:LightGreen">Hue/Saturation/Value ✔</span></li>
<li><span style="color:LightGreen">Invert Color ✔</span></li>
<li><span style="color:DimGrey">Light Falloff -</span></li>
<li><span style="color:GoldenRod">Mix Color O (see Mix converter)</span></li>
<li><span style="color:DimGrey">RGB Curves -</span></li>
<li><span style="color:DimGrey">Color Balance -</span></li>
<li><span style="color:DimGrey">Color Correction -</span></li>
<li><span style="color:DimGrey">Hue Correct -</span></li>
<li><span style="color:IndianRed">Exposure ❌</span></li>
<li><span style="color:IndianRed">Tonemap ❌</span></li>
<br>
<li><span style="color:IndianRed">Alpha Over ❌</span></li>
<li><span style="color:IndianRed">Z Combine ❌</span></li>
<li><span style="color:IndianRed">Alpha Convert ❌</span></li>
<li><span style="color:IndianRed">Convert Colorspace ❌</span></li>
<li><span style="color:LightGreen">Set Alpha ✔</span></li>
</ul>
</details>

<details>
<summary>Converter</summary>
<ul>
<li><span style="color:GoldenRod">Blackbody O (Missing rec709->linear, very minor color difference)</span></li>
<li><span style="color:LightGreen">Clamp ✔</span></li>
<li><span style="color:DimGrey">Color Ramp -</span></li>
<li><span style="color:GoldenRod">Combine Color O (No colorspace option for YUV/YCbCr)</span></li>
<li><span style="color:LightGreen">Combine XYZ ✔</span></li>
<li><span style="color:DimGrey">Float Curve -</span></li>
<li><span style="color:LightGreen">Map Range ✔</span></li>
<li><span style="color:GoldenRod">Math O</span></li>
No dynamic inputs, the following operations do not work correctly: Smooth Minimum, Smooth Maximum<br>
Divide does not handle division by 0 as Blender
<li><span style="color:GoldenRod">Mix O</span></li>
No dynamic inputs, no non-uniform vector factor, the following blending modes do not work correctly: Overlay, Soft Light, Linear Light<br>
Divide does not handle division by 0 as Blender
<li><span style="color:LightGreen">RGB to BW ✔</span></li>
<li><span style="color:GoldenRod">Separate Color O (No colorspace option for YUV/YCbCr)</span></li>
<li><span style="color:LightGreen">Separate XYZ ✔</span></li>
<li><span style="color:IndianRed">Vector Math ❌</span></li>
<li><span style="color:LightGreen">Wavelength ✔</span></li>
</ul>
</details>

<details>
<summary>Transform</summary>
<ul>
<li><span style="color:GoldenRod">Rotate O (No bicubic interpolation)</span></li>
<li><span style="color:GoldenRod">Scale O (No bicubic interpolation)</span></li>
<li><span style="color:GoldenRod">Transform O (No bicubic interpolation)</span></li>
<li><span style="color:GoldenRod">Translate O (No bicubic interpolation)</span></li>
<li><span style="color:IndianRed">Corner Pin ❌</span></li>
<li><span style="color:IndianRed">Crop ❌</span></li>
<li><span style="color:IndianRed">Displace ❌</span></li>
<li><span style="color:IndianRed">Flip ❌</span></li>
<li><span style="color:IndianRed">Map UV ❌</span></li>
<li><span style="color:IndianRed">Lens Distortion ❌</span></li>
<li><span style="color:IndianRed">Movie Distortion ❌</span></li>
</ul>
</details>

<details>
<summary>Texture</summary>
<ul>
<li><span style="color:IndianRed"> Brick Texture ❌</span></li>
<li><span style="color:IndianRed"> Checker Texture ❌</span></li>
<li><span style="color:IndianRed"> Gabor Texture ❌</span></li>
<li><span style="color:IndianRed"> Gradient Texture ❌</span></li>
<li><span style="color:IndianRed"> Magic Texture ❌</span></li>
<li><span style="color:IndianRed"> Noise Texture ❌</span></li>
<li><span style="color:IndianRed"> Voronoi Texture ❌</span></li>
<li><span style="color:IndianRed"> Wave Texture ❌</span></li>
<li><span style="color:IndianRed"> White Noise Texture ❌</span></li>
</ul>
</details>

<details>
<summary>Filter</summary>
<ul>
<li><span style="color:IndianRed"> Whatever blurs I can implement ❌</span></li>
<li><span style="color:IndianRed"> Anti-Aliasing ❌</span></li>
<li><span style="color:IndianRed"> Despeckle ❌</span></li>
<li><span style="color:IndianRed"> Dilate/Erode ❌</span></li>
<li><span style="color:IndianRed"> Filter FIlter :) ❌</span></li>
<li><span style="color:IndianRed"> Glare ❌</span></li>
<li><span style="color:IndianRed"> Kuwahara ❌</span></li>
<li><span style="color:IndianRed"> Pixelate ❌</span></li>
<li><span style="color:IndianRed"> Posterize ❌</span></li>
<li><span style="color:IndianRed"> Sun Beams ❌</span></li>
</ul>
</details>

Implemented nodes: 22/56