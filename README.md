# ComfyUI-Blenderesque-Nodes
Blender-like nodes for ComfyUI.<br>

<h1>Work In Progress<br>
Nodes might change and you will need to redo your workflow</h1>

<img src=mary_combo.png style="width:50%;height:50%">

Install:
```
cd ./custom_nodes
git clone https://github.com/a-One-Fan/ComfyUI-Blenderesque-Nodes
pip install -r ./ComfyUI-Blendersque-Nodes/requirements.txt
```

Usage:<br><br>
Nodes will accept most inputs, akin to Blender's implicit conversion between vectors, floats, colors, and image types thereof.<br>
Nodes will output Blender-like data, and an image for general Comfy usage.<br>
Differently sized images are automatically cropped/padded like Blender's compositor.
You should prefer to use the Blender-like data output between these nodes, to reduce loss in precision from excessive colorspace transforms, or avoid very small images inteded for a simple preview, among other possible issues.<br>

Currently implemented nodes:<br><br>
<span style="color:LightGreen">Something with ✅</span> = Implemented<br>
<span style="color:IndianRed">Something with ❌</span> = Not implemented<br>
<span style="color:DimGrey">Something with -</span> = Probably won't be implemented?<br>
<SPAN STYLE="color:GoldenRod">Something with 🔵</span> = Partially implemented, useable<br>
Hopefully I'll figure out a way to do colorramps at some point.<br>
Blender lacks some compositor-applicable shader nodes, so this list will also include some shader nodes.<br>
Currently the list is incomplete.<br>
<br>
Hopefully in the future I'll look into integrating this with https://github.com/AIGODLIKE/ComfyUI-BlenderAI-node for maximum convenience. I highly recommend that Blender addon!<br>
<details>
<summary>Meta/Bugs/TODO</summary>
<ul>
<li><span style="color:IndianRed">Integrate with https://github.com/AIGODLIKE/ComfyUI-BlenderAI-node ❌</span></li>
<li><span style="color:IndianRed">Dynamic inputs ❌</span></li>
<li><span style="color:GoldenRod">Dynamic widgets 🔵</span></li>
<li><span style="color:IndianRed">UV Input Node (For mapping textures) ❌</span></li>
<li><span style="color:IndianRed">Resize Canvas Node ❌</span></li>
<li><span style="color:IndianRed">Extract Data Node (Get image, mask, canvas xy, float, etc.) ❌</span></li>
<li><span style="color:IndianRed">Merged input sockets and default values ("widgets") ❌</span></li>
<li><span style="color:IndianRed">Merged output blender and image sockets ❌</span></li>
<li><span style="color:IndianRed">Low precision on image transforms, teethy edges ❌</span></li>
</ul>
</details>

<details>
<summary>Input</summary>
<ul>
<li><span style="color:LightGreen">Value ✅</span></li>
<li><span style="color:LightGreen">RGB ✅</span></li>
<li><span style="color:IndianRed">Bokeh Image ❌</span></li>
<li><span style="color:IndianRed">UV Node (For mapping textures) ❌</span></li>
Most other input nodes seem redundant or not applicable.
</ul>
</details>

<details>
<summary>Color</summary>
<ul>
<li><span style="color:LightGreen">Brightness/Contrast ✅</span></li>
<li><span style="color:LightGreen">Gamma ✅</span></li>
<li><span style="color:LightGreen">Hue/Saturation/Value ✅</span></li>
<li><span style="color:LightGreen">Invert Color ✅</span></li>
<li><span style="color:DimGrey">Light Falloff -</span></li>
<li><span style="color:GoldenRod">Mix Color 🔵 (see Mix converter)</span></li>
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
<li><span style="color:LightGreen">Set Alpha ✅</span></li>
</ul>
</details>

<details>
<summary>Converter</summary>
<ul>
<li><span style="color:GoldenRod">Blackbody 🔵 (Missing rec709->linear, very minor color difference)</span></li>
<li><span style="color:LightGreen">Clamp ✅</span></li>
<li><span style="color:DimGrey">Color Ramp -</span></li>
<li><span style="color:GoldenRod">Combine Color 🔵 (No colorspace option for YUV/YCbCr)</span></li>
<li><span style="color:LightGreen">Combine XYZ ✅</span></li>
<li><span style="color:DimGrey">Float Curve -</span></li>
<li><span style="color:LightGreen">Map Range ✅</span></li>
<li><span style="color:LightGreen">Math ✅</span></li>
<li><span style="color:GoldenRod">Mix 🔵 (no non-uniform vector factor, no use alpha)</span></li>
<li><span style="color:LightGreen">RGB to BW ✅</span></li>
<li><span style="color:GoldenRod">Separate Color 🔵 (No colorspace option for YUV/YCbCr)</span></li>
<li><span style="color:LightGreen">Separate XYZ ✅</span></li>
<li><span style="color:IndianRed">Vector Math ❌</span></li>
<li><span style="color:LightGreen">Wavelength ✅</span></li>
<br>
<li><span style="color:IndianRed">Extract Data ❌</span></li>
</ul>
</details>

<details>
<summary>Transform</summary>
<ul>
<li><span style="color:GoldenRod">Rotate 🔵 (No bicubic interpolation)</span></li>
<li><span style="color:GoldenRod">Scale 🔵 (No bicubic interpolation)</span></li>
<li><span style="color:GoldenRod">Transform 🔵 (No bicubic interpolation)</span></li>
<li><span style="color:GoldenRod">Translate 🔵 (No bicubic interpolation)</span></li>
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