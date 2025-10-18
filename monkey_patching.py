import asyncio
import inspect
import nodes
import execution

from .nodes_base import BlenderData

old_amnol = execution._async_map_node_over_list

# https://github.com/comfyanonymous/ComfyUI/blob/master/execution.py
# !!! Possibly sensitive to Comfy updates
async def patched_async_map_node_over_list(prompt_id, unique_id, obj, input_data_all: dict, func, allow_interrupt=False, execution_block_cb=None, pre_execute_cb=None, hidden_inputs=None):
    if type(obj).__name__.find("Blender") == -1:
        new_in = {}
        it: dict = {}
        try:
            if isinstance(obj, type):
                it = obj.INPUT_TYPES()
            else:
                it = type(obj).INPUT_TYPES()
        except Exception as e: # Possibly unnecessary, but just in case
            print(f"Bad type: \n{e}\n{obj}\n{type(obj)}\n\n\n")
            return await old_amnol(prompt_id, unique_id, obj, input_data_all, func, allow_interrupt, execution_block_cb, pre_execute_cb, hidden_inputs)
        
        for k in input_data_all.keys():
            v = input_data_all[k]
            vnew = []
            for vi in v:
                if type(vi) is BlenderData:
                    target_type = it.get("required", {}).get(k) or it.get("optional", {}).get(k)
                    target_type = target_type[0]
                    converted = ""
                    if target_type == "FLOAT":
                        converted = vi.as_primitive_float()
                    elif target_type == "INT":
                        converted = int(vi.as_primitive_float())
                    elif target_type == "IMAGE":
                        converted = vi.as_rgb()
                    elif target_type == "MASK":
                        converted = vi.as_comfy_mask()
                    elif target_type == "LATENT":
                        converted = vi.as_vector()
                    else:
                        raise Exception(f"Unhandled Blenderesque monkey patching for type {target_type}, input types {it}")
                    vnew.append(converted)
                else:
                    vnew.append(vi)
            new_in[k] = vnew
        input_data_all = new_in

    return await old_amnol(prompt_id, unique_id, obj, input_data_all, func, allow_interrupt, execution_block_cb, pre_execute_cb, hidden_inputs)

execution._async_map_node_over_list = patched_async_map_node_over_list