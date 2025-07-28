import os
import shutil
from pathlib import Path

comfy_dir = Path(__file__).parent.parent.parent / "comfy"

model_management_path = str(comfy_dir / "model_management.py")
original_model_management_path = str(comfy_dir / "model_management_original.py")
is_patched = os.path.exists(original_model_management_path)


def _apply_cuda_safe_patch():
    """Apply a permanent patch that avoid torch cuda init during snapshots"""

    shutil.copy(model_management_path, original_model_management_path)
    print(
        "[memory_snapshot_helper] ==> Applying CUDA-safe patch for model_management.py"
    )

    with open(model_management_path, "r") as f:
        lines = f.readlines()

    # Ensure the necessary imports are present
    if not any("import os" in line for line in lines):
        lines.insert(0, "import os\n")
    
    patched_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Patch 1: Replace get_torch_device return statement
        if "return torch.device(torch.cuda.current_device())" in line:
            # Get the indentation of the current line
            indent = len(line) - len(line.lstrip())
            indent_str = " " * indent
            
            # Replace with safe CUDA detection
            patched_lines.extend([
                f"{indent_str}try:\n",
                f"{indent_str}    if torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES', '') != '':\n",
                f"{indent_str}        return torch.device(torch.cuda.current_device())\n",
                f"{indent_str}    else:\n",
                f"{indent_str}        return torch.device('cpu')\n",
                f"{indent_str}except (RuntimeError, AssertionError):\n",
                f"{indent_str}    return torch.device('cpu')\n"
            ])
            print("[memory_snapshot_helper] ==> Applied patch 1: get_torch_device")
            
        # Patch 2: Replace get_device_properties call
        elif "props = torch.cuda.get_device_properties(device)" in line:
            # Get the indentation of the current line
            indent = len(line) - len(line.lstrip())
            indent_str = " " * indent
            
            # Replace with safe GPU properties check
            patched_lines.extend([
                f"{indent_str}try:\n",
                f"{indent_str}    if torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES', '') != '':\n",
                f"{indent_str}        props = torch.cuda.get_device_properties(device)\n",
                f"{indent_str}    else:\n",
                f"{indent_str}        return False\n",
                f"{indent_str}except (RuntimeError, AssertionError):\n",
                f"{indent_str}    return False\n"
            ])
            print("[memory_snapshot_helper] ==> Applied patch 2: should_use_fp16")
            
        else:
            # Keep the original line
            patched_lines.append(line)
            
        i += 1

    # Save the patched version
    with open(model_management_path, "w") as f:
        f.writelines(patched_lines)

    print("[memory_snapshot_helper] ==> Successfully patched model_management.py")


if not is_patched:
    _apply_cuda_safe_patch()
