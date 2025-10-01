"""
Test script to validate the Qwen Image Edit endpoint implementation
This script validates the structure and imports without running the actual Modal service
"""
import sys
import os

# Add the current directory to path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

print("Validating Qwen Image Edit endpoint implementation...")

try:
    import ast
    with open('misc/qwen_image_edit_endpoint.py', 'r') as f:
        source = f.read()
        # Parse the file to check for syntax errors
        tree = ast.parse(source)
        print("✓ Syntax validation passed")
        
    # Check for required components
    required_components = [
        'modal.App',
        'QwenImageEditService',
        'ImageEditRequest',
        'inference',
        'modal.enter(snap=True)',
        'load',
        'FlowMatchEulerDiscreteScheduler',
        'QwenImageEditPlusPipeline',
        'NunchakuQwenImageTransformer2DModel'
    ]
    
    for component in required_components:
        if component in source:
            print(f"✓ Found required component: {component}")
        else:
            print(f"✗ Missing required component: {component}")
    
    # Check for GPU configuration
    if 'gpu="L40S"' in source:
        print("✓ GPU configuration found")
    else:
        print("✗ GPU configuration missing")
        
    # Check for Modal volumes
    if 'CONTAINER_CACHE_VOLUME' in source and 'volumes={' in source:
        print("✓ Modal volume configuration found")
    else:
        print("✗ Modal volume configuration missing")
        
    print("\nValidation completed successfully!")
    print("\nTo deploy this application, run:")
    print("modal deploy misc/qwen_image_edit_endpoint.py")
    
except SyntaxError as e:
    print(f"✗ Syntax error in the file: {e}")
except Exception as e:
    print(f"✗ Error validating the file: {e}")