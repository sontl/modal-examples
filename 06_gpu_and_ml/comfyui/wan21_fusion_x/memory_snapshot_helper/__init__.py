import os
import torch
import logging

from aiohttp import web
from server import PromptServer

# Modal-aware CUDA initialization
def _initialize_cuda():
    """Initialize CUDA when the module is loaded (Modal snapshot aware)"""
    try:
        # Only initialize CUDA if we're not in a Modal snapshot phase
        # During snapshots, CUDA is not available even if configured
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            torch.cuda.init()
            torch.cuda.empty_cache()
            logging.info(f"[memory_snapshot_helper] CUDA initialized on import: {torch.cuda.device_count()} devices")
        else:
            logging.info("[memory_snapshot_helper] CUDA not available on import (likely during Modal snapshot)")
    except Exception as e:
        logging.info(f"[memory_snapshot_helper] CUDA initialization skipped: {e}")

# Initialize CUDA when module is imported (safe for Modal snapshots)
_initialize_cuda()

# ------- API Endpoints -------


@PromptServer.instance.routes.post("/cuda/set_device")
async def set_current_device(request):
    import torch
    import logging
    
    try:
        # Set CUDA environment for Modal
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        
        # Force torch to reinitialize CUDA if available (Modal post-snapshot)
        if torch.cuda.is_available():
            # Clear any existing CUDA context
            torch.cuda.empty_cache()
            
            # Reinitialize CUDA context
            torch.cuda.init()
            
            # Verify CUDA is working
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            
            # Test CUDA functionality
            test_tensor = torch.tensor([1.0]).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            
            logging.info(f"[memory_snapshot_helper] CUDA restored successfully: {device_count} devices, current: {current_device}")
            
            return web.json_response({
                "status": "success", 
                "cuda_available": True,
                "device_count": device_count,
                "current_device": current_device,
                "message": "CUDA restored and verified"
            })
        else:
            logging.warning("[memory_snapshot_helper] CUDA not available (likely during Modal snapshot)")
            return web.json_response({
                "status": "success", 
                "cuda_available": False,
                "message": "CUDA not available - this is normal during Modal snapshots"
            })
            
    except Exception as e:
        logging.error(f"[memory_snapshot_helper] Error setting CUDA device: {e}")
        return web.json_response({
            "status": "error", 
            "message": str(e),
            "cuda_available": False
        }, status=500)


# Empty for ComfyUI node registration
NODE_CLASS_MAPPINGS = {}
