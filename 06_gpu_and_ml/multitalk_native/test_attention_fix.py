#!/usr/bin/env python3
"""
Test script to verify the attention implementation fix
"""

import os
import sys

def test_transformers_attention():
    """Test that transformers can be loaded with eager attention"""
    
    # Set environment variable
    os.environ["TRANSFORMERS_ATTN_IMPLEMENTATION"] = "eager"
    
    try:
        from transformers import AutoConfig, AutoModel
        
        print("✓ Transformers imported successfully")
        
        # Test loading a config with eager attention
        config = AutoConfig.from_pretrained(
            "TencentGameMate/chinese-wav2vec2-base",
            attn_implementation='eager'
        )
        
        print("✓ Config loaded with eager attention")
        print(f"Attention implementation: {getattr(config, 'attn_implementation', 'not set')}")
        
        # Test setting output_attentions
        config.output_attentions = True
        print("✓ output_attentions set successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_wav2vec_loading():
    """Test wav2vec model loading with attention fix"""
    
    try:
        from transformers import Wav2Vec2Model
        
        # This should work with eager attention
        model = Wav2Vec2Model.from_pretrained(
            "TencentGameMate/chinese-wav2vec2-base",
            attn_implementation='eager'
        )
        
        print("✓ Wav2Vec2 model loaded successfully with eager attention")
        
        # Test setting output_attentions on the config
        model.config.output_attentions = True
        print("✓ output_attentions set on model config")
        
        return True
        
    except Exception as e:
        print(f"❌ Wav2Vec loading error: {e}")
        return False

if __name__ == "__main__":
    print("Testing attention implementation fixes...")
    print("=" * 50)
    
    success1 = test_transformers_attention()
    print()
    success2 = test_wav2vec_loading()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("✅ All attention tests passed!")
    else:
        print("❌ Some tests failed - attention fix may need adjustment")