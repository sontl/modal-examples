#!/usr/bin/env python3
"""
Test script to verify model caching improvements in MultiTalk Fixed
"""

import modal
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multitalk_fixed import app, MultiTalkFixed

def test_model_caching():
    """Test that models are properly cached and not re-downloaded"""
    
    print("🧪 Testing MultiTalk model caching...")
    
    # Create a MultiTalk instance
    multitalk = MultiTalkFixed()
    
    # Test the verification methods
    print("\n📋 Testing model verification methods...")
    
    with app.run():
        # Test setup multiple times to ensure caching works
        print("\n🔄 First setup (may download models)...")
        multitalk.setup.remote()
        
        print("\n🔄 Second setup (should use cached models)...")
        multitalk.setup.remote()
        
        print("\n🔄 Third setup (should use cached models)...")
        multitalk.setup.remote()
    
    print("\n✅ Model caching test completed!")
    print("Check the logs above - you should see 'All models already complete!' after the first run")

if __name__ == "__main__":
    test_model_caching()