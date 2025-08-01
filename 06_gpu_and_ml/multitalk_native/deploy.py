#!/usr/bin/env python3
"""
Deployment script for MultiTalk Native on Modal.com
"""

import subprocess
import sys
from pathlib import Path

def deploy_multitalk():
    """Deploy MultiTalk Native to Modal"""
    script_path = Path(__file__).parent / "multitalk_simple.py"
    
    print("🚀 Deploying MultiTalk Native to Modal...")
    
    try:
        # Deploy the main application
        result = subprocess.run([
            "modal", "deploy", str(script_path)
        ], check=True, capture_output=True, text=True)
        
        print("✅ Deployment successful!")
        print(result.stdout)
        
        # Get the app URL
        print("\n📱 To access the web interface, run:")
        print(f"modal serve {script_path}::gradio_interface")
        
        print("\n🔧 To test the CLI, run:")
        print(f"modal run {script_path} --audio-path 'path/to/audio.wav' --image-path 'path/to/image.jpg'")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Deployment failed: {e}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)

def test_deployment():
    """Test the deployed MultiTalk application"""
    print("🧪 Testing MultiTalk deployment...")
    
    try:
        # Test that the app is accessible
        result = subprocess.run([
            "modal", "app", "list"
        ], check=True, capture_output=True, text=True)
        
        if "multitalk-simple" in result.stdout:
            print("✅ MultiTalk Simple app found in Modal")
        else:
            print("❌ MultiTalk Simple app not found")
            return False
            
        print("✅ Deployment test passed!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy MultiTalk Native")
    parser.add_argument("--test", action="store_true", help="Test deployment")
    
    args = parser.parse_args()
    
    if args.test:
        test_deployment()
    else:
        deploy_multitalk()