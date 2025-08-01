#!/usr/bin/env python3
"""
Production API Test Script
==========================
Test the production API with various email examples
"""

import requests
import json
import time

def test_production_api():
    """Test the production API."""
    
    base_url = "http://localhost:8080"
    
    print("🧪 TESTING PRODUCTION API")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        ("Hi IT Team, I'm requesting access to the Sales Dashboard for Q3 analytics.", "Work"),
        ("Hi buddy, how are you doing? I was thinking about our conversation last week.", "Personal"),
        ("CRITICAL: Database breach detected! Immediate action required!", "Urgent"),
        ("Thank you for your prompt response.", "Standard"),
        ("CONGRATULATIONS! You've won $$1,000,000! Click here to claim!", "Spam")
    ]
    
    print("🔍 Testing email classification...")
    print("-" * 50)
    
    for i, (email, expected) in enumerate(test_cases, 1):
        try:
            print(f"\n{i}. Testing: {expected}")
            print(f"   Email: {email[:50]}...")
            
            response = requests.post(
                f"{base_url}/classify",
                json={'email': email},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                predicted = result['category']
                confidence = result['confidence']
                
                is_correct = predicted == expected
                status = "✅" if is_correct else "❌"
                
                print(f"   {status} Predicted: {predicted} ({confidence:.1%} confidence)")
                print(f"   Expected: {expected}")
                
            else:
                print(f"   ❌ API Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test health endpoint
    print("\n🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Health check passed: {health['status']}")
            print(f"   Model loaded: {health['model_loaded']}")
            print(f"   Model version: {health['model_version']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Test model info
    print("\n🔍 Testing model info...")
    try:
        response = requests.get(f"{base_url}/model_info", timeout=5)
        if response.status_code == 200:
            info = response.json()
            print(f"✅ Model info retrieved:")
            print(f"   Version: {info['model_version']}")
            print(f"   Categories: {', '.join(info['categories'])}")
            print(f"   Accuracy target: {info['accuracy_target']}")
        else:
            print(f"❌ Model info failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Model info error: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Production API testing completed!")

if __name__ == "__main__":
    test_production_api()
