"""
Test script for the FastAPI backend.
Run this to verify your API is working correctly.
"""
import requests
import sys
from pathlib import Path

# Default API URL (adjust if needed)
API_BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("\n🔍 Testing /health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        print("✅ Health check passed!")
        print(f"   Status: {data['status']}")
        print(f"   Service: {data['service']}")
        print(f"   GPU Available: {data['gpu_available']}")
        print(f"   Available Models: {len(data['available_models'])} found")
        
        if data['available_models']:
            print("\n   Models:")
            for model in data['available_models'][:5]:  # Show first 5
                print(f"     - {model}")
        else:
            print("   ⚠️  No models found! Train some models first.")
            return False
        
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_detect():
    """Test detection endpoint."""
    print("\n🔍 Testing /detect endpoint...")
    
    test_cases = [
        {
            "text": "The quick brown fox jumps over the lazy dog. This is a natural sentence.",
            "expected": "Should detect as real (human-written)"
        },
        {
            "text": "As an AI language model, I can help you with various tasks. I am designed to assist users with information.",
            "expected": "Should detect as fake (AI-generated)"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n  Test Case {i}: {test_case['expected']}")
        print(f"  Text: {test_case['text'][:60]}...")
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/detect",
                json={"text": test_case["text"]},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            print(f"  ✅ Prediction: {'FAKE' if data['is_fake'] else 'REAL'}")
            print(f"     Probability (fake): {data['probability']:.2%}")
            print(f"     Confidence: {data['confidence']:.2%}")
            print(f"     Model: {data['model_info']['model_name']}")
            
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Detection failed: {e}")
            return False
    
    return True


## Pair detection test removed (endpoint no longer supported)


def test_models_endpoint():
    """Test models listing endpoint."""
    print("\n🔍 Testing /models endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/models", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        print("✅ Models endpoint passed!")
        print(f"   Available Models: {len(data['available_models'])}")
        print(f"   Cached Models: {len(data['cached_models'])}")
        print(f"   Cached Extractors: {len(data['cached_extractors'])}")
        
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Models endpoint failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("🚀 FastAPI Backend Test Suite")
    print("=" * 60)
    print(f"Testing API at: {API_BASE_URL}")
    
    # Check if server is running
    try:
        requests.get(f"{API_BASE_URL}/", timeout=5)
    except requests.exceptions.RequestException:
        print("\n❌ Cannot connect to API server!")
        print(f"   Make sure the server is running at {API_BASE_URL}")
        print("\n   To start the server:")
        print("   cd deepfake-text-detector")
        print("   uvicorn api.app:app --reload")
        sys.exit(1)
    
    # Run tests
    results = {
        "Health Check": test_health(),
        "Detection": test_detect(),
        "Models Listing": test_models_endpoint()
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your API is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
