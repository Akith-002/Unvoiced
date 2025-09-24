#!/usr/bin/env python3
"""
Simple test to check if Flask server is accessible
"""

import requests
import time

def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get("http://127.0.0.1:5000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running and accessible")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"❌ Server responded with status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing Flask server health...")
    if test_health():
        print("Server is ready for testing!")
    else:
        print("Server is not accessible. Please start Flask server first.")