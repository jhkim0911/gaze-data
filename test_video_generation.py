#!/usr/bin/env python3
"""
Debug why gemini-3-flash-preview fails with video.
List available models and test exact sync pattern.
"""
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import time

load_dotenv("/u/arkimjh/code/ECCV-jh/.env")
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("ERROR: API Key not found")
    exit(1)

client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})

print("=" * 60)
print("Listing all available models...")
print("=" * 60)

try:
    models = client.models.list()
    flash_models = []
    for model in models:
        name = model.name if hasattr(model, 'name') else str(model)
        if 'flash' in name.lower() or 'gemini-3' in name.lower():
            flash_models.append(name)
            print(f"  {name}")
    
    print(f"\nFound {len(flash_models)} flash/gemini-3 models")
except Exception as e:
    print(f"Failed to list models: {e}")

# Check if gemini-3-flash-preview exists specifically
print("\n" + "=" * 60)
print("Checking gemini-3-flash-preview model info...")
print("=" * 60)
try:
    model_info = client.models.get(model="models/gemini-3-flash-preview")
    print(f"Model: {model_info.name}")
    print(f"Display Name: {getattr(model_info, 'display_name', 'N/A')}")
    print(f"Supported Methods: {getattr(model_info, 'supported_generation_methods', 'N/A')}")
    print(f"Input Token Limit: {getattr(model_info, 'input_token_limit', 'N/A')}")
except Exception as e:
    print(f"Failed to get model info: {e}")

# Test text-only generation with gemini-3-flash-preview
print("\n" + "=" * 60)
print("Test: gemini-3-flash-preview with TEXT ONLY")
print("=" * 60)
try:
    resp = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents="Say hello in one word."
    )
    print(f"SUCCESS: {resp.text}")
except Exception as e:
    print(f"FAILED: {e}")

# Now test with video
VIDEO_PATH = "/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social/social-iq/gaze_videos/_0at8kXKWSw_sam3rf_viz.mp4"

print("\n" + "=" * 60)
print("Uploading video...")
print("=" * 60)
video_file = client.files.upload(file=VIDEO_PATH, config={"mime_type": "video/mp4"})
print(f"Name: {video_file.name}")
print(f"URI: {video_file.uri}")
print(f"State: {video_file.state}")
print(f"Mime Type: {video_file.mime_type}")

while video_file.state == types.FileState.PROCESSING:
    time.sleep(2)
    video_file = client.files.get(name=video_file.name)

print(f"Final state: {video_file.state}")

# Test with video
print("\n" + "=" * 60)
print("Test: gemini-3-flash-preview with VIDEO")
print("=" * 60)
try:
    resp = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[video_file, "What do you see?"]
    )
    print(f"SUCCESS: {resp.text[:200]}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    
    # Print more debug info
    if hasattr(e, 'response'):
        print(f"Response: {e.response}")
    if hasattr(e, 'args'):
        print(f"Args: {e.args}")

# Cleanup
try:
    client.files.delete(name=video_file.name)
    print(f"\nDeleted: {video_file.name}")
except:
    pass
