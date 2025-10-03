"""
Simple test script to verify x-ai/grok-4-fast:free model works with OpenRouter
"""
import os
from openai import OpenAI

# Load environment variables manually from .env file
def load_env_file(filepath="app/.env"):
    """Simple .env file loader"""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print(f"✓ Loaded environment from {filepath}")
    else:
        print(f"⚠️  Warning: {filepath} not found")

load_env_file()

def test_grok_model():
    """Test the Grok 4 Fast Free model via OpenRouter"""
    
    # Configuration
    api_key = os.getenv("OPENROUTER_API_KEY")
    use_openrouter = os.getenv("USE_OPENROUTER", "false").lower() == "true"
    
    print("=" * 60)
    print("🧪 Testing Grok 4 Fast Free Model")
    print("=" * 60)
    print(f"\n✓ USE_OPENROUTER: {use_openrouter}")
    print(f"✓ API Key present: {bool(api_key)}")
    print(f"✓ API Key (first 20 chars): {api_key[:20] if api_key else 'None'}...")
    
    if not api_key:
        print("\n❌ ERROR: OPENROUTER_API_KEY not found in .env file")
        return False
    
    if not use_openrouter:
        print("\n⚠️  WARNING: USE_OPENROUTER is not set to 'true'")
        print("   Set USE_OPENROUTER=true in your .env file")
        return False
    
    # Create OpenRouter client
    print("\n📡 Creating OpenRouter client...")
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    
    # Test models to try
    models_to_test = [
        "x-ai/grok-4-fast:free",
        "google/gemini-2.5-flash:free",
    ]
    
    for model in models_to_test:
        print(f"\n{'─' * 60}")
        print(f"🤖 Testing model: {model}")
        print(f"{'─' * 60}")
        
        try:
            # Simple test prompt
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Respond concisely."
                    },
                    {
                        "role": "user",
                        "content": "Say 'Hello! I am working correctly.' and nothing else."
                    }
                ],
                max_tokens=50,
                temperature=0.1,
                extra_headers={
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "Safety Copilot Test"
                }
            )
            
            result = response.choices[0].message.content
            print(f"✅ SUCCESS!")
            print(f"📝 Response: {result}")
            print(f"🔢 Tokens used: {response.usage.total_tokens if hasattr(response, 'usage') else 'N/A'}")
            
            return True
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ FAILED!")
            print(f"💥 Error: {error_msg}")
            
            # Detailed error analysis
            if "invalid model" in error_msg.lower():
                print("\n🔍 Analysis: Model ID is not recognized by OpenRouter")
                print("   - Check https://openrouter.ai/models for valid model IDs")
            elif "401" in error_msg or "unauthorized" in error_msg.lower():
                print("\n🔍 Analysis: API key authentication failed")
                print("   - Verify your OPENROUTER_API_KEY is correct")
            elif "403" in error_msg or "forbidden" in error_msg.lower():
                print("\n🔍 Analysis: Access denied to this model")
                print("   - Your API key may not have access to this model")
            elif "429" in error_msg or "rate limit" in error_msg.lower():
                print("\n🔍 Analysis: Rate limit exceeded")
                print("   - Wait a moment and try again")
            else:
                print("\n🔍 Analysis: Unknown error")
            
            continue
    
    return False


def test_streaming():
    """Test streaming with Grok model"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    print("\n" + "=" * 60)
    print("🌊 Testing Streaming Mode")
    print("=" * 60)
    
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        print("\n📡 Streaming response from x-ai/grok-4-fast:free...")
        print("─" * 60)
        
        stream = client.chat.completions.create(
            model="x-ai/grok-4-fast:free",
            messages=[
                {
                    "role": "user",
                    "content": "Count from 1 to 5, one number per line."
                }
            ],
            stream=True,
            max_tokens=50,
            extra_headers={
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "Safety Copilot Test"
            }
        )
        
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
        
        print("\n" + "─" * 60)
        print(f"✅ Streaming SUCCESS!")
        print(f"📝 Full response: {full_response}")
        return True
        
    except Exception as e:
        print(f"\n❌ Streaming FAILED: {str(e)}")
        return False


if __name__ == "__main__":
    print("\n🚀 Starting Grok Model Tests\n")
    
    # Test 1: Basic model test
    success = test_grok_model()
    
    # Test 2: Streaming test
    if success:
        test_streaming()
    
    print("\n" + "=" * 60)
    print("✨ Tests Complete!")
    print("=" * 60 + "\n")
