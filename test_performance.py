"""
Performance Testing Script for Blazing Fast WebSocket Agent

Run this to measure and verify optimization improvements
"""

import asyncio
import time
import json
from typing import Dict, List


async def test_agent_performance():
    """Test agent response time with different models"""
    
    print("🚀 Testing Agent Performance\n")
    print("=" * 60)
    
    # Import after FastAPI app is available
    from fastapiepcl.app.services.tool_agent import run_tool_based_agent
    from fastapiepcl.app.services.data_cache import get_cache_stats, clear_all_caches
    
    # Test queries
    test_queries = [
        "Show top 10 departments by incident count",
        "What are the most common hazard types?",
        "Analyze incident trends by location"
    ]
    
    # Models to test
    models = [
        ("google/gemini-flash-1.5:free", "Gemini Flash (FASTEST)"),
        ("z-ai/glm-4.6", "Grok Fast (Reasoning)"),
    ]
    
    results: Dict[str, List[float]] = {}
    
    for model_id, model_name in models:
        print(f"\n📊 Testing: {model_name}")
        print("-" * 60)
        
        model_times = []
        
        for i, query in enumerate(test_queries, 1):
            print(f"\nQuery {i}: {query[:50]}...")
            
            start_time = time.time()
            first_token_time = None
            event_count = 0
            
            try:
                async for event in run_tool_based_agent(
                    query=query,
                    model=model_id
                ):
                    event_count += 1
                    
                    # Track first token
                    if first_token_time is None and event["type"] in ["thinking_token", "tool_result"]:
                        first_token_time = time.time() - start_time
                    
                    # Complete
                    if event["type"] == "complete":
                        total_time = time.time() - start_time
                        model_times.append(total_time)
                        
                        print(f"  ✅ Total time: {total_time*1000:.0f}ms")
                        print(f"  ⚡ First token: {(first_token_time or 0)*1000:.0f}ms")
                        print(f"  📦 Events: {event_count}")
                        break
                    
                    # Error
                    if event["type"] == "error":
                        print(f"  ❌ Error: {event.get('message', 'Unknown')}")
                        break
            
            except Exception as e:
                print(f"  ❌ Exception: {str(e)}")
                continue
            
            # Small delay between queries
            await asyncio.sleep(0.5)
        
        results[model_name] = model_times
        
        # Show average
        if model_times:
            avg_time = sum(model_times) / len(model_times)
            print(f"\n  📈 Average: {avg_time*1000:.0f}ms")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 60)
    
    for model_name, times in results.items():
        if times:
            avg = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"\n{model_name}:")
            print(f"  Average: {avg*1000:.0f}ms")
            print(f"  Min:     {min_time*1000:.0f}ms")
            print(f"  Max:     {max_time*1000:.0f}ms")
    
    # Cache stats
    print("\n" + "=" * 60)
    print("💾 CACHE STATISTICS")
    print("=" * 60)
    
    stats = get_cache_stats()
    
    print(f"\nWorkbook Cache:")
    print(f"  Hits:     {stats['workbook_cache']['hits']}")
    print(f"  Misses:   {stats['workbook_cache']['misses']}")
    print(f"  Hit Rate: {stats['workbook_cache']['hit_rate']:.1f}%")
    
    print(f"\nQuery Cache:")
    print(f"  Hits:     {stats['query_cache']['hits']}")
    print(f"  Misses:   {stats['query_cache']['misses']}")
    print(f"  Hit Rate: {stats['query_cache']['hit_rate']:.1f}%")


async def test_cache_performance():
    """Test cache performance improvement"""
    
    print("\n" + "=" * 60)
    print("💾 Testing Cache Performance")
    print("=" * 60)
    
    from fastapiepcl.app.services.agent import load_default_sheets
    from fastapiepcl.app.services.data_cache import (
        get_cached_workbook,
        cache_workbook,
        clear_all_caches
    )
    
    # Clear cache first
    clear_all_caches()
    print("\n✅ Cache cleared")
    
    # First load (cache miss)
    print("\n📥 First load (cache MISS)...")
    start = time.time()
    workbook = load_default_sheets()
    cache_workbook(workbook)
    first_load_time = time.time() - start
    print(f"  Time: {first_load_time*1000:.0f}ms")
    
    # Second load (cache hit)
    print("\n⚡ Second load (cache HIT)...")
    start = time.time()
    cached_workbook = get_cached_workbook()
    second_load_time = time.time() - start
    print(f"  Time: {second_load_time*1000:.0f}ms")
    
    # Calculate improvement
    if second_load_time > 0:
        speedup = first_load_time / second_load_time
        print(f"\n🚀 Speedup: {speedup:.1f}x faster!")
        print(f"  Saved: {(first_load_time - second_load_time)*1000:.0f}ms")


async def benchmark_event_batching():
    """Benchmark event batching overhead"""
    
    print("\n" + "=" * 60)
    print("📦 Testing Event Batching")
    print("=" * 60)
    
    # Simulate events
    events = [{"type": "progress", "message": f"Step {i}"} for i in range(100)]
    
    # Without batching (send each)
    print("\n❌ Without batching (100 events)...")
    start = time.time()
    sent_count = 0
    for event in events:
        # Simulate JSON serialization
        json.dumps(event)
        sent_count += 1
    no_batch_time = time.time() - start
    print(f"  Time: {no_batch_time*1000:.0f}ms")
    print(f"  Events sent: {sent_count}")
    
    # With batching (batch of 5)
    print("\n✅ With batching (batch size 5)...")
    start = time.time()
    sent_count = 0
    batch = []
    for event in events:
        batch.append(event)
        if len(batch) >= 5:
            # Simulate sending batch
            json.dumps(batch)
            sent_count += 1
            batch = []
    if batch:  # Send remaining
        json.dumps(batch)
        sent_count += 1
    batch_time = time.time() - start
    print(f"  Time: {batch_time*1000:.0f}ms")
    print(f"  Batches sent: {sent_count}")
    
    # Calculate improvement
    reduction = (1 - sent_count / 100) * 100
    speedup = no_batch_time / batch_time if batch_time > 0 else 0
    
    print(f"\n🚀 Improvement:")
    print(f"  Events reduced: {reduction:.0f}%")
    print(f"  Speedup: {speedup:.1f}x faster")


async def main():
    """Run all performance tests"""
    
    print("\n" + "=" * 60)
    print("⚡ BLAZING FAST AGENT - PERFORMANCE TESTS")
    print("=" * 60)
    
    try:
        # Test 1: Cache performance
        await test_cache_performance()
        
        # Test 2: Event batching
        await benchmark_event_batching()
        
        # Test 3: Agent performance (requires running FastAPI server)
        print("\n" + "=" * 60)
        print("🤖 Agent Performance Test")
        print("=" * 60)
        print("\n⚠️  Note: This requires the FastAPI server to be running")
        print("   Start server with: uvicorn fastapiepcl.app.main:app")
        
        try:
            await test_agent_performance()
        except ImportError:
            print("\n⚠️  Skipping agent test (server not running)")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✅ Performance tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
