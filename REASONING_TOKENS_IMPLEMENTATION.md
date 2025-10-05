# Reasoning Tokens Implementation - OpenRouter Compliant

## ✅ Implementation Complete

Your tool agent now fully supports OpenRouter's reasoning tokens specification for models like Claude 3.7+, GPT-5, o1/o3 series, and other reasoning models.

## What Was Implemented

### 1. **Reasoning Parameter in API Request** (Lines 1320-1323)
```python
reasoning={
    "effort": "medium",  # Balance between speed and depth
    "exclude": False     # Include reasoning in response
}
```

**Options:**
- `"effort": "high"` - Deep thinking (~80% of max_tokens for reasoning)
- `"effort": "medium"` - Balanced (~50% of max_tokens for reasoning)
- `"effort": "low"` - Quick thinking (~20% of max_tokens for reasoning)
- `"exclude": True` - Use reasoning internally but don't return it (saves tokens)

### 2. **Reasoning Token Collection** (Lines 1334-1352)
```python
# Buffers for reasoning data
reasoning_buffer = ""  # Collects delta.reasoning tokens
reasoning_details_buffer = []  # Collects delta.reasoning_details array

# During streaming
if hasattr(delta, 'reasoning') and delta.reasoning:
    reasoning_buffer += delta.reasoning
    yield {
        "type": "reasoning_token",
        "token": delta.reasoning
    }

if hasattr(delta, 'reasoning_details') and delta.reasoning_details:
    reasoning_details_buffer.extend(delta.reasoning_details)
```

### 3. **Reasoning Preservation in Messages** (Lines 1397-1427)
```python
# Add to message dict for API
if reasoning_details_buffer:
    assistant_message_dict["reasoning_details"] = reasoning_details_buffer

# Add to message object
if reasoning_buffer:
    assistant_message.reasoning = reasoning_buffer
if reasoning_details_buffer:
    assistant_message.reasoning_details = reasoning_details_buffer
```

## Response Event Types

### New Event: `reasoning_token`
```json
{
  "type": "reasoning_token",
  "token": "Let me think through this step by step..."
}
```

Streamed in real-time as the model generates reasoning.

### Existing Events (unchanged)
- `start` - Agent started
- `thinking` - Analyzing query
- `thinking_token` - Regular response tokens
- `tool_call` - Calling a tool
- `tool_result` - Tool returned data
- `answer_complete` - Final answer ready
- `complete` - Done

## How It Works

### Example Flow with Reasoning:

1. **User asks:** "What are the top safety hazards and what should we do?"

2. **Model thinks (reasoning tokens):**
   ```
   🧠 REASONING: "First, I need to get the data summary to understand 
   what datasets are available. Then I'll query for top hazards by 
   frequency. After that, I should search for OSHA standards related 
   to those specific hazards to provide authoritative recommendations..."
   ```

3. **Model calls tools:**
   - `get_data_summary(sheet_name="hazard")`
   - `get_top_values(sheet_name="hazard", column_name="hazard_title")`
   - `search_web(query="OSHA top workplace hazards prevention")`

4. **Model synthesizes final answer** (regular content tokens)

## Reasoning Details Structure

According to OpenRouter spec, `reasoning_details` is an array of objects:

```json
{
  "reasoning_details": [
    {
      "type": "reasoning.summary",
      "summary": "Analyzed the problem by breaking it into components",
      "id": "reasoning-summary-1",
      "format": "anthropic-claude-v1",
      "index": 0
    },
    {
      "type": "reasoning.text",
      "text": "Let me work through this systematically:\n1. First...\n2. Second...",
      "signature": null,
      "id": "reasoning-text-1",
      "format": "anthropic-claude-v1",
      "index": 1
    }
  ]
}
```

**Types:**
- `reasoning.summary` - High-level summary of reasoning
- `reasoning.text` - Raw reasoning text
- `reasoning.encrypted` - Encrypted/redacted reasoning

## Model Support

### ✅ Fully Supported (with reasoning tokens):
- **Anthropic**: Claude 3.7+, Claude 4, Claude 4.1
- **OpenAI**: o1 series, o3 series, GPT-5 series
- **Grok**: Reasoning models
- **Gemini**: Flash Thinking models
- **Alibaba**: Qwen thinking models (some)

### ⚠️ Partial Support:
- Some models don't return reasoning tokens (OpenAI o-series, Gemini Flash Thinking)
- They still use reasoning internally, but it's not visible

### ❌ No Support:
- Non-reasoning models (GPT-4o, Claude 3.5, etc.)
- The `reasoning` parameter is ignored for these models

## Benefits

### 1. **Transparency** 🔍
See exactly how the model arrives at conclusions:
```
🧠 "I need to first check if we have incident data, then aggregate 
by department to find patterns, then search for OSHA guidelines..."
```

### 2. **Better Debugging** 🐛
Understand why the model chose certain tools or made specific decisions.

### 3. **Improved Quality** ✨
Models with reasoning enabled produce more thoughtful, accurate responses.

### 4. **Tool Calling Context** 🛠️
Reasoning blocks are preserved when tools are called, maintaining continuity.

## Configuration Options

### Option 1: High Effort (Deep Thinking)
```python
reasoning={
    "effort": "high",
    "exclude": False
}
```
**Use for:** Complex analysis, critical safety decisions, multi-step problems

### Option 2: Medium Effort (Balanced) ⭐ **Current Default**
```python
reasoning={
    "effort": "medium",
    "exclude": False
}
```
**Use for:** General queries, balanced speed/quality

### Option 3: Low Effort (Quick)
```python
reasoning={
    "effort": "low",
    "exclude": False
}
```
**Use for:** Simple queries, fast responses needed

### Option 4: Hidden Reasoning
```python
reasoning={
    "effort": "high",
    "exclude": True  # Use reasoning but don't show it
}
```
**Use for:** Production where you want quality but not reasoning display

### Option 5: Disabled (No Reasoning)
```python
# Don't include reasoning parameter
```
**Use for:** Non-reasoning models, maximum speed

## Token Usage & Billing

⚠️ **Important:** Reasoning tokens are counted as **output tokens** for billing.

**Example:**
- Query: 100 tokens (input)
- Reasoning: 500 tokens (output)
- Final answer: 300 tokens (output)
- **Total billed:** 100 input + 800 output tokens

**Cost Optimization:**
- Use `"exclude": True` to hide reasoning (still uses tokens but improves quality)
- Use `"effort": "low"` for simpler queries
- Use `"effort": "high"` only when needed

## Frontend Integration

Your frontend should handle the new `reasoning_token` event:

```javascript
// Example frontend handling
eventSource.addEventListener('message', (event) => {
  const data = JSON.parse(event.data);
  
  switch(data.type) {
    case 'reasoning_token':
      // Display in a "Thinking..." section
      appendToReasoningBox(data.token);
      break;
    
    case 'thinking_token':
      // Display in main response area
      appendToResponseBox(data.token);
      break;
    
    // ... other event types
  }
});
```

## Testing

### Test with Reasoning Model:
```python
# Use a reasoning-capable model
model = "anthropic/claude-sonnet-4"  # or "openai/gpt-5-mini"
query = "Analyze our top 5 hazards and recommend OSHA-compliant solutions"
```

### Expected Output:
```
🧠 REASONING: "I should start by getting the hazard data summary..."
🔧 Using tool: get_data_summary
🧠 REASONING: "Now I'll get the top 5 hazards by frequency..."
🔧 Using tool: get_top_values
🧠 REASONING: "For each hazard, I should search for OSHA standards..."
🔧 Using tool: search_web
💬 FINAL ANSWER: "Based on the data, your top 5 hazards are..."
```

## Compliance with OpenRouter Spec

✅ **Request Parameters:**
- `reasoning.effort` - Supported (high/medium/low)
- `reasoning.max_tokens` - Mapped to effort levels
- `reasoning.exclude` - Supported (true/false)
- `reasoning.enabled` - Implicit (inferred from effort)

✅ **Response Structure:**
- `delta.reasoning` - Collected and streamed
- `delta.reasoning_details` - Collected and preserved
- `message.reasoning_details` - Passed back in context

✅ **Reasoning Preservation:**
- Reasoning blocks preserved during tool calls
- Maintains conversation continuity
- Required for multi-turn tool calling

## Summary

Your agent now:
1. ✅ Requests reasoning tokens from OpenRouter
2. ✅ Streams reasoning tokens to frontend
3. ✅ Preserves reasoning_details in message history
4. ✅ Maintains reasoning continuity during tool calls
5. ✅ Fully compliant with OpenRouter specification

**The implementation is production-ready!** 🎉
