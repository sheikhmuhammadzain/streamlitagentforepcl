# Serper API Web Search + Image Search Integration

## Overview
Successfully integrated Serper API for both web search and image search into the tool-based agent to provide safety standards, OSHA regulations, best practices with authoritative sources, plus visual references like safety signs, PPE, and diagrams.

## What Was Added

### 1. **Web Search Tool** (Lines 35-101)
```python
async def search_web(query: str, num_results: int = 5) -> str:
    """Search the web for safety standards, best practices, and solutions"""
```

**Features:**
- Searches Google via Serper API
- Returns titles, links, snippets, and positions
- **NEW:** Includes thumbnail images for results (when available)
- **NEW:** Knowledge graph images (logos, diagrams)
- Handles up to 10 results per query
- 10-second timeout for reliability

**API Key:** `1a7343c2485b3e95dde021b5bb0b24296f6ce659` (hardcoded as requested)

### 2. **Image Search Tool** (Lines 104-160)
```python
async def search_images(query: str, num_results: int = 10) -> str:
    """Search for images related to safety topics"""
```

**Features:**
- Dedicated image search endpoint
- Returns full-size image URLs and thumbnails
- Includes image dimensions (width/height)
- Provides source website and context link
- Up to 10 high-quality images per search
- Perfect for finding safety signs, PPE, diagrams, infographics

**Returns:**
```json
{
  "images": [
    {
      "title": "Workplace Safety Signs",
      "imageUrl": "https://example.com/image.jpg",
      "thumbnailUrl": "https://...",
      "source": "example.com",
      "link": "https://example.com/page",
      "width": 1200,
      "height": 800
    }
  ]
}
```

### 3. **Tool Definitions** (Lines 1063-1106)
Added to the TOOLS array for AI function calling:

**search_web:**
```json
{
  "name": "search_web",
  "description": "Search the web for safety standards, best practices, regulations...",
  "parameters": {
    "query": "Search query focused on safety standards...",
    "num_results": "Number of results (default 5, max 10)"
  }
}
```

**search_images:**
```json
{
  "name": "search_images",
  "description": "Search for images related to safety topics...",
  "parameters": {
    "query": "Image search query (e.g., 'workplace hazard signs', 'PPE equipment')...",
    "num_results": "Number of images (default 10, max 10)"
  }
}
```

### 4. **Tool Registration** (Lines 1111-1122)
```python
TOOL_FUNCTIONS = {
    ...
    "search_web": search_web,
    "search_images": search_images  # NEW
}
```

### 5. **Async Execution Support** (Lines 1372-1376, 1433-1436)
Handles async tools properly:
```python
if asyncio.iscoroutinefunction(TOOL_FUNCTIONS[function_name]):
    result = await TOOL_FUNCTIONS[function_name](**function_args)
else:
    result = TOOL_FUNCTIONS[function_name](**function_args)
```

### 6. **Enhanced System Prompt** (Lines 1253-1254)
- Lists `search_web` and `search_images` as available tools
- Instructs AI to cite authoritative sources (OSHA, NIOSH)
- Encourages including links to safety standards
- Can now provide visual references (images) in responses

## How It Works

### Example Flow 1: Web Search
1. **User asks:** "What are the OSHA requirements for fall protection?"

2. **AI decides to use search_web:**
   ```json
   {
     "query": "OSHA fall protection standards requirements",
     "num_results": 5
   }
   ```

3. **Serper API returns (with images):**
   ```json
   {
     "results": [
       {
         "title": "Fall Protection - 1926.501 | OSHA",
         "link": "https://www.osha.gov/laws-regs/regulations/...",
         "snippet": "Each employee on a walking/working surface...",
         "thumbnail": "https://encrypted-tbn0.gstatic.com/images?q=..."
       }
     ],
     "knowledge_graph": {
       "title": "OSHA",
       "imageUrl": "https://upload.wikimedia.org/.../OSHA_logo.png"
     }
   }
   ```

4. **AI synthesizes response:**
   ```
   ## What's Happening
   OSHA requires fall protection at 6 feet in construction...
   
   ## Why It Matters
   Falls are the leading cause of death in construction...
   
   ## What To Do About It
   1. Install guardrails on all elevated surfaces
   2. Provide personal fall arrest systems
   
   **Sources:**
   - [OSHA 1926.501](https://www.osha.gov/...)
   - [NIOSH Fall Prevention](https://www.cdc.gov/...)
   ```

### Example Flow 2: Image Search
1. **User asks:** "Show me examples of proper PPE for confined spaces"

2. **AI decides to use search_images:**
   ```json
   {
     "query": "confined space PPE safety equipment",
     "num_results": 10
   }
   ```

3. **Serper API returns:**
   ```json
   {
     "images": [
       {
         "title": "Confined Space Entry PPE Kit",
         "imageUrl": "https://example.com/ppe-kit.jpg",
         "thumbnailUrl": "https://...",
         "source": "safetysupplies.com",
         "width": 1200,
         "height": 800
       }
     ]
   }
   ```

4. **AI can reference images in response:**
   ```
   ## Essential PPE for Confined Spaces
   
   Here are the key equipment items:
   1. Self-contained breathing apparatus (SCBA)
   2. Full-body harness with retrieval line
   3. Gas detection monitor
   
   **Visual References:**
   - [PPE Kit Example](https://example.com/ppe-kit.jpg)
   - [Harness Setup Diagram](https://example.com/harness.jpg)
   ```

## When AI Uses These Tools

### Web Search (`search_web`)
The AI will automatically use this when:
- User asks about safety standards or regulations
- Recommendations need authoritative backing
- Compliance information is required
- Best practices from industry experts are needed
- Specific OSHA/NIOSH guidelines are referenced

**Example queries AI generates:**
- `"OSHA fall protection standards"`
- `"workplace chemical hazard prevention best practices"`
- `"confined space entry safety requirements"`
- `"NIOSH lifting guidelines"`
- `"lockout tagout procedures OSHA"`

### Image Search (`search_images`)
The AI will automatically use this when:
- User asks to "show" or "see" examples
- Visual references would help explain concepts
- User mentions signs, posters, diagrams, or equipment
- Training materials need visual aids
- PPE or safety equipment identification is needed

**Example queries AI generates:**
- `"workplace hazard warning signs"`
- `"PPE safety equipment types"`
- `"confined space entry diagram"`
- `"OSHA safety poster"`
- `"fire extinguisher types chart"`
- `"proper lifting technique infographic"`

## Response Formats

### Web Search Results
```json
{
  "query": "OSHA fall protection",
  "results_count": 5,
  "results": [
    {
      "title": "Fall Protection - OSHA",
      "link": "https://www.osha.gov/...",
      "snippet": "Requirements for fall protection...",
      "position": 1,
      "thumbnail": "https://encrypted-tbn0.gstatic.com/images?q=..."
    }
  ],
  "knowledge_graph": {
    "title": "OSHA",
    "type": "Government agency",
    "description": "Occupational Safety and Health Administration",
    "imageUrl": "https://upload.wikimedia.org/.../OSHA_logo.png"
  },
  "search_metadata": {
    "total_results": "1,234,567"
  }
}
```

### Image Search Results
```json
{
  "query": "workplace safety signs",
  "images_count": 10,
  "images": [
    {
      "title": "Workplace Safety Signs Collection",
      "imageUrl": "https://example.com/signs.jpg",
      "thumbnailUrl": "https://example.com/thumb.jpg",
      "source": "safetysigns.com",
      "link": "https://example.com/page",
      "width": 1920,
      "height": 1080
    }
  ],
  "search_metadata": {
    "total_results": "45,678"
  }
}
```

## Dependencies Added

```python
import httpx  # For async HTTP requests
```

Make sure to install:
```bash
pip install httpx
```

## Benefits

1. **Authoritative Sources**: Cites OSHA, NIOSH, industry standards
2. **Compliance Ready**: Provides regulatory requirements
3. **Up-to-date**: Gets latest safety guidelines from the web
4. **Actionable**: Links to official documentation
5. **Credible**: Backs recommendations with expert sources
6. **Visual Learning**: Provides images for better understanding
7. **Training Ready**: Can find safety posters, diagrams, and infographics
8. **Equipment Identification**: Shows examples of proper PPE and safety equipment

## Testing

### Test Web Search:
- "What are the safety requirements for confined spaces?"
- "Show me OSHA regulations for chemical handling"
- "What are best practices for preventing workplace falls?"

### Test Image Search:
- "Show me examples of proper PPE for construction"
- "What do confined space warning signs look like?"
- "Find safety infographics about lifting techniques"
- "Show me different types of fire extinguishers"

### Test Combined (Data + Web + Images):
- "Analyze our fall incidents and show me OSHA standards with visual examples"
- "What hazards do we have most and what do the warning signs look like?"

The agent will:
1. Analyze your data (if relevant)
2. Search web for standards and regulations
3. Find relevant images for visual reference
4. Combine insights with authoritative sources
5. Provide actionable recommendations with links and images
