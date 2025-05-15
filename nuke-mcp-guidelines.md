# NukeMCP Usage Guide & Stability Enhancements

This document outlines the enhanced features of the NukeMCP system and provides guidance on how to use it effectively and stably.

## Overview of Enhancements

The NukeMCP system has been enhanced with:

1. **Workflow Rules** - Professional node graph organization rules and best practices
2. **Stability Improvements** - More robust code execution and error handling
3. **Template System** - Ready-made node configurations for common tasks
4. **Safety Checks** - Validation of nodes, parameters, and connections to prevent crashes
5. **Node Organization** - Tools for automatically arranging nodes with backdrops

## Stable Node Creation Patterns

When creating nodes, follow these guidelines for maximum stability:

### Creating Individual Nodes

```python
# Example using create_node tool
result = await mcp.create_node(
    node_type="Grade",                    # Use exact node type names
    position=[100, 100],                  # Explicit positioning
    parameters={
        "label": "Overall Grade",         # Use labels instead of renaming
        "channels": "rgb"                 # Only use parameters that exist on this node type
    }
)
```

### Using Workflow Templates

```python
# Create an entire keying setup at once
result = await mcp.create_workflow_template(
    template_type="keying",
    root_position=[100, 100],
    add_backdrop=True
)
```

Available templates:
- `keying` - Standard keying workflow with despill
- `color_correction` - Color correction chain with unpremult/premult
- `lens_distortion` - Lens distortion/redistortion setup
- `3d_simple` - Basic 3D setup with camera, card and renderer

### Creating 3D Setups

3D node setups follow a specific stable creation pattern:

1. Create all nodes first (Camera, Card, Scene, etc.)
2. Set positions for all nodes
3. Connect nodes only after all nodes have been created

The `create_workflow_template` tool handles this sequence automatically.

## Node Organization

### Auto Layout

```python
# Auto-arrange selected nodes
result = await mcp.auto_layout_nodes(selected_only=True)
```

### Organized Layout with Backdrops

```python
# Create a professional node graph with category grouping
result = await mcp.organize_node_graph(
    selected_only=False,
    direction="vertical"  # or "horizontal"
)
```

This tool:
- Groups nodes by category (inputs, color, FX, etc.)
- Creates color-coded backdrops for each category
- Arranges nodes in a clean, consistent flow

## Workflow Rules and Best Practices

The enhanced system enforces professional compositing practices:

1. **B-pipe Structure** - For Merge nodes, main pipeline connects to input B (1)
2. **Pre/Post Processing** - Unpremult before color correction, Premult after
3. **Clear Labeling** - Use labels instead of renaming nodes
4. **Organized Layout** - Vertical stacking with consistent spacing
5. **Smart Defaults** - Sensible default settings for common node types

The system will provide feedback when these rules are not followed, and in some cases will automatically apply fixes.

## Stability Guidelines

### Node Type Names

Use exact node type names:
- ✓ "Merge2" (not "Merge")
- ✓ "ColorCorrect" (not "ColorCorrection")
- ✓ "Premult" (not "Premultiply")

### BackdropNode Creation

For backdrop nodes, use the special create_node handler - the system will automatically use the correct method internally:

```python
result = await mcp.create_node(
    node_type="BackdropNode",
    position=[0, 0],
    parameters={
        "label": "COLOR",
        "note_font_size": 42,
        "tile_color": "0xC67171FF"  # Red
    }
)
```

### Parameter Validation

Only set parameters that exist on the node type:

```python
# Good practice - parameters are validated
result = await mcp.modify_node(
    name="Grade1",
    parameters={
        "channels": "rgb",  # Common parameter
        "white": [1.2, 1.2, 1.2, 1]  # Grade-specific parameter 
    }
)
```

### Node Flow Pattern

The most stable node creation pattern is:

1. Create all nodes without connecting
2. Position all nodes
3. Connect nodes only after all creation is complete

The template system follows this pattern automatically.

## Additional Tools

### Script Info

```python
# Get information about the current script
script_info = await mcp.get_script_info()
```

### Node Info

```python
# Get detailed info about a specific node
node_info = await mcp.get_node_info(node_name="Grade1")
```

### Code Execution

```python
# Execute Python code in Nuke with safety measures
result = await mcp.execute_nuke_code(code="""
import nuke
# Get all selected nodes
selected = [n.name() for n in nuke.selectedNodes()]
output["selected"] = selected
""")
```

## Using the Integrated Panel

1. In Nuke, go to **NukeMCP > Show Panel**
2. Set the port (default: 9876)
3. Click **Start Server**
4. Connect from your client application

## Troubleshooting

If you encounter stability issues:

1. Always create nodes first, then connect them
2. Use exact node type names from the supported list
3. Validate parameters before setting them
4. For complex setups, use the workflow templates
5. Check server logs for detailed error messages

## Workflow Reference

| Task | Recommended Tool |
|------|-----------------|
| Creating a single node | `create_node()` |
| Building a keying setup | `create_workflow_template(template_type="keying")` |
| Setting up color correction | `create_workflow_template(template_type="color_correction")` |
| Creating a 3D setup | `create_workflow_template(template_type="3d_simple")` |
| Organizing the node graph | `organize_node_graph()` |
| Auto-arranging selected nodes | `auto_layout_nodes(selected_only=True)` |
| Checking node parameters | `get_node_info()` |