# Nuke-MCP

A bridge between The Foundry's Nuke and AI systems using the Model Context Protocol (MCP).

![Nuke MCP](mcp.PNG)

## Overview

Nuke-MCP allows AI assistants to interact with Nuke through a socket connection, enabling them to:

- Get information about Nuke scripts
- Create, modify, and delete nodes
- Position nodes in the node graph
- Connect nodes together
- Control playback and rendering
- Execute arbitrary Python code in Nuke

## Components

1. **Nuke Addon** (`nuke_mcp_addon.py`): A Nuke script that creates a socket server within Nuke
2. **MCP Server** (`nuke_mcp_server.py`): A Python server that connects to the Nuke addon and exposes tools to AI systems
3. **Entry Point** (`main.py`): A simple script to start the MCP server

## Installation

### Prerequisites

- The Foundry's Nuke (any recent version should work)
- Python 3.7+
- FastMCP package

### Installing the FastMCP package

```bash
pip install fastmcp
```

### Installing Nuke-MCP

1. Clone or download this repository
2. Copy `nuke_mcp_addon.py` to your Nuke scripts folder or a location in your Nuke Python path

## Usage

### Installing the Nuke Addon

1. **Copy the Addon File**:
   - Take the `nuke_mcp_addon.py` file and place it in a location where Nuke can find it:
     - Copy it to your Nuke scripts folder (usually in `~/.nuke` on Linux/Mac or in your home directory on Windows)
     - Or place it in a folder that's in your Nuke Python path

2. **Create a Startup Script** (Recommended):
   - Create or edit an existing `init.py` file in your `.nuke` directory
   - Add the following line to automatically load the addon when Nuke starts:
     ```python
     import nuke_mcp_addon
     ```

3. **Manual Loading** (Alternative):
   - If you don't want to load it automatically, you can manually load it each time by:
     - Opening Nuke
     - Going to the Script Editor panel
     - Running `import nuke_mcp_addon`

### Docking the NukeMCP Panel

By default, the NukeMCP panel opens as a floating window. If you prefer to have it docked in Nuke's interface, you can modify the `NukeMCPPanel` class in `nuke_mcp_addon.py`:

```python
# Find the NukeMCPPanel class definition (around line 380)
class NukeMCPPanel(nukescripts.PythonPanel):
    def __init__(self):
        nukescripts.PythonPanel.__init__(self, 'Nuke MCP', 'com.example.NukeMCP')
        # ... existing code ...

# Add this method to enable docking
    def addToPane(self):
        pane = nuke.getPaneFor('Properties.1')
        if not pane:
            pane = nuke.getPaneFor('Viewer.1')
        self.setMinimumSize(300, 200)  # Set a reasonable minimum size
        return pane.addPermanentAsQWidget(self)

# Modify the show_panel function to use docking
def show_panel():
    """Show the NukeMCP panel"""
    global _panel
    if _panel is None:
        _panel = NukeMCPPanel()
    
    # Show as docked panel instead of floating window
    pane = _panel.addToPane()
    if pane:
        _panel.setParent(pane)
```

You can also modify how the panel appears in the menu. Based on your menu.py file, you have:

```python
# MCP Tools
nuke.toolbar("Nodes").addCommand('DUGCO/MCP/NukeMCP Panel', 'nuke_mcp_addon.show_panel()')
```

To ensure it's properly integrated with your existing toolbar structure, make sure this line is uncommented in your menu.py file.

## Troubleshooting

To use Nuke-MCP with Claude Desktop, follow these steps:

1. **Download and Install Claude Desktop**:
   - Download Claude Desktop from Anthropic's website
   - Install and set up your account

2. **Enable Developer Mode**:
   - Open Claude Desktop
   - Go to Settings
   - Enable Developer Mode (this might require specific permissions)

3. **Edit the Configuration File**:
   - In Claude Desktop settings, click "Edit Config" (or directly edit the `claude_desktop_config.json` file)
   - Add the Nuke-MCP server configuration:

```json
{
  "mcpServers": {
    "nuke": {
      "command": "python",
      "args": [
        "/path/to/your/nuke-mcp/main.py"
      ],
      "trusted": true
    }
  }
}
```

4. **Replace the Path**:
   - Update the path with the actual path to where you saved the `main.py` file
   - Make sure to use the full absolute path
   - For Windows, use either forward slashes or escaped backslashes:
     ```
     "args": ["Z:/path/to/nuke_mcp/main.py"]
     ```
     or
     ```
     "args": ["Z:\\path\\to\\nuke_mcp\\main.py"]
     ```
   - Note: The `"trusted": true` flag is required for full functionality

5. **Restart Claude Desktop**:
   - Save the configuration file
   - Restart Claude Desktop to apply the changes

Now when you use Claude Desktop, you can instruct it to interact with Nuke through the MCP connection.

## System Architecture

The system consists of three main components that work together:

1. **Nuke Addon**: Runs inside Nuke and creates a socket server that listens for commands
2. **MCP Server**: Acts as middleware between AI systems and Nuke
3. **AI Client**: Connects to the MCP Server to control Nuke (e.g., Claude Desktop)

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│             │     │             │     │             │
│  AI Client  │◄───►│  MCP Server │◄───►│  Nuke Addon │
│ (Claude)    │     │             │     │ (in Nuke)   │
│             │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
```

## Available Tools

The MCP server exposes the following tools:

- `get_script_info()`: Get information about the current Nuke script
- `get_node_info(node_name)`: Get details about a specific node
- `create_node(node_type, name, position, inputs, parameters)`: Create a new node
- `modify_node(name, parameters, position, inputs)`: Change an existing node
- `delete_node(name)`: Delete a node
- `position_node(name, x, y)`: Position a node in the node graph
- `connect_nodes(output_node, input_node, input_index)`: Connect nodes
- `render(frame_range, write_node, proxy_mode)`: Render frames
- `viewer_playback(action, start_frame, end_frame)`: Control Viewer playback
- `execute_nuke_code(code)`: Run Python code in Nuke
- `auto_layout_nodes(selected_only)`: Automatically arrange nodes
- `set_frames(first_frame, last_frame, current_frame)`: Set frame range
- `create_viewer(input_node)`: Create a Viewer node

## Examples

### Creating a Simple Blur Node

```
create_node(node_type="Blur", parameters={"size": 10})
```

### Building a Grade Node Chain

```
create_node(node_type="Read", name="input", parameters={"file": "/path/to/image.exr"})
create_node(node_type="Grade", name="grade1", position=[200, 0])
create_node(node_type="Grade", name="grade2", position=[400, 0])
create_node(node_type="Write", name="output", position=[600, 0], parameters={"file": "/path/to/output.exr"})
connect_nodes(output_node="input", input_node="grade1")
connect_nodes(output_node="grade1", input_node="grade2")
connect_nodes(output_node="grade2", input_node="output")
```

### Rendering Output

```
render(frame_range="1-10", write_node="output")
```

## Example Commands for Claude

When working with Claude Desktop, you can give it natural language instructions to control Nuke. Here are some example commands:

### Basic Script Analysis
```
"Can you tell me what's in my current Nuke script?"
```

### Creating a Complete Compositing Setup
```
"Create a compositing setup in Nuke with a Read node, Color Correct, Grade, and Write node, all properly connected."
```

### Building a Green Screen Keying Setup
```
"Create a professional green screen keying setup in Nuke with Keyer, Despill, Edge Blur, and compositing over a background."
```

### Adding Special Effects
```
"Build a particle system in Nuke using Noise, ParticleEmitter, and Merge nodes to create a fire effect."
```

### Enhancing an Existing Script
```
"Analyze my current Nuke script, then enhance it by creating a color grading chain with separate adjustments for shadows, midtones, and highlights. Add a subtle film grain effect and a gentle vignette."
```

### 3D Scene Setup
```
"Set up a 3D scene in Nuke with a Camera, 3D objects, and proper lighting."
```

## Extending the MCP Tools

You can extend the system by adding new tools to the `nuke_mcp_server.py` file. The MCP tools are defined using the `@mcp.tool()` decorator pattern.

### Tool Structure

Each tool follows this basic structure:

```python
@mcp.tool()
def tool_name(ctx: Context, param1: type, param2: type = default_value) -> str:
    """
    Tool description for documentation.
    
    Parameters:
    - param1: Description of param1
    - param2: Description of param2 with default value
    """
    try:
        logger.info(f"Tool called: tool_name with params")
        nuke = get_nuke_connection()
        result = nuke.send_command("command_name", {
            "param1": param1,
            "param2": param2
        })
        
        # Process result and format response
        return "Human-readable response"
    except Exception as e:
        logger.error(f"Error in tool_name: {str(e)}")
        return f"Error message: {str(e)}"
```

### Adding a New Tool

To add a new tool:

1. **Identify the functionality** you want to add
2. **Add a new method in the Nuke addon** (in `nuke_mcp_addon.py`):
   - Add the method to the `NukeMCPServer` class
   - Add an entry to the `handlers` dictionary in the `execute_command` method

3. **Create a corresponding tool** in `nuke_mcp_server.py`:
   - Use the `@mcp.tool()` decorator
   - Define parameters with type hints
   - Add a descriptive docstring
   - Implement the logic to call the corresponding Nuke command

4. **Test the new tool** thoroughly

### Example: Adding a Transform Node Tool

Here's an example of how you might add a specialized tool for creating a Transform node:

```python
@mcp.tool()
def create_transform(
    ctx: Context,
    name: str = None,
    position: List[int] = None,
    rotation: float = 0.0,
    scale: float = 1.0,
    center: List[float] = None
) -> str:
    """
    Create a Transform node with specified parameters.
    
    Parameters:
    - name: Optional name for the Transform node
    - position: Optional [x, y] position coordinates for the node
    - rotation: Rotation in degrees
    - scale: Uniform scale factor
    - center: Optional [x, y] center of transformation
    """
    try:
        logger.info(f"Tool called: create_transform")
        nuke = get_nuke_connection()
        
        # Prepare parameters for the Transform node
        parameters = {
            "rotate": rotation,
            "scale": scale
        }
        
        if center:
            parameters["center"] = center
        
        # Call the generic create_node command
        result = nuke.send_command("create_node", {
            "node_type": "Transform",
            "name": name,
            "position": position,
            "parameters": parameters
        })
        
        actual_name = result.get("name", "unknown")
        return f"Created Transform node named '{actual_name}' with rotation={rotation}° and scale={scale}"
    except Exception as e:
        logger.error(f"Error in create_transform: {str(e)}")
        return f"Error creating Transform node: {str(e)}"
```

This pattern allows you to build specialized tools that provide higher-level functionality while leveraging the existing command infrastructure.

## Troubleshooting

### Connection Issues

If Claude cannot connect to Nuke, check the following:

1. **Nuke is Running**: Make sure Nuke is open and running
2. **Addon is Active**: Verify the NukeMCP panel shows "Running on port 9876"
3. **Port Configuration**: Ensure the port matches in both Nuke and the MCP server
4. **Firewall Settings**: Check if your firewall is blocking the connection

### Common Errors

- **"Could not connect to Nuke"**: Make sure the Nuke addon is running and using the correct port
- **"Socket timeout while waiting for response"**: The operation in Nuke may be taking too long
- **"Failed to create node"**: Check if the node type name is correct

## Development

This is an initial implementation. Future improvements could include:

- Support for more specialized Nuke functionality (Roto, Tracking, etc.)
- Better error handling and timeout management
- Support for template-based workflows
- Integration with additional asset management systems

## License

[MIT License](LICENSE)
