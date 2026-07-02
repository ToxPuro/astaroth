# ⚠️ WARNING ⚠️
# **This is a LLM based analysis.**
# **THE BRANCH SHOULD NEVER BE MERGED TO master OR develop BRANCHES!**

# LLM Information
- **Service:** Oulu University Lehmus AI
- **Model:** Gemma4 26B MoE

# Overview
The `ac-interpreter` is a command-line interface (CLI) utility designed for interactive testing and debugging of the Astaroth library. It provides a REPL-like environment (Read-Eval-Print Loop) that allows users to manually trigger core library functions such as mesh initialization, kernel execution, data I/O, and field reductions.

# Directory Structure & File Descriptions

- `CMakeLists.txt`: Build configuration file used to compile the interpreter.
- `main.c`: The core implementation of the command-line interface, containing the main command loop and the logic for handling various interactive commands.

# Command Reference

The interpreter processes the following commands via standard input:

| Command | Arguments | Description |
| :--- | :--- | :--- |
| `init` | `nx ny nz` | Initializes the mesh dimensions and creates the host mesh and device context. |
| `load_random` | None | Initializes the device mesh with randomized data and applies periodic boundary conditions. |
| `read` | None | Loads a mesh from a file into the device. |
| `write` | None | Saves the current device mesh to a file. |
| `launch` | `<kernel_name>` | Launches a specified Astaroth kernel on the device. |
| `reduce` | `<reduction_type>` | Performs a specific scaling reduction (e.g., `RTYPE_MAX`, `RTYPE_MIN`) on the device fields. |
| `reduce_all` | None | Performs both `RTYPE_MAX` and `RTYPE_MIN` reductions across all fields. |
| `exit` / `quit` | None | Safely destroys the mesh and device context and exits the program. |

# Key Dependencies
- `astaroth.h`: Core Astaroth library definitions and function prototypes.
- `astaroth_utils.h`: Utility functions and macros used within the interpreter.
