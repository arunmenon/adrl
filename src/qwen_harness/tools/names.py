"""Canonical tool name strings. Port of tools/tool-names.ts.

The system prompt interpolates these rather than hard-coding, so renames
stay consistent between prompt text and tool declarations. Upstream also
migrates legacy names (search_file_content -> grep_search, replace -> edit,
task -> agent) when loading old sessions.
"""


class ToolNames:
    LS = "list_directory"
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    EDIT = "edit"
    GLOB = "glob"
    GREP = "grep_search"
    SHELL = "run_shell_command"
    MEMORY = "save_memory"
    READ_MANY_FILES = "read_many_files"
    TODO_WRITE = "todo_write"
    WEB_FETCH = "web_fetch"
    AGENT = "agent"
