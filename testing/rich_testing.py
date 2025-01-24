import json
from rich.console import Console
from rich.table import Table
from rich.json import JSON

# Initialize Rich Console
console = Console()

# Example structured data
data = {
    "name": "Alice",
    "age": 28,
    "is_active": True,
    "roles": ["editor", "contributor"],
    "stats": {
        "articles": 42,
        "followers": 1500,
        "following": 200,
    },
}

# 1. Standard Print vs Rich Text
print("=== Standard Print ===")
print("Hello, world!")

console.print("\n=== Rich Text ===")
console.print("[bold green]Hello, world![/bold green]")

# 2. Standard Print for Key-Value Data vs Rich Table
print("\n=== Standard Key-Value ===")
for key, value in data.items():
    print(f"{key}: {value}")

console.print("\n=== Rich Table ===")
table = Table(title="User Data", show_header=True, header_style="bold cyan")
table.add_column("Key", style="dim")
table.add_column("Value")
for key, value in data.items():
    table.add_row(key, str(value))
console.print(table)

# 3. Standard JSON Dump vs Rich JSON
print("\n=== Standard JSON ===")
print(json.dumps(data, indent=4))

console.print("\n=== Rich JSON ===")
console.print(JSON(json.dumps(data)))

# 4. Standard Progress Bar vs Rich Progress (Demo)
import time
from rich.progress import track

print("\n=== Standard Progress Bar (Simulated) ===")
for i in range(5):
    print(f"Processing step {i + 1} / 5...")
    time.sleep(0.5)

console.print("\n=== Rich Progress Bar ===")
for step in track(range(5), description="Processing..."):
    time.sleep(0.5)
