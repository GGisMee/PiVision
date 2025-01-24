from rich.panel import Panel
from rich.console import Console
console = Console()

panel = Panel("[bold green]Success![/bold green] Operation completed.")
console.print(panel)

from rich.status import Status
import time

with console.status("[bold green]Loading...[/bold green]"):
    time.sleep(2)
console.print("[green]Done![/green]")


from pprint import pprint

data = {
    "name": "Alice",
    "details": {
        "age": 25,
        "city": "New York",
        "hobbies": ["reading", "biking", "coding"]
    }
}
pprint(data, width=40)

from pprint import pprint

data = {
    "name": "Alice",
    "details": {
        "age": 25,
        "city": "New York",
        "hobbies": ["reading", "biking", "coding"]
    }
}
pprint(data, width=40)

from rich.markdown import Markdown

markdown = Markdown("# This is a title\n\n- Item 1\n- Item 2")
console.print(markdown)
