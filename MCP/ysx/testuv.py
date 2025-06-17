import asyncio
import json
import subprocess
from typing import Dict, Any

async def call_mcp_time_server(method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    proc = subprocess.Popen(
        [r"/Users/bytedance/.nvm/versions/node/v22.16.0/bin/npx", "-y", "mcp-time-server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1 
    )

    request = {
        "jsonrpc": "2.0",
        "method": method,
        "params":params,
        "id": 1   
    }
    print(request)
    proc.stdin.write(json.dumps(request) + "\n")
    proc.stdin.flush()

    line1=proc.stdout.readline()
    print("Raw response:", line1)#注意它会先输出一条垃圾日志
    line2=proc.stdout.readline()
    print("Raw response:", line2)
    return json.loads(line2)

async def main():
    try:
        response = await call_mcp_time_server(
            method="tools/call",
            params={
            "name": "get_current_time",
            "arguments": {"timezone": "America/New_York"}
        }
        )
        print("Parsed response:", response)
    except Exception as e:
        print("Error:", e)

asyncio.run(main())