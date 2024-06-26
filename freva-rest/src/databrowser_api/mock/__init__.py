"""Read test entries from a databrowser."""

from pathlib import Path

import aiohttp


async def read_data(core: str, hostname: str = "localhost", port: str = "8983") -> None:
    """Read mock databrowser data."""
    datapath = Path(__file__).parent.absolute()
    data_json = (datapath / f"{core}.json").read_text().replace("$CWD", str(datapath))
    url = f"http://{hostname}:{port}/solr/{core}/update/json?commit=true"
    headers = {"content-type": "application/json"}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
        async with session.post(url, data=data_json, headers=headers) as res:
            _ = await res.text()
