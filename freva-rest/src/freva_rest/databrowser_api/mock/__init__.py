"""Read test entries from a databrowser."""

import asyncio
from pathlib import Path

import httpx


async def read_data(core: str, uri: str) -> None:
    """Read mock databrowser data."""
    datapath = Path(__file__).parent.absolute()
    data_json = (datapath / f"{core}.json").read_text().replace("$CWD", str(datapath))
    url = f"{uri}/solr/{core}/update/json?commit=true"
    headers = {"content-type": "application/json"}
    async with httpx.AsyncClient(timeout=5.0) as client:
        res = await client.post(url, content=data_json, headers=headers)
        try:
            res.raise_for_status()
        except httpx.HTTPStatusError as e:  # pragma: no cover
            if e.response.status_code == 503:
                await asyncio.sleep(1)
            else:
                raise
