"""Test for adding tools."""

import os
import time
from typing import Dict, Tuple

import mock
import requests


def test_add_the_tool_success(
    subporcess_mocker: Tuple[str, Dict[str, Dict[str, str]]]
) -> None:
    """Test what happens if you add a tool."""

    test_server, tokens = subporcess_mocker
    url = f"{test_server}/tool"

    # Let's add a usertool
    res = requests.post(
        f"{url}/add",
        headers=tokens["johndoe"],
        json={"tool-path": "examples/gnuR-script/tool.toml"},
        stream=True,
        timeout=5,
    )
    assert res.status_code == 200
    errors = [
        line for line in res.iter_lines() if "error" in line.decode().lower()
    ]
    assert not errors

    # Let's do this again and make sure the system complains.
    res = requests.post(
        f"{url}/add",
        headers=tokens["johndoe"],
        json={"tool-path": "examples/gnuR-script", "force": True},
        stream=True,
        timeout=5,
    )
    assert res.status_code >= 400 and res.status_code < 500

    # Check if alice doesn't see johns tool
    res = requests.get(f"{url}/overview", headers=tokens["alicebrown"], timeout=5)
    assert res.status_code == 200
    assert len(res.json()) == 0

    # Since jane is admin everyone will see the tool if she adds one
    res = requests.post(
        f"{url}/add",
        headers=tokens["janedoe"],
        json={"tool-path": "examples/gnuR-script", "force": True},
        stream=True,
        timeout=5,
    )
    assert res.status_code == 200
    errors = [
        line for line in res.iter_lines() if "error" in line.decode().lower()
    ]
    assert not errors
    time.sleep(2)
    # Check if alice can see the tool now
    res = requests.get(
        f"{url}/overview",
        headers=tokens["alicebrown"],
        timeout=5,
    )
    assert res.status_code == 200
    assert len(res.json()) > 0


def test_add_the_tool_failure(
    subporcess_mocker: Tuple[str, Dict[str, Dict[str, str]]]
) -> None:
    """Test if what happens if tools are no good for adding."""

    test_server, tokens = subporcess_mocker
    url = f"{test_server}/tool"
    res = requests.post(
        f"{url}/add",
        headers=tokens["johndoe"],
        json={"tool-path": "examples/fix-me/tool-1.toml"},
        stream=True,
        timeout=5,
    )
    assert res.status_code == 422
    res = requests.post(
        f"{url}/add",
        headers=tokens["johndoe"],
        json={"tool-path": "examples/fix-me/tool-2.toml"},
        stream=True,
        timeout=5,
    )
    assert res.status_code == 422
    res = requests.post(
        f"{url}/add",
        headers=tokens["johndoe"],
        json={"tool-path": "examples/fix-me/tool-3.toml"},
        stream=True,
        timeout=5,
    )
    assert res.status_code == 500

    res = requests.post(
        f"{url}/add",
        headers=tokens["johndoe"],
        json={"tool-path": "examples/fix-me/tool-4.toml"},
        stream=True,
        timeout=5,
    )
    assert res.status_code == 422

    res = requests.post(
        f"{url}/add",
        headers=tokens["johndoe"],
        json={"tool-path": "foo/tool-1.toml"},
        stream=True,
        timeout=5,
    )
    assert res.status_code == 500
    res = requests.post(
        f"{url}/add",
        headers=tokens["johndoe"],
        json={"url": "http://foo.git"},
        stream=True,
        timeout=5,
    )
    assert res.status_code == 422
    env = os.environ.copy()
    env["MOCK_RETURN_CODE"] = "1"
    with mock.patch.dict(os.environ, env, clear=True):
        # Let's add a usertool
        res = requests.post(
            f"{url}/add",
            headers=tokens["johndoe"],
            json={"url": "http://foo.git", "tool-name": "bar"},
            stream=True,
            timeout=5,
        )
        assert res.status_code == 500

        res = requests.post(
            f"{url}/add",
            headers=tokens["johndoe"],
            json={"tool-path": "examples/rust-build"},
            stream=True,
            timeout=5,
        )
        assert res.status_code == 500
