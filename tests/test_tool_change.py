"""Test changing the tool."""

from typing import Dict, Tuple

import requests


def test_change_tool(
    subporcess_mocker: Tuple[str, Dict[str, Dict[str, str]]]
) -> None:
    """Test what happens if we want to change a tool."""

    test_server, tokens = subporcess_mocker
    url = f"{test_server}/tool"
    # Add a tool we haven't added so far.
    res = requests.post(
        f"{url}/add",
        headers=tokens["johndoe"],
        json={"tool-path": "examples/rust-build"},
        stream=True,
        timeout=5,
    )
    # Wait for the tool to be added.
    _ = list(res.iter_lines())
    assert res.status_code == 200
    res = requests.get(
        f"{url}/overview",
        headers=tokens["johndoe"],
        timeout=5,
    )

    assert res.status_code == 200
    assert "example-rust-build" in res.json().keys()

    res = requests.post(
        f"{url}/change",
        json={"tool": "example-rust-build"},
        timeout=5,
        headers=tokens["johndoe"],
    )
    assert res.status_code == 422

    res = requests.post(
        f"{url}/change",
        headers=tokens["johndoe"],
        json={
            "tool": "example-rust-build",
            "versions": ["0.0.2"],
            "visible": False,
        },
        timeout=5,
    )
    assert res.status_code == 200
    res = requests.get(
        f"{url}/overview",
        headers=tokens["johndoe"],
        timeout=5,
    )
    assert res.status_code == 200
    assert "example-rust-build" not in res.json().keys()
    res = requests.get(
        f"{url}/overview",
        headers=tokens["johndoe"],
        timeout=5,
        params={"display-all": True},
    )
    assert res.status_code == 200
    assert "example-rust-build" in res.json().keys()

    res = requests.post(
        f"{url}/change",
        json={
            "tool": "example-rust-build",
            "versions": ["0.0.2"],
            "visible": True,
        },
        headers=tokens["johndoe"],
        timeout=5,
    )
    assert res.status_code == 200
    res = requests.get(
        f"{url}/overview",
        headers=tokens["johndoe"],
        timeout=5,
    )
    assert res.status_code == 200
    assert "example-rust-build" in res.json().keys()

    res = requests.post(
        f"{url}/change",
        json={"tool": "example-rust-build", "make-global": True},
        headers=tokens["johndoe"],
        timeout=5,
    )
    assert res.status_code == 200
    res = requests.get(
        f"{url}/overview",
        headers=tokens["alicebrown"],
        timeout=5,
    )
    assert res.status_code == 200
    assert "example-rust-build" in res.json().keys()
    res = requests.post(
        f"{url}/change",
        json={"tool": "example-rust-build", "visible": False},
        headers=tokens["janedoe"],
        timeout=5,
    )
    assert res.status_code == 200
    res = requests.get(
        f"{url}/overview",
        headers=tokens["alicebrown"],
        timeout=5,
    )
    assert res.status_code == 200
    assert "example-rust-build" not in res.json().keys()
