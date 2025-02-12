"""Command line argument completion definitions."""

import argparse
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

from freva_client import databrowser
from freva_client.cli.cli_app import app

from .cli_utils import parse_cli_args


class Completer:
    """Base class for command line argument completers."""

    def __init__(
        self,
        argv: List[str],
        choices: Optional[Dict[str, Tuple[str, str]]] = None,
        shell: str = "bash",
        strip: bool = False,
        flags_only: bool = False,
    ):
        self.choices = choices or {}
        self.strip = strip
        self.argv = argv
        self.flags_only = flags_only
        if shell == "zsh":
            self.get_print = self._print_zsh
        elif shell == "fish":
            self.get_print = self._print_fish
        else:
            self.get_print = self._print_default

    def _print_zsh(self, choices: Dict[str, Tuple[str, str]]) -> List[str]:
        out = []
        for key, _help in choices.items():
            if key.startswith("-"):
                out.append(f"{key}[{_help}]")
            else:
                out.append(f"{key}: {_help}")
        return out

    def _print_fish(self, choices: Dict[str, Tuple[str, str]]) -> List[str]:
        out = []
        for key, _help in choices.items():
            out.append(f"{key}: {_help}")
        return out

    def _print_default(self, choices: Dict[str, Tuple[str, str]]) -> List[str]:
        out = []
        for key, _help in choices.items():
            if not key.startswith("-"):
                out.append(f"{key}: {_help}")
            else:
                out.append(key)
        return out

    def _get_choices(self) -> Dict[str, Tuple[str, str]]:
        """Get the choices for databrowser command."""

        facet_args = []
        for arg in self.argv:
            if len(arg.split("=")) == 2:
                facet_args.append(arg)
        search_keys = parse_cli_args(facet_args)
        search = databrowser.metadata_search(
            flavour="freva",
            time=None,
            host=None,
            time_select="flexible",
            bbox=None,
            bbox_select="flexible",
            multiversion=False,
            extended_search=True,
            fail_on_error=False,
            **search_keys,
        )
        choices = {}
        for att, values in search.items():
            if att not in search_keys:
                keys = ",".join([v for n, v in enumerate(values)])
                choices[att] = (keys, "")
        return choices

    @property
    def command_choices(self) -> Dict[str, Tuple[str, str]]:
        """Get the command line arguments for all sub commands."""

        if self.flags_only:
            return self.choices
        choices = self._get_choices()
        return {**self.choices, **choices}

    def formatted_print(self) -> None:
        """Print all choices to be processed by the shell completion function."""

        out = self.get_print(self.command_choices)
        for line in out:
            if line.startswith("-") and self.strip and not self.flags_only:
                continue
            if not line.startswith("-") and self.flags_only:
                continue
            print(line)

    @classmethod
    def parse_choices(cls, argv: list[str]) -> "Completer":
        """Create the completion choices from given cmd arguments."""
        parser = argparse.ArgumentParser(
            description="Get choices for command line arguments"
        )
        parser.add_argument(
            "--strip",
            help="Do not print options starting with -",
            default=False,
            action="store_true",
        )
        parser.add_argument(
            "--shell",
            help="Set the target shell type.",
            default=False,
        )
        parser.add_argument(
            "--flags-only",
            help="Only print options starting with -",
            default=False,
            action="store_true",
        )
        cli_app, args = parser.parse_known_args(argv)
        main_choices = {c.name: (c, c.help) for c in app.registered_commands}
        choices = {}
        if args and args[0] == "freva-databrowser":
            _ = args.pop(0)
        if args[0] in main_choices:
            signature = inspect.signature(
                cast(Callable[..., Any], main_choices[args[0]][0].callback)
            )
            for param, value in signature.parameters.items():
                if param in args or param == "search_keys":
                    continue
                if hasattr(value.annotation, "__metadata__"):
                    option = value.annotation.__metadata__
                else:
                    option = value.default
                choices[option.param_decls[-1]] = option.help
        else:
            choices = {k: v[1] for k, v in main_choices.items()}
        return cls(
            args,
            choices,
            shell=cli_app.shell,
            strip=cli_app.strip,
            flags_only=cli_app.flags_only,
        )
