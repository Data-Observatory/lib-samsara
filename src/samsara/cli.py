"""
Module that contains the command line app.

Why does this file exist, and why not put this in __main__?

  You might be tempted to import things from __main__ later, but that will cause
  problems: the code will get executed twice:

  - When you run `python -msamsara` python will execute
    ``__main__.py`` as a script. That means there will not be any
    ``samsara.__main__`` in ``sys.modules``.
  - When you import __main__ it will get executed again (as a module) because
    there"s no ``samsara.__main__`` in ``sys.modules``.

  Also see (1) from http://click.pocoo.org/5/setuptools/#setuptools-integration
"""

import argparse

parser = argparse.ArgumentParser(description="Not implemented, still in development.")
parser.add_argument(
    "names",
    metavar="NAME",
    nargs=argparse.ZERO_OR_MORE,
    help="A name of something.",
)


def main(args=None):
    args = parser.parse_args(args=args)
    print(args.names)
