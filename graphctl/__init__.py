"""graphctl -- deterministic controller for the graph loop.

The agent never has to remember where it is: ``python -m graphctl next`` prints
the briefing for the next node, ``verify`` runs that node's machine checks,
``done`` advances, ``jobs`` manages long-running background processes and the
Cursor ``stop`` hook calls ``followup`` to keep the loop going without a human.
"""
__version__ = "1.0.0"
