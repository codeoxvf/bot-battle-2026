# agario-competitor

Starter repo for a competition bot.

## First run

```bash
uv sync
uv run interactive 3:bots/my_bot.py
```

This template installs the published `agario-kit` package from PyPI. The local
interactive launcher expects `count:path` specs whose counts sum to `n - 1`.
For the current 4-player game, that means the counts must sum to `3`.

To play manually against example bots instead, run:

```bash
uv run interactive 2:bots/my_bot.py 1:bots/other_bot.py
```

To watch a non-interactive simulation, run:

```bash
uv run simulation 4:bots/my_bot.py
```

## Writing a bot

- Put your bot logic in `bots/my_bot.py`.
- Import `Game` from `helper.game`.
- Read visible state from `game.state`.
- Return moves using the `lib.interface.events.moves` models.

## Updating during the competition

When the organisers publish a new platform version, update with:

```bash
uv lock --upgrade-package agario-kit==2026.1.1
uv sync
```
