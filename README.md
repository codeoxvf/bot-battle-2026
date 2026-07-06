# agario-competitor

Starter repo for a competition bot.

## First run
1. [Install UV](https://docs.astral.sh/uv/getting-started/installation/)

2. Run these from this folder
```bash
uv sync --upgrade
uv run interactive 7:bots/my_bot.py
```

This template installs the published `agario-kit` package from PyPI. The local
interactive launcher expects `count:path` specs whose counts sum to `n - 1`.
For the current 8-player game, that means the counts must sum to `7`.

To play manually against example bots instead, run:

```bash
uv run interactive 2:bots/my_bot.py 5:bots/other_bot.py
```

To watch a non-interactive simulation, run:

```bash
uv run simulation 8:bots/my_bot.py
```

## Writing a bot

- Put your bot logic in `bots/my_bot.py`.
- Import `Game` from `helper.game`.
- Read visible state from `game.state`.
- Return moves using the `lib.interface.events.moves` models.

## Updating during the competition

We may make changes to the game engine during the event. When a new platform version is published, please run this command to bring your version of the game engine up to date:

```bash
uv sync --upgrade
```

We will send a message on [Discord](https://discord.gg/24We3YWM7e) if this happens.