# Contributing

Fixes, tests, documentation, speedrun claims and bench rows are welcome. Fork, change, open a pull request. I read every one. Questions go in Discussions.

## Set up and test

```sh
uv sync            # on Linux: uv sync --extra cpu
uv run pytest
uv run ruff check .
uv run ruff format .
```

Add or change a test that fails without your fix. On Linux MLX runs on the CPU. That is enough for everything but speed and the Metal kernels.

## What CI checks

CI runs lint, the tests, a package build and a docs build, on Ubuntu, on the CPU. It never runs the Metal kernels. If you change them, say so in the pull request, and I check them on a Mac before I merge.

## How my own changes arrive

I develop in a private repository, where AI coding agents write most of the code and their checks run. Each change is exported here as one commit. When I merge your pull request here, the same commits are applied there with your name on them.

## A change to the model or its training

Write the test down before you run it: the metric, the baseline and the margin. Billet's twin is the baseline, and [Reproduce](docs/reproduce.md) has its command. Train both arms on the same tokens, on a Mac, and send the logs with the pull request. A result that fails its test is worth sending too.

For speedrun claims and bench rows, [the speedrun page](docs/speedrun.md) says what to send.

## Rules

- **Licence.** Apache-2.0 in, Apache-2.0 out. There is no CLA. By opening a pull request you confirm you may contribute it. Copied code needs a compatible licence and a `# SOURCE:` line.
- **AI assistance.** Welcome. Say how much and which tool. You have read every line, and you answer for it.
- **Data.** Only permissively licensed or public-domain data. No personal data.
- **Conduct.** The [code of conduct](.github/CODE_OF_CONDUCT.md) applies.
- **Security.** Never in a public issue: see [SECURITY.md](.github/SECURITY.md).
- **Agents.** A coding agent should read [AGENTS.md](AGENTS.md) first.
