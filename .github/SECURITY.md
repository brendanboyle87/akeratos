# Security policy

## Reporting

Report a vulnerability privately: open this repository's **Security** tab and choose **Report a vulnerability**. Do not open a public issue, pull request or discussion about it.

Say what you found, how to reproduce it, the commit or release you tested, and what an attacker could do with it.

I am the only maintainer, and nobody else watches the reports.

| Step | Target |
|---|---|
| I acknowledge your report | 3 working days |
| I tell you whether I accept it | 10 working days |
| I update you while a fix is in progress | every 10 working days |

I fix an accepted report in a private advisory, credit you unless you ask me not to, and publish the advisory with the fix.

## Supported versions

The latest release on PyPI and the `main` branch. Fixes ship in a new release.

## Scope

In scope: code that reads untrusted files (checkpoints, safetensors files, exported models, training data); the GitHub Actions workflows; the integrity of what I publish on PyPI and Hugging Face.

Out of scope: what a model writes (open an ordinary issue); flaws in MLX or another upstream project, which go to that project; scanner output with no shown impact; modified forks.

## Checking a download

Every model folder on Hugging Face holds a `SHA256SUMS` file. In the folder, run `shasum -a 256 -c SHA256SUMS`.
