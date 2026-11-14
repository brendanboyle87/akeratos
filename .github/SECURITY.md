# Security policy

## Reporting

Report a vulnerability privately: open this repository's **Security** tab and choose **Report a vulnerability**. Do not open a public issue, pull request or discussion about it.

Say what you found, how to reproduce it, the commit or release you tested, and what an attacker could do with it.

Only Cogvera Labs sees a report.

| Step | Target |
|---|---|
| Acknowledgement of your report | 3 working days |
| A decision to accept it or not | 10 working days |
| Updates while a fix is in progress | every 10 working days |

Cogvera Labs fixes an accepted report in a private advisory, credits you unless you ask otherwise, and publishes the advisory with the fix.

## Supported versions

The latest release on PyPI and the `main` branch. Fixes ship in a new release.

## Scope

In scope: code that reads untrusted files (checkpoints, safetensors files, exported models, training data); the GitHub Actions workflows; the integrity of what Cogvera Labs publishes on PyPI and Hugging Face; a released model that repeats personal or private text.

Out of scope: anything else a model writes (open an ordinary issue); flaws in MLX or another upstream project, which go to that project; scanner output with no shown impact; modified forks.

## Checking a download

Every model folder on Hugging Face holds a `SHA256SUMS` file. In the folder, run `shasum -a 256 -c SHA256SUMS`.
