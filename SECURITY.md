# Security

## Reporting a vulnerability

Please report it privately through GitHub's "Report a vulnerability" on the Security tab of the repository (private vulnerability reporting), rather than a public issue. Include what you found, how to reproduce it, and the version.

You can expect an acknowledgement within a few days. This is a small project: there is no bug bounty and no guaranteed timeline, but reports are taken seriously and fixed first.

## What counts

The areas most worth attention: the local server's access rules (host, origin, token), the path jail and command policy, the scratch-copy runner's environment handling, applying and rolling back changes, the desktop window's isolation, and parsing of model files.

## What the project claims, and does not

[docs/SECURITY_MODEL.md](docs/SECURITY_MODEL.md) describes the protections. [docs/LIMITS.md](docs/LIMITS.md) lists what is **not** protected, notably what the command sandbox does and does not cover (it is an OS-level seatbelt on macOS and, with bubblewrap, Linux, not a vault against code you approve), and that nobody outside the project has reviewed the code.

## Supported versions

Pre-1.0: fixes go into the latest release only.
