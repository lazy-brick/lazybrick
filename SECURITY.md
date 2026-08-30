# Security policy

lazybrick is pre-alpha. Security fixes are applied to the current `main` branch
and the latest published release when practical; older pre-alpha releases may
not receive backports.

## Reporting a vulnerability

Do not open a public issue with exploit details, secrets, private artifacts, or
affected-user data.

Use the repository's private **Report a vulnerability** flow under GitHub's
Security tab when it is available. If it is unavailable, contact a maintainer
through their GitHub profile and request a private reporting channel without
including sensitive details in the initial message.

Include the affected version or commit, impact, prerequisites, minimal
reproduction, and any suggested mitigation. Remove credentials and proprietary
data. We do not promise a response or remediation deadline while the project is
pre-alpha, but we will avoid public disclosure until a reasonable fix and
coordination plan exists.

## In scope

Examples include mutable or spoofed provenance, unsafe deserialization, command
or path injection, artifact or evidence tampering, secret leakage, dependency
or plugin execution risks, remote-code bypasses, and unsafe handling of model or
dataset inputs.

Plugin subprocesses are not a filesystem, network, or GPU sandbox. Running an
untrusted plugin is therefore a known trust decision, not a vulnerability by
itself; a bypass of a documented restriction or a misleading safety claim is in
scope.
