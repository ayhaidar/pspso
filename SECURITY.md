# Security Policy

## Supported versions

Security fixes are currently applied to the active version 1.0 development code.
After publication, fixes will target the latest `1.0.x` release. Older
development snapshots are not supported.

## Reporting a vulnerability

Please report suspected vulnerabilities privately through
[GitHub Security Advisories](https://github.com/ayhaidar/pspso/security/advisories/new).
Include the affected version, a minimal reproduction, the expected impact, and
any suggested mitigation. Please do not include sensitive details in a public
issue before a fix is available.

## Deployment boundary

PSPSO is designed as a local experiment tool. Its dashboard and API do not
provide user authentication, authorization, or TLS termination. The dashboard
therefore binds to loopback by default for normal single-user use. Other bind
addresses such as `0.0.0.0` are supported and print an exposure warning. Any
network deployment must add its own HTTPS, authentication, access control, and
network isolation.

PSPSO model and preprocessor artifacts use Python machine-learning serialization
formats. Only load artifacts created in a trusted workspace; tampered or
untrusted serialized files can execute code when loaded.
