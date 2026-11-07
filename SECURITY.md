# Security policy

## Supported versions

Security fixes go into the latest minor release (currently the 0.x line). Please upgrade before reporting.

## Reporting a vulnerability

Please do **not** open a public issue. Report privately through GitHub's
["Report a vulnerability"](https://github.com/BowlOfData/maki/security/advisories/new) form, or email
bowlofdata@gmail.com. Include the maki version, a minimal reproduction and the impact you see.

You can expect an acknowledgement within a week. Fixes are released as a patch version and credited in the
[changelog](CHANGELOG.md) unless you prefer otherwise.

## Scope

Of particular interest: SSRF or URL-validation bypasses in `Connector`, plugin sandbox escapes (path handling,
`DANGEROUS_METHODS` gating), authentication bypass or denial of service in `maki serve`, and secret leakage in logs.

Running `maki serve` on a non-loopback address without an API key and TLS is a configuration problem, not a
vulnerability; the CLI warns about it.
