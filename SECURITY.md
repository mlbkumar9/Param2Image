# Security Policy

## Supported versions

This project is research/source software for a Param2Image cGAN pipeline. Use the latest `master` branch.

## Reporting a vulnerability

If you discover a security issue (for example accidental exposure of credentials, unsafe deserialization of untrusted checkpoints, or path traversal in data loading), please open a private security advisory on GitHub or contact the repository owner.

Please do **not** open a public issue that includes secrets, private datasets, or exploit details.

## Notes

- Do not commit API keys, cloud credentials, or private simulation data.
- Treat untrusted `.pth` checkpoints like untrusted code: load only from sources you trust.
