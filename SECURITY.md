# Security Policy

## Supported versions

| version | supported |
|---|---|
| 3.6.x (`main`) | yes |
| earlier | no |

## Reporting a vulnerability

Please report security issues privately through GitHub's [private vulnerability reporting](https://github.com/mlbkumar9/Param2Image/security/advisories/new). Do not open a public issue for them. You can expect an acknowledgement within a week.

## Loading checkpoints

PyTorch checkpoints are pickle files and can execute code when loaded. `p2i-evaluate` and `p2i-predict` load full checkpoints, including their stored configuration, so only load checkpoints from sources you trust.
