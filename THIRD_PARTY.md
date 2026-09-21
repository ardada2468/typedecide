# Third-party work this project builds on

Nothing here is vendored. The two reference repositories are cloned locally by
`bench/fetch_fixtures.sh` and are gitignored, because they are other people's code
and data and should be obtained from their own sources.

## openjev — TheoLeeCJ/openjev (MIT)

<https://github.com/TheoLeeCJ/openjev>

- `benchmarks/data/authored144.jsonl` is the fixture every accuracy number in this
  repository is measured on. We do not redistribute it.
- The metric, mean family balanced accuracy, follows openjev's
  `benchmarks/manifests/metric-contract.json`.
- The letter-slot readout and the prefix-verification discipline originate here.

## sarvam-jev — SAGAR-TAMANG/sarvam-jev

<https://github.com/SAGAR-TAMANG/sarvam-jev>

The immediate predecessor to this project: the same mechanism on Sarvam's Indic
models, with a wllama browser demo. Our browser engine is a response to that one and
differs in the ways recorded in `README.md`.

## Jev

Architectural background on Jev itself is from
[archerhume's teardown](https://archerhume.com/posts/jevs-architecture-unmasked/).

None of these projects are affiliated with this one, and none of them have endorsed
it. Published numbers quoted for comparison are theirs, measured on their hardware.
