# Contribute

## Code of conduct

We follow [The Apache Software Foundation's code of conduct](https://www.apache.org/foundation/policies/conduct.html). Be open, empathetic, collaborative, inquisitive, carefully-worded, and concise. We do not support harrassment, non-welcoming behavior, and other behaviors detrimental to our community.

## GitHub preferred

Developer communications should primarily live in GitHub issues and PRs, as this best helps with asynchronous communications and future reference

## Chat!

When in doubt, [stop by the Slack channel](https://join.slack.com/t/graphistry-community/shared_invite/zt-53ik36w2-fpP0Ibjbk7IJuVFIRSnr6g)

## Report bugs and propose features

There are more ways to contribute than code. Filing bugs, including for usability, or coming up with and voting on big feature win ideas, are great as well!

Please use GitHub issues to report and discuss topics and discuss them. Search for open/closed ones first.

When filing a bug, pleasse provide a fully reproducible code snippet. We should be able to copy-paste it into a Jupyter notebook and reproduce the problem.

## Improve docs and share examples

Docs can always be better:
* Feel free to expand on the core docs
* Styling can always use care
* Examples of use cases, integrations, and techniques can always help others

## PRs welcome

We are happy to accept PRs! 

* Check for `good first issue` and `help wanted` tags for ideas on where to start
* If you have something specific you'd like to add, we are happy to provide guidance via GitHub on where to start and what to add to land it
* Data integrations, convenience methods, fixes, and more are all welcome
* When in doubt, ask on an Issue / PR / Slack!

### Git conventions

**Commits should be atomic**. Every commit -- or squashed PR -- should be a self-contained addition/removal so we can cherrypick them as needed. 

For example, if you fix some bug, refactor some code, and update dependencies while there... split that into three commits: `fix()`, `refactor()`, and `garden()`.

**We use [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/).** 

Messages should look like:

```
fix(some feature name): verb action taken
```

The commit types are `fix()`, `feat()`, `infra()`, `garden()` / `refactor()`, `docs()`, `security()`.

**Descriptive PRs and CHANGELOG.md**

* Every PR should detail the additions/removals/fixes and breaking changes
* When adding features, try to add examples in the PR
* Manually update CHANGELOG.md as part of the PR

**Automation**

* PRs must pass CI, including style checks
* Maintainers are responsible for publishing