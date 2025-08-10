See [ai/README.md](ai/README.md) for AI assistant development guidance.

## Documentation Validation

When working with RST documentation files:
1. Run `./bin/validate-docs.sh` to check RST syntax before committing
2. Use `./bin/validate-docs.sh --changed` to check only modified files
3. The `.rstcheck.cfg` file configures which Sphinx-specific warnings to ignore
4. Documentation CI will automatically validate RST syntax on push

## Important Instruction Reminders

Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.