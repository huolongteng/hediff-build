# HEDiff Build Harness

This repository uses GitHub-hosted Ubuntu runners to resolve and validate the
three native HEDiff dependency stacks without consuming local compute. Each job
prefers binary wheels. It falls back to building wheels from source only when a
complete binary resolution is unavailable, and uploads those fallback wheels.

Formal model training and HEDiff experiments run locally; this repository is
only an environment builder and compatibility gate.
