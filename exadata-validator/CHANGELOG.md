# Changelog

## Unreleased

- Breaking: numerical metadata bounds must be finite JSON numbers representable
  as doubles. Numeric strings and booleans are rejected with InvalidDataModelError.
- Bind numerical bounds as SQL parameters instead of SQL expressions.
