# The fitted settings, definitions

- `fitted_values.py`: `FITTED`, the settings the fit chose and what its search found.

## Must not get wrong

- **Two parts only**: `settings` (engine setting names and values) and `fitted_on`.
- **Empty means nothing is fitted**; the status then starts with "none". A test ties
  the two together.
- **Write values only from a holder** of a finished search, never by hand to make a
  scene look good.
