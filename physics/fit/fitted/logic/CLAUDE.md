# The fitted settings, logic

- `load_fitted.py`: the fitted values as engine `Settings`, from the shipped record or
  from a JSON file of the same shape.

## Must not get wrong

- **A name that is no setting raises**, and so does any part other than `settings` and
  `fitted_on`. A file from the old engine (with a "start" part) is refused, not half read.
- **The engine checks the values**: a value `Settings` refuses raises here too.
- **Loading never changes `Settings()`.**
