# Changes
Most recent releases are shown at the top. Each release shows:

- **Added**: New classes, methods, functions, etc
- **Changed**: Additional parameters, changes to inputs or outputs, etc
- **Fixed**: Bug fixes that don't change documented behaviour

Note that the top-most release is changes in the unreleased master branch on Github. Parentheses after an item show the name or github id of the contributor of that change.

Keep a Changelog](https://keepachangelog.com/en/1.0.0/), [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2019.03.25]
### Added
- CHANGES.md
- new `arche.rules.duplicates.find_by()` to find duplicates by chosen columns

    import arche
    from arche.readers.items import JobItems
    df = JobItems(0, "235801/1/15").df
    arche.rules.duplicates.find_by(df, ["title", "category"]).show()

- `basic_json_schema().json()` prints a schema in JSON format
- `Result.show()` to print a rule result, e.g.

    from arche.rules.garbage_symbols import garbage_symbols
    from arche.readers.items import JobItems
    items = JobItems(0, "235801/1/15")
    garbage_symbols(items).show()

### Changed
- Tags rule returns unused tags, #2
- `basic_json_schema()` prints a schema as a python dict

### Deprecated
- `Arche().basic_json_schema()` deprecated in favor of `arche.basic_json_schema()`
### Removed
### Fixed
- `Arche().basic_json_schema()` not using `items_numbers` argument


## 2019.03.18

- Last release without CHANGES updates