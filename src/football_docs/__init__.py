"""Docs-build tooling shared by the football-* documentation sites.

Four modules, each loaded by module name so that a consumer repo needs nothing but the
installed football-core:

- :mod:`football_docs.roles`: a griffe extension that renders the Sphinx roles of numpy
  docstrings as cross-reference links or plain code at build time;
- :mod:`football_docs.pages`: the checks a public page must pass (no internal process codes,
  every number sourced, no raw role in the built site) and the home page's results block,
  copied by code from the README's generated card blocks;
- :mod:`football_docs.theme`: the site theme (CSS variables, a few rules, the logo) generated
  from the chart themes in ``football_core.viz``, light and dark;
- :mod:`football_docs.meta`: the repository metadata check: author, CI and Pages wiring.

**Dependencies.** ``roles`` imports ``griffe`` (the ``docs`` dependency group, through
mkdocstrings); ``meta`` imports ``yaml`` (PyYAML, also a transitive ``docs``-group dependency,
through mkdocs); ``theme`` imports ``football_core.viz`` (the ``viz`` extra, which the ``dev``
group pulls); ``pages`` needs the standard library only. Developers and CI install every group
(``uv sync --all-groups``), because the doctests of this package run with the test suite.
Nothing in ``football_core`` imports this package, so it stays outside the model repos'
prediction-cache hash.
"""
