"""Recipe loading and validation (YAML -> reference Recipe/Step types).

Recipe format (see recipes/example_salad.yaml):
    name, ingredients [list], verbs [list], steps [{verb, ingredient, note?}]

Validation (brief section 12): loading FAILS LOUDLY if a step uses a verb
not in `verbs` or an ingredient not in `ingredients`. The recipe is a
config file, never hard-coded.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from reference.state_machine_ref import Recipe, Step


def load_recipe(path) -> Recipe:
    """Parse and validate one recipe YAML.

    Input: path to a YAML file.
    Output: reference.state_machine_ref.Recipe (name, ingredients, verbs,
    steps).
    Raises ValueError naming the exact bad step/field on any violation.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: recipe must be a mapping, got {type(raw).__name__}")

    for key in ("name", "ingredients", "verbs", "steps"):
        if key not in raw:
            raise ValueError(f"{path}: missing required key '{key}'")

    name = str(raw["name"])
    ingredients = [str(x) for x in raw["ingredients"]]
    verbs = [str(x) for x in raw["verbs"]]
    if not ingredients:
        raise ValueError(f"{path}: 'ingredients' is empty")
    if not verbs:
        raise ValueError(f"{path}: 'verbs' is empty")
    if len(set(ingredients)) != len(ingredients):
        raise ValueError(f"{path}: duplicate entries in 'ingredients'")
    if len(set(verbs)) != len(verbs):
        raise ValueError(f"{path}: duplicate entries in 'verbs'")

    steps: list[Step] = []
    for i, s in enumerate(raw["steps"]):
        if not isinstance(s, dict) or "verb" not in s or "ingredient" not in s:
            raise ValueError(f"{path}: step {i} must map verb+ingredient, got {s!r}")
        if s["verb"] not in verbs:
            raise ValueError(f"{path}: step {i} uses verb '{s['verb']}' "
                             f"not in verbs {verbs}")
        if s["ingredient"] not in ingredients:
            raise ValueError(f"{path}: step {i} uses ingredient "
                             f"'{s['ingredient']}' not in ingredients "
                             f"{ingredients}")
        steps.append(Step(verb=str(s["verb"]), ingredient=str(s["ingredient"]),
                          note=str(s.get("note", ""))))
    if not steps:
        raise ValueError(f"{path}: 'steps' is empty")
    return Recipe(name=name, ingredients=ingredients, verbs=verbs, steps=steps)
