"""Add compatible model transformations here; do not edit the baseline vendor code.

A recipe takes an eager L0 model and a JSON-serializable configuration dictionary,
then returns the transformed model. Never write files from a recipe: the runner
owns artifact saving. Training recipes should explicitly opt into train mode.
"""


def baseline(model, config):
    """No transformation: establish reproducibility and validate the workflow."""
    return model


RECIPES = {'baseline': baseline}
