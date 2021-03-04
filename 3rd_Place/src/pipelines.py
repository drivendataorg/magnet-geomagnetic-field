from typing import Dict, Any
from preprocessing.transformer import NoOp, FeatureFilter
from preprocessing.transformer import Normalize, ToDtype, FillNaN
from preprocessing.transformer import MakeSureFeatures
from sklearn.pipeline import Pipeline


# to use a pipeline from a config file,
# the instance must be imported in this file
# and add it to the dictionary with a unique key
library = {'feature_filter': FeatureFilter,
           'normalize': Normalize,
           'to_float32': ToDtype,
           'fillnan': FillNaN,
           'make_sure_features': MakeSureFeatures}


def build_pipeline(pipeline_config: Dict[str, Dict[str, Any]]) -> Pipeline:
    """
    A function to build a pipeline from a config file
    # Params
    pipeline_config: `Dict[str, Dict[str, Any]]`
        A OrderedDict of dictionaries where each key is string from
        the library dict define above, and its value is a dictionary of
        valid parameters for the choosen transformer class.
        structure:
            {'name_of_the_pipeline': parameters}
            name_of_the_pipeline: a key in the pipeline library dict define above
            parameters: this must be dictionary with valid parameters values

    # Example
    >>> pipeline_config = {'normalize': {},  # no parameters needit
                           'to_float32': {}, # no parameters needit
                           }
    >>> build_pipeline(pipeline_config)
    Pipeline(steps=[('normalize', Normalize()),
                    ('to_float32', ToDtype())])

    # Returns
    Pipeline:
        A full parametrized sklearn Pipeline
    """
    if len(pipeline_config) == 0:
        return Pipeline(steps=[('no_op', NoOp())])
    steps = []
    for pipeline_name, pipeline_config in pipeline_config.items():
        pipeline_instance = library[pipeline_name]
        steps.append((pipeline_name, pipeline_instance(**pipeline_config)))
    return Pipeline(steps=steps)
