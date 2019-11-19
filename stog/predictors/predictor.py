from typing import List
import json

from stog.utils.registrable import Registrable
from stog.utils.checks import ConfigurationError
from stog.utils.string import JsonDict, sanitize
from stog.data import DatasetReader, Instance
from stog.data.dataset_builder import load_dataset_reader

# a mapping from model `type` to the default Predictor for that type
DEFAULT_PREDICTORS = {
    'Seq2Seq': 'Seq2Seq',
    'DeepBiaffineParser': 'BiaffineParser',
    'STOG': 'STOG',
}


class Predictor(Registrable):
    """
    a ``Predictor`` is a thin wrapper around an AllenNLP model that handles JSON -> JSON predictions
    that can be used for serving models through the web API or making predictions in bulk.
    """
    def __init__(self, model, dataset_reader: DatasetReader) -> None:
        self._model = model
        self._dataset_reader = dataset_reader

    def load_line(self, line: str) -> JsonDict:  # pylint: disable=no-self-use
        """
        If your inputs are not in JSON-lines format (e.g. you have a CSV)
        you can override this function to parse them correctly.
        """
        return json.loads(line)

    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        return json.dumps(outputs) + "\n"

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        return self.predict_instance(instance)

    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        return sanitize(outputs)

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Converts a JSON object into an :class:`~allennlp.data.instance.Instance`
        and a ``JsonDict`` of information which the ``Predictor`` should pass through,
        such as tokenised inputs.
        """
        raise NotImplementedError

    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        instances = self._batch_json_to_instances(inputs)
        return self.predict_batch_instance(instances)

    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        return sanitize(outputs)

    def _batch_json_to_instances(self, json_dicts: List[JsonDict]) -> List[Instance]:
        """
        Converts a list of JSON objects into a list of :class:`~allennlp.data.instance.Instance`s.
        By default, this expects that a "batch" consists of a list of JSON blobs which would
        individually be predicted by :func:`predict_json`. In order to use this method for
        batch prediction, :func:`_json_to_instance` should be implemented by the subclass, or
        if the instances have some dependency on each other, this method should be overridden
        directly.
        """
        instances = []
        for json_dict in json_dicts:
            instances.append(self._json_to_instance(json_dict))
        return instances

    @classmethod
    def from_path(cls, archive_path: str, predictor_name: str = None) -> 'Predictor':
        """
        Instantiate a :class:`Predictor` from an archive path.

        If you need more detailed configuration options, such as running the predictor on the GPU,
        please use `from_archive`.

        Parameters
        ----------
        archive_path The path to the archive.

        Returns
        -------
        A Predictor instance.
        """
        # Comment it out for cyclic imports
        # return Predictor.from_archive(load_archive(archive_path), predictor_name)
        raise NotImplementedError

    @classmethod
    def from_archive(cls, archive, predictor_name: str = None) -> 'Predictor':
        """
        Instantiate a :class:`Predictor` from an :class:`~allennlp.models.archival.Archive`;
        that is, from the result of training a model. Optionally specify which `Predictor`
        subclass; otherwise, the default one for the model will be used.
        """
        # Duplicate the config so that the config inside the archive doesn't get consumed
        config = archive.config.duplicate()

        if not predictor_name:
            model_type = config.get("model").get("model_type")
            if not model_type in DEFAULT_PREDICTORS:
                raise ConfigurationError(f"No default predictor for model type {model_type}.\n"\
                                         f"Please specify a predictor explicitly.")
            predictor_name = DEFAULT_PREDICTORS[model_type]

        word_splitter = None
        if config['model'].get('use_bert', False):
            word_splitter=config['data'].get('word_splitter', None)
        dataset_reader = load_dataset_reader(
            config["data"]["data_type"], word_splitter=word_splitter)
        if hasattr(dataset_reader, 'set_evaluation'):
            dataset_reader.set_evaluation()

        model = archive.model
        model.eval()

        return Predictor.by_name(predictor_name)(model, dataset_reader)
