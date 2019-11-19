from overrides import overrides
import re
import json
import sys
from stog.utils.registrable import Registrable
from stog.utils.checks import ConfigurationError
from stog.utils.string import JsonDict, sanitize
from stog.data import DatasetReader, Instance
from stog.predictors.predictor import Predictor
from stog.utils.string import START_SYMBOL, END_SYMBOL
from stog.data.dataset_readers.amr_parsing.amr import AMRGraph
from stog.utils.exception_hook import ExceptionHook

sys.excepthook = ExceptionHook()


@Predictor.register('STOG')
class STOGPredictor(Predictor):
    """
    Predictor for the :class:`~stog.models.stog. model.
    """

    def predict(self, source: str) -> JsonDict:
        return self.predict_json({"source": source})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"source": "..."}``.
        """
        source = json_dict["source"]
        return self._dataset_reader.text_to_instance(source)

    @overrides
    def predict_batch_instance(self, instances):
        outputs = []
        gen_vocab_size = self._model.vocab.get_vocab_size('decoder_token_ids')
        _outputs = super(STOGPredictor, self).predict_batch_instance(instances)
        for instance, output in zip(instances, _outputs):
            gold_amr = instance.fields['amr'].metadata
            copy_vocab = instance.fields['src_copy_vocab'].metadata
            node_indexes = output['nodes']
            head_indexes = output['heads']
            head_label_indexes = output['head_labels']
            corefs = output['corefs']

            nodes = []
            head_labels = []
            copy_indicators = []

            for i, index in enumerate(node_indexes):
                # Lookup the node.
                if index >= gen_vocab_size:
                    copy_index = index - gen_vocab_size
                    nodes.append(copy_vocab.get_token_from_idx(copy_index))
                    copy_indicators.append(1)
                else:
                    nodes.append(self._model.vocab.get_token_from_index(index, 'decoder_token_ids'))
                    copy_indicators.append(0)
                # Lookup the head label.
                head_labels.append(self._model.vocab.get_token_from_index(
                    head_label_indexes[i], 'head_tags'))

            if END_SYMBOL in nodes:
                nodes = nodes[:nodes.index(END_SYMBOL)]
                head_indexes = head_indexes[:len(nodes)]
                head_labels = head_labels[:len(nodes)]
                corefs = corefs[:len(nodes)]

            outputs.append(dict(
                nodes=nodes,
                heads=head_indexes,
                corefs=corefs,
                head_labels=head_labels,
                copy_indicators=copy_indicators,
                gold_amr=gold_amr
            ))

        return outputs

    @overrides
    def dump_line(self, output):
        # return ' '.join(output['nodes']) + '\n'
        pred_graph = AMRGraph.from_prediction(output)
        amr = output['gold_amr']
        gold_graph = amr.graph
        amr.graph = pred_graph

        string_to_print = str(amr).replace(
            "# ::save-date", "# ::tgt_ref {}\n# ::tgt_pred {}\n# ::save-date".format(
                " ".join(output["nodes"]),
                " ".join(gold_graph.get_tgt_tokens()
                         )
            )
        )
        return string_to_print + '\n\n'
