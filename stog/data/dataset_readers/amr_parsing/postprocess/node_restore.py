from stog.data.dataset_readers.amr_parsing.io import AMRIO


class NodeRestore:

    def __init__(self, node_utils):
        self.node_utils = node_utils

    def restore_instance(self, amr):
        graph = amr.graph
        for node in graph.get_nodes():
            instance = node.instance
            new_instance = self.node_utils.get_frames(instance)[0]
            if instance != new_instance:
                graph.replace_node_attribute(node, 'instance', instance, new_instance)
            continue

    def restore_file(self, file_path):
        for amr in AMRIO.read(file_path):
            self.restore_instance(amr)
            yield amr


if __name__ == '__main__':
    import argparse

    from stog.data.dataset_readers.amr_parsing.node_utils import NodeUtilities as NU

    parser = argparse.ArgumentParser('node_restore.py')
    parser.add_argument('--amr_files', nargs='+', required=True)
    parser.add_argument('--util_dir', default='./temp')

    args = parser.parse_args()

    node_utils = NU.from_json(args.util_dir, 0)

    nr = NodeRestore(node_utils)

    for file_path in args.amr_files:
        with open(file_path + '.frame', 'w', encoding='utf-8') as f:
            for amr in nr.restore_file(file_path):
                f.write(str(amr) + '\n\n')
