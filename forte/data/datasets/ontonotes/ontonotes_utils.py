from forte.data import DataPack
from forte.data.ontology import ontonotes_ontology as Ont

__all__ = [
    "write_tokens_to_file",
]


def write_tokens_to_file(pred_pack: DataPack, gold_pack: DataPack,
                         pred_path: str, gold_path: str):
    pred_data, gold_data = [
        pack.get_data(
            context_type=Ont.Sentence, request={
                Ont.Token: [],
                Ont.PredicateMention: {"unit": "Token"},
                Ont.PredicateArgument: {"unit": "Token"},
                Ont.PredicateLink: {"fields": ["arg_type"]},
            })
        for pack in [pred_pack, gold_pack]]

    f_pred = open(pred_path, "w")
    f_gold = open(gold_path, "w")
    for pred_sentence, gold_sentence in zip(pred_data, gold_data):
        words = pred_sentence["Token"]["text"]
        # It is necessary to use gold predicates for evaluation.
        props = ["-"] * len(words)
        predicates = gold_sentence["PredicateMention"]["unit_span"][:, 0]
        for idx in predicates:
            props[idx] = words[idx]

        for sent, f in [(pred_sentence, f_pred),
                        (gold_sentence, f_gold)]:
            col_labels = [["*"] * len(words) for _ in range(len(predicates))]
            pred_to_args = {pred_id: [] for pred_id in predicates}
            n_links = len(sent["PredicateLink"]["arg_type"])

            for link_id in range(n_links):
                # TODO: This is ridiculously convoluted.
                arg_type = sent["PredicateLink"]["arg_type"][link_id]
                parent_id = sent["PredicateLink"]["parent"][link_id]
                child_id = sent["PredicateLink"]["child"][link_id]
                pred_id = sent["PredicateMention"]["unit_span"][parent_id][0]
                arg_span = sent["PredicateArgument"]["unit_span"][child_id]
                pred_to_args[pred_id].append(
                    (arg_span[0], arg_span[1], arg_type))

            for i, pred_id in enumerate(predicates):
                # To make sure CoNLL-eval script count matching predicates
                # as correct.
                flags = [False] * len(words)
                for start, end, label in pred_to_args[pred_id]:
                    if max(flags[start:end + 1]):
                        continue
                    col_labels[i][start] = "(" + label + col_labels[i][start]
                    col_labels[i][end] = col_labels[i][end] + ")"
                    for j in range(start, end + 1):
                        flags[j] = True
                # Add unpredicted verb (for predicted SRL).
                if not flags[pred_id]:
                    col_labels[i][pred_id] = "(V*)"

            # Print a labeled sentence
            for label_column in col_labels:
                assert len(label_column) == len(col_labels)
            for i in range(len(props)):
                f.write(props[i].ljust(15))
                for label_column in col_labels:
                    f.write(label_column[i].rjust(15))
                f.write("\n")
            f.write("\n")
    f_pred.close()
    f_gold.close()
