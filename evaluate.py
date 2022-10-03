#!/usr/bin/env python3

from typing import List, Optional, Tuple

BRACKET_OPEN = "["
BRACKET_CLOSE = "]"
# PREFIX_INTENT = "IN:"
# PREFIX_SLOT = "SL:"

PREFIX_INTENT = "in:"
PREFIX_SLOT = "sl:"

class Node:
    """
    A generalization of Root / Intent / Slot / Token
    """
    def __init__(self, label: str) -> None:
        self.label: str = label
        self.children: List[Node] = []
        self.parent: Optional[Node] = None

    def validate_node(self) -> None:
        for child in self.children:
            child.validate_node()

    def list_nonterminals(self):
        non_terminals: List[Node] = []
        for child in self.children:
            if type(child) != Root and type(child) != Token:
                non_terminals.append(child)
                non_terminals += child.list_nonterminals()
        return non_terminals

    def get_token_indices(self) -> List[int]:
        indices: List[int] = []
        if self.children:
            for child in self.children:
                if type(child) == Token:
                    indices.append(child.index)
                else:
                    indices += child.get_token_indices()
        return indices

    def get_token_span(self) -> Optional[Tuple[int, int]]:
        indices = self.get_token_indices()
        if indices:
            return (min(indices), max(indices) + 1)
        return None

    def get_flat_str_spans(self) -> str:
        str_span: str = str(self.get_token_span()) + ": "
        if self.children:
            for child in self.children:
                str_span += str(child)
        return str_span

    def __repr__(self) -> str:
        str_repr: str = ""
        if type(self) == Intent or type(self) == Slot:
            str_repr = BRACKET_OPEN
        if type(self) != Root:
            str_repr += str(self.label) + " "
        if self.children:
            for child in self.children:
                str_repr += str(child)
        if type(self) == Intent or type(self) == Slot:
            str_repr += BRACKET_CLOSE + " "
        return str_repr


class Root(Node):
    def __init__(self) -> None:
        super().__init__("ROOT")

    def validate_node(self) -> None:
        super().validate_node()
        for child in self.children:
            if type(child) == Slot or type(child) == Root:
                raise TypeError(
                    "A Root's child must be an Intent or Token: " + self.label)
            elif self.parent is not None:
                raise TypeError(
                    "A Root should not have a parent: " + self.label)


class Intent(Node):
    def __init__(self, label: str) -> None:
        super().__init__(label)

    def validate_node(self) -> None:
        super().validate_node()
        for child in self.children:
            if type(child) == Intent or type(child) == Root:
                raise TypeError(
                    "An Intent's child must be a slot or token: " + self.label)


class Slot(Node):
    def __init__(self, label: str) -> None:
        super().__init__(label)

    def validate_node(self) -> None:
        super().validate_node()
        for child in self.children:
            if type(child) == Slot or type(child) == Root:
                raise TypeError("An Slot's child must be an intent or token: "
                                + self.label)


class Token(Node):
    def __init__(self, label: str, index: int) -> None:
        super().__init__(label)
        self.index: int = index

    def validate_node(self) -> None:
        if len(self.children) > 0:
            raise TypeError("A Token {} can't have children: {}".format(
                self.label, str(self.children)))


class Tree:
    def __init__(self, top_repr: str) -> None:
        self.root = Tree.build_tree(top_repr)
        try:
            self.validate_tree()
        except ValueError as v:
            raise ValueError("Tree validation failed: {}".format(v))

    @staticmethod
    def build_tree(top_repr: str) -> Root:
        root = Root()
        node_stack: List[Node] = [root]
        token_count: int = 0

        for item in top_repr.split():
            if item == BRACKET_CLOSE:
                if not node_stack:
                    raise ValueError("Tree validation failed")
                node_stack.pop()

            elif item.startswith(BRACKET_OPEN):
                label: str = item[1:]
                if label.startswith(PREFIX_INTENT):
                    node_stack.append(Intent(label))
                elif label.startswith(PREFIX_SLOT):
                    node_stack.append(Slot(label))
                else:
                    raise NameError(
                        "Nonterminal label {} must start with {} or {}".format(
                            label, PREFIX_INTENT, PREFIX_SLOT))

                if len(node_stack) < 2:
                    raise ValueError("Tree validation failed")
                node_stack[-1].parent = node_stack[-2]
                node_stack[-2].children.append(node_stack[-1])

            else:
                token = Token(item, token_count)
                token_count += 1
                if not node_stack:
                    raise ValueError("Tree validation failed")
                token.parent = node_stack[-1]
                node_stack[-1].children.append(token)

        if len(node_stack) > 1:
            raise ValueError("Tree validation failed")

        return root

    def validate_tree(self) -> None:
        try:
            self.root.validate_node()
            for child in self.root.children:
                child.validate_node()
        except TypeError as t:
            raise ValueError("Failed validation for {} \n {}".format(
                self.root, str(t)))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.root == other.root

    def __repr__(self) -> str:
        return repr(self.root).strip()

from itertools import zip_longest
from typing import Counter, Dict, Optional
import argparse


class Calculator:
    def __init__(self, strict: bool = False) -> None:
        self.num_gold_nt: int = 0
        self.num_pred_nt: int = 0
        self.num_matching_nt: int = 0
        self.strict: bool = strict

    def get_metrics(self):
        precision: float = (
            self.num_matching_nt / self.num_pred_nt) if self.num_pred_nt else 0
        recall: float = (
            self.num_matching_nt / self.num_gold_nt) if self.num_gold_nt else 0
        f1: float = (2.0 * precision * recall /
                     (precision + recall)) if precision + recall else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def add_instance(self, gold_tree: Tree,
                     pred_tree: Optional[Tree] = None) -> None:
        node_info_gold: Counter = self._get_node_info(gold_tree)
        self.num_gold_nt += sum(node_info_gold.values())

        if pred_tree:
            node_info_pred: Counter = self._get_node_info(pred_tree)
            self.num_pred_nt += sum(node_info_pred.values())
            self.num_matching_nt += sum(
                (node_info_gold & node_info_pred).values())

    def _get_node_info(self, tree) -> Counter:
        nodes = tree.root.list_nonterminals()
        node_info: Counter = Counter()
        for node in nodes:
            node_info[(node.label, self._get_span(node))] += 1
        return node_info

    def _get_span(self, node):
        return node.get_flat_str_spans(
        ) if self.strict else node.get_token_span()

def evaluate(labels, predictions) -> dict:

    instance_count: int = 0
    exact_matches: int = 0
    invalid_preds: float = 0
    labeled_bracketing_scores = Calculator(strict=False)
    tree_labeled_bracketing_scores = Calculator(strict=True)


    for gold_line, pred_line in zip_longest(labels, predictions):
        gold_line = gold_line.replace(": ", ":").replace("[ ","[").strip()
        pred_line = pred_line.replace(": ", ":").replace("[ ","[").strip()

        # print("prediction:", pred_line)
        # print("label:", gold_line)

        # try:
        gold_tree = Tree(gold_line)
        instance_count += 1
        # except ValueError:
        #     print("FATAL: found invalid line in gold file:", gold_line)
        #     # quit()

        try:
            pred_tree = Tree(pred_line)
            labeled_bracketing_scores.add_instance(gold_tree, pred_tree)
            tree_labeled_bracketing_scores.add_instance(
                gold_tree, pred_tree)
        except ValueError:
            # print("WARNING: found invalid line in pred file:", pred_line)
            invalid_preds += 1
            labeled_bracketing_scores.add_instance(gold_tree)
            tree_labeled_bracketing_scores.add_instance(gold_tree)
            continue

        if str(gold_tree) == str(pred_tree):
            exact_matches += 1

    exact_match_fraction: float = (
        exact_matches / instance_count) if instance_count else 0
    tree_validity_fraction: float = (
        1 - (invalid_preds / instance_count)) if instance_count else 0

    return {
        "exact_match":
        exact_match_fraction,
        "labeled_bracketing_scores":
        labeled_bracketing_scores.get_metrics(),
        "tree_labeled_bracketing_scores":
        tree_labeled_bracketing_scores.get_metrics(),
        "tree_validity":
        tree_validity_fraction
    }

def evaluate_predictions(labels, predictions):
    results = []
    for gold_line, pred_line in zip_longest(labels, predictions):
        try:
            results.append(evaluate([gold_line], [pred_line]))
        except:
            results.append({
                    "exact_match": 0,
                    "labeled_bracketing_scores": {'precision': 0.0, 'recall': 0.0, 'f1': 0.},
                    "tree_labeled_bracketing_scores": {'precision': 0.0, 'recall': 0.0, 'f1': 0.},
                    "tree_validity":0.
                    })
    
    return results

    
    
