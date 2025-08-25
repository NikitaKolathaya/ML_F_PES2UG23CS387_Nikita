import torch

def get_entropy_of_dataset(tensor: torch.Tensor):
    target_col = tensor[:, -1]  
    classes, counts = torch.unique(target_col, return_counts=True)
    probabilities = counts.float() / target_col.size(0)
    entropy = -torch.sum(probabilities * torch.log2(probabilities))
    return entropy.item()


def get_avg_info_of_attribute(tensor: torch.Tensor, attribute: int):
    total_rows = tensor.size(0)
    attribute_col = tensor[:, attribute]
    values, counts = torch.unique(attribute_col, return_counts=True)

    avg_info = torch.tensor(0.0)
    for v, cnt in zip(values, counts):
        subset = tensor[attribute_col == v]
        entropy_subset = get_entropy_of_dataset(subset)
        weight = cnt.item() / total_rows
        avg_info += weight * entropy_subset
    return avg_info.item()


def get_information_gain(tensor: torch.Tensor, attribute: int):
    dataset_entropy = get_entropy_of_dataset(tensor)
    avg_info = get_avg_info_of_attribute(tensor, attribute)
    info_gain = dataset_entropy - avg_info
    return round(info_gain, 4)


def get_selected_attribute(tensor: torch.Tensor):
    num_attributes = tensor.size(1) - 1  
    info_gains = {}

    for attr in range(num_attributes):
        ig = get_information_gain(tensor, attr)
        info_gains[attr] = ig

    best_attribute = max(info_gains, key=info_gains.get)
    return info_gains, best_attribute


class DecisionNode:
    def __init__(self, attribute=None, is_leaf=False, prediction=None):
        self.attribute = attribute          
        self.children = {}                  
        self.is_leaf = is_leaf
        self.prediction = prediction        


def build_tree(tensor: torch.Tensor):
    target_col = tensor[:, -1]
    classes, counts = torch.unique(target_col, return_counts=True)

    if classes.size(0) == 1:
        return DecisionNode(is_leaf=True, prediction=classes.item())

    if tensor.size(1) == 1:
        majority_class = classes[torch.argmax(counts)].item()
        return DecisionNode(is_leaf=True, prediction=majority_class)

    _, best_attr = get_selected_attribute(tensor)
    node = DecisionNode(attribute=best_attr)

    attr_values = torch.unique(tensor[:, best_attr])
    for v in attr_values:
        subset = tensor[tensor[:, best_attr] == v]
       
        reduced_subset = torch.cat(
            (subset[:, :best_attr], subset[:, best_attr+1:]), dim=1
        )
        child = build_tree(reduced_subset)
        node.children[v.item() if subset.dtype in (torch.int64, torch.int32) else str(v.item())] = child

    return node


def print_tree(node: DecisionNode, depth=0):
    indent = "  " * depth
    if node.is_leaf:
        print(f"{indent}Leaf → Class {node.prediction}")
    else:
        print(f"{indent}Attribute {node.attribute}")
        for val, child in node.children.items():
            print(f"{indent} Value {val}:")
            print_tree(child, depth + 1)
