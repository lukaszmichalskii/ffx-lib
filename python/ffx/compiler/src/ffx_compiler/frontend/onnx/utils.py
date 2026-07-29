from typing import Any, Dict

import onnx


def extract_attributes(node: onnx.NodeProto) -> Dict[str, Any]:
    return {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}


def get_attribute_with_default(
    node: onnx.NodeProto,
    attr_name: str,
    default: Any = None,
) -> Any:
    for attr in node.attribute:
        if attr.name == attr_name:
            return onnx.helper.get_attribute_value(attr)
    return default
