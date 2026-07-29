from ffx_compiler.frontend.frontend_error import FrontendError
from ffx_compiler.frontend.onnx.parser import OnnxParser
from ffx_compiler.frontend.parser import Parser


def parser_factory(model_format: str) -> Parser:
    if model_format == "onnx":
        return OnnxParser()
    raise FrontendError(f"Unsupported model format: {model_format}")
