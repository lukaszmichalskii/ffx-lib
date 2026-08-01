import logging
import shutil
from pathlib import Path

from ffx_compiler.backend import Codegen, codegen_factory
from ffx_compiler.error import FfxError
from ffx_compiler.frontend import Parser
from ffx_compiler.frontend.parser_factory import parser_factory
from ffx_compiler.middle_end import Optimizer, optimizer_factory

logger = logging.getLogger(__name__)


class Compiler:
    def __init__(
        self,
        model_path: Path,
        output_path: Path,
        level: int = 0,
        runtime: str = "ffx_runtime",
        override: bool = False,
    ):
        self.model_path = model_path
        self.output_path = output_path

        if not self.model_path.exists():
            raise FfxError(f"Model file not found at path: {self.model_path}")

        export_format = self.model_path.suffix.removeprefix(".")
        self.frontend: Parser = parser_factory(model_format=export_format)
        self.middle_end: Optimizer = optimizer_factory(runtime=runtime, level=level)
        self.backend: Codegen = codegen_factory(runtime=runtime)

        self.__make_workspace(override)

    def compile(self) -> int:
        logger.info(f"Starting AOT compilation for model: {self.model_path.name}")

        # IR synthesis pass
        graph_ir = self.frontend.parse_to_intermediate_representation(self.model_path)
        logger.debug(
            f"Parsed IR graph '{graph_ir.name}' with {len(graph_ir.nodes)} nodes."
        )
        for node in graph_ir.nodes:
            print(node.type)

        # optimization pass
        graph_ir_optimized = self.middle_end.optimize_intermediate_representation(
            graph_ir
        )

        # cpp source and binary payload synthesis
        model_source, model_data = (
            self.backend.intermediate_representation_to_source_code(graph_ir_optimized)
        )
        logger.debug("Source code and data synthesis done")

        # emit workspace artifacts
        for filename, source_code in model_source.items():
            self.output_path.joinpath(filename).write_text(
                source_code, encoding="utf-8"
            )

        data = self.output_path.joinpath("model.data")
        data.write_bytes(model_data)
        logger.info(
            f"Compilation completed successfully. Artifacts written to: {self.output_path.resolve()}"
        )

        return 0

    def __make_workspace(self, override: bool = False) -> None:
        if self.output_path.exists() and any(self.output_path.iterdir()):
            if not override:
                raise FfxError(
                    f"Output directory '{self.output_path}' exists and is not empty. "
                    "Use --force to overwrite."
                )
            shutil.rmtree(self.output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
