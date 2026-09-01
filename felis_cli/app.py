from cliff.app import App
from cliff.commandmanager import CommandManager


class FelisApp(App):
    def __init__(self) -> None:
        super().__init__(
            description=(
                "Run the FELIS YOLO-based camera-trap processing pipeline. "
                "Use a subcommand to predict detections, extract metadata, "
                "aggregate results, validate overlays, or run the full workflow."
            ),
            version="0.1.0",
            command_manager=CommandManager("felis.commands"),
            deferred_help=True,
        )

    def initialize_app(self, argv):  # noqa: D401
        # Commands are registered programmatically below
        # since we aren't packaging entry_points here.
        from .commands.predict import Predict
        from .commands.get_exif import GetExif
        from .commands.validate import Validate
        from .commands.aggregate import Aggregate
        from .commands.run import RunPipeline

        self.command_manager.add_command("predict", Predict)
        self.command_manager.add_command("exif", GetExif)
        self.command_manager.add_command("validate", Validate)
        self.command_manager.add_command("aggregate", Aggregate)
        self.command_manager.add_command("run", RunPipeline)
