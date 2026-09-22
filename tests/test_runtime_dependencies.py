import unittest

from scripts.check_runtime_dependencies import unexpected_pip_check_lines


class RuntimeDependencyTests(unittest.TestCase):
    def test_headless_profile_accepts_only_gui_opencv_alias_errors(self) -> None:
        output = "\n".join(
            [
                "felis 0.1.0 requires opencv-python, which is not installed.",
                "ultralytics 8.4.143 requires opencv-python, which is not installed.",
            ]
        )

        self.assertEqual(
            unexpected_pip_check_lines(output, "opencv-python-headless"), []
        )

    def test_runtime_check_preserves_unrelated_dependency_errors(self) -> None:
        output = "felis 0.1.0 requires pandas, which is not installed."

        self.assertEqual(
            unexpected_pip_check_lines(output, "opencv-python-headless"), [output]
        )

    def test_runtime_check_accepts_clean_pip_check_output(self) -> None:
        self.assertEqual(
            unexpected_pip_check_lines(
                "No broken requirements found.", "opencv-python"
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
