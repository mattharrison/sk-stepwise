from pathlib import Path
import doctest


def test_readme_doctests_pass():
    readme = Path(__file__).resolve().parents[1] / "README.md"
    result = doctest.testfile(
        str(readme),
        module_relative=False,
        optionflags=doctest.ELLIPSIS,
    )
    assert result.failed == 0
