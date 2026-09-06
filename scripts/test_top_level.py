import contextlib
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import check_top_level as gate
from dspy_exports import ExportError, PinnedModules, PublicExports


class ExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repository = Path(temporary.name)
        self.git("init", "--quiet")

    def git(self, *arguments):
        return subprocess.check_output(
            ["git", "-C", str(self.repository), *arguments], text=True
        ).strip()

    def exports_from(self, sources):
        for name, source in sources.items():
            path = self.repository / "dspy" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
        self.git("add", "dspy")
        return PublicExports(PinnedModules(self.repository, self.git("write-tree")))

    def test_reexports_keep_the_defining_module(self):
        exports = self.exports_from(
            {
                "__init__.py": "from dspy.tools import Tool\n",
                "tools/__init__.py": "from .public import Tool\n",
                "tools/public.py": "class Tool: pass\n",
                "avatar.py": "class Tool: pass\n",
            }
        ).exports()
        self.assertEqual(exports, {"Tool": "tools/public.py::Tool"})
        self.assertEqual(gate.missing_exports(exports, {}, {"avatar.py": {}}), exports)
        self.assertEqual(
            gate.missing_exports(
                exports, {"symbols": {"tools/public.py::Tool": {}}}, {}
            ),
            {},
        )

    def test_a_decided_package_does_not_cover_its_imports(self):
        exports = self.exports_from(
            {
                "__init__.py": "from dspy.clients import LM\n",
                "clients/__init__.py": "from .lm import LM\n",
                "clients/lm.py": "class LM: pass\n",
            }
        ).exports()
        self.assertEqual(
            gate.missing_exports(exports, {}, {"clients/__init__.py": {}}), exports
        )

    def test_imported_all_and_relative_imports(self):
        exports = self.exports_from(
            {
                "__init__.py": "from dspy.core import *\n",
                "core/__init__.py": "from .types import *\nfrom .types import __all__ as __all__\n",
                "core/types.py": "__all__ = ('Message',)\nclass Message: pass\n",
            }
        ).exports()
        self.assertEqual(exports, {"Message": "core/types.py::Message"})

    def test_import_aliases_and_instances_resolve_to_the_class(self):
        exports = self.exports_from(
            {
                "__init__.py": "from dspy.settings import settings as active\ncontext = active.context\n",
                "settings.py": "class Settings: pass\nsettings = Settings()\n",
            }
        ).exports()
        self.assertEqual(
            exports,
            {"active": "settings.py::Settings", "context": "settings.py::Settings"},
        )

    def test_aliases_retain_the_binding_that_existed_at_assignment(self):
        exports = self.exports_from(
            {
                "__init__.py": "from dspy.one import Item\nAlias = Item\nfrom dspy.two import Item\n",
                "one.py": "class Item: pass\n",
                "two.py": "class Item: pass\n",
            }
        ).exports()
        self.assertEqual(exports, {"Alias": "one.py::Item", "Item": "two.py::Item"})

    def test_local_variables_do_not_become_module_exports(self):
        exports = self.exports_from(
            {
                "__init__.py": "def run():\n    private_to_function = 1\nclass Worker:\n    field = 2\n",
            }
        ).exports()
        self.assertEqual(
            exports, {"run": "__init__.py::run", "Worker": "__init__.py::Worker"}
        )

    def test_a_name_cannot_refer_to_a_later_definition(self):
        resolver = self.exports_from(
            {"__init__.py": "Alias = Later\nclass Later: pass\n"}
        )
        with self.assertRaisesRegex(ExportError, "name used before binding"):
            resolver.exports()

    def test_annotation_without_a_value_does_not_bind_a_name(self):
        self.assertEqual(
            self.exports_from({"__init__.py": "unset: int\n"}).exports(), {}
        )

    def test_unreadable_star_import_fails(self):
        resolver = self.exports_from({"__init__.py": "from dspy.missing import *\n"})
        with self.assertRaisesRegex(ExportError, "cannot read dspy.missing"):
            resolver.exports()

    def test_missing_or_dynamic_all_fails(self):
        declarations = [
            "",
            "__all__ = build_names()",
            "__all__ = ['Thing']\n__all__ += ['Other']",
            "__all__ = ['Thing']\n__all__.append('Other')",
            "__all__ = [1]",
            "__all__ = ['Thing']\n__all__[0] = 'Other'",
            "__all__ = ['Thing']\nalias = __all__\nalias.append('Other')",
            "names = ['Thing']\n__all__ = names\nnames.append('Other')",
        ]
        for declaration in declarations:
            with self.subTest(declaration=declaration):
                resolver = self.exports_from(
                    {
                        "__init__.py": "from dspy.types import *\n",
                        "types.py": f"class Thing: pass\n{declaration}\n",
                    }
                )
                with self.assertRaises(ExportError):
                    resolver.exports()

    def test_all_cannot_name_an_undefined_export(self):
        resolver = self.exports_from(
            {
                "__init__.py": "from dspy.types import *\n",
                "types.py": "__all__ = ['Missing']\n",
            }
        )
        with self.assertRaisesRegex(
            ExportError, "unresolved export dspy.types.Missing"
        ):
            resolver.exports()

    def test_cyclic_reexports_fail(self):
        resolver = self.exports_from(
            {
                "__init__.py": "from dspy.one import Item\n",
                "one.py": "from dspy.two import Item\n",
                "two.py": "from dspy.one import Item\n",
            }
        )
        with self.assertRaisesRegex(ExportError, "cyclic export"):
            resolver.exports()

    def test_cyclic_star_imports_fail(self):
        resolver = self.exports_from(
            {
                "__init__.py": "from dspy.one import *\n",
                "one.py": "from dspy.two import *\n",
                "two.py": "from dspy.one import *\n",
            }
        )
        with self.assertRaisesRegex(ExportError, "cyclic star imports"):
            resolver.exports()

    def test_conditional_rebinding_is_not_silently_ignored(self):
        resolver = self.exports_from(
            {
                "__init__.py": "class Item: pass\nif enabled:\n    Item = replacement\n",
            }
        )
        with self.assertRaisesRegex(ExportError, "unsupported binding"):
            resolver.exports()

    def test_reads_the_pinned_source_once_even_if_checkout_changes(self):
        resolver = self.exports_from(
            {
                "__init__.py": "from dspy.items import First, Second\n",
                "items.py": "class First: pass\nclass Second: pass\n",
            }
        )
        (self.repository / "dspy/items.py").write_text("broken checkout")
        with patch.object(resolver.modules, "git", wraps=resolver.modules.git) as git:
            expected = {"First": "items.py::First", "Second": "items.py::Second"}
            self.assertEqual(resolver.exports(), expected)
            self.assertEqual(resolver.exports(), expected)
            self.assertEqual(git.call_count, 2)

    def test_source_read_failure_is_not_an_empty_module(self):
        resolver = self.exports_from(
            {"__init__.py": "from dspy.items import *\n", "items.py": "__all__ = []\n"}
        )
        with (
            patch.object(
                resolver.modules, "git", side_effect=ExportError("read failed")
            ),
            self.assertRaisesRegex(ExportError, "read failed"),
        ):
            resolver.exports()


class PinnedExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        modules = PinnedModules(gate.DSPY, gate.pinned_tree(gate.pinned_tag()))
        cls.exports = PublicExports(modules).exports()
        cls.ledger = gate.tomllib.loads(gate.LEDGER.read_text())
        cls.unported = gate.tomllib.loads(gate.UNPORTED.read_text())["modules"]

    def test_all_122_pinned_exports_have_qualified_decisions(self):
        self.assertEqual(len(self.exports), 122)
        self.assertEqual(
            gate.missing_exports(self.exports, self.ledger, self.unported), {}
        )
        self.assertEqual(self.exports["Tool"], "adapters/types/tool.py::Tool")
        self.assertEqual(self.exports["context"], "dsp/utils/settings.py::Settings")

    def test_removing_the_public_tool_decision_fails(self):
        ledger = {table: dict(rows) for table, rows in self.ledger.items()}
        del ledger["symbols"]["adapters/types/tool.py::Tool"]
        self.assertEqual(
            gate.missing_exports(self.exports, ledger, self.unported),
            {"Tool": "adapters/types/tool.py::Tool"},
        )

    def test_gate_reports_resolution_failure(self):
        output = io.StringIO()
        with (
            patch.object(
                gate.PublicExports,
                "exports",
                side_effect=ExportError("unreadable package"),
            ),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(gate.main(), 1)
        self.assertIn("unreadable package", output.getvalue())


if __name__ == "__main__":
    unittest.main()
