"""Resolve public DSPy bindings from a pinned Git tree without importing DSPy."""

from __future__ import annotations

import ast
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from importlib.util import resolve_name
from pathlib import Path


class ExportError(ValueError):
    pass


class PinnedModules:
    def __init__(self, repository: Path, tree: str):
        self.repository = repository
        self.tree = tree
        self.paths: dict[str, str] = {}
        self.syntax: dict[str, ast.Module] = {}
        for path in self.git(
            "ls-tree", "-r", "--name-only", tree, "--", "dspy"
        ).splitlines():
            if path.endswith(".py"):
                module = (
                    path.removesuffix(".py").removesuffix("/__init__").replace("/", ".")
                )
                if module in self.paths:
                    raise ExportError(f"ambiguous module {module}")
                self.paths[module] = path

    def git(self, *arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(self.repository), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise ExportError(
                f"git {' '.join(arguments)} failed: {result.stderr.strip()}"
            )
        return result.stdout

    def path(self, module: str) -> str:
        if module not in self.paths:
            raise ExportError(f"cannot read {module} at {self.tree}")
        return self.paths[module]

    def read(self, module: str) -> ast.Module:
        if module in self.syntax:
            return self.syntax[module]
        path = self.path(module)
        try:
            self.syntax[module] = ast.parse(
                self.git("show", f"{self.tree}:{path}"), filename=path
            )
        except SyntaxError as error:
            raise ExportError(f"cannot parse {path}: {error}") from error
        return self.syntax[module]

    def imported_module(self, module: str, node: ast.ImportFrom) -> str:
        if not node.level:
            return node.module or ""
        package = (
            module
            if self.path(module).endswith("/__init__.py")
            else module.rpartition(".")[0]
        )
        try:
            return resolve_name("." * node.level + (node.module or ""), package)
        except ImportError as error:
            raise ExportError(
                f"invalid relative import in {module}: {error}"
            ) from error


@dataclass(frozen=True)
class ImportedName:
    module: str
    name: str | None


@dataclass(frozen=True)
class UnboundName:
    module: str
    name: str


@dataclass(frozen=True)
class DefinedName:
    module: str
    name: str
    value: ast.AST | None = None
    target: DefinedName | ImportedName | UnboundName | None = None


type ExportBinding = DefinedName | ImportedName | UnboundName
type Namespace = dict[str, DefinedName | ImportedName]


def assigned_names(node: ast.AST) -> Iterator[str]:
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        yield node.name
    elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
        yield node.id
    elif isinstance(node, ast.alias):
        yield node.asname or node.name.split(".")[0]
    else:
        for child in ast.iter_child_nodes(node):
            yield from assigned_names(child)


def original_binding(binding: ExportBinding | None) -> ExportBinding | None:
    while isinstance(binding, DefinedName) and binding.target is not None:
        binding = binding.target
    return binding


class PublicExports:
    def __init__(self, modules: PinnedModules):
        self.modules = modules
        self.loading: set[str] = set()
        self.namespaces: dict[str, Namespace] = {}

    def module_bindings(self, module: str) -> Namespace:
        if module in self.namespaces:
            return self.namespaces[module]
        if module in self.loading:
            raise ExportError(f"cyclic star imports in {module}")
        self.loading.add(module)
        try:
            bindings: Namespace = {}
            for statement in self.modules.read(module).body:
                bindings.update(self.statement_bindings(module, statement, bindings))
            self.namespaces[module] = bindings
            return bindings
        finally:
            self.loading.remove(module)

    def statement_bindings(
        self, module: str, statement: ast.stmt, namespace: Namespace
    ) -> Namespace:
        match statement:
            case ast.ImportFrom():
                return self.imported_names(module, statement)
            case ast.Import():
                return {
                    imported.asname or imported.name.split(".")[0]: ImportedName(
                        imported.name
                        if imported.asname
                        else imported.name.split(".")[0],
                        None,
                    )
                    for imported in statement.names
                }
            case ast.ClassDef() | ast.FunctionDef() | ast.AsyncFunctionDef():
                return {statement.name: DefinedName(module, statement.name)}
            case ast.Assign() | ast.AnnAssign():
                return self.assigned_bindings(module, statement, namespace)
            case _:
                return self.unsupported_bindings(module, statement, namespace)

    def imported_names(self, module: str, statement: ast.ImportFrom) -> Namespace:
        source = self.modules.imported_module(module, statement)
        bindings = {}
        for imported in statement.names:
            if imported.name == "*":
                bindings.update(
                    {
                        name: ImportedName(source, name)
                        for name in self.export_names(source)
                    }
                )
            else:
                bindings[imported.asname or imported.name] = ImportedName(
                    source, imported.name
                )
        return bindings

    def assigned_bindings(
        self, module: str, statement: ast.Assign | ast.AnnAssign, namespace: Namespace
    ) -> Namespace:
        if statement.value is None:
            return {}
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
        )
        if not all(isinstance(target, ast.Name) for target in targets):
            return self.unsupported_bindings(module, statement, namespace)
        referenced = self.expression_target(module, statement.value, namespace)
        return {
            target.id: DefinedName(module, target.id, statement.value, referenced)
            for target in targets
        }

    @staticmethod
    def unsupported_bindings(
        module: str, statement: ast.stmt, namespace: Namespace
    ) -> Namespace:
        names = set(assigned_names(statement))
        if "*" in names:
            raise ExportError(f"conditional star import in {module}")
        export_list = original_binding(namespace.get("__all__"))
        if export_list is not None and any(
            isinstance(node, ast.Name)
            and original_binding(namespace.get(node.id)) == export_list
            for node in ast.walk(statement)
        ):
            names.add("__all__")
        return {name: DefinedName(module, name, statement) for name in names}

    def expression_target(
        self, module: str, value: ast.AST | None, bindings: Namespace
    ) -> ExportBinding | None:
        match value:
            case ast.Name(id=name):
                return bindings.get(name, UnboundName(module, name))
            case ast.Call(func=function):
                return self.expression_target(module, function, bindings)
            case ast.Attribute(value=owner, attr=name):
                target = self.expression_target(module, owner, bindings)
                if isinstance(target, ImportedName) and target.name is None:
                    return ImportedName(target.module, name)
                return target
            case _:
                return None

    def resolve_definition(self, binding: ExportBinding) -> DefinedName:
        visited: set[tuple[str, str | None]] = set()
        while True:
            if isinstance(binding, UnboundName):
                raise ExportError(
                    f"name used before binding: {binding.module}.{binding.name}"
                )
            if isinstance(binding, ImportedName):
                key = (binding.module, binding.name)
                if key in visited:
                    raise ExportError(f"cyclic export {binding.module}.{binding.name}")
                visited.add(key)
                bindings = self.module_bindings(binding.module)
                if binding.name not in bindings:
                    raise ExportError(
                        f"unresolved export {binding.module}.{binding.name}"
                    )
                binding = bindings[binding.name]
            elif binding.target is not None:
                binding = binding.target
            elif binding.value is None or isinstance(binding.value, ast.expr):
                return binding
            else:
                raise ExportError(
                    f"unsupported binding {binding.module}.{binding.name}: {ast.unparse(binding.value)}"
                )

    def export_names(self, module: str) -> list[str]:
        binding = self.resolve_definition(ImportedName(module, "__all__"))
        try:
            names = ast.literal_eval(binding.value)
        except (ValueError, TypeError) as error:
            raise ExportError(
                f"{module}.__all__ must be a literal list or tuple of names"
            ) from error
        if not isinstance(names, (list, tuple)) or not all(
            isinstance(name, str) for name in names
        ):
            raise ExportError(
                f"{module}.__all__ must be a literal list or tuple of names"
            )
        return list(names)

    def exports(self) -> dict[str, str]:
        exports = {}
        bindings = self.module_bindings("dspy")
        if "__getattr__" in bindings:
            raise ExportError(
                "dynamic dspy.__getattr__ exports cannot be enumerated statically"
            )
        for name, binding in bindings.items():
            if not name.startswith("_"):
                definition = self.resolve_definition(binding)
                path = self.modules.path(definition.module).removeprefix("dspy/")
                exports[name] = f"{path}::{definition.name}"
        return exports
