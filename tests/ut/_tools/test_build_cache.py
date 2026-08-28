# Copyright (c) 2026 Huawei Technologies Co., Ltd.

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
ENGINE = REPO_ROOT / "csrc" / "scripts" / "build_cache.py"


def _write_builder(tmp_path: Path) -> Path:
    builder = tmp_path / "builder.py"
    builder.write_text(
        """
from pathlib import Path
import sys

output_dir = Path(sys.argv[1])
counter = Path(sys.argv[2])
artifact_name = sys.argv[3]
mode = sys.argv[4]

count = int(counter.read_text()) if counter.exists() else 0
count += 1
counter.write_text(str(count))

output_dir.mkdir(parents=True, exist_ok=True)
content = "fixed-artifact" if mode == "fixed" else f"artifact-{count}"
(output_dir / artifact_name).write_text(content)
""",
        encoding="utf-8",
    )
    return builder


def _run_cache(
    *,
    cache_root: Path,
    prepared_input: Path,
    output_dir: Path,
    builder: Path,
    counter: Path,
    recipe_value: str,
    domain: str = "custom_operator",
    artifact_name: str = "kernel.o",
    mode: str = "counted",
    operator_source: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(ENGINE),
        "run",
        "--cache-root",
        str(cache_root),
        "--domain",
        domain,
        "--unit",
        "test_unit",
        "--output-dir",
        str(output_dir),
        "--prepared-input",
        str(prepared_input),
        "--recipe-value",
        recipe_value,
        "--environment-profile",
        "ascendc" if domain == "custom_operator" else "host-cxx",
        "--environment-tool",
        sys.executable,
    ]

    if domain == "third_party":
        command.extend(["--artifact-include", artifact_name])

    if domain == "custom_operator":
        source = operator_source or prepared_input
        command.extend(
            [
                "--soc",
                "ascend910b",
                "--operator",
                "test_operator",
                "--action",
                "TestOperator-0",
                "--operator-source",
                str(source),
            ]
        )

    command.extend(
        [
            "--",
            sys.executable,
            str(builder),
            str(output_dir),
            str(counter),
            artifact_name,
            mode,
        ]
    )

    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def test_custom_operator_hit_and_input_invalidation(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "kernel.cpp").write_text("int value = 1;\n", encoding="utf-8")

    prepared = tmp_path / "prepared"
    shutil.copytree(source, prepared)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    cache_root = tmp_path / "build_cache"
    counter = tmp_path / "counter"
    builder = _write_builder(tmp_path)

    first = _run_cache(
        cache_root=cache_root,
        prepared_input=prepared,
        operator_source=source,
        output_dir=output_dir,
        builder=builder,
        counter=counter,
        recipe_value="optimization=-O2",
    )
    assert first.returncode == 0, first.stderr
    assert "[build-cache] MISS" in first.stdout
    assert counter.read_text() == "1"

    shutil.rmtree(output_dir)
    output_dir.mkdir()

    second = _run_cache(
        cache_root=cache_root,
        prepared_input=prepared,
        operator_source=source,
        output_dir=output_dir,
        builder=builder,
        counter=counter,
        recipe_value="optimization=-O2",
    )
    assert second.returncode == 0, second.stderr
    assert "[build-cache] HIT" in second.stdout
    assert counter.read_text() == "1"
    assert (output_dir / "kernel.o").read_text() == "artifact-1"

    (prepared / "kernel.cpp").write_text("int value = 2;\n", encoding="utf-8")
    shutil.rmtree(output_dir)
    output_dir.mkdir()

    third = _run_cache(
        cache_root=cache_root,
        prepared_input=prepared,
        operator_source=source,
        output_dir=output_dir,
        builder=builder,
        counter=counter,
        recipe_value="optimization=-O2",
    )
    assert third.returncode == 0, third.stderr
    assert "[build-cache] MISS" in third.stdout
    assert counter.read_text() == "2"


def test_operator_text_hash_is_identity_layer_not_action_key(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "kernel.cpp").write_text("int source = 1;\n", encoding="utf-8")

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "kernel.cpp").write_text("prepared\n", encoding="utf-8")

    output = tmp_path / "output"
    output.mkdir()
    cache_root = tmp_path / "cache"
    counter = tmp_path / "counter"
    builder = _write_builder(tmp_path)

    first = _run_cache(
        cache_root=cache_root,
        prepared_input=prepared,
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
        recipe_value="same-recipe",
    )
    assert first.returncode == 0, first.stderr

    operator_root = cache_root / "custom_operator" / "ascend910b" / "test_operator"
    versions = [path for path in operator_root.iterdir() if path.is_dir()]
    assert len(versions) == 1
    first_version = versions[0].name

    (source / "kernel.cpp").write_text("int source = 2;\n", encoding="utf-8")
    shutil.rmtree(output)
    output.mkdir()

    second = _run_cache(
        cache_root=cache_root,
        prepared_input=prepared,
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
        recipe_value="same-recipe",
    )
    assert second.returncode == 0, second.stderr

    versions = [path for path in operator_root.iterdir() if path.is_dir()]
    assert len(versions) == 2
    assert first_version in {path.name for path in versions}


def test_recipe_change_invalidates_cache(tmp_path: Path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "source.cc").write_text("source\n", encoding="utf-8")

    output = tmp_path / "output"
    output.mkdir()
    cache_root = tmp_path / "cache"
    counter = tmp_path / "counter"
    builder = _write_builder(tmp_path)

    first = _run_cache(
        cache_root=cache_root,
        prepared_input=prepared,
        output_dir=output,
        builder=builder,
        counter=counter,
        recipe_value="optimization=-O2",
    )
    assert first.returncode == 0, first.stderr

    shutil.rmtree(output)
    output.mkdir()

    second = _run_cache(
        cache_root=cache_root,
        prepared_input=prepared,
        output_dir=output,
        builder=builder,
        counter=counter,
        recipe_value="optimization=-O0",
    )
    assert second.returncode == 0, second.stderr
    assert "[build-cache] MISS" in second.stdout
    assert counter.read_text() == "2"


def test_identical_rebuild_output_is_still_cacheable(tmp_path: Path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "source.cc").write_text("source\n", encoding="utf-8")

    output = tmp_path / "output"
    output.mkdir()
    cache_root = tmp_path / "cache"
    counter = tmp_path / "counter"
    builder = _write_builder(tmp_path)

    first = _run_cache(
        cache_root=cache_root,
        prepared_input=prepared,
        output_dir=output,
        builder=builder,
        counter=counter,
        recipe_value="recipe-a",
        mode="fixed",
    )
    assert first.returncode == 0, first.stderr

    # Keep the old output in place. Recipe changes force a MISS, but the
    # compiler rewrites exactly the same bytes. The artifact must still be
    # discovered via metadata change and saved under the new cache key.
    second = _run_cache(
        cache_root=cache_root,
        prepared_input=prepared,
        output_dir=output,
        builder=builder,
        counter=counter,
        recipe_value="recipe-b",
        mode="fixed",
    )
    assert second.returncode == 0, second.stderr
    assert "[build-cache] MISS" in second.stdout
    assert "[build-cache] SAVED" in second.stdout
    assert counter.read_text() == "2"


def test_corrupted_artifact_falls_back_to_rebuild(tmp_path: Path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "source.cc").write_text("source\n", encoding="utf-8")

    output = tmp_path / "output"
    output.mkdir()
    cache_root = tmp_path / "cache"
    counter = tmp_path / "counter"
    builder = _write_builder(tmp_path)

    first = _run_cache(
        cache_root=cache_root,
        prepared_input=prepared,
        output_dir=output,
        builder=builder,
        counter=counter,
        recipe_value="stable",
    )
    assert first.returncode == 0, first.stderr

    cached_artifacts = list(cache_root.rglob("artifacts/kernel.o"))
    assert len(cached_artifacts) == 1
    cached_artifacts[0].write_text("corrupted", encoding="utf-8")

    shutil.rmtree(output)
    output.mkdir()

    second = _run_cache(
        cache_root=cache_root,
        prepared_input=prepared,
        output_dir=output,
        builder=builder,
        counter=counter,
        recipe_value="stable",
    )
    assert second.returncode == 0, second.stderr
    assert "[build-cache] MISS" in second.stdout
    assert counter.read_text() == "2"


def test_third_party_cache_restores_products(tmp_path: Path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "third_party.cc").write_text("source\n", encoding="utf-8")

    output = tmp_path / "output"
    output.mkdir()
    cache_root = tmp_path / "cache"
    counter = tmp_path / "counter"
    builder = _write_builder(tmp_path)

    first = _run_cache(
        cache_root=cache_root,
        prepared_input=prepared,
        output_dir=output,
        builder=builder,
        counter=counter,
        recipe_value="build_shared_libs=OFF",
        domain="third_party",
        artifact_name="libtest.a",
    )
    assert first.returncode == 0, first.stderr

    shutil.rmtree(output)
    output.mkdir()

    second = _run_cache(
        cache_root=cache_root,
        prepared_input=prepared,
        output_dir=output,
        builder=builder,
        counter=counter,
        recipe_value="build_shared_libs=OFF",
        domain="third_party",
        artifact_name="libtest.a",
    )
    assert second.returncode == 0, second.stderr
    assert "[build-cache] HIT" in second.stdout
    assert counter.read_text() == "1"
    assert (output / "libtest.a").is_file()
