# `pyspark-iceberg-reader` — Specification

> Status: **draft for build**. This document is the source of truth for requirements,
> the Iceberg v2 format notes gathered from research, the architecture, and the
> algorithms. `CLAUDE.md` is the short operating guide that points back here.

---

## 1. Purpose and goal

Build a small, source-only Python/PySpark package that **materializes an Apache Iceberg
v2 table as a Spark `DataFrame`, given only the location of a `*.metadata.json` file** and
the filesystem access already available to the running notebook or Lakeflow task.

The package is **not** distributed as a wheel. It is imported as plain source from the
workspace/repo. The only runtime assumption is that a `SparkSession` (`spark`) exists.

### Non-goals (for this version)

- No Iceberg **v3** support yet (deletion vectors, row lineage, variant/geo types,
  multi-arg transforms). The design must leave a clean seam to add it — see §13.
- No writes, no commits, no maintenance (compaction, expiry). **Read-only.**
- No catalog. We never register the table anywhere. We read from the metadata path.
- No external authentication. We use whatever filesystem access the cluster already has.

---

## 2. Public API

A single top-level function. Keyword-only options to keep call sites explicit.

```python
from pyspark_iceberg_reader import read_iceberg_table

df = read_iceberg_table(
    metadata_location: str,            # e.g. "s3://bucket/db/table/metadata/0000x-uuid.metadata.json"
    *,
    snapshot_id: int | None = None,    # default: the snapshot the metadata marks current
    normalizer: "PathNormalizer | None" = None,  # optional path-scheme remapper (§9)
    spark: "SparkSession | None" = None,
) -> "pyspark.sql.DataFrame"
```

### Parameter semantics

- **`metadata_location`** — full path to a single `*.metadata.json` (or `*.metadata.json.gz`).
  The caller picks *which* metadata file. We do **not** scan the `metadata/` folder to guess
  the newest one; the metadata file given is authoritative.
- **`snapshot_id`** — optional override.
  - If `None`: use the metadata's `current-snapshot-id`. If that is also null/absent, the
    table has no current data → return an **empty** DataFrame with the current schema.
  - If provided: it **must** exist in the metadata's `snapshots` list, else raise
    `SnapshotNotFoundError`. **Never** fall back to "latest" and **never** guess.
- **`normalizer`** — optional `PathNormalizer` (a `str → str` callable). Use
  `DEFAULT_NORMALIZER` for built-in scheme remapping (`s3://` → `s3a://`, `wasbs://` →
  `abfss://`, etc.) when the cluster's filesystem differs from what the writer recorded.
  Pass `None` (default) to use paths verbatim. See §9.
- **`spark`** — if `None`, resolve via `SparkSession.getActiveSession()` then
  `SparkSession.builder.getOrCreate()`. In the target environment a session always exists.

### Returns / errors

- Returns a lazy Spark `DataFrame` over the table's **current schema** at the chosen snapshot,
  with all live data files included and all applicable delete files applied.
- Typed exceptions (see §10): `UnsupportedFormatVersionError`, `SnapshotNotFoundError`,
  `MetadataParseError`, `UnsupportedFeatureError`.

---

## 3. Runtime environment and hard constraints

| Item | Value |
|---|---|
| Platform | Databricks Runtime **17.3 LTS** |
| Engine | Apache **Spark 4.0.0**, Scala 2.13 |
| Python | **3.11.x** or newer |
| Iceberg format version | **v2 only** (reject v1 with a clear message, or read v1-compatible metadata defensively — see §13) |
| Cluster JARs | **None may be added.** The OSS `iceberg-spark-runtime` path is explicitly out (it needs a catalog backing that the platform forbids). |
| `pip` packages | At most an **optional** `pyiceberg`; the package must run fully without it. |
| Filesystem | DBFS, AWS S3, Azure storage — accessed **only** through Spark, using existing cluster credentials. No SDK auth, no credential passing. |
| Notebook-only APIs | Avoid `dbutils`/`display`. The package must work in a plain Python task. The only runtime API used is the `SparkSession`. |

### Spark capabilities we rely on (verify early on DBR 17.3)

- **Avro reader**: `spark.read.format("avro")` (bundled in Databricks Runtime). Used for the
  manifest list and manifest files in the native planner.
- **Parquet reader** with **field-ID matching**: `spark.sql.parquet.fieldId.read.enabled`
  (Spark ≥ 3.3). Lets us project by Iceberg field ID instead of by name — required for schema
  evolution (renames/reorders/added columns). See §7.4.
- **File metadata columns**: `_metadata.file_path` and `_metadata.row_index` (parquet,
  Spark ≥ 3.5). `row_index` gives the in-file row position needed for position deletes.
  Note: `input_file_name()` is removed in DBR 17.3 — **use `_metadata.file_path`**.

> Each of these three must be validated against DBR 17.3 in the first integration test
> (`tests/test_runtime_capabilities.py`). If `_metadata.row_index` behaves unexpectedly,
> position-delete handling needs a fallback (§7.6).

---

## 4. Apache Iceberg v2 — research notes (the important bits)

These notes are a working summary distilled from the Iceberg table spec; they exist so the
implementer does not have to re-derive the rules. The spec is the final authority.

### 4.1 On-disk layout

```
<table-root>/
├── data/                       # data files (Parquet here) + delete files may also live here
│   └── ... .parquet
└── metadata/
    ├── <version>-<uuid>.metadata.json     # table metadata; several exist, one is "current"
    ├── snap-<snapshot-id>-<n>-<uuid>.avro  # manifest LIST, one per snapshot
    ├── <uuid>-m<n>.avro                    # manifest files (data and/or delete manifests)
    └── (version-hint.text)                 # only with Hadoop tables; we ignore it — caller gives the path
```

Iceberg writes **absolute** paths inside metadata by default. We generally use them verbatim
(§9 covers scheme/mount normalization).

### 4.2 `metadata.json` (the fields we use)

- `format-version` → must be `2` (gate; see §13 for v1-in-v2 tolerance).
- `table-uuid`, `location`.
- `current-schema-id` + `schemas[]` → the **current schema** is the one whose `schema-id`
  matches `current-schema-id`. Each schema lists fields with **stable `id`s** (field IDs).
- `default-spec-id` + `partition-specs[]` → partition specs (there can be **several**; partition
  evolution). Each spec has `spec-id` and fields with `source-id`, `field-id`, `transform`,
  `name`.
- `current-snapshot-id` → the snapshot the metadata marks current (**our default**).
- `snapshots[]` → each: `snapshot-id`, `parent-snapshot-id`, `sequence-number`, `timestamp-ms`,
  `manifest-list` (path to the manifest-list Avro), `summary` (incl. `operation`), `schema-id`.
- `refs` → named branches/tags; `main` usually points at `current-snapshot-id`. **Out of scope**
  for selection in v1 (we select by `current-snapshot-id` or explicit `snapshot_id`).
- `last-sequence-number`, `snapshot-log`, `metadata-log`, `sort-orders`, `properties`.

### 4.3 Manifest list (Avro) — one per snapshot

Each row is a `manifest_file` describing one manifest:

- `manifest_path`, `manifest_length`
- `partition_spec_id` → which spec the manifest's files were partitioned by
- `content` → **`0` = data manifest, `1` = delete manifest** (v2)
- `sequence_number`, `min_sequence_number` → used for **inheritance** (§4.5)
- `added_snapshot_id`
- counts: `added_files_count`, `existing_files_count`, `deleted_files_count`, row counts
- `partitions` → per-partition-field value summaries (for pruning; optional to use)

### 4.4 Manifest file (Avro) — lists data or delete files

Each row is a `manifest_entry`:

- `status` → **`0` = EXISTING, `1` = ADDED, `2` = DELETED**
- `snapshot_id` (nullable, inheritable)
- `sequence_number` → the **data sequence number** (nullable, inheritable)
- `file_sequence_number` (nullable, inheritable)
- `data_file` (struct):
  - `content` → **`0` = DATA, `1` = POSITION DELETES, `2` = EQUALITY DELETES** (v2)
  - `file_path`, `file_format` (`PARQUET`/`ORC`/`AVRO`)
  - `partition` (struct of partition values under this manifest's `partition_spec_id`)
  - `record_count`, `file_size_in_bytes`
  - column stats: `column_sizes`, `value_counts`, `null_value_counts`, `nan_value_counts`,
    `lower_bounds`, `upper_bounds` (maps keyed by field ID; usable for pruning, optional)
  - `equality_ids` → for equality deletes: the field IDs that define equality
  - `split_offsets`, `sort_order_id`, `key_metadata`

### 4.5 Inheritance (critical, easy to get wrong)

A manifest entry's `sequence_number` / `file_sequence_number` / `snapshot_id` may be **null** to
avoid rewriting files on commit retries. Resolve as:

- If `status == 1` (ADDED) and the value is null → **inherit** from the `manifest_file` in the
  manifest list (`sequence_number`; and `added_snapshot_id` for the snapshot id).
- If `status == 0` (EXISTING) or `2` (DELETED) → the values **must be present explicitly**; do
  not inherit. (A missing value here is a corrupt/unsupported manifest → raise.)

The **data sequence number is immutable** once a file is added and is what governs delete
application. The file sequence number is **not** usable for delete pruning.

### 4.6 What is "live" at a snapshot

For the chosen snapshot: read its manifest list → the referenced manifests. Within those
manifests, a file entry is **live** iff `status ∈ {ADDED, EXISTING}` (i.e. exclude `DELETED`
tombstones). This applies to both data manifests and delete manifests.

### 4.7 Delete files and how they apply (the correctness core)

Two kinds in v2 (deletion vectors are v3):

- **Position deletes** (`content = 1`): rows of `(file_path, pos)` identifying the row position
  to drop in a specific data file. May also carry the deleted `row` (ignored by us). They are
  **path-scoped**.
- **Equality deletes** (`content = 2`): rows giving values for the columns named by
  `equality_ids`; any data row matching those values is dropped. **Null matches null**
  (Iceberg uses null-safe equality here, unlike SQL `=`).

**Application rules, by data sequence number, same partition:**

- A **position** delete `P` applies to data file `D` when `P.seq >= D.seq` (the `>=` lets a
  delete remove rows added in the *same* commit) **and** `P` references `D`'s path. Because a
  data file's path is unique and a position delete referencing it can only be written at or
  after the file's creation, in practice a position-delete row that names `D`'s path always
  applies. → Implement as an anti-join on `(file_path, pos)`.
- An **equality** delete `E` applies to data file `D` when `E.seq > D.seq` (strictly greater)
  **and** same partition. The strict `>` is what prevents an equality delete from eating
  newly-inserted rows with the same key. → Implement as a sequence-gated, partition-scoped
  anti-join on the equality columns (null-safe).

> Column metrics on delete files (lower/upper bounds, counts) can prune which deletes overlap a
> data file. This is an **optimization**, not correctness; defer it (§8).

### 4.8 Schema evolution and field IDs

- Columns are tracked by **stable field ID**, not name. A rename changes the name, not the ID.
- Data files written under older schemas may have fewer/renamed/reordered columns. To produce
  the **current** schema we must map physical columns to logical columns **by field ID**,
  fill **null** for columns added after a file was written, and drop columns not in the current
  schema. → Parquet field-ID read (§7.4).
- **Type promotion** is allowed (e.g. `int`→`long`, `float`→`double`, decimal precision
  increase). Field-ID read alone does not always promote; we cast post-read where needed
  (§7.4, and the limitation in §11).

### 4.9 Hidden partitioning (affects output)

Partition values come from **transforms** of source columns and are **not** separate stored
columns (except identity transforms, where the source column itself is in the data file).
Therefore the output DataFrame is exactly the **table's data columns** — we do **not** synthesize
partition columns. Partition values are used internally only for delete scoping/pruning.

---

## 5. Architecture

```
read_iceberg_table(metadata_location, snapshot_id, normalizer, spark)
        │
        ▼
   metadata.py        ── parse metadata.json (driver, tiny) → TableMetadata
        │                 resolve schema, partition specs, target snapshot
        ▼
   planning/native.py ── NativePlanner: reads manifest list + manifests via Spark Avro
        │
        ▼  ScanPlan  (the shared IR: data files + delete files, with seq/partition/spec)
        │
        ▼
   execution.py       ── Spark engine:
        │                 read data files → apply position deletes → apply equality deletes
        │                 → project to current schema (field-ID) → DataFrame
        ▼
   pyspark.sql.DataFrame
```

The key design idea: **planning and execution are separate concerns.** The heavy, must-be-scalable
work (reading data, applying deletes) lives once in `execution.py`. This is the main DRY lever.

### 5.1 The shared IR (`planning/base.py`)

```python
@dataclass(frozen=True)
class DataFileTask:
    path: str
    file_format: str            # "PARQUET" (others rejected in v1, §11)
    spec_id: int
    partition: tuple            # normalized partition key (spec_id-qualified)
    data_sequence_number: int
    record_count: int

@dataclass(frozen=True)
class DeleteFile:
    path: str
    file_format: str
    content: int                # 1 = position, 2 = equality
    spec_id: int
    partition: tuple
    data_sequence_number: int
    equality_field_ids: tuple   # empty for position deletes

@dataclass(frozen=True)
class ScanPlan:
    snapshot_id: int
    current_schema: "IcebergSchema"     # resolved current schema (field IDs + types)
    partition_specs: dict[int, "PartitionSpec"]
    data_files: list[DataFileTask]      # may be large; see §8 for memory notes
    delete_files: list[DeleteFile]
```

### 5.2 Planner interface

```python
class ScanPlanner(Protocol):
    def plan(self, metadata: TableMetadata, snapshot_id: int) -> ScanPlan: ...
```

`NativePlanner` in `native.py` implements this protocol. It reads manifest lists and manifest
files via Spark Avro, applies the inheritance rules (§4.5), filters to live files (§4.6), and
returns the global sequence-gated form. `execution.py` computes the delete gate itself from the
sequence numbers in the plan.

---

## 6. Proposed module layout

```
pyspark-iceberg-reader/
├── CLAUDE.md
├── specification.md
├── README.md
├── pyproject.toml                      # metadata + dev tooling only (not for cluster install)
├── src/pyspark_iceberg_reader/
│   ├── __init__.py                     # exports read_iceberg_table
│   ├── reader.py                       # top-level orchestration
│   ├── metadata.py                     # metadata.json parse + snapshot resolution
│   ├── schema.py                       # Iceberg schema <-> Spark StructType, field-ID metadata
│   ├── manifests.py                    # Avro manifest-list / manifest reading + inheritance (native)
│   ├── execution.py                    # shared Spark read + delete application + projection
│   ├── deletes.py                      # position + equality delete application helpers
│   ├── paths.py                        # path scheme/mount normalization + Spark file reads
│   ├── errors.py                       # typed exceptions
│   └── planning/
│       ├── __init__.py                 # sub-package marker
│       ├── base.py                     # ScanPlan IR + ScanPlanner Protocol
│       └── native.py                   # NativePlanner (Spark Avro)
└── tests/
    ├── test_runtime_capabilities.py    # avro, field-id read, _metadata.row_index on DBR 17.3
    ├── test_metadata.py                # snapshot resolution, v1 rejection, empty snapshot
    ├── test_native_planner.py          # inheritance, status filtering, IR shape
    ├── test_deletes_position.py
    ├── test_deletes_equality.py        # sequence gating + null-safe matching
    └── test_schema_evolution.py        # add/drop/rename/reorder, type promotion
```

---

## 7. Algorithms in detail

### 7.1 Reading the metadata.json (driver, small)

Read it through Spark so we inherit cluster filesystem access and avoid SDKs:

```python
text = spark.read.text(metadata_location, wholetext=True).first()[0]
meta = json.loads(text)
```

`spark.read.text` transparently handles a `.gz` metadata file by extension. Validate
`format-version == 2` (or v1-tolerant per §13). Build `TableMetadata`.

### 7.2 Snapshot resolution (`metadata.py`)

```
if snapshot_id is None:
    target = meta["current-snapshot-id"]
    if target is None: return empty_df(current_schema)   # no data yet
else:
    target = snapshot_id
    if target not in {s["snapshot-id"] for s in meta["snapshots"]}:
        raise SnapshotNotFoundError(target)
```

Resolve the snapshot's `manifest-list` path and its `schema-id`. The **current schema** for
output is `current-schema-id` (the table's current schema), per the read-the-table-as-it-is-now
convention; the snapshot's own `schema-id` is used to interpret older files via field IDs.

### 7.3 Native planning (`native.py` + `manifests.py`)

All reads via Spark so this scales to very large manifest counts.

1. **Read the manifest list** (one Avro file):
   `spark.read.format("avro").load(manifest_list_path)`.
   Split rows by `content`: `0` → data manifests, `1` → delete manifests. Keep
   `manifest_path`, `partition_spec_id`, `sequence_number`, `added_snapshot_id`.
2. **Read data manifests** in bulk: `spark.read.format("avro").load(*data_manifest_paths)`.
   - Apply **inheritance** (§4.5): fill `sequence_number`/`snapshot_id` from the owning
     manifest where null and `status == 1`.
   - Keep entries with `status ∈ {0,1}`. Project `data_file.file_path`, `file_format`,
     `partition`, `record_count`, resolved `data_sequence_number`, `partition_spec_id`.
3. **Read delete manifests** the same way → `DeleteFile`s with `content`, `equality_ids`,
   `data_sequence_number`, `partition`, `spec_id`.
4. Materialize the file lists into the `ScanPlan`. See §8 for how much we pull to the driver
   and how to avoid blowing it up.

> Inheritance must be done per owning manifest, so we cannot blindly union all manifests before
> filling sequence numbers. Either read manifest-by-manifest attaching its `sequence_number` as a
> literal column, or union with the owning manifest's `sequence_number` carried alongside.

### 7.4 Schema mapping and field-ID projection (`schema.py`)

To read older files into the current schema robustly:

1. Convert the Iceberg **current schema** to a Spark `StructType` where each field carries its
   Iceberg field ID in metadata: `StructField(name, sparkType, metadata={"parquet.field.id": id})`,
   recursively for structs/lists/maps.
2. Enable field-ID matching for the read:
   `spark.conf.set("spark.sql.parquet.fieldId.read.enabled", "true")` (set narrowly; restore if
   we change session conf — prefer reading with an explicit `.schema(...)` so Spark maps by ID).
3. Read data with `.schema(read_schema)`. Spark then matches parquet columns by field ID:
   renames/reorders resolve correctly, columns added later read as **null** for older files.
4. **Type promotion**: where a file stores a narrower type than the current schema (e.g. `int`
   under a now-`long` column), apply an explicit `cast` to the current type after read. Track the
   set of promotions from the schema diff. (Known limitation for exotic promotions — §11.)

### 7.5 Data read strategy (`execution.py`)

Two modes, chosen from the plan:

- **No equality deletes present** → bulk read: `spark.read.schema(read_schema).parquet(*data_paths)`,
  carrying `_metadata.file_path` and `_metadata.row_index`. Apply position deletes (§7.6).
  Single job, maximal parallelism.
- **Equality deletes present** → we need each data row tagged with its file's
  `(data_sequence_number, spec_id, partition)` for gating. Group data files by the distinct
  `(spec_id, partition, data_sequence_number)` tuple — typically far fewer groups than files —
  read each group with those values attached as literal columns, and `unionByName` the groups.
  Then apply position deletes (§7.6) and equality deletes (§7.7).

Both modes end by **projecting to the current schema** (dropping the internal `_metadata`/seq/
partition helper columns).

### 7.6 Position-delete application (`deletes.py`)

```
pos = spark.read.parquet(*position_delete_paths)          # columns: file_path, pos
data = data.withColumn("_rid", col("_metadata.row_index"))
          .withColumn("_fp",  col("_metadata.file_path"))
result = data.join(
    pos.select(col("file_path").alias("_fp"), col("pos").alias("_rid")),
    on=["_fp", "_rid"], how="left_anti"
)
```

Notes: normalize `file_path` on both sides (§9) before joining, since the delete file stores the
data file's path exactly as the writer wrote it. If `_metadata.row_index` proves unreliable on
DBR 17.3, fall back to a per-file `row_number()` over an order that matches physical order — this
is fragile, so the runtime test in §3 gates it.

### 7.7 Equality-delete application (`deletes.py`)

Per equality delete "shape" (the set of `equality_field_ids`) and per partition group:

```
For each equality delete file E (content=2):
  eq_cols = current names of fields in E.equality_field_ids
  E_rows  = spark.read.parquet(E.path).select(eq_cols).distinct()
  # E applies to data rows where data.data_sequence_number < E.seq
  #   AND data.partition == E.partition (same spec)
  candidate = data.where((col("_seq") < E.seq) & partition_matches(E))
  deleted   = candidate.join(E_rows, on=[eqNullSafe(c) for c in eq_cols], how="left_semi")
  data      = data.subtract-by-key(deleted)     # implement as a keyed left-anti, not RDD.subtract
```

Implementation reality: do this efficiently by unioning all equality-delete rows of the same
shape with their `seq` and `partition`, then a single gated null-safe anti-join rather than a
Python loop over files. Use `eqNullSafe` (`<=>`) for every equality column so **null matches
null** per §4.7. Tag data rows with a stable surrogate id (e.g. `_fp` + `_rid`) so the anti-join
removes exactly the matched rows without disturbing duplicates that should remain.

### 7.8 Final assembly (`reader.py`)

`parse metadata → resolve snapshot → NativePlanner.plan() → execute_scan_plan(plan) → DataFrame`.
Return lazily; do not `.collect()` data. The only driver-side materialization is metadata-scale
(file lists), per §8.

---

## 8. Scalability design notes

- **All bulk reads go through Spark**: manifests (Avro) and data/deletes (Parquet). Nothing about
  data volume is handled on the driver.
- **Driver-side footprint is metadata-scale**: the file *lists* in `ScanPlan`. For tables with
  millions of files this can still be hundreds of MB of path strings. Mitigations, in order:
  1. Project manifests to only the columns needed before collecting.
  2. Group/dedupe by `(spec_id, partition, seq)` so the data-read driver structure is the number
     of groups, not files, where possible.
  3. If file counts are extreme, **batch** the data read (read in chunks and `unionByName`),
     keeping any single driver list bounded.
- **Avoid wide shuffles** in delete application: position deletes are typically small relative to
  data; broadcast the position/equality delete side when the optimizer doesn't already.
- **Never** read the whole `data/` folder. Only files named by the snapshot's manifests are valid;
  the folder also contains orphans and other snapshots' files.

---

## 9. Filesystem and path handling (`paths.py`)

- Access is **only** via Spark; no cloud SDKs, no credentials in code.
- Iceberg stores absolute paths. Schemes seen: `dbfs:/...`, `s3://...`/`s3a://...`,
  `abfss://...@...dfs.core.windows.net/...`, `wasbs://...`. Pass them to Spark as-is by default.
- Provide a small, optional **path normalization/remap** hook for the case where the metadata's
  recorded scheme/host differs from what the cluster mounts (e.g. `s3a` vs `s3`, or a mounted
  path). This is also applied symmetrically to both sides of delete joins so paths compare equal.
- Resolve any relative paths against the table `location` from the metadata (rare, but spec-legal).

---

## 10. Error handling and validation

| Exception | Raised when |
|---|---|
| `UnsupportedFormatVersionError` | `format-version` not handled (e.g. v3, or v1 if we choose to reject) |
| `MetadataParseError` | metadata.json missing required fields / malformed |
| `SnapshotNotFoundError` | explicit `snapshot_id` absent from `snapshots` |
| `UnsupportedFeatureError` | non-Parquet data files, unsupported transform, corrupt inheritance, etc. |

Validation principle: **fail loudly and specifically** rather than silently returning wrong rows.
A wrong row count from a missed delete is the worst outcome; prefer an explicit error.

---

## 11. Known limitations and open risks

1. **Equality deletes are the highest-risk area.** Sequence gating + partition scoping + null-safe
   matching must be exactly right. This is the priority for tests (and the PyIceberg oracle, §12).
2. **Type promotion**: common promotions (`int`→`long`, `float`→`double`, decimal precision) are
   handled by post-read cast; uncommon ones may need explicit support — fail with
   `UnsupportedFeatureError` rather than guess.
3. **Partition-spec evolution**: when files in one snapshot span multiple specs, partition equality
   for delete scoping must compare within compatible specs. v1 handles identity and single-active
   spec cleanly; multi-spec equality-delete scoping is a tested edge case and a candidate for the
   PyIceberg planner.
4. **PyIceberg storage access**: PyIceberg uses its own FileIO (fsspec/pyarrow) and may **not**
   inherit Databricks' S3/ADLS credentials automatically, so the `"pyiceberg"` planner can require
   credential configuration that `"native"` avoids. Keep `"native"` the default; document this.
5. **`_metadata.row_index`** dependency for position deletes — gated by the runtime test (§3).
6. **Driver memory** on tables with extreme file counts — see §8 batching.
7. **ORC/Avro data files** are out of scope in v1 (we expect Parquet). Reject with a clear error.

---

## 12. Testing strategy

- **Runtime capability test** first (§3): proves avro read, field-ID read, and
  `_metadata.row_index` on DBR 17.3.
- **Unit tests** with hand-built fixtures (tiny Iceberg tables generated once and committed to
  `tests/fixtures/`): copy-on-write, position-delete, equality-delete, mixed, and a
  schema-evolved table (add/drop/rename/reorder + one promotion).
- **Oracle parity test**: when `pyiceberg` is importable, compare the native planner's resolved
  file set and the final DataFrame (sorted, schema-aligned) against PyIceberg's read of the same
  metadata. This catches subtle delete/inheritance bugs cheaply.
- **Snapshot tests**: default snapshot == `current-snapshot-id`; explicit older `snapshot_id`
  returns the historical view; bad id raises; null current → empty.

---

## 13. Forward path to Iceberg v3

Keep these seams clean so v3 is additive, not a rewrite:

- Version gate lives in one place (`metadata.py`). v3 adds **deletion vectors** (Puffin) which
  replace position-delete files — `deletes.py` gets a third strategy behind the same interface.
- New types (variant, geometry/geography) and **multi-argument transforms** touch `schema.py` and
  partition handling only.
- **Row lineage** is read-side metadata; ignore unless surfaced.
- The spec is explicitly read-permissive about v1-compatible metadata appearing inside a v2 table
  (allowed by the spec so tables upgrade without rewriting). Decide deliberately whether to accept
  such mixed metadata or reject; document the choice in `metadata.py`.

---

## 14. Coding standards

- Python 3.11 typing throughout; `from __future__ import annotations`.
- Pure functions where possible; the `SparkSession` is passed in, never created implicitly deep in
  the call tree.
- DRY: one Iceberg-type-mapping table, one path-normalization function, one delete-join helper.
  Both planners share the IR and the execution engine.
- No `print`; use `logging`. No `dbutils`/`display`. No global session mutation that isn't restored.
- Small, focused modules per §6; each public function documented with the spec section it implements.
