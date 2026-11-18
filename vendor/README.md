# Vendored model implementations

These packages are installed as part of TinySegment, retaining upstream import namespaces for checkpoint compatibility. Edit them directly for experiments. Do not install overlapping upstream distributions into the same environment.

`provenance.json` identifies the imported versions. `source-hashes.json` records the initial imported baseline, including the documented EfficientViT patches; it is not automatically regenerated after experimental architecture changes. License and package metadata are preserved in `licenses/`.

EfficientViT includes SAM, backbone, inference NN operations, and utilities. Its model initializer exports only SAM, its NN initializer omits training-only drop layers, and optional/lazy Triton imports preserve CPU compatibility. The SAM architecture math is unchanged. Other vendored packages retain their imported Python sources unchanged.
