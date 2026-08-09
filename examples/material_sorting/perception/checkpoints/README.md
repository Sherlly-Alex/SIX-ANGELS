# Perception Checkpoints

Place the offline competition detector at:

```text
best.pt
```

The current checkpoint contains these model labels:

```text
0 pink
1 yellow
2 brown
3 material_box
4 packaging_box
```

`perception/box_detect.py` reads `model.names` at startup and rejects a model
whose label set does not match these five canonical names.
