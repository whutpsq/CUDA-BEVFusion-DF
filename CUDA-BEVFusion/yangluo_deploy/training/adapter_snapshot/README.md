# Yangluo training adapter snapshot

This directory is the local, self-contained snapshot of the Yangluo dataset
adapter that originally lived at `H:/df_code/bevfusion/yangluo_adapter`.
The external training checkout is intentionally left unchanged.

The snapshot records the corrected training contract established by the
vehicle projection investigation:

- source point clouds and 3D boxes: RFU (`x` right, `y` front, `z` up);
- model/training coordinates: FLU (`x` front, `y` left, `z` up);
- front image topic/source `cam5`: camera calibration entry `0`;
- rear image topic/source `cam10`: camera calibration entry `10`.

`convert_yangluo.py` therefore uses:

```python
CAMERA_MAP = (("CAM_FRONT", "cam5", "0"), ("CAM_BACK", "cam10", "10"))
```

The mapping is explicit because live topic numbers are not a general promise
that they equal calibration IDs.  If the physical calibration changes again,
update this map and regenerate the PKLs before retraining.  A checkpoint trained
with a different camera mapping must not be reused as the final production
model; retrain, export ONNX, and rebuild the TensorRT plans on Thor.

For ResNet50 training, first copy or synchronize this directory into the
training BEVFusion checkout as `yangluo_adapter`, then use
`../make_resnet50_config.py` to derive the isolated ResNet50 configuration.
The generator removes old checkpoint resume/load settings and replaces the
camera backbone without changing the passenger-car path.
