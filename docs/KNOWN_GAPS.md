# Known Gaps

- segmentation manifest building relies on filename heuristics and should be validated visually per dataset
- segmentation support is currently 2D TensorFlow/Keras, not full volumetric MONAI-style 3D MRI segmentation
- class semantics depend on the downloaded dataset's annotation quality
- no active-learning loop or model monitoring has been added yet
