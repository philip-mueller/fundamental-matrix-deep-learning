from train.utils.model_testing import test_model
from train.utils.metrics import *

# --- Testing the three models against the test set of the synthetic set with which it was trained ---
#results = test_model('models/final/bottleneck_only', epoch=25, dataset_name='synthetic_horse_rotation', norm='fro', metrics=[epi_abs, epi_sqr, ssd, sed], verbose=1)
#results = test_model('models/final/derived_features', epoch=35, dataset_name='synthetic_horse_rotation', norm='fro', metrics=[epi_abs, epi_sqr, ssd, sed], verbose=1)
results = test_model('models/final/derived_features_imgs', epoch=24, dataset_name='synthetic_horse_rotation', norm='fro', metrics=[epi_abs, epi_sqr, ssd, sed], verbose=1)


# --- Testing the three models against a thresholded (from a texture dataset) silhouette set for which it was not trained (bad results are expected) ---
# Note: Using TRAIN set as it contains more samples and here it is not important to use TEST set because it was neither trained nor tuned against any of these sets

#results = test_model('models/final/bottleneck_only', epoch=25, dataset_name='templeSparseRing_silhouette', type='TRAIN', norm='fro', metrics=[epi_abs, epi_sqr, ssd, sed], verbose=1)
#results = test_model('models/final/derived_features', epoch=35, dataset_name='templeSparseRing_silhouette', type='TRAIN', norm='fro', metrics=[epi_abs, epi_sqr, ssd, sed], verbose=1)
#results = test_model('models/final/derived_features_imgs', epoch=24, dataset_name='templeSparseRing_silhouette', type='TRAIN', norm='fro', metrics=[epi_abs, epi_sqr, ssd, sed], verbose=1)


print(results)

