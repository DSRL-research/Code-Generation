"""The machine-learning pieces that the ML operators (operations/ml/) are built from (DESIGN.md section 11).

- features.py    FeatureSpec: lags of the target and calendar features of ts
- algorithms.py  the algorithms a chain may name (scikit-learn), with their allowed parameters
- trained.py     TrainedModel (estimator + features + card) and AdaptedEstimator (source + local correction)
- metrics.py     forecast errors: MAE, RMSE, MAPE
- profiles.py    the hour-of-day profile of a series: the embedding of a microgrid
- vectors.py     distance and cosine similarity of two vectors
- sharing.py     models between sandboxes: export a model as plain numbers, load one (LinearWeights)

The models are deliberately simple: the point of the sandbox is how capabilities are obtained, derived and
reused across microgrids, not the accuracy of the models.
"""
