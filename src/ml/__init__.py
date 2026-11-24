"""The machine-learning pieces that the ML operators (operations/ml/) are built from (DESIGN.md sections 11, 17).

- features.py    ColumnSpec: the feature columns of a model; reading them as numbers
- algorithms.py  the algorithms a chain may name (scikit-learn), with their allowed parameters
- trained.py     TrainedModel (estimator + features + card) and AdaptedEstimator (source + local correction)
- metrics.py     forecast errors: MAE, RMSE, MAPE
- profiles.py    the hour-of-day profile of a series: the embedding of a microgrid
- vectors.py     distance and cosine similarity of two vectors
- sharing.py     models between sandboxes: export a model as plain numbers, load one (LinearWeights)

The models are deliberately simple: the point of the sandbox is how capabilities are obtained, derived and
reused across microgrids, not the accuracy of the models.
"""
