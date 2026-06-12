"""
XGBoost model for football match outcome prediction.

Designed to be trained on the "recent era" (2010+) slice of the full
international results dataset (data.historical.fetch_results()), using
gradient-boosted trees over the same 6-feature vector as the other
predictors. Predicts: P(home win), P(draw), P(away win)

Complements the existing models:
  - Elo+Poisson:        scoreline distribution, expected goals
  - LogisticRegression: calibrated linear baseline, WC-only training set
  - RandomForest:       bagged-tree ensemble, recent-era history
  - XGBoost:            boosted-tree ensemble, recent-era history — used
                         as a further independent cross-check

Usage
-----
    from models.xgboost_model import XGBoostMatchPredictor

    model = XGBoostMatchPredictor()
    model.train(X, y)  # X, y from features.build_match_features_df()
    probs = model.predict(home_elo=2100, away_elo=2050, ...)
    # → {"home_win": 0.42, "draw": 0.28, "away_win": 0.30}
"""
import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn(
        "xgboost not installed. XGBoostMatchPredictor will use Elo fallback.\n"
        "Install with: pip install xgboost",
        ImportWarning,
        stacklevel=2,
    )

from models.features import (
    build_feature_row,
    build_match_features_df,
    FEATURE_COLS,
    LABEL_MAP,
)


class XGBoostMatchPredictor:
    """
    Gradient-boosted-tree classifier over the same 6-feature vector as
    LogisticMatchPredictor / RandomForestMatchPredictor, intended to be
    trained on the recent-era (2010+) slice of international results.
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 4,
                 learning_rate: float = 0.05, random_state: int = 42):
        self.pipeline = None
        if XGBOOST_AVAILABLE:
            self.pipeline = Pipeline([
                ("clf", XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    random_state=random_state,
                    n_jobs=-1,
                )),
            ])
        self.trained = False
        self.cv_accuracy: float | None = None
        self.n_train: int = 0

    def train(self, X, y, cv: int = 5) -> "XGBoostMatchPredictor":
        if not XGBOOST_AVAILABLE or self.pipeline is None:
            print("  xgboost not available — using Elo fallback instead of training.")
            return self
        self.pipeline.fit(X, y)
        self.trained = True
        self.n_train = len(y)
        if len(np.unique(y)) == 3:
            cv_scores = cross_val_score(
                self.pipeline, X, y,
                cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
                scoring="accuracy",
            )
            self.cv_accuracy = float(np.mean(cv_scores))
            print(f"  XGBoost trained on {self.n_train} matches | {cv}-fold CV accuracy: {self.cv_accuracy:.1%}")
        return self

    @classmethod
    def train_from_history(cls, wc_df, elos: dict, form_lookup: dict, **kwargs) -> "XGBoostMatchPredictor":
        model = cls(**kwargs)
        X, y = build_match_features_df(wc_df, elos, form_lookup)
        model.train(X, y)
        return model

    def predict(self, home_elo, away_elo, is_neutral=False, home_form=1.5, away_form=1.5,
                 home_scored=1.3, away_scored=1.3, home_conceded=1.1, away_conceded=1.1) -> dict[str, float]:
        if not self.trained:
            return self._elo_fallback(home_elo, away_elo)
        x = build_feature_row(home_elo, away_elo, is_neutral, home_form, away_form,
                               home_scored, away_scored, home_conceded, away_conceded).reshape(1, -1)
        proba = self.pipeline.predict_proba(x)[0]
        classes = self.pipeline.classes_
        prob_dict = {LABEL_MAP[c]: float(proba[i]) for i, c in enumerate(classes)}
        return prob_dict

    @staticmethod
    def _elo_fallback(home_elo: float, away_elo: float) -> dict[str, float]:
        exp_h = 1.0 / (1.0 + 10.0 ** ((away_elo - (home_elo + 100)) / 400.0))
        elo_diff = abs(home_elo - away_elo)
        draw_base = max(0.20, 0.30 - elo_diff / 2000.0)
        h = exp_h * (1.0 - draw_base)
        a = (1.0 - exp_h) * (1.0 - draw_base)
        return {"home_win": h, "draw": draw_base, "away_win": a}

    def feature_importance(self) -> dict[str, float]:
        if not self.trained:
            return {}
        importances = self.pipeline.named_steps["clf"].feature_importances_
        return dict(zip(FEATURE_COLS, importances.tolist()))

    def summary(self) -> str:
        lines = [
            "XGBoostMatchPredictor",
            f"  Trained: {self.trained}",
            f"  Training samples: {self.n_train}",
            f"  CV Accuracy: {self.cv_accuracy:.1%}" if self.cv_accuracy else "  CV: N/A",
            f"  Features: {FEATURE_COLS}",
        ]
        return "\n".join(lines)
