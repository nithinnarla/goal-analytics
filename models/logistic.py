"""
Logistic Regression model for football match outcome prediction.

Trained on historical World Cup matches (1930-2022, ~900 matches).
Predicts: P(home win), P(draw), P(away win)

Complements the Elo+Poisson model:
  - Elo+Poisson: scoreline distribution, expected goals (good for match predictor)
  - LogisticRegression: calibrated outcome probabilities (good for bracket simulation)

Usage
-----
    from models.logistic import LogisticMatchPredictor

    model = LogisticMatchPredictor()
    model.train(X, y)  # X, y from features.build_match_features_df()
    probs = model.predict(home_elo=2100, away_elo=2050, ...)
    # → {"home_win": 0.42, "draw": 0.28, "away_win": 0.30}
"""
import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn(
        "scikit-learn not installed. LogisticMatchPredictor will use Elo fallback.\n"
        "Install with: pip install scikit-learn",
        ImportWarning,
        stacklevel=2,
    )

from models.features import (
    build_feature_row,
    build_match_features_df,
    FEATURE_COLS,
    LABEL_MAP,
)


class LogisticMatchPredictor:
    """
    Multinomial logistic regression for 3-way match outcome prediction.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 500):
        self.pipeline = None
        if SKLEARN_AVAILABLE:
            self.pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    C=C,
                    multi_class="multinomial",
                    solver="lbfgs",
                    max_iter=max_iter,
                    random_state=42,
                )),
            ])
        self.trained = False
        self.cv_accuracy: float | None = None
        self.n_train: int = 0

    # ─── Training ──────────────────────────────────────────────────────────────

    def train(self, X, y, cv: int = 5) -> "LogisticMatchPredictor":
        """
        Train on feature matrix X and outcome labels y {0,1,2}.
        Also runs stratified k-fold CV to report accuracy.
        """
        if not SKLEARN_AVAILABLE or self.pipeline is None:
            print("  sklearn not available — using Elo fallback instead of training.")
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
            print(
                f"  LogReg trained on {self.n_train} WC matches | "
                f"{cv}-fold CV accuracy: {self.cv_accuracy:.1%}"
            )
        return self

    @classmethod
    def train_from_history(
        cls,
        wc_df,
        elos: dict,
        form_lookup: dict,
        **kwargs,
    ) -> "LogisticMatchPredictor":
        """Convenience factory: train directly from WC match DataFrame."""
        model = cls(**kwargs)
        X, y = build_match_features_df(wc_df, elos, form_lookup)
        model.train(X, y)
        return model

    # ─── Prediction ─────────────────────────────────────────────────────────────

    def predict(
        self,
        home_elo: float,
        away_elo: float,
        is_neutral: bool = False,
        home_form: float = 1.5,
        away_form: float = 1.5,
        home_scored: float = 1.3,
        away_scored: float = 1.3,
        home_conceded: float = 1.1,
        away_conceded: float = 1.1,
    ) -> dict[str, float]:
        """
        Predict match outcome probabilities.
        Returns dict: {home_win, draw, away_win}
        """
        if not self.trained:
            # Fallback to Elo-derived probabilities
            return self._elo_fallback(home_elo, away_elo)

        x = build_feature_row(
            home_elo, away_elo, is_neutral,
            home_form, away_form,
            home_scored, away_scored,
            home_conceded, away_conceded,
        ).reshape(1, -1)

        proba = self.pipeline.predict_proba(x)[0]
        # Classes are sorted: 0=away_win, 1=draw, 2=home_win
        classes = self.pipeline.classes_
        prob_dict = {LABEL_MAP[c]: float(proba[i]) for i, c in enumerate(classes)}
        return prob_dict

    @staticmethod
    def _elo_fallback(home_elo: float, away_elo: float) -> dict[str, float]:
        """Pure Elo probability (no training required)."""
        exp_h = 1.0 / (1.0 + 10.0 ** ((away_elo - (home_elo + 100)) / 400.0))
        # Allocate draw probability proportional to closeness
        elo_diff = abs(home_elo - away_elo)
        draw_base = max(0.20, 0.30 - elo_diff / 2000.0)
        h = exp_h * (1.0 - draw_base)
        a = (1.0 - exp_h) * (1.0 - draw_base)
        return {"home_win": h, "draw": draw_base, "away_win": a}

    # ─── Model info ─────────────────────────────────────────────────────────────

    def feature_importance(self) -> dict[str, list[float]]:
        """
        Return coefficients per feature per class.
        Useful for understanding which features matter most.
        """
        if not self.trained:
            return {}
        coef = self.pipeline.named_steps["clf"].coef_
        return {
            LABEL_MAP[c]: coef[i].tolist()
            for i, c in enumerate(self.pipeline.classes_)
        }

    def summary(self) -> str:
        lines = [
            "LogisticMatchPredictor",
            f"  Trained: {self.trained}",
            f"  Training samples: {self.n_train}",
            f"  CV Accuracy: {self.cv_accuracy:.1%}" if self.cv_accuracy else "  CV: N/A",
            f"  Features: {FEATURE_COLS}",
        ]
        return "\n".join(lines)
