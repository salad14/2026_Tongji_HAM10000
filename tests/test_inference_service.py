import unittest

from src.inference import PatientMetadata, PredictorNotConfiguredError, predict
from src.inference.schema import CLASS_LABELS, PredictionResult


class FixedPredictor:
    def predict(self, image, metadata, variant):
        probabilities = {label: 0.0 for label in CLASS_LABELS}
        probabilities["nv"] = 1.0
        return PredictionResult(
            variant=variant,
            probabilities=probabilities,
            predicted_class="nv",
            provider="fixed",
        )


class InferenceServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = PatientMetadata(age=50, sex="female", localization="back")

    def test_default_predictor_is_not_configured_before_model_integration(self) -> None:
        with self.assertRaises(PredictorNotConfiguredError):
            predict(image=None, metadata=self.metadata, variant="meta_only")

    def test_explicit_predictor_still_uses_the_shared_contract(self) -> None:
        result = predict(
            image=b"image",
            metadata=self.metadata,
            variant="fusion",
            predictor=FixedPredictor(),
        )

        self.assertEqual(result.provider, "fixed")
        self.assertEqual(result.predicted_class, "nv")
        self.assertEqual(sum(result.probabilities.values()), 1.0)


if __name__ == "__main__":
    unittest.main()
