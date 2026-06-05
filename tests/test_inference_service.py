import unittest
from unittest.mock import patch

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

    def test_default_predictor_uses_configured_pytorch_provider(self) -> None:
        with patch("src.inference.service.get_predictor", return_value=FixedPredictor()) as get_predictor:
            result = predict(image=None, metadata=self.metadata, variant="meta_only")

        get_predictor.assert_called_once_with("meta_only")
        self.assertEqual(result.provider, "fixed")
        self.assertEqual(result.predicted_class, "nv")

    def test_missing_weights_raise_configuration_error(self) -> None:
        with patch("src.inference.service.get_predictor", side_effect=FileNotFoundError("missing")):
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
