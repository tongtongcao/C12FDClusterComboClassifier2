package org.example;

import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.translate.TranslateException;
import ai.djl.inference.Predictor;
import ai.djl.translate.Batchifier;

import java.io.IOException;
import java.nio.file.Paths;

/**
 * ===========================
 * Custom Input Class for Track Prediction
 * ===========================
 */
class TrackInput {

    /**
     * Feature array: length = 12 (6 avgWire + 6 slope)
     */
    float[] features;

    /**
     * Constructor: validates length and normalizes input features.
     *
     * @param features float array of length 12
     */
    public TrackInput(float[] features) {
        if (features.length != 12) {
            throw new IllegalArgumentException("Expected 12 features");
        }
        this.features = normalize(features);
    }

    /**
     * Normalize features.
     * - First 6 values (avgWire) are divided by 112.0
     * - Remaining 6 values (slope) are unchanged
     *
     * @param feats input feature array
     * @return normalized feature array
     */
    private float[] normalize(float[] feats) {
        float[] norm = new float[12];

        // Normalize avgWire
        for (int i = 0; i < 6; i++) {
            norm[i] = feats[i] / 112.0f;
        }

        // Copy slope features as-is
        for (int i = 6; i < 12; i++) {
            norm[i] = feats[i];
        }

        return norm;
    }
}

/**
 * ===========================
 * Main Inference Program
 * ===========================
 */
public class Main {

    public static void main(String[] args) {

        // -----------------------------
        // 1. Translator: TrackInput -> Float (track probability)
        // -----------------------------
        Translator<TrackInput, Float> translator = new Translator<TrackInput, Float>() {

            @Override
            public NDList processInput(TranslatorContext ctx, TrackInput input) {
                NDManager manager = ctx.getNDManager();
                // Shape: (1, 12) for single sample
                NDArray x = manager.create(input.features).reshape(1, input.features.length);
                return new NDList(x);
            }

            @Override
            public Float processOutput(TranslatorContext ctx, NDList list) {
                NDArray result = list.get(0); // Shape: (1,)
                return result.toFloatArray()[0]; // Extract single predicted value
            }

            @Override
            public Batchifier getBatchifier() {
                return null; // Single-sample inference (no batching)
            }
        };

        // -----------------------------
        // 2. Define model loading criteria
        // -----------------------------
        Criteria<TrackInput, Float> criteria = Criteria.builder()
                .setTypes(TrackInput.class, Float.class)
                .optModelPath(Paths.get("nets/mlp_default.pt"))  // TorchScript model path
                .optEngine("PyTorch")
                .optTranslator(translator)
                .optProgress(new ProgressBar())
                .build();

        // -----------------------------
        // 3. Load model and run inference
        // -----------------------------
        try (ZooModel<TrackInput, Float> model = criteria.loadModel();
             Predictor<TrackInput, Float> predictor = model.newPredictor()) {

            // Example input (12 float features)
            float[] exampleFeatures = new float[]{
                    44.6000f, 43.3333f, 41.0000f, 38.8571f, 35.1667f, 33.4286f,
                    -0.3232f, -0.1155f, 0.0009f, -0.1015f, -0.1506f, -0.3012f
            };

            TrackInput input = new TrackInput(exampleFeatures);

            Float probability = predictor.predict(input);
            System.out.printf("Predicted track probability: %.4f%n", probability);

        } catch (IOException | ModelNotFoundException | MalformedModelException | TranslateException e) {
            throw new RuntimeException("Model inference failed", e);
        }
    }
}