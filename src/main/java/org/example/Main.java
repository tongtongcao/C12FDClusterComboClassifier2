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

// ===========================
// Custom Input Class
// ===========================
class TrackInput {
    float[] features;   // 长度 = 12 (avgWire，slope 等输入特征)

    public TrackInput(float[] features) {
        this.features = features;
    }
}

// ===========================
// Main Inference Program
// ===========================
public class Main {

    public static void main(String[] args) {

        // -----------------------------
        // 1. Translator: 输入 TrackInput → 输出 Float (轨迹概率)
        // -----------------------------
        Translator<TrackInput, Float> translator = new Translator<TrackInput, Float>() {

            @Override
            public NDList processInput(TranslatorContext ctx, TrackInput input) {
                NDManager manager = ctx.getNDManager();
                // shape: (1, 12)
                NDArray x = manager.create(input.features).reshape(1, input.features.length);
                return new NDList(x);
            }

            @Override
            public Float processOutput(TranslatorContext ctx, NDList list) {
                NDArray result = list.get(0);  // shape: (1,)
                return result.toFloatArray()[0];  // 取出单个预测值
            }

            @Override
            public Batchifier getBatchifier() {
                return null;  // 单样本推理
            }
        };

        // -----------------------------
        // 2. 定义模型加载 Criteria
        // -----------------------------
        Criteria<TrackInput, Float> criteria = Criteria.builder()
                .setTypes(TrackInput.class, Float.class)
                .optModelPath(Paths.get("nets/mlp_default.pt"))  // TorchScript 模型路径
                .optEngine("PyTorch")
                .optTranslator(translator)
                .optProgress(new ProgressBar())
                .build();

        // -----------------------------
        // 3. 加载模型并执行推理
        // -----------------------------
        try (ZooModel<TrackInput, Float> model = criteria.loadModel();
             Predictor<TrackInput, Float> predictor = model.newPredictor()) {

            // 示例输入 (12 个 float 特征)
            float[] exampleFeatures = new float[]{44.6000f,43.3333f,41.0000f,38.8571f,35.1667f,33.4286f,-0.3232f,-0.1155f,0.0009f,-0.1015f,-0.1506f,-0.3012f};

            TrackInput input = new TrackInput(exampleFeatures);

            Float probability = predictor.predict(input);
            System.out.printf("Predicted track probability: %.4f%n", probability);

        } catch (IOException | ModelNotFoundException | MalformedModelException | TranslateException e) {
            throw new RuntimeException("Model inference failed", e);
        }
    }
}
