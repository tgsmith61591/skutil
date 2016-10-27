import java.io.*;
import hex.genmodel.easy.RowData;
import hex.genmodel.easy.EasyPredictModelWrapper;
import hex.genmodel.easy.prediction.*;

public class main {
    private static String modelClassName = "{{ model_class_name }}";
    public static void main(String[] args) throws Exception {

        hex.genmodel.GenModel rawModel;
        rawModel = (hex.genmodel.GenModel) Class.forName(modelClassName).newInstance();
        EasyPredictModelWrapper model = new EasyPredictModelWrapper(rawModel);
        {{ fields }}
        BinomialModelPrediction p = model.predictBinomial(row);
        System.out.println("Label (aka prediction) is flight departure delayed: " + p.label);
        System.out.print("Class probabilities: ");

        for (int i = 0; i < p.classProbabilities.length; i++) {
            if (i > 0) {
                System.out.print(",");
            }
        System.out.print(p.classProbabilities[i]);
        }
        System.out.println("");
    }
}