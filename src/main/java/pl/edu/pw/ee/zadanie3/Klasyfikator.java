package pl.edu.pw.ee.zadanie3;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import static libsvm.svm.svm_predict;
import static libsvm.svm.svm_train;

/*

https://www.csie.ntu.edu.tw/~cjlin/libsvm/
http://svmlight.joachims.org/

 X = csvread('spambase.data');
 labels = X(:,58);
features = X(:,1:end-1);
features_sparse = sparse(features);
libsvmwrite('spambase.train', labels, features_sparse);

Mieszam plik z danymi tak żeby dane w każdej próbce mniej więcej odzwierciedlały
całą próbkę:
> shuf spambase.train > spambase.train.rand 

Dzielę pilk na 10 równych porcji:
> split -l 460 spambase.train.rand spambase.train.rand.

W wyniku poprzedniego polecenia otrzymuję 10 plików po 460 linii, oraz 1 plik z 
jedną linią:
-rw-rw-r--. 1 bladekp bladekp 460945 11-20 14:50 spambase.train
-rw-rw-r--. 1 bladekp bladekp 460945 11-20 15:07 spambase.train.rand
-rw-rw-r--. 1 bladekp bladekp  45934 11-20 15:08 spambase.train.rand.aa
-rw-rw-r--. 1 bladekp bladekp  44481 11-20 15:08 spambase.train.rand.ab
-rw-rw-r--. 1 bladekp bladekp  45551 11-20 15:08 spambase.train.rand.ac
-rw-rw-r--. 1 bladekp bladekp  45683 11-20 15:08 spambase.train.rand.ad
-rw-rw-r--. 1 bladekp bladekp  47293 11-20 15:08 spambase.train.rand.ae
-rw-rw-r--. 1 bladekp bladekp  46110 11-20 15:08 spambase.train.rand.af
-rw-rw-r--. 1 bladekp bladekp  46573 11-20 15:08 spambase.train.rand.ag
-rw-rw-r--. 1 bladekp bladekp  47497 11-20 15:08 spambase.train.rand.ah
-rw-rw-r--. 1 bladekp bladekp  45545 11-20 15:08 spambase.train.rand.ai
-rw-rw-r--. 1 bladekp bladekp  46191 11-20 15:08 spambase.train.rand.aj
-rw-rw-r--. 1 bladekp bladekp     87 11-20 15:08 spambase.train.rand.ak

Łączę plik z jedną linią do ostatniego pliku, w ten sposób ostatni plik będzie 
miał 461 linii, a pozostałe 460:
> cat spambase.train.rand.ak >> spambase.train.rand.aj
> rm spambase.train.rand.ak


*/
public class Klasyfikator {
    
    public static void main(String args[]) throws IOException{
        // nazwy plików z danymi
        List<String> paths = 
                Stream
                        .iterate(97, n -> n + 1)
                        .limit(10)
                        .map(n -> "src/main/resources/spambase.train.rand.a".concat("" + (char) n.byteValue()))
                        .collect(Collectors.toList());
        
        // tworzymy pliki do cross classification
        for(int i =0; i<10; i++){
            List<String> pathsT = new ArrayList<>(paths);
            pathsT.remove(i);
            String[] arr = new String[9];
            arr = pathsT.toArray(arr);
            List<String> ls = Stream.of(arr)
                .map(Paths::get)
                .flatMap(ThrowingFunction.wrap(Files::lines))
                .collect(Collectors.toList());
            Files.write(Paths.get("src/main/resources/spambase.train.rand.without_file_"+(i+1)), ls);
        }
        
        // dla każdego pliku z danymi treningowymi wołamy kolejno plik z danymi testującymi
        for (int i =0; i<10; i++){
            new svm_train().run(("-s 0 -t 2 -g 0.01 -q src/main/resources/spambase.train.rand.without_file_"+(i+1)+" src/main/resources/model_file").split(" "));
            svm_predict.start(("src/main/resources/spambase.train.rand.a"+((char)(97+i))+" src/main/resources/model_file src/main/resources/output_file").split(" "));
        }
    }

}
