# Convolutional Neural Network 

[Go to German Version](#german-version)

**Convolutional neural networks (CNNs)** have revolutionized the field of computer vision, enabling applications ranging from image segmentation and object recognition to more complex tasks like detecting diseases such as cancer in medical images (LeCun et al., 2015; Kumar et al., 2024). See the summary report [here](https://github.com/alexhubbe/convolutional_neural_networks/blob/main/reports/summary_report.md).

This repository showcases key techniques for improving CNN performance using the **[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)** dataset—a collection of 60,000 32×32 color images across 10 classes (Figure 1). 

<p align="center">
<img src="https://github.com/alexhubbe/image_classification/blob/main/reports/images/example_images.jpg" width="100%" alt="Example images used for classification">
</p>  
Figure 1: Example images from the CIFAR-10 dataset, showcasing the 10 classes used for classification. 

<br><br>

This project is divided into two parts:
- In [Image Classification with Transfer Learning](https://github.com/alexhubbe/convolutional_neural_networks/blob/main/notebooks/01_ah_model.ipynb), I show key strategies for improving performance when working with CNNs under a transfer learning approach, which is a widely recommended practice (Goodfellow et al., 2016; Géron, 2022). 
- In [Image Classification with AutoGluon's AutoMM](https://github.com/alexhubbe/convolutional_neural_networks/blob/main/notebooks/02_ah_autogluon.ipynb), I used the powerful AutoGluon's [AutoMM](https://auto.gluon.ai/stable/tutorials/multimodal/index.html) to achieve high performance results. 

## Key Findings  
- Transfer learning with fine-tuning of the base model, adjusting learning rates, modifying batch size, tuning hyperparameters, and resizing input images significantly improves model performance (Figures 2 and 3). Combining these techniques with L2 regularization and dropout can also help mitigate overfitting (Figure 4).
- **AutoGluon's AutoMM** achieved **99.03% accuracy** on the test dataset with just 30 minutes of training. Compared to publicly available models, AutoMM ranks within the top 8% of models on CIFAR-10 and is only 0.47% below the best model. 

<p align="center">
<img src="https://github.com/alexhubbe/image_classification/blob/main/reports/images/mobilenet_summary.png" width="100%" alt="Mobilenet model summary">
</p>

Figure 2: MobileNet models summary diagram. 

<p align="center">
<img src="https://github.com/alexhubbe/image_classification/blob/main/reports/images/nasnetmobile_summary%20.png" width="70%" alt="NasNetMobile model summary">
</p>  

Figure 3: NasNetMobile models summary diagram.

<div style="display: flex; justify-content: center; gap: 20px;">
    <img src="https://github.com/alexhubbe/image_classification/blob/main/reports/images/overfit.jpg" width="48%" alt="Train Validation loss curve">
    <img src="https://github.com/alexhubbe/image_classification/blob/main/reports/images/betterfit.jpg" width="48%" alt="Train Validation loss curve">
</div>  

Figure 4: Training and validation loss curves. On the left side, a case of overfitting, and on the right side, a better fit resulting from hyperparameter changes and L2 regularization and dropout.
<br>

Below is a succinct description of the steps taken in this project.

## [Image Classification with Transfer Learning](https://github.com/alexhubbe/convolutional_neural_networks/blob/main/notebooks/01_ah_model.ipynb)  

This section shows key techniques for improving performance in CNNs using transfer learning. To achieve this, I chose the `MobileNet` and `NASNetMobile` architectures, as they offer the fastest inference times among the `Keras` models available for transfer learning.  

However, `MobileNet`, which has the fastest inference time, does not support **32 × 32** images for transfer learning. Therefore, I used `NASNetMobile` instead.  

If the primary objective were to maximize model accuracy, larger architectures like `NASNetLarge` would be preferable, as performance on ImageNet is positively correlated with performance on other datasets when using transfer learning (Kornblith et al., 2019). Another option would be `AutoGluon's AutoMM` (see my notebook [here](https://github.com/alexhubbe/image_classification/blob/main/notebooks/02_ah_autogluon.ipynb)). Additional alternatives can be found [here](https://paperswithcode.com/sota/image-classification-on-cifar-10).  

### **Model Architectures and Training Approaches**  

#### **MobileNet Models**  
- **No transfer learning** is used.  
- Baseline model (**Model I**) uses fixed hyperparameters: **30 epochs, batch size of 128, and learning rate of 1E-3**.  
- Subsequent models (**II.1 to II.5**) incorporate **L2 regularization** and **Dropout** to mitigate overfitting. Variations include:  
  - **Model II.1**: Baseline with L2 and Dropout.  
  - **Model II.2**: Fixed learning rate of 1E-2.  
  - **Model II.3**: Decaying learning rate starting at 1E-2.  
  - **Model II.4**: Batch size reduced to 32.  
  - **Model II.5**: Automated hyperparameter tuning.  

#### **NASNetMobile Models**  
- **Transfer learning** is used, with the base model's weights **frozen** by default.  
- Baseline model (**Model I**) uses fixed hyperparameters: **30 epochs, batch size of 128, and learning rate of 1E-3**.  
- **Model II**: Resizes images to **224×224** to match ImageNet’s input size.  
- **Model III**: Unfreezes the base model for fine-tuning with a learning rate of **1E-4**.

## [Image Classification with AutoGluon's AutoMM](https://github.com/alexhubbe/convolutional_neural_networks/blob/main/notebooks/02_ah_autogluon.ipynb)  

Using AutoGluon's [AutoMM](https://auto.gluon.ai/stable/tutorials/multimodal/index.html), I achieved **99.03% accuracy** on the CIFAR-10 test dataset with just **30 minutes of training**. This result ranks within the **top 8%** of models and is only **0.47% lower** than the best publicly available model (see [here](https://paperswithcode.com/sota/image-classification-on-cifar-10)).  

## Tools and Technologies  
- **Libraries**: autogluon, graphviz, keras, matplotlib, numpy, pandas, scikit-learn, tensorflow

## Project organization

```
├── .gitignore                                <- Files and directories to be ignored by Git
|
├── LICENSE                                   <- License type 
|
├── README.md                                 <- Main README explaining the project
|
├── environments                              <- Requirements files to reproduce the analysis environments
|   ├── autogluon_environment.yml             <- Environment for running '02_ah_autogluon.ipynb' 
|   ├── deep_learning_environment.yml         <- Environment for running '01_ah_model.ipynb', 'ah_appendix.ipynb'
|
├── models                                    <- Trained and serialized models, model predictions, or model summaries
|   ├── best_model.keras                      <- Stores the best-performing model during training based on validation sparse categorical accuracy (used in MobileNet Model II.5)
|   └── kt_mobilenet_tuning                   <- Directory to store Keras hyperparameter optimization results
|       ├── mobilenet_bayesian                <- Directory for trials from the optimization procedure
|
├── notebooks                                 <- Jupyter notebooks
|   ├── 01_ah_model.ipynb                     <- Image Classification with Transfer Learning
|   ├── 02_ah_autogluon.ipynb                 <- Image Classification with Autogluon`s AutoMM
|   ├── ah_appendix.ipynb                     <- Model Summary Diagram
|
├── references                                <- Data dictionaries, manuals, and other explanatory materials
|   ├── data_contextualization.md             <- Description of the dataset
|
├── reports                                   <- Generated analyses in HTML, PDF, LaTeX, etc., and results
|   ├── summary_report.md                     <- Summary report
│   └── images                                <- Images used in the project
|      ├── betterfit.jpg                      <- Figure with a better fit in the train and validation loss curve
|      ├── example_images.jpg                 <- Figure to Illustrate the Pictures within Each Class
|      ├── mobilenet_summary.png              <- MobileNet Models Summary Diagram
|      ├── nasnetmobile_summary .png          <- NasNetMobile Models Summary Diagram
|      ├── overfit.jpg                        <- Figure depicting overfit in the train and validation loss curve
```
## Contributing
All contributions are welcome!

### Issues
Submit issues for:
- Recommendations or improvements
- Additional analyses or models
- Feature enhancements
- Bug reports

### Pull Requests
- Open an issue before starting work.
- Fork the repository and clone it.
- Create a branch and commit your changes.
- Push your changes and open a pull request for review.

## References
- Géron, A. (2022). Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow. "O'Reilly Media, Inc.".  
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.  
- Kornblith, S., Shlens, J., & Le, Q. V. (2019). "Do better ImageNet models transfer better?" Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 2661-2671.  
- Kumar, Y., Shrivastav, S., Garg, K., Modi, N., Wiltos, K., Woźniak, M., & Ijaz, M. F. (2024). Automating cancer diagnosis using advanced deep learning techniques for multi-cancer image classification. *Scientific Reports, 14*(1), 25006.  
- LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature, 521*(7553), 436-444.

# German Version

# Convolutional Neural Network

**Convolutional Neural Networks (CNNs)** haben das Feld der Computer Vision revolutioniert und ermöglichen Anwendungen von Bildsegmentierung und Objekterkennung bis hin zu komplexeren Aufgaben wie der Erkennung von Krankheiten wie Krebs in medizinischen Bildern (LeCun et al., 2015; Kumar et al., 2024). Siehe den Zusammenfassungsbericht [hier](https://github.com/alexhubbe/convolutional_neural_networks/blob/main/reports/summary_report.md).

Dieses Repository zeigt Schlüsseltechniken zur Verbesserung der CNN-Leistung unter Verwendung des **[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)**-Datensatzes – einer Sammlung von 60.000 32×32 Farbbildern in 10 Klassen (Abbildung 1).

<p align="center">
<img src="https://github.com/alexhubbe/image_classification/blob/main/reports/images/example_images.jpg" width="100%" alt="Beispielbilder für die Klassifizierung">
</p>  
Abbildung 1: Beispielbilder aus dem CIFAR-10-Datensatz, die die 10 Klassen für die Klassifizierung zeigen.

<br><br>

Dieses Projekt ist in zwei Teile unterteilt:
- In [Bildklassifizierung mit Transfer Learning](https://github.com/alexhubbe/convolutional_neural_networks/blob/main/notebooks/01_ah_model.ipynb) zeige ich wichtige Strategien zur Verbesserung der Leistung bei der Arbeit mit CNNs unter einem Transfer-Learning-Ansatz, der eine weit verbreitete Praxis ist (Goodfellow et al., 2016; Géron, 2022).
- In [Bildklassifizierung mit AutoGluon's AutoMM](https://github.com/alexhubbe/convolutional_neural_networks/blob/main/notebooks/02_ah_autogluon.ipynb) habe ich das leistungsstarke AutoGluon's [AutoMM](https://auto.gluon.ai/stable/tutorials/multimodal/index.html) verwendet, um hochleistungsfähige Ergebnisse zu erzielen.

## Wichtige Erkenntnisse
- Transfer Learning mit Feinabstimmung des Basismodells, Anpassung der Lernraten, Änderung der Batch-Größe, Optimierung der Hyperparameter und Änderung der Bildgröße verbessert die Modellleistung erheblich (Abbildungen 2 und 3). Die Kombination dieser Techniken mit L2-Regularisierung und Dropout kann auch dazu beitragen, Overfitting zu reduzieren (Abbildung 4).
- **AutoGluon's AutoMM** erreichte eine **Genauigkeit von 99,03 %** auf dem Testdatensatz mit nur 30 Minuten Training. Im Vergleich zu öffentlich verfügbaren Modellen liegt AutoMM in den **Top 8 %** der Modelle auf CIFAR-10 und ist nur **0,47 %** unter dem besten Modell.

<p align="center">
<img src="https://github.com/alexhubbe/image_classification/blob/main/reports/images/mobilenet_summary.png" width="100%" alt="Mobilenet-Modellzusammenfassung">
</p>

Abbildung 2: Zusammenfassungsdiagramm der MobileNet-Modelle.

<p align="center">
<img src="https://github.com/alexhubbe/image_classification/blob/main/reports/images/nasnetmobile_summary%20.png" width="70%" alt="NasNetMobile-Modellzusammenfassung">
</p>  

Abbildung 3: Zusammenfassungsdiagramm der NasNetMobile-Modelle.

<div style="display: flex; justify-content: center; gap: 20px;">
    <img src="https://github.com/alexhubbe/image_classification/blob/main/reports/images/overfit.jpg" width="48%" alt="Trainings- und Validierungsverlustkurve">
    <img src="https://github.com/alexhubbe/image_classification/blob/main/reports/images/betterfit.jpg" width="48%" alt="Trainings- und Validierungsverlustkurve">
</div>  

Abbildung 4: Trainings- und Validierungsverlustkurven. Links ein Fall von Overfitting, rechts ein besserer Fit durch Hyperparameteränderungen sowie L2-Regularisierung und Dropout.
<br>

Unten finden Sie eine prägnante Beschreibung der Schritte in diesem Projekt.

## [Bildklassifizierung mit Transfer Learning](https://github.com/alexhubbe/convolutional_neural_networks/blob/main/notebooks/01_ah_model.ipynb)

Dieser Abschnitt zeigt wichtige Techniken zur Verbesserung der Leistung von CNNs mit Transfer Learning. Dafür habe ich die Architekturen `MobileNet` und `NASNetMobile` gewählt, da sie die schnellsten Inferenzzeiten unter den für Transfer Learning verfügbaren `Keras`-Modellen bieten.

Allerdings unterstützt `MobileNet`, das die schnellste Inferenzzeit hat, keine **32 × 32** Bilder für Transfer Learning. Daher habe ich stattdessen `NASNetMobile` verwendet.

Wenn das primäre Ziel die Maximierung der Modellgenauigkeit wäre, wären größere Architekturen wie `NASNetLarge` vorzuziehen, da die Leistung auf ImageNet positiv mit der Leistung auf anderen Datensätzen korreliert, wenn Transfer Learning verwendet wird (Kornblith et al., 2019). Eine weitere Option wäre `AutoGluon's AutoMM` (siehe mein Notebook [hier](https://github.com/alexhubbe/image_classification/blob/main/notebooks/02_ah_autogluon.ipynb)). Weitere Alternativen finden Sie [hier](https://paperswithcode.com/sota/image-classification-on-cifar-10).

### **Modellarchitekturen und Trainingsansätze**

#### **MobileNet-Modelle**
- **Kein Transfer Learning** wird verwendet.
- Das Basismodell (**Modell I**) verwendet feste Hyperparameter: **30 Epochen, Batch-Größe von 128 und Lernrate von 1E-3**.
- Nachfolgende Modelle (**II.1 bis II.5**) integrieren **L2-Regularisierung** und **Dropout**, um Overfitting zu reduzieren. Variationen umfassen:
  - **Modell II.1**: Baseline mit L2 und Dropout.
  - **Modell II.2**: Feste Lernrate von 1E-2.
  - **Modell II.3**: Abnehmende Lernrate, beginnend bei 1E-2.
  - **Modell II.4**: Batch-Größe auf 32 reduziert.
  - **Modell II.5**: Automatisierte Hyperparameteroptimierung.

#### **NASNetMobile-Modelle**
- **Transfer Learning** wird verwendet, wobei die Gewichte des Basismodells standardmäßig **eingefroren** sind.
- Das Basismodell (**Modell I**) verwendet feste Hyperparameter: **30 Epochen, Batch-Größe von 128 und Lernrate von 1E-3**.
- **Modell II**: Ändert die Bildgröße auf **224×224**, um der Eingabegröße von ImageNet zu entsprechen.
- **Modell III**: Entsperrt das Basismodell für die Feinabstimmung mit einer Lernrate von **1E-4**.

## [Bildklassifizierung mit AutoGluon's AutoMM](https://github.com/alexhubbe/convolutional_neural_networks/blob/main/notebooks/02_ah_autogluon.ipynb)

Mit AutoGluon's [AutoMM](https://auto.gluon.ai/stable/tutorials/multimodal/index.html) habe ich eine **Genauigkeit von 99,03 %** auf dem CIFAR-10-Testdatensatz mit nur **30 Minuten Training** erreicht. Dieses Ergebnis liegt in den **Top 8 %** der Modelle und ist nur **0,47 %** niedriger als das beste öffentlich verfügbare Modell (siehe [hier](https://paperswithcode.com/sota/image-classification-on-cifar-10)).

## Tools und Technologien
- **Bibliotheken**: autogluon, graphviz, keras, matplotlib, numpy, pandas, scikit-learn, tensorflow

## Projektorganisation

├── .gitignore                                <- Dateien und Verzeichnisse, die von Git ignoriert werden sollen
|
├── LICENSE                                   <- Lizenztyp 
|
├── README.md                                 <- Haupt-README mit einer Erklärung des Projekts
|
├── environments                              <- Anforderungsdateien zur Reproduktion der Analyseumgebungen
|   ├── autogluon_environment.yml             <- Umgebung für die Ausführung von '02_ah_autogluon.ipynb' 
|   ├── deep_learning_environment.yml         <- Umgebung für die Ausführung von '01_ah_model.ipynb', 'ah_appendix.ipynb'
|
├── models                                    <- Trainierte und serialisierte Modelle, Modellvorhersagen oder Modellzusammenfassungen
|   ├── best_model.keras                      <- Speichert das während des Trainings am besten abschneidende Modell basierend auf der Validierungsgenauigkeit (verwendet im MobileNet-Modell II.5)
|   └── kt_mobilenet_tuning                   <- Verzeichnis zur Speicherung der Keras-Hyperparameteroptimierungsergebnisse
|       ├── mobilenet_bayesian                <- Verzeichnis für Versuche aus dem Optimierungsverfahren
|
├── notebooks                                 <- Jupyter-Notebooks
|   ├── 01_ah_model.ipynb                     <- Bildklassifikation mit Transfer Learning
|   ├── 02_ah_autogluon.ipynb                 <- Bildklassifikation mit Autogluons AutoMM
|   ├── ah_appendix.ipynb                     <- Modellzusammenfassungsdiagramm
|
├── references                                <- Datenwörterbücher, Handbücher und andere erklärende Materialien
|   ├── data_contextualization.md             <- Beschreibung des Datensatzes
|
├── reports                                   <- Generierte Analysen in HTML, PDF, LaTeX usw. und Ergebnisse
|   ├── summary_report.md                     <- Zusammenfassender Bericht
│   └── images                                <- Im Projekt verwendete Bilder
|      ├── betterfit.jpg                      <- Abbildung mit einer besseren Anpassung in der Trainings- und Validierungsverlustkurve
|      ├── example_images.jpg                 <- Abbildung zur Veranschaulichung der Bilder innerhalb jeder Klasse
|      ├── mobilenet_summary.png              <- MobileNet-Modellzusammenfassungsdiagramm
|      ├── nasnetmobile_summary.png           <- NasNetMobile-Modellzusammenfassungsdiagramm
|      ├── overfit.jpg                        <- Abbildung, die Overfitting in der Trainings- und Validierungsverlustkurve zeigt


## Mitwirken
Alle Beiträge sind willkommen!

### Probleme
Reichen Sie Probleme ein für:
- Empfehlungen oder Verbesserungen
- Zusätzliche Analysen oder Modelle
- Funktionserweiterungen
- Fehlerberichte

### Pull Requests
- Öffnen Sie ein Issue, bevor Sie mit der Arbeit beginnen.
- Forken Sie das Repository und klonen Sie es.
- Erstellen Sie einen Branch und committen Sie Ihre Änderungen.
- Pushen Sie Ihre Änderungen und öffnen Sie einen Pull Request zur Überprüfung.

## Referenzen
- Géron, A. (2022). Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow. "O'Reilly Media, Inc.".
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
- Kornblith, S., Shlens, J., & Le, Q. V. (2019). "Do better ImageNet models transfer better?" Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 2661-2671.
- Kumar, Y., Shrivastav, S., Garg, K., Modi, N., Wiltos, K., Woźniak, M., & Ijaz, M. F. (2024). Automating cancer diagnosis using advanced deep learning techniques for multi-cancer image classification. *Scientific Reports, 14*(1), 25006.
- LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature, 521*(7553), 436-444.
