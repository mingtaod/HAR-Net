# HAR-Net
Human Activity Recognition based on Convolutional Neural Network

Below is the abstract of he paper proposing the method: 
Wearable computing and context awareness are the focuses of study in the field of artificial intelligence recently. One of the most appealing as well as challenging applications is the Human Activity Recognition (HAR) utilizing smart phones. Conventional HAR based on Support Vector Machine relies on manually extracted features. This approach is time and energy consuming in prediction due to the partial view toward which features to be extracted by human. With the rise of deep learning, artificial intelligence has been making progress toward being a mature technology. This paper proposes a new approach based on deep learning called HAR-Net to address the HAR issue. The study used the data collected by gyroscopes and acceleration sensors in android smart phones. The HAR-Net fusing the hand-crafted features and high-level features extracted from convolutional neural network to make prediction. The performance of the proposed method was proved to be higher than the original MC-SVM approach. The experimental results on the UCI dataset demonstrate that fusing the two kinds of features can make up for the shortage of traditional feature engineering and deep learning techniques.


<img src="./model_illustration/Model Structure.PNG" />
Figure1. Overall structure of the model proposed

<img src="./model_illustration/Separable Model Illustration.PNG" />
Figure2. Illustration of separable convolution

<img src="./model_illustration/Inception Block.PNG" />
Figure3. The inception block adopted in the paper and the proposed model
