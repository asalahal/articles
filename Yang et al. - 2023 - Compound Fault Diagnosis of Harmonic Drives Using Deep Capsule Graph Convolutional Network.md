This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TIE.2022.3176280, IEEE

Transactions on Industrial Electronics

IEEE TRANSACTIONS ON INDUSTRIAL ELECTRONICS

## Compound Fault Diagnosis of Harmonic Drives Using Deep Capsule Graph Convolutional Network


Guo Yang, Hui Tao, Ruxu Du, and Yong Zhong, _Member, IEEE_






_**Abstract**_ **—Harmonic drive is a key component of the**
**industrial robot. Because of its large reduction ratio and**
**excessive dynamic loading, various kinds of faults may**
**occur. In particular, since the robot is an integrated**
**system, it is not unusual to have multiple harmonic drives**
**malfunction simultaneously, which is difficult to diagnose.**
**In practice, these kinds of compound faults are often**
**mislabeled as single faults causing missing repair. In this**
**paper, we propose a deep capsule graph convolutional**
**network (DCGCN) approach to diagnose compound faults**
**of harmonic drives. First, the multi-sensor data is used to**
**obtain the frequency spectrum of the fault signal and**
**construct the label relationship map of the adjacency**
**matrix. Second, the deep capsule network is used to learn**
**the representation of the fault vector, and the graph**
**convolutional network is used to learn the relationship**
**between different single-label faults. Third, the two**
**networks are combined to obtain diagnosing results.**
**Finally, the dynamic routing algorithm and the margin loss**
**function** **are** **used** **to** **optimize** **the** **DCGCN.** **The**
**experimental results show that the proposed DCGCN can**
**effectively diagnose compound faults under varying**
**working** **conditions,** **outperforming** **other** **existing**
**state-of-the-art methods.**


_**Index Terms**_ **—Harmonic drive, compound faults, Deep**
**Capsule Graph Convolutional Network (DCGCN), varying**
**working conditions.**


I. I NTRODUCTION
# H ARMONIC drives are the key transmission component of the industrial robot, which have the advantages of small

size, lightweight, large reduction ratio, high efficiency, and
high precision [1]-[3]. The performance of harmonic drive is


 This work was supported in part by the National Natural Science
Foundation of China under Grant 62103152, the Opening Project of
National and local joint Engineering Research Center for industrial
friction and lubrication technology, the Natural Science Foundation of
Guangdong Province (2020A1515010621, 2022A1515011479),
Guangzhou Applied Basic Research Foundation (202102020360), KEY
Laboratory of Robotics and Intelligent Equipment of Guangdong
Regular Institutions of Higher Education (Grant No.2017KSYS009), and
Innovation Center of Robotics and Intelligent Equipment).
(Corresponding author: Yong Zhong.)

Guo Yang, Hui Tao, Ruxu Du, and Yong Zhong are with the
Shien-Ming Wu School of Intelligent Engineering, South China
University of Technology, Guangzhou 511442, China (e-mail:
201910108802@mail.scut.edu.cn; 202010109516@mail.scut.edu.cn;
duruxu@scut.edu.cn; zhongyong@scut.edu.cn).



not only easily affected by the manufacturing error and
assembly error of the product but also related to the working
conditions [4]. The fault diagnosis of the harmonic drives
mainly relies on simple vibration measuring instruments and
the mature experience of skilled workers, which leads to
unreliable diagnosis results and low efficiency.

The single-fault diagnosis of rotating machinery equipment
(except for harmonic drive) has received lots of research
interests [5]-[8]. However, the feature of compound faults is
complex and the identification is difficult, which has
increasingly aroused scholars’ attention [9]. The challenges of
the compound fault diagnosis of the harmonic drives are
summarized as follows: 1) There are various combinations of
compound faults in harmonic drives. However, we often can
only collect data of various single-label faults and lack various
types of compound fault data. 2) The physical structure of the
harmonic drive is quite complex and it is different on each axis
of the industrial robot, and the fault characteristics are also
different. 3) Unlike traditional gearboxes fixed on static
objects, the harmonic drives installed on the industrial robots
have dynamic spatial motion, and their compound fault signals
are formed by a highly nonlinear combination of individual
fault components. Therefore, there is still a lack of an effective
fault diagnosis method for the compound fault of harmonic
drives.

For the compound fault diagnosis (CFD) of rotating
machinery equipment, the existing technologies mainly
include: 1) Analytical model-based CFD. It needs to master the
operating mechanism of the complex system. Zhang _et al_ [10]
used intrinsic mode functions (IMFs) to decompose the
mechanical faults of impeller blowers, but the number of IMF
features for multi-classification is often difficult to determine.
2) Qualitative knowledge base CFD. It needs to accumulate
rich technology and mature experience of experts. Ma _et al._

 [11] used Personalized Binary Correlation (PBR) to convert

multi-label fault diagnosis of roller bearings into multiple
independent binary classification problems, which has the
disadvantage of a large number of models. 3)
Data-driven-based CFD. The intelligent fault analysis method
based on big data is the mainstream of current research [12].
For example, Jia _et al._ [13] used Fast Fourier Transform (FFT)
and Deep Neural Network (DNN) to isolate the failure modes
of planetary gearboxes. Huang _et al._ [14] adopted a multi-label
classifier to realize the compound fault diagnosis of the
gearbox. Huang _et al._ [14] used Convolutional Neural Network
(CNN) and softmax function to realize multi-label fault






0278-0046 (c) 2021 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.

Authorized licensed use limited to: Universita' Politecnica delle Marche. Downloaded on November 11,2022 at 18:34:06 UTC from IEEE Xplore. Restrictions apply.


This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TIE.2022.3176280, IEEE

Transactions on Industrial Electronics

IEEE TRANSACTIONS ON INDUSTRIAL ELECTRONICS



diagnosis of the gearbox. However, it is difficult to measure the
true performance of multi-label classification in the network
model by only using the precision index alone. Yadavendra

[16] utilized the InceptionV3 pre-trained model to achieve the
multi-label classification of the human cell atlas. Zhu _et al._ [17]
adopted the CNN based on a capsule network to classify the
compound fault of bearing.

Although the above-mentioned deep learning methods
perform well in classification problems, they are limited to
extracting euclidean spatial feature data, and the network model
lacks the learning of label relationship features. Recently, graph
neural networks (GNNs) have become a research hotspot. The
map constructed in the graph network can effectively reveal the
relationship between the tags in the image, and it can achieve
excellent multi-classification performance [18]-[19]. Chen _et_
_al._ [18] constructed the graph convolutional network (GCN) to
realize automatic recognition of multiple objects in the picture.
Hu _et al._ [19] proposed a graph attention network (GAT) that
combines GCN and attention mechanism, which can further
improve the ability to recognize various objects in the picture.
However, the compound samples need to participate in the
process of network model training, but for the CFD of harmonic
drives, the compound fault data is scarce. Therefore, there is an
urgent need for a method that only utilizes various single-fault
samples for network training and can successfully diagnose
compound faults in the testing phase.

To solve the aforementioned problem, an intelligent method
named deep capsule graph convolutional network (DCGCN) is
proposed to achieve the compound fault diagnosis of harmonic
drives. We construct the intelligent diagnosis model with two
sub-networks. One adopts capsule network to obtain the feature
vector of multiple single-faults, and the other uses the graph


Fig. 1. The architecture of the proposed DCGCN.


_A._ _Architecture of DCGCN_


_1)_ _Multi-signal Fusion Module:_ This module is used for
multi-channel signal processing and data fusion to obtain more



convolutional network to learn the topology between various
single-faults. Then, the generated classifier is used to perform
dot product on the feature matrix obtained by the two
sub-networks, and output the predicted probability of various
single-fault components in the compound fault. Next, dynamic
routing algorithms and margin loss functions are used to
optimize DCGCN. The experimental results show that the
proposed method is effective to identify and diagnose each fault
component in the compound faults. The main contributions of
the article are as follows:

1) To solve the problem that the compound fault samples of

different combinations are difficult to collect, the
proposed DCGCN network only needs various types of
single-label fault samples to realize the compound fault
diagnosis of the harmonic drive.
2) The constructed intelligent diagnosis model-DCGCN

contains two sub-networks, one uses the capsule
network to capture the vector features of single-label
fault samples, and the other uses the graph convolutional
neural network to learn the relationship between various
types of single-label faults.
3) It is the first time to realize the compound fault diagnosis

of harmonic drives and obtain better diagnosis results
than other existing methods, which has very important
engineering significance.


II. M ATERIALS AND M ETHODS


The proposed DCGCN is composed of four modules, as
shown in Fig.1, including the multi-signal fusion module, the
representation learning module, the graph learning module, and
the generated classifiers module.


health status information of mechanical equipment. Signal
preprocessing includes the following steps: First, the collected
acceleration vibration signals from the three sensors were used



0278-0046 (c) 2021 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.

Authorized licensed use limited to: Universita' Politecnica delle Marche. Downloaded on November 11,2022 at 18:34:06 UTC from IEEE Xplore. Restrictions apply.


This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TIE.2022.3176280, IEEE

Transactions on Industrial Electronics

IEEE TRANSACTIONS ON INDUSTRIAL ELECTRONICS



to resampled by moving the fixed window to obtain five times
the amount of data of the original signal. Second, the original
signals were performed Fast Fourier transform (FFT) to obtain
the frequency spectrum. Third, the spectrum was used to
normalize for speeding up the convergence of the network. We
use the data fusion method to compose the preprocessed data
collected by the three channels into the input unit in the capsule
network module, as shown in Fig. 2. Specifically, each input
network unit has three layers, and each layer contains data for
more than one motion cycle of a sensor channel.


Fig. 2. Principles of multi-sensor data fusion.


In addition, the weight adjustment strategy first uses data
augmentation [20] to generate abnormal single-label fault data,
and then adjusts the proportion of various single-label fault
samples in the training set to construct a balanced dataset. It
allows the model to fully learn the characteristics of various
single-label fault components during the training phase.

_2)_ _Representation Learning Module:_ This module uses a
capsule network to learn representative features of fault signals
after data preprocessing. It includes the following three
modules: convolution operation、primary capsules and digital
capsules. For a given input _z_, the nonlinear representation 𝑔 𝑖
can be obtained by the convolutional layer through the rectified
linear unit (RELU) or sigmoid activation, the formula is as
shown:


𝑔 𝑖 = 𝑓(𝐾𝑒𝑟 𝑖 ⊗𝑧+ 𝑏 𝑖 )             (1)


where _Ker_ _i_ and _b_ _i_ represent the weights and bias of the
convolutional kernel. Besides, _f_ and  indicate the nonlinear
activation function and convolutional calculation.


Then, a novel nonlinear activation function called “squash”
is used to transform the output vector length into the interval of

[0,1] in primary capsules, which can represent the probability
of various categories. When the given input vector is _u_, the
formula of the squash function is as follows:


𝑠𝑞𝑢𝑎𝑠ℎ(𝑢) = 𝑢‖𝑢‖ / (1 + ‖𝑢‖ [2] )           (2)


The output of the primary layer of the lower level capsules is
named _y_ _[Pcpas]_ . _y_ _ji_ is produced by multiplying the _y_ _i_ _[Pcpas ]_ with the
transformation matrix _W_ _ij_, and _d_ _j_ is the weighted sum of all
middle prediction vectors. They can be obtained as follows:


𝑦 𝑗𝑖 = 𝑊 𝑖𝑗 𝑦 𝑖𝑃𝑐𝑎𝑝𝑠 (3)


𝑑 𝑗 = ∑𝑐 𝑖 𝑖𝑗 𝑦 𝑗𝑖 (4)


𝐷𝑖𝑔𝑖𝑡𝐶𝑎𝑝𝑠

𝑣 𝑗 = 𝑠𝑞𝑢𝑎𝑠ℎ(𝑑 𝑗 )              (5)



where _v_ _j_ _[DigitCaps]_ is the output vector of the higher level of the
digital capsule _j_ . _C_ represents the number of labels, and _c_ _ij_ is the
coefficient updated by the agreement-based dynamic routing
algorithm, which can be expressed as:


𝐶
𝑐 𝑖𝑗 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝑏 𝑖𝑗 ) = 𝑒𝑥𝑝(𝑏 𝑖𝑗 ) / ∑ 𝑐=1 𝑒𝑥𝑝(𝑏 𝑖𝑐 )   (6)


where _b_ _ij_ is the log prior probabilities that the capsule _i_ couple
the capsule _j_ . The "Routing agreement" [21] aims to construct a
complex nonlinear mapping between two consecutive capsule
layers in a clustering manner, which is denoted as follows:



𝑃 𝑖𝑗 = 𝑃(𝐿 𝑗 / 𝐿 𝑖 )                 (9)


_P(L_ _j_ _| L_ _i_ _)_ is used to measure the probability of the appearance of
the _L_ _j_ label when the _L_ _i_ label appears. Similar to the standard
convolution logic of CNN, the graph convolution layer
performs convolution operations through the neighbors of a specific graph node. The node feature  _h_ _ui_ of the graph
convolution is obtained by applying the activation function _f_ on
the neighbor feature _h_ _uj_, which can be obtained as follows:


1
ℎ [̅] 𝑢 𝑖 = 𝑓(∑ 𝑗∈𝑁 𝑖 𝑐 𝑖𝑗 ℎ 𝑢 𝑗 𝑤)             (10)


where _N_ _i_ is the index set of neighbor nodes of the node _u_ _i_, _w_ is a
learned weight, and _c_ _ij_ is a constant parameter for the edge
( _u_ _i_, _u_ _j_ ).

GCN takes the feature matrix as input, and each node has a
feature vector. Then, label vectorization (Glove) is used to
represent a specific word of the dictionary and calculate this
space to find the relationships between the labels. To avoid the
problem of overfitting, the weighted adjacency matrix _A_ _ij_ is
calculated as follow:


𝐴
𝑖𝑗 = { [0, 𝑖𝑓 𝑃] 1, 𝑖𝑓 𝑃 [𝑖𝑗] 𝑖𝑗 ≥𝜏 [< 𝜏][               (11) ]


where  = 0.1 was used as the threshold to filter the pairs of _A_ _ij_ .
Besides, the reweighted adjacency matrix _A’_ _ij_ is continue used
to prevent over-smoothing by introducing parameter _p_ ( _p_ =0.25),
which is used to calibrate the weights assigned to the node itself
and other related nodes.



𝐷𝑖𝑔𝑖𝑡𝐶𝑎𝑝𝑠
, 𝑦 𝑗𝑖 ) = < 𝑣 𝑗



𝑎
𝑖𝑗 = 𝑅𝑜𝑢𝑡𝑖𝑛𝑔(𝑣 𝑗



𝐷𝑖𝑔𝑖𝑡𝐶𝑎𝑝𝑠
, 𝑦 𝑗𝑖      -  (7)



𝑏 𝑖𝑗 = 𝑏 𝑖𝑗′ + 𝑎 𝑖𝑗 (8)


Routing(.) is a scalar product function, and _b_ _ij_ is updated by
adding _a_ _ij_ to the previous 𝑏 𝑖𝑗′ .

_3)_ _Graph Learning Module:_ This module uses GCNs to
identify the various components of a compound fault, which
can learn the label characteristics of each single-label fault
signal. The graph is a structure used to encode the relationship
between various objects. Objects are represented by "nodes",
and relationships between nodes are represented by "edges".
Edges use weights to indicate the strength of the relationship
between nodes. We use nodes to represent different types of
single-label faults, and then get the adjacency matrix 𝐴∈𝑅 [𝐶×𝐶]
( _C_ is the number of labels) used to represent the topological
relationship. The element _P_ _ij_ in _A_ is calculated as:



𝑖𝑗′ + 𝑎 𝑖𝑗 (8)



Routing(.) is a scalar product function, and _b_ _ij_ is updated by
adding _a_ _ij_ to the previous 𝑏 𝑖𝑗′ .



0278-0046 (c) 2021 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.

Authorized licensed use limited to: Universita' Politecnica delle Marche. Downloaded on November 11,2022 at 18:34:06 UTC from IEEE Xplore. Restrictions apply.


This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TIE.2022.3176280, IEEE

Transactions on Industrial Electronics

IEEE TRANSACTIONS ON INDUSTRIAL ELECTRONICS



𝐴 ′𝑖𝑗 = { [𝑝 / ∑] 1 −𝑝, 𝑖𝑓 𝑖= 𝑗 𝐶𝑗=1,𝑖≠𝑗 𝐴 𝑖𝑗, 𝑖𝑓 𝑖≠𝑗         (12)


Therefore, GCNs that contain multi-layer graph convolution
operations were used as a label relationship extractor of the
complex and interconnected tasks in real life.

_4)_ _Generated Classifiers Module:_ The generated classifier of
this module performs dot product on the output feature matrices
of the two learning modules to obtain the result of the output
possibility of each label. First, data features of length _N_
obtained by the representation module will use vectorized
neurons to represent them instead of using scalar data. Second,
the graph features from the adjacency matrix will also be
obtained by GCN, which will stack multiple graph
convolutional layers and activation functions. Then, we will
perform dot product on the two output features obtained to
consider the contribution of data samples and graphics to
category prediction. In addition, 𝑊 [̅] is the set of correlation 𝑤̅ 𝑖
of each category learned through GCN. Finally, the calculation
process of sample multi-label prediction results is as follows:


𝑊 [̅ ] = {𝑤̅ 𝑖 } 𝐶𝑖=1, (𝑊 [̅ ] ∈𝑅 [𝐶×𝑁] )            (13)


𝑦̅ = 𝑊 [̅ ] 𝑥, (𝑦∈𝑅 [𝐶] )              (14)


where _C_ is the number of categories, _x_ is the feature vector
learned by the capsule network, and  _y_ is the predicted multi-label learned by the classifiers. _y_ is the ground truth label,

Therefore, our proposed DCGCN method for intelligent
diagnosis of compound faults can be summarized by
pseudo-code, as shown in Table I:

TABLE I

A LGORITHM OF DCGCN M ODEL


_B._ _Margin Loss Function and Evaluation Metrics_


In our designed neural network, we want to detect multiple
digits in one data sample through a proper loss function. The
margin loss function [22] is used to increase the feature
discrimination of different types of faults and can optimize the
classification performance of the network model. It adopts the
novel separate margin loss _L_ _c_ for each category _c_ digit, which is
calculated as follow:


𝐿 𝑐 = 𝐾 𝑐 𝑚𝑎𝑥(0, 𝑏 [+] −‖𝑣 𝑐 ‖) [2] + 𝜆𝐾 𝑐′ 𝑚𝑎𝑥(0, ‖𝑣 𝑐 ‖ −𝑏 − ) 2 (15)


Where _K_ _c_ =1 if an object of class _c_ appears, and _K’_ _c_ =1- _K_ _c_ .
Besides, _b_ _[+]_ =0.9 and _b_ _[-]_ =0.1 means the lower and upper
boundary of  _v_ _c_  . The down-weighting parameter  (  _=0.5_ ) is
used to stop the initial learning from shrinking the activity



vectors of all classes. Finally, the total margin loss computing is
the sum of the losses of all classes.


In addition, we use Precision, Recall and F1-score evaluation
indicators [23] to measure the performance of the network.
True Positive (TP) means that the positive sample is
successfully predicted to be positive. True Negative (TN)
means that negative samples are successfully predicted to be
negative. False Positive (FP) means that negative samples are
incorrectly predicted as positive. False Negative (FN) means
that the positive samples are incorrectly predicted as negative.


_Precision_ = _TP_ / ( _TP_ + _FP_ )               (16)


_Recall_ = _TP_ / ( _TP_ + _FN_ )                  (17)


_F1-score_ = (2* _Precision_  - _Recall_ ) / ( _Precision_ + _Recall_ )  (18)


Formulas (16)-(18) are used to calculate the Precision, Recall,
and F1-score of the accumulation of various types of fault
components in the compound fault sample.


_C._ _Output Principle of the DCGCN_


The diagnosis of compound faults in mechanical equipment
is a multi-label classification problem. One-Hot coding is used
to represent samples in multi-label classification tasks, and a
sample may contain multiple labels at the same time.
Traditional intelligent fault diagnosis methods usually identify
the compound fault as a single fault or are only suitable for
identifying double-label compound faults. As the compound
fault contains more fault components, it will become more
difficult to accurately identify all types of faults. However, our
proposed method can diagnose the triple-label compound faults
of harmonic drives, as shown in Fig. 3. The biggest advantage
of our proposed method is that it can successfully identify
compound faults with more fault components, and there is no
missed detection or false detection, which is more suitable for
real and complex electromechanical system scenarios.


Fig. 3. Comparison of different diagnosis models.


In addition, our compound diagnostic model outputs the
possibility of various individual labels in each test sample and
then uses a threshold (  =0.5) to filter the output of each label.
When the predicted probability value of a certain label is
greater than , it is considered that the compound fault contains
this type of fault.



0278-0046 (c) 2021 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.

Authorized licensed use limited to: Universita' Politecnica delle Marche. Downloaded on November 11,2022 at 18:34:06 UTC from IEEE Xplore. Restrictions apply.


This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TIE.2022.3176280, IEEE

Transactions on Industrial Electronics

IEEE TRANSACTIONS ON INDUSTRIAL ELECTRONICS



III. E XPERIMENTS


_A._ _Setup of the Experimental platform_


Fig. 4. Industrial robot vibration test bench.


The industrial robot vibration test bench includes a six-axis
industrial robot, a servo controller, a SKF vibration analyzer
(System: IMx-P), three vibration acceleration sensors (Model:
CMSS2200), and a personal computer, as shown in Fig. 4.
Three vibration acceleration sensors are installed orthogonally
at the end of the industrial robot to collect the vibration signal
of the robot [20]. The 4 [th], 5 [th] and 6 [th] axes of industrial robots all
use harmonic drives. During the experiment, the range of
motion angles of the 4 [th], 5 [th], and 6 [th] axes are ±105 [0], ±123 [0], and
±110 [0] respectively. The sampling frequency is 2560Hz.

To more clearly express the mechanical and physical
mechanism of the 6-axis tandem industrial robot in the
vibration test bench, the structure diagram of the whole
machine is shown as follow:


Fig. 5. Schematic diagram of the mechanical structure of an industrial robot.


The schematic diagram of the industrial robot is shown in Fig.
5. The 1 [st] axis, the 2 [nd ] axis, and the 3 [rd] axis are driven by RV
gears. The RV reducer has the characteristics of simple
structure, high reliability, and few failures. The harmonic
drives of the 4 [th] axis, 5 [th] axis and, 6 [th] axis adopts the principle of
harmonic transmission instead of gear transmission. The
harmonic drive is composed of a wave generator, flexspline,
and circular spline. It relies on the elastic deformation of the
flexspline to realize movement transmission. Besides, the
harmonic drive is extremely sensitive to the manufacturing
process and assembly errors. Therefore, multiple harmonic
drives of industrial robots sometimes will cause abnormal

vibrations at the same time.



The physical structures of the various types of harmonic
drives in Fig. 5 are different, so their fault characteristics are
also different. We use the frequency in Table II to observe their
motion characteristics. The frequency spectrum of the collected
vibration signal contains the frequency components of various
parts, such as motor, pulley, camshaft, flexspline, and circular
spline. The various input speeds of the motor will run under 3
different levels of payload (0/3/8Kg). In Table II, the motors
used for driving have four speeds of 1247/2492/3739/4985
(r/min), corresponding to L/M/H/S respectively. The
calculation formula of the frequency spectral components of
each type of harmonic drive can refer to the literature [4]. These
frequency components will be used for subsequent neural
network analysis. The different types of harmonic drives are
respectively installed in the 4 [th], 5 [th], and 6 [th] axis of the industrial
robot, and the reduction ratio of these products is 50/100/50
respective. Due to the consideration of external output
connection (belt and pulley), the total revised reduction ratio of
these products is 74.538/84.1666/50 respective.


TABLE II

F REQUENCY C OMPONENT D IAGRAM OF V IBRATION A CCELERATION



Circular


spline

base

frequency

(Hz)



Cam

shaft

base

frequency

(Hz)



Flexspline

meshing
frequency

(Hz)



Motor


base

frequency

(Hz)



Model


LHS-25-5


0-C-III


LHSG-17

-100-C-IV


LCSG-17

50-C-II-S


2



Motor


speed
(r/min)



1247 20.78 13.94 56.32 0.28

2492 41.53 27.86 112.56 0.56

3739 62.32 41.80 168.88 0.84

4985 83.08 55.73 225.16 1.11


1247 20.78 24.69 49.88 0.25

2492 41.53 49.35 99.68 0.49

3739 62.32 74.04 149.56 0.74

4985 83.08 98.71 199.40 0.99


1247 20.78 20.78 83.96 0.42

2492 41.53 41.53 167.79 0.83

3739 62.32 62.32 251.76 1.25

4985 83.08 83.08 335.66 1.66



_B._ _The Construction of the Dataset_


The collected continuous vibration data larger than one
motion period are preprocessed to form a sample unit of the
data set. Each sample unit contains 32*32*3=3072 data points.
There are a total of four different operating speeds (L/M/H/S).
Normal signals and various single-label fault signals are used to
construct the training dataset of the network. Then, various
single-label samples and compound fault samples are used to
compose the testing dataset. Besides, Fault 1, Fault 2, and Fault
3 of abnormal vibration occurred on the harmonic drive of the
4 [th] axis, 5 [th] axis, and 6 [th] axis respectively. These three types of
faults are abnormal jitters, and Fault 2 is severe jitter
accompanied by harsh noises. The description of the dataset
was given in Table III.


Table III

D ESCRIPTION OF T HE D ATASET USED FOR DCGCN N ETWORK


Speed Label Number of sample units


Training Normal/ Fault 1/
L/M/H/S 2850/2850/2850/2850
Dataset Fault 2/ Fault 3


Normal/ Fault 1/ Fault



Testing
L/M/H/S
Dataset



2/ Fault 3/ Fault 1&2/

Fault 1&3/ Fault 2&3/


Fault 1&2&3



1425/1425/1425/1425

/1425/1425/1425/1425



0278-0046 (c) 2021 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.

Authorized licensed use limited to: Universita' Politecnica delle Marche. Downloaded on November 11,2022 at 18:34:06 UTC from IEEE Xplore. Restrictions apply.


This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TIE.2022.3176280, IEEE

Transactions on Industrial Electronics

IEEE TRANSACTIONS ON INDUSTRIAL ELECTRONICS



The collected vibration acceleration data is divided into a
training dataset and a testing dataset after data preprocessing.
The training dataset is composed of the normal data, fault 1,
fault 2, and fault 3 signals, and the sample amount of each type
of health status data in the training dataset under various speed
conditions is 2850. The testing dataset includes various
single-label health status samples and compound fault samples,
and the sample amount of each type of health status data in the
testing dataset under various speed conditions is 1425. The
training dataset samples are all single-label samples, and the
testing dataset samples contain both single-label and
multi-label compound fault samples. There are four types of
health status data in the training dataset and eight types of
health status data in the testing dataset.


IV. R ESULTS AND D ISCUSSIONS


_A._ _The Results of FFT Analysis_


The vibration acceleration signal collected on every
single-label fault and the compound fault of the harmonic
drives under H working conditions with sensor 2 is shown in
Fig. 6.


Fig. 6. Vibration acceleration signal of (a) Fault 1. (b) Fault 2. (c) Fault 3. (d)

Compound fault (Fault1 & Fault2 & Fault3).


Moreover, the frequency spectrum obtained by performing
FFT transformation on the fault vibration acceleration signals
of the harmonic drives is shown in Fig. 7.



Fig. 7. The frequency spectrum of (a) Fault 1. (b) Fault 2. (c) Fault 3. (d)

Compound fault (Fault1 & Fault2 & Fault3).


From Fig. 6 and Fig. 7, it can be seen that the vibration
accelerations and frequency spectrums of each type of fault are
very different. It is difficult to use traditional various fault
diagnosis methods to directly separate and diagnose various
components in the compound fault signal of the harmonic drive.
However, in our model, the frequency spectrum of the vibration
acceleration signal of various single-label faults is directly used
for subsequent network model learning, which can successfully
achieve the CFD.


_B._ _Parameters of the Proposed Network_


To give full play to the performance of the network model,
the parameter selection of the proposed DCGCN network is
particularly important. The detailed structural parameters of our
proposed network are given in Table IV.


Table IV

T HE N ETWORK S TRUCTURE OF DCGCN M ODEL


Layer name DCGCN Model

_**Capsule network**_

InputSize Input unit=32x32x3

Conv1 Conv2d(3,256,kernel size=(5,5),stride=(1,1),padding=(1,1))

PrimaryCaps(num_conv_units=32,in_channels=256,
PrimaryCap

out_channels=8,kernel_size=9,stride=2)


DigitCaps(in_dim=8,in_caps=32 * 6 * 6, out_caps=3,
DigitCaps

out_dim=16,num_routing=3)


Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)),
Visual Feature

output_shape=(48,))(capsule)

_**GCN network**_

GC1 x1=GraphConvolution(in_channel, 2), LeakyReLU(0.2)

GC2 x2=GraphConvolution(2, 3), LeakyReLU(0.2)
Graph Feature x2.transpose(0, 1)

Dot Product x3= torch.matmul(Visual Feature, sigmoid(x2))

L2 Norm _Lc_ =Margin loss function(), Output dimension = [3, 1]
The input size of the sample unit is 32x32x3. The capsule
network consists of a convolutional layer with a ReLU
activation function and a capsule layer (includes a primary
capsule and a digital capsule). The GCN network contains two
graph convolutional layers with LeakyReLU activation
functions. The number of iterations during the routing
algorithm is 3. The learning rate is 5e-6, the batch size is 6, the
epoch number is 35, and the optimizer is Adam. The loss
function of the network is margin loss. Besides, the key
parameters of the DCGCN model are given in Table V, which
is obtained from debugging experience.


Table V

T HE K EY P ARAMETERS OF DCGCN MODEL


Parameters   _b_ _[+]_ _b_ _[-]_ _In_channel_ _t_ _p_

value 0.5 0.5 0.9 0.1 300 0.1 0.25


The configuration of the laptop used for the experiment is:
Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz 2.59 GHz.


_C._ _The Results of Compound Fault Diagnosis_


_1) Experimental Results of Data Fusion:_


According to industry test specifications, three vibration
acceleration sensors are used to collect vibration information in
three orthogonal directions. The DCGCN model is used to



0278-0046 (c) 2021 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.

Authorized licensed use limited to: Universita' Politecnica delle Marche. Downloaded on November 11,2022 at 18:34:06 UTC from IEEE Xplore. Restrictions apply.


This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TIE.2022.3176280, IEEE

Transactions on Industrial Electronics

IEEE TRANSACTIONS ON INDUSTRIAL ELECTRONICS



evaluate the performance of each sensor and multi-sensor data
fusion. We use the possibilities identification value of each
fault in the compound fault to verify the compound fault
diagnosis performance of the proposed method under the S
speed working conditions. The average output probabilities
obtained from five repeated experiments are shown in Table
VI:


Table VI

D IAGNOSIS R ESULTS OF D IFFERENT D ATA C HANNELS


Average output probabilities of DCGCN under S speed (%)
Label

sensor 1 sensor 2 sensor 3 Multi-sensors

Fault 1 83.26 92.01 94.84 81.46

Fault 2 77.19 94.81 85.33 70.11

Fault 3 90.34 98.19 95.05 83.65


It can be seen from Table VI that the three types of
single-fault contained in the compound fault samples used for
testing have been accurately identified, and the DCGCN
method performs well in each channel and data fusion. Besides,
the installation direction of vibration sensor 1 is parallel to the
rotation center of the output shaft, and the installation direction
of vibration sensors 2 and 3 is perpendicular to the rotation
center of the shaft. The vibration information captured by
sensors 2 and 3 is the most sensitive. From the comparison data
in Table VI, it can also be found that the best results are
obtained using the data of sensor 2. Therefore, we will use the
data of this channel for subsequent analysis.


_2) Output Probabilities of Compound Fault Diagnosis:_


The fault samples of four different speed conditions (L, M, H,
and S) are used to verify the compound fault diagnosis
performance of the DCGCN model. First, various types of
single-label fault samples under the same working condition
were used to train the model. Second, the triple-label compound
fault samples were used for testing. In each working condition,
the number of three types of single-label fault samples used for
training is 450, and the numbers of the triple-label compound
fault samples used for testing are 445, 420, 440, and 360,
respectively. The predicted output probabilities of each
compound fault sample of the last testing epoch under the S
speed condition are shown in Fig. 8.


Fig. 8. Output probabilities of the compound fault samples under S speed.


It can be seen from Fig. 8 that under the S speed condition,
the predicted output probabilities of each fault component in
each triple compound fault sample is greater than 0.8, which is
far greater than the set threshold value of 0.5. The average
output probabilities under the S speed condition are 0.9019,
0.9188, and 0.9588. Furthermore, the evaluation index



Precision, Recall, and F1-score of the compound fault samples
under L/M/H/S working conditions can calculate according to
formula (16) to (18), which are given in Table VII:


Table VII

E VALUATION INDEX RESULTS OF COMPOUND FAULTS UNDER DIFFERENT


WORKING CONDITIONS


Speed Precision Recall F1-score

L 1.000 0.999 0.999

M 1.000 1.000 1.000

H 1.000 0.993 0.996

S 1.000 1.000 1.000


It can be seen from Table VII that the proposed model can
obtain ideal evaluation index values under varying working
conditions. This also shows that the model has excellent
compound fault diagnosis capabilities.


_D._ _Comparison Results of Different Methods_


There are three main categories of compound fault diagnosis
approaches. (1) Traditional diagnosis methods: IMFs and PBR;
(2) End-to-end machine learning models: FFT-DNN,
CNN-Multilabel classifier (CNN-MLC), Deep CNN (DCNN),
InceptionV3 (ICN), and Capsule Network (CapsNet). (3)
Graph-based neural networks: GCN and GAT. To make a fair
comparison, we use the same dataset of the S speed working
conditions to conduct experiments and perform five repetitions
to obtain average probabilities. IMFs use the first 6-order
energy spectrum as the feature vector and adopt the
multi-output SVM classifier. The hidden units of each layer of
the FFT-DNN model are 1536, 768, 256, 64, and 3 respectively.
DCNN uses binary cross-entropy, and the last layer of
activation function is Softmax. CNN-MLC uses the Margin
loss function, and the last layer of activation function is sigmoid.
The Inceptionv3 (ICN) pre-trained model contains 3 Inception
modules. The loss function of the CapsNet is Margin loss. The
feature learning module of GCN is the resnext50_32x4d
pre-training model and contains a 2-layer graph convolutional
layer. The dropout value of GraphAttentionLayer in GAT is 0.6,
alpha is 0.2, and concat is True. The results obtained by various
methods are given in Table VIII.

It can be seen from Table VIII that:
1) Various traditional analysis methods and machine

learning methods can obtain excellent performance in the
fault diagnosis of the single-label fault (fault 3).
2) In the identification of dual-label compound faults (fault

1&2, fault 1&3, fault 2&3), many intelligent methods
can only identify the fault components of part of the
compound fault combination samples.
3) DCGCN can accurately diagnose each fault component

of the triple-label compound faults (fault 1&2&3), which
is better than only using CapsNet or GCN.
Therefore, the DCGCN method obtains the highest output
possibility value in various compound fault diagnosis scenarios,
which also shows that the feature vector and the relationship
graph are helpful to identify multi-label compound faults,
thereby improving the performance of our proposed model.



0278-0046 (c) 2021 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.

Authorized licensed use limited to: Universita' Politecnica delle Marche. Downloaded on November 11,2022 at 18:34:06 UTC from IEEE Xplore. Restrictions apply.


This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TIE.2022.3176280, IEEE

Transactions on Industrial Electronics

IEEE TRANSACTIONS ON INDUSTRIAL ELECTRONICS


Table VIII

T HE A VERAGE O UTPUT P ROBABILITY OF C OMPOUND F AULT D IAGNOSIS


Average Output Probability of Each Type of Fault(%)
Methods
Normal Fault 1 Fault 2 Fault 3 Fault 1&2 Fault 1&3 Fault 2&3 Fault 1&2&3


IMFs [10] 100 100 76.80 100 54.13, 38.98 77.24, 64.41 38.59, 66.82 77.24, 38.98, 52.49


PBR [11] 75.76 100 100 100 57.23, 24.49 50.67, 75.34 24.00, 56.57 57.23, 24.49, 69.08


FFT-DNN [13] 99.96 100 99.98 99.99 91.83, 78.05 87.84, 84.00 74.57, 62.97 45.58, 21.54, 9.16


CNN-MLC [14] 100 100 100 100 52.60, 55.50 52.20, 52.61 53.97, 54.96 50.36, 55.96, 55.73


DCNN [14] 100 100 100 100 47.27, 52.73 48.18, 51.82 48.25, 51.75 28.51, 34.25, 37.24


ICN [16] 97.14 86.55 99.07 99.29 27.48, 24.11 77.84, 97.90 52.04, 25.81 63.88, 0.05, 11.31


CapsNet [17] 92.21 91.12 92.39 90.75 87.06, 51.21 86.71, 45.23 82.08, 63.13 83.04, 52.20, 45.16


GCN [18] 72.87 96.35 98.22 94.30 80.30, 56.87 55.66, 93.87 89.86, 24.23 64.82, 33.57, 20.24


GAT [19] 84.44 71.41 96.57 86.24 18.12, 69.03 92.03, 6.23 61.08, 7.51 51.12, 31.99, 27.62


**DCGCN** **94.03** **95.63** **95.16** **99.86** **94.72, 98.77** **97.19, 96.02** **96.75, 98.29** **93.28, 95.84, 98.14**



_E._ _Ablation Experiment and Discussion:_


The essence of compound fault diagnosis is the problem of
multi-label identification. Even if a higher average output
probability value can be obtained, the three evaluation
indicators of Precision, Recall, and F1-score are still needed to
measure the true performance of the network model. We only
use 3 types of single-label fault samples (data volume of each
type is 450) for training, and the test set contains triple-label
compound fault samples (data volume is 360) and single-label
fault samples (data volume is 360). The results obtained are
shown in Table IX:


Table IX

E VALUATION RESULTS OF ABLATION EXPERIMENTS


Method Precision Recall F1-score Training time (s)

IMFs 0.992 0.494 0.743 53

FFT-DNN 1.000 0.485 0.742 115

CNN-MLC 0.667 1.000 0.833 65

CapsNet 1.000 0.764 0.837 260

GCN 0.817 0.599 0.654 594

**DCGCN** **1.000** **1.000** **1.000** **265**


It can be found from Table IX that when no compound fault
samples participate in training, the precision of GCN is 0.817,
and the precision of CapsNet and DCGCN methods are both 1.
Although the precision of these methods is very high, the values
of Recall and F1-score are quite different. For example, for a
fault 1 sample with a true label of (1,0,0), the output probability
of CapsNet is (0.46, 0.32, 0.13), so that the sample is
incorrectly judged as not containing any type of fault
components. In addition, it can be seen from Table VIII that our
DCGCN model can accurately identify each component in all
compound fault samples, and the predicted average value is far
greater than the threshold of 0.5.

The results of Table VIII and Table IX show that the
proposed DCGCN model has the following advantages: 1) It
can realize the diagnosis of compound faults without the need
for compound samples to participate in model training. 2)
When diagnosing the compound fault samples in the test set, it



will not miss the detection, nor will the single fault sample be
falsely detected or more faults are detected, and the predicted
probability value of each single-label fault component has been
improved. This not only guarantees the ideal Precision, Recall,
and F1-score results, but also improves the output probability
of various fault components in the compound fault sample.


V. C ONCLUSION AND F UTURE WORK


To solve the problem of compound fault diagnosis for
harmonic drives installed on industrial robot, we design a new
intelligent fault diagnosis architecture-DCGCN, which only
requires various single-label fault samples to achieve the fault
diagnosis of compound faults. In our approach, the capsule
network is employed to effectively capture the vector features
of fault samples, and the graph convolutional network is
employed to learn the relationship of various single-label faults.
The feature matrix obtained by these two sub-networks is
utilized to output the predicted probabilities of the various
single-fault component in each compound fault sample through
the dot product. The dynamic routing algorithm and margin
loss function are used to optimize the performance of DCGCN.
Through the real experiments of the industrial robot platform,
the compound fault diagnosis results of DCGCN are better than
IMFs, PBR, DCNN, CNN-MLC, CapsNet, GCN, and GAT.
Moreover, DCGCN can obtain high Precision, Recall, and
F1-score, which proves our model has excellent diagnostic
performance and has the potential to apply in real factory
scenarios.

In the future, we will further simplify the DCGCN network
structure and improve the model's fault diagnosis performance
under variable working conditions. In addition, we will also
study the identification of weak faults in the compound faults of
the harmonic drives.


R EFERENCES


[1] Tuttle, Timothy D. Understanding and modeling the behavior of a

harmonic drive gear transmission. No. AI-TR-1365.
MASSACHUSETTS INST OF TECH CAMBRIDGE ARTIFICIAL



0278-0046 (c) 2021 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.

Authorized licensed use limited to: Universita' Politecnica delle Marche. Downloaded on November 11,2022 at 18:34:06 UTC from IEEE Xplore. Restrictions apply.


This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TIE.2022.3176280, IEEE

Transactions on Industrial Electronics

IEEE TRANSACTIONS ON INDUSTRIAL ELECTRONICS



INTELLIGENCE LAB, 1992.

[2] Park, In-Gyu, Il-Kyeom Kim, and Min-Gyu Kim. "Vibrational

characteristics of developed harmonic reduction gear and fault diagnosis
by campbell diagram." In _2015 15th International Conference on Control,_
_Automation and Systems (ICCAS)_, pp. 2062-2065. _IEEE_, 2015.

[3] Adams, Christian, Adam Skowronek, Joachim Bös, and Tobias Melz.

"Vibrations of Elliptically Shaped Bearings in Strain Wave Gearings."
_Journal of Vibration and Acoustics_ 138, no. 2 (2016).

[4] G. Yang, Y. Zhong, L. Yang and R. Du, "Fault Detection of Harmonic

Drive Using Multiscale Convolutional Neural Network," in _IEEE Trans._
_Instrum. Meas_, vol. 70, pp. 1-11, 2021, Art no. 3502411, DOI:
10.1109/TIM.2020.3024355.

[5] L. Wen, X. Li, L. Gao, and Y. Zhang, “A new convolutional neural

network-based data-driven fault diagnosis method,” _IEEE Trans. Ind._
_Electron_ ., vol. 65, no. 7, pp. 5990-5998, Jul. 2018.

[6] J. Liu, Y. Hu, Y. Wang, B. Wu, J. Fan, Z. Hu, An integrated multi-sensor

fusion-based deep feature learning approach for rotating machinery
diagnosis, _Meas. Sci. Technol_ . 29 (2018).

[7] G. Jiang, H. He, J. Yan, and P. Xie, “Multiscale convolutional neural

networks for fault diagnosis of wind turbine gearbox,” _IEEE Trans. Ind._
_Electron_ ., vol. 66, no. 4, pp.3196-3207, Apr. 2019.

[8] C. Zhang and Y. Liu, "A Two-Step Denoising Strategy for Early-Stage

Fault Diagnosis of Rolling Bearings," in _IEEE Trans. Instrum. Meas_ ., vol.
69, no. 9, pp. 6250-6261, Sept. 2020, doi: 10.1109/TIM.2020.2969092.

[9] X. Yang, K. Ding, G. He, and Y. Li, “Double-dictionary signal

decomposition method based on split augmented Lagrangian shrinkage
algorithm and its application in gearbox hybrid faults diagnosis,” _J. Sound_
_Vib_ ., vol. 432, pp. 484–501, Oct. 2018.

[10] J. Zhang, Q. Zhang, X. He, G. Sun and D. Zhou, "Compound-Fault

Diagnosis of Rotating Machinery: A Fused Imbalance Learning Method,"
in _IEEE Trans. Control Syst. Technol,_ vol. 29, no. 4, pp. 1462-1474, July
2021, doi: 10.1109/TCST.2020.3015514.

[11] X. Ma, Y. Hu, M. Wang, F. Li and Y. Wang, "Degradation State Partition

and Compound Fault Diagnosis of Rolling Bearing Based on
Personalized Multilabel Learning," in _IEEE Trans. Instrum. Meas,_ vol.
70, pp. 1-11, 2021, Art no. 3520711, doi: 10.1109/TIM.2021.3091504.

[12] C. Sun, X. Chen, R. Yan, and R. X. Gao, “Composite graph-based sparse

subspace clustering for machine fault diagnosis,” _IEEE Trans. Instrum._
_Meas_ ., to be published, doi: 10.1109/TIM.2019.2923829.

[13] F. Jia, Y. Lei, J. Lin, X. Zhou, and N. Lu, “Deep neural networks: A

promising tool for fault characteristic mining and intelligent diagnosis of
rotating machinery with massive data,” _Mech. Syst. Signal Process_ ., vol.
72–73, pp. 303–315, May 2016.

[14] R. Huang, W. Li and L. Cui, "An Intelligent Compound Fault Diagnosis

Method Using One-Dimensional Deep Convolutional Neural Network
With Multi-Label Classifier," _2019 IEEE International Instrumentation_
_and Measurement Technology Conference (I2MTC),_ 2019, pp. 1-6, doi:
10.1109/I2MTC.2019.8827030.

[15] R. Huang, Y. Liao, S. Zhang and W. Li, "Deep Decoupling Convolutional

Neural Network for Intelligent Compound Fault Diagnosis," in _IEEE_
_Access_, vol. 7, pp. 1848-1858, 2019, doi:
10.1109/ACCESS.2018.2886343.

[16] Yadavendra and S. Chand, "Multiclass and Multilabel Classification of

Human Cell Components Using Transfer Learning of InceptionV3
Model," _2021 International Conference on Computing, Communication,_
_and_ _Intelligent_ _Systems_ _(ICCCIS),_ 2021, pp. 523-528, doi:
10.1109/ICCCIS51004.2021.9397165.

[17] Z. Zhu, G. Peng, Y. Chen, and H. Gao, “A convolutional neural network

based on a capsule network with strong generalization for bearing fault
diagnosis,” _Neurocomputing_, vol. 323, pp. 62–75, Jan. 2019.

[18] Z. Chen, et al. "Multi-label image recognition with graph convolutional

networks." _Proceedings of the IEEE/CVF Conference on Computer_
_Vision and Pattern Recognition_ . 2019.

[19] B. Hu, K. Guo, X. Wang, J. Zhang and D. Zhou, "RRL-GAT: Graph

Attention Network-driven Multi-Label Image Robust Representation
Learning," in _IEEE_ _Internet_ _of_ _Things_ _Journal_, doi:
10.1109/JIOT.2021.3089180.

[20] G. Yang, Y. Zhong, L. Yang, H. Tao, J. Li and R. Du, "Fault Diagnosis of

Harmonic Drive With Imbalanced Data Using Generative Adversarial
Network," in _IEEE Trans. Instrum. Meas_, vol. 70, pp. 1-11, 2021, Art no.
3519911, doi: 10.1109/TIM.2021.3089240.

[21] S. Sabour, N. Frosst, and G. E. Hinton, “Dynamic routing between

capsules,” Oct. 2017, _arXiv:1710.09829_ . [Online]. Available:
https://arxiv.org/abs/1710.09829.




[22] R. Gao, F. Yang, W. Yang, and Q. Liao, “Margin loss: Making faces more

separable,” _IEEE Signal Process. Lett_ ., vol. 25, no. 2, pp. 308–312, Feb.
2018.

[23] Y. He and Z. Liu, "A Feature Fusion Method to Improve the Driving

Obstacle Detection Under Foggy Weather," in _IEEE Trans. Transport._
_Electrific._, vol. 7, no. 4, pp. 2505-2515, Dec. 2021, doi:
10.1109/TTE.2021.3080690.


**Guo Yang** received the M.S degree in
mechanical engineering from Guangdong
University of Technology, Guangzhou, China, in
2018.

He is currently a Ph.D. candidate with the
Shien-Ming Wu School of Intelligent
Engineering, South China University of
Technology, Guangzhou, China. His main
research focuses on intelligent monitoring,
diagnosis and control of advanced
manufacturing systems.


**Hui Tao** received the M.S. degree in control
theory and control engineering from Wuhan
University of Science and Technology, Wuhan,
China, in 2011.
He is currently a Ph.D. candidate with the
Shien-Ming Wu School of Intelligent
Engineering, South China University of
Technology, Guangzhou, China. His main
research focuses on intelligent monitoring,
diagnosis and control.


**Ruxu Du** received the M.S. degree in automatic
control from the South China University of
Technology, Guangzhou, China, in 1983, and
the Ph.D. degree in mechanical engineering from
the University of Michigan, Ann Arbor, MI, USA,
in 1989. From 1991 to 2001, he taught at the
University of Windsor, Windsor, ON, Canada; at
the University of Miami, Coral Gables, FL, USA;
and at Chinese University of Hong Kong
(CUHK), Shatin, Hong Kong. Since 2002, he has
been a Professor with the Department of Mechanical and Automation
Engineering, and the Director of the Institute of Precision Engineering,
CUHK. His research interests include precision engineering, condition
monitoring, fault diagnosis, manufacturing processes (metal forming,
machining, plastic injection molding, etc.), and robotics.
Dr. Du became a Fellow of the America Society of Mechanical
Engineers (ASME) in 2009, a Fellow of the Society of Manufacturing
Engineers (SME), and the Hong Kong Institute of Engineers (HKIE) in
2012, and an Academician of Canadian Academy of Engineering in
2017.


**Yong Zhong** (S’12–M’17) received the B.Eng.
degree in mechanical design, manufacturing and
automation from the Huazhong University of
Science and Technology, Hubei, China, in 2011,
and the M.Eng. degree in control engineering
from the University of Chinese Academy of
Sciences, Beijing, China, in 2014, and the Ph.D.
degree in Mechanical and Automation
Engineering from the Chinese University of Hong
Kong, Shatin, Hong Kong, in 2017. He was a
research fellow at National University of Singapore from 2017 to 2019.

He is currently an Assistant Professor with Shien-Ming Wu School of
Intelligent Engineering, South China University of Technology. His
research interests include bioinspired robots, soft robots, intelligent
diagnosis and control.



0278-0046 (c) 2021 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.

Authorized licensed use limited to: Universita' Politecnica delle Marche. Downloaded on November 11,2022 at 18:34:06 UTC from IEEE Xplore. Restrictions apply.


