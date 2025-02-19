2021                                                                                                                                                                                                                                                                                            1
            RA-GCN: Graph Convolutional Network for Disease
                         Prediction Problems with Imbalanced Data
               Mahsa Ghorbani∗†, Anees Kazi†, Mahdieh Soleymani Baghshah∗, Hamid R. Rabiee∗, Nassir Navab†,‡,∗Department of Computer Engineering, Sharif University of Technology, Tehran, Iran
                †Computer Aided Medical Procedures, Department of Informatics, Technical University of Munich, Germany
                                   ‡Whiting School of Engineering, Johns Hopkins University, Baltimore, USA
   Abstract—Disease  prediction  is  a  well-known  classiﬁcation
problem in medical applications. Graph Convolutional Networks
(GCNs) provide a powerful tool for analyzing the patients’ fea-
tures relative to each other. This can be achieved by modeling the
problem as a graph node classiﬁcation task, where each node is a
patient. Due to the nature of such medical datasets, class imbal-
ance is a prevalent issue in the ﬁeld of disease prediction, where
the distribution of classes is skewed. When the class imbalance
is present in the data, the existing graph-based classiﬁers tend
to be biased towards the major class(es) and neglect the samples
in the minor class(es). On the other hand, the correct diagnosis                  Fig. 1: Node classiﬁcation task. Label for a subset of nodes
of the rare positive cases (true-positives) among all the patients                and the graph between all nodes are available. The goal is to
is vital in a healthcare system. In conventional methods, such                    predict the label of unlabeled (grey) nodes.
imbalance is tackled by assigning appropriate weights to classes
in the loss function which is still dependent on the relative values
of weights, sensitive to outliers, and in some cases biased towards
the  minor  class(es).  In  this  paper,  we  propose  a  Re-weighted             networks and have achieved unprecedented performance on a
Adversarial Graph Convolutional Network (RA-GCN) to prevent                       broad range of tasks over the past years [4], [5].
the graph-based classiﬁer from emphasizing the samples of any
particular class. This is accomplished by associating a graph-                       Prevalent  deep  neural  networks  do  not  incorporate  the
based  neural  network  to  each  class,  which  is  responsible  for             interaction and association between the patients in their archi-
weighting the class samples and changing the importance of each                   tecture. Considering the relationships between the patients is
sample for the classiﬁer. Therefore, the classiﬁer adjusts itself and             beneﬁcial as it helps to analyze and study the similar cohort
determines  the  boundary  between  classes  with  more  attention                of patients together. By viewing the patients as nodes and
to the important samples.The parameters of the classiﬁer and
weighting  networks  are  trained  by  an  adversarial  approach.                 their associations as edges, graphs provide a natural way of
We show experiments on synthetic and three publicly available                     representing the interactions among a population. Moreover,
medical datasets.Our results demonstrate the superiority of RA-                   when a graph between patients is constructed from a subset
GCN compared to recent methods in identifying the patient’s                       of their features, those features are summarized in the graph
status on all three datasets. The detailed analysis of our method is
provided as quantitative and qualitative experiments on synthetic                 edges and omitted from the feature set. This results in feature
datasets.                                                                         dimensionality reduction to avoids the overﬁtting due to the
   Keywords—Disease  prediction,  Graphs,  Graph  convolutional                   large number of features [6], [7], [8]. Consequently, the focus
networks, Node classiﬁcation, Imbalanced classiﬁcation                            of developing deep learning methods for such data with an
                                                                                  underlyinggraphstructurehaswitnessedatremendousamount
                           I.  INTRODUCTION                                       of  attention  over  the  last  few  years  [9].  Geometric  deep
                                                                                  learning[10]istheﬁeldthatstudiesextendingneuralnetworks
   Disease prediction using medical data has recently shown a                     to graph-structured data by including neighborhoods between
potential application area for machine learning-based meth-                       nodes. Studies in this ﬁeld explore the methods of generalizing
ods  [1],  [2].  Disease  prediction  analyzes  effective  factors                thekeyconceptsofdeeplearning,suchasconvolutionalneural
for  each  patient  to  distinguish  healthy  and  diseased  cases                networks, to graph-structured data and propose graph-based
or  to  recognize  the  type  of  disease.  Predicting  the  disease              neural network architectures. By modeling the medical data
state has considerable value in the clinical domain because                       with a graph, the disease prediction problem turns into the
the  identiﬁcation  of  patients  with  a  higher  probability  of                node classiﬁcation task (depicted in Fig. 1) which has been
developing a chronic disease enhances the chance for better                       explored recently by utilizing Graph Neural Network (GNN)
treatments  [3].  Deep  neural  networks  are  popular  machine                   models [11], [12], [13]. As shown in Fig. 1, the task is to
learning models that started to outperform other methods in a                     predict the class label of the unknown test samples within the
wide range of domains. Most of the state-of-the-art machine                       cohort (patients with gray nodes).
learning methods across a variety of areas adopted deep neural                       However, there are still open challenges, especially in the
  Corresponding authors: Mahsa Ghorbani, Hamid R. Rabiee                          biomedical datasets including missing values [14], dataset size
Email-addresses: mahsa.ghorbani@tum.de, mahsa.ghorbani@sharif.edu,                [15], and low contrast imaging [16]. Such challenges hinder
rabiee@sharif.edu                                                                 the optimal performance of any model for the classiﬁcation

2                                                                                                                                                                                                                                                                                            2021
task. One of these challenges is also the class imbalance which                                          II.  RELATED WORK
happens when the number of samples across the classes are
disproportional. The class imbalance can vary from a slight to                     Our method is mainly related to GCNs, their importance in
a severe one where the major class contains tens, hundreds,                     disease prediction, and the effect of class imbalance in medical
or thousands of times more samples than the minor one. It is                    datasets on the decision of the classiﬁers. Accordingly, this
common to describe the imbalance of the dataset in terms of                     section is dedicated to reviewing the literature on the related
imbalance ratio (ratio of the major class size to the minor class               areas.
[17]) or summarize the distribution as a percentage of samples                  Graph Convolutional Networks: Due to the success of con-
in each class. Most of the algorithms and loss functions for                    volutional neural networks on common and regular domains
training  a  classiﬁer  aim  to  maximize  accuracy  metric  for                such as computer vision and speech recognition, most of the
training  data,  so  they  have  reasonable  outcomes  when  the                GNNs are devoted tore-deﬁningthe concept of convolution on
classiﬁcation  task  is  deﬁned  on  the  datasets  with  balanced              graphs (see review paper of [19]). Depending on the deﬁnition
classes [17]. However, in an imbalanced setting, maximizing                     of  ﬁltering,  GNNs  are  categorized  into  spectral-based  (as
the accuracy encourages the classiﬁer to favor the major class.                 reviewed in [20], [21], [22], [10]) and spatial-based models (as
Accordingly, a high overall accuracy classiﬁer does not imply                   reviewed in [23]). ChebyNet [22] is among the known spectral
a well-discriminating decision boundary between the major                       methods which developed a deep learning approach to deal
and minor classes in imbalanced datasets.                                       with graph-structured data. Further, [18] proposeda variant
   In this paper, we propose a model that handles the class                     of GCNswhich simpliﬁed the ChebyNet without the need to
imbalance by modifying the objective function and prevents                      perform the convolution in the spectral domain and bridged
bias towards either of the classes. We adopt a well-known                       the gap between spectral and spatial-based approaches [24].
variant of Graph Convolutional Networks (GCNs) proposed                         Many variants of GCNs are proposed which include applying
by  Kipf  et  al.  [18]  which  has  achieved  high  performance                the  attention  mechanism  to  the  graphs  [25],  increasing  the
for the node classiﬁcation task in different applications (See                  receptive ﬁeld of convolutional ﬁlters [26], and improving the
review  paper  [19]).  In  our  proposed  model,  in  addition  to              scalability by sampling techniques [27], [28], [29].
the GCN-based classiﬁer, a class-speciﬁc GCN is used which                      GCNs  in  disease  prediction: In recent years, GCNs have
is  responsible  for  providing  a  weight  distribution  over  the             been adopted in different applications, especially in medical
training samples of the corresponding class. These weights                      domains e.g., brain analysis [30], mammogram analysis [31],
are employed in the weighted cross-entropy as the objective                     and segmentation [32]. The disease prediction problem has
function. Modifying the objective function by giving more or                    been also widely explored by GCN-based methods [12], [33],
less penalty on samples, purposefully biases the classiﬁer to                   [34], [35]. Following the promising results of GCNs in the
provide a more accurate prediction of high weighted samples                     medical domain, [12] exploit GCNs in the disease prediction
and  build  a  discriminating  hidden  space  across  them.  The                problem  with  multi-modal  datasets.  They  chose  ChebyNet
classiﬁer and the weighting networks are trained in an iterative                with constant ﬁlter size as the classiﬁer and investigated the
adversarial process. The classiﬁer minimizes the penalty of                     afﬁnity graph construction by computing the distance between
misclassiﬁcation  and  the  weighting  networks  maximize  the                  the subjects for brain analysis. InceptionGCN [36] extends
weights of the misclassiﬁed nodes against the classiﬁer. The                    ChebyNet and designs ﬁlters with different kernel sizes, and
proposed method is named Re-weighted Adversarial Graph                          chooses the optimal one for the ﬁnal disease prediction. A
Convolutional Network (RA-GCN) which is designed for node                       branch of methods focuses on improving the graph structure.
classiﬁcation task in the imbalanced datasets. Our contribu-                    [37] and [38] start with a pre-constructed graph and update
tions are:                                                                      it during the training, but [39] learn the whole graph from
                                                                                features directly in an end-to-end manner. Besides the methods
   •A novel model to address class imbalance for disease                        that concentrate on the graph structure, a set of methods inves-
      prediction using GCNs.                                                    tigate the missing-value problem as an important challenge of
   •Proposing separate weighting networks that learn a para-                    medical datasets in disease prediction [40], [41]. However, to
      metricfunctionforweightingeachsampleineachclassto                         the best of our knowledge, none of the previous work consider
      helptheclassiﬁerﬁtbetterbetweenclassesandavoidbias                        the class imbalance problem in the graph-based methods for
      towards either of the classes by adopting an adversarial                  disease prediction.
      training approach.                                                        Class imbalance in non-medical and medical applications:
   •Proposing a simple and easy-to-learn model,compared to                      Methods for handling class imbalance for non-graph datasets
      the previous methods, by weighting the existing training                  can be categorized as data-level and cost-sensitive approaches.
      samples instead of generating new samples.                                In  the  data-level  approach,  re-sampling  is  done  to  balance
   •Demonstrating  RA-GCN’s  ability  to  enhance  the  per-                    the classes [42], [43], [44]. There are different forms of re-
      formance of the GNNs with experiments on both real                        sampling  such  as  randomly  or  targeted  under-sampling  the
      medical and designed synthetic datasets which are similar                 major class(es) and oversampling the minor one(s). Even in
      to real ones in the number of samples and features. In                    solving class imbalance for non-graph datasets, oversampling
      this paper, we demonstrate the superiority of RA-GCN                      the minority class(es) causes the overﬁtting to the duplicate
      in  terms  of  performance  and  stability  over  the  recent             samples drawn from the minor class(es). On the other hand,
      methods.                                                                  under-sampling  the  majority  class(es)  causes  the  exclusion

Ghorbani et al.:Accepted for publication in Medical Image Analysis (2021)                                                                                                                                                                        3
of samples that are required for discrimination [45]. These                    two multivariate Gaussian distributions. The goal of the third
weaknesses lead many methods to cost-sensitive approaches.                     component  is  to  minimize  the  Kullback-Leibler  divergence
Modifying the cost function is a common solution for dealing                   between these two distributions. Cross-entropy without any
with the problem [46], [47], [48]. This approach alleviates the                modiﬁcations is used as the main objective function and the
problem by assigning different weights to each class, so the                   second and third components regularize the classiﬁer to avoid
misclassiﬁcation of samples in the minor class(es) penalizes                   overﬁtting to the imbalanced data. Although both the proposed
the  classiﬁer  more  intensely  [49].  The  deﬁnition  of  class              method (RA-GCN) and DR-GCN use adversarial training to
weights is a key point to address in training the classiﬁer [17].              face  the  class  imbalance  issue,  they  have  totally  different
It should be noted that in this approach, all the samples in                   approaches. DR-GCN uses adversarial training between the
each class have the same weight. The idea of re-weighting the                  discriminator and the conditional generator of cGAN along
samples has been also employed in the boosting algorithms                      with the training process of the main classiﬁer. This causes
(e.g., AdaBoost [50]). However, boosting algorithms have a                     the DR-GCN to struggle with the convergence and stability
different approach to learn the classiﬁer. Boosting algorithms                 problems  in  the  training  of  cGAN.  The  problem  becomes
improve the prediction by training a sequence of weak models                   serious  in  the  domains  with  a  limited  number  of  data  for
where each of them compensates for the weaknesses of its                       training, such as medical datasets. However, the proposed RA-
predecessors,  and  their  weighted  linear  combination  forms                GCN utilizes an adversary between the main classiﬁer and
a  strong  classiﬁer.  AdaBoost  is  a  boosting  algorithm  that              the weighting networks with the purpose of learning weights
iteratively identiﬁes misclassiﬁed samples and increases their                 for training samples which is a much simpler and easier task
weights which makes the next classiﬁer put extra attention on                  than generating new samples as in DR–GCN. In addition to
them. On the contrary, RA-GCN learns one strong classiﬁer                      that, DR-GCN forces the separation between the classes in
instead of an ensemble of weak classiﬁers. Furthermore, RA-                    the hidden space of the classiﬁer by adding another cGAN
GCN learns the weighting function based on the classiﬁer’s                     with its own loss function. Nevertheless, RA-GCN trains the
output in an adversarial manner, instead of using a pre-deﬁned                 classiﬁer’s parameters directly by modifying the classiﬁcation
re-weighting  function.  Due  to  the  sensitivity  of  decision-              loss function and penalizing the misclassiﬁcation based on the
making  in  the  medical  domain,  some  methods  exclusively                  sample weights which are learned by weighting networks.
concentrate on the problem of class imbalance in this ﬁeld
[51], [52]. Cancer [53], diabetes [54], Parkinson’s [55], and                                               III.  METHOD
Alzheimer’s diagnosis [56] are some examples that have been                       Graph-based deep learning methods have recently become
studied widely in this domain.                                                 popular because of the novelty and success in different ap-
Class  imbalance  in  graph-structured  data:  The  studies                    plications. One of the important ﬂaws of recently proposed
related  to  the  imbalanced  classiﬁcation  in  graph-structured              GNNs,  is  their  inability  to  deal  with  imbalanced  datasets.
data are limited. One of the reasons is that re-sampling is                    In  this  section,  we  provide  the  details  of  our  method  to
challenging in the context of graphs since each data sample                    dynamically address the class imbalance in GCNs. To this
not  only  contains  features  but  also  has  relations  to  others.          end,  we  propose  a  model  to  simultaneously  learn  graph-
Although  the  generation  of  features  for  a  new  sample  has              based  weighting  networks  to  automatically  weight  the  loss
several approaches, adding a new sample to the graph needs                     value calculated by the classiﬁer’s output for each sample.
to determine its relationwith respect to the rest of the sam-                  In the following, we ﬁrst provide the notation and deﬁnition
ples. Additionally, over-sampling changes the original graph                   of  the  problem,  next  an  overview  of  the  speciﬁc  type  of
structure, which is an inﬂuential factor in the trained model.                 graph-convolutional layer (GC-layer) utilized in our model is
On  the  other  hand,  removing  data  samples  from  a  class                 described. Finally, the proposed method is discussed in detail.
might  affect  a  signiﬁcant  number  of  connections  and  may
create disconnected components. Dual Regularized GCN (DR-                      A.   Problem Deﬁnition and Preliminary
GCN) is a variant of GCNs dedicated to addressing the class
imbalance problem in the GCN architecture [57]. DR-GCN                            AssumethatthegraphG isgivenwithN  nodes,represented
has three components. First one is the GCN-based classiﬁer.                    byG(V,E,X),  whereV  is  the  set  of  nodes  (|V|= N ),
As second component, a conditional Generative Adversarial                      E  is the set of edges, andX∈RN×F  indicates the node
Net (cGAN) including a discriminator and generator is added                    feature matrix.A∈RN×N  is the unweighted and undirected
to the proposed architecture. After the training, the generator                adjacency matrix of the graph.Every node    vi has a corre-
can generate samples with the desired label and the discrim-                   sponding feature vectorxi, a one-hot label vectoryi, and the
inator can discriminate the fake and real samples. Since the                   true class labelci∈C  whereC  is the set of classes. The
low-dimensional representation of nodes from the classiﬁer’s                   label information is available for a subset of nodes and the
hiddenspaceistheinputtothecGAN,theﬁrstcomponentgets                            task is to learn the parametric functionfθ(X,A) which takes
updated during the training of cGAN and the separation be-                     the adjacency matrix and node features as input, and its goal
tween classes will be enhanced in its hidden space. In order to                is to predict the true label of the unlabeled nodes. It should
prevent the network from overﬁtting with the labeled samples,                  be noted that a probabilistic classiﬁer predicts a probability
a distribution alignment between the labeled and unlabeled                     distribution over the|C|classes and the class with maximum
samples is imposed as the third component. For this purpose,                   probability is selected as the label. In our proposed methodqi
it is assumed that the labeled and unlabeled samples followed                  is the output probability distribution of classiﬁer deﬁned over

4                                                                                                                                                                                                                                                                                            2021
|C|classes for the samplexiwherec-th element represents                         for  all  samples  by  dynamically  increasing  the  weights  of
the conﬁdence of the classiﬁer about assigning labelctoxi.                      misclassiﬁed samples against the true-classiﬁed ones. In this
Thus, the problem will be formulated as follow:                                 way, the classiﬁer is forced to reﬁne its decision boundary
                            Q = fθ(X,A),                               (1)      and focus on the misclassiﬁed samples to correct them for
                                                                                both minor and major classes. Fig. 2 depicts an overview of
whereQ∈RN×|C|is the prediction matrix of the classiﬁer                          the proposed model.
for all nodes including unlabeled ones.                                            As it is demonstrated in Fig. 2, RA-GCN consists of two
                                                                                components: 1) A Classiﬁer network, 2) A set of weighting
B. Background: Graph Convolutional Network (GCN)                                networks which are described below.
   In  traditional  neural  networks  (such  as  multi-layer  per-              Classiﬁer network (D):Intheproposedmodel,D isageneric
ceptron  with  fully  connected  layers),  there  is  no  explicit              arbitrary classiﬁer. In our method, we use GCN which takes
relation between the data samples, and they are assumed to                      the  adjacency  matrix  and  node  features  and  its  task  is  to
be independent. GCNs aim to take the neighborhood graph                         predict the class label of the input nodes. As we said before, a
between the samples into consideration and create the feature                   probabilistic classiﬁer predicts a probability distribution over
map of each node not only by its own features but also using                    classes. Fig. 3 represents that the classiﬁer predicts the label
its neighbors [58]. For our method, we employed the deﬁnition                   for  node  (a)  with  high  probability,  but  node  (b)  is  more
ofGC-layerproposedbyKipfetal.[18]intheirmethodwhich                             challenging to decide. Since our classiﬁcation task has a single
is a spectral-rooted spatial convolution network that has shown                 labelpernode,weuseasoftmaxactivationfunctionintheﬁnal
prospering results in the node classiﬁcation task [23] and has                  layer of classiﬁerD.
inﬂuenced many other contributions [24]. The utilized GC-                       Weighting networks (Wcs): In order to learn the weighting
layer ﬁrst aggregates each node’s features and its neighbors                    function for samples of each class, a separate GCN is dedi-
(based on the structure of the graph) and then ﬁnds a latent                    cated to each class to learn the weights for its samples. The
representationforeachnodeusingafullyconnectedlayer(FC-                          weighting network (W-GCN) for classcis denoted byWc.
layer). AssumeA∈RN×N  is the adjacency matrix of the                            EveryWc takes the samples from classcand the adjacency
given graph andX∈RN×F  indicates the features of nodes.                         matrix between them as input and assigns a weight to each
LetDeg be a degree matrix, andIN  be the identity matrix                        sample. It is worth noting that GCN utilizes the neighbors
of sizeN . The graph convolution which we also use in our                       of a node if they exist; otherwise, a node is considered as
method comprises of the following two steps:                                    an isolated one with a self-loop edge. Therefore, if taking
                         −1          −1                                         the class-speciﬁc sub-graphs for weighting networks leads to
   Step1:X′=     ˆDeg       2  ˆA  ˆDeg2X , where  ˆA = A + IN  and             isolated nodes, GCN still works properly. To limit the weights
  ˆDegii= ∑   j ˆAij.                                                           of samples and make the sum of weights in every class in the
   Step2:Z = fθ(X′), wherefθ is an FC-layer with parame-                        same range, we normalize the sample weights of each class
tersθand an arbitrary non-linear activation function.                           using a softmax function.
   In the ﬁrst step, the input graph structure is changed by                    End-to-end   adversarial   training   process:  As  discussed
addingaself-loopforeverynode,andthenthefeaturesofeach                           above, to emphasize the misclassiﬁed samples and the cor-
node are replaced by an average between the node features                       rectly classiﬁed ones but with low conﬁdence, we design an
and  its  neighbors  in  the  graph.  Then,  in  the  second  step,             adversary process between the classiﬁer networkD  and the
the updated features are given to an FC-layer for mapping                       weighting networksWcis. To this end, we adopt the weighted
to  a  latent  space.  In  the  rest  of  the  paper,  ”GCN”  refers            cross-entropy  as  the  classiﬁer  loss  function  and  make  the
to  a  network  composed  of  the  described  GC-layers  except                 weightingnetworkstoprovidethesampleweightsdynamically
otherwise expressed.                                                            for the classiﬁer. We use a two-player min-max game [60] to
                                                                                formulate the adversarial process, where the classiﬁer and the
C. The Proposed Model                                                           set of weighting networks both try to optimize the following
   Weighted cross-entropy loss (LWCE ) is a prevalent objec-                    objective:
tive function for the classiﬁcation of imbalanced data, which                                                    |C|∑|YcL|∑
is deﬁned as follows:                                                                             maxW csminD  −          wiycilog(qci)   ,                    (3)
                                 |C|∑|Yc                                                                       c=1  i=1              
                                       L|∑
                 LWCE  =−                 βcycilog(qci),                (2)                                  Dynamic weighted cross entropy
                                c=1   i=1                                       such thatqiandwiare the outputs of networksD  andWci
whereyciandqciare thec-th element ofyiandqi, respectively,                      respectively,  for  the  samplexi (Wci  is  the  corresponding
andYcL is the set of labeled nodes with labelc.LWCE  needs                      weighting network forxi). This is adversarial training because
predeﬁned weights for samples in every class (βc). This is                      while the classiﬁer minimizes the objective function, W-GCNs
usually done by weighting samples of each class proportional                    have to maximize it. During the training, the classiﬁer tries to
to the inverse of the class frequency [59]. However, the class-                 correctly classify the training samples, especially the samples
weighting method assigns the same weight for the incorrect                      with high weights, to minimize the Eq. 3. Whereas, the ad-
samples as the samples that are already classiﬁed correctly.                    versary makes the weighting networks update their parameters
We  modify  this  approach  by  learning  appropriate  weights                  by putting larger weights on challenging nodes (the samples

Ghorbani et al.:Accepted for publication in Medical Image Analysis (2021)                                                                                                                                                                        5
Fig. 2: The proposed RA-GCN model. The input graph with the corresponding node features is processed by two components.
In the ﬁrst one (the upper part), the classiﬁer, after processing the input data by several layers of GC, maps the input to a
new features space. In the ﬁnal step of classiﬁcation, the classiﬁer predicts the label of each input node which indicates its
conﬁdence about assigning the input node to every class (qis). In the second component (the lower part), a weighting network
composed of multiple GC-layers is utilized for each class to assign appropriate weight to the samples of the corresponding
class so as to improve the classiﬁer’s performance and prevent it from biasing towards either of the classes. The outputs of
the components go to the objective function for updating their parameters. The classiﬁer intends to minimize the objective and
the weighting networks aim to maximize it in an adversary manner.
                                                                                 limited by applying a softmax normalization. Alternatively,
                                                                                 when the provided weights by weighting networks increase
                                                                                 the objective function value, the classiﬁer faces a high penalty.
                                                                                 This makes the classiﬁer update its parameters and try to boost
                                                                                 its performance with respect to the latest generated weights
                                                                                 without  bias  towards  either  of  classes.  Analogous  to  [60],
                                                                                 the parameters ofWcs have to update after the training of
                                                                                 D  is completed, however optimizingD  to full convergence
                                                                                 is highly expensive; therefore, we updateWcsfor one step
                                                                                 after updating the classiﬁerD  fork steps. At the end of the
                                                                                 adversarial game, there will be an equilibrium between the
                                                                                 classiﬁer and the weighting networks. We hypothesize that
                                                                                 the dynamically changing node weights put more attention
                                                                                 on  the  samples  that  might  be  ignored  when  the  classiﬁer
Fig. 3: Conﬁdence of the classiﬁer in the classiﬁcation task.                    is  trained  by  the  conventional  WCE  or  CE  loss  functions.
                                                                                 However,  in  the  real  datasets,  there  are  samples  in  every
Node(a) isan easysample forclassiﬁcationsince itis farfrom                       class that are signiﬁcantly different from other observations.
the decision boundary and the classiﬁer is conﬁdent about its                    In Fig. 4, nodes colored in red are such samples that have the
decision for node (a)                                                            same labels as yellow nodes but lie at an abnormal distance
. Node (b) is more challenging because its features are more                     from other samples. These samples are conventionally called
similar to the nodes with the opposite label (squared) and the                   outliers.Inthepresenceoftheoutliers,theweightingnetworks
classiﬁer assigns it to both classes with close probabilities.                   put  more  weight  on  outliers  to  maximize  the  penalty  for
                                                                                 the classiﬁer. This makes the classiﬁer overﬁt on outliers or
                                                                                 alternate  between  the  correct  classiﬁcation  of  outliers  and
that are not correctly classiﬁed or those on which the classiﬁer                 ﬁnding an appropriate boundary which results in instability.
is not conﬁdent about its decision) to maximize the objective                    Using GCN as weighting network is beneﬁcial to overcome
function. However, as explained before, In order to prevent                      this problem by avoiding sharp weighting distributions. This
the weighting networks from increasing the sample weights                        happens by smoothing the features of each node by its same-
without limitation and do not prioritize any class to another                    labeled neighbors and generating a softened weight distribu-
one, the total summation of sample weights in each class is

6                                                                                                                                                                                                                                                                                            2021
tion  as  a  result.  In  addition  to  that,  we  add  another  term               Algorithm  1: Training Procedure of RA-GCN.k is
and call it ”entropy term”. The entropy term is the sum of                          the number of steps to update the classiﬁer towards
weight distribution entropies. This term acts as a regularizer                      the convergence. We usedk=1    in our experiments to
that  penalizes  such  sharp  weighting  distributions  where  a                    avoid expensive updates.
few samples are assigned a large weight compared to other                            Input: Attributed GraphG, node featuresX ,
samples.α is a free coefﬁcient that determines the importance                                  adjacency matrixA, and labeled nodesYL
of  the  regularizer  term  against  the  weighted  cross-entropy.                   Output: Classiﬁer for node classiﬁcation (D)
By  adding  the  entropy  term,  the  overall  objective  function                   for number of training iterations do
becomes as follow:                                                                        for k steps do
                 |C|∑|YcL|∑                         |C|∑ |YcL|∑                                Get the output of the classiﬁer network (qis for
  maxW csminD  −           wiycilog(qci)   +α(−               wilog(wi))                         training samples);
                 c=1  i=1                           c=1  i=1                                   Get the outputs of weighting networks (wis for
                                                                                         training samples) ;
             Dynamic weighted cross entropy             Entropy term          (4)
                                                                                               Update the parameters of the classiﬁer (θD) by
 Eventually, for the testing phase, we remove the weighting                                      descending their gradient :∑|Yc
                                                                                                         ∇θD−∑|C|c=1           i=1  wiycilog(qci)L|
                                                                                          end
                                                                                          Get the output of the classiﬁer network;
                                                                                          Get the output of weighting networks ;
                                                                                          Update the parameters of each weighting network
                                                                                            (θW c) by ascending the gradient:
                                                                                               ∇θW  c−∑|YcL|i=1  wiycilog(qci)+ α(−∑|YcL|i=1 wilog(wi))
                                                                                     end
                                                                                  A. Graph Construction
                                                                                     For graph construction, one or multiple features from the
                                                                                  original datasets are chosen (calledXadj). We compute the
                                                                                  distance between every pair of nodes according toXadjand
                                                                                  connect  the  nodes  within  a  distance  less  than  a  threshold
Fig.  4:  Outliers  in  the  dataset.  Circles  (blue  samples)  are              γ.γ is the hyper-parameter chosen empirically (e.g., cross-
the samples from the ﬁrst class and squares (yellow and red                       validation)  for  each  dataset.  Mathematically,  the  adjacency
samples) are the samples from the second class. Samples in                        matrix between samples (nodes) is deﬁned as follows:
                                                                                                           {
red have an abnormal distance from other points with the same                                      aij=      1,   ifdist(xadji ,xadjj  )<γ
label (yellow ones). These points are called outliers. Outliers                                              0,   otherwise                                      (5)
can be the result of many circumstances such as an error in                       whereaijis the element in thei-th row andj-th column
experiments, data collection, or labeling.
                                                                                  of adjacency matrix (A) andxadji    represents the adjacency
networksWcs and obtain an efﬁcient classiﬁer for the test set.                    feature(s) of nodei. It should be noted that the afﬁnity graph
Algorithm 1 describes the training steps of RA-GCN.                               is constructed between all the pairs of nodes (train, validation,
                                                                                  and test) as we follow the transductive setting. After graph
                                                                                  construction,Xadjwill be excluded from the input features to
                 IV.  EXPERIMENTS AND RESULTS                                     avoid redundancy.
   To  evaluate  our  model’s  performance  as  well  as  other                   B. Method Comparisons
existing  methods,  we  construct  the  population  graph  based
on a subset of features in the original imbalanced datasets.                         One of the most effective methods for dealing with the class
Unlike [39], [12], ﬁnding the best way for graph construction                     imbalance problem is the weighted cross-entropy (Eq.2), and
is not the focus of this paper. Therefore, we follow the similar                  the weights are given by:
steps of [12] for graph construction, explained in the next
subsection. Moreover, to keep the setting simple and avoid                                                  βc=1− |YcL|∑|C|
the effect of edge weights on the classiﬁcation task, a binary                                                              i=1|YiL|,                          (6)
graph is constructed between samples.                                             where|YcL|is the number of nodes in the training set with
   Our experiments are divided into two sets of studies on real                   labelc. The intuition behind this approach of weighting is to
and synthetic datasets.For each one, we introducethe datasets,                    make the sum of the sample weights of all classes equal.
the corresponding task, the experimental setup, and ﬁnally, the                      We compare the results of our proposed method with multi-
results and discussion of the experiments are provided.                           layer perceptron (MLP) as a non-graph feature-based model

Ghorbani et al.:Accepted for publication in Medical Image Analysis (2021)                                                                                                                                                                        7
and the GCN proposed by Kipf et al. [18] as the baselines                        to  cover  different  challenges  in  the  targeted  problem.  An
since they are successful neural networks for the classiﬁcation                  overview of the real datasets is provided in Table I. In the
task [19]. As it is described in Section III-B, GCN utilizes FC-                 following section, we will describe each dataset in detail.
layers in its architecture, but it incorporates the graph between
samples at the start of every layer. Therefore, depending on                     TABLE  I:  Details  about  the  real  datasets.  Table  has  the
the  graph  structure,  the  extra  information  in  the  graph  for             characteristics of the real datasets including the number of
GCN can either improve the class imbalance by enforcing the                      samples, number of features, density of graph, and imbalance
separation between classes, or it can intensify the imbalance                    ratio. All the datasets have two classes, and the imbalance
problem by the domination of major class samples throughout                      ratio is reported as the number of samples in the major class
the  feature  propagation  [57].  Despite  the  strength  of  MLP                divided by the number of samples in the minor one.
and  GCN,  the  class  distribution  is  not  considered  in  their                            No. of    No. of   No. of Features                 Imbalance
                                                                                   Datasets   Samples   Features    for Graph    Adj Density        Ratio
architecture. To deal with this issue in training both methods,
we  use  weighted  cross-entropy  as  loss  function  and  add                     Diabetes    768         7           1           0.086      500 /268 = 1  .866
                                                                                    PPMI       324       300           4           0.0005      249 /75 = 3 .320
”weighted” to their name in experiments. DR-GCN [57] is                           Haberman     306         2           1           0.076       225 /81 = 2 .778
the recent method dedicated to the class imbalance problem
in the GCN. DR-GCN does not follow conventional methods.                            1) Datasets:  Pima Indian Diabetes (Diabetes) [65]: The
It adds regularization terms to enforce the separation between                   dataset is produced by the ”National Institute of Diabetes and
classes in the latent space by including cGAN.                                   Digestive and Kidney Diseases”. The goal of this dataset is
                                                                                 to recognize the diabetic status of patients (binary classiﬁca-
C. Metrics                                                                       tion). Every patient has 7  numeric features which show the
   In the following experiments, accuracy, macro F1, and ROC                     diagnostic measurements of diabetes including the number of
AUC are reported as measurements to assess the compared                          pregnancies, plasma glucose, blood pressure, skin thickness,
methods. The value of all the metrics is in the range [0,1],                     Body  Mass  Index  (BMI),  insulin  level,  diabetes  pedigree
and the higher values show better classiﬁcation performance.                     function, and age. We use plasma glucose of patients for graph
Accuracy  is  one  of  the  most  frequently  used  metrics  in                  construction (withγ=4    in Eq. 5) and the rest measurements
classiﬁcation. High accuracy is achievable in an imbalanced                      as node features.
dataset by a biased model towards the major class. However,                         Parkinson’s Progression Markers Initiative (PPMI) [66]:
along with other metrics, accuracy can reﬂect the effect of                      This dataset is about detecting Parkinson’s disease vs. normal
the class imbalance problem on the classiﬁers’ performance                       samples (binary classiﬁcation). PPMI dataset contains brain
[61]. For measuring accuracy, all samples, without considering                   MRI and non-imaging information. Non-imaging information
their class, contribute equally to compute the metric. To reﬂect                 includes Uniﬁed Parkinson’s Disease Rating Scale (UPDRS is
the classiﬁers’ performance on the minor classes as well as                      a numeric rating used to gauge the severity and progression
the major classes, we add ROC AUC and macro F1. Since                            ofParkinson’sdisease),MontrealCognitiveAssessmentscores
in their calculation, each metric is computed independently                      (MoCA is a cognitive screening test for detecting brain dis-
for  each  class  and  then  averaged  (by  treating  all  classes               eases with numeric scores), and demographic information (age
equally), they are good measures to be studied besides the                       and gender). The MRI images are processed and then are used
other metrics when the class imbalance exists in the dataset.                    asthefeaturesofsamples.A3D-autoencoderisusedtoencode
ROCAUCisasingle-valuedscore widelyusedinthepresence                              raw image intensities, and the encoded representation is used
of class imbalance [62], but it should be noted that it can                      as the input features. More details about this process are de-
be sub-optimal in case of the low sample size of minority                        scribed in [68] and [69]. Non-imaging features are utilized for
instances [63]. Therefore, to thoroughly analyze the models’                     graph construction (patients with equal test scores in UPDRS
performance, macro F1 is also utilized as a preferable and                       andMOCAwhohavethesamegenderandhaveagedifference
reliable metric. F1 score is the weighted average of precision                   less  than  2  are  connected).  The  encoded  representation  by
and recall, and both false positives and false negatives are                     trained3D-autoencoderwithimagingfeaturesareusedasnode
considered in its calculation. When false positive and false                     features.
negative have similar costs, accuracy is a better measurement,                      Haberman’s  survival  (Haberman)  [67]: This dataset is
but in uneven class distribution, where the cost of classes are                  the output of the study about patients’ survival status who
different, we should look at both precision and recall [64]. The                 had  undergone  surgery  for  breast  cancer,  which  was  done
details about the deﬁnition of the metrics are provided in A.                    between 1958 and 1970 at the University of Chicago’s Billings
                                                                                 Hospital. The dataset has two classes for predicting the sur-
                                                                                 vival status of patients. The dataset contains numeric features
D. Experiments on Real Datasets                                                  including the number of auxiliary nodes (Lymph Nodes), age,
   In this section, we evaluate the proposed RA-GCN on three                     and operation year. The number of auxiliary nodes is used for
real datasets. The datasets are Pima Indian Diabetes (Diabetes)                  graph construction (withγ=2  ) and the rest features are node
[65], Parkinson’s Progression Markers Initiative (PPMI) [66],                    features.
and Haberman’s survival (Haberman) [67]. All the datasets                           2) Implementation  Details:  In  all  experiments,  for  RA-
are  medical  datasets,  and  they  are  different  in  the  number              GCN and also the rest of the methods, the hyper-parameters
of samples, the number of features, and the imbalance ratio                      are chosen based on their best performance on the validation

8                                                                                                                                                                                                                                                                                            2021
set. In RA-GCN, for the classiﬁer, we chose a two-layer GCN                      bias with unweighted loss function is towards the major class.
(one hidden layer and one output layer) with 4 hidden units for                  It can be seen from Table III that the results of the DR-GCN
Diabetes and PPMI and 2 hidden units for Haberman datasets.                      are also interesting. Although it improves the performance of
To simplify our model, we use the same structure and setting                     the GCN, it seems that it is still stuck in the trap of class
for the class-speciﬁc weighting networks. We adopted a two-                      imbalance due to the low value of macro F1. The proposed
layer GCN with two hidden units for the weighting networks                       RA-GCN  performs  the  best  in  improving  the  challenge  of
in the following experiments. Rectiﬁed Linear Unit (ReLU)                        imbalance.
[70] is used as the activation function of the hidden layers. We                    In terms of the imbalance ratio and the size of the dataset,
have also applied a dropout layer with a dropping rate of 0.5 to                 the Haberman dataset is similar to the PPMI, but its number
avoid overﬁtting. The entropy term’s coefﬁcientα is set to 0.5,                  of input features is much lower. By comparing the weighted
0.1,1forDiabetes,PPMI,andHabermandatasetsrespectively.                           versionofMLPandGCNwithunweightedonesinthisdataset,
In order to train the networks regarding the objective function                  it  can  be  concluded  that  the  effect  of  weighting  is  more
in Eq. 4, stochastic gradient descent using Adam optimizer                       substantialinthiscase.Thegraphinformationisimprovingthe
[71] is used with the learning rate of 0.001 for the classiﬁer                   results, but its effect is more limited than other datasets. The
D  and 0.01 for the weighting networks. We use 60%   , 20%   ,                   results of the DR-GCN are even worse than weighted GCN.
and 20%     for train, validation, and test split, respectively in all           The limited number of samples and features besides the high
datasets and the imbalance ratio is kept in the splits. Inspired                 imbalance ratio might cause poor performance since it is hard
by  Parisot  et  al.  [12],  the  absolute  difference  between  the             to train a generative model with these limitations. RA-GCN
features is used as a distance for graph construction, and the                   performs the best in this case, especially in improving the
graphsaresimple.AlltheimplementationsareinPyTorch[72]                            macro F1.
and the results of competitors are obtained with the authors’                       From the experiments, it can be concluded that: 1) None of
source code. For all methods, the results are reported on the                    the metrics is sufﬁcient to judge a model and totally reﬂect the
test set using the best conﬁguration (selected based on the                      performanceoftheminorclassagainstthemajorone.2)Using
macro F1 value on validation set) for each method per dataset.                   GC-layers  instead  of  FC-layers  for  feature  propagation  is
The reported results are the mean and standard deviation of                      helpful. 3) In highly imbalanced datasets, weighting deﬁnitely
all the metrics for each method on 5  different random splits                    helps the performance, and learning weights enhances the ﬁnal
of data.                                                                         classiﬁer.
   3) Results and Discussion: Accuracy, macro F1, and ROC                        TABLE II: Results of RA-GCN and compared methods on
AUContherealdatasetsareprovidedinTables II,III,andIV.                            Diabetes dataset.
The boxplots of these results are also provided in Fig. 5 for
visual comparison. We compare RA-GCN with MLP, GCN,                                     Method            Accuracy          Macro F1            ROC AUC
their weighted versions, and one recent method (DR-GCN).                            MLP-unweighted      0.58±0.079        0.43±0.028          0.55±0.117
                                                                                     MLP-weighted       0.62±0.045        0.60±0.046          0.68±0.044
   For  all  datasets,  MLP  weighted  or  unweighted  perform                      GCN-unweighted      0.70±0.107        0.65±0.097          0.72±0.195
worse than most of the graph-based methods. Although GCN                             GCN-weighted       0.71±0.051        0.64±0.143          0.66±0.272
                                                                                        DR-GCN          0.71±0.021        0.66±0.049          0.75±0.041
performs better than MLP, it has a high variance in all three                        RA-GCN (ours)     0.74±0.026         0.73±0.021          0.80±0.014
datasets.  The  performance  of  DR-GCN  varies  for  different
datasets. It shows high stability and good performance in the                    TABLE III: Results of RA-GCN and compared methods on
ﬁrst two datasets, but its efﬁciency drops on the Haberman. As                   PPMI dataset.
can be seen, our proposed method, RA-GCN, performs better
than all the other methods for all datasets, especially in macro                        Method            Accuracy          Macro F1            ROC AUC
F1 that indicates the classiﬁer’s performance for both minor                        MLP-unweighted      0.74±0.062        0.46±0.058          0.56±0.067
                                                                                     MLP-weighted       0.60±0.098        0.49±0.074          0.45±0.056
and major classes. The effect of class imbalance is evident in                      GCN-unweighted      0.68±0.101        0.48±0.050          0.55±0.078
all the datasets.                                                                    GCN-weighted       0.64±0.119        0.51±0.087          0.50±0.052
                                                                                        DR-GCN         0.76±0.016         0.56±0.037          0.60±0.055
   Diabetes dataset is the best-chosen dataset in terms of the                       RA-GCN (ours)      0.71±0.109        0.58±0.096          0.56±0.045
imbalance ratio and dataset size. It is obvious that utilizing
the graph makes a signiﬁcant improvement in this dataset.                        TABLE IV: Results of RA-GCN and compared methods on
The effect of weighting in this dataset for MLP is substantial,                  Haberman dataset.
but its effect for GCN is not considerable; while, the problem
of class imbalance still exists for both. RA-GCN is noticeably                          Method            Accuracy          Macro F1            ROC AUC
the best and most reliable method on this dataset and has an                        MLP-unweighted      0.71±0.033        0.46±0.041          0.45±0.177
                                                                                     MLP-weighted       0.71±0.046        0.56±0.114          0.65±0.113
improvement in all metrics.                                                         GCN-unweighted      0.74±0.020        0.47±0.055          0.55±0.060
   PPMI dataset is the most challenging one due to high di-                          GCN-weighted       0.71±0.032        0.57±0.058          0.46±0.095
                                                                                        DR-GCN          0.58±0.178        0.44±0.134          0.56±0.063
mensional input features, a low number of samples, and a high                        RA-GCN (ours)     0.75±0.065         0.65±0.081          0.61±0.193
imbalance ratio. Given such a ratio, all the metrics should be
considered for judging a classiﬁer’s performance. Due to the                     E. Experiments on Synthetic Datasets
high imbalance ratio, the unweighted methods achieve higher
accuracy and ROC AUC than weighted versions. However, the                           In this section, we investigate the effect of different factors
low value of macro F1 is a witness that the trained classiﬁer’s                  on the proposed model’s performance on synthetic datasets.

Ghorbani et al.:Accepted for publication in Medical Image Analysis (2021)                                                                                                                                                                        9
        (a) Box plots of Accuracy, Macro F1, and ROC AUC of the compared methods on Pima Indian Diabetes (Diabetes) dataset
(b) Box plots of Accuracy, Macro F1, and ROC AUC of the compared methods on Parkinson’s Progression Markers Initiative (PPMI) dataset
        (c) Box plots of Accuracy, Macro F1, and ROC AUC of the compared methods on Haberman’s survival (Haberman) dataset
Fig. 5: Box plot of results on real datasets. Each row corresponds to the results of methods on a dataset which from left to
right includes accuracy, macro F1, and ROC AUC. From top to bottom, rows are results of methods on Diabetes, PPMI, and
Haberman datasets. Each boxplot summarizes the values of a metric on 5  different random splits of the dataset indicating how
the values are spread out.
We generate the synthetic datasets to ﬁrst examine the effect                      instead of graph construction in pre-processing step.
of imbalance ratio on the performance of the proposed model                           For generating synthetic samples, we use the scikit-learn
in Section IV-E2, then, in Section IV-E3, an ablation study is                     library in Python [73]. The algorithm of data generation is
performed to evaluate how different variants of the weighting                      adopted from [74] which is based on creating cluster points
networks in the proposed model inﬂuence the results, after-                        normally  distributed  around  vertices  on  a  hypercube.  The
ward, the sensitivity of the model to its parameterα in the                        algorithm  introduces  interdependence  between  the  features
objective function is examined in Section IV-E4. In the ﬁnal                       and also is able to add various types of noise to the data.
experiment,  the  effect  of  graph  structure  on  the  results  of               The library also supports data generation with an imbalanced
all compared methods is assessed by changing the sparsity                          setting. In order to build a relevant graph with correlations
of constructed graphs in Section IV-E5. In addition to these                       to the output label, we generate a dataset with 2F  features
experiments,  we  provide  a  qualitative  study  to  investigate                  (whereF  is a hyper-parameter) and use the ﬁrst half of the
the problem by visualization in Appendix A. We have also                           features for the graph construction and the second half as the
explored  the  class  imbalance  issue  in  a  more  challenging                   node features. The graph with node features provides us an
situation for multi-class imbalanced data in Appendix B. The                       appropriate graph-based dataset with the desired imbalance
last experiment in Appendix C is an study about the compared                       ratio.
methods’  performance  when  another  component  for  graph-                          1) Implementation Details:  We follow the same network
construction  is  added  to  their  structure  to  learn  the  graph               architectures  for  the  RA-GCN  as  the  real  datasets  as  well

10                                                                                                                                                                                                                                                                                          2021
as training settings and parameters. The architecture of the                        The results of the second experiment are provided in Fig. 7.
classiﬁer for all methods is identical. The classiﬁer’s dropout                  These datasets are more challenging than the previous ones.
and learning rate are common hyper-parameters between all                        Although the imbalance ratio is changing slowly, changes in
methods and are tuned for each method and dataset separately                     the  results  are  much  dramatic.  Once  again,  the  difference
based  on  the  validation  results. 60%      of  the  data  is  used            between the results of weighted methods and unweighted ones
for  training, 20%     for  validation,  and 20%     for  testing.  The          proves that class imbalance can have a huge effect on the
imbalance ratio is kept the same in the splits. We chose the                     classiﬁcation results. For higher imbalance ratios, unweighted
best classiﬁer of each method based on its macro F1 value on                     methods  are  biased  towards  the  major  class,  and  although
the validation set and reported its results on the test set. Same                they have high accuracy, they perform poorly based on the
as Section IV-D, we follow the DR-GCN paper to set up its                        other measures. Interestingly, in the case of GCN with static
parameters. The reported results are the mean and standard                       weighting, the accuracy and ROC AUC are not high. This
deviation of all the metrics for each method on 5  different                     means  that  the  learned  classiﬁer  chooses  to  ignore  many
random  splits  of  data.  In  the  following  experiments,  we                  samples in the major class against the correct classiﬁcation of
generate 1000    nodes with 10   features for graph construction                 a few samples in the minor one resulting in a high macro F1.
and 10  features as node features. The distance between nodes                    The instability of DR-GCN is more serious in this experiment.
in Eq. 5 is cosine distance.                                                     Although  it  improves  the  results  of  the  unweighted  GCN
   2) Effect of Imbalance Ratio:  In this experiment, we test                    with a clear margin, its results are competitive with weighted
the performance of RA-GCN w.r.t imbalance ratio on a set of                      GCN. Moreover, for highly imbalanced datasets, the results
synthetic datasets consisting of balanced, low imbalanced, and                   of weighted MLP and weighted GCN are competitive, and for
highlyimbalanceddata.Inthissection,weusetheconnectivity                          high imbalance ratios, MLP has better performance, so the
threshold (γ) equal to 0.5.                                                      graph is not helpful in these cases. We can say that in the
We have two experiments that are both done on binary-class                       case of GCNs, as the size of the major class grows and the
datasets. In the ﬁrst experiment, the ratio of the data in the                   minor one shrinks, the minor class samples, due to their small
major class goes from 50%     to 80%   , and in the second one,                  number, do not have many co-labeled neighbors in the graph.
it goes from 85%     to 95%     at a more granular level. The ﬁrst               Therefore, they are more affected by the major class samples,
experiment’s goal is to investigate theeffect of class imbalance                 which  causes  a  drop  in  weighted  GCN  performance.  This
by varying the dataset from a balanced dataset to a highly                       observation is also validated by Shi et al. [57]. RA-GCN deals
imbalanced  one.  In  the  second  one,  we  want  to  study  the                with the problem by putting more weights on the neglected
effect of varying imbalance ratios at a higher granular level in                 nodes and highlighting them for the classiﬁer to focus more
the highly imbalanced datasets. The number of samples in all                     on the difference between classes instead of just the number
the datasets is 1000   . Appendix Table VI contains information                  of correctly classiﬁed samples.RA-GCN shows a much higher
about the imbalanced ratio and the adjacency matrix in these                     macro F1 and a competitive ROC AUC. In terms of accuracy,
synthetic datasets.                                                              it beats the other ones when the percentage of the majority
   Results:  The  ﬁrst  experimental  results  are  provided  in                 class varies from 85%     to 90%   . The results demonstrate the
Fig.  6.  When  the  dataset  is  completely  balanced,  there  is               superiority and robustness of RA-GCN in severely imbalanced
no  difference  between  weighted  and  unweighted  methods.                     datasets.
However, RA-GCN has a minor improvement, which implies
that  weighting  (independent  of  the  class  imbalance  issue)                    3) Effect of Modiﬁcation in The Weighting Networks:  In
can help the neural network to ﬁnd a better low-dimensional                      this section, we investigate the model in detail for a better
space  in  which  classes  are  more  separable  than  the  other                understanding of weighting networks’ role in the proposed
methods. In severe situations, the performance of all methods,                   model.  For  this  purpose,  two  synthetic  datasets  containing
including  the  proposed  method,  drops.  From  the  results  in                1000     nodes with a moderate imbalance ratio (75%     of data
Fig. 6, we can conclude that employing the graph results in                      in the major class) and a high imbalance ratio (95%     of data
a better performance. By increasing the imbalance ratio, the                     in  the  major  class)  are  generated.  The  threshold  for  graph
accuracy of unweighted MLP increases, but macro F1 drops                         construction is 0.5  to follow the same steps as the previous
signiﬁcantly. One interesting point is that unweighted GCN                       experiment. It should be noted the results are provided with
keeps its performance even when the unweighted MLP starts                        a  similar  structure  for  the  weighting  networks  in  all  the
to drop, which shows that utilizing graphs is helpful to deal                    compared variants. The compared variations of the RA-GCN
with the class imbalance in these datasets. Weighting is an                      are as follows.
improvement for both GCN and MLP architectures, especially                       No weighting networks: In this case, the weighting network
when  the  imbalance  ratio  increases;  however,  it  should  be                is discarded from model and only the classiﬁer is trained.
noted  that  the  improvement  is  minor  for  GCN.  DR-GCN                      Class-weighting networks: This variant is the class-weighted
instability can also be concluded from Fig. 6. Although this                     version of the proposed RA-GCN. It has the same structure
methodismorecomplicatedthanmerelyweightingtheclasses,                            as  RA-GCN,  but  instead  of  the  softmax  function,  we  ﬁrst
it ends up with high variance results, which might be due to                     calculate the output of weighting networks for all samples,
the convergence problem in the GAN-based methods. For all                        then apply an average function on the samples of each class
imbalanced datasets, the RA-GCN is better in all measures. It                    to get class weights. In the last step, we normalize the class
has a stable performance by changing the imbalance ratio.                        weights and use them in the WCE loss function.

Ghorbani et al.:Accepted for publication in Medical Image Analysis (2021)                                                                                                                                                                      11
                        (a) Accuracy                                             (b) Macro F1                                   (c) ROC AUC
Fig. 6: The effect of changing the imbalance ratio on the performance. The ﬁgure shows the results of the compared methods.
In each ﬁgure, by varying the imbalance ratio, the datasets change from balanced dataset to low imbalanced, and highly
imbalanced ones.
                        (a) Accuracy                                             (b) Macro F1                                    (c) ROC AUC
Fig. 7: The effect of changing the imbalance ratio on the performance at a granular level. The ﬁgure illustrates the results of
the compared methods. All the datasets are highly imbalanced.
One  sample-weighting  network  for  all  samples:  In  this                     network is a difﬁcult task and needs a network with more
variant of RA-GCN, instead of having a separate weighting                        parameters  that  might  overﬁt  due  to  the  few  number  of
network for each class, one sample-weighting network with                        samples. On the other hand, although sharing the parameters
the same structure is supposed to learn the weights for all                      between the networks in the moderate imbalanced dataset has
training samples.                                                                better results than one weighting network for all samples, it
Separate sample-weighting network per class with shared                          is more unstable with lower performance in a high imbalance
parameters:  In  this  form  of  model,  like  the  original  RA-                setting.Whentheparametersaresharedbetweentheweighting
GCN, one weighting network is responsible for the weights                        networks, they use the same low-dimensional representation
of samples in each class, but all the network parameters are                     (hidden  space  of  the  network)  to  ﬁt  a  weight  distribution
shared between W-GCNs except the parameters of the last                          over classes. This can be a limitation when the best weight
layer.                                                                           distributions  over  classes  need  more  than  one  layer  (the
                                                                                 last  one  layer  which  is  not  shared  between  classes)  to  be
   Results: Table V shows the results of different models on                     learned. Therefore, dedicating a separate weighting network
both moderate (75%     of data in the major class) and highly                    to each class makes the problem simpler for each network
imbalanced  (95%      of  data  in  the  major  class)  datasets.  In            and increases the ability of the model to learn complicated
both datasets, the imbalance issue is apparent in the results                    weight distributions.
of the classiﬁer without weighting networks. The results are
competitive  on  the  moderate  imbalanced  dataset  (right-side
of  Table  V)  and  the  main  difference  between  the  models                     4) Parameter Sensitivity:  RA-GCN algorithm involves the
shows up in the highly imbalanced setting. By comparing the                      parameterα as a coefﬁcient of a regularization term added to
method for class-weighting (the second row) against sample-                      the objective function. In order to evaluate how changes to the
weighting  methods,  it  can  be  concluded  that  the  sample-                  parameterα affect the performance on node classiﬁcation, we
weighting  is  superior  approach  over  the  class-weighting  in                conduct the experiments on two synthetic datasets. Likewise
terms of accuracy, macro F1 and ROC AUC, especially for                          the previous experiment, we use two datasets of 1000    nodes
highly imbalanced datasets.                                                      with 95%     and 75%     of nodes in the major class. For this ex-
From the results we can infer that ﬁnding appropriate weights                    periment, we ﬁxed the structure of the classiﬁer and weighting
for  all  samples  from  different  classes  with  one  weighting                networks and changed the coefﬁcient of entropy term in Eq. 4.

12                                                                                                                                                                                                                                                                                          2021
TABLE V: Ablation study for RA-GCN. The table shows the results of different variants of RA-GCN with respect to the
design of weighting networks and the effect of each variation on the performance.
            Class distribution                               [0.95-0.05]                                                    [0.75-0.25]
             Method/Metric              Accuracy            Macro F1              ROC AUC              Accuracy            Macro F1              ROC AUC
         No weighting networks      0.952 ±0.0012        0.524 ±0.0226         0.403 ±0.0225        0.914 ±0.0037       0.872 ±0.0055         0.964 ±0.0025
           (GCN-unweighted)
        Class-weighting networks     0.874 ±0.0954       0.616 ±0.0903         0.74±0.0713          0.913 ±0.0227       0.884 ±0.0325         0.943 ±0.0326
      One sample-weighting network   0.918 ±0.0273       0.698 ±0.0731         0.864 ±0.0665        0.913 ±0.0284       0.887 ±0.0375         0.956 ±0.0266
             for all samples
     Separate sample-weighting network0.933 ±0.0301      0.695 ±0.1380         0.783 ±0.2169        0.921 ±0.0201       0.895 ±0.0299         0.956 ±0.0298
      per class with shared parameters
     Separate sample-weighting network0.934 ±0.0102     0.710 ±0.0366         0.871 ±0.0451        0.927 ±0.0163       0.905 ±0.0224         0.960 ±0.0188
        per class (proposed model)
   Results: Fig. 8 shows the effect of increasing the coefﬁcient                   propagation through the connections between samples from
of entropy term on the results. In the moderate imbalanced                         differentclasses.Especiallywhenthenumberofsamplesinthe
dataset (the blue line), the performance of RA-GCN across                          minor one is low, the inter-class connectivities become more
different values ofα is more stable than the other dataset.                        dominant. Further, as the graph convolution takes an average
From this observation, we can conclude that a high entropy                         between the node features and its neighbors, even two GC
weight distribution can also work out for this type of dataset.                    layered networks may lead to the smoothed features.
This conﬁrms the results from the previous experiment saying                          The second experiment’sresults, withthe moderatelyimbal-
that learning the class-weights instead of sample-weights has                      anced dataset, are depicted in Fig. 10. This study also implies
more acceptable results in the moderate imbalanced dataset                         that tuning the graph threshold can have a high impact even
than the highly imbalanced one. Moreover, when we discard                          when the dataset is not highly imbalanced. The best results of
the entropy term from the objective function (α= 0  ), the re-                     the classiﬁer belong to the RA-GCN when the graph threshold
sults on the moderate imbalanced data get unstable with higher                     issetto0.4.TheweightedandunweightedGCNperformances
variance than for the other values ofα. On the other hand, in                      start  to  drop  dramatically  forγ> 0.3  when  encountering
the highly imbalanced dataset (orange line), the coefﬁcient of                     a highly imbalanced dataset (previous experiment); however,
entropy is more effective. In this dataset, when theα is zero                      they  are  more  stable  on  the  moderate  imbalanced  datasets
or close to zero (α=0 .001  ), the performance drops, because,                     (current experiment). This conﬁrms the importance of graph
with  a  small  number  of  samples  in  the  minor  class,  the                   construction alongside the class imbalance issue for the node
weighting network is more likely to assign high weights to a                       classiﬁcation task.
few number of samples. The experimental results demonstrate
that the entropy term is more effective for the learning process                                                V.  CONCLUSION
of sample-weighting when the class imbalance issue gets more                          In  this  paper,  we  proposed  RA-GCN,  a  new  model  to
severe.                                                                            tackle the class imbalance problem by dynamically learning
   5) Effect of Graph Sparsity: Although this paper’s primary                      the cross-entropy sample weights besides the GNN-based clas-
focus  is  to  deal  with  the  challenge  of  class  imbalance  by                siﬁer using adversarial training. We studied the behavior of the
learning the weights automatically, we cannot avoid the fact                       proposed model on various input data compared to baselines
that GCNs are sensitive to the graph structure [36], [28]. In                      and the one recent method [57] in this ﬁeld. We tested the
this section, we study the effect of the graph construction on                     methodonthreerealandasetofsyntheticdatasets.Ourresults
the imbalanced classiﬁcation problem. For this purpose, we                         indicate that the proposed model enhances the classiﬁer based
generate two sets of datasets. In each set, the imbalance ratio                    on the adopted metrics and outperforms the static weighting
is ﬁxed. Following the graph construction from Section IV-A,                       of samples with margin. To investigate the impact of different
we vary the thresholdγ for graph construction from 0.1  to                         factorsontheclassimbalanceproblem,wegeneratedsynthetic
0.9. When the threshold is 0.1, the graph is sparse (refer to                      datasets with different imbalance ratios, different sparsity for
Eq.5), and when the threshold is 0.9, the graph is becoming                        the population graph, and a different number of classes. Syn-
denser, and even samples from different classes are connected.                     thetic experiments show that although the competitive meth-
All the datasets contain 1000    samples. In the ﬁrst and second                   ods  have  acceptable  performance  on  moderate  imbalanced
sets of the generated datasets, the percentage of data in the                      datasets, their efﬁciency drops in highly imbalanced ones. It is
major class is 95   and 75 , respectively. The information about                   also evident that in GCN-based methods, the graph structure
the sparsity of graphs is provided in Table VII.                                   is crucial due to feature propagation throughout the networks.
   Results: The results of the highly imbalanced setting are                       The detailed beneﬁts of the proposed method in dealing with
depicted in Fig. 9. The results of weighted and unweighted                         graph-based class imbalanced datasets are demonstrated in the
MLP are constant as they are not dependent on the graph.                           quantitative and qualitative experimental results.
The results frequently illustrate that as the threshold increases                     Although  in  this  paper,  we  aimed  at  handling  the  class
and the graph becomes dense, the performance of the graph-                         imbalance issue for the graph-based data, a similar approach
based classiﬁers drops. This reveals that tuning the threshold                     can be adopted to tackle the imbalance problem in the non-
for  graph  construction  is  essential.  This  experiment  once                   graph datasets. For this purpose, an appropriate neural network
again  conﬁrms  that  in  the  case  of  an  imbalanced  dataset,                  architecture has to be used as the classiﬁer and the weighting
a  dense  graph  might  corrupt  the  performance  by  feature                     networks(e.g.,convolutionalneuralnetworkforimagingdata).

Ghorbani et al.:Accepted for publication in Medical Image Analysis (2021)                                                                                                                                                                      13
                            (a) Accuracy                                        (b) Macro F1                                 (c) ROC AUC
Fig. 8:Performance evaluation of RA-GCN on varying amount of α. The inﬂuence of the entropy term in the objective function
is controlled byα. Adjustingα is more effective when the imbalance ratio increases.
                        (a) Accuracy                                               (b) Macro F1                                    (c) ROC AUC
Fig. 9: The effect of changing the threshold for graph construction on the performance when 95%     of data is in the major class.
The ﬁgure depicts the performance of the compared methods. In each ﬁgure, the threshold for connecting nodes changes from
0.1  to 0.9. Increasing the threshold makes the distant nodes to be connected in the graph.
                        (a) Accuracy                                               (b) Macro F1                                    (c) ROC AUC
Fig. 10: The effect of changing the threshold for graph construction on the performance when 75%     of data is in the major
class. The ﬁgure depicts the performance of the compared methods. In each ﬁgure, the threshold for connecting nodes changes
from 0.1  to 0.9. Increasing the threshold makes the distant nodes to be connected in the graphs.
Moreover, we should note that in the case of a large number                       label classiﬁcation problem where each data can have more
of classes in which training the weighting networks is com-                       than one label in the dataset is also challenging.
putationally inefﬁcient, a general weighting network instead
of class-speciﬁc ones could present a scalable solution. For                                                       APPENDIX
future research studies, we point out two possible directions.                        The utilized metrics in the experiments are deﬁned based
The proposed model is constructed based on the transductive                       on the following four measurements:
setting.  It  would  be  interesting  to  investigate  the  models’                   True-positive (tpc): The number of nodes whose true label
performance in inductive settings where one has no access                         iscand the predicted class by the classiﬁer is alsoc.
to testing node features during the training. Additionally, the                       True-negative (tnc): The number of nodes whose true label
way one can generalize the proposed RA-GCN to a multi-                            is notcand the predicted class by the classiﬁer is also notc.

14                                                                                                                                                                                                                                                                                          2021
   False-positive (fpc): The number of nodes whose true label                       TABLE  VII:  Setting  of  the  synthetic  datasets.  The  table
is notcbut the predicted class by the classiﬁer isc.                                contains the imbalance ratio and the density of the generated
   False-negative  (fnc):  The  number  of  nodes  whose  true                      adjacency matrix for evaluating the performance of RA-GCN
label iscbut the predicted class by the classiﬁer is notc.                          with changing the threshold for connecting the nodes in the
   The following metrics are reported in this section.                              graphconstructionstep(dataofexperimentsinSectionIV-E5).
   Accuracy: Accuracy is the ratio of nodes that are classiﬁed                                                                 Adj Density
correctly, without considering whether that node is a member                                   Threshold (γ)      IR = 750/250 = 3      IR = 950/50 = 19
                                                                                                                (Datasets of Fig. 10)  (Datasets of Fig. 9)
of major classes or minor classes.                                                                  0.1                0.0037                0.0049
                                      ∑|C|                                                          0.2                0.0215                0.0301
                            Acc=          c=1 tpc                                                   0.3                0.0518                0.0727
                                           N                                       (7)              0.4                0.0894                0.1242
                                                                                                    0.5                0.1347                0.1813
   MacroF1:Intheclassiﬁcationproblem,theF1-scorecanbe                                               0.6                0.1899                0.2433
deﬁned for every class. F1-score is the harmonic mean of the                                        0.7                0.2580                0.3117
                                                                                                    0.8                0.3406                0.3886
precision and recall of the classiﬁer performance for that class.                                   0.9                0.4355                0.4735
The precision of classcis the number of correctly identiﬁed
samples of classcdivided by the number of samples predicted
as classc. The recall of classcis the number of correctly                           that how the class imbalance changes the behavior of a linear
identiﬁed samples of classcdivided by the number of samples                         classiﬁer trained by RA-GCN and other competitors. In the
whose true label isc. In binary or multi-class classiﬁcation,                       second experiment, the imbalance issue is studied in datasets
macro averaging means assigning equal weight to all classes                         with 3, 4, and 5  classes with various class distributions, and
and compute an average over their scores. Thus, macro F1 is                         in the last experiment, one of the latest methods for graph
a useful metric when there is a class imbalance issue in the                        learning is employed to study how adding graph learning to
dataset. Macro F1 can be calculated as follow:                                      the classiﬁcation task effects the performance in the presence
                                     |C|∑                                           of class imbalance.
             MacroF1=     1                        2tpc
                                |C|c=1    2tpc+ fpc+ fnc            (8)             A. Qualitative Study
   Area  Under  Receiver  Operating  Characteristic  Curve                             In this section, we investigate the problem by visualization.
(ROC  AUC): An  ROC  curve  plots  true-positive  rate  vs.                         For  this  purpose,  we  generate 1000     samples  distributed  in
false-positive  rate  at  different  classiﬁcation  threshold.ROC                   two classes with a class imbalance ratio of 90:10. Samples
AUC means the area under the ROC curve. The ROC AUC                                 of  the  ﬁrst  class  are  drawn  from  a 4-dimensional  Gaus-
of  a  classiﬁer  is  the  probability  that  a  randomly  selected                 sian distribution with mean 0  and identity covariance matrix
positive sample will be ranked higher than a randomly selected                      (N(0 4×1,I4×4)). On the other hand, the samples from the
negative sample.                                                                    second  class  are  drawn  from  another 4-dimensional  Gaus-
   The details about the synthetic datasets of experiments in                       sian distribution with mean 1  and covariance matrix 0.3I4×4
Section IV-E2 (the effect of imbalance ratio) and experiments                       (N(1 4×1,I4×4)). Out of the four features, two are used for
in Section IV-E5 (the effect of graph sparsity) are provided in                     graphconstruction,andtheothertwoareinputfeatures.The2-
Tables. VI and VII, respectively.                                                   dimensional visualization of the input features is illustrated in
TABLE VI:Setting of the synthetic datasets with moderate                            Fig. 11. In all of the subﬁgures of Fig. 11, each color (purple,
imbalance ratio. The table contains the imbalance ratio and                         yellow)  or  shape  (circle,  square)  represents  the  class.  The
the density of the adjacency matrix generated for evaluating                        orange line shows the respective classiﬁer after training. The
the performance of RA-GCN with changing the imbalance                               table below each subﬁgure reports the performance metrics of
ratios (data of experiments in Section IV-E2).                                      the classiﬁer. To avoid a confusing diagram, we have abstained
                                                                                    from drawing the graph between samples.
         Dataset           Adj Density     Imbalance Ratio       %    of data in       For simplicity, we train each classiﬁer in a linear manner.
                                                                the major class     The  learned  classiﬁer  of  DR-GCN  is  not  provided  here,
                             0.109          500 /500 = 1              50            because the idea of DR-GCN is to apply regularization to
  Moderate imbalanced        0.115         600 /400 = 1  .5           60
      data of Fig. 6         0.126        700 /300 = 2  .333          70            the hidden space. Since the linear classiﬁer does not have any
                             0.144          800 /200 = 4              80            hidden layer, the output of DR-GCN is the same as GCN-
                             0.156        850 /150 = 5  .667          85            unweighted. Fig. 11 depicts the visualization of the trained
                             0.159        860 /140 = 6  .143          86
                             0.161        870 /130 = 6  .692          87            classiﬁers with different methods.
                             0.164        880 /120 = 7  .334          88               Fig.11aindicatesthatasimpleunweightedclassiﬁerignores
    Highly imbalanced        0.166        890 /110 = 8  .091          89
      data of Fig. 7         0.169         900 /100 = 9  .0           90            most of the minor class samples (Yellow). It acquires high
                             0.171        910 /90 = 10  .112          91
                             0.173        920 /80 = 11  .500          92            accuracy and low macro F1, as expected. On the other hand,
                             0.176        930 /70 = 13  .286          93            MLP-weighted (Fig. 11b) ignores lots of samples in the major
                             0.178        940 /60 = 15  .667          94
                             0.181          950 /50 = 19              95            class because one sample in the minor class is weighted 9
                                                                                    times more. On the other hand, it can be seen from Fig. 11c
   In this section, we provide three additional experiments on                      and 11d that multiplying the features by the modiﬁed adja-
synthetic datasets. In the ﬁrst experiment, it has been shown                       cency matrix reduces the variance of features and results in

Ghorbani et al.:Accepted for publication in Medical Image Analysis (2021)                                                                                                                                                                      15
the denser clusters with potential class overlap. Although the                  TABLE VIII: Details about the generated synthetic datasets
samples’ clustering can make them more class separable, the                     for multi-class classiﬁcation problem. The table contains the
inter-class overlap might make the problem of class imbalance                   number of classes, class distributions and the density of the
more serious for the graph-based methods. In Fig. 11c, the                      constructed adjacency matrix.
high density of samples in one class changes the behavior of                               Dataset   Number of     Percentage of Data  Adj Density
the unweighted GCN in comparison to the unweighted MLP.                                                Classes     Spread in Classes
Due to the high imbalance, the model becomes highly biased                                  DS1           3            70, 20, 10         0.0025
                                                                                            DS2           3            50,40,10           0.0020
towards the major class. This bias encourages the classiﬁer to                              DS3           4          60, 25, 10, 5        0.0023
misclassify all the minor class samples, represented by very                                DS4           4          45, 35, 12, 8        0.0018
                                                                                            DS5           4          35, 30, 25, 10       0.0016
similarfeaturesduetolowvarianceinmodiﬁedfeaturesspace.                                      DS6           5         50, 20, 15, 10, 5     0.0015
On  the  other  hand,  in  Fig.  11d,  two  outliers  in  the  minor
class that are away from other samples can deteriorate the
classiﬁer  learned  with  conventional  weighted  cross-entropy.                Table IX, the overall drop in performance across the methods
This is because the classiﬁer prefers to correctly classify the                 proves that an increase in the number of classes makes the
outliers (due to the high weight) instead of many samples in                    classiﬁcation task more difﬁcult. When there are three classes,
the other class. Hence, outliers (noise or mislabeled data) have                the  results  on  DS1  and  DS2  are  consistent,  although  the
a signiﬁcant effect. Fig. 11e depicted the classiﬁer learned by                 distribution of data is different. When there are four classes,
RA-GCN. The weights trained by RA-GCN are also illustrated                      the problem of class imbalance is more obvious, as can be
in Fig. 11e in the heatmap. To better visualize weights, the                    seen in DS3, DS4, and DS5 compared to DS1 and DS2. With
logarithm  of  weights  is  scaled  to [0,1]  in  this  ﬁgure.  As              the four class scenarios (DS3, DS4, and DS5), the impact of
can be seen, each sample is differently weighted, unlike the                    class imbalance is higher in DS3, where there are three minor
weighted  cross-entropy.  RA-GCN  resolves  the  problem  by                    classes against one major class. By comparing the weighted
automatically weighing the samples of each class. The entropy                   and unweighted methods, it can be concluded that weighting is
term in the objective function from Eq.4 prevents the classiﬁer                 beneﬁcial for all datasets and methods. With four classes, for
fromemphasizingoutliers.Althoughtheminorclass’ssamples                          the DS5 in which the class imbalance is less severe, weighting
have  high  weights,  RA-GCN  assigns  more  weights  to  the                   is less effective than for DS3 and DS4. DR-GCN improves the
misclassiﬁed and boundary points of the major class which                       resultsofGCN-unweightedinDS1,DS3,andDS6.Inallthese
keeps the classiﬁer balanced and hinders it from sacriﬁcing                     sets, there is one major class. Needless to mention that the
the class samples against each other. Metrics also conﬁrm the                   information provided by the graph is helpful in all datasets.
superiority of the RA-GCN in all measurements.                                  All the results demonstrate that the proposed method, RA-
                                                                                GCN, outperforms all the competitors. This implies the power
B. Experiments on Synthetic Datasets for Multi-Class Classi-                    of automatically and dynamically weighting for multi-class
ﬁcation                                                                         datasets.
   So far, we have been through analysis, results, and vali-
dations on binary class datasets covering the different levels                  C. End-to-end Graph Construction
of  difﬁculty.  In  this  section,  we  target  more  challenging
conditions. We examine the proposed method when the class                          In the previous experiments, an unsupervised approach for
imbalance issue happens in the multi-class classiﬁcation prob-                  graph  construction  is  used.  In  this  experiment,  we  intend
lem. In this case, there are more than two classes. It should be                to  study  the  performance  of  the  proposed  method  besides
noted that the datasets are not multi-label, which means that                   learning the graph in an end-to-end manner. For this purpose,
each sample belongs to only one class.                                          we  adopt  the  method  proposed  by  Cosmo  et  al.  [39]  for
   Dataset: We generate 6  different datasets with 3, 4, and                    graph learning. The method proposed by Cosmo et al. [39]
5  classes.  All  the  datasets  contain 2000     samples.  Like  the           is  one  of  the  latest  graph-learning  methods  which  aims  to
previous experiment, 10   features are the node features, and                   learn the best graph in an end-to-end manner to optimally
the other10  are utilized for graph construction. The generated                 support the node-classiﬁcation task. In the method, ﬁrst, the
datasets cover the class imbalance problem in different ways.                   node features are embedded into a lower-dimensional space by
For  example,  in  three  classes  scenario,  we  test  two  cases:             an MLP network, then a sigmoid function is applied on the
1)  Two  classes  being  minor  classes  (0.7,0.2,0.1   for  data               linear transformation of the Euclidean distances between every
distribution), 2) One class being the minor class (0.5,0.4,0.1                  pair of nodes in the low-dimensional space. The parameters
for data distribution). More details about the six datasets are                 of the linear transformation are hyper-parameters and need to
provided in Table VIII. The adjacency density results from                      be set by the results on validation set. The outcome provides
thresholding the graphs with 0.5.                                               a weighted graph to the rest of the model to complete the
   Results: In this experiment, there are more classes, so the                  node classiﬁcation task. We added graph-learning method to
problemgetsmorechallenging.ThemacroF1istheaverageof                             RA-GCN and GCN as a component called graph-constructor.
F1-score over all classes. Hence, improvement in the macro F1                   For DR-GCN, since the implementation is done by authors,
means an improvement in all classes, on average, despite their                  to make sure that we are fair to their method, we used the
size, and a small increase is meaningful. From the results in                   learned graph by GCN-unweighted and input that graph to

               16                                                                                                                                                                                                                                                                                          2021
               (a)                                 (b)                                 (c)                                 (d)                                 (e)
                     MLP-unweighted                      MLP-weighted                        GCN-unweighted                      GCN-weighted                        RA-GCN (ours)
                        Accuracy          0.884            Accuracy         0.817               Accuracy           0.9              Accuracy        0.769               Accuracy         0.873
                        Macro F1          0.494            Macro F1         0.696               Macro F1          0.474            Macro F1         0.658               Macro F1         0.762
                        ROC AUC           0.504            ROC AUC          0.867               ROC AUC            0.5             ROC AUC          0.872              ROC AUC           0.907
               Fig. 11: Visualization of the trained classiﬁer on a dataset with two input features. Figure represents the analysis of linear
               classiﬁers trained by evaluated methods. MLP, GCN (in weighted and unweighted form), and the proposed method (RA-GCN)
               are compared. Due to the linearity of the classiﬁers (without hidden space), the output of DR-GCN is the same as GCN-
               unweighted. In all of the sub-ﬁgures, the orange line represents the learned classiﬁer. In ﬁgure (a),(b),(c) and (d) two colors
               represent the classes. In sub-ﬁgure (e) color represents the weight learned by the proposed method. Horizontal and vertical
               axes show the ﬁrst and second input features respectively. Since GCN-based methods (c, d, and e) multiply the input feature
               by adjacency matrix and use a new feature space, their classiﬁer is drawn in their own feature space.
               TABLE IX: Results of the methods on the multi-class datasets. Table includes the performance of the compared methods for
               different levels of difﬁculty for the node classiﬁcation task. The ﬁrst row represents the class distributions of the corresponding
               dataset in graphical form.
               DR-GCN. Therefore, the construction of the input graph for                                  the poor performance of DR-GCN can come from the fact
               DR-GCN is not end-to-end. The results are reported on two                                   that the learned graph by GCN-unweighted has deteriorated
               synthetic datasets with class distributions [0.95,0.05]   (highly                           its outcomes.
               imbalanced  dataset)  and [0.75,0.25]   (moderate  imbalanced
               dataset).                                                                                                              ACKNOWLEDGEMENTS
                  Results:  The  results  of  the  experiment  are  provided  in                              The authors would like to thank Mojtaba Bahrami for all
               Table X. When the graph-constructor component is added to                                   his help in editing and proof reading the article.
               each method, the number of the parameters grows, and the
               need for more data increases. In addition to that, one of the
               most important factors for constructing the best graph is to                                                                  REFERENCES
               have enough samples that enables the model to detect similar
               samples. In the imbalanced datasets, usually, the number of                                  [1]S.  Uddin,  A.  Khan,  M.  E.  Hossain,  and  M.  A.  Moni,  “Comparing
               samples  in  the  minor  class  is  limited.  This  issue  can  be                                different supervised machine learning algorithms for disease prediction,”
                                                                                                                 BMC Medical Informatics and Decision Making, vol. 19, no. 1, pp. 1–
               a  reason  that  the  results  of  classiﬁcation  with  end-to-end                                16, 2019.
               graph  construction  for  each  method  cannot  reach  the  best                             [2]S. Mohan, C. Thirumalai, and G. Srivastava, “Effective heart disease
               results of their counterpart with manual graph construction.                                      prediction  using  hybrid  machine  learning  techniques,”  IEEE  Access,
                                                                                                                 vol. 7, pp. 81542–81554, 2019.
               A comparison between the results of the two datasets also                                    [3]M.  Bayati,  S.  Bhaskar,  and  A.  Montanari,  “A  low-cost  method  for
               conﬁrms that the methods with graph-constructor shows more                                        multiple disease prediction,” in AMIA Annual Symposium Proceedings,
               deﬁciency on the higher imbalanced dataset. However, in the                                       vol. 2015.   American Medical Informatics Association, 2015, p. 329.
               graph construction based on the distance between nodes and                                   [4]A. S. Lundervold and A. Lundervold, “An overview of deep learning in
                                                                                                                 medical imaging focusing on mri,” Zeitschrift f¨ur Medizinische Physik,
               without parameters [12], the number of samples in each class                                      vol. 29, no. 2, pp. 102–127, 2019.
               is not involved. It should be also noted that as the imbalance                               [5]R. Yamashita, M. Nishio, R. K. G. Do, and K. Togashi, “Convolutional
               issue  biases  the  classiﬁer,  it  can  also  bias  the  structure  of                           neural networks: an overview and application in radiology,” Insights into
                                                                                                                 imaging, vol. 9, no. 4, pp. 611–629, 2018.
               the ﬁnal graph. Hence, better handling of the issue leads to                                 [6]C. M. Bishop, “Pattern recognition,”      Machine learning, vol. 128, no. 9,
               a better graph and results. This can be seen in the results                                       2006.
               where RA-GCN has achieved superiority over other methods                                     [7]R. Caruana, S. Lawrence, and C. L. Giles, “Overﬁtting in neural nets:
                                                                                                                 Backpropagation, conjugate gradient, and early stopping,” in Advances
               when the graph is constructed during the training. Moreover,                                      in neural information processing systems, 2001, pp. 402–408.
DistributionofData      DS1:70,20,10               DS2:50,40,10              DS3:60,25,10,5              DS4:45,35,12,8              DS5:35,30,25,10           DS6:50,20,15,10,5
 Method/Metric       Acc         MacroF1         Acc         MacroF1        Acc        MacroF1          Acc        MacroF1          Acc        MacroF1         Acc         MacroF1
 MLP-unweighted   0.81∓0.015    0.54∓0.018    0.8∓0.014     0.56∓0.01   0.72∓0.008    0.38∓0.012    0.69∓0.008    0.4∓0.008     0.64∓0.009    0.51∓0.018    0.65∓0.011   0.31∓0.026
 MLP-weighted     0.77∓0.018    0.67∓0.02     0.78∓0.018    0.7∓0.019   0.66∓0.042    0.54∓0.036    0.64∓0.023    0.56∓0.031    0.62∓0.022    0.59∓0.018    0.61∓0.032   0.47∓0.012
 GCN-unweighted   0.86∓0.004    0.59∓0.011    0.86∓0.008   0.64∓0.041   0.79∓0.01     0.44∓0.035   0.76∓0.018     0.49∓0.046   0.72∓0.021     0.59∓0.041    0.7∓0.023     0.4∓0.053
 GCN-weighted     0.84∓0.018    0.75∓0.021    0.84∓0.017    0.77∓0.02   0.74∓0.062    0.61∓0.047    0.74∓0.036    0.65∓0.041     0.7∓0.03     0.65∓0.026    0.68∓0.023   0.55∓0.028
   DR-GCN         0.85∓0.012    0.66∓0.031    0.83∓0.044   0.62∓0.020   0.75∓0.022    0.45∓0.037    0.67∓0.038    0.44∓0.026    0.71∓0.041    0.57∓0.026    0.69∓0.037   0.42∓0.062
 RA-GCN(ours)    0.88∓0.023    0.79∓0.031    0.88∓0.013    0.8∓0.023    0.76∓0.037   0.63∓0.027    0.76∓0.012    0.67∓0.016    0.72∓0.013     0.68∓0.017    0.68∓0.028   0.56∓0.038

Ghorbani et al.:Accepted for publication in Medical Image Analysis (2021)                                                                                                                                                                      17
             TABLE X: Results of the compared methods with adding the graph-constructor component to the structure.
                   Class distribution                                [0.95-0.05]                                                            [0.75-0.25]
                    Method/Metric             Accuracy              Macro F1                 ROC AUC                 Accuracy              Macro F1                ROC AUC
                  GCN - unweighted       0.949 ±0.0037           0.589 ±0.036            0.526 ±0.0207          0.948 ±0.0091          0.924 ±0.014            0.974 ±0.0032
                  (Best threshold=0.2)
                    GCN - weighted       0.958 ±0.0142          0.789 ±0.0429            0.859 ±0.0443          0.964 ±0.0037          0.951 ±0.0053           0.978 ±0.0037
                  (Best threshold=0.3)
                      RA - GCN           0.975 ±0.0061          0.854 ±0.0385            0.895 ±0.037           0.934 ±0.0171          0.902 ±0.0281           0.958 ±0.0228
                  (Best threshold=0.3)
                      DR - GCN           0.793 ±0.1414          0.622 ±0.1114            0.711 ±0.1256          0.966 ±0.0091          0.953 ±0.0129           0.984 ±0.0039
                  (Best threshold=0.3)
                  GCN - unweighted       0.919 ±0.0233          0.504 ±0.0103            0.617 ±0.0565          0.877 ±0.0164          0.809 ±0.0286           0.931 ±0.0185
                  + Graph Constructor
                    GCN - weighted       0.804 ±0.0827          0.559 ±0.0551            0.717 ±0.0239          0.886 ±0.0209          0.857 ±0.0233           0.933 ±0.0196
                  + Graph Constructor
                      DR - GCN           0.895 ±0.0321          0.532 ±0.0292            0.663 ±0.0334          0.767 ±0.0224          0.687 ±0.0216           0.788 ±0.0383
                  + Graph Constructor
                      RA - GCN           0.892 ±0.0347          0.643 ±0.0396            0.775 ±0.0488          0.906 ±0.0121          0.876 ±0.0157           0.953 ±0.0121
                  + Graph Constructor
 [8]H.ZhangandM.Gabbouj,“Featuredimensionalityreductionwithgraph                                    [25]P.  Veli      ˇckovi´c,  G.  Cucurull,  A.  Casanova,  A.  Romero,  P.  Lio,  and
      embedding  and  generalized  hamming  distance,”  in  2018  25th  IEEE                               Y. Bengio, “Graph attention networks,” in  ICLR, 2017.
      International Conference on Image Processing (ICIP).     IEEE, 2018,                          [26]Q. Li, X.-M. Wu, H. Liu, X. Zhang, and Z. Guan, “Label efﬁcient semi-
      pp. 1083–1087.                                                                                       supervised learning via graph ﬁltering,” in Proceedings of the IEEE
 [9]Z. Zhang, P. Cui, and W. Zhu, “Deep learning on graphs: A survey,”                                     Conference  on  Computer  Vision  and  Pattern  Recognition,  2019,  pp.
      IEEE Transactions on Knowledge and Data Engineering, 2020.                                           9582–9591.
[10]M. M. Bronstein, J. Bruna, Y. LeCun, A. Szlam, and P. Vandergheynst,                            [27]W.  Hamilton,  Z.  Ying,  and  J.  Leskovec,  “Inductive  representation
      “Geometric deep learning: going beyond euclidean data,”  IEEE Signal                                 learning on large graphs,” in Advances in neural information processing
      Processing Magazine, vol. 34, no. 4, pp. 18–42, 2017.                                                systems, 2017, pp. 1024–1034.
[11]A.  Kazi,  S.  Shekarforoush,  S.  A.  Krishna,  H.  Burwinkel,  G.  Vivar,                     [28]E.  Rossi,  F.  Frasca,  B.  Chamberlain,  D.  Eynard,  M.  Bronstein,  and
      B. Wiestler, K. Kort¨um, S.-A. Ahmadi, S. Albarqouni, and N. Navab,                                  F.  Monti,  “Sign:  Scalable  inception  graph  neural  networks,”   arXiv
      “Graph  convolution  based  attention  model  for  personalized  disease                             preprint arXiv:2004.11198, 2020.
      prediction,” in International Conference on Medical Image Computing                           [29]M.Ramezani,W.Cong,M.Mahdavi,A.Sivasubramaniam,andM.Kan-
      and Computer-Assisted Intervention.   Springer, 2019, pp. 122–130.                                   demir, “Gcn meets gpu: Decoupling “when to sample” from “how to
[12]S. Parisot, S. I. Ktena, E. Ferrante, M. Lee, R. G. Moreno, B. Glocker,                                sample”,” Advances in Neural Information Processing Systems, vol. 33,
      and  D.  Rueckert,  “Spectral  graph  convolutions  for  population-based                            2020.
      disease prediction,” in International conference on medical image com-                        [30]X.  Li  and  J.  Duncan,  “Braingnn:  Interpretable  brain  graph  neural
      puting and computer-assisted intervention.    Springer, 2017, pp. 177–                               network for fmri analysis,” bioRxiv, 2020.
      185.                                                                                          [31]H. Du, J. Feng, and M. Feng, “Zoom in to where it matters: a hier-
[13]M.  Ghorbani,  M.  S.  Baghshah,  and  H.  R.  Rabiee,  “Mgcn:  Semi-                                  archical graph based model for mammogram analysis,” arXiv preprint
      supervised classiﬁcation in multi-layer graphs with graph convolutional                              arXiv:1912.07517, 2019.
      networks,” in Proceedings of the 2019 IEEE/ACM International Confer-                          [32]R. D. Soberanis-Mukul, N. Navab, and S. Albarqouni, “Uncertainty-
      ence on Advances in Social Networks Analysis and Mining, 2019, pp.                                   based graph convolutional networks for organ segmentation reﬁnement,”
      208–211.                                                                                             in Medical Imaging with Deep Learning.   PMLR, 2020, pp. 755–769.
[14]C.-H. Cheng, J.-R. Chang, and H.-H. Huang, “A novel weighted distance                           [33]R. Anirudh and J. J. Thiagarajan, “Bootstrapping graph convolutional
      threshold method for handling medical missing values,” Computers in                                  neural networks for autism spectrum disorder classiﬁcation,” inICASSP
      Biology and Medicine, vol. 122, p. 103824, 2020.                                                     2019-2019 IEEE International Conference on Acoustics, Speech and
[15]R. Ashraf, M. A. Habib, M. Akram, M. A. Latif, M. S. A. Malik,                                         Signal Processing (ICASSP).   IEEE, 2019, pp. 3197–3201.
      M. Awais, S. H. Dar, T. Mahmood, M. Yasir, and Z. Abbas, “Deep                                [34]Y. Li, B. Qian, X. Zhang, and H. Liu, “Graph neural network-based
      convolution neural network for big data medical image classiﬁcation,”                                diagnosis prediction,” Big Data, vol. 8, no. 5, pp. 379–390, 2020.
      IEEE Access, vol. 8, pp. 105659–105670, 2020.                                                 [35]N.  Ravindra,  A.  Sehanobish,  J.  L.  Pappalardo,  D.  A.  Haﬂer,  and
[16]B.  Subramani  and  M.  Veluchamy,  “A  fast  and  effective  method  for                              D. van Dijk, “Disease state prediction from single-cell data using graph
      enhancement of contrast resolution properties in medical images,” Mul-                               attention networks,” in Proceedings of the ACM Conference on Health,
      timedia Tools and Applications, vol. 79, no. 11, pp. 7837–7855, 2020.                                Inference, and Learning, 2020, pp. 121–130.
[17]Y. Sun, A. K. Wong, and M. S. Kamel, “Classiﬁcation of imbalanced                               [36]A.  Kazi,  S.  Shekarforoush,  S.  A.  Krishna,  H.  Burwinkel,  G.  Vivar,
      data: A review,” International journal of pattern recognition and artiﬁ-                             K. Kort¨um, S.-A. Ahmadi, S. Albarqouni, and N. Navab, “Inceptiongcn:
      cial intelligence, vol. 23, no. 04, pp. 687–719, 2009.                                               receptive ﬁeld aware graph convolutional network for disease predic-
[18]T. N. Kipf and M. Welling, “Semi-supervised classiﬁcation with graph                                   tion,” in International Conference on Information Processing in Medical
      convolutional networks,” in ICLR, 2017.                                                              Imaging.   Springer, 2019, pp. 73–85.
[19]Z.  Wu,  S.  Pan,  F.  Chen,  G.  Long,  C.  Zhang,  and  S.  Y.  Philip,  “A                   [37]Y.  Huang  and  A.  C.  Chung,  “Edge-variational  graph  convolutional
      comprehensive survey on graph neural networks,” IEEE Transactions                                    networks  for  uncertainty-aware  disease  prediction,”  in  International
      on Neural Networks and Learning Systems, 2020.                                                       Conference on Medical Image Computing and Computer-Assisted In-
[20]D.  I.  Shuman,  S.  K.  Narang,  P.  Frossard,  A.  Ortega,  and  P.  Van-                            tervention.   Springer, 2020, pp. 562–572.
      dergheynst, “The emerging ﬁeld of signal processing on graphs: Ex-                            [38]X. Song, F. Zhou, A. F. Frangi, J. Cao, X. Xiao, Y. Lei, T. Wang,
      tending high-dimensional data analysis to networks and other irregular                               and  B.  Lei,  “Graph  convolution  network  with  similarity  awareness
      domains,” IEEE signal processing magazine, vol. 30, no. 3, pp. 83–98,                                and adaptive calibration for disease-induced deterioration prediction,”
      2013.                                                                                                Medical Image Analysis, vol. 69, p. 101947, 2021.
[21]J. Bruna, W. Zaremba, A. Szlam, and Y. LeCun, “Spectral networks and                            [39]L.Cosmo,A.Kazi,S.-A.Ahmadi,N.Navab,andM.Bronstein,“Latent-
      locally connected networks on graphs,” 2013.                                                         graph  learning  for  disease  prediction,”  in  International  Conference
[22]M. Defferrard, X. Bresson, and P. Vandergheynst, “Convolutional neural                                 on  Medical  Image  Computing  and  Computer-Assisted  Intervention.
      networks on graphs with fast localized spectral ﬁltering,” inAdvances                                Springer, 2020, pp. 643–653.
      in neural information processing systems, 2016, pp. 3844–3852.                                [40]X. Wu, W. Lan, Q. Chen, Y. Dong, J. Liu, and W. Peng, “Inferring
[23]S. Zhang, H. Tong, J. Xu, and R. Maciejewski, “Graph convolutional                                     lncrna-disease associations based on graph autoencoder matrix comple-
      networks:  a  comprehensive  review,”  Computational  Social  Networks,                              tion,” Computational biology and chemistry, vol. 87, p. 107282, 2020.
      vol. 6, no. 1, p. 11, 2019.                                                                   [41]W.  Wang,  J.  Luo,  C.  Shen,  and  N.  H.  Tu,  “A  graph  convolutional
[24]M.Balcilar,G.Renton,P.H      ´eroux,B.Gauzere,S.Adam,andP.Honeine,                                     matrix completion method for mirna-disease association prediction,” in
      “Bridging the gap between spectral and spatial domains in graph neural                               International Conference on Intelligent Computing.   Springer, 2020, pp.
      networks,” arXiv preprint arXiv:2003.11702, 2020.                                                    201–215.

18                                                                                                                                                                                                                                                                                          2021
[42]E. Burnaev, P. Erofeev, and A. Papanov, “Inﬂuence of resampling on                              [64]H.  Das,  P.  K.  Pattnaik,  S.  S.  Rautaray,  and  K.-C.  Li,        Progress  in
      accuracy of imbalanced classiﬁcation,” inEighth International Confer-                                Computing, Analytics and Networking: Proceedings of ICCAN 2019.
      ence on Machine Vision (ICMV 2015), vol. 9875.   International Society                               Springer Nature, 2020, vol. 1119.
      for Optics and Photonics, 2015, p. 987521.                                                    [65]J. W. Smith, J. Everhart, W. Dickson, W. Knowler, and R. Johannes,
[43]Q. Dong, S. Gong, and X. Zhu, “Imbalanced deep learning by minority                                    “Using the adap learning algorithm to forecast the onset of diabetes
      class incremental rectiﬁcation,”IEEE transactions on pattern analysis                                mellitus,” in Proceedings of the Annual Symposium on Computer Appli-
      and machine intelligence, vol. 41, no. 6, pp. 1367–1381, 2018.                                       cation in Medical Care.    American Medical Informatics Association,
[44]F.Rayhan, S.Ahmed, A.Mahbub,R. Jani,S.Shatabda, andD. M.Farid,                                         1988, p. 261.
      “Cusboost: Cluster-based under-sampling with boosting for imbalanced                          [66]K. Marek, D. Jennings, S. Lasch, A. Siderowf, C. Tanner, T. Simuni,
      classiﬁcation,” in2017 2nd International Conference on Computational                                 C. Coffey, K. Kieburtz, E. Flagg, S. Chowdhury et al., “The parkinson
      Systems and Information Technology for Sustainable Solution (CSITSS).                                progression marker initiative (ppmi),” Progress in neurobiology, vol. 95,
      IEEE, 2017, pp. 1–5.                                                                                 no. 4, pp. 629–635, 2011.
[45]H.-I. Lin and C.-M. Nguyen, “Boosting minority class prediction on                              [67]S. J. Haberman, “Generalized residuals for log-linear models,” in        Pro-
      imbalanced point cloud data,” Applied Sciences, vol. 10, no. 3, p. 973,                              ceedings of the 9th international biometrics conference, 1976, pp. 104–
      2020.                                                                                                122.
[46]C.Zhang,K.C.Tan,H.Li,andG.S.Hong,“Acost-sensitivedeepbelief                                     [68]C. Baur, B. Wiestler, S. Albarqouni, and N. Navab, “Deep autoencoding
      network  for  imbalanced  classiﬁcation,” IEEE  transactions  on  neural                             models for unsupervised anomaly segmentation in brain mr images,” in
      networks and learning systems, vol. 30, no. 1, pp. 109–122, 2018.                                    International MICCAI Brainlesion Workshop.   Springer, 2018, pp. 161–
[47]T. W. Cenggoro        et al., “Deep learning for imbalance data classiﬁcation                          169.
      using class expert generative adversarial network,” Procedia Computer                         [69]G. Vivar, A. Kazi, H. Burwinkel, A. Zwergal, N. Navab, and S.-A.
      Science, vol. 135, pp. 60–67, 2018.                                                                  Ahmadi, “Simultaneous imputation and disease classiﬁcation in incom-
[48]A. Sze-To and A. K. Wong, “A weight-selection strategy on training                                     plete medical datasets using multigraph geometric matrix completion
      deep neural networks for imbalanced classiﬁcation,” in International                                 (mgmc),” arXiv preprint arXiv:2005.06935, 2020.
      Conference Image Analysis and Recognition.    Springer, 2017, pp. 3–                          [70]V.NairandG.E.Hinton,“Rectiﬁedlinearunitsimproverestrictedboltz-
      10.                                                                                                  mann machines,” in Proceedings of the 27th international conference on
[49]P. Wang, S. Li, F. Ye, Z. Wang, and M. Zhang, “Packetcgan: Exploratory                                 machine learning (ICML-10), 2010, pp. 807–814.
      study of class imbalance for encrypted trafﬁc classiﬁcation using cgan,”                      [71]D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,”
      in ICC 2020-2020 IEEE International Conference on Communications                                     arXiv preprint arXiv:1412.6980, 2014.
      (ICC).   IEEE, 2020, pp. 1–7.                                                                 [72]A. Paszke, S. Gross, S. Chintala, G. Chanan, E. Yang, Z. DeVito, Z. Lin,
[50]Y. Freund, R. E. Schapire        et al., “Experiments with a new boosting                              A. Desmaison, L. Antiga, and A. Lerer, “Automatic differentiation in
      algorithm,” in icml, vol. 96.   Citeseer, 1996, pp. 148–156.                                         pytorch,” 2017.
                                                                                                    [73]F.  Pedregosa,  G.  Varoquaux,  A.  Gramfort,  V.  Michel,  B.  Thirion,
[51]L. Zhang, H. Yang, and Z. Jiang, “Imbalanced biomedical data classiﬁ-                                  O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vander-
      cation using self-adaptive multilayer elm combined with dynamic gan,”                                plas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duch-
      Biomedical engineering online, vol. 17, no. 1, p. 181, 2018.                                         esnay, “Scikit-learn: Machine learning in Python,”  Journal of Machine
[52]Y. Zhao, Z. S.-Y. Wong, and K. L. Tsui, “A framework of rebalancing                                    Learning Research, vol. 12, pp. 2825–2830, 2011.
      imbalanced  healthcare  data  for  rare  events’  classiﬁcation:  a  case  of                 [74]I. Guyon, “Design of experiments of the nips 2003 variable selection
      look-alike sound-alike mix-up incident detection,” Journal of healthcare                             benchmark,” in NIPS 2003 workshop on feature extraction and feature
      engineering, vol. 2018, 2018.                                                                        selection, vol. 253, 2003.
[53]S. Fotouhi, S. Asadi, and M. W. Kattan, “A comprehensive data level
      analysisforcancerdiagnosisonimbalanceddata,”Journalofbiomedical
      informatics, vol. 90, p. 103089, 2019.
[54]M. Alghamdi, M. Al-Mallah, S. Keteyian, C. Brawner, J. Ehrman, and
      S.Sakr,“Predictingdiabetesmellitususingsmoteandensemblemachine
      learning approach: The henry ford exercise testing (ﬁt) project,”PloS
      one, vol. 12, no. 7, p. e0179805, 2017.
[55]P.  Vuttipittayamongkol  and  E.  Elyan,  “Improved  overlap-based  un-
      dersampling for imbalanced dataset classiﬁcation with application to
      epilepsy and parkinson’s disease,” International journal of neural sys-
      tems, vol. 30, no. 08, p. 2050043, 2020.
[56]R. Cruz, M. Silveira, and J. S. Cardoso, “A class imbalance ordinal
      method  for  alzheimer’s  disease  classiﬁcation,”  in 2018  International
      Workshop on Pattern Recognition in Neuroimaging (PRNI).     IEEE,
      2018, pp. 1–4.
[57]M. Shi, Y. Tang, X. Zhu, D. Wilson, and J. Liu, “Multi-class imbalanced
      graph convolutional network learning,” in Proceedings of the Twenty-
      NinthInternationalJointConferenceonArtiﬁcialIntelligence,IJCAI-20.
      International Joint Conferences on Artiﬁcial Intelligence Organization,
      7 2020, pp. 2879–2885.
[58]J. Park, J. Lee, I.-J. Kim, and K. Sohn, “Sumgraph: Video summariza-
      tion via recursive graph modeling,” in 16th European Conference on
      Computer Vision, ECCV 2020.   Springer, 2020, pp. 647–663.
[59]Y. Cui, M. Jia, T.-Y. Lin, Y. Song, and S. Belongie, “Class-balanced
      loss based on effective number of samples,” in Proceedings of the IEEE
      Conference  on  Computer  Vision  and  Pattern  Recognition,  2019,  pp.
      9268–9277.
[60]I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley,
      S.Ozair,A.Courville,andY.Bengio,“Generativeadversarialnetworks,”
      Communications of the ACM, vol. 63, no. 11, pp. 139–144, 2020.
[61]P. Branco, L. Torgo, and R. P. Ribeiro, “A survey of predictive modeling
      on imbalanced domains,” ACM Computing Surveys (CSUR), vol. 49,
      no. 2, pp. 1–50, 2016.
[62]H. He and Y. Ma, “Imbalanced learning: foundations, algorithms, and
      applications,” 2013.
[63]A.  Fern      ´andez,  S.  Garc´ıa,  M.  Galar,  R.  C.  Prati,  B.  Krawczyk,  and
      F. Herrera, Learning from imbalanced data sets.   Springer, 2018.

