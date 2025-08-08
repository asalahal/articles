[Journal of Ethnopharmacology 297 (2022) 115109](https://doi.org/10.1016/j.jep.2022.115109)


Contents lists available at ScienceDirect

# Journal of Ethnopharmacology


[journal homepage: www.elsevier.com/locate/jethpharm](https://www.elsevier.com/locate/jethpharm)

## TCM herbal prescription recommendation model based on multi-graph convolutional network


Wen Zhao [a], Weikai Lu [b], Zuoyong Li [c], Chang’en Zhou [a], Haoyi Fan [d], Zhaoyang Yang [a],
Xuejuan Lin [a] [,] [*], Candong Li [a] [,] [** ]


a _School of Traditional Chinese Medicine, Fujian University of Traditional Chinese Medicine, Fuzhou, 350122, China_
b _School of Electronic, Electrical Engineering and Physics, Fujian University of Technology, Fuzhou, 350118, China_
c _Fujian Provincial Key Laboratory of Information Processing and Intelligent Control, College of Computer and Control Engineering, Minjiang University, Fuzhou, 350121,_

_China_
d _School of Computer Science and Technology, Harbin University of Science and Technology, Harbin, 150080, China_



A R T I C L E I N F O


_Keywords:_
Artificial intelligence
Graph convolutional network
TCM herbal Prescription
_N_ -ary relationship
Intelligent recommendation



A B S T R A C T


_Ethnopharmacological relevance:_ The recommendation of herbal prescriptions is a focus of research in traditional
Chinese medicine (TCM). Artificial intelligence (AI) algorithms can generate prescriptions by analysing symptom
data. Current models mainly focus on the binary relationships between a group of symptoms and a group of TCM
herbs. A smaller number of existing models focus on the ternary relationships between TCM symptoms,
syndrome-types and herbs. However, the process of TCM diagnosis (symptom analysis) and treatment (pre­
scription) is, in essence, a “multi-ary” ( _n_ -ary) relationship. Present models fall short of considering the _n_ -ary
relationships between symptoms, state-elements, syndrome-types and herbs. Therefore, there is room for
improvement in TCM herbal prescription recommendation models.
_Purpose:_ To portray the _n_ -ary relationship, this study proposes a prescription recommendation model based on a
multigraph convolutional network (MGCN). It introduces two essential components of the TCM diagnosis pro­
cess: state-elements and syndrome-types.
_Methods:_ The MGCN consists of two modules: a TCM feature-aggregation module and a herbal medicine pre­
diction module. The TCM feature-aggregation module simulates the _n_ -ary relationships between symptoms and
prescriptions by constructing a symptom-‘state element’-symptom graph ( _S_ _e_ ) and a symptom-‘syndrome-type’symptom graph ( _T_ _s_ ). The herbal medicine prediction module inputs state-elements, syndrome-types and symp­
tom data and uses a multilayer perceptron (MLP) to predict a corresponding herbal prescription. To verify the
effectiveness of the proposed model, numerous quantitative and qualitative experiments were conducted on the

_Treatise on Febrile Diseases_ dataset.

_Results:_ In the experiments, the MGCN outperformed three other algorithms used for comparison. In addition, the
experimental data shows that, of these three algorithms, the SVM performed best. The MGCN was 4.51%, 6.45%
and 5.31% higher in Precision@5, Recall@5 and F1-score@5, respectively, than the SVM. We set the _K_ -value to 5
and conducted two qualitative experiments. In the first case, all five herbs in the label were correctly predicted
by the MGCN. In the second case, four of the five herbs were correctly predicted.
_Conclusions:_
Compared with existing AI algorithms, the MGCN significantly improved the accuracy of TCM herbal
prescription recommendations. In addition, the MGCN provides a more accurate TCM prescription herbal
recommendation scheme, giving it great practical application value.



_Abbreviations:_ TCM, traditional Chinese medicine; AI, artificial intelligence; MGCN, multigraph convolutional network; S e, symptom-‘state element’-symptom

’
graph; T s, symptom-‘syndrome-type -symptom graph; MLP, multilayer perceptron; SVM, support vector machine; LR, logistic regression; PTM, potential topic model;
GCN, graph convolutional network; OvR, one-vs-rest; S e GCN, state-element graph convolutional network; TsGCN, TCM syndrome-type graph convolutional network.

 - Corresponding author.
** Corresponding author.
_E-mail addresses:_ [wynn.wen.zhao@fjtcm.edu.cn](mailto:wynn.wen.zhao@fjtcm.edu.cn) (W. Zhao), [550521071@qq.com](mailto:550521071@qq.com) (W. Lu), [fzulzytdq@126.com](mailto:fzulzytdq@126.com) (Z. Li), [2007038@fjtcm.edu.cn](mailto:2007038@fjtcm.edu.cn) (C. Zhou),
[isfanhy@hrbust.edu.cn (H. Fan), yzy813@126.com (Z. Yang), lxjfjzy@126.com (X. Lin), fjzylcd@126.com (C. Li).](mailto:isfanhy@hrbust.edu.cn)


[https://doi.org/10.1016/j.jep.2022.115109](https://doi.org/10.1016/j.jep.2022.115109)
Received 16 December 2021; Received in revised form 11 February 2022; Accepted 12 February 2022

Available online 25 February 2022
0378-8741/© 2022 Published by Elsevier B.V.


_W. Zhao et al._ _Journal of Ethnopharmacology 297 (2022) 115109_



**1. Introduction**


In traditional Chinese medicine (TCM) diagnosis and treatment,
doctors usually prescribe herbs based on the patient’s symptoms.
However, this process is generally considered a “black box” method,
which is seemingly highly subjective. This impression stems from TCM’s
unique theoretical system and abstract philosophy. The foothold of TCM
theory is in its holism, which emphasizes the wholeness of the human
body and its relationship with the natural environment (Wang and Xu,
2014). TCM uses this rather abstract concept to understand the form and
function of the human body and evaluate and regulate its different states
in health and disease (Zhao et al., 2020). In essence, the “black box”
diagnosis and treatment process is neither mysterious nor subjective. A
typical clinical process can be summarized as follows: First, the TCM
doctor collects the patient’s symptom information through the four
methods of diagnosis (四诊), namely, diagnosis through observation
(望), auscultation and olfaction (闻), inquiry (问), and pulse feeling and
palpation (切). After the information is collected, the underlying
disease-location (病位) and disease-nature (病性) (a component of the
state-element 状态要素) are analysed using the holistic concepts of TCM.
Second, the disease-location (for example, liver, heart, or spleen) and
disease-nature (for example, cold, heat, or deficiency) are combined to
form a particular syndrome-type (证型). Finally, a prescription is rec­
ommended based on the syndrome-type. In short, the so-called “black
box” method is actually a process of induction and generalization of the
state-elements and syndrome-types within the TCM framework of
diagnostics.

Fig. 1 elaborates on the importance of state-elements and syndrometypes during induction and generalization in TCM clinical practice, ac­
cording to the TCM prescription _Sini San_ (四逆散). The diagnosis and
treatment process can generally be broken down into four steps: (1)
Symptom information gathering, where the doctor gathers the patient’s
symptom information through the four methods of diagnosis. The
symptom set in this example comprises “diarrhoea”, “abdominal pain”,
and “cold hands and feet”. (2) Induction of state-elements, where stateelements are determined after a comprehensive analysis of the symp­
toms. In this case, the disease-location (“intestine” and “liver”) and
disease-nature (“ _qi_ stagnation” and “cold”) constitute the patient’s stateelement. (3) Induction of syndrome-type, where a syndrome-type is
formed through a process of recombination of the disease-location and
disease-nature. In this case, based on the main pathological state of the
illness, the “intestine”, “liver”, “ _qi_ stagnation”, and “cold” are correlated
and recombined with a set of TCM rules to deduce the syndrome-type



“Liver _qi_ stagnation”. (4) Selection of herbal prescriptions. The doctor
selects a corresponding prescription to treat the disease based on the
resulting state-element and syndrome-type. In this case, the prescription
consists of “ _Bupleurum falcatum_ subsp. _Cernuum_ (Ten.) Arcang."(柴胡),
“ _Solanum truncicola_ Bitter"(枳实), “ _Paeonia lactiflora_ Pall."(芍药), and
“ _Cucumis melo_ var. _honey-dew_ Hassib"(炙甘草). A search in the TCM
database indicates that the combination is “ _Sini San_ ”, a prescription in
the book _Treatise on Febrile Diseases_ . Understanding the process makes it
obvious that the “black box” method comprises the second and third
steps in the TCM diagnosis and treatment sequence. Fundamentally, this
process is an intricate connection between symptoms, state-elements,
syndrome-types, and herbal prescriptions. It is not a simple binary
relationship but is an _n_ -ary relationship.
Using artificial intelligence (AI) algorithms to make herbal pre­
scriptions is an important topic in TCM research. Existing TCM pre­
scription recommendation models can generally be divided into two
categories: _non-topic_ and _topic_ models. Non-topic models treat diagnostic
data as label classification problems and focus on simulating the binary
relations between a set of symptoms and a prescription set. In other
words, they overlook steps two and three of the diagnosis and treatment
process. Their resulting accuracy is often less than ideal. Representative
algorithms in this category comprise somewhat traditional machine
learning algorithms, such as support vector machine (SVM; Chang and
Lin, 2011) and logistic regression (LR; Hosmer, 2013). In contrast, topic
models induce and classify the implicit relationship between symptoms
and herbs according to specific topics (Yao et al., 2015, 2018; Ma and
Zhang, 2018; Blei et al., 2003; Zhao et al., 2018; Zhou et al., 2021; Wang
et al., 2019). Fundamentally, topic modelling is an optimized form of
non-topic modelling that introduces topic modules to simulate the
ternary relationship between symptoms and herbal prescriptions. An
example algorithm is the potential topic model (PTM) constructed by
Yao et al. (2015, 2018) and Ma et al. (2018). They modelled “syndro­
me-type” topics and attempted to mine the ternary relationship between
symptoms and herbs, thereby significantly improving the overall accu­
racy of the recommendation model.
In recent years, the graph convolutional network (GCN) has gathered
widespread attention due to its outstanding performance and high
interpretability. The GCN excels in mining the complex relationships
between pairs of entities, especially in image recognition where it can
determine the underlying correlations between image pairs. TCM diag­
nosis and treatment processes can be treated as an “image” sketched out
by the four methods of diagnosis to represent the patient’s illness. Then,
a corresponding treatment is prescribed by mining the complex



**Fig. 1.** Diagram of the TCM diagnosis and treatment process.


2


_W. Zhao et al._ _Journal of Ethnopharmacology 297 (2022) 115109_



correlations within the “image”. This assumption enabled past re­
searchers to build recommendation models based on a GCN. Li et al.

(2018) and Ruan et al. (2019) used a GCN to simulate the process of
prescription generation. Jin et al. (2020b) used the embedding method
of GCN to simulate the process of generating TCM syndromes to gain
better prescription recommendations.
The current study is inspired by GCN and PTM and proposes a TCM
prescription recommendation model named the _multi-graph convolu­_
_tional network_ (MGCN). The MGCN comprises a TCM featureaggregation module and a herbal medicine prediction module. In the
TCM feature-aggregation module, a symptom–‘state-element’–symptom
graph ( _S_ _e_ ) and a symptom–‘syndrome-type’–symptom graph ( _T_ _s_ ) are
constructed to extract features by graph convolution, and the potential
correlations between symptoms, state-elements and syndrome-types are
fully explored. This simulates the _n_ -ary relationships between symptoms
and prescriptions in TCM diagnosis and treatment. In the herbal medi­
cine prediction module, the fused state-elements, syndrome-types, and
symptom information are used as model input. A multilayer perceptron
(MLP) is then used to predict the corresponding herbal prescription in
the form of probability values. The model performance was tested on the
_Treatise on Febrile Diseases_ dataset, which is classical, widely recognized
TCM literature. Quantitative analysis and case analysis experiments
show that MGCN significantly improves the accuracy of herbal pre­
scription recommendations compared with those of existing machine
learning algorithms.
The current study report is organized as follows. Section 2 reviews
the related literature. Section 3 introduces the methods and models.

Section 4 describes the experimental procedures and results, then dis­
cusses the influence of the ablation study and the choice of essential
model parameters. A conclusion summarises the study.


**2. Related work**


The SVM and LR algorithms can be used to provide intelligent herbal
prescription recommendations based on non-topic models. They focus
on simulating the binary relationships between symptoms and pre­
scriptions. Through data training, the SVM completes binary classifica­
tions on optimal separating hyperplanes in the feature space. This
approach is excellent for small sample sets but not when handling multilabel classifications. In this experiment, kernel function and the one-vsrest (OvR) strategy were used to enhance the SVM, enabling non-linear
multi-label classification. LR classification is widely used in medicine
and was used to establish a regression formula for the decision boundary
based on data training.
Topic model algorithms have become a prevalent solution for
recommendation models in recent years. They improve recommenda­
tions by targeted mining of the ternary relationships between symptoms,
syndrome-types, and herbs. For example, Yao and Ma used PTM (Yao
et al., 2018) to explore the relationships between symptoms and herbs.
Not only did they incorporate syndrome-types into the equation, they
also tried to use TCM methodologies such as “herb-pairs” (药对) and
“Junchen Zuozhu” (君臣佐使; combination rules for TCM prescriptions)
to create a variety of topic models. In addition, Zhao et al. (2018) pro­
posed a double-end fusion recommendation topic model, while Zhou
et al. (2021) proposed an integrating phenotype and molecule infor­
mation topic model and Wang et al. (2019) devised a knowledge graph
topic model, which increased the capability to mine the underlying as­
sociations between symptoms and herbs. These models exceeded the
performance of traditional non-topic models in terms of generalizability
and herbal recommendation. However, none of these methods consid­
ered the correlation between state-elements and symptoms. Further­
more, they neglected to fully simulate the _n_ -ary relationship between
symptoms and herbs.
The GCN has received widespread attention for its outstanding per­
formance and interpretability. The GCN (Kipf and Welling, 2017;
Hamilton et al., 2018; Veliˇckovi´c et al., 2018) extends convolutional



neural networks (CNNs) on the graph data structure. It identifies un­
derlying associations by mining the characteristics of various nodes and
their relationships. This algorithm is extensively used in computer
vision, natural language processing, recommendation systems, and
other fields (Wang et al., 2020; Li and Yang, 2017; Ruan et al., 2017). In
the computer vision field, Johnson et al. (2018) developed an end-to-end
method to generate images from scene graphs using a GCN. This method
is a better solution for machine image recognition and image generation.
In the natural language processing field, Koncel-Kedziorski et al. (2019)
used a GCN to build an end-to-end trainable system for graph-to-text
conversion. In recommendation technology, Chang et al. (2020) and
Jin et al. (2020a) used a GCN to implement intelligent recommendation
processes to provide customers with better recommendation services.
Inspired by the successful applications of PTM and GCN, this study
proposes the novel MGCN network to further enhance the performance
of the herbal prescription recommendation model. The main contribu­
tions of this study are:


(1) The MGCN introduces state-elements and syndrome-types to
simulate the _n_ -ary relationships between symptoms and herbal
prescriptions.


**3. Proposed methods and models**


The MGCN consists of a TCM feature-aggregation module and a
herbal prediction module, as shown in Fig. 2. In the TCM featureaggregation module, the state-element and syndrome-type induction
phases are embedded into the model by convolution. This simulates the
_n_ -ary relationship in the process from symptoms to herbs and simulates
the second and third steps of the TCM diagnosis and treatment process.
Furthermore, a symptom-‘state-element’-symptom graph ( _S_ _e_ ) and a
symptom-‘syndrome-type’-symptom graph ( _T_ _s_ ) graph are constructed to
extract features by convolution, enabling the potential correlations be­
tween symptoms, state-elements, and syndrome types to be fully
explored. In the herbal prediction module, a fusion of state-elements,
syndrome-types, and symptoms further enriches the diversity of the
input data. The results provide additional evidence for the effectiveness
of computerized TCM herbal prescription recommendations. In addi­
tion, MGCN self-learning is achieved using MLP iterative training. The
following section introduces the architecture of the MGCN and describes
its principal features.


_3.1. TCM feature-aggregation module_


The TCM feature-aggregation module was developed with the help of
GCN to simulate the key aspects of TCM diagnosis and treatment. The
module mainly functions through the construction of _S_ _e_ and _T_ _s_ graphs
and feature aggregation. The specific steps are as follows:


(1) Construction of _S_ _e_ and _T_ _s_ graphs


Firstly, _X_ _ps_ contains the set of symptoms in each experimental dataset
and represents them in the model with multi-hot encoding. Let _X_ _p_ = {X 1,
X 2, … X n }, where n is the total number of clinical data points in the
dataset, then _X_ _i_ = {S 1,S 2, … S t }, where t is the number of symptom types
contained in the dataset. If symptom S 1 appears in clinical data i, it is set
to “1 [′′], otherwise it is set to “0". Secondly, each clinical data point p in
the dataset can be expressed as {{s 1, s 2, …, s k }, {h 1, h 2, …, h m }, {se 1, se 2,
…, se z }, {ts 1, ts 2, …, ts j }}, where s 1 − s k represents the symptom
collection, h 1 − h m represents the herbal collection, se 1 − se z represents
the state-element collection, and ts 1 − ts j represents the syndrome-type

’
collection of the clinical data. Finally, the symptom-‘state-element symptom graph ( _S_ _e_ ) and symptom-‘syndrome-type’-symptom graph
( _T_ _S_ ), respectively, are defined to generate a convoluted representation of
the relationship between symptoms and symptoms.
In this paper, _T_ _S_ and _S_ _e_ are expressed as undirected graphs.



3


_W. Zhao et al._ _Journal of Ethnopharmacology 297 (2022) 115109_


**Fig. 2.** Schematic diagram of the multi-graph convolutional network (MGCN).


(2) In the TCM feature-aggregation module in the MGCN, features are extracted by constructing a symptom–‘state-element’–symptom graph ( _S_ _e_ ) and a symp­

tom–‘syndrome-type’–symptom graph ( _T_ _s_ ) for convolution to fully explore the potential correlations among symptoms, state-elements, and syndrome-types.
(3) In the herbal prediction module of the MGCN, the fused state-elements, syndrome-types, and symptom data are designated as inputs and an MLP is applied to
predict the corresponding herbal prescriptions based on probability values.
(4) Experiments were conducted on the _Treatise of Febrile Diseases_ dataset, which contains 1329 symptoms, 1641 state-elements, 358 syndrome-types, and 1627
herbs. It was found that the MGCN significantly improves the accuracy of herbal prescription recommendations compared with those of existing machine
learning algorithms.



Therefore, each can be expressed by a symptom-symptom relation ma­
trix. _R_ _s1,s2_ denotes the relationship between symptoms s 1 and s 2 . The
variable _N_ represents the number of state-elements, and se( _N_ ) denotes
the set of _N_ or more state-elements. The value range of _N_ is 1–5. Notably,
the method of constructing _S_ _e_ can be implied as follows: If two symptoms
and multiple state-elements are present simultaneously, there is a
particular relationship between the two symptoms. Specifically, the el­
ements in the relationship matrix of graph _S_ _e_ can be defined by the
following relations:

R s1 _,_ s2 R s2 _,_ s1 ={10 _,,_ ifotherwise(s1 _,_ s2 _,_ se _._ (N))appearsinaparticularclinicaldatasimultaneously


(1)

Similarly, t su can be defined as any of the syndrome types, and the
relationship matrix of the _T_ _S_ graph can be defined as:

R s1 _,_ s2 R s2 _,_ s1 = { 10 _,,_ if otherwise(s1 _,_ s2 _,_ tsu _._ ) appearsinaparticularclinicaldatasimultaneously _._


(2)


(2) Feature aggregation


First, _S_ _e_ and _T_ _S_ are encoded with different GCNs, respectively. This
step forms the new symptom-embedding representation _Z_ _pe_, which in­
tegrates symptoms and state-element features, and _Z_ _pt_, which integrates
symptoms and syndrome-type features. Then, _Z_ _pe_ and _Z_ _pt_ are merged to
obtain new symptom representations. Finally, the average embedding
representations of the symptom groups in each clinical data point are
calculated to obtain the clinical data point embedding representation
_Z_ _ps_ . This representation is then used as input for the final classifier
(herbal prediction module).


4



For _S_ _e_, the process of aggregating symptoms to their first-order
neighbouring nodes can be defined as:



_Z_ _pe_ [1] [=][ ReLU] ~~_D_~~
(



1
2 _pe_ - _S_ _e_ - ~~_D_~~



1
2 _pe_ - _W_ _pe_ + _b_ _pe_



(3)
)



where _D_ _pe_ ∈ R _[t]_ [×] _[t ]_ is the degree matrix of _S_ _e_, _W_ _pe_ is the weight matrix, _b_ _pe_
is the bias matrix, and ReLu(⋅) is the non-linear activation function.
Similarly, the node aggregation process of graph _T_ _s_ can be defined as:



_Z_ _pt_ [1] [=][ ReLU] _D_
(



1 1
2 _pt_ - _T_ _s_ - ~~_D_~~ 2 _pt_ - _W_ _pt_ + _b_ _pt_



1
2 _pt_ - _T_ _s_ - ~~_D_~~



(4)
)



Significantly, feature aggregation can be further extended to multi­
ple layers. For the _n_ [th ] graph convolution layer, the aggregation process
can be defined as:



_Z_ _pe_ _[l]_ [=][ ReLU] ~~_D_~~
(



1
2 _pe_ - _S_ _e_ - ~~_D_~~



1
2 _pe_ - _Z_ _pe_ _[l]_ [−] [1] [*] _[ W]_ _[pe]_ [ +] _[ b]_ _[pe]_



(1 _< l_ ≤ _L_ ) (5)
)



_Z_ _pt_ _[l]_ [=][ ReLU] _D_
(



1 1
2 _pt_ - _T_ _s_ - ~~_D_~~ 2 _pt_ - _Z_ _pt_ _[l]_ [−] [1] [*] _[ W]_ _[pt]_ [ +] _[ b]_ _[pt]_



1
2 _pt_ - _T_ _s_ - ~~_D_~~



(1 _< l_ ≤ _L_ ) (6)
)



where _l_ is the current number of layers. In addition, removing normal­
ization operations on _S_ _e_ and _S_ _t_ is beneficial to performance improvement
when there is only one GCN layer. In this case, we define the entire
feature aggregation operation as:



_Z_ _pe_ _[L]_ [=][ ReLU] ( _S_ _e_ - _W_ _pe_ + _b_ _pe_ ) (7)



_Z_ _pt_ _[L]_ [=][ ReLU] ( _T_ _s_ - _W_ _pt_ + _b_ _pt_ ) (8)


Then, a fusion module is used to obtain comprehensive embedding
representations of the clinical data point:


_W. Zhao et al._ _Journal of Ethnopharmacology 297 (2022) 115109_


​

​


​ ​


​ ​ ​


​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​

​ ​ ​



_Z_ _ps_ = _D_ [−] _ps_ [1] [*] _[Cat]_ ( _Z_ _pe_ _[L]_ _[,][ Z]_ _pt_ _[L]_


​

​


​ ​


​ ​ ​


​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​

​ ​ ​



(9)
)


​

​


​ ​


​ ​ ​


​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​

​ ​ ​



where _D_ _ps_ ∈ R _[n]_ [×] _[n ]_ is a diagonal matrix where each non-zero value on the
diagonal stores the total number of symptoms in a clinical data point,
and _Cat_ ( ⋅) is a matrix connection operation.
For ease of understanding, the key symbols and their definitions are
listed in Table 1.


_3.2. Herb prediction module_


The MLP was applied to construct the herbal medicine prediction
module. This enabled the self-learning ability of the MGCN and
completed the recommendations process. In the feature-aggregation
module, the output of each sample is _Z_ _ps_, which is a new embedding
expression that incorporates symptoms, state-elements, and syndrome
types. In the herbal medicine prediction module, _Z_ _ps_ was used as input
for the MLP. Finally, a set of probability values were output corre­
sponding to the probability of each herb being recommended. The
specific steps are as follows:


(1) MLP classifier.


A two-layer MLP was used as our final multi-label classifier:


​

​


​ ​


​ ​ ​


​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​

​ ​ ​



) (10)


​

​


​ ​


​ ​ ​


​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​

​ ​ ​



the experimental evaluation and implementation methods. In addition,
an ablation study is presented to verify the validity of the different
components of the MGCN model. Finally, screening of the tunable pa­
rameters is discussed in detail to illustrate the effect of different pa­
rameters on the MGCN’s performance and outcomes.


_4.1. Datasets_


A total of 358 data items in the _Treatise on Febrile Diseases_ dataset

were used as samples. Each item comprises symptoms, state-elements,
syndrome-types, and herb information, as shown in Table 2. Specif­
ically, the input items of this dataset contain a total of 1329 symptoms,
1641 state-elements, and 358 syndrome-types, while the output items
contain 1627 herb data points. Due to the small sample size of the
dataset, a 5-fold cross-validation method was used to test the general­
izability of the algorithm. The data grouping is shown in Table 3.


_4.2. Evaluation methods_


The performance of MGCN was assessed with three metrics
commonly used to evaluate recommendations: Precision, Recall, and F1score. Precision@ _K_ is the precision rate of the MGCN, representing the
proportion of correct predictions made by the herb prediction algorithm,
which is a direct indication of its performance. Recall@ _K_ is the recall of
the MGCN, representing the proportion of herbs in the true labels that
are correctly predicted by the algorithm, indicating how well the correct

​ prediction labels cover all correct labels. The F1-score is the weighted
​ average of the two metrics, which provides a more objective represen­

tation of performance. The methods are defined as follows:


[(] _[ P]_ _pre, K_ ) ∩ _P_ _label_ |
Precision@K = [|] _[ To][p]_ (12)
_K_


[(] _[ P]_ _pre, K_ ) ∩ _P_ _label_ |
Recall@K = [|] _[ To][p]_ (13)
| _P_ _label_ |


F1 − score@K = [2][*][Precision][@][K][ ∗] [Recall][@][K] (14)
Precision@k ​ + ​ Recall@K


​ ​ ​


where _K_ is the number of herbs set by the experimental algorithm, which
was uniformly set to 5 or 10 in this study. _Top (P_pre, K)_ represents the
predicted _K_ herb set with the highest probability. _P_label_ represents the
herb set used to test comparisons in the original dataset; that is, the data
label. All indicators were positively correlated, with higher values
indicating better results.


_4.3. Implementation details_


This experiment was performed in the Python 3.6 environment with
the Pytorch 1.6 deep learning library. The computer comprised an Intel
Core i5-6300HQ processor, Nvidia GeForce GTX 960M graphics card,


​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​

​ ​ ​



_y_ _h_ = _sigmoid_ ( _W_ 2 ReLU( _W_ 1 - _Z_ _ps_ + _b_ 1


​

​


​ ​


​ ​ ​


​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​

​ ​ ​



) + _b_ 2


​

​


​ ​


​ ​ ​


​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​

​ ​ ​



where _W_ 1 and ​ _W_ 2 are the weight matrices of the first and second
layers, respectively, and _b_ 1 ​ and ​ _b_ 2 are the bias vectors of the first and
second layers, respectively. _y_ _h_ is the herbal probability vector.


(2) Multi-label cross-entropy loss function.


A loss function was set to calculate the gap between the model pre­
dictions and the actual output used to train the model. For the multilabel classification task, the most-used loss function is the crossentropy loss function:


​ ​


​ ​ ​


​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​

​ ​ ​



​

​


L = − _H_ 1 ​ ​ ​ ​ ​


​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​

​ ​ ​



​

​


​ ​

∑( _t_ _h_ log ​ y h + (1 − _t_ _h_ ) ​ log(1 − ​ y h )) (11)

_h_ ∈ _H_


​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​

​ ​ ​



​

​


​ ​


​ ​ ​


where _H_ represents the number of neurons in the output layer; that is,
the number of types of herbs. t h = (t h {0,1}) and y h (0 ≤ y h ≤ 1) represent
the actual labels and the predicted values of the model, respectively.


**4. Experimental results**


To verify the effectiveness of the MGCN in making herbal recom­
mendations, the model output was compared with those of two nontopic model algorithms (SVM and LR) and a topic model algorithm
(PTM). The _Treatise on Febrile Diseases_ dataset was chosen for experi­
mental learning and training. This section introduces that dataset and


**Table 1**
Symbol definitions.

Symbol Definition


​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​

​ ​ ​



​

​


​ ​


​ ​ ​


_X_ _ps_ Symptom matrix containing all symptoms of the clinical dataset
_T_ _s_ _,_ _S_ _e_ Symptom-‘syndrome-type’-symptom graph, and symptom-‘state-element’-symptom graph
_D_ _t_ _,_ _D_ _e_ Degree matrix of _T_ _s_ and _S_ _e_
_Z_ _[n]_ _pt_ _[,]_ _[Z]_ _[n]_ _pe_ The ​ node ​ feature ​ representation ​ of ​ the ​ output ​ of ​ the ​ _n_ [th] ​ layer ​ graph ​ convolution ​ layer ​ of ​ _T_ _s_ and _S_ _e_ (n = 1 _,_ 2 _,..._ )
_D_ _ps_ Diagonal matrix containing the number of symptoms per clinical data point
_W_ _pt_ _,_ _b_ _pt_ Weight and bias matrix of graph convolution layer in _T_ _s_
_W_ _pe_ _,_ _b_ _pe_ Weight and bias matrix of graph convolution layer in _S_ _e_
_Z_ _pt_ _,_ _Z_ _pe_ Final ​ prescription ​ feature ​ representation obtained from _T_ _s_ and _S_ _e_
_Z_ _ps_ Clinical data point feature representation after fusion
_W_ _i_ _,_ _b_ _i_ The MLP weight and bias matrix in layer _i_ (i = 1,2)
_y_ _h_ Herbal probability vector
_N_ Threshold of _S_ _e_


5


_W. Zhao et al._ _Journal of Ethnopharmacology 297 (2022) 115109_


**Table 2**

_Treatise on Febrile Diseases_ dataset.


Symptoms State elements TCM syndrome Herbs Prescription



泄泻 [Diarrhoea],
腹痛 [Stomach ache],
手足冷 [Cold limbs]



肠 [Intestines],
肝 [Liver],
气滞 [ _Qi_ stagnation],
寒 [Cold]



肝气郁滞 [Syndrome of liver _Qi_ stagnation] 柴胡 [ _Bupleurum falcatum_ subsp. _Cernuum_ (Ten.) Arcang.],
枳实 [ _Solanum truncicola_ Bitter],
芍药 [ _Paeonia lactiflora_ Pall.],
炙甘草 [ _Cucumis melo_ var. _honey-dew_ Hassib]



四逆散 [ _Sini San_ ]



**Table 3**

Statistical evaluation of the dataset.


Dataset Symptoms State TCM Herbs Prescriptions
elements syndromes


All 1329 1641 358 1627 112

Training 1063 1313 286 1302 90
Testing 266 328 72 325 22


and Windows 10 operating system.
Merit-based screening of all tunable parameters was performed to
ensure the accuracy of the experiments. Due to the many tunable pa­
rameters involved in the MGCN, the selection process is described in
detail in Section 5. Considering the experimental results under different
parameters, the optimal number of hidden layers containing GCN
modules was set to 600. The optimal number of hidden layers for the
MLP module was set to 800. For the feature extraction module, the ideal
set value of _N_ in the _Se_ graph was 3, and the ideal number of graph
convolutional layers was 1. Notably, the optimizer chosen for this study
was the Adam optimizer (Kingma and Ba, 2017). The learning rate lr was
set to 2.5 × 10 [−] [4], the L2 regularisation coefficient was set to 10 [−] [4], and
the dropout ratio was set to 0.2. In addition, for the tuning parameters of
the three compared algorithms, (1) the kernel function of the SVM was
set as a Gaussian kernel, the kernel coefficient = 0.3, and the regular­
isation strength = 3; (2) For LR, the regularisation strength was set to 3,
and (3) for PTM, _K_ = 30, _α_ = 0.2, and _β_ = 0.1.


_4.4. Quantitative experimental results_


Table 4 shows the experimental results of the developed model and
the three algorithms used for comparison. The MGCN outperformed the
three algorithms in the accuracy, Recall, and F1-score evaluation met­
rics. Specifically, the MGCN was 62.70% at P@5, 39.44% at P@10,
73.39% at R@5, 87.68% at R@10, 67.52% at F@5, and 54.33% at
F@10. All quantitative data exploring the MGCN outperformed the three
comparative algorithms. In addition, the experimental results showed
that the SVM had the best performance of the three algorithms. The
MGCN was 4.51%, 6.45%, and 5.31% higher than the SVM in terms of
P@5, R@5, and F@5, respectively. This quantitative experiment shows
that the MGCN provides markedly improved accuracy in making TCM
prescription recommendations.


_4.5. Qualitative experimental results_


Table 5 shows two cases of the MGCN where _K_ = 5. The model

correctly predicted the herbs that are underlined and in bold type. In the
first case, all five herbs in the label were correctly predicted by the
MGCN. In the second case, four of the five herbs were correctly pre­
dicted. There was one misreported herb: Cucumis melo var. honey-dew


**Table 4**

Comparison of experimental results using different baselines.


Algorithm P@5 P@10 R@5 R@10 F@5 F@10


SVM 0.5819 0.3717 0.6694 0.8148 0.6221 0.5103

LR 0.5456 0.3661 0.6302 0.8092 0.5844 0.5039

PTM 0.4450 0.3113 0.4832 0.6694 0.4630 0.4248

MGCN **0.6270** **0.3943** **0.7339** **0.8768** **0.6752** **0.5433**



Hassib. Although this herb was misreported, it could still play an adju­
vant role and would not affect the overall treatment outcome. These

results demonstrate that the MGCN can provide a more accurate scheme
for herbal recommendations than existing methods and has substantial
practical value.


_4.6. Ablation analysis_


The MGCN model was divided into three components, a StateElement Graph Convolutional Network (S e GCN), TCM Syndrome-type
Graph Convolutional Network (T s GCN), and MLP. Their contributions
to the recommendation model were evaluated. Component 1 uses only
state-elements ( _S_ _e_ ) for graph convolution to extract features, and
component 2 uses only syndrome-types ( _T_ _s_ ) for graph convolution
feature extraction. Component 3, or MLP, is not embedded in the graph
convolution module but is directly mapped from symptoms to pre­
scriptions. The model containing all three components simultaneously is
the MGCN. The experimental results are illustrated in Table 6.
According to the quantitative experimental data, the MGCN, S e GCN
and T s GCN gave better results than the MLP, to various degrees. Hence,
the data shows that graph convolution performs better in terms of Pre­
cision, Recall, and F1 scores. This demonstrates that the embedding
graph convolution module plays a significant role in the MGCN and all of
its components are effective. Furthermore, the experimental effect of
T s GCN is slightly better than that of S e GCN, which means that the
contribution of syndrome-types is slightly greater than that of stateelements. However, both undoubtedly play positive roles to various
degrees. The experimental results of MGCN are better than those of
T s GCN and S e GCN, thus further validating the importance of embedding
syndrome-types and state-elements within the model.


_4.7. Influence of hyperparameters_


(1) Effect of GCN layers


The current study used different GCN layers (1, 2 or 3) to investigate
whether the MGCN utilizes a deeper GCN. Given that the MLP does not
involve a GCN, only MGCN, S e GCN and S t GCN were tested. The results
are shown in Fig. 3.
One characteristic of GCN is that its nodes can collect information
from neighbouring nodes. The first layer collects the information of the
first-order neighbours and the second layer does so for the second-order
neighbours. The neighbouring relationship defined in this study is
determined by whether two symptoms emerge concurrently with the
state-element and syndrome-type, which is a first-order relationship.
The experimental results also show that the MGCN performance de­
creases as the number of GCN layers increases. Therefore, a GCN layer
setting of 1 is the optimal parameter for the MGCN.


(2) Effect of the number of MLP hidden layers


The model was tested using various MLP hidden layer nodes (100,
200, 300, 400, 500, 600, 700, 800, 900 or 1000) to explore the effect on
MGCN performance. The results are shown in Fig. 4.
The number of nodes in the hidden layer of the MLP determines the
complexity of the mapping relationship that can be fitted which, in this
study, refers to the symptom-herb mapping relationship. Notably, if the



6


_W. Zhao et al._ _Journal of Ethnopharmacology 297 (2022) 115109_


**Table 5**

Herbal recommendations made by the MGCN.


Symptom State element TCM syndrome Recommended herbs


MGCN Ground-truth



腹痛 [Stomach ache],
腹胀 [Abdominal

distension],
肠鸣 [Borborygmus],
干呕 [Vomiting],
心烦 [Upsetting]



胃 [Stomach],
心 [Heart],
气虚 [ _Qi_ deficiency],
气逆 [Reversed flow
of _Qi_ ],
湿 [Wet],
热 [Hot]



腹痛 [Stomach ache], 胃 [Stomach], 中虚湿热痞重证 **半夏** [ _**Pinellia ternata**_ **(Thunb.)** **黄连** **[** _**Coptis chinensis**_ **Franch.]**,
腹胀 [Abdominal 心 [Heart], and hot syndromes] **Makino** ], **半夏** [ _**Pinellia ternata**_ **(Thunb.)**
distension], 气虚 [ _Qi_ deficiency], **大枣** **[** _**Ziziphus jujuba**_ **var.** _**Jujuba**_ **]**, **Makino** ],
肠鸣 [Borborygmus], 气逆 [Reversed flow **人参** **[** _**Aralia ginseng**_ **(C.A.Mey.)** **人参** **[** _**Aralia ginseng**_ **(C.A.Mey.)**
干呕 [Vomiting], of _Qi_ ], **Baill.]**, **Baill.]**,
心烦 [Upsetting] 湿 [Wet], **炙甘草** **[** _**Cucumis melo**_ **var.** _**honey-**_ **炙甘草** **[** _**Cucumis melo**_ **var.** _**honey-**_

热 [Hot] _**dew**_ **Hassib]**, _**dew**_ **Hassib]**,

**黄连** **[** _**Coptis chinensis**_ **Franch.]** **大枣** **[** _**Ziziphus jujuba**_ **var.** _**Jujuba**_ **]**

骨节疼痛 [Joint pain], 心 [Heart], 寒湿体痛证 [Bodily cold wet and pain **炮附子** **[** _**Aconitum uncinatum**_ **L.]**, **芍药** **[** _**Paeonia lactiflora**_ **Pall.]**,
身痛 [Bodily pain], 肾 [Kidney], syndromes] **白术** **[** _**Atractylodes macrocephala**_ **人参** **[** _**Aralia ginseng**_ **(C.A.Mey.)**
手足冷 [Cold limbs], 阳虚 [ _Yang_ **Koidz.]**, **Baill.]**,
脉沉 [Heavy pulse] deficiency], **芍药** **[** _**Paeonia lactiflora**_ **Pall.]**, 茯苓 [ _Salix polia_ C.K.Schneid.],



中虚湿热痞重证 [Serious deficiency wet **半夏** [ _**Pinellia ternata**_ **(Thunb.)**
and hot syndromes] **Makino** ],
**大枣** **[** _**Ziziphus jujuba**_ **var.** _**Jujuba**_ **]**,
**人参** **[** _**Aralia ginseng**_ **(C.A.Mey.)**
**Baill.]**,
**炙甘草** **[** _**Cucumis melo**_ **var.** _**honey-**_
_**dew**_ **Hassib]**,
**黄连** **[** _**Coptis chinensis**_ **Franch.]**

寒湿体痛证 [Bodily cold wet and pain **炮附子** **[** _**Aconitum uncinatum**_ **L.]**,
syndromes] **白术** **[** _**Atractylodes macrocephala**_
**Koidz.]**,
**芍药** **[** _**Paeonia lactiflora**_ **Pall.]**,
**人参** **[** _**Aralia ginseng**_ **(C.A.Mey.)**
**Baill.]**,
炙甘草 [ _Cucumis melo_ var. _honey-dew_
Hassib]



心 [Heart],
肾 [Kidney],
阳虚 [ _Yang_
deficiency],
寒 [Cold],
湿 [Wet],
关节 [Joint]



**芍药** **[** _**Paeonia lactiflora**_ **Pall.]**,
**人参** **[** _**Aralia ginseng**_ **(C.A.Mey.)**
**Baill.]**,
茯苓 [ _Salix polia_ C.K.Schneid.],
**白术** **[** _**Atractylodes macrocephala**_
**Koidz.]**,
**炮附子** **[** _**Aconitum uncinatum**_ **L.]**



**Table 6**

Comparison of the experimental results of different sub-models.


Method P@5 P@10 R@5 R@10 F@5 F@10


MGCN **0.6270** **0.3943** **0.7339** **0.8768** **0.6752** **0.5433**

T s GCN 0.6107 0.3850 0.7163 0.8578 0.6584 0.5309

S e GCN 0.6050 0.3847 0.7078 0.8552 0.6515 0.5300

MLP 0.5803 0.3753 0.6776 0.8367 0.6247 0.5178



number of nodes in the hidden layer is too low, the problem of under­
fitting is likely to occur. If the number of nodes in the hidden layer is too
high, the model’s generalizability will likely be reduced. Therefore,
ideal results are obtained when the number of hidden layer nodes is
moderate. The experimental results show that the ideal number of nodes
for both the MGCN- and GCN-embedded sub-model is 600, while the
ideal number of nodes for the MLP sub-model (without embedded GCN)
is 800. In addition, the difference between the results with 800 and 600

nodes shows that GCN feature extraction can make new features more



**Fig. 3.** Effects of GCN layers (1, 2, and 3).


7


_W. Zhao et al._ _Journal of Ethnopharmacology 297 (2022) 115109_


**Fig. 4.** Effects of varying the number of nodes in the MLP module’s hidden layer.



distinguishable, thereby reducing the complexity of the mapping rela­
tionship. In other words, after embedding the GCN, a relatively low
number of hidden layer nodes can be used to fit a more complex map­
ping relationship, which further demonstrates the effectiveness of
embedded GCN components in the model.


(3) Effect of the _S_ _e_ graph’s _N_ -value



In the embedded state-element induction module, the number of
state-elements that are needed to accurately determine the correlation
between two symptoms needs to be ascertained. A hyperparameter _N_ in
the _Se_ graph is established to determine this correlation. Since there are,
at most, five state-elements that can appear simultaneously in the two
clinical datasets, _N_ is set to {1, 2, 3, 4 and 5}. The outcomes were
assessed in S e GCN and MGCN to determine the best _N_ -value and the



**Fig. 5.** Effects of N-values.


8


_W. Zhao et al._ _Journal of Ethnopharmacology 297 (2022) 115109_



results are shown in Fig. 5.
When _N_ = 3, the MGCN and its sub-method S e GCN achieve the best
results, indicating that there is an optimal value of _N_ . After several
comparisons, the correlation between the two symptoms can be deter­
mined more accurately based on three concurrent state-elements.


**5. Conclusions**


To improve the accuracy of the TCM prescription recommendation
model, a PTM and GCN-inspired model called MGCN was proposed and
developed. Unlike existing models, state-elements and syndrome-types
were incorporated to simulate an _n_ -ary relationship between symp­
toms, state-elements, syndrome-types, and herbs. On the one hand, in
the MGCN feature-aggregation module, features were extracted by
constructing a symptom-‘state-element’-symptom graph ( _S_ _e_ ) and a
symptom-‘syndrome-type’-symptom graph ( _T_ _s_ ) for convolution to fully
explore the potential correlations among symptoms, state-elements, and
syndrome-types. On the other hand, in the MGCN herbal medicine
prediction module, the fused state-element, syndrome-type, and symp­
tom information was used as input and applied to the MLP to predict the
corresponding TCM prescriptions in the form of probability values.
The _Treatise on Febrile Diseases_ dataset was chosen for experimental
testing to assess the performance of the MGCN. The dataset contains
1329 symptoms, 1641 state-elements, 358 syndrome-types, and 1627
herbal medicines. The results indicate that, compared with SVM, MGCN
is 4.51%, 6.45% and 5.31% higher in P@5, R@5 and F@5. Hence, it
significantly improves the accuracy of TCM prescription recommenda­
tions. In the future, a TCM prescription recommendation model based on
a hypergraph will be developed to further explore the potential for
integration between TCM diagnosis and treatment and computer
algorithms.


**Author contribution statement**


The study’s overall design was made by **Candong Li**, **Wen Zhao**,
**Haoyi Fan** and **Zuoyong Li** . The _Treatise on Febrile Diseases_ dataset was
collected and analysed by **Wen Zhao**, **Weikai Lu** and **Chang’en Zhou** .
Quantitative and qualitative experimental analyses were performed by
**Weikai Lu**, **Chang’en Zhou** and **Haoyi Fan** . **Wen Zhao** and **Weikai Lu**
wrote the manuscript, which was reviewed and revised by **Zhaoyang**
**Yang**, **Xuejuan Lin** and **Candong Li** .


**Declaration of competing interest**


The authors declare no conflicts of interest.


**Funding statement**


This work was partially supported by the National Natural Science
Foundation Joint Fund Project [No. U1705286], the Pre-research Proj­
ect in Manned Space Field [No. 020104], a research project of the School
of Management, Fujian University of Chinese Medicine [No. X2021018],
the National Natural Science Foundation of China [No. 61972187] and
the Natural Science Foundation of Fujian Province [No. 2020J02024].


**Appendix A. Supplementary data**


[Supplementary data to this article can be found online at https://doi.](https://doi.org/10.1016/j.jep.2022.115109)
[org/10.1016/j.jep.2022.115109.](https://doi.org/10.1016/j.jep.2022.115109)



**References**


[Blei, D., Ng, A., Jordan, M., 2003. Latent dirichlet allocation. J. Mach. Learn. Res. 3,](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref1)
[993–1022.](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref1)

[Chang, J., Gao, C., He, X., Jin, D., Li, Y., 2020. Bundle recommendation with graph](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref2)
[convolutional networks. Proc. ACM/SIGIR Conf. Res. Develop. Inform. Retri.](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref2)
[1673–1676.](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref2)

[Chang, C., Lin, C., 2011. LIBSVM: a library for support vector machines. ACM Transact.](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref3)
[Intellig. Sys. Technol.(TIST) 2 (3), 1–27.](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref3)
Hamilton, W., Ying, R., Leskovec, J., 2018. Inductive representation learning on large
[graphs. Available from: http://arxiv.org/abs/1706.02216. (Accessed 10 September](http://arxiv.org/abs/1706.02216)
2018).
[Hosmer, D., 2013. Applied logistic regression. In: Lemeshow, S., Sturdivant, R.X. (Eds.),](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref5)
[Introduction to the Logistic Regression Model. Wiley, New York, NY, USA, pp. 1–7.](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref5)
[Jin, B., Gao, C., He, X., Jin, D., Li, Y., 2020a. Multi-behavior recommendation with graph](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref6)
[convolutional networks. In: Proc. ACM/SIGIR Conference on Research and](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref6)
[Development in Information Retrieval, pp. 659–668.](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref6)
[Jin, Y., Zhang, W., He, X., Wang, X., Wang, X., 2020b. Syndrome-aware herb](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref7)
[recommendation with multi-graph convolution network. In: Proc. 2020 IEEE 36th](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref7)
[International Conference on Data Engineering. ICDE), pp. 145–156.](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref7)
[Johnson, J., Gupta, A., Li, F., 2018. Image generation from scene graphs. In: Proc. CVF/](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref8)
[IEEE Conference on Computer Vision and Pattern Recognition. CVPR),](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref8)
[pp. 1219–1228.](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref8)
Kingma, D., Ba, J., 2017. Adam: a method for stochastic optimization. Available from:

[http://arxiv.org/abs/1412.6980. (Accessed 30 January 2017).](http://arxiv.org/abs/1412.6980)
Kipf, T., Welling, M., 2017. Semi-supervised classification with graph convolutional
[networks. Available from: http://arxiv.org/abs/1609, 02907. Accessed: 22 February](http://arxiv.org/abs/1609)
2017.

Koncel-Kedziorski, R., Bekal, D., Luan, Y., Lapata, M., Hajishirzi, H., 2019. Text
[generation from knowledge graphs with graph transformers. Available from: htt](http://arxiv.org/abs/1904.02342)
[p://arxiv.org/abs/1904.02342. (Accessed 18 May 2019).](http://arxiv.org/abs/1904.02342)
Li, W., Sun, X., Ren, X., Yang, Z., 2018. Exploration on generating traditional Chinese
medicine prescription from symptoms with an end-to-end method. Available from:
[http://arxiv.org/abs/1801.09030. (Accessed 21 May 2018).](http://arxiv.org/abs/1801.09030)
Li, W., Yang, Z., 2017. Distributed representation for traditional Chinese medicine herb
[via deep learning models. Available from: http://arxiv.org/abs/1711.01701.](http://arxiv.org/abs/1711.01701)
(Accessed 6 November 2017).
[Ma, J., Zhang, Y., 2018. Using topic models for intelligent computer-aided diagnosis in](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref14)
[traditional Chinese medicine. In: Proc. International Conference on Intelligent](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref14)
[Networking and Collaborative Systems (INCoS), pp. 504–507.](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref14)
[Ruan, C., Wang, Y., Zhang, Y., Ma, J., Chen, H., Aickelin, U., Zhu, S., Zhang, T., 2017.](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref15)
[THCluster: herb supplements categorization for precision traditional Chinese](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref15)
[medicine. In: Proc. 2017 IEEE International Conference on Bioinformatics and](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref15)

[Biomedicine (BIBM), pp. 417–424.](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref15)
[Ruan, C., Wang, Y., Zhang, Y., Yang, Y., 2019. Exploring regularity in traditional Chinese](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref16)
[medicine clinical data using heterogeneous weighted networks embedding. In:](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref16)
[International Conference on Database Systems for Advanced Applications. Springer,](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref16)
[VeliˇckoviCham, pp. 310´c, P., Cucurull, G., Casanova, A., Romero, A., Li–313.](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref16) o, P., Bengio, Y., 2018. Graph `
[attention networks. Available from: http://arxiv.org/abs/1710.10903. (Accessed 4](http://arxiv.org/abs/1710.10903)
February 2018).
[Wang, X., Zhang, Y., Wang, X., Chen, J., 2019. A knowledge graph enhanced topic](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref18)
[modeling approach for herb recommendation. In: International Conference on](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref18)
[Database Systems for Advanced Applications. Springer, Cham, pp. 709–724.](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref18)
[Wang, Y., Xu, A., 2014. Zheng: a systems biology approach to diagnosis and treatments.](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref19)
[Science 346 (6216), S13–S15.](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref19)
[Wang, Z., Li, L., Yan, J., Yao, Y., 2020. Approaching high-accuracy side effect prediction](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref20)
[of traditional Chinese medicine compound prescription using network embedding](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref20)
[and deep learning. IEEE Access 2020 (8), 82493–82499.](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref20)
[Yao, L., Zhang, Y., Wei, B., Zhang, W., Jin, Z., 2018. A topic modeling approach for](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref21)
[traditional Chinese medicine prescriptions. Proc. IEEE Trans. Knowl. Data Eng. 30](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref21)
[(6), 1007–1021.](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref21)
[Yao, L., Zhang, Y., Wei, B., Wang, W., Zhang, Y., Ren, X., Bian, Y., 2015. Discovering](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref22)
[treatment patterns in Traditional Chinese Medicine clinical cases by exploiting](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref22)
[supervised topic model and domain knowledge. J. Biomed. Inf. 58, 260–267.](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref22)
[Zhao, G., Zhuang, X., Wang, X., Ning, W., Li, Z., Wang, J., Chen, Q., Mo, Z., Chen, B.,](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref23)
[Chen, H., 2018. Data-driven traditional Chinese medicine clinical herb modeling and](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref23)
[herb pair recommendation. Proc. IEEE Intern. Conf. Dig. Home (ICDH) 160–166.](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref23)
[Zhao, W., Lin, X., Min, L., Wang, Y., Wang, F., Li, C., 2020. Connotation and extension of](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref24)
[TCM thoughts. Chin. J. Tradit. Chin. Med. Pharm. 35 (1), 46–49 [in Chinese].](http://refhub.elsevier.com/S0378-8741(22)00147-7/sref24)
Zhou, W., Yang, K., Zeng, J., Lai, X., Wang, X., Ji, C., Li, Y., Zhang, P., Li, S., 2021.
FordNet: recommending traditional Chinese medicine formula via deep neural
[network integrating phenotype and molecule. Available from: https://doi.org/10.10](https://doi.org/10.1016/j.phrs.2021.105752)
[16/j.phrs.2021.105752. (Accessed 2 September 2021).](https://doi.org/10.1016/j.phrs.2021.105752)



9


