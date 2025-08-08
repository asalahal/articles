_Briefings in Bioinformatics_, 2022, **23**, 1–11


**https://doi.org/10.1093/bib/bbab511**

**Problem Solving Protocol**

# **Predicting drug–drug interactions by graph** **convolutional network with multi-kernel**


Fei Wang, Xiujuan Lei, Bo Liao and Fang-Xiang Wu


Corresponding author: Fang-Xiang Wu, 3B42, 57 Campus Drive, Saskatoon, S7N 5A9, Canada. Tel: ++(306)966-5280; Fax: +1(306)966-5280;
E-mail: fangxiang.wu@usask.ca


Abstract


Drug repositioning is proposed to find novel usages for existing drugs. Among many types of drug repositioning approaches, predicting
drug–drug interactions (DDIs) helps explore the pharmacological functions of drugs and achieves potential drugs for novel treatments.
A number of models have been applied to predict DDIs. The DDI network, which is constructed from the known DDIs, is a common
part in many of the existing methods. However, the functions of DDIs are different, and thus integrating them in a single DDI graph
may overlook some useful information. We propose a graph convolutional network with multi-kernel (GCNMK) to predict potential
DDIs. GCNMK adopts two DDI graph kernels for the graph convolutional layers, namely, increased DDI graph consisting of ‘increase’related DDIs and decreased DDI graph consisting of ‘decrease’-related DDIs. The learned drug features are fed into a block with
three fully connected layers for the DDI prediction. We compare various types of drug features, whereas the target feature of drugs
outperforms all other types of features and their concatenated features. In comparison with three different DDI prediction methods,
our proposed GCNMK achieves the best performance in terms of area under receiver operating characteristic curve and area under
precision-recall curve. In case studies, we identify the top 20 potential DDIs from all unknown DDIs, and the top 10 potential DDIs from
the unknown DDIs among breast, colorectal and lung neoplasms-related drugs. Most of them have evidence to support the existence
of their interactions. fangxiang.wu@usask.ca


Keywords: graph convolutional network, drug–drug interaction, drug features, drug repositioning



Introduction


Drug repositioning is to find novel usages for existing
drugs. The safety and other properties of the existing drugs, which have been approved to sell on the
market, have been studied clearly. Therefore, drug
repositioning helps save time and reduce costs of drug
development greatly. Several successful drugs have been
proposed by drug repositioning approaches, such as
sildenafil, thalidomide, zidovudine, minoxidil and celecoxib [1].

In order to increase the prediction efficiency, many
computational approaches have been utilized to predict
potential drugs for different diseases. A main field is
predicting potential links between drugs and related elements, such as drug–disease associations (DDAs) [2–7],
drug–target interactions [8–13] and drug–drug interactions (DDIs) [14–21]. When predicting DDAs, Luo _et al._ calculated similarities and constructed a similarity network

[2, 3]. Random walk was employed to calculate the probabilities of DDAs. Li _et al._ utilized a convolutional neural



network (CNN) model to conduct a binary classification
of DDAs, based on the known DDAs and drug/disease
feature vectors [5]. In the study of DTI, deep learning (DL)
approaches are effective tools to predict potential DTIs.
Wen _et al._ constructed a deep-belief network to predict
potential DTIs [9]. Monteiro _et al._ combined a CNN with
a deep neural network (DNN) to make predictions, where
the CNN was used to produce novel representations of
feature vectors and the DNN was employed to predict
DTIs [12].

The DDIs refer to the pharmacological and clinical
responses to a drug combination, different from the
known effects of two drugs when used alone. The
prediction of DDIs helps researchers to have a deep
understanding of the mechanisms of actions of drugs.
In order to analyze DDIs, various types of drug features
have been studied, such as chemical substructures, side
effects, targets, pathways and enzymes, etc.

A number of approaches have been proposed to predict DDIs based on one or more types of drug features.
Ferdousi _et al._ calculated drug–drug similarities based



**Fei Wang** is current a PhD candidate in the Division of Biomedical Engineering at the University of Saskatchewan. His research topic is bioinformatics,
machine/deep learning and drug repositioning.
**Xiujuan Lei** has been a professor in Shaanxi Normal University since 2013. Her research interests include bioinformatics, data mining, swarm intelligence
computing and machine/deep learning.
**Bo Liao** is currently working in Hainan Normal University as a professor. His current research interests include bioinformatics, data mining and machine/deep
learning.
**Fang-Xiang Wu** is currently a full professor in three Departments (Computer Science, Biomedical Engineering and Mechanical Engineering) at the University of
Saskatchewan. His research interests include artificial intelligence, machine/deep learning, computational biology and bioinformatics, medical image analytics,
complex network analytics. He is an IEEE senior member.
**Received:** September 14, 2021. **Revised:** October 28, 2021. **Accepted:** November 7, 2021
© The Author(s) 2021. Published by Oxford University Press. All rights reserved. For Permissions, please email: journals.permissions@oup.com


2 | _Wang_ et al.


Figure 1. The architecture of GCNMK. **A** : Constructing two DDI graphs from increased, decreased interactions, and inputting drug attributes.
**B** : Generating the feature representation of drugs by GCN. **C** : Predicting DDIs.



on various types of features and utilized a positive similarity threshold to determine the potential DDIs [14].
However, the similarities of many DDIs are negative,
while they cannot be predicted by a constant positive
value. Yan _et al._ used a _k_ -nearest neighbor procedure after



generating similarities of known DDIs and employed a
regularized least squares classifier to predict potential
DDIs [15]. In the classifier, both positive samples and negative samples are essential. In predicting potential DDIs,
the positive samples are those known DDIs, whereas the


negative samples are the unknown DDIs. Zheng _et al._
used an SVM model to produce reliable negative samples
from the unknown samples and made a further prediction [16]. Zhang _et al._ proposed a multi-modal autoencoder (MDAE) with positive-unlabeled learning to predict
potential DDIs [21].

The DDIs can be utilized to construct a DDI graph,
where nodes are drugs and edges are interactions among
drugs. Zhou _et al._ used a Markov clustering algorithm
on the DDI graph to predict potential drug combinations

[17]. Additionally, researchers can combine the drug features with the network structures to predict potential
interactions. Zhang _et al._ used a random walk algorithm
on the DDI graph [18], while the transition probabilities
were based on the drug–drug similarity matrices.

Graph convolutional network (GCN) [22] is a variant
of CNN on the graph, while the graph is used as a kernel. Researchers utilize GCN to produce low-dimensional
representation vectors of drugs by learning topological
structures of drugs in the DDI graph. Feng _et al._ combined
GCN with a deep neural network to generate feature
representation matrix and predict potential DDIs [19].
Huang _et al._ added a skip graph to reflect the indirect connections in the original DDI graph and made predictions
based on both the original DDI graph and the skip graph

[20].

In many DDI prediction methods, researchers do not
distinguish the responses of DDIs. All known DDIs are
labeled as positive samples and used to construct the DDI
graph. However, there are many types of DDIs relating to
various mechanisms. About half of them are ‘increase’
related, such as ‘DRUG A may increase the activities of
DRUG B’, another half of them are ‘decrease’-related,
such as ‘The metabolism of DRUG A can be decreased

when combined with DRUG B’.


In this work, we aim to learn novel embeddings from
those two types of DDIs. As discussed above, GCN is
an effective structure to utilize both DDI graphs and
drug feature vectors. We propose a graph convolutional
network with multi-kernel (GCNMK) to predict potential
increased DDIs. We firstly construct an increased DDI
graph and a decreased DDI graph from the increase’related and ‘decrease’-related DDIs, respectively. Two
GCN layers are combined to learn low-dimensional
representation vectors of drugs with those two graphs
and various types of drug features. After generating the
node embeddings, two drug vectors are concatenated
to be the vector of a DDI. Finally, a block with three
fully connected layers is used to make predictions.
In the experiments, we investigate the prediction
performance of our proposed model on various types
of drug features, including chemical substructures, side
effects, targets, pathways and enzymes, etc. We compare
three state-of-the-art methods with our GCNMK. The

results demonstrate that our GCNMK outperforms other
competing methods in predicting potential DDIs. In case
studies, we predict potential DDIs, and most of them have
evidence to support the existence of their interactions.



_GCNMK_ | 3


Methods


In this section, we introduce the architecture of our
GCNMK model, as shown in Figure 1. In Figure 1 **A**, an
increased DDI graph and a decreased DDI graph are
constructed from the ‘increase’-related and ‘decrease’
related DDIs, respectively. The two graphs and drug feature matrices are fed into two GCN blocks, respectively.
In Figure 1 **B**, these two GCN blocks form the GCN layer
_L_ 1, whereas layer _L_ 2 contains the third block. An addition
procedure, whose output is a linear combination of its
inputs, is adopted in each block to generate drug embeddings from both increased and decreased DDI graphs.The
low-dimensional representation vectors of drugs are produced after the layer _L_ 2 . In Figure 1 **C**, the feature vectors
of two drugs are concatenated to form a DDI vector. A
block with three fully connected layers is employed to
predict potential DDIs.


**DDI graphs and drug feature matrix**

A DDI graph _G_ = _(V_, _E)_ represents a collection of _n_ nodes
and _m_ edges, while nodes are drugs and edges are DDIs,
which is described by an association matrix _A_ . The DDI
refers to the pharmacological and clinical responses to
a drug combination, different from the known effects of
two drugs when used alone. If there is a known response
between drugs _i_ and _j_, in the association matrix _A_, _A(i_, _j)_ =
1. Otherwise, _A(i_, _j)_ = 0. The DDI graph is undirected, that
is, _A(i_, _j)_ = _A(j_, _i)_ .

There are various types of responses between two
drugs, including analgesic activity, risk or severity of
heart failure, serum concentration, therapeutic efficacy,
etc. We divide them into two groups. One group contains
DDIs that increase one of the responses, whereas another
group contains DDIs that decrease one of the responses.
Two DDI graphs _G_ _I_ and _G_ _D_ are constructed based on
those two groups of DDIs, respectively. Their association
matrices are denoted by _A_ _I_ and _A_ _D_ .

Another matrix is the drug feature matrix _H_ [0] . In order
to make a distinction, the feature matrix together with
the graph _G_ _I_ is marked as _H_ _[i]_ _I_ [, whereas the other one is]
_H_ _[i]_ _D_ [, at the] _[ i]_ [-th layer of GCNs.]


**Feature representations of drugs**

In this study, we construct two DDI graphs _G_ _I_ and _G_ _D_
for the increased and decreased DDIs, respectively. Our
purpose is to use GCN layers to learn features from both
two graphs. In layer _L_ 1, two blocks are adopted, each has
an input graph, as shown in Figure 1 **B** . The propagation
rules of linear transformation are as follows:


_H_ [1] _II_ [=] _[ F]_ _[I]_ _[H]_ [0] _I_ _[W]_ _I_ [0] (1)


_H_ [1] _ID_ [=] _[ F]_ _[I]_ _[H]_ [0] _I_ _[W]_ [′][0] _I_ (2)


_H_ [1] _DD_ [=] _[ F]_ _[D]_ _[H]_ [0] _D_ _[W]_ _D_ [0] (3)


_H_ [1] _DI_ [=] _[ F]_ _[D]_ _[H]_ [0] _D_ _[W]_ [′][0] _D_ (4)


4 | _Wang_ et al.


where _H_ [1] _II_ [and] _[ H]_ [1] _DD_ [are the node embedding matrices]
transferring within each block, respectively. _H_ [1] _ID_ [and] _[ H]_ [1] _DI_
transferring between the two blocks in layer _L_ 1 . _F_ _I_ =



In order to prevent the over-fitting problem, an _L_ 2  regularization is adopted:



� _D_ − [1] 2



− _I_ [1] 2 _A_ � _I_ � _D_ − _I_ [1] 2



− [1] − [1]

_I_ 2, _F_ _D_ = [�] _D_ _D_ 2



− [1] − [1]

2 2
_D_ _[A]_ [�] _[D]_ [�] _[D]_ _D_



� _w_ [2] (9)


_w_



� _D_ − _I_ 2 _A_ � _I_ � _D_ − _I_ 2, _F_ _D_ = [�] _D_ − _D_ 2 _[A]_ [�] _[D]_ [�] _[D]_ − _D_ 2 [.][ �] _[A]_ _[I]_ [ =] _[ A]_ _[I]_ [+] _[I]_ [ and][ �] _[A]_ _[D]_ [ =] _[ A]_ _[D]_ [+] _[I]_ [ are]

the association matrices of the graph _G_ _I_ and _G_ _D_, respec� _D_ tively. _D_ _(i_, _i) I_ = is the identity matrix. [�] _j_ _[A]_ [�] _[D]_ _[(][i]_ [,] _[ j][)]_ [ are the degree diagonal matrices.][�] _D_ _I_ _(i_, _i)_ = [�] _j_ _[A]_ [�] _[I]_ _[(][i]_ [,] _[ j][)]_ [ and] _[W]_ _I_ [0] [,]

_W_ [′][0] _I_ [,] _[ W]_ _D_ [0] [and] _[ W]_ [′] _D_ [0] [are the weight matrices.]
In each block, an addition procedure is adopted before
the activation function as follows:



_L_ 2 = _[λ]_

2 _N_



_H_ [1] _I_ [=] _[ σ(][H]_ [1] _II_ [+] _[ H]_ [1] _DI_ _[)]_ (5)


_H_ [1] _D_ [=] _[ σ(][H]_ [1] _DD_ [+] _[ H]_ [1] _ID_ _[)]_ (6)


where _H_ [1] _I_ [and] _[ H]_ [1] _D_ [are the outputs.] _[ σ]_ [ is the activation]
function, which is ReLU in this study.

The GCN layer _L_ 2 contains Block 3, which is used to
integrate the outputs from two blocks in layer _L_ 1 as
follows:


_Z_ = _σ(H_ [2] _I_ [+] _[ H]_ [2] _D_ _[)]_ [ =] _[ σ(][F]_ _[I]_ _[H]_ [1] _I_ _[W]_ _I_ [1] [+] _[ F]_ _[D]_ _[H]_ [1] _D_ _[W]_ _D_ [1] _[)]_ (7)


where _Z_ is the final representation matrix of drugs.


**Predicting DDIs**

The Block 4 with three fully connected layers is utilized to
predict DDIs in our model, as shown in Figure 1 **C** . Before
Block 4, a concatenation layer is used to generate the
DDI feature matrix. The inputs of concatenation layer are
representation matrix _Z_, and DDI information matrix _D_ .
For a pair of drugs _i_ and _j_ in _D_, its DDI feature vector is the
concatenation of _Z_ _i_ and _Z_ _j_, represented as [ _Z_ _i_, _Z_ _j_ ], where _Z_ _i_
and _Z_ _j_ are the feature vectors of drugs _i_ and _j_ in _Z_, which
is fed into Block 4.


In Block 4, the number of neurons in each layer is 64,
16 and 1. The DDI prediction is formulated as a binary
classification that the output values are the probabilities
of how likely a drug pair is a true DDI. The activation
function is ReLU in hidden layers and Sigmoid in output
layer.

The cross-entropy loss function is used in our GCNMK
model:



_BCE_ = − [1]

_N_



�[ _y_ _ij_ log _p_ _ij_ + _(_ 1 − _y_ _ij_ _)_ log _(_ 1 − _p_ _ij_ _)_ ] (8)

_ij_



where _λ_ is a hyper-parameter, _w_ is an element in the
parameter matrices _W_ _I_ [0] [,] _[ W]_ [′] _I_ [0] [,] _[ W]_ _D_ [0] [,] _[ W]_ [′] _D_ [0] [,] _[ W]_ _I_ [1] [and] _[ W]_ _D_ [1] [. As a]
result, the loss function for training our GCNMK model is
_L_ = _BCE_ + _L_ 2 .


Experiments


In this section, we illustrate the performances of our
proposed model in various types of data and compare
it with three state-of-the-art DDI prediction algorithms.
Five aspects are discussed in the following five subsections: datasets in both our proposed model and the competing models; experiment setting; visualization analysis
of embedding features; results of competing methods;
case studies of our proposed model.


**Datasets**


In order to make a fair comparison between various types
of features and methods, we choose the drugs that have
all types of features in both our proposed methods and
the competing methods. In our study, we download DDIs
from the DrugBank database (Version 5.1.8) [23], whereas
the numbers of ‘increase’-related and ‘decrease’-related

DDIs are 40 202 and 40 500, respectively, among 613 FDAapproved drugs.

Eight types of features are compared in the experiments, as described in Table 1. It should be mentioned
that the node2vec feature matrix is generated from the
whole DDI graph _G_ _all_ = _G_ _I_ ∪ _G_ _D_ and that there is an information leak in it. The features about associated drugs,
enzymes, side effects, substructures and targets are generated from the corresponding databases, as listed in
Table 1. The pathway feature vectors of drugs are based
on the drug-related targets and target–pathway associations. The prototype ranked list (PRL) feature vector is
generated by merging a group of profiles of a given drug
into a single ranked list [29]. The profiles are downloaded
from the Library of Integrated Network-based Cellular
Signatures (LINCS) database [28].


**Experiment setting**

In this study, we use 5-fold cross-validation (5-CV) to
evaluate the prediction performance of our GCNMK
model and the competing methods. The known DDIs
are represented as positive samples, and the unknown
DDIs are represented as negative samples. The number
of positive samples is 80 702, whereas that of negative
samples is 106 876. In order to make the training
data balanced, 80 702 negative samples are randomly
selected. Both the positive samples and the selected
negative samples are divided into five subsets randomly.
At each time, a positive subset and a negative subset



where _N_ is the sample size, _y_ _ij_ ∈ [0, 1] is the true label for
the interaction between drug _i_ and _j_ . ‘1’ represents the
label of a positive sample, whereas ‘0’ represents that of
a negative sample. _p_ _ij_ is the predicted probability.


_GCNMK_ | 5


**Table 1.** The types of features, their dimensions, and resources/methods


**Feature types** **Dimensions** **Resources**


Associated Drugs 613 DrugBank ([23])
Enzymes 454 DrugBank
Pathways 533 DrugBank, CTD ([24]) and KEGG ([25])
Side Effects 4859 SIDER ([26])
Substructures 811 DrugBank
Targets 2670 DrugBank and CTD
Node2vec 613 DrugBank, [27] and [20]
PRL 978 LINCS ([28] and [29])


Figure 2. The influence of learning rate _lr_ .


Figure 3. The influence of _L_ 2 -regularization coefficient _λ_ .



are selected as the testing set, whereas the remaining
subsets are selected as the training set. After five times,
all subsets are used up to be testing sets, and the
predicting results are produced.



In order to avoid using the testing information in the
training procedure and make the testing procedure more
accurate, the DDIs in the testing set are deleted from _G_ _I_
and _G_ _D_ at each training.


6 | _Wang_ et al.


Figure 4. The influence of embedding size _d_ .


Figure 5. The influence of feature type.


In experiments,the area under receiver operating characteristic curve (AUC-ROC) and area under precisionrecall curve (AUC-PR) are used to measure the performance of results. The higher the values are, the more
reliable the model is.


We adjust the parameters in order to achieve optimal
performances. For the learning rate _lr_, _L_ 2 -regularization
coefficient _λ_, and embedding size _d_, we search for
the optimal values with the nominal values _lr_ =0.0005,
_λ_ =0.0005, _d_ =128. When optimizing the influence of a
specific parameter, the other two parameters are set to be
the nominal values. After optimization, its optimal value
is used to update its nominal value. In those experiments,
the target information is used to construct the drug
feature matrix _H_ [0] .



The learning rate _lr_ ∈ [0.1, 0.01, 0.001, 0.0001, 0.00001,0.000001]. After achieving that the optimal value is
around 0.001, we set the learning rate to be in a refined
range [0.0001,0.0002,...,0.0009,0.001,0.002,...,0.009]. In
order to show them clearly, we use two histograms to
depict the AUC-ROC and AUC-PR values under different
_lr_ values, as shown in Figure 2. When _lr_ increases from
0.000001 to 0.002, the general trend of AUC-ROC and
AUC-PR is ascending. When _lr_ is larger than 0.002, the
AUC-ROC and AUC-PR are reducing. Therefore, we set
the learning rate _lr_ to be 0.002 in our proposed GCNMK
model.


The _L_ 2 -regularization coefficient _λ_ ∈ [0.1, 0.01, 0.001, 0.
0001, 0.00001, 0.000001]. The optimal value is around
0.0001. Then _λ_ is set to be in a refined range [0.00001,


_GCNMK_ | 7


Figure 6. The t-SNE visualization analysis of embeddings.


**Table 2.** The prediction performances of the competing methods


**Methods** **AUC-ROC** **AUC-PR**


**Ave.** **Std.** **Rank** **Ave.** **Std.** **Rank**


**GCNMK** **0.9557** 0.0017 1 **0.9508** 0.0012 1

GCNMK-5 0.9337 0.0042 2 0.9292 0.0048 2

DPDDI 0.9126 0.0003 3 0.9131 0.0003 4

SkipGNN 0.8589 0.0005 5 0.8604 0.0005 5

MDAE 0.8981 0.0015 4 0.9232 0.0013 3


Ave.: The average value across ten repeats. Std.: The standard deviation across ten repeats. Rank: The ranks are based on the average values.



0.00002,...,0.00009, 0.0001, 0.0002,...,0.0009]. All the AUCROC and AUC-PR values are shown in Figure 3. When
_λ_ increases from 0.000001 to 0.0003, the AUC-ROC and
AUC-PR increase slightly. When _λ_ is larger than 0.0003,
the AUC-ROC and AUC-PR are decreasing. Therefore, we
set _λ_ to be 0.0003 in our proposed GCNMK model.

The embedding size _d_ ∈ [32, 64, 96, 128, 160, 192, 224,
256, 288,320]. The prediction performance changes a
little when the embedding size varies, as depicted in
Figure 4. When _d_ is increasing from 32 to 160, the AUCROC and AUC-PR are increasing. When _d_ is larger than
160, the AUC-ROC and AUC-PR are becoming smaller. We
set the optimal embedding size _d_ to be 160 in our GCNMK
model.


Various types of features are used in our GCNMK
model. The histograms of their prediction performance
are shown in Figure 5. Although the node2vec [27] feature
has a problem of information leak, its prediction performance is the worst among the eight types of features.
The PRL [29] feature produces the 2nd-worst prediction
results. The differences of the AUC-ROC and AUC-PR

of the other six types of features are not large, and
the target feature of drugs achieves the best prediction
performance among them. Therefore, in the following
comparison, we use the target feature of drugs in our
GCNMK model.


We compare our methods with three DDI prediction
methods, which are DPDDI [19], SkipGNN [20] and MDAE

[21]. The parameters are set to be the optimal values



as described in their methods. The type of feature
used in DPDDI is the associated drugs. In SkipGNN, it
is node2vec. Five types of features are used in MDAE,
including associated drugs, enzymes, pathways, targets
and substructures. Additionally, the same five types
of features are used in our GCNMK model, which is
represented as GCNMK-5 in Table 2.


**Visualization analysis of embedding features**

In order to study the embedding performance of our proposed model, we employ t-distributed stochastic neighbor embedding (t-SNE) [32] to visualize DDIs based on
the embedding features learned from our model. t-SNE
is applied to reduce the dimensionality of embedding
features to 2 and plot a 2-D figure, as shown in Figure 6.
The green dots are known DDIs, whereas the red dots are
unknown DDIs. Based on Figure 6, we can see that most
of dots are gathered in two areas. Specially, the known
DDIs are located at the lower half of the figure, whereas
the unknown DDIs are located on the upper right quarter
of the figure, which can explain the performance of our
model.


**Results**


The prediction performances of all competing methods
are listed in Table 2. Each method is repeated 10 times
to generate an average value and an SD of the AUC-ROC
and AUC-PR metrics. The GCNMK and GCNMK-5, whose
performance ranks are 1 and 2 in terms of AUC-ROC


8 | _Wang_ et al.


**Table 3.** The top 20 predicted DDIs


**Rank** **Drug A** **Drug B** **Evidence Source** **Description**



1 Imipramine Olanzapine Drugs.com Using imipramine together with olanzapine may
increase side effects such as drowsiness.

2 Olanzapine Theophylline TWOSIDE Using the drug combination may increase the side
effect of anaemia.

3 Desipramine Olanzapine Drugs.com Using desipramine together with olanzapine may
increase side effects such as drowsiness.

4 Sulfadiazine Trimethoprim TWOSIDE Using the drug combination may increase the side
effect of anaemia.

5 Cimetidine Tramadol Drugs.com Cimetidine may increase the blood levels and
effects of tramadol.

6 Sulfamethoxazole Trimethoprim TWOSIDE Using the drug combination may increase the side
effect of anaemia folate deficiency.
7 Hydrochloroth- Metoprolol Drugs.com Using metoprolol and hydrochlorothiazide together
iazide



7 Hydrochloroth- Metoprolol Drugs.com Using metoprolol and hydrochlorothiazide together
iazide may lower your blood pressure and slow your heart


rate.

8 Ofloxacin Ticlopidine N.A. N.A.
9 Dextromethor- Quinidine Drugs.com Using dextromethorphan together with quinidine
phan may increase the effects of dextromethorphan.



9 Dextromethor- Quinidine Drugs.com Using dextromethorphan together with quinidine
phan may increase the effects of dextromethorphan.

10 Tolbutamide Vincristine N.A. N.A.

11 Estradiol Progesterone TWOSIDE Using the drug combination may increase the side
effect of anaemia.

12 Fosinopril Hydrochloroth- Drugs.com Their effects may be additive on lowering your
iazide



12 Fosinopril Hydrochloroth- Drugs.com Their effects may be additive on lowering your
iazide blood pressure.

13 Nicotine Vincristine TWOSIDE Using the drug combination may increase the side
effect of anaemia.

14 Hydrochloroth- Pindolol Drugs.com Using pindolol and hydrochlorothiazide together
iazide



14 Hydrochloroth- Pindolol Drugs.com Using pindolol and hydrochlorothiazide together
iazide may lower your blood pressure and slow your heart


rate.

15 Lorazepam Ranitidine TWOSIDE Using the drug combination may increase the side
effect of anaemia.

16 Promethazine Pseudoephedrine TWOSIDE Using the drug combination may increase the side
effect of anaemia.

17 Theophylline Vincristine TWOSIDE Using the drug combination may increase the side
effect of neutropenia.
18 Panobinostat Rosiglitazone N.A. N.A.
19 Hydralazine Reserpine N.A. N.A.
20 Ranitidine Teniposide N.A. N.A.



N.A.: The evidence of the given DDI is not available till now.


and AUC-PR, respectively, are our proposed methods. The
ranks of other three competing methods are from 3 to 5.

We compare GCNMK model with others in different
aspects. There is only one graph kernel in DPDDI method

[19], which is the graph of all known DDIs _G_ _all_ = _G_ _I_ ∪ _G_ _D_ .
The AUC-ROC and AUC-RP values produced by GCNMK
model are about 4% larger than those of DPDDI. Referring to the results in Figure 5, our GCNMK model still
achieves better performance than DPDDI when using the
same type of feature. The results indicate that using
two DDI graphs _G_ _I_ and _G_ _D_ can improve the prediction
performance.

There are two graph kernels in SkipGNN [20] that
one kernel is _G_ _all_ and another kernel _G_ _skip_ is based on
_G_ _all_ . The GCNMK generates 10% larger AUC-ROC and
AUC-RP values than SkipGNN. In this way, the graphs
_G_ _I_ and _G_ _D_ work better in predicting potential DDIs. One
possible reason is that the ratio of edges in _G_ _all_ is about
43% in our datasets, and it is nearly 95% in _G_ _skip_ . Adding
such an almost fully connected graph can not improve
the prediction performance.



Five types of features are used to identify the drug
representation feature vectors in GCNMK-5 and MDAE

[21]. In the results, the GCNMK-5 outperforms MDAE.
Furthermore, the GCNMK achieves better prediction performance than GCNMK-5, which indicates that multiple
types of features do not achieve better results than a
single type of feature.

In summary, our proposed GCNMK model achieves
the best prediction performance among all competing
methods in terms of AUC-ROC and AUC-PR.


**Case studies**


In case studies, all 106 876 unknown DDIs are fed into
our GCNMK model. A larger prediction score of two drugs
suggests that they have a higher probability of having
an interaction. We generate a ranked list of DDIs in
descending order according to their prediction scores.

The top 20 predicted DDIs are listed in Table 3. We
verify them with TWOSIDE database [30] and Drug
Interactions Checker [31], and collect the descriptions


_GCNMK_ | 9


**Table 4.** The top ten predicted DDIs of breast neoplasms-related drugs


**Rank** **Drug A** **Drug B** **Evidence Source** **Description**


1 **Verapamil** Mefloquine Drugs.com Using mefloquine together with verapamil can
increase the risk of irregular heart rhythm that may
be serious and potentially life-threatening.
2 **Sulindac** Methazolamide N.A. N.A.

3 **Ranitidine** **Vinblastine** TWOSIDE Using the drug combination may increase the side
effect of neutropenia.
4 **Rosiglitazone** Metformin TWOSIDE Using the drug combination may increase the side
effect of anaemia vitamin b12 deficiency.
5 **Quinine** Nizatidine TWOSIDE Using the drug combination may increase the side
effect of chest pain.
6 **Sulindac** Theobromine N.A. N.A.

7 **Ranitidine** Sunitinib TWOSIDE Using the drug combination may increase the side
effect of anaemia.

8 **Ranitidine** Teniposide N.A. N.A.
9 **Ranitidine** **Vinorelbine** TWOSIDE Using the drug combination may increase the side
effect of anaemia.

10 **Sulfasalazine** Isosorbide TWOSIDE Using the drug combination may increase the side
effect of anaemia.


The breast neoplasms-related drugs are in bold.


**Table 5.** The top ten predicted DDIs of colorectal neoplasms-related drugs


**Rank** **Drug A** **Drug B** **Evidence Source** **Description**


1 **Simvastatin** Niacin TWOSIDE Using the drug combination may increase the side
effect of iron deficiency anaemia.
2 **Fluorouracil** Lorazepam TWOSIDE Using the drug combination may increase the side
effect of iron deficiency anaemia.
3 **Meloxicam** **Methotrexate** TWOSIDE Using the drug combination may increase the side
effect of iron deficiency anaemia.
4 **Fluorouracil** Tramadol TWOSIDE Using the drug combination may increase the side
effect of anaemia.

5 **Famotidine** Primidone TWOSIDE Using the drug combination may increase the side
effect of haemorrhagic anaemia.
6 **Dacarbazine** Phenytoin N.A. N.A.
7 **Famotidine** Progesterone TWOSIDE Using the drug combination may increase the side
effect of atrial fibrillation.

8 **Fluorouracil** Oxymetholone N.A. N.A.
9 **Doxorubicin** Lynestrenol N.A. N.A.
10 **Simvastatin** Trifluoperazine TWOSIDE Using the drug combination may increase the side
effect of pancytopenia.


The colorectal neoplasms-related drugs are in bold.



about their interactions. For instance, the description of
‘Imipramine-Olanzapine’ is ‘Using imipramine together
with olanzapine may increase side effects such as
drowsiness’. We can see that 15 DDIs are confirmed in
either Drugs.com or TWOSIDE. The results indicate that
our proposed GCNMK model is effective in predicting
novel DDIs. Other five DDIs, ‘Ofloxacin–Ticlopidine’,
‘Tolbutamide–Vincristine’, ‘Panobinostat–Rosiglitazone’,
‘Hydralazine–Reserpine’ and ‘Ranitidine–Teniposide’,
deserve to be confirmed by further experiments. Additionally, the drug ‘Vincristine’ appears in three predicted
DDIs, two of which have been confirmed. More attention
should be paid on ‘Tolbutamide–Vincristine’.

Especially, in order to study the potential DDIs, which
are related to a given disease, we generate the diseaserelated drugs from CTD database. Those drugs have
been used to treat the given disease. In our datasets,



the numbers of breast, colorectal and lung neoplasmsrelated drugs are 64, 31 and 36, respectively. The
unknown DDIs that are connected with those drugs are
predicted. The predicted results are listed in Tables 4, 5,
and 6.

In the predicted results of breast neoplasms-related
DDIs, 7 out of 10 DDIs have been confirmed to have
interactions in either TWOSIDE or Drugs.com. Especially, there are two confirmed DDIs, each of which
consists of two breast neoplasms-related drugs. The
other three DDIs, ‘Sulindac–Methazolamide’, ‘Sulindac–
Theobromine’ and ‘Ranitidine–Teniposide’, deserve to
be confirmed by further experiments. Especially, among
the 10 predicted DDIs, the drug ‘Ranitidine’ appears in
four DDIs, whereas three DDIs have been confirmed.
The DDI ‘Ranitidine–Teniposide’ should attract more
attention.


10 | _Wang_ et al.


**Table 6.** The top ten predicted DDIs of lung neoplasms-related drugs


**Rank** **Drug A** **Drug B** **Evidence Source** **Description**


1 **Sulindac** Methazolamide N.A. N.A.

2 **Rosiglitazone** Metformin TWOSIDE Using the drug combination may increase the side
effect of anaemia vitamin b12 deficiency.
3 **Theophylline** **Vincristine** TWOSIDE Using the drug combination may increase the side
effect of neutropenia.
4 **Sulindac** Theobromine N.A. N.A.

5 **Methotrexate** Meloxicam TWOSIDE Using the drug combination may increase the side
effect of iron deficiency anaemia.
6 **Theophylline** Thalidomide TWOSIDE Using the drug combination may increase the side
effect of anaemia.

7 **Ifosfamide** Ofloxacin Drugs.com Chemotherapy with ifosfamide may reduce the
plasma concentrations of oral ofloxacin.
8 **Theophylline** Olanzapine TWOSIDE Using the drug combination may increase the side
effect of anaemia.

9 **Sulindac** Isosorbide TWOSIDE Using the drug combination may increase the side
effect of pancytopenia.
10 **Melatonin** Tacrolimus TWOSIDE Using the drug combination may increase the side
effect of pancytopenia.


The lung neoplasms-related drugs are in bold.



In the predicted results of colorectal neoplasmsrelated DDIs, 7 out of 10 DDIs have been confirmed
to have interactions in TWOSIDE. The other three

interactions, ‘Dacarbazine–Phenytoin’, ‘Fluorouracil–
Oxymetholone’ and ‘Doxorubicin–Lynestrenol’, could be
potential DDIs.

In the predicted results of lung neoplasms-related
DDIs, 8 out of 10 DDIs have been confirmed to have
interactions in either TWOSIDE or Drugs.com. The other
two DDIs, ‘Sulindac–Methazolamide’ and ‘Sulindac–
Theobromine’, are also on the predicted list of breast
neoplasms.

These neoplasms-related case studies demonstrate
the usefulness of our GCNMK model in identifying
potential DDIs for specific disease-related drugs.


Conclusion and Discussion


In this study, we have proposed a GCNMK model for predicting DDIs. The ‘increase’-related DDIs and ‘decrease’related DDIs are used to construct two DDI graphs, which
are the graph kernels in our model. Then novel embeddings of drugs are produced by three GCN blocks. A DDI
feature vector is the concatenation of two drug feature
vectors. A block of three fully connected layers is used
as a predictor. Comprehensive experiments have been
conducted to evaluate the performance of GCNMK and
other methods. In the experiments, our GCNMK model
outperforms all other methods. In the case studies, most
of the predicted DDIs have evidence to support the existence of their interactions. Therefore, benefiting from the
two graph kernels, our GCNMK model can be used to
predict DDIs effectively.

Even so, there is a limitation in our proposed model.
When constructing the DDI graphs and generating the
set of drugs, the drugs in the experiment have at least one
DDI. We remove the drugs that do not have any known



DDIs. As a result, our model can not identify DDIs among
isolated drugs.

There are several directions of future work along this
study. In the DDI graphs of GCNMK, the edges belong to
the same type. We could adapt this to any heterogeneous
network, such as drug–disease network. The descriptions
of drug–diseases associations consist of two types: therapeutic and marker/mechanism, which may be useful
for employing a GCN model. Another future direction is
to distinguish more types of predicted DDIs. According
to their functions, each type of DDI may be used to
construct a graph kernel, and the model has potential to
identify the specific type of a predicted DDI.


**Key Points**


  - We propose a graph convolutional network
with multi-kernel, termed GCNMK, for predicting

DDIs.

  - The DDIs are divided into two groups, which are
increased and decreased DDIs.

  - GCNMK uses GCN blocks to capture structure
features from both increased and decreased DDI

graphs and uses fully connected layers as the
predictor.


Data availability


The datasets were derived from the following sources
in the public domain: the drug–drug interactions from
[https://www.drugbank.ca/, the drug–enzyme associa-](https://www.drugbank.ca/)
[tions from https://www.drugbank.ca/ [23], the pathway–](https://www.drugbank.ca/)
[target associations from https://www.genome.jp/kegg/](https://www.genome.jp/kegg/pathway.html)
[pathway.html](https://www.genome.jp/kegg/pathway.html) [25] and [http://ctdbase.org/](http://ctdbase.org/) [24], the
[drug–side effect associations from http://sideeffects.e](http://sideeffects.embl.de/)
[mbl.de/ [26], the drug chemical substructures from](http://sideeffects.embl.de/)


[https://www.drugbank.ca/, the drug–target associations](https://www.drugbank.ca/)
[from https://www.drugbank.ca/ and http://ctdbase.org/,](https://www.drugbank.ca/)
[the drug perturbation profiles from https://www.ncbi.](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742)
[nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742 [28].](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742)


Acknowledgments


The authors thank the anonymous reviewers for their
valuable suggestions.


Author contributions statement


F.W. and F.X.W. discussed the algorithms and conceived
the experiments; F.W. implemented the algorithms
and experiments, analyzed the results and wrote
the manuscript; F.X.W., X.L. and B.L. reviewed the
manuscript.


Funding


Natural Science and Engineering Research Council of
Canada (NSERC); China Scholarship Council (CSC); the
National Natural Science Foundation of China (Grant No.
2020YFB2104400, 61428209).


References


1. Pushpakom S, Iorio F, Eyers PA, _et al._ Drug repurposing:
progress, challenges and recommendations. _Nat Rev Drug Discov_
2019; **18** (1):41–58.
2. Luo H, Wang J, Li M, _et al._ Drug repositioning based on comprehensive similarity measures and Bi-Random walk algorithm.
_Bioinformatics_ 2016; **32** (17):2664–71.
3. Luo H, Wang J, Li M, _et al._ Computational drug repositioning
with random walk on a heterogeneous network. _IEEE/ACM Trans_
_Comput Biol Bioinform_ 2018; **16** (6):1890–900.
4. Jiang HJ, Huang YA, You ZH. Predicting drug-disease associations
via using gaussian interaction profile and kernel-based autoencoder. _Biomed Res Int_ 2019; **2019** :1–11.

5. Li Z, Huang Q, Chen X, _et al._ Identification of drug-disease associations using information of molecular structures and clinical
symptoms via deep convolutional neural network. _Front Chem_

2020; **7** :1–14.

6. Yu Z, Huang F, Zhao X, _et al._ Predicting drug-disease associations
through layer attention graph convolutional network. _Brief Bioin-_
_form_ 2021; **22** (4):1–11.
7. Wang F, Ding Y, Lei X, _et al._ Identifying gene signatures for cancer
drug repositioning based on sample clustering. _IEEE/ACM Trans_
_Comput Biol Bioinform_ [2020;1–13. 10.1109/TCBB.2020.3019781.](https://doi.org/10.1109/TCBB.2020.3019781)
8. Luo Y, Zhao X, Zhou J, _et al._ A network integration approach
for drug-target interaction prediction and computational drug
repositioning from heterogeneous information. _Nat Commun_
2017; **8** (1):1–13.
9. Wen M, Zhang Z, Niu S, _et al._ Deep-learning-based drug-target
interaction prediction. _J Proteome Res_ 2017; **16** (4):1401–9.
10. Wang H, Wang J, Dong C, _et al._ A novel approach for drug-target
interactions prediction based on multimodal deep autoencoder.
_Front Pharmacol_ 2020; **10** :1–19.

11. Hu S, Zhang C, Chen P, _et al._ Predicting drug-target interactions
from drug structure and protein sequence using novel convolutional neural networks. _BMC Bioinformatics_ 2019; **20** (25):1–12.



_GCNMK_ | 11


12. Monteiro NRC, Ribeiro B, Arrais J, _et al._ Drug-target interaction
prediction: end-to-end deep learning approach. _IEEE/ACM Trans_
_Comput Biol Bioinform_ [2020;1–12. 10.1109/TCBB.2020.2977335.](https://doi.org/10.1109/TCBB.2020.2977335)
13. Jiang M, Li Z, Zhang S, _et al._ Drug-target affinity prediction using graph neural network and contact maps. _RSC Adv_
2020; **10** (35):20701–12.
14. Ferdousi R, Safdari R, Omidi Y. Computational prediction of
drug-drug interactions based on drugs functional similarities. _J_
_Biomed Inform_ 2017; **70** :54–64.
15. Yan C, Duan G, Pan Y, _et al._ DDIGIP: predicting drug-drug
interactions based on Gaussian interaction profile kernels. _BMC_
_Bioinformatics_ 2019; **20** (15):1–10.
16. Zheng Y, Peng H, Zhang X, _et al._ DDI-PULearn: a positiveunlabeled learning method for large-scale prediction of drugdrug interactions. _BMC Bioinformatics_ 2019; **20** (19):1–12.
17. Zhou B, Wang R, Wu P, _et al._ Drug repurposing based on drugdrug interaction. _Chem Biol Drug Des_ 2015; **85** (2):137–44.
18. Zhang W, Chen Y, Liu F, _et al._ Predicting potential drug-drug
interactions by integrating chemical, biological, phenotypic and
network data. _BMC Bioinformatics_ 2017; **18** (1):1–12.
19. Feng YH, Zhang SW, Shi JY. DPDDI: a deep predictor for drugdrug interactions. _BMC Bioinformatics_ 2020; **21** (1):1–15.
20. Huang K, Xiao C, Glass LM, _et al._ SkipGNN: predicting molecular interactions with skip-graph networks. _Sci Rep_ 2020; **10** (1):

1–16.

21. Zhang Y, Qiu Y, Cui Y, _et al._ Predicting drug-drug interactions using multi-modal deep auto-encoders based network
embedding and positive-unlabeled learning. _Methods_ 2020; **179** :

37–46.

22. Kipf TN, Welling M. Semi-supervised classification with graph
convolutional networks. arXiv preprint arXiv:1609.02907.

2016;1–14.

23. Wishart DS, Feunang YD, Guo AC, _et al._ DrugBank 5.0: a major
update to the DrugBank database for 2018. _Nucleic Acids Res_
2018; **46** [(D1):D1074–82. 10.1093/nar/gkx1037 [dataset].](https://doi.org/10.1093/nar/gkx1037)
24. Davis AP, Grondin CJ, Johnson RJ, _et al._ Comparative toxicogenomics database (CTD): update 2021. _Nucleic Acids Res_
2021; **49** [(D1):D1138–43. 10.1093/nar/gkaa891.](https://doi.org/10.1093/nar/gkaa891)
25. Kanehisa M, Furumichi M, Sato Y, _et al._ KEGG: integrating viruses
and cellular organisms. _Nucleic Acids Res_ 2021; **49** (D1):D545–51.
[10.1093/nar/gkaa970.](https://doi.org/10.1093/nar/gkaa970)
26. Kuhn M, Letunic I, Jensen L, _et al._ The SIDER database of
drugs and side effects. _Nucleic Acids Res_ 2016; **44** (D1):D1075–9.
[10.1093/nar/gkv1075.](https://doi.org/10.1093/nar/gkv1075)
27. Grover A, Leskovec J. node2vec: Scalable feature learning for
networks. In: _Proceedings of the 22nd ACM SIGKDD interna-_
_tional conference on knowledge discovery and data mining_ . Association for Computing Machinery, New York, NY, USA, 2016,

855–64.

28. Subramanian A, Narayan R, Corsello SM, _et al._ A next generation
connectivity map: L1000 platform and the first 1,000,000 profiles. _Cell_ 2017; **171** [(6):1437–52. 10.1016/j.cell.2017.10.049.](https://doi.org/10.1016/j.cell.2017.10.049)
29. Iorio F, Bosotti R, Scacheri E, _et al._ Discovery of drug mode of
action and drug repositioning from transcriptional responses.
_Proc Natl Acad Sci_ 2010; **107** (33):14621–6.
30. Tatonetti NP, Patrick PY, Daneshjou R, _et al._ Data-driven prediction of drug effects and interactions. _Sci Transl Med_ 2012; **4** (125):

1–26.

[31. Drugs.com. Drug Interactions Checker. https://www.drugs.com/](https://www.drugs.com/drug_interactions.html)
[drug_interactions.html, September 13, 2021.](https://www.drugs.com/drug_interactions.html)
32. Laurens VM, Geoffrey H. Visualizing data using t-SNE. _J Mach_
_Learn Res_ 2008; **9** (11):1–27.


