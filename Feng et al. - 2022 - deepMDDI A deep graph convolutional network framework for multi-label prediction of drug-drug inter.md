[Analytical Biochemistry 646 (2022) 114631](https://doi.org/10.1016/j.ab.2022.114631)


Contents lists available at ScienceDirect

# Analytical Biochemistry


[journal homepage: www.elsevier.com/locate/yabio](https://www.elsevier.com/locate/yabio)

## deepMDDI: A deep graph convolutional network framework for multi-label prediction of drug-drug interactions


Yue-Hua Feng [a], Shao-Wu Zhang [a] [,] [*], Qing-Qing Zhang [a], Chu-Han Zhang [b], Jian-Yu Shi [c] [,] [** ]


a _Key Laboratory of Information Fusion Technology of Ministry of Education, School of Automation, Northwestern Polytechnical University, Xi’an, 710072, China_
b _School of Software, Northwestern Polytechnical University, Xian, 710072, China_
c _School of Life Sciences, Northwestern Polytechnical University, Xi’an, 710072, China_



A R T I C L E I N F O


_Keywords:_
Drug-drug interactions
Graph convolution network
Multi-label prediction
Transductive prediction
Inductive prediction


**1. Introduction**



A B S T R A C T


It is crucial to identify DDIs and explore their underlying mechanism (e.g., DDIs types) for polypharmacy safety.
However, the detection of DDIs in assays is still time-consuming and costly, due to the need for experimental
search over a large space of drug combinations. Thus, many computational methods have been developed to
predict DDIs, most of them focusing on whether a drug interacts with another or not. And a few deep learningbased methods address a more realistic screening task for identifying various DDI types, but they assume a DDI
only triggers one pharmacological effect, while a DDI can trigger more types of pharmacological effects. Thus,
here we proposed a novel end-to-end deep learning-based method (called deepMDDI) for the Multi-label pre­
diction of Drug-Drug Interactions. deepMDDI contains an encoder derived from relational graph convolutional
networks and a tensor-like decoder to uniformly model interactions. deepMDDI is not only efficient for DDI
transductive prediction, but also inductive prediction. The experimental results show that our model is superior
to other state-of-the-art deep learning-based methods. We also validated the power of deepMDDI in the DDIs
multi-label prediction and found several new valid DDIs in the case study. In conclusion, deepMDDI is beneficial
to uncover the mechanism and regularity of DDIs.



Polypharmacy, also termed as drug combination, is becoming a
promising strategy for treating complex diseases (e.g., diabetes and
cancer) in recent years [1]. When two or more drugs are taken together,
they may trigger unexpected side effects, adverse reactions, and even
serious toxicity [2]. The pharmacological effects triggered by multiple
drugs in the treatment are named drug-drug interactions (DDIs). DDIs
can be divided into two cases. One case is that a pair of drugs triggers
only one pharmacological effect, another is that a pair of drugs causes
two or more related pharmacological effects. We call the former a
single-fold interaction and the latter a multi-fold interaction. For
example, the interaction between Sucralfate and Metoclopramide tells
“Sucralfate may decrease the excretion rate of Metoclopramide, result­
ing in a higher serum level”. The pair of these two drugs may trigger two
related pharmacokinetic effects, that is, Excretion and Serum Concen­
tration. Therefore, it is crucial to identify DDIs and unravel their un­
derlying mechanisms for polypharmacy safety. However, it is still both



time-consuming and costly to detect DDIs among a large scale of drug
pairs in assays. Over the past decade, this build-up of experimentally
determined DDI entries boosts the application of computational
methods to find the potential DDIs [3], especially machine
learning-based methods.
Various machine learning methods have been proved as promising
methods to provide preliminary screening of DDIs for further experi­
mental validation with the advantages of both high efficiency and low
costs. Generally, machine learning-based methods [4–15] use the
approved DDIs training the predictive models to infer the potential DDIs
among massive unlabeled drug pairs by extracting the drug features
from diverse drug property sources, such as chemical structure [4,6–9],
targets [4–7], anatomical taxonomy [5,8,10] and phenotypic observa­
tion [5,7,9,10], or extracting the drug similarity features [5,6,9,10,
16–18]. However, most existing methods for DDI prediction focus on
whether a drug interacts with another or not. In addition, they have
limited ability to represent structural data [19–21] (e.g., DDIs).
Without hand-drafted feature engineering, advanced machine




 - Corresponding author.
** Corresponding author.
_E-mail addresses:_ [zhangsw@nwpu.edu.cn (S.-W. Zhang), jianyushi@nwpu.edu.cn (J.-Y. Shi).](mailto:zhangsw@nwpu.edu.cn)


[https://doi.org/10.1016/j.ab.2022.114631](https://doi.org/10.1016/j.ab.2022.114631)
Received 30 December 2021; Received in revised form 10 February 2022; Accepted 23 February 2022

Available online 25 February 2022
0003-2697/© 2022 Elsevier Inc. All rights reserved.


_Y.-H. Feng et al._ _Analytical Biochemistry 646 (2022) 114631_



learning methods (i.e, deep learning) provide a good representation of
diverse structural data, such as Graph Neural Network (GNN) specif­
ically designed for graph-structure data. Deep learning (DL) has been
widely utilized in drug discovery [22], including drug molecular activity
prediction [23], molecular property prediction [24–26], target identi­
fication [27,28], de novo molecular design [29,30], DTI (drug-target
interaction) prediction [31,32] as well as DDI (drug-drug interaction)
prediction [11,33,34]. Specifically, a few DL-based predictive methods
of DDI [11,12,35] have been developed to address a harder screening
task of identifying pharmacological effects caused by known DDIs. For
example, DeepDDI designs a nine-layer deep neural network to predict
86 types of DDIs by using the structural information of drug pairs as
inputs [11]. Lee et al. predict the pharmacological effects of DDIs by
using drug similarity profiles including the structural similarity profile,
Gene Ontology term similarity profiles, and target gene similarity pro­
files of known drug pairs to train the three-layer autoencoder and an
eight-layer deep feed-forward network [35]. DDIMDL predicts DDI
events by using the drug similarity features computed from chemical
substructures, targets, enzymes, and pathways to separately train
three-layer deep neural networks (DNNs), and then averages (sums up)
individual predictions of those trained DNNs as the final prediction [12].
Despite these efforts on identifying pharmacological effects of DDIs,
they still exist the following improvements for DL-based methods. (1)
Existing DL-based methods require the known DDI as input, while the
interactions of most drug pairs are unknown. Therefore, it is necessary to
develop new algorithms to identify whether an unknown drug pair has
one or more pharmacological effects. (2) More DDIs can form an inter­
action network that helps to improve the predictors performance,
however existing DL-based methods treat drug pairs as independent
samples, ignoring the structural relationship between DDI entries.
To address the above issues, we proposed a novel deepMDDI method
to identify whether an unknown drug pair results in one or more phar­
macological effects. The main contributions of our work are as follows:
(1) deepMDDI leverages an encoder by an deep relational graph con­
volutional network (R-GCN) to capture the topological features of DDI
network. (2) deepMDDI employs a tensor-like decoder to uniformly
model both single-fold interactions and multi-fold interactions for
identifying whether an unlabeled type-specific drug pair results in one
or more pharmacological effects.


**2. Materials and methods**


_2.1. Datasets_


We built the DDI dataset by collecting DDI entries from DrugBank
(July 16, 2020) [36] in the following steps. First, after downloading the
completed XML-formatted database (including the comprehensive pro­
files of 11,440 drugs), we selected all small-molecule drugs (i.e., 3373
drugs) and their 1,138,353 DDIs entries and parsed all the sentences of
their DDIs. Secondly, we utilized a keyword dictionary pattern-matching
algorithm [37] to collect sentence patterns where 274 patterns in total
are found (Table S1). All sentence patterns were further categorized into
11 types [38], including Absorption, Metabolism, Serum Concentration,
Excretion, Activity Decrease, Activity Increase, Toxicity Activity,
Adverse Effect, Antagonism Effect, Synergy Effect, and PD triggered by
PK. For example, the sentence pattern “The absorption of Drug Rosu­
vastatin can be decreased when combined with Drug Sodium bicar­
bonate” falls into Absorption. The sentence pattern “Orlistat can cause a
decrease in the absorption of Cyclosporine resulting in a reduced serum
concentration and potentially a decrease in efficacy” falls into Absorp­
tion, Serum Concentration, and Antagonism Effect. It means that the
pair of Orlistat and Cyclosporine has three types of interactions simul­
taneously, called a multi-fold DDI in our model. The type annotations of
sentence patterns were marked. In addition, chemical structures and
binding proteins (including drug target, enzyme, transporter, and
pathway protein information) of these drugs were also collected to



calculate drug similarities. After discarding the drugs without chemical
structure or binding protein, we eventually obtained 859,662 drug-drug
interaction entries among 2926 drugs and we organized them as a DDI
matrix, which was used as the input of our model.
We also performed statistics on the DDI network of our dataset.
Fig. 2-E shows the statistics on different pharmacological effects caused
by DDIs. And Fig. 2-F shows the proportional distribution of the number
of single-fold and multi-fold DDIs. Other details of the datasets are
provided in Tables S2–S3 (in supplementary).


_2.2. Problem formulation_


Suppose n drugs **D** = {d i } in the DDIs network, x drugs **D** x = {d j } is
new drugs and k interactions **L** = {l ij } among drugs. The traditional
DDI binary prediction, DDI multi-classification, DDI multi-label pre­
diction in the network and DDI multi-label prediction for new drug are
different pharmacological tasks.


- The task of traditional DDI binary prediction learns a function
mapping **F** : **D** × **D** →{0 _,_ 1} to deduce potential interactions be­
tween unlabeled drug pairs among **D** (Fig. 1-A).

- The task of DDI multi-classification identifies what pharmacological
effects caused by known DDIs are (Fig. 1-B). It learns a function
mapping **F** : **L** →{ _t_ _i_ } _, i_ = 1 _,_ 2 _,_ … _,m_, where _t_ _i_ is the pharmacological
effect type of DDIs, and _m_ is the total number of all pharmacological
effects.

- The task of DDI multi-label prediction directly discriminates whether
an unknown drug pair in the DDIs network results in one or more
pharmacological effects of interest (Fig. 1-C). It learns a set of
functions mapping **F** _t_ _i_ : **D** × **D** →[0 _,_ 1] _, i_ = 1 _,_ 2 _,_ … _,m_, where _t_ _i_ is the
pharmacological effect type of DDIs, and _m_ is the total number of all
pharmacological effects.

- The task of DDI multi-label prediction directly discriminates whether
an unknown drug pair between drugs in the DDIs network and new
drug out the network results in one or more pharmacological effects
of interest (Fig. 1-D). It learns a set of functions mapping **F** _t_ _i_ : **D** ×
**D** _x_ →[0 _,_ 1] _, i_ = 1 _,_ 2 _,_ … _,m_, where _t_ _i_ is the pharmacological effect type
of DDIs, and _m_ is the total number of all pharmacological effects.


This work focuses on the task of DDI multi-label prediction (task3
and task4) since the second task is just its degraded version. Referring to
DDI-triggered pharmacological effects as interaction types, we represent
a set of multiple relation DDIs as a complex network _G_ ( **V** _,_ **R** ), where
vertices are drugs and edges between vertices are multiple relation in­
teractions (Fig. 2-A). Let **V** = { _v_ 1 _, v_ 2 _,_ … _, v_ _n_ } be the vertex set, **T** = { _t_ 1 _, t_ 2 _,_
… _, t_ _m_ } be the interaction type set, and ( _v_ _i_ _, t_ _r_ _, v_ _j_ ) be the interaction of type
_t_ _r_ caused by the pair of drug _d_ _i_ and drug _d_ _j_ . Furthermore, _G_ is decom­
posed into _m_ sliced sub-networks _G_ = { _G_ 1 _, G_ 2 _,_ … _, G_ _m_ } regarding inter­
action types (Fig. 2-A). Each slice, denoted as _G_ _r_, is represented a
symmetric adjacent matrix _**A**_ _[r]_ _n_ × _n_ [= {] _[a]_ _[r]_ _ij_ [}] _[,][ i][,][ j]_ [ =][ 1] _[,]_ [ 2] _[,]_ [ …] _[,][ n]_ [;] _[ r]_ [ =][ 1] _[,]_ [ 2] _[,]_ [ …] _[,][ m][,]_

where _a_ _[r]_ _ij_ [=][ 1 indicates an approved interaction of type ] _[t]_ _[r ]_ [between drug ]

_d_ _i_ and drug _d_ _j_, and _a_ _[r]_ _ij_ [=][ 0 otherwise. These binary adjacent matrices ]

naturally form a 3-order multiple relational tensor **R** ∈ R _[n]_ [×] _[n]_ [×] _[m ]_ (Fig. 2A). Besides, pairwise similarities of all drugs among **D** are organized
into a similarity matrix, **S** = { _s_ _u,v_ } ∈[0 _,_ 1] _, u,_ _v_ = 1 _,_ 2 _,_ … _,_ _n_ .


_2.3. Feature extraction_


In addition to interaction entries, we extracted drug chemical
structures, which are represented by SMILES strings, as well as drug
binding proteins (DBPs), including targets, enzymes, transporters, and
carriers. Drug chemical structures were encoded into feature vectors by
Extended Connectivity Fingerprints (ECFPs) [39] and MACCSkeys
(Molecular ACCess System keys) Fingerprints [39], respectively. ECFPs
represent a molecular structure through circular atom neighborhoods as



2


_Y.-H. Feng et al._ _Analytical Biochemistry 646 (2022) 114631_


**Fig. 1.** Four tasks in DDIs prediction: (A) DDI binary prediction. (B) DDI multi-classification. (C) DDI multi-label prediction in the network. (D) DDI multi-label
prediction for new drug.


**Fig. 2.** Overall framework of deepMDDI for the DDI multi-label prediction. (A) Decomposition of the multiple relation DDIs network. The multiple relation DDI
network is decomposed into _m_ sliced (i.e., type number) sub-networks, which are represented by _m_ adjacent matrices and are taken as the input of the encoder. **(B)**
Encoder. It constructs a _p_ -layer relation GCN (R-GCN) to encode drugs in the multiple relation DDI network into embedding vectors (i.e., rows in the colorful matrix)
by capturing their complex topological properties. A residual strategy (i.e., the black arrow) is added from the second hidden layer to the last hidden layer.
Meanwhile, a drug similarity matrix is employed to constrain similar drugs as close as possible in the embedding space (i.e., the purple matrix). **(C)** Decoder. It is a
tensor factorization-like matrix operation, which integrates the embedding feature matrix, type-specific feature importance matrices { **D** _**t**_ _r_ }, and an average feature
association matrix **R** to reconstruct the multiple relation DDIs network. **(D)** An example to illustrate a layer of R-GCN in the encoder. An interest node (i.e., blue node)
aggregates both the features of its first-order neighbor nodes (i.e., orange) and its own in each of _m_ sliced networks to update its features (i.e., green bar). Then, all the
updated features are accumulated and passed through a ReLU activation function to produce its final embedding (i.e., the colorful vector). The whole DDI network is
propagated by a p-layer R-GCN to capture the information of its p-th order neighbors. **(E)** Statistics on different pharmacological effects caused by DDIs. From the left
to the right, the interaction types are: Absorption, Metabolism, Serum Concentration, Excretion, Activity Decrease, Activity Increase, Toxicity Activity, Adverse
Effect, Antagonism Effect, Synergy Effect, and PD triggered by PK. Y-axis indicates their occurring numbers. **(F)** Proportional distribution of the number of single-fold
and multi-fold DDIs. 79.6% DDIs are single-fold, 19.36% are two-fold and 1.04% are three-fold. Other details of the datasets are in Table S2 and Table S3. (For
interpretation of the references to color in this figure legend, the reader is referred to the Web version of this article.)



a 1024-dimensional binary vector, where each element denotes the
presence or the absence of a specific functional substructure. In contrast,
the MACCSkeys Fingerprints represent a molecular structure as
166-dimensional binary vector w.r.t. a set of pre-defined substructures.



These two fingerprints are computed by the RDKit Package implemented
in Python, and the radius of ECFPs neighborhood is set to 4.
Moreover, we consider DBPs (targets, transporters, enzymes, and
carrier proteins) as the third type of drug feature, because they are



3


_Y.-H. Feng et al._ _Analytical Biochemistry 646 (2022) 114631_



crucial factors when a DDI occurs. Sequentially, the drug is represented
as a 3334-dimensional binary vector in which each element indicates
whether the drug binds to a specific protein. Finally, the Tanimoto co­
efficient (TC) [40] is used to calculate drug similarities from drug fea­
tures including the ECFPs_4, MACCSkeys Fingerprints, and DBPs. Last,
we averaged these three kinds of similarities as the final similarity to
regularize the distribution of latent embeddings Z n×k in our model.


_2.4. deepMDDI model_


Upon the above representation of multiple pharmacological effects
by DDIs, we cast the task of DDI prediction as to the multiple relational
link prediction, and design an end-to-end model deepMDDI to address
this task. deepMDDI contains an encoder **F** _e_ and a decoder **F** _d_ .
Derived from the relation GCN (R-GCN) [33,41–43], we construct a
multi-layer R-GCN in which encoder **F** _e_ extracts a global latent feature
matrix **Z** _n_ × _k_ ( _k_ **≪** _n_ ) by capturing the topological feature matrices { **Z** _[r]_ _n_ × _k_ [}]
of all drugs across { _G_ _r_ }. However, the primary multi-layer GCN causes
the over-smoothing issue that makes all the nodes in a network have
highly similar feature values. To relax the over-smoothing issue, **F** _e_
doesn’
t use the outputting embedding representations of its final layer,
but it sums the embedding representations (named residuals) of its

**Z** . In
hidden layers together as its final embedding feature matrix
addition, considering a few of possible missing interactions among the
network, **F** _e_ utilizes a pre-defined drug similarity matrix to constrain
the similar drugs closer to each other in the embedding space.
Since the original decoder in the primary GCN [43] is just an inner
production ZZ [T ] between drug embedding vectors, it cannot reflect the
essence of multiple relation interactions. R-GCN employs RESCAL [44],
which utilizes m additional type-specific feature association matrices M r
to capture the essence of multiple relation interactions (i.e., ZM r Z [T] ).
Inspired by literature [33,45], we suppose that feature importance
varies across interaction types, and we also assume that interaction
types are not completely independent to each other. Therefore, our
decoder **F** _d_ adopts a tensor factorization-like matrix operation to
integrate the embedding feature matrix **Z**, _m_ type-specific feature
importance matrices, and an average feature association matrix to
reconstruct the multiple relation DDIs network (i.e., **ZD** _**t**_ _r_ **RD** _**t**_ _r_ **Z** _[T]_ ).
Finally, our deepMDDI trains **F** _e_ and **F** _d_ simultaneously to obtain
an end-to-end model for implementing the DDI multi-label prediction.


_2.4.1. Encoder in relation graph convolutional network_
We employed the extension GCN (i.e., R-GCN) to extract the node
embedding in the multiple relation DDI network (Fig. 2-B). First, the
network G is decomposed into m sliced sub-networks { _G_ 1 _,G_ 2 _,_ … _,G_ _m_ }, in
which each slice accounts for a specific interaction type (Fig. 2-A). Then,

both the feature vector **h** [(] _i_ [0][)] of drug _d_ _i_ (or node _v_ _i_ ) and those of its
neighbors in _G_ _r_ are aggregated by a graph convolutional operation. After
that, similar aggregations across all the sliced subnetworks are further

summed up to generate the updated feature vector **h** [(] _i_ [1][)] of drug _d_ _i_ . Such a
single layer of R-GCN integrates the topological neighborhood of drug _d_ _i_
across interaction types which it involves. For any layer in a multi-layer
R-GCN (Fig. 2-E), the general propagation rule is defined as:



higher-order topological features of multi-type DDI network [46].
However, it usually causes the ‘over smoothing’ issue derived from GCN

[46], where the features of the neighboring drugs, even all drugs in the
case of many layers, are extremely similar. As a result, a good GCN
contains only a few hidden layers (e.g., the number of layers is less than
or equal to 2) [41–43]. To enhance the ability of GCN’s network rep­
resentation, a residual strategy is adopted to relax the ‘over smoothing’
issue for multi-layer R-GCN(Fig. 2-B). Let the final embedding features
outputted by the Encoder **F** _e_ be **z** _i_ . For a _p_ -layer R-GCN, we set **z** _i_ as:



_**p**_
**z** _**i**_ = ∑

_**k**_ =2



**h** _[k]_ _i_ (2)



Notedly, this sum requires that the dimensions in different layers are
the same. Due to the first hidden layer accounts for the dimension

reduction of the high-dimensional one-hot features **h** [(] _i_ [0][)] [, the residual ]
strategy just starts the sum from the second hidden layer.
Moreover, it is anticipated that two interacting drugs are close in the
embedding space generated by **F** _e_ . Thus, possible interactions can be
deduced among those close drugs according to their embedding features

[43]. However, the existing interaction with missing label between two
drugs possibly causes their remoteness in the network. Missing in­
teractions between these drugs would aggravate the learning of **F** e .
Therefore, under the consideration that similar drugs tend to interact in
terms of chemical structures [2] or binding proteins [47], pre-defined
drug similarities, taken as a regularization item s i _,_ j ⋅z i − z j [2] 2 [, is ]
employed to constrain similar drugs as close as possible in the embed­
ding space(Fig. 2-B). Refer to Section Loss Function for details.


_2.4.2. Decoder_

Once the encoder **F** _e_ generates drug embedding features { **z** _**i**_ }, which
integrates topological information across interaction types, the decoder
**F** _d_ sequentially employs { **z** _**i**_ } to reconstruct the multiple relation DDI

network _G_ [̃] . In the case of binary DDI prediction, the inner production
**z** _**i**_ **z** _**[T]**_ _**j**_ [indicates how likely drug ] _[d]_ _[i ]_ [interacts with drug ] _[d]_ _[j]_ [. In order to reflect ]

the difference between interaction types, R-GCN employs **z** _**i**_ **M** _**r**_ **z** _**[T]**_ _**j**_ [to ]
calculate the likelihood of being a type-specific interaction, where { **M** _**r**_ }
are _m_ specific-type associative matrices. Inspired by literature [48], we
suppose that feature importance varies across interaction types, and we
also assume that interaction types are not completely independent to
each other. Therefore, our decoder **F** _d_ adopts a tensor factorization-like
matrix operation **z** _**i**_ **D** _**t**_ _r_ **RD** _**t**_ _r_ **z** _**[T]**_ _**j**_ [to calculate the type-specific interaction ]
likelihood. Thus, how likely the pair of drug _d_ _i_ and drug _d_ _j_ triggers an
_t_ _r_ -type pharmacological effect can be formally defined as the scoring
function:



(3)
)



( **z** _**i**_ **D** _**t**_ _r_ **RD** _**t**_ _r_ **z** _**[T]**_ _**j**_



_f_ ( _v_ _i_ _, r, v_ _j_



) = _σ_



⎛



1

_c_ _i,r_



_h_ [(] _i_ _[k]_ [+][1][)] = _σ_



⎛



_r_

⎝ [∑]



⎞

⎠



⎞



⎠ _,_ (1)



~~_w_~~ [(] _r_ _[k]_ [)] _[h]_ [(] _j_ _[k]_ [)] + _w_ [(] _r_ _[k]_ [)] _[h]_ _i_ [(] _[k]_ [)]



where **z** _**i**_ and **z** _**j**_ are the 1 × _k_ embedding vectors of drug nodes _v_ _i_ and _v_ _j_
respectively, **D** _**t**_ _r_ is a _k_ × _k_ feature importance diagonal matrices con­
cerning type _t_ _r_, **R** is a _k_ × _k_ feature association matrix across different
interaction types, and all of { **D** _**t**_ _r_ } and **R** (similar as matrices { **M** _**r**_ } in
traditional R-GCN) are parameter matrices that need to be learned
during the model training. _σ_ ( ⋅) is the Sigmoid function that converts the
confidence score of being an _t_ _r_ -type interaction into a probability value
of [0 _,_ 1].


_2.4.3. Loss function_
The encoder **F** _e_ and the decoder **F** _d_ can be trained as an end-to-end
model of multiple relation DDI prediction. The loss function of deep­
MDDI is composed of two components. The first one measures the dif­
ference between the original multiple relation interaction network _G_

and the reconstructed network _G_ [̃] . The second one is a regularization
item, which keeps the similar drugs as close as possible in the embedding

space.



_r_



⎝ [∑] _j_ ∈ _N_ _i_ _[r]_



_j_ ∈ _N_ _i_ _[r]_



where _c_ _i,r_ = ⃒⃒ _N_ _ri_ ⃒⃒ is a normalization constant, _N_ _ri_ [denotes the set of ] _[d]_ _[i]_ [’][s ]

neighbors in _G_ _r_, _h_ [(] _i_ _[k]_ [)] is the input feature vector, **w** [(] _r_ _[k]_ [)] is the trainable
weight matrix in the _k_ -th layer of _G_ _r_ in R-GCN, and σ is a non-linear
element-wise activation function (i.e., ReLU). Last, the aggregation
process is propagated through _p_ layers of R-GCN to obtain the final

embedding feature vector **h** [(] _i_ _[p]_ [)] of drug _d_ _i_ .
Such a multi-layer propagation of R-GCN enables the extraction of



4


_Y.-H. Feng et al._ _Analytical Biochemistry 646 (2022) 114631_



Let _a_ _[r]_
_ij_ [be the true label of a triplet ][(] _[v]_ _[i]_ _[,][ r][,][ v]_ _[j]_ [)][ for the pair of drug ] _[d]_ _[i ]_ [and ]

drug _d_ _j_ in the _t_ _r_ -th slice network _G_ _r_, and _p_ _[r]_ _ij_ [be the predicted probability ]

of being interaction of type _r_ . For the _t_ _r_ -th slice network _G_ _r_, its loss
function _L_ _[r]_
_ij_ [is defined by a binary cross-entropy as follows: ]



)(1 − log( _p_ _[r]_ _ij_



(4)
)))



_2.5. Strategy of multi-label prediction for new drugs_


There are two crucial scenarios in DDI prediction. The first is the
prediction of unobserved interactions between known drugs (trans­
ductive prediction), while the second is the prediction of the interactions
between known drugs and new drugs (inductive prediction). The orig­
inal version of our deepMDDI only can perform a transductive predic­
tion of multiple relation DDIs.
In contrast, since new drugs don’t attend the training or have no
known interaction with any approved drug, the interaction prediction of
new drugs is a task of inductive inference (Fig. 3). To cope with such a
task, we extended deepMDDI by utilizing extra properties (e.g., struc­
tures, targets, side effects) owned by both the approved and new drugs,
such that deepMDDI can perform the inductive prediction of multiple
relation DDIs.

Formally, suppose that n approved drugs and x investigational new
drugs. For approved drugs, let Z n be their n × k topological feature
matrix, which is learned from the encoder of deepMDDI, and S n be their
n × n similarity matrix (see also Section 2.3 Feature extraction **)**, which
can be regarded as an extra feature matrix. Let S x be the x × n similarity
matrix, which contains the pairwise similarity between investigational
drugs and approved drugs. Inspired by work [49], the key step to predict
interactions between approved drugs and investigational drugs is the
bridge between the topological feature matrix and the similarity matrix.
We build the bridge by a KNN regression (implemented by SciKit-learn
python package) which learns a mapping g(S n ) = Z n based on approved
drugs with default regression parameters. With the learned regression
model g, we can map S x to Z x, which is the estimated x × k topological
feature matrix of investigational drugs (Fig. 3). Once Z x is determined, a
tensor factorization-like matrix operation z x **D** _**t**_ _r_ R **D** _**t**_ _r_ z [T] n [(the decoder of ]
deepMDDI) is performed to infer how likely investigational drugs
interact with approved drugs regarding type-specific interactions.


_2.5.1. Assessment metrics_

To make a fair comparison with other approaches, which adopted 5fold cross-validation (5-CV) to evaluate the prediction in multiclassification (Section Comparison with other three existing models in



_L_ _[r]_ _ij_ [= −] ∑



) + (1 − _a_ _[r]_ _ij_



_i,j_



( _a_ _[r]_ _ij_ [log] ( _p_ _[r]_ _ij_



The positive samples are taken as the interactions in _G_ _r_ while the
negative samples are randomly sampled among its unlabeled drug pairs.
The number of negative samples is the same as that of positive samples.

_L_ =
For all the sliced networks, the global loss function is defined as

_m_
∑ _L_ _[r]_ [. ]



_r_ =1



_L_ _[r]_
_ij_ [. ]



Let **S** = { _s_ _i,j_ } ∈[0 _,_ 1] _, i, j_ = 1 _,_ 2 _,_ … _, n_ be the drug similarity matrix.
The regularization item is defined as:



_n_


2

_Reg_ = ∑( _s_ _i,j_ ⋅ **z** _i_ − **z** _j_ 2 ) (5)

_i,j_ =1


where **z** _i_ and **z** _j_ are the embedding representations of drug _d_ _i_ and drug _d_ _j_
generated by the encoder respectively. It can be written in an elegant
matrix form as follows:


_Reg_ = 2 _α_ ⋅tr( **Z** [T] **LZ** ) (6)



where **Z** is an _n_ × _k_ feature matrix stacked by feature vectors, **L** = **D** − **S**
is a Laplace matrix, **D** is an _n_ × _n_ diagonal matrix derived from **S** and its
element _D_ _i,i_ = [∑] _s_ _i,j_ _, i_ ∕= _j_ . This regularization item utilizes pre-defined

_j_

drug similarities to constrain similar drugs as close as possible in the
embedding space. This idea is similar to that in literature [48].
Therefore, the final loss of deepMDDI is as follows:



_j_



_s_ _i,j_ _, i_ ∕= _j_ . This regularization item utilizes pre-defined



_m_
_Loss_ = ∑ _L_ _ij_ _[r]_ [+][ 2] _[α]_ [⋅tr] ( **Z** [T] **LZ** ) (7)

_r_ =1


where α is a hyper parameter to adjust the weight of similarity constraint
in the training phase.



**Fig. 3.** Framework of multi-label prediction for new drugs. First, it employs a method of regression g(S n ) = Z n for constructing the bridge between the topological
feature matrix and the similarity matrix. And then map the similarity matrix S x of new drugs to their topological feature matrix Z x by parameters of function g. At last,
a tensor factorization-like matrix operation z x **D** _**t**_ _r_ R **D** _**t**_ _r_ z [T] n [(the decoder of deepMDDI) is performed to infer how likely new drugs interact with known drugs regarding ]
type-specific interactions.


5


_Y.-H. Feng et al._ _Analytical Biochemistry 646 (2022) 114631_


​


​


​



DDI multi-classification), we use the same split of the training set and
the testing set. The training set is used to train the learning model and
the testing set is used to measure the generalization performance of the
model on unlabeled data. In multi-label prediction (Section Performance
of deepMDDI in multi-label DDI prediction), we use 75% samples of the
DDIs datasets as the training set, 5% samples as the validation data, and
the remaining 20% samples as the testing data. In each experiment, the
splitting process is usually repeated 20 times with different random
seeds and the average performance of these repetitions is reported as the
final performance.
Since our task is a multi-label prediction problem, a group of metrics
is used to measure the prediction, including the area under the receiver
operating characteristic curve (AUC), the area under the precision-recall
curve (AUPR), Accuracy, Recall, Precision, and F1-score. Remarkably,
Recall, Precision, and F1-score have their macro versions and micro
versions, respectively. Macro metrics reflect the average performance
across different interaction types. For example, Macro Precision is
defined as the average of the Precision values of different interaction
types. In contrast, Micro metrics are analogous to corresponding metrics
in binary classification by summing the numbers of true positive, false
positive, true negative, and false negative samples across all interaction
types, respectively. Their definitions are as follows:


​


​


​



(8)


​ (9)


​ (10)


​



Accuracy = [1]
_l_


​


​


​



∑ _i_ = _l_ 1 _TP_ _i_ + _FPTP_ _ii_ + + _TN TN_ _ii_ + _FN_ _i_


​


​


​



Macro ​ Precision = [1]
_l_


​


​



_l_


​

∑

_i_ =1


​


​



_TP_ _i_


​ _TP_ _i_ + _FP_ _i_


​


​



​


_l_

Macro ​ Recall = [1] _l_ ∑ _i_ =1


​



​


_TP_ _i_


​ _TP_ _i_ + _FN_ _i_


​



​


​


Macro _F_ 1 = [2][ ×] _Macro Precision_ _[ macro Precision]_ + [ ×] _Macro Recall_ _[ Macro Recall]_ (11)


​



DDIs as the classifier. Differently, in terms of model architecture,
DeepDDI is a model for homogeneous interaction feature (i.e., chemical
structure) whereas both Lee’s model and DDIMDL are two models for
accommodating heterogeneous DDI features (e.g., pathway, GO terms,
and binding proteins).
Moreover, to cope with the high dimension of the DDI feature, they
utilized various tricks to enhance their models. DeepDDI [11] employed
the Principal Component Analysis (PCA) to reduce the feature dimen­
sion before training the nine-layer DNN. Lee et al. [35] first utilized
three three-layer autoencoders for three sources of raw DDI features
respectively, and then concatenated three sources of dimension-reduced
features as the training feature of the eight-layer DNN. DDIMDL [12]
trained four three-layer DNNs for four sources of DDI features respec­
tively, and averaged the individual predictions of those trained DNNs as
the final prediction.
These methods are designed for the multi-classification [50] of
single-fold DDIs where a drug pair is allowed to have only one interac­
tion type, of which each has a single exclusive type. They determine the
pharmacological effect type for a given DDI, while our deepMDDI ex­
ceeds the task with the direct discrimination of whether an unknown

drug pair results in one or more pharmacological effects of interest.
Thus, our deepMDDI is accommodated to the version of
multi-classification task. In detail, all DDIs are divided into training
samples and test samples. All the DDIs in each type are split into the
training set (80%) and the test set (20%) for each type. In detail, the
DDIs belonging to this DDI type are considered as the positive samples,
and the DDIs not belonging to this DDI type are considered as the

​ negative samples. We implemented DeepDDI, Lee’s model, and DDIMDL

with their published source codes and the default parameters. The
comparison results in Table 1 show that our deepMDDI achieves the best

​ performance with the significant improvements of 0.1–1.1%, 5.9–9.8%,

1.7–12.2%, 0.6–1.8%, and 1.7–12.2% against other state-of-the-art
methods in terms of Macro AUC, Macro AUPR, Micro Precision, Accu­
racy, and Micro F1-score respectively.


_3.2. Performance of deepMDDI in DDI multi-label prediction_


​

Existing methods only consider the DDI multi-classification, where a
DDI triggers only one pharmacological effect. However, ~20% DDIs are
of multi-fold interactions, where a DDI causes two or more related
pharmacological effects. Since the classification of multi-fold in­
teractions requires multiple classifiers, existing methods cannot handle
this task. Owning to the decoder, our deepMDDI is capable to address
the issue of predicting the multi-fold interactions. In this sense, we run a
DDI multi-label prediction to demonstrate the good predictive perfor­
mance of deepMDDI. In addition, for each type of interaction, we
sampled the same number of negative drug pairs (not having this
interaction type) as that of positive drug interactions (having this
interaction type) to train the model. The results in Table 2 show that our
deepMDDI can effectively predict the single-fold and multi-fold DDIs.
If there is more than one single type-specific interaction score be­
tween two drugs calculated by the decoder of deepMDDI is greater than
the threshold, this drug pair is considered to have multi-fold interactions
simultaneously. Besides performance measurement of each single-fold
DDIs prediction, we use macro metrics (average all single-fold perfor­
mances of each metric) to measure the overall performance of
deepMDDI.
In order to further verify the performance of our deepMDDI to pre­
dict the new DDIs and their interaction types in unknown DDIs, the
inspiring prediction impels us to perform a novel transductive inference
of potential DDIs among all drug pairs and their interaction types. Such
an inference validates the performance of deepMDDI in practice. To
accomplish this task, we first used the whole dataset with known DDIs to
train deepMDDI, and then we employed the trained deepMDDI to infer
how likely unlabeled drug pairs trigger specific pharmacological effects
among 11 interaction types. After that, we ranked these unlabeled drug



​


​


Micro ​ Precision =



​


​


_l_
∑ _i_ =1 _[TP]_ _[i]_


​ _l_

~~∑~~ _i_ =1 _[TP]_ _[i]_ [ +] _[ FP]_ _[i]_



​


​


​ (12)



​


​


​


_Micro F_ 1 = [2][ ×] _Micro Precision_ _[ micro Precision]_ + [ ×] _Micro Recall_ _[ Micro Recall]_ (13)


where _TP_ _i_, _TN_ _i_, _FP_ _i_ and _FN_ _i_ represent the number of true positive, true
negative, false positive and false negative samples in the _i_ -type DDI
prediction, respectively; _l_ is the number of DDI interaction types. In
addition, to considering both Precision and Recall, we selected the
threshold value, which achieves the maximum value of F1 in each
interaction type, as the type-specific threshold.


**3. Results and discussion**


We designed some experiments to address the following questions:
(1) Does deepMDDI improve DDI multi-classification? (2) Can deep­
MDDI achieve a good predictive performance in DDI multi-label pre­
diction? (3) How do both the residual strategy and the similarity
regularization in the encoder help the prediction?


_3.1. Comparison with other three existing models in DDI multi-_
_classification_


In order to answer the first question, we compared deepMDDI with
other three state-of-the-art DDI multi-classification models which are
deep learning-based models, including DeepDDI [11], Lee’s model [35],
and DDIMDL [12]. In common, these methods first treat rows in a drug
similar matrix as corresponding drug feature vectors, then set the
concatenation of two feature vectors as the feature vector of a DDI, and
the last train a multi-layer DNN with both feature vectors and types of



​


​


​


6


_Y.-H. Feng et al._ _Analytical Biochemistry 646 (2022) 114631_


**Table 1**
Results of deepMDDI and other three methods for DDI multi-classification.


Model Micro AUC Macro AUC Macro AUPR Micro Recall Micro Precision Accuracy Micro ​ F 1


DeepDDI 0.985 0.972 0.733 0.788 0.788 0.961 0.788

Lee’s **0.987** 0.982 0.694 0.683 0.683 0.949 0.683

DDIMDL 0.983 0.975 0.708 0.771 0.771 0.958 0.771

deepMDDI 0.982 **0.983** **0.792** **0.805** **0.805** **0.967** **0.805**


**Table 2**

Performance of deepMDDI for DDIs multi-label prediction.


Relation types AUC AUPR Accuracy Precision Recall F1


single-fold Absorption 0.989 0.982 0.967 0.954 0.982 0.968

Metabolism 0.948 0.932 0.898 0.850 0.943 0.894

Serum Concentration 0.965 0.938 0.937 0.920 0.971 0.939

Excretion 0.937 0.860 0.904 0.867 0.971 0.940

Activity Decrease 0.995 0.992 0.980 0.975 0.985 0.980
Activity Increase 0.986 0.979 0.959 0.944 0.977 0.960
Toxicity Activity 0.970 0.968 0.923 0.929 0.931 0.923

Adverse Effect 0.980 0.966 0.957 0.937 0.980 0.958

Antagonism Effect 0.992 0.989 0.972 0.963 0.982 0.978
Synergy Effect 0.995 0.993 0.9778 0.970 0.985 0.978
PD triggered by PK 0.990 0.986 0.968 0.954 0.984 0.969

multi-fold Macro metrics 0.977 0.962 0.949 0.933 0.972 0.953



​


pairs in each interaction type according to their type-specific predicting
scores. Finally, we picked up top-20 type-specific candidates in each
interaction type and validated them by both the latest version of
DrugBank (version 5.1.8, on January 18, 2021) and the online Drug
[Interaction Checker tool (Drugs.com).](http://Drugs.com)
The validation was performed in both single-fold interactions and
multi-fold interactions respectively. The detailed results are listed in
Table S4 (in supplementary). In the prediction results of single-fold in­
teractions, 40 out of 220 predicted DDI candidates (18.2%) are
confirmed. The average rank of 40 verified DDIs is 7.75, indicating that
our deepMDDI can effectively detect the potential DDIs as well as
different types of DDIs. We further picked up some validated DDI can­
didates (i.e., Case 31, Case 34, Case15 and Case 16) to show how DDI
prediction contributes to synergistic drug combination and drug
contraindication. For example, when two drugs of Pregabalin and Ben­
moxin are combined, the therapeutic efficacy of Benmoxin can be
increased (Case 31). In addition, the therapeutic efficacy of Mebanazine
can be increased when used in combination with Pregabalin (Case 34).
In contrast, the risk or severity of QTc prolongation can be increased
when Quinidine is combined with Promethazine (Case 15). Besides, the
risk or severity of serotonin syndrome can be increased when Linezolid
is combined with Ergotamine (Case 16). These results manifest that the
deepMDDI can provide a preliminary screening for synergistic drug
combination and drug contraindication.
In the prediction results of multi-fold interactions, 17 out of 50 twofold predicted candidates and 8 out of 60 three-fold predicted candidates
are confirmed, respectively. The detailed results are listed in Table S5 (in
supplementary). As illustrated, we picked up a two-fold interaction case
(Case 8) and a three-fold interaction case (Case 18) to show how
deepMDDI contributes to finding multi-fold interaction cases. For the
example of two-fold interaction, DrugBank states “Acebutolol may in­
crease the arrhythmogenic activities of Digoxin”, while DDI Checker
states “Using Acebutolol together with Digoxin may slow your heart rate
and lead to increased side effects.” (Case 8). Both statements show that
the pair of Digoxin and Acebutolol triggers an activity decrease and
further results in a PD adverse effect. For the example of three-fold
interaction, two statements are similarly found, but contain three
pharmacological effects as follows “Voriconazole may increase the
blood levels and effects of Trazodone” and “The risk or severity of QTc
prolongation can be increased when Trazodone is combined with Vor­
iconazole” (Case 18). The pair of Voriconazole and Trazodone increases



​


both PK serum and PD synergy of Trazodone, but also increases the risk
of adverse effects as well. In total, these newly-predicted DDIs demon­
strate the potentials of our deepMDDI in practice.


_3.3. Influence of hidden layers, residual strategy and similarity_
_regularization in encoder_


In this section, we investigated how three factors (i.e., the number of
hidden layers, the similarity regularization, and the residual strategy) in
the encoder affect the performance of deepMDDI. First, after removing
the similarity regularization and the residual strategy in deepMDDI, we
adopted deepMDDI two variants, that is, deepMDDI with 2 hidden layers
(denoted as deepMDDI-2) and 4 hidden layers (denoted as deepMDDI4). The two hidden layers in deepMDDI-2 contain [1024, 128] neu­
rons, and four hidden layers in deepMDDI-4 contain [1024, 128, 128,
128] neurons, respectively. On the architecture of deepMDDI-4, we
added the residual strategy to generate an additional variant of deep­
MDDI (denoted as deepMDDI-4-R). If the similarity regularization is
further added to deepMDDI-4-R, the variant of deepMDDI-4-R is the full
architecture of deepMDDI. The results of deepMDDI and its variants (i.
e., deepMDDI-2, deepMDDI-4 and deepMDDI-4-R) are shown in Table 3,
from which we can obtain three following crucial points.


(1) deepMDDI-4 is worse than deepMDDI-2 in all the measuring
metrics. Obviously, the increment of the number of hidden layers
decreases the predictive performance because of the “over
smoothing” issue derived from GCN.
(2) Compared with deepMDDI-2 and deepMDDI-4, deepMDDI-4-R
owing to the residual strategy achieves the significant improve­
ment. Thus, the residual strategy can relax the “over smoothing”
issue in the case of deeper GCN architecture.
(3) Compared with these variants, the full architecture of deepMDDI
having the additional similarity regularization further improves
the prediction. Thus, the similarity regularization helps constrain
similar drugs as close as possible in the embedding space to cope
with the issue that missing interaction label between similar
drugs causes their remoteness in the network.


In summary, with the help of residual strategy, deepMDDI can
accommodate deep GCN architecture (e.g., containing _>_ 2 layers). Also,
its similarity regularization further helps capture missing interactions.



​


7


_Y.-H. Feng et al._ _Analytical Biochemistry 646 (2022) 114631_


**Table 3**

Performance of similarity constraint and residual strategy in deepMDDI.


Variant Macro AUC Macro AUPR Accuracy Macro Precision Macro Recall Macro F1 AP@50


deepMDDI-2 0.953 0.913 0.906 0.881 0.958 0.916 0.909
deepMDDI-4 0.928 0.901 0.878 0.847 0.925 0.885 0.838
deepMDDI-4-R 0.971 0.954 0.948 0.927 **0.973** 0.949 **0.959**
deepMDDI **0.977** **0.962** **0.949** **0.933** 0.972 **0.953** 0.951



_3.4. Influence of different implementations in decoder_


Since the decoder in deepMDDI is loosely coupled with the encoder,
we should adopt various decoder models. In this section, we compared
three implementations of the decoder, including the inner production
**z** _**i**_ **z** _**[T]**_ _**j**_ [in the traditional GCN, the type-specific association ] **[z]** _**[i]**_ **[M]** _**[r]**_ **[z]** _**[T]**_ _**j**_ [in R- ]

GCN, as well as our type-specific importance association z i **D** _**t**_ _r_ R **D** _**t**_ _r_ z [T] j [. ]

According to their original algorithms, these three implementations are
denoted as InnerProd, RESCAL, and DEDICOM, respectively. See Section
Decoder for details.

The comparison results in Table 4 show that InnerProd is the worst
and DEDICOM is the best. The potential reason for DEDICOM signifi­
cantly outperforming two other models is as follows. The inner pro­
duction **z** _**i**_ **z** _**[T]**_ _**j**_ [only indicates how likely drug ] _[d]_ _[i ]_ [interacts with drug ] _[d]_ _[j]_ [, but ]
it cannot model interaction types. In contrast, RESCAL reflects the dif­
ference between interaction types and models the likelihood of being a

_m_
type-specific interaction by additional type-specific feature associa­
tion matrices { **M** _**r**_ }. Compared with RECAL, to indicate how likely the
pair of drug _d_ _i_ and drug _d_ _j_ triggers an _t_ _r_ -type pharmacological effect,
DEDICOM employs a global feature association matrix **R**, as well as _m_
additional type-specific diagonal matrices { **D** _**t**_ _r_ }, which reflects that
feature importance varies across interaction types.


_3.5. Performance of deepMDDI in DDI multi-label prediction for new_
_drugs_


Our model deepMDDI can not only predict unobserved interactions
between known drugs (the transductive scenario) but also the in­
teractions between known drugs and new drugs (the inductive scenario)
with a simple modification (Section Strategy of DDI multi-label predic­
tion for new drugs)
. In the transductive scenario, different from the edge split strategy in
the first scenario (Section Assessment), the second scenario requires the
splitting in terms of drugs. In detail, we randomly split all drugs into a
training set (80%) and a testing set (20%), and then the interactions
among the training drugs are selected as the training samples. The
testing drugs are regarded as new drugs, and only the interactions be­
tween the testing drugs and the training drugs are selected as the testing
samples, which are blind to the training drugs. The splitting process is
usually repeated 20 times with different random seeds. The average
performance of these repetitions is reported as the final performance.
The results in Table 5 show that our deepMDDI can effectively predict
DDI for new drugs.
We also perform a novel inductive inference of potential DDIs be­
tween known drugs and five new drugs in the investigational or exper­
imental process selected randomly from DrugBank. In detail, we first
used the whole dataset with known DDIs to train deepMDDI, and then
mapped the similarity matrix of known drugs to their topological feature
matrix for the regression model. Then, based on the regression model,


**Table 4**

Performance of different implementations in decoder.



the similarity matrix of new drugs is mapped to their topological feature
matrix. Last, a tensor factorization-like matrix operation (the decoder of
deepMDDI) is performed to infer how likely new drugs interact with
approved drugs regarding type-specific interactions (Section Decoder
for details). After that, we picked up the top-5 type-specific candidates in
each interaction type, and further validated them by the latest version of
DrugBank (version 5.1.8, on January 18, 2021). There are total six
predicted DDI candidates are confirmed. The results indicate deepMDDI
can effectively predict the potential DDIs as well as different types be­
tween known drugs and new drugs. The detailed results are listed in
Table S6 (in supplementary). All parameters involved in deepMDDI are
presented in Section Parameter settings of supplement file.


**4. Conclusions**


In this work, we proposed a novel end-to-end deep learning-based
method (called deepMDDI) to predict DDIs as well as their types, help­
ing to explore the underlying mechanism of DDIs. deepMDDI designs an
encoder by the deep relational graph convolutional networks con­
straining with similarity regularization to capture the topological fea­
tures of DDI network, and employs a tensor-like decoder to uniformly
model both single-fold interactions and multi-fold interactions. Thus,
deepMDDI can effectively discriminate whether an unlabeled typespecific drug pair in DDI network results in one or more pharmacolog­
ical effects, and also predict the potential DDIs between the known drugs
in the DDI network and new drugs outside the network. The superiority
of our deepMDDI is demonstrated by comparing it with state-of-the-art
deep learning-based methods in the task of DDI multi-classification.
deepMDDI achieves an inspiring performance for predicting DDIs in
the case of both single-fold DDIs and multi-fold DDIs, and its power of
predicting DDIs is further validated in the case study by the latest
version of DrugBank and an online Drug Interaction Checker tool.
deepMDDI is beneficial for the inference of drug combinations in
treating complex diseases.


**Declarations**


Ethics approval and consent to participate.
No ethics approval was required for the study.


**Consent for publication**


Not applicable.


**Availability of data and materials**


The datasets generated and analyzed during the current study and
[the code of MTDDI are openly available at the website of https://github.](https://github.com/NWPU-903PR/MTDDI)

[com/NWPU-903PR/MTDDI.](https://github.com/NWPU-903PR/MTDDI)



Model AUC Macro AUPR Accuracy Macro Precision Macro Recall Macro F1 AP@50


InnerProd 0.634 0.588 0.591 0.571 0.947 0.704 0.125

RESCAL 0.896 0.879 0.832 0.797 0.917 0.850 0.869

DEDICOM **0.977** **0.962** **0.949** **0.933** **0.972** **0.953** **0.951**


8


_Y.-H. Feng et al._ _Analytical Biochemistry 646 (2022) 114631_


**Table 5**

Performance of deepMDDI for DDIs multi-label prediction for new drugs.


Relation types AUC AUPR Accuracy Precision Recall F1


single-fold Absorption 0.880 0.910 0.829 0.878 0.764 0.817

Metabolism 0.860 0.882 0.776 0.766 0.795 0.780

Serum Concentration 0.772 0.818 0.696 0.684 0.729 0.706

Excretion 0.742 0.778 0.669 0.646 0.748 0.694

Activity Decrease 0.861 0.874 0.778 0.756 0.821 0.787
Activity Increase 0.842 0.855 0.777 0.774 0.784 0.779
Toxicity Activity 0.876 0.874 0.802 0.777 0.849 0.811

Adverse Effect 0.867 0.889 0.787 0.787 0.786 0.786

Antagonism Effect 0.868 0.891 0.784 0.779 0.793 0.786
Synergy Effect 0.870 0.902 0.813 0.840 0.774 0.806
PD triggered by PK 0.838 0.842 0.762 0.734 0.821 0.776

multi-fold Macro metrics 0.843 0.865 0.770 0.765 0.788 0.775



**Funding**


This work has been supported by the National Natural Science
Foundation of China (No.62173271, PI：SWZ, No. 61873202, PI：SWZ
and No. 61872297, PI: JYS) and Shaanxi Provincial Key R&D Program,
China (No. 2020 KW-063, PI: JYS). The funding body did not play any
roles in the design of the study, collection, analysis, and interpretation of
data, and in writing the manuscript.


**CRediT authorship contribution statement**


**Yue-Hua Feng:** Methodology, performed the experiments, Writing –
original draft. **Shao-Wu Zhang:** modified manuscript. **Qing-Qing**
**Zhang:** collected the datasets. **Chu-Han Zhang:** Data curation. **Jian-Yu**
**Shi:**
Formal analysis, modified manuscript, All authors read and
approved the final manuscript.


**Declaration of competing interest**


None of the authors has any competing interests.


**Acknowledgments**


We acknowledge anonymous reviewers for the valuable comments
on the original manuscript.


**Abbreviations**


DDIs Drug-Drug Interactions
GCN Graph Convolution Network
DBPs Drug-Binding Proteins
AUC Area Under the receiver operating characteristic Curve
AUPR Area Under the Precision-Recall curve

ACC ACCuracy
R-GCN Relational Graph Convolution Network

SMILES
Simplified Molecular Input Line Entry System
MACCSkeys Molecular ACCess System keys
PK PharmacoKinetic

PD PharmacoDynamic


**Appendix A. Supplementary data**


[Supplementary data to this article can be found online at https://doi.](https://doi.org/10.1016/j.ab.2022.114631)
[org/10.1016/j.ab.2022.114631.](https://doi.org/10.1016/j.ab.2022.114631)


**References**


[[1] F. Cheng, I. K, A.L. Barab´asi, Network-based prediction of drug combinations, Nat.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref1)
[Commun. 10 (1) (2019) 1197.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref1)

[[2] S.R. Niu J, D.E. Mager, Pharmacodynamic drug-drug interactions, Clin. Pharmacol.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref2)
[Ther. 105 (6) (2019) 1395–1406.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref2)




[[3] M. Sun, et al., Graph convolutional networks for computational drug development](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref3)
[and discovery, Briefings Bioinf. 21 (3) (2020) 919–935.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref3)

[[4] T. Takeda, et al., Predicting drug-drug interactions through drug structural](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref4)
[similarities and interaction networks incorporating pharmacokinetics and](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref4)
[pharmacodynamics knowledge, J. Cheminf. 9 (2017) 16.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref4)

[[5] D. Sridhar, S. Fakhraei, Getoor, A probabilistic approach for collective similarity-](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref5)
[based drug-drug interaction prediction, Bioinformatics 32 (20) (2016) 3175–3182.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref5)

[[6] W. Zhang, et al., Predicting potential drug-drug interactions by integrating](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref6)
[chemical, biological, phenotypic and network data, BMC Bioinf. 18 (1) (2017) 18.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref6)

[[7] Z. Wen, et al., SFLLN: a sparse feature learning ensemble method with linear](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref7)
[neighborhood regularization for predicting drug–drug interactions, J. Inf. Sci. 497](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref7)
[(2019) 189–201.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref7)

[[8] K. Andrej, et al., Predicting potential drug-drug interactions on topological and](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref8)
[semantic similarity features using statistical learning, PLoS One 13 (5) (2018)](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref8)
[e0196865.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref8)

[[9] H. Yu, et al., Predicting and understanding comprehensive drug-drug interactions](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref9)
[via semi-nonnegative matrix factorization, BMC Syst. Biol. 12 (Suppl 1) (2018) 14.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref9)

[[10] A. Gottlieb, et al., INDI: a computational framework for inferring drug interactions](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref10)
[and their associated recommendations, Mol. Syst. Biol. 8 (1) (2012).](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref10)

[[11] J.Y. Ryu, H.U. Kim, S.Y. Lee, Deep learning improves prediction of drug-drug and](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref11)
[drug-food interactions, Proc. Natl. Acad. Sci. U. S. A. 115 (18) (2018)](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref11)
[E4304–E4311.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref11)

[[12] Y. Deng, et al., A multimodal deep learning framework for predicting drug-drug](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref12)
[interaction events, Bioinformatics 36 (15) (2020) 4316–4322.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref12)

[[13] F. Cheng, Z. Zhao, Machine learning-based prediction of drug-drug interactions by](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref13)
[integrating drug phenotypic, therapeutic, chemical, and genomic properties, J. Am.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref13)
[Med. Inf. Assoc. 21 (e2) (2014) e278–e286.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref13)

[[14] P. Zhang, et al., Label propagation prediction of drug-drug interactions based on](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref14)
[clinical side effects, Sci. Rep. 5 (1) (2015) 12339.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref14)

[[15] G. Shtar, L. Rokach, B. Shapira, Detecting drug-drug interactions using artificial](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref15)
[neural networks and classic graph similarity measures, PLoS One 14 (8) (2019)](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref15)
[e0219796.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref15)

[[16] J.-Y. Shi, et al., Detecting drug communities and predicting comprehensive](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref16)
[drug–drug interactions via balance regularized semi-nonnegative matrix](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref16)
[factorization, J. Cheminf. 11 (1) (2019).](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref16)

[[17] R. Ferdousi, R. Safdari, Y. Omidi, Comput. predic. drug-drug interact. based on](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref17)
[drugs funct. similar. 70 (2017) 54.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref17)

[[18] N. Rohani, C. Eslahchi, A. Katanforoush, ISCMF: integrated similarity-constrained](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref18)
[matrix factorization for drug–drug interaction prediction, Network Model. Analy.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref18)
[Health Inform. Bioinform. 9 (1) (2020).](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref18)

[[19] Y. Jing, et al., Deep learning for drug design: an artificial intelligence paradigm for](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref19)
[drug discovery in the big data era, AAPS J. 20 (4) (2018) 79.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref19)

[[20] Y. Zhang, et al., Predicting drug-drug interactions using multi-modal deep auto-](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref20)
[encoders based network embedding and positive-unlabeled learning, Methods 179](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref20)
[(2020) 37–46.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref20)

[[21] S.S. Deepika, T.V. J, J.o.B.I. Geetha, A meta-learning framework using](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref21)
[representation learning to predict drug-drug interaction, J. Biomed. Inf. 84 (84)](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref21)
[(2018) 136–147.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref21)

[[22] Y. LeCun, Y. Bengio, G. Hinton, Deep learning, Nature 521 (7553) (2015) 436–444.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref22)

[[23] J. Ma, et al., Deep neural nets as a method for quantitative structure-activity](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref23)
[relationships, J. Chem. Inf. Model. 55 (2) (2015) 263–274.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref23)

[[24] E.N. Feinberg, et al., Improvement in ADMET prediction with multitask deep](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref24)
[featurization, J. Med. Chem. 63 (16) (2020) 8835–8848.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref24)

[[25] F. Montanari, et al., Modeling physico-chemical ADMET endpoints with multitask](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref25)
[graph convolutional networks, Molecules 25 (1) (2019).](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref25)

[[26] B. Ramsundar, et al., Is multitask deep learning practical for pharma? J. Chem. Inf.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref26)
[Model. 57 (8) (2017) 2068–2076.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref26)

[[27] S.K. Mohamed, V. Novacek, A. Nounu, Discovering protein drug targets using](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref27)
[knowledge graph embeddings, Bioinformatics 36 (2) (2020) 603–610.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref27)

[[28] S. Pittala, et al., Relation-weighted Link Prediction for Disease Gene Identification](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref28) _._
[arXiv Preprint arXiv:2011.05138, 2020, 2020.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref28)

[[29] A. Zhavoronkov, et al., Deep learning enables rapid identification of potent DDR1](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref29)
[kinase inhibitors, Nat. Biotechnol. 37 (9) (2019) 1038–1040.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref29)

[[30] P. Gainza, et al., Deciphering interaction fingerprints from protein molecular](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref30)
[surfaces using geometric deep learning, Nat. Methods 17 (2) (2020) 184–192.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref30)



9


_Y.-H. Feng et al._ _Analytical Biochemistry 646 (2022) 114631_




[[31] T. Wen, R.B. J, o.C.I. J, Altman, and modeling, Graph Convolutional Neural](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref31)
[Networks for Predicting Drug-Target Interactions 59, 2019, pp. 4131–4149, 10.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref31)

[[32] Z. Wang, M. Zhou, C. Arnold, Toward heterogeneous information fusion: bipartite](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref32)
[graph convolutional networks for in silico drug repurposing, Bioinformatics 36](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref32)
[(Suppl_1) (2020) i525–i533.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref32)

[[33] M. Zitnik, M. Agrawal, J. Leskovec, Modeling polypharmacy side effects with graph](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref33)
[convolutional networks, Bioinformatics 34 (13) (2018) i457–i466.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref33)

[[34] A.K. Nyamabo, H. Yu, J.Y. Shi, SSI-DDI: Substructure-Substructure Interactions for](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref34)
[Drug-Drug Interaction Prediction, Brief Bioinform, 2021.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref34)

[[35] P.C. Lee G, J. Ahn, Novel deep learning model for more accurate prediction of](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref35)
[drug-drug interaction effects, BMC Bioinf. 20 (2019) 415.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref35)

[[36] D.S. Wishart, et al., DrugBank 5.0: a major update to the DrugBank database for](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref36)
[2018, Nucleic Acids Res. 46 (D1) (2017) D1074–D1082.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref36)

[[37] S.W. Zhang, et al., PPLook: an Automated Data Mining Tool for Protein-Protein](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref37)
[Interaction, 11, 2010, 1): p. 326-326.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref37)

[[38] H. Seung, Lee, et al., Pharmacokinetic and pharmacodynamic insights from](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref38)
[microfluidic intestine-on-a-chip models, Expet Opin. Drug Metabol. Toxicol. 15](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref38)
[(12) (2019) 1005–1019.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref38)

[[39] D. Rogers, M. H, Extended-connectivity fingerprints, J. Chem. Inf. Model. 50 (5)](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref39)
[(2010) 742–754.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref39)

[[40] M. Levandowsky, D. Winter, Distance between sets, Nature 234 (5323) (1971)](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref40)
[34–35.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref40)




[[41] M. Schlichtkrull, T.N. K, P. Bloem, et al., Modeling Relational Data with Graph](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref41)
[Convolutional Networks, Springer, Cham, 2017.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref41)

[[42] T.N. Kipf, M. Welling, Semi-supervised Classification with Graph Convolutional](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref42)
[Networks, 02907, 2016.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref42) **arXiv:1609** .

[[43] T.N. Kipf, M. Welling, Variational Graph Auto-Encoders, 07308, 2016.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref43) **arXiv:1611** .

[[44] M. Nickel, V. Tresp, H.P. Kriegel, A three-way model for collective learning on](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref44)
[multi-relational data, in: International Conference on International Conference on](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref44)
[Machine Learning, 2011.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref44)

[[45] C.F. Ee Papalexakis, N.D. Sidiropoulos, Tensors for data mining and data fusion:](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref45)
[models, applications, and scalable algorithms, ACM Trans. Intell. Syst. Technol. 8](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref45)
[(2157–6904) (2016) 44.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref45)

[[46] X. Yue, Z. W, J. Huang, et al., Graph embedding on biomedical networks: methods,](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref46)
[applications, and evaluations, Bioinformatics 36 (Suppl 1) (2019).](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref46)

[[47] Z. Yu, F. H, X. Zhao, W. Xiao, W. Zhang, Predicting Drug-Disease Associations](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref47)
[through Layer Attention Graph Convolutional Network, Brief Bioinform., 2020,](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref47)
[p. 243.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref47)

[[48] D. Wang, C. Peng, W. Zhu, Structural deep network embedding, in: Acm Sigkdd](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref48)
[International Conference on Knowledge Discovery & Data Mining, 2016.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref48)

[[49] J.Y. Shi, et al., Detecting drug communities and predicting comprehensive drug-](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref50)
[drug interactions via balance regularized semi-nonnegative matrix factorization,](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref50)
[J. Cheminf. 11 (1) (2019) 28.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref50)

[[50] G. Tsoumakas, I. Katakis, Multi-label classification: an overview, Int. J. Data](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref49)
[Warehous. Min. 3 (3) (2009) 1–13.](http://refhub.elsevier.com/S0003-2697(22)00087-2/sref49)



10


