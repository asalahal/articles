IEEE TRANSACTIONS ON EMERGING TOPICS IN COMPUTING, VOL. 12, NO. 1, JANUARY-MARCH 2024 163

## Fusing Higher and Lower-Order Biological Information for Drug Repositioning via Graph Representation Learning


Bo-Wei Zhao, Lei Wang, Peng-Wei Hu, Leon Wong, Xiao-Rui Su, Bao-Quan Wang,
Zhu-Hong You _, Member, IEEE_, and Lun Hu _, Member, IEEE_



_**Abstract**_ **—Drug repositioning is a promising drug devel-**
**opment technique to identify new indications for existing**
**drugs. However, existing computational models only make**
**use of lower-order biological information at the level of**
**individual drugs, diseases and their associations, but few**
**of them can take into account higher-order connectivity**
**patterns presented in biological heterogeneous information**
**networks (HINs). In this work, we propose a novel graph**
**representation learning model, namely FuHLDR, for drug**
**repositioning by fusing higher and lower-order biological**
**information. Specifically, given a HIN, FuHLDR first learns**
**the representations of drugs and diseases at a lower-order**
**level by considering their biological attributes and drug-**
**disease associations (DDAs) through a graph convolutional**
**network model. Then, a meta-path-based strategy is de-**
**signed to obtain their higher-order representations involv-**
**ing the associations among drugs, proteins and diseases.**
**Their integrated representations are thus determined by**
**fusing higher and lower-order representations, and finally**
**a Random Vector Functional Link Network is employed by**
**FuHLDR to identify novel DDAs. Experimental results on**
**two benchmark datasets demonstrate that FuHLDR per-**
**forms better than several state-of-the-art drug reposition-**
**ing models. Furthermore, our case studies on Alzheimer’s**


Manuscript received 1 May 2022; revised 30 September 2022; accepted 16 January 2023. Date of publication 31 January 2023; date
of current version 15 March 2024. This work was supported in part by
the Natural Science Foundation of Xinjiang Uygur Autonomous Region
under Grant 2021D01D05, in part by the Pioneer Hundred Talents Program of Chinese Academy of Sciences, in part by the National Natural
Science Foundation of China under Grants 62172355 and 61702444, in
part by Awardee of the NSFC Excellent Young Scholars Program under
Grant 61722212, in part by Tianshan youth - Excellent Youth under
Grant 2019Q029, in part by Tianchi Doctoral Program of Xinjiang Uygur
Autonomous Region, and in part by the Qingtan scholar talent project
of Zaozhuang University. _(Corresponding authors: Zhu-Hong You; Lun_
_Hu.)_
Bo-Wei Zhao, Peng-Wei Hu, Xiao-Rui Su, Bao-Quan Wang, and
Lun Hu are with the Xinjiang Technical Institute of Physics and
Chemistry, Chinese Academy of Sciences, Urumqi 830011, China (e[mail: zhaobowei19@mails.ucas.ac.cn; hupengwei@hotmail.com; sux-](mailto:zhaobowei19@mails.ucas.ac.cn)
[iaorui19@mails.ucas.ac.cn; wangbq@ms.xjb.ac.cn; hulun@ms.xjb.ac.](mailto:suxiaorui19)
[cn).](mailto:hulun@ms.xjb.ac.cn)
Lei Wang and Leon Wong are with the Big Data and Intelligent
Computing Research Center, Guangxi Academy of Sciences, Nanning
[530007, China (e-mail: leiwang@gxas.cn; lghuang@gxas.cn).](mailto:leiwang@gxas.cn)
Zhu-Hong You is with the School of Computer Science, Northwestern Polytechnical University, Xi’an 710129, China (e-mail:
[zhuhongyou@nwpu.edu.cn).](mailto:zhuhongyou@nwpu.edu.cn)
This article has supplementary downloadable material available at
[https://doi.org/10.1109/TETC.2023.3239949, provided by the authors.](https://doi.org/10.1109/TETC.2023.3239949)
Digital Object Identifier 10.1109/TETC.2023.3239949



**disease and Breast neoplasms indicate that the rich higher-**
**order biological information gains new insight into drug**
**repositioning with improved accuracy.**


_**Index Terms**_ **—Drug repositioning, drug-disease associ-**
**ation, higher and lower-order information, information fu-**
**sion, graph representation learning.**


I. I NTRODUCTION

S A promising drug development strategy, drug reposi# A tioning or drug repurposing, aims to identify new clinical

effects for approved drugs in relatively short-time and low-cost,
and thereby has attracted more attention. Drug repositioning
techniques have been widely applied in the treatment for a
variety of diseases [1]. Recently, due to the global spread of
corona virus disease 2019 (COVID-19), many researchers have
attempted to utilize approved drugs to treat COVID-19 in the
absence of specific medicine. Taking Remdesivir as an example,
evidence collected from clinical trials indicates its potential
ability to be a therapeutic drug for COVID-19 [2].
With the explosive growth of large-scale biological and medical data, a variety of computational models have been specifically proposed for drug repositioning, and they are regarded
as a complementary, yet effective, tool for discovering new
indications of approved drugs [3]. According to the difference in
their principle learning objectives, these computational models
are mainly divided into three categories, i.e., network-based,
recommendation system-based and machine learning-based
models [4].
Network-based models predict unknown DDAs by considering the structural information of DDA networks. For example,
Luo et al. [5] adopt a bi-random walk model to predict the hidden
relationshipsbetweendrugsanddiseasesbyrandomlytraversing
a DDA network. DRRS [6] is a heterogeneous network-based
model, and it makes use of a fast Singular Value Thresholding
algorithmtoscorepairsofdrugsanddiseaseswhoseassociations
are unknown. Recommendation system-based algorithms consider DDAs as the relationships between users and items, and
identify diseases to which drugs are most likely to be related
based on known effects of drugs. For example, NRLMF [7]
combines logistic matrix factorization and neighborhood regularization to discover novel DDAs. SCMFDD [8] obtains the
features of drugs and diseases for DDA prediction through



2168-6750 © 2023 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.


Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:26:36 UTC from IEEE Xplore. Restrictions apply.


164 IEEE TRANSACTIONS ON EMERGING TOPICS IN COMPUTING, VOL. 12, NO. 1, JANUARY-MARCH 2024



similarity-constrained matrix factorization. Dai et al. [9] present
a novel recommendation system-based model to predict novel
drug indications based on the features of drugs and diseases
extracted from the factorization of a DDA matrix. Although
effective, these models are not applicable to make an accurate
prediction for new drugs or diseases, as these models seriously
rely on the known relationships between drugs and diseases [10].

As an alternative way to network-based and recommendation
system-based models, machine learning-based models consider
the drug repurposing problem as a binary classification issue
and make use of different machine learning models to predict unknownDDAs [8].Gottliebetal. [11] firstcalculatethedrug-drug
and disease-disease similarities with respective measures, and
then combine them as the input features of a Logistic Regression
(LR) classifier to predict unknown DDAs. PreDR [12] presents
variousmethodstocalculatethepropertiesofdrugsanddiseases,
and then constructs a Support Vector Machine (SVM) classifier
with a specific kernel function for DDA prediction.

Recently, multi-information fusion techniques have been
widely applied to predict novel DDAs on a vast amount of
biological heterogeneous information [13]. Wang et al. [14]
first construct a heterogeneous information network (HIN) by
integrating drug-disease, drug-protein, protein-protein, proteindisease networks, and then utilize a graph convolution network
(GCN) model to obtain the embeddings of drug-disease pairs,
which are used to compute the association probability through
a Multilayer Perceptron layer. NIMCGCN [15] combines the
similarity networks of miRNA-miRNA and disease-disease to
capture their feature representations with GCN, and then adopts
a neural inductive matrix completion strategy to infer unknown
miRNA-disease associations. LAGCN [16] obtains the feature
vectors of drugs and diseases by a graph convolution algorithm on multiple networks, and then an attention mechanism
is applied to integrate these feature vectors for predicting new
associations. DeepR2cov [17] first learns low-dimensional vectors of drugs by deep neural networks, and then uses these
representation vectors to discover candidates. DRHGCN [18]
is also a heterogeneous information fusion algorithm for drug
repositioning based on GCN. It extracts different topological
features from a DDA network and the similarity networks of
drug-drug and disease-disease, and fuses them to improve the
accuracy of drug repositioning.

However, existing computational models only make use of
lower-order biological information at the level of individual
molecules and associations, but few of them can take into
account higher-order connectivity patterns presented in a biological HIN. In particular, due to the lack of insufficient DDAs
for drug repositioning [19], it is necessary for us to additionally
integrate other kinds of associations, such as drug-protein and
protein-disease associations considered in our work, such that a
more informative HIN can be constructed for drug repositioning.
Obviously, the introduction of additional associations further
enhancestheconnectivityofHINandsoastoenrichitsstructural
information. Moreover, the sparsity of HINs further contains the
performance of network-based models, as they only make use
of lower-order biological information at the level of individual
drugs, diseases and their associations. It has been pointed out
by [20] that higher-order connectivity patterns involving more



than two nodes are of significance to conduct an accurate network analysis. Hence, we intend to explore the possibility of
fusing higher and lower-order information observed in a HIN
to obtain a better representation of drugs and diseases, thus
achieving the improved performance of drug repositioning.

In this work, a novel prediction algorithm, namely FuHLDR,
is proposed to effectively learn the representations of drugs and
diseases by making use of higher and lower-order information
in a HIN with different graph representation learning models,
as shown in Fig. 1. To begin with, FuHLDR takes advantage
of a graph convolutional network (GCN) to obtain the lowerorder representations of drugs and diseases by aggregating their
neighboring information in a HIN. Furthermore, several meta
paths are specifically designed by FuHLDR to define the higherorder connectivity patterns of interest, and a meta-path-based
graph representation learning model, i.e., metapath2vec [21], is
adopted to learn the higher-order representations of drugs and
diseases in the light of meta paths. Finally, FuHLDR employs a
Random Vector Functional Link Network (RVFL) model as its
classifier to complete the prediction task of drug repositioning by
fusing the higher and lower-order representations of drugs and
diseases. The main contributions of our work are summarized

as follows.

r Higher and lower-order biological information presented

in HIN are taken into account simultaneously to better
learn the representations of drugs and diseases with different graph representation learning models.
r An effective drug repositioning algorithm, namely
FuHLDR, is proposed to precisely identify novel DDAs
by fusing the higher and lower-order representations of
drugs and diseases through an RVFL classifier.
r Experimental results on two benchmark datasets demon
strate that FuHLDR performs better than several state-ofthe-art drug repositioning algorithms under ten-fold crossvalidation. Furthermore, our case studies on Alzheimer’s
disease and Breast neoplasms indicate that the rich higherorder biological information gains new insight into drug
repositioning with improved accuracy.


II. M ETHODS


FuHLDR is composed of three steps, including the construction of biological attribute matrices, higher and lower-order
representation learning, and DDA prediction. Before describing
the details of these three steps, we first present the definition of
HIN involved in our work as below.


_A. Definition of HIN_


TobetterdescribetheframeworkofFuHLDR,athree-element
tuple, i.e., _HIN_ ( **V** _,_ **E** _,_ **C** ), is defined for HIN of interest. In
particular, **V** = _{V_ _[dr]_ _, V_ _[di]_ _, V_ _[pr]_ _}_ is a set of all drugs ( _V_ _[dr]_ ),
diseases ( _V_ _[di]_ ) andproteins ( _V_ _[pr]_ ) thatareinvolvedtoconstructa
HIN, **E** = _{E_ _[dd]_ _, E_ _[dp]_ _, E_ _[pd]_ _}_ denotes all links in the HIN, including DDAs ( _E_ _[dd]_ ), drug-protein associations ( _E_ _[dp]_ ) and proteindisease associations ( _E_ _[pd]_ ), **C** = [ **C** _[dr]_ _,_ **C** _[di]_ _,_ **C** _[pr]_ ] _[T]_ _∈_ _R_ _[|]_ **[V]** _[|×][d]_ [1],
where _|_ **V** _|_ is the size of **V**, represents the integration of biological attribute matrices for all nodes in **V**, Moreover, _N_, _M_ and _P_
are used to denote the sizes of _V_ _[dr]_, _V_ _[di]_ and _V_ _[pr]_ respectively. _T_



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:26:36 UTC from IEEE Xplore. Restrictions apply.


ZHAO et al.: FUSING HIGHER AND LOWER-ORDER BIOLOGICAL INFORMATION FOR DRUG REPOSITIONING 165


Fig. 1. The overall framework of FuHLDR. Given a HIN that contains three kinds of biomolecules, i.e., drugs, proteins and diseases, and their
associations, a biological attribute matrix **C** is first constructed by calculating the similarities of biomolecules with their biological knowledge. Then,
FuHLDR learns the lower-order representations of drugs and diseases with a graph convolutional network model. After that, a meta-path-based
strategy is designed to define higher-order connectivity patterns of interest, and a heterogeneous Skip-Gram model is then used to obtain higherorder representations of drugs and diseases. Finally, FuHLDR integrates higher and lower-order representations of drugs and diseases to infer new
DDAs by using a Random Vector Functional Link Network.



represents a set of possible node types and in our work, we have
**T** = _{drug, protein, disease}_ . For an arbitrary node _v_ _i_ _∈_ **V**,
its corresponding node type is denoted as _t_ _i_ _∈_ **T** .


_B. Constructing_ **C**


To construct **C**, three kinds of biological attributes are
collected from different databases and tools. Specifically,
**C** _[dr]_ is the attribute matrix of drugs, and its elements are
the molecular fingerprints of drugs obtained by the RDkit toolkit [22] to deal with the Simplified Molecular Input
Line Entry System (SMILES) [23] derived from DrugBank
database [24]; **C** _[di]_ is the semantic similarity matrix of diseases constructed with the data of Medical Subject heading
(MeSH) descriptor [25] that can be downloaded in the online
(https://www.nlm.nih.gov/mesh/meshhome.html) and ontology
similarity [26]; **C** _[pr]_ is the sequence attribute matrix of proteins
obtained by means of the 3-mer algorithm applied to compute the sequence information of proteins [27] from STRING
database [28]

Since **C** _[dr]_ and **C** _[di]_ are high-dimensional, we additionally
make use of autoencoder [29] to reduce their dimensions to
_N × d_ 1 and _M × d_ 1 respectively. Regarding **C** _[pr]_, the 3-mer
algorithm allows us to predefine its dimension, which is set as
_P × d_ 1 in our work. In this regard, the operation of dimension
reduction is not applied to **C** _[pr]_ . Once obtaining **C** _[dr]_, **C** _[di]_ and
**C** _[pr]_, we can construct **C** according to its definition.


_C. Learning Lower-Order Representations_


In social networks, each user contains some attributes, such
as age, gender, etc., which are all necessary to describe the
identity of user. Similarly, biological networks also preserve
such a property. However, the natural biological attributes are
more complex and cannot be directly trained with machine
learning algorithms. Although there have been some previous



Here, **A** [˜] = **A** + **I** where **A** is the adjacency matrix of HIN
and **I** is a identify matrix, **D** [˜] is the degree matrix of **A** [˜], **H** _[′]_ _∈_
R _[|][V][ |×][d]_ [2] and **H** _[′′]_ _∈_ R _[|][V][ |×][d]_ [2], respectively, are the first-order and
second-order neighbor aggregation representations of nodes.
Hence, **B** = **H** _[′′]_ ( _{V_ _[dr]_ _, V_ _[di]_ _}_ ) _∈_ R [(] _[N]_ [+] _[M]_ [)] _[×][d]_ [2] is used to denote
the lower-order representation vectors of drugs and diseases
derived from the hidden layer **H** _[′′]_ .


_D. Learning Higher-Order Representations_


The heterogeneous information in a HIN is introduced by the
different types of nodes, i.e., drugs, diseases and proteins, and
also by their different types of associations, i.e., DDAs, drugprotein associations and protein-disease associations. However,
due to the sparsity of HIN, the neighboring information of nodes



computational methods proposed for processing the attributes
of biomolecules, they share some common disadvantages. For
example, drugs with similar molecular structures are possible
to be described with much different SMILES strings [30], and
this may yield inaccurate prediction results by confusing the
classifier. Furthermore, the missing of original biological data
is commonly encountered during the collection process, and
the corresponding elements in the attribute matrix are normally
substituted with 0. Such a simple substitution can also negatively
affect the accuracy of identifying novel DDAs. Therefore, a
graph convolutional network (GCN) model [31] is incorporated
into FuHLDR to learn the lower-order representations of nodes
by aggregating their neighboring information in a given HIN.

In particular, GCN learns the neighbor information of nodes
through the propagation of a multi-layer neural network with the
following update rules.



**H** _[′]_ = _σ_ ( **D** [˜] _[−]_ 2 [1]


**H** _[′′]_ = _σ_ ( **D** [˜] _[−]_ 2 [1]




[1] ˜

2 **A** ˜ **D** _[−]_ 2 [1]



2 **H** _[′]_ **W** _[′′]_ ) (2)



2 **CW** _[′]_ ) (1)




[1] ˜

2 **A** ˜ **D** _[−]_ 2 [1]



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:26:36 UTC from IEEE Xplore. Restrictions apply.


166 IEEE TRANSACTIONS ON EMERGING TOPICS IN COMPUTING, VOL. 12, NO. 1, JANUARY-MARCH 2024



is insufficient to precisely learn the node representations. In this
regard, it is necessary to capture the higher-order connectivity
patterns of nodes, which provide us an alternative view to unlock
the full potential of HIN.

To this end, a meta-path learning model, i.e., metapath2vec [21], is employed to find the optimal higher-order
representation of each node, and it is mainly composed of
two steps. First, a meta-path-based random walk strategy is
adopted to capture diverse semantic relations among drugs,
proteins and diseases. After that, the higher-order representation of each node is obtained with a heterogeneous SkipGram model. In this work, a meta-path scheme _MP_ : _drug →_
_protein →_ _drug →_ _disease_ isdesignedtosimulatetheprocess
of how similar drugs act on the same disease. Therefore, given
a set of network paths **P** = _{p_ _i_ _}_, each path _p_ _i_ is composed
of _l_ nodes denoted as _v_ 1 _→_ _v_ 2 _→· · · →_ _v_ _i_ _→_ _v_ ( _i_ +1) _→· · · →_
_v_ _l_ ( _v_ 1 _, . . ., v_ _l_ _∈_ **V** ). Following the scheme _MP_, the transition
probability from _v_ _i_ to _v_ ( _i_ +1) is defined as:


_prob_ ( _v_ _i_ +1 _|v_ _i_ _, MP_ )



be randomly generated and remain unchanged in the training

process.

For an arbitrary node _v ∈{_ **V** _[dr]_ _,_ **V** _[di]_ _}_, its integrated representation is defined as **X** _v_ = [ **G** _v_ _,_ **B** _v_ ] by fusing higher and
lower-order representations, and _h_ _k_ = [ **X** ( _v_ ) _i_ _,_ **X** ( _v_ ) _j_ ] _∈_ _E_ _[dd]_

denotes the feature of the relationship between a drug node ( _v_ _i_ )
and a disease node ( _v_ _j_ ). Hence, FuHLDR constructs a RVFL
classifier with the loss function defined as below.



2 2
+ _λ||β||_ _k_ = 1 _,_ 2 _, . . ., K_ (6)



_Loss_ =



�


_k_



( _t_ _k_ _−_ _d_ _[T]_ _k_ _[β]_ [)]



1
_|Nb_ ( _v_ _i_ _,φ_ ( _t_ _i_ )) _|_ ( _v_ _i_ +1 _, v_ _i_ ) _∈_ _E, t_ _i_ +1 = _φ_ ( _t_ _i_ )
0 ( _v_ _i_ +1 _, v_ _i_ ) _∈_ _E, t_ _i_ +1 _̸_ = _φ_ ( _t_ _i_ )
0 ( _v_ _i_ +1 _, v_ _i_ ) _/∈_ _E_



In (6), _K_ is the number of positive and negative samples,
_t_ _k_ is denoted as the target result of _h_ _k_, _λ_ is the regularization
parameter, _β_ indicates output weights, and _d_ _k_ is the feature
vector by splicing the original and random features of _h_ _k_ . The
values of _t_ _k_ and _d_ _k_ can be obtained with (7) and (8) respectively.


_t_ _k_ = _d_ _[T]_ _k_ _[β]_ (7)


_d_ _k_ = Γ( _g_ ( _w_ _k_ _[T]_ _[h]_ _[k]_ [+] _[ b]_ _[k]_ [)] _[, h]_ _[k]_ [)] (8)


In (8), _w_ _k_ is a random weight with a uniform distribution
within [ _−s,_ + _s_ ], _g_ ( _·_ ) is an activation function, Γ( _·_ ) is concatenation function, and _b_ _k_ is a bias. Finally, the results of FuHLDR
prediction are represented as the matrix **R** _∈_ R _[E]_ _[dd]_ . The detailed
procedure of FuHLDR is presented in _Algorithm 1_ . One should
note that the values of these parameters, i.e., _d_ 1, _d_ 2, _d_ 3, _l_ and _w_
are defined as 64, 64, 64, 10 and 10, respectively.


III. E XPERIMENT


_A. Dataset_


To evaluate the performance of FuHLDR, two actual datasets
are adopted to construct HINs, and they are B-Dataset and FDataset. There are three kinds of biological networks, i.e., drugdisease, drug-protein, and protein-disease networks, to compose
these two datasets.


Regarding B-Dataset, its drug-disease network is collected
from CTD database [35] by following Zhang et al.’s instruction [8], and a total of 18,416 DDAs, 269 drugs and 598 diseases are contained in this network. Moreover, the drug-protein
network is obtained from the DrugBank database [24], and it
contains 213 drugs, 357 proteins and 3,110 drug-protein associations. The protein-disease network is downloaded from the DisGeNET database [36], and it consists of 805 proteins, 142 diseases and 5,898 protein-disease associations. When compared
with B-Dataset, F-Dataset is sparser in terms of the number of
DDAs. It is composed of 593 drugs, 313 diseases, 1,933 DDAs,
3,243 drug-protein associations and 54,265 protein-disease associations. In the F-Dataset, drug-disease, drug-protein, and
protein-disease networks are obtained from Gottlieb et al. [11],
DrugBank and DisGeNET respectively.

Moreover, an in-depth investigation has also been conducted
to examine the difference between B-Dataset and F-Dataset by
analyzing all associations in them, and finding several things
as follows. First, F-Dataset is much sparser than B-Dataset, as
the densities of B-Dataset and F-Dataset are 0.1144 and 0.0104

respectively. Second, the number of associations in F-Dataset



=



⎧
⎨

⎩



(3)



where _φ_ ( _t_ _i_ ) returnsthetypeofnextnodegiventhetypeofcurrent
node is _t_ _i_ by traversing the route of MP, and _Nb_ ( _v_ _i_ _, φ_ ( _t_ _i_ )) is the
set of _v_ _i_ ’s neighboring nodes with the node type _φ_ ( _t_ _i_ ). With
(3), each path in **P** is able to be determined by integrating the
semantic relations among different biomolecules into the heterogeneous Skip-Gram model. The higher-order representations
of nodes can thus be learned by maximizing the conditional
probability defined as:



�

_v_ _j_ _∈Nb_ ( _v_ _i_ _,φ_ ( _t_ _i_ ))



_argmax_

**F**



�

_v_ _i_ _∈_ **V**



_logprob_ ( _v_ _j_ _|v_ _i_ _,_ **P** ) (4)



where _F_ = _F_ _i_ is the set of higher-order representations of all
nodes, _prob_ ( _v_ _j_ _|v_ _i_ _,_ **P** ) represents the conditional probability of
visiting _v_ _j_ given the current node _v_ _i_ in the context of **P**, and its
definition is given as below.


_e_ _[F]_ _[j]_ _[F]_ _[i]_
_prob_ ( _v_ _j_ _|v_ _i_ _,_ **P** ) = (5)
~~�~~ _e_ _[F]_ _[k]_ _[F]_ _[i]_


_v_ _k∈_ **V**


Hence, according to **F**, a ( _N_ + _M_ ) _× d_ 3 matrix **G** is constructed to denote the higher-order representations of drugs and
disease.


_E. Predicting DDAs_


After obtaining the higher and lower-order feature representations of drugs and diseases from HIN, FuHLDR next targets to
predict the associations between drugs and diseases with their
integrated representations. According to [32], [33], we select
RVFL [34] as the classifier of FuHLDR by fusing multi-order
biological information to discover potential DDAs. In particular, RVFL is a feature discriminator employed to distinguish
different types of associations, and it is different from the general neural network due to the fact that its weight matrix can



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:26:36 UTC from IEEE Xplore. Restrictions apply.


ZHAO et al.: FUSING HIGHER AND LOWER-ORDER BIOLOGICAL INFORMATION FOR DRUG REPOSITIONING 167



**Algorithm 1:** The Complete Procedure of FuHLDR.

**Input** : graph _HIN_ ( **V** _,_ **E** _,_ **C** )
representation sizes: _d_ 1 _, d_ 2 _, d_ 3
random walk length _l_
context size: _w_

the number of hidden neurons: _δ_

the regularization parameter: _λ_
**Output** : the prediction matrix **R**
1: Initialization: **R**

2: Constructing **C**
3: Calculating the Molecular fingerprint attribute
matrix of drugs **C** _[dr]_

4: Calculating the semantic similarity matrix of
diseases **C** _[di]_


5: Calculating the sequence attribute matrix of proteins
**C** _[pr]_


6: Reducing dimensions by the autoencoder



7: **C** =



⎡ **CC** _[dr][di]_


**C** _[pr]_

⎣



⎤⎦ _∈_ R _[|]_ **[V]** _[|×][d]_ 1



8: Learning lower-order representations
9: **B** = _GCN_ ( **C** _,_ **A** _, d_ 2 )
10: Learning higher-order representations
11: **G** = _matepath_ 2 _vec_ ( **E** _, d_ 3 _, l, w_ )
12: Predicting DDAs
13: **for** each _e_ _ij_ = _< v_ _i_ _, v_ _j_ _>∈_ _E_ _[dd]_ **do**
14: the features matrix of drugs and diseases
**X** _v_ = [ **G** _v_ _,_ **B** _v_ ]
15: the input features of the classifier
_h_ _k_ = [ **X** _v_ ( _v_ _i_ ) _,_ **X** _v_ ( _v_ _j_ )]
16: **R** = _RV FL_ ( _h_ _k_ _, δ, λ_ )
17: **end for**

18: **return R**


TABLE I

T HE D ETAILS OF T WO B ENCHMARK D ATASETS


is twice as many as in B-Dataset, and accordingly the richer
structural information in F-Dataset allows FuHLDR to better

capture more expressive representations of drugs and diseases
from the higher-order perspective. The details of two benchmark
datasets are presented in Table I.

Another point worth noting is that both B-Dataset and FDataset are imbalanced due to the fact that the number of negative samples is greater than that of positive samples. Therefore,
we randomly choose unknown drug-disease pairs with an equal
size of positive samples as negative samples to evaluate the
performance of FuHLDR.



_B. Evaluation Metrics_


To evaluate the predictive performance of FuHLDR from different perspectives, several evaluation metrics, including AUC,
AUPR, Matthew’s correlation coefficient (MCC), Precision
(Prec.), Recall (Rec.), and F1-score (F1), are used. Among them,
AUC and AUPR are the areas under the Receiver Operating
Characteristic (ROC) curve and the Precision-Recall (PR) curve
respectively. The definition of MCC, Prec., Rec. and F1 are
defined as:


_MCC_


_TP × TN −_ _FP × FN_

=
~~�~~ ( _TP_ + _FP_ ) _×_ ( _TP_ + _FN_ ) _×_ ( _TN_ + _FP_ ) _×_ ( _TN_ + _FN_ )

(9)


_TP_
_Prec._ = (10)
_TP_ + _FP_


_TP_
_Rec._ = (11)
_TP_ + _FN_


2 _TP_
_F_ 1 = (12)
2 _TP_ + _FP_ + _FN_


whereTP,TN,FPandFNrepresentthenumbersoftruepositives,
true negatives, false positives and false negatives respectively for
predicted DDAs.

In the experiments, the performance of FuHLDR is evaluated
by following a 10-fold cross-validation (CV) scheme. Specifically, a benchmark dataset is divided into 10-folds. Each fold is
alternatively taken as a testing dataset while the rest compose the
training dataset. The results of the 10-fold CV on B-Dataset and
F-Dataset are shown in Supplementary Table S1-2 and Figure
S1-2 _(available online)_ .


_C. Parameter Selection_


To achieve the best performance of FuHLDR, the parameters
involved in training the RVFL classifier with (6) have to be
tuned during the training. As has been pointed out by [37], more
neurons are used to construct the hidden layer if the features are
more complex or the set of training samples is huge. Therefore,
the number of hidden neurons, i.e., _δ_ is set to 2 _[a]_ where the
value of a is varied from the set _{_ 10 _,_ 11 _,_ 12 _,_ 13 _,_ 14 _,_ 15 _,_ 16 _}_ .
Besides, the regularization parameter _λ_ is set as 1 _/_ 2 _[s]_, where the
value of _s_ is varied from the set _{−_ 6 _, −_ 4 _, −_ 2 _,_ 3 _,_ 6 _,_ 12 _}_ . Given
different combinations of _δ_ and _s_, the performance of FuHLDR
is presented in Fig. 2, where we note that FuHLDR achieves
its best predictive performance on the B-Dataset when _a_ = 16
and _s_ = 3, and on the F-Dataset when _a_ = 14 and _s_ = _−_ 2.
Furthermore, we have also constructed experiments to analyze
the parameters involved in representation learning. First, the
lower and higher-order representation dimensions are set to _d_ 2
and _d_ 3 respectively, and their values are varied from the set
_{_ 32 _,_ 64 _,_ 128 _}_ . Given different combinations of _d_ 2 and _d_ 3, the

–
performance of FuHLDR is presented in Fig. 2(c) (d), where
we note that FuHLDR achieves its best predictive performance
on benchmark datasets when _d_ 2 = 64 and _d_ 3 = 128. Second,
regarding the random walk length _l_ and the context size _w_,



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:26:36 UTC from IEEE Xplore. Restrictions apply.


168 IEEE TRANSACTIONS ON EMERGING TOPICS IN COMPUTING, VOL. 12, NO. 1, JANUARY-MARCH 2024


Fig. 2. Hyperparameters selected for FuHLDR model. (a)-(b) present the process of parameter tuning for RVFL on the B-Dataset and F-Dataset.
(c)-(d) present the process of parameter tuning for lower and higher-order representation dimension on the B-Dataset and F-Dataset. (e)-(f) present
the process of parameter tuning for higher representation learning on B-Dataset and F-Dataset.



we vary the values of _l_ and _w_ from the sets _{_ 5 _,_ 10 _,_ 20 _}_ and
_{_ 2 _,_ 5 _,_ 7 _,_ 10 _}_ respectively. Given different combinations of _l_ and
_w_, the performance of FuHLDR is presented in Fig. 2(e)–(f),
and we note that the best predictive performance of FuHLDR is
achieved on the benchmark datasets when _l_ = 10 and _w_ = 7.

Regarding the hardware environment, all experiments have
been conducted on an AMD Ryzen machine equipped with an



8-core CPU at 3.9 GHz, 128 GB of RAM and NVIDIA GeForce

GTX-2080Ti*2.

To evaluate the operational efficiency of FuHLDR, we have
reported the running time taken FuHLDR in 10-fold crossvalidation. In particular, FuHLDR costs 966.73 seconds and
19.11 seconds to complete the cross-validation experiment on
B-Dataset and F-Dataset respectively. Since B-Dataset has many



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:26:36 UTC from IEEE Xplore. Restrictions apply.


ZHAO et al.: FUSING HIGHER AND LOWER-ORDER BIOLOGICAL INFORMATION FOR DRUG REPOSITIONING 169


TABLE II
T HE AUC AND AUPR V ALUES OF THE V ARIOUS M ETHODS BY 10- FOLD CV ON T WO D ATASETS



more DDAs than F-Dataset, FuHLDR takes more time to compute the loss function of RVFL for B-Dataset with a larger value
of _K_ .


_D. Comparison With Several State-of-the-Art Drug_
_Repositioning Models_


Experiments have been conducted to evaluate the performance of FuHLDR by comparing it with five state-of-the-art
drug repositioning models, i.e., NRLMF [7], NIMCGCN [15],
DRRS [6], SCMFDD [8], LAGCN [16], DeepR2cov [17] and
DRHGCN [18].

The AUC and AUPR scores yielded by all competing models
on B-Dataset and F-Dataset are presented Table II. We note
that FuHLDR has a superior performance in terms of AUC
across all datasets, as it performs better by 6.57%, 9.27%,
3.72%, 10.87% and 4.52% than NRLMF, NIMCGCN, DRRS,
SCMFDD and DRHGCN, respectively. The performance of
FuHLDR in terms of AUPR shows a bigger margin, as the
average AUPR value of FuHLDR is larger by 43.96%, 48.81%,
44.41%, 60.71% and 15.16% than NRLMF, NIMCGCN, DRRS,
SCMFDD and DRHGCN, respectively. In this regard, we reason
that the consideration of higher and lower-order biological information considerably improves the learning ability of FuHLDR
for improved accuracy in DDA prediction. Besides, due to the
imbalance between positive and negative samples, the AUPR
values of some models are lower than expected.

Although FuHLDR yields the best performance in terms of
AUC and AUPR, an in-depth analysis is conducted on the results
in Table II from another perspective. The reasons accountable for
the poor performance of prediction models used for comparison
are two-fold. First, most of them ignore higher-order connectivity patterns presented in biological HINs. Taking NIMCGCN as
an example, it mainly considers the lower-order representations
between molecules as the features to train a classifier. Since
FuHLDR takes into account simultaneously higher and lowerorder biological information presented in HINs, its learning ability is improved to better obtain the representations of drugs and
diseasesinamorecomprehensivemanner.Second,allprediction
models except FuHLDR fail to consider the heterogeneous
information resulting from the introduction of proteins when
learning the lower-order representations. In doing so, the node
representations learned by these models have a limited ability in
aggregating neighboring information especially when the HINs
of interest are generally sparse. Hence, the consideration of



protein-related associations enriches the neighboring information, thus allowing FuHLDR to better learn the representations
of drugs and diseases at a lower-order level.

To conduct the experiments with LAGCN and DeepR2Cov,
we first download their source codes from the Github sites
provided in their original work, and compile these codes to run
these prediction models for performance comparison. Regarding
their parameter settings used for training, we explicitly adopt the
default parameter values recommended in their original work
for obtaining a fair comparison. Their experimental results of
10-fold CV on B-Dataset and F-Dataset are presented in Table II.
On average, FuHLDR performs better by 7.37% and 56.46%
than LAGCN in terms of AUC and AUPR respectively. When
comparedwithDeepR2Cov,itgives18.05%and16.21%relative
improvement in AUC and AUPR respectively.

Regarding the unsatisfactory performance of LAGCN and
DeepR2Cov in drug repositioning, the reasons are two-fold.
First, LAGCN fails to take into account rich higher-order structural information in a HIN, and hence the representations learned
by its layer attention graph convolutional network are less informative to infer novel DDAs. Second, although DeepR2Cov
preserves long-range structure dependency by designing different meta-paths, its performance is limited without considering
the lower-order biological information, which explicitly affects
the accuracy of DDA prediction according to our ablation study.

Moreover, we also note that the performance of FuHLDR on
F-Dataset is better than on B-Dataset. The reason for this phenomenon is that all associations of B-Dataset are less than that

of F-Dataset, which weakens the learning ability of FuHLDR on
B-Dataset. In summary, the promising performance of FuHLDR
is a strong indicator that FuHLDR is a useful tool to discover
novel DDAs.


_E. Classifier Selection_


To verify the rationality behind the use of RVFL, additional
experiments have been conducted by implementing several variants of FuHLDR integrated with different classifiers, including
KNN (K-Nearest Neighbors) classification, SVM (Support Vector Machine), LR (Logistic Regression), RF (Random Forest)
classification and RVFL [34]. As shown in Table III and Fig. 3,
FuHLDR yields the best performance in terms of AUC and
AUPR on B-Dataset when taking RVFL as its classifier. Among
all classifiers used for comparison, RVFL improves evaluation scores by 8.92%, 6.21%, 32.89%, 10.26%, 26.19%, and
11.99% when compared with RF respectively. More detailed



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:26:36 UTC from IEEE Xplore. Restrictions apply.


170 IEEE TRANSACTIONS ON EMERGING TOPICS IN COMPUTING, VOL. 12, NO. 1, JANUARY-MARCH 2024


TABLE III
T HE P ERFORMANCE OF D IFFERENT C LASSIFIERS BY 10-F OLD CV ON THE B-D ATASET


Fig. 3. The ROC and PR curves are obtained by different classifiers on the B-Dataset, and they are presented in subfigures (a)-(b), respectively.



experimental results of these different classifiers under 10-fold
CV are provided in the supplementary material, available online.

Another point worth noting in Table III is that machine
learning-based classifiers, i.e., KNN, SVM and LR, yield a fair
performance in terms of AUC, mainly because of their limited ability in handling high-dimensional and complex feature
information contained in HINs. Although the performance of
FuHLDR is slightly improved when integrated with RF, it is still
worse than RVFL, which are classifiers with neural networks.
Since neural network solves diverse and high-dimensional problems by constructing a nonlinear complex relationship model,
RVFL has a better discriminant ability in predicting the associations between drugs and diseases with their features _X_ _v_ . To be
specific, the reasons ascribed for the promising performance of
RVFL are two-fold. First, RVFL contains an enhancement space
layer, which could enhance feature representations of nodes by
projecting input features into an enhanced feature space. Second,
it could concatenate such space to the output layer along with
the original feature representations. As mentioned above, the
accuracy of FuHLDR prediction has been greatly improved for
candidate drugs or diseases through RVFL.


_F. Ablation Analysis_


To better investigate the influence of higher and lower-order
biological information on the performance of FuHLDR for DDA
prediction, we have developed two variants of FuHLDR, i.e.,
FuHLDR-L and FuHLDR-H. In particular, FuHLDR-L only
makes use of the representations of drugs and diseases learned



from lower-order biological information, while FuHLDR-H
rests on higher-order representations of drugs and diseases to
complete the task of DDA prediction. Their experimental results
of 10-fold CV on the B-Dataset are presented in Table IV and
Fig. 4, where several things can be noted.

First, FuHLDR-L achieves the worst performance among
FuHLDR and its variants. In this regard, only relying on
the lower-order biological information may not be sufficiently
enough to achieve desired DDA prediction performance. Hence,
for newly discovered diseases, the lack of sufficient biological
knowledge decreases the accuracy of discovering their potential
DDAs with FuHLDR-L. Second, FuHLDR-H shows a bigger margin in performance against FuHLDR-L. On average,
FuHLDR-H performs better by 13.13%, 11.73%, 26.47% and
13.45% than FuHLDR-L in terms of AUC, AUPR, MCC and
F1 respectively. In this regard, the consideration of higherorder connectivity patterns allows FuHLDR-H to better unlock the full potential of HIN on the task of DDA prediction. Last, a further improvement is made by FuHLDR that
combines the advantages of FuHLDR-L and FuHLDR-H by
fusing higher and lower-order representations of drugs and
diseases.


_G. Complexity Analysis_


When analyzing the computational complexity of FuHLDR,
we decompose it into four parts corresponding to the workflow of FuHLDR. First, there are three matrices, i.e., **C** _[dr]_,



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:26:36 UTC from IEEE Xplore. Restrictions apply.


ZHAO et al.: FUSING HIGHER AND LOWER-ORDER BIOLOGICAL INFORMATION FOR DRUG REPOSITIONING 171


TABLE IV
T HE P ERFORMANCE OF F U HLDR-L, F U HLDR-H AND F U HLDR BY 10-F OLD CV ON B-D ATASET


Fig. 4. The ROC and PR curves of FuHLDR-L, FuHLDR-H and FuHLDR on B-Dataset in the ablation study, and they are presented in subfigures
(a)-(b), respectively.



**C** _[di]_ and **C** _[pr]_, required to construct **C**, and their computational complexities are _O_ ( _N_ ), _O_ ( _M_ [2] ), and _O_ ( _PS_ ) respectively given that _S_ is the sequence length. Moreover, the dimensionality reduction applied to **C** _[dr]_ and **C** _[di]_ takes time
_O_ ( _d_ 1 _M_ [2] ). Second, the computational complexity of learning
lower-order representations is _O_ ( _|E|d_ 1 _d_ 2 ). Third, the computational complexity of learning higher-order representations is
more complicated than that of learning lower-order representations, and it is composed of two steps as indicated by (3)
and (4). These two steps take time _O_ ( _lw|V |_ ) and _O_ ( _lwd_ 3 _|V |_ )
respectively, and we have _O_ ( _lw|V |_ ) _< O_ ( _lwd_ 3 _|V |_ ). Last, regarding the DDA prediction with the RVFL classifier, this
process has a complexity of _O_ ((2( _d_ 2 + _d_ 3 ) _K_ ) [3] ) where _|E|_,
_d_ 1, _d_ 2, _d_ 3, _K_, _l_ and _w_ are constant values. Hence, the overall
computational complexity of FuHLDR is _O_ ( _PS_ + _|E|d_ 1 _d_ 2 +
_lwd_ 3 _|V |_ + (2( _d_ 2 + _d_ 3 ) _K_ ) [3] ). Considering the inequality _w <_
_l < d_ 1 _≤_ _d_ 2 _≤_ _d_ 3 _< M < S < P < |V | < K_, the computational complexity of FuHLDR can be further simplified to
_O_ ( _K_ [3] ).


_H. Case Study_


To further prove the applicability of FuHLDR, two diseases,
including Alzheimer’s disease (AD) and Breast neoplasms (BN)
of B-Dataset, are selected as case studies to discover their
potential therapeutic drugs. AD is a degenerative disease of
the nervous system that mainly occurs in the elderly, and the
number of patients is increasing year by year [38]. Table V shows
the top 10 drug candidates identified by FuHLDR for AD, and
among them 8 candidates have been confirmed in the literature



TABLE V
T HE T OP 10 C ANDIDATE D RUGS P REDICTED BY F U HLDR FOR AD


TABLE VI
T HE T OP 10 C ANDIDATE D RUGS P REDICTED BY F U HLDR FOR BN


(PMID is the PubMed identifier). BN is a common disease
in women with a high incidence rate, which is challenging
in clinical treatment [39]. Table VI presents the top 10 drug
candidates predicted by FuHLDR for the potential treatment of
BN, and among them 9 candidates have been confirmed in the



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:26:36 UTC from IEEE Xplore. Restrictions apply.


172 IEEE TRANSACTIONS ON EMERGING TOPICS IN COMPUTING, VOL. 12, NO. 1, JANUARY-MARCH 2024


Fig. 5. A diagram of the partial relationships of between Caffeine and Alzheimer’s disease in the B-Dataset. The green nodes represent drugs,
the red nodes represent diseases, the purple nodes represent proteins, and solid lines represent known links.



literature. It is worth noting that the effects of Cimetidine [40]
and Ketoconazole [41] on BN have just recently been proposed.
Take as Ketoconazole an example, its ability to act on the zinc
finger protein to inhibit BN again justifies FuHLDR’s ability to
tap into new DDAs by introducing proteins, and further provides
a possible direction for drug research and disease treatment. This
finding fully demonstrates the reliability of FuHLDR in drug
repurposing, and further provides a possible direction for drug
research and disease treatment.

Moreover, an in-depth analysis is also performed on Caffeine,
which is a verified drug candidate of AD predicted by FuHLDR,
from the perspective of higher-order connectivity patterns. By
visualizing a partial HIN involving AD and Caffeine from
B-Dataset, we intuitively find from Fig. 5 that there are many
network paths between them that satisfy the designed meta-path
strategy (i.e., _drug →_ _protein →_ _drug →_ _disease_ ), and thus
the path **P** is proved to be effective. More importantly, proteins
play a critical role as the intermediaries between AD and
Caffeine, as the size of proteins in Fig. 5 is larger, which
indicates that they contribute more when FuHLDR predicts the
association between AD and Caffeine. Obviously, FuHLDR is
verified to be effective in aggregating the higher-order structure
information of HIN based on the meta-paths connecting drugs,
proteins and diseases.
Similarly, wealsopresent anin-depthanalysis of thepredicted
drugs predicted to be associated with BN. In doing so, it is
anticipated to demonstrate the predictive ability of FuHLDR



from the perspective at a lower-order level. Pearson productmoment correlation coefficients of the lower-order representation in _B_ ( _V_ _[dr]_ ) are adopted to indicate the similarities between
predicted drugs and approved drugs as shown in Fig. 6. One
should note that each of the predicted drugs is highly similar to
some of the approved drugs according to the distribution of dark
blocks and chrominance index. Hence, a conclusion could be
made that FuHLDR has a strong performance to discover these
candidate drugs at a lower-order level by considering DDAs
and the biological attributes of drugs and diseases through the
GCN.

Even for those DDAs predicted by FuHLDR but not verified,
evidence can also be found through a literature review to indicate
their potential ability for treatment. Taking AD as an example,
although its association with Olanzapine has not been officially declared in pharmacological studies at the present stage,
there are many publications that provide evidence supporting
our prediction result to some extent. First, Onaka et al. [42]
argue that cognitive dysfunction in AD could be ameliorated
with Olanzapine. Second, the study on the interaction between
Schizophrenia (SZ) and AD indicates that these two diseases
are similar, as they have a shared which is clear overlap of
white matter deficit patterns [43]. Since the association between
Olanzapine and SZ is verified in the B-Dataset, we have reason
to believe that Olanzapine has a certain effect on the treatment of
AD. In other words, the ability of FuHLDR in predicting novel
DDAs can thus be verified.



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:26:36 UTC from IEEE Xplore. Restrictions apply.


ZHAO et al.: FUSING HIGHER AND LOWER-ORDER BIOLOGICAL INFORMATION FOR DRUG REPOSITIONING 173


Fig. 6. The similarity of biological representation between predicted drugs and approved drugs for Breast neoplasms. The horizontal axis
represents the approved drugs and the vertical axis represents the predicted drugs.



TABLE VII
T HE T OP 10 C ANDIDATE D RUGS P REDICTED BY F U HLDR-L FOR AD


TABLE VIII
T HE T OP 10 C ANDIDATE D RUGS P REDICTED BY F U HLDR-L FOR BN


Since we have developed two variants of FuHLDR, i.e.,
FuHLDR-L and FuHLDR-H, to demonstrate the respective impacts of lower and higher-order representations on the task of
drug repositioning, they are also explicitly applied to predict
the potential drugs for the treatment of AD and BN, and experimental results are presented in Tables VII, VIII, IX, and
X, where top-10 drug candidates predicted by FuHLDR-L and
FuHLDR-H are listed. It is observed that both FuHLDR-L and

FuHLDR-H perform worse than FuHLDR in this case study, as
only around half of the top-10 drug candidates have been verified
by the literature. In light of the case study for AD and BN, the
performances of FuHLDR-L and FuHLDR-H demonstrate their



TABLE IX
T HE T OP 10 C ANDIDATE D RUGS P REDICTED BY F U HLDR-H FOR AD


TABLE X
T HE T OP 10 C ANDIDATE D RUGS P REDICTED BY F U HLDR-H FOR BN


own ability in predicting novel DDAs, and a further improvement is made by FuHLDR by combining their advantages.
Moreover, we have also analyzed the reasons why lower and
higher-order representations could benefit drug repositioning.
Taking caffeine as an example, its association with AD is
identified by FuHLDR with the incorporation of higher-order
representations learned through the designed meta-path. Another example is presented to indicate how the lower-order
representations learned from the biological knowledge lead to
the discovery of the associations between BN and BN-related
drugs.



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:26:36 UTC from IEEE Xplore. Restrictions apply.


174 IEEE TRANSACTIONS ON EMERGING TOPICS IN COMPUTING, VOL. 12, NO. 1, JANUARY-MARCH 2024



The above analysis confirms that by fusing higher and lowerorder biological information in HIN, FuHLDR is a useful tool
for drug repositioning due to its promising performance.


IV. D ISCUSSION


In summary, the above experimental results have demonstrated the promising performance of FuHLDR in predicting
novel DDAs. On the one hand, for a variety of biological HINs,
FuHLDR can consider the higher and lower-order biological
information for better learning the representations of drugs and
diseases [44]. To be more specific, a meta-path-based graph
representation learning model is adopted by FuHLDR to learn
the higher-order representation of drugs and diseases. Unlike
previous homogenous representation learning, meta-path-based
representation learning can ensure to preserve different kinds
of biological associations, as well as their network path information, in the representations of drugs and diseases. Moreover, although the high-throughput experiments considerably
increase the biological information of molecules, the corresponding databases are far from complete especially for new
discovered molecules. It is for this reason that FuHLDR utilizes

a graph convolution network model to capture the lower-order
representation of each drug, or disease, by aggregating their
neighboring information from both biological attributes and
network structure, thus alleviating the negative impact resulting
from the missing biological information. In this regard, the
consideration of higher and lower-order biological information
allows FuHLDR to fully unlock the potential of HIN for improved accuracy of drug repositioning.
On the other hand, after fusing the higher and lower-order
representations of drugs and diseases, FuHLDR takes advantage
of RVFL to precisely predict unknown DDAs. Compared with
conventional classifiers, RVFL is composed of a large number
of hidden neurons and enhanced nodes, which further enhance
the ability of FuHLDR to handle high-dimensional features with
excellent performance for drug repositioning.
As a popular validation strategy, independent verification
aims to evaluate the generalization ability of the proposed
method for unseen data. The independence between training and
testing data requires that the testing data should not only have
no overlap with the training data, but should also follow the
same distribution as the training data. However, since FuHLDR
rests on higher-order connectivity patterns to conduct the representation learning procedure, it is incapable of obtaining the
representations of query drugs and diseases that are not found to
constitute any meta-path in the training dataset. In this regard,
independent verification may not be a proper validation strategy
for FuHLDR.

To further verify this statement, we have also conducted independent verification experiments on F-Dataset and B-Dataset
by using them as training and testing datasets alternatively. The
results demonstrate that FuHLDR yields a poor performance, as
its AUC, AUPR and F1 scores are much less than those obtained
in 10-fold cross-validation. Considering the fact that few drugs
and diseases are shared by F-Dataset and B-Dataset, FuHLDR
could not be able to effectively learn the representations for most



of drugs and disease in the testing dataset, thus leading to its poor
performance in the independent verification.
Although the experiment results have demonstrated the
promising performance of FuHLDR, there are still some limitations to be addressed in future work. First, in addition to the
three kinds of associations considered in our work, there are
also other kinds of associations, such as drug-drug interactions.
In this regard, we would like to include more heterogeneous
information to construct a HIN, from which FuHLDR is able
to learn more expressive network representations of drugs and
diseases. However, along with the increase in the heterogeneous information, the redundancy in higher and lower-order
biological information could be more severe, thus raising a
new challenge to the information fusion ability of FuHLDR. To
response this demand, we intend to construct a meta-learningbased framework to integrate multiple biological information.
Last, regarding the learning of higher-order representation,
there is only one meta-path designed when we integrate metapath2vec into FuHLDR. As a part of our future work, we
are interested in exploring the possibility of taking into account more different meta-paths for improved performance of
FuHLDR.


V. C ONCLUSION


Drug repositioning provides a low-cost and effective way
to discover new indications of approved drugs. However, its
development is constrained by the lack of sufficient DDAs.
To overcome this problem, different kinds of biological information, including protein-related associations, drug SMILES
data, disease MeSH descriptions and protein sequences, are
considered in our work to compose a more complicated HIN.
After that, a novel drug repositioning model, i.e., FuHLDR, is
proposed to identify potential DDAs. To do so, different neural
network models are adopted by FuHLDR to learn higher and
lower-order representations of drugs and diseases. By fusing
these representations, FuHLDR makes use of RVFL to complete
the drug repositioning task. A serious of extensive experiments
have demonstrated the superior performance of FuHLDR when
comparing it with several state-of-the-art drug repositioning
models. Moreover, our case studies also indicate that FuHLDR is
a useful tool to filter out convincing drug candidates for diseases.
Hence, the consideration of higher and lower-order biological
informationprovidesnewinsightintodrugrepositioningbyfully
unlocking the potential of biological HINs.


R EFERENCES


[1] A. Badkas, S. De Landtsheer, and T. Sauter, “Topological network
measures for drug repositioning,” _Brief. Bioinf._, vol. 22, no. 4, 2021,
Art. no. bbaa357.

[2] R. T. Eastman et al., “Remdesivir: A review of its discovery and development leading to emergency use authorization for treatment of COVID-19,”
_ACS Central Sci._, vol. 6, no. 5, pp. 672–683, 2020.

[3] J. Li, S. Zheng, B. Chen, A. J. Butte, S. J. Swamidass, and Z. Lu, “A survey
of current trends in computational drug repositioning,” _Brief. Bioinf._,
vol. 17, no. 1, pp. 2–12, 2016.

[4] M. Yang, L. Huang, Y. Xu, C. Lu, and J. Wang, “Heterogeneous graph
inference with matrix completion for computational drug repositioning,”
_Bioinformatics_, vol. 36, no. 22/23, pp. 5456–5464, 2020.



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:26:36 UTC from IEEE Xplore. Restrictions apply.


ZHAO et al.: FUSING HIGHER AND LOWER-ORDER BIOLOGICAL INFORMATION FOR DRUG REPOSITIONING 175




[5] H. Luo et al., “Drug repositioning based on comprehensive similarity

measures and bi-random walk algorithm,” _Bioinformatics_, vol. 32, no. 17,
pp. 2664–2671, 2016.

[6] H. Luo, M. Li, S. Wang, Q. Liu, Y. Li, and J. Wang, “Computational

drug repositioning using low-rank matrix approximation and randomized
algorithms,” _Bioinformatics_, vol. 34, no. 11, pp. 1904–1912, 2018.

[7] Y. Liu, M. Wu, C. Miao, P. Zhao, and X.-L. Li, “Neighborhood regularized

logistic matrix factorization for drug-target interaction prediction,” _PLoS_
_Comput. Biol._, vol. 12, no. 2, 2016, Art. no. e1004760.

[8] W. Zhang, X. Yue, F. Huang, R. Liu, Y. Chen, and C. Ruan, “Predicting

drug-disease associations and their therapeutic function based on the drugdisease association bipartite network,” _Methods_, vol. 145, pp. 51–59, 2018.

[9] W. Dai et al., “Matrix factorization-based prediction of novel drug in
dications by integrating genomic space,” _Comput. Math. Methods Med._,
vol. 2015, 2015, Art. no. 275045.

[10] H.Luo,M.Li,M.Yang,F.-X.Wu,Y.Li,andJ.Wang,“Biomedicaldataand

computational models for drug repositioning: A comprehensive review,”
_Brief. Bioinf._, vol. 22, no. 2, pp. 1604–1619, 2021.

[11] A. Gottlieb, G. Y. Stein, E. Ruppin, and R. Sharan, “PREDICT: A method

for inferring novel drug indications with application to personalized
medicine,” _Mol. Syst. Biol._, vol. 7, no. 1, 2011, Art. no. 496.

[12] Y. Wang, S. Chen, N. Deng, and Y. Wang, “Drug repositioning by kernel
based integration of molecular structure, molecular activity, and phenotype
data,” _PLoS One_, vol. 8, no. 11, 2013, Art. no. e78518.

[13] B.-W. Zhao, L. Hu, Z.-H. You, L. Wang, and X.-R. Su, “HINGRL:

Predicting drug–disease associations with graph representation learning
on heterogeneous information networks,” _Brief. Bioinf._, vol. 23, no. 1,
2022, Art. no. bbab515.

[14] Z. Wang, M. Zhou, and C. Arnold, “Toward heterogeneous information

fusion: Bipartite graph convolutional networks for in silico drug repurposing,” _Bioinformatics_, vol. 36, no. Supplement_1, pp. i525–i533, 2020.

[15] J. Li, S. Zhang, T. Liu, C. Ning, Z. Zhang, and W. Zhou, “Neural inductive

matrix completion with graph convolutional networks for miRNA-disease
association prediction,” _Bioinformatics_, vol. 36, no. 8, pp. 2538–2546,
2020.

[16] Z. Yu, F. Huang, X. Zhao, W. Xiao, and W. Zhang, “Predicting drug–

disease associations through layer attention graph convolutional network,”
_Brief. Bioinf._, vol. 22, no. 4, 2021, Art. no. bbaa243.

[17] X. Wang et al., “DeepR2cov: Deep representation learning on heteroge
neous drug networks to discover anti-inflammatory agents for COVID-19,”
_Brief. Bioinf._, vol. 22, no. 6, 2021, Art. no. bbab226.

[18] L. Cai et al., “Drug repositioning based on the heterogeneous information

fusion graph convolutional network,” _Brief. Bioinform._, vol. 22, no. 6,
2021, Art. no. bbab319.

[19] X. Zeng, S. Zhu, X. Liu, Y. Zhou, R. Nussinov, and F. Cheng, “deepDR:

A network-based deep learning approach to in silico drug repositioning,”
_Bioinformatics_, vol. 35, no. 24, pp. 5191–5198, 2019.

[20] L. Hu, J. Zhang, X. Pan, H. Yan, and Z.-H. You, “HiSCF: Leveraging

higher-order structures for clustering analysis in biological networks,”
_Bioinformatics_, vol. 37, no. 4, pp. 542–550, 2021.

[21] Y. Dong, N. V. Chawla, and A. Swami, “metapath2vec: Scalable represen
tation learning for heterogeneous networks,” in _Proc. 23rd ACM SIGKDD_
_Int. Conf. Knowl. Discov. Data Mining_, 2017, pp. 135–144.

[22] G. Landrum, “RDKit documentation,” _Release_, vol. 1, no. 1-79, 2013,

Art. no. 4.

[23] D. Weininger, “SMILES, a chemical language and information system. 1.

Introduction to methodology and encoding rules,” _J. Chem. Inf. Comput._
_Sci._, vol. 28, no. 1, pp. 31–36, 1988.

[24] D. S. Wishart et al., “DrugBank 5.0: A major update to the DrugBank

database for 2018,” _Nucleic Acids Res._, vol. 46, no. D1, pp. D1074–D1082,
2018.

[25] Z.-H. Guo et al., “MeSHHeading2vec: A new method for representing

mesh headings as vectors based on graph embedding algorithm,” _Brief._
_Bioinf._, vol. 22, no. 2, pp. 2085–2095, 2021.

[26] M. Yang, G. Wu, Q. Zhao, Y. Li, and J. Wang, “Computational drug

repositioning based on multi-similarities bilinear matrix factorization,”
_Brief. Bioinf._, vol. 22, no. 4, 2021, Art. no. bbaa267.

[27] X.-N. Fan and S.-W. Zhang, “LPI-BLS: Predicting lncRNA–protein inter
actions with a broad learning system-based stacked ensemble classifier,”
_Neurocomputing_, vol. 370, pp. 88–93, 2019.

[28] D. Szklarczyk et al., “STRING v11: Protein–protein association networks

with increased coverage, supporting functional discovery in genome-wide
experimental datasets,” _Nucleic Acids Res._, vol. 47, no. D1, pp. D607–
D613, 2019.

[29] C.-Y. Liou, W.-C. Cheng, J.-W. Liou, and D.-R. Liou, “Autoencoder for

words,” _Neurocomputing_, vol. 139, pp. 84–96, 2014.




[30] Y. Cheng, Y. Gong, Y. Liu, B. Song, and Q. Zou, “Molecular design in drug

discovery: A comprehensive review of deep generative models,” _Brief._
_Bioinform._, vol. 22, no. 6, 2021, Art. no. bbab344.

[31] T. N. Kipf and M. Welling, “Semi-supervised classification with graph

convolutional networks,” 2016, _arXiv:1609.02907_ .

[32] R. Katuwal, P. N. Suganthan, and L. Zhang, “An ensemble of decision trees

with random vector functional link networks for multi-class classification,”
_Appl. Soft Comput._, vol. 70, pp. 1146–1153, 2018.

[33] R. Katuwal and P. N. Suganthan, “Stacked autoencoder based deep ran
dom vector functional link neural network for classification,” _Appl. Soft_
_Comput._, vol. 85, 2019, Art. no. 105854.

[34] L. Zhang and P. N. Suganthan, “A comprehensive evaluation of random

vector functional link networks,” _Inf. Sci._, vol. 367, pp. 1094–1105, 2016.

[35] A. P. Davis et al., “The comparative toxicogenomics database: Update

2017,” _Nucleic Acids Res._, vol. 45, no. D1, pp. D972–D978, 2017.

[36] J. Piñero et al., “DisGeNET: A comprehensive platform integrating infor
mation on human disease-associated genes and variants,” _Nucleic Acids_
_Res._, 2017, vol. 45, Art. no. gkw943.

[37] Q. Shi, R. Katuwal, P. Suganthan, and M. Tanveer, “Random vector

functional link neural network based ensemble deep learning,” _Pattern_
_Recognit._, vol. 117, 2021, Art. no. 107978.

[38] R. Mayeux and M. Sano, “Treatment of Alzheimer’s disease,” _New Eng-_

_land J. Med._, vol. 341, no. 22, pp. 1670–1679, 1999.

[39] S.-H. Ueng, T. Mezzetti, and F. A. Tavassoli, “Papillary neoplasms of the

breast: A review,” _Arch. Pathol. Lab. Med._, vol. 133, no. 6, pp. 893–907,
2009.

[40] F. Taghipour et al., “Modulatory effects of metformin alone and in com
bination with cimetidine and Ibuprofen on T cell-related parameters in a
breast cancer model,” _Iranian J. Allergy Asthma Immunol._, vol. 20, no. 5,
2021, Art. no. 600.

[41] D. L. Doheny et al., “Antifungal ketoconazole inhibits tumor-specific

transcription factor tGLI1 leading to suppression of breast cancer stem
cells and brain metastasis,” _Cancer Res._, vol. 80, no. 16_Supplement,
pp. 5025–5025, 2020.

[42] Y. Onaka, S. Wada, M. Yoneyama, T. Yamaguchi, and K. Ogita, “Ef
fect of olanzapine on trimethyltin-induced cognitive dysfunction and
neurodegeneration,” in _Proc. Annu. Meeting Japanese Pharmacological_
_Soc._ [, 2018, Art. no. PO2–1. [Online]. Available: https://doi.org/10.1254/](https://doi.org/10.1254/jpssuppl.WCP2018.0_PO2-1-31)
[jpssuppl.WCP2018.0_PO2-1-31](https://doi.org/10.1254/jpssuppl.WCP2018.0_PO2-1-31)

[43] P. Kochunov et al., “A white matter connection of schizophrenia and

Alzheimer’s disease,” _Schizophrenia Bull._, vol. 47, no. 1, pp. 197–206,
[2020. [Online]. Available: https://doi.org/10.1093/schbul/sbaa078](https://doi.org/10.1093/schbul/sbaa078)

[44] Y. Dai, C. Guo, W. Guo, and C. Eickhoff, “Drug–drug interaction pre
diction with Wasserstein adversarial autoencoder-based knowledge graph
embeddings,” _Brief. Bioinform._, vol. 22, no. 4, 2021, Art. no. bbaa256.


**BO-WEI ZHAO** is currently working toward the
PhD degree in computer application technology
with the University of Chinese Academy of Sciences (UCAS), Beijing, China, since 2021. He
is currently working with the Xinjiang Technical Institute of Physics and Chemistry, Chinese
Academy of Sciences, Urumqi, China. His current research interests include machine learning, complex networks analysis, graph neural
network, and their applications in bioinformatics.


**LEI WANG** received the PhD degree from the
School of Computer Science Technology, China
University of Mining and Technology, Jiangsu,
China, in 2018. He is currently working with
Big Data and Intelligent Computing Research
Center, Guangxi Academy of Science, Nanning,
China. His research interests include data mining, pattern recognition, machine learning, deep
learning, computational biology, and bioinformatics. He acted as reviewers for many international journals, such as the _Scientific Reports_,
_Current Protein & Peptide Science_, _Computational Biology and Chem-_
_istry_, _Soft Computing_, and _Journal of Computational Biology_ .



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:26:36 UTC from IEEE Xplore. Restrictions apply.


176 IEEE TRANSACTIONS ON EMERGING TOPICS IN COMPUTING, VOL. 12, NO. 1, JANUARY-MARCH 2024



**PENG-WEI HU** received the BE degree in software engineering from the Foreign Trade and
Business College, Chongqing Normal University, Chongqing, China, in 2011, and the PhD
degree from the Department of Computing,
Hong Kong Polytechnic University, Hong Kong,
in 2019. Currently, he is a research scientist with
IBM Research China. His research interests include machine learning, data mining, data fusion, and biomedical informatics.


**LEON WONG** received the PhD degree in computer application technology from the University
of Chinese Academy of Sciences (UCAS), Beijing, China. He is currently a post-doctoral with
Big Data and Intelligent Computing Research
Center, Guangxi Academy of Science, Nanning,
China. His current research interests include artificial intelligence, deep learning, complex networks analysis, evolutionary computation, and
their applications in bioinformatics.


**XIAO-RUI** **SU** received the BE degree in
software engineering from Xiamen University
(XMU), Xiamen, China, in 2019. She is currently working toward the PhD degree in computer application technology with the University
of Chinese Academy of Sciences (UCAS), Beijing, China, since 2019. She is currently working
with the Xinjiang Technical Institute of Physics
and Chemistry, Chinese Academy of Sciences,
Urumqi, China. Her current research interests
include artificial intelligence, data mining, complex networks analysis, knowledge graph representation learning, and
their applications in bioinformatics.



**BAO-QUAN WANG** received the MSc and PhD
degrees from the Xinjiang Institute of Physical
and Chemical Technology, Chinese Academy of
Sciences, in 2017 and 2020, respectively. His
main research interests include Big Data analysis and time series data mining.


**ZHU-HONG YOU** (Member, IEEE) received the
BE degree in electronic information science
and engineering from Hunan Normal University,
Changsha, China, in 2005, and the PhD degree in control science and engineering from the
University of Science and Technology of China
(USTC), Hefei, China, in 2010. From June 2008
to November 2009, he was a visiting research
fellow with the Center of Biotechnology and Information, Cornell University. He is currently a
professor with the School of Computer Science,
Northwestern Polytechnical University. He has published more than 170
research papers in refereed journals and conferences in the areas of
pattern recognition, bioinformatics, and complex-network analysis. He
holds more than 10 patents. His current research interests include neural networks, intelligent information processing, sparse representation,
and its applications in bioinformatics.


**LUN HU** (Member, IEEE) received the BEng
degree from the Department of Control Science
and Engineering, Huazhong University of Science and Technology, Wuhan, China, in 2006,
and the MSc and PhD degrees from the Department of Computing, Hong Kong Polytechnic University, Hong Kong, in 2008 and 2015,
respectively. He joined the Xinjiang Technical
Institute of Physics and Chemistry, Chinese
Academy of Sciences, Urumqi, China, in 2020
as a professor of computer science. His research interests include machine learning, complex network analytics
and their applications in bioinformatics.



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:26:36 UTC from IEEE Xplore. Restrictions apply.


