_Briefings in Bioinformatics_, 2022, **23** ( **3** ), 1–13


**https://doi.org/10.1093/bib/bbac140**

**Problem Solving Protocol**

# **Attention-based Knowledge Graph Representation** **Learning for Predicting Drug-drug Interactions**


Xiaorui Su, Lun Hu, Zhuhong You, Pengwei Hu and Bowei Zhao


Corresponding author. Lun Hu, Xinjiang Technical Institute of Physics & Chemistry, Chinese Academy of Science, Urumqi, China.
Email: hulun@ms.xjb.ac.cn


Abstract


Drug–drug interactions (DDIs) are known as the main cause of life-threatening adverse events, and their identification is a key
task in drug development. Existing computational algorithms mainly solve this problem by using advanced representation learning
techniques. Though effective, few of them are capable of performing their tasks on biomedical knowledge graphs (KGs) that provide
more detailed information about drug attributes and drug-related triple facts. In this work, an attention-based KG representation
learning framework, namely DDKG, is proposed to fully utilize the information of KGs for improved performance of DDI prediction. In
particular, DDKG first initializes the representations of drugs with their embeddings derived from drug attributes with an encoder–
decoder layer, and then learns the representations of drugs by recursively propagating and aggregating first-order neighboring
information along top-ranked network paths determined by neighboring node embeddings and triple facts. Last, DDKG estimates the
probability of being interacting for pairwise drugs with their representations in an end-to-end manner. To evaluate the effectiveness
of DDKG, extensive experiments have been conducted on two practical datasets with different sizes, and the results demonstrate that
DDKG is superior to state-of-the-art algorithms on the DDI prediction task in terms of different evaluation metrics across all datasets.


Keywords: drug–drug interactions, graph neural network, knowledge graph, attention-based representation learning



Introduction


Drug–drug interactions (DDIs) refer to changes in the
actions, or side effects, of drugs when they are taken
at the same time or successively [1, 2]. In general, DDIs
can be classified into three categories, including pharmaceutical interactions, pharmacokinetic interactions and
pharmacodynamic interactions [3]. In addition, as has
been reported by [4], the adverse drug reaction (ADR)
rates are 18.6%, 81.4% and 100% when 2–5 kinds of
drugs, 6–10 kinds of drugs and 10 kinds of drugs are
taken together, respectively. Obviously, DDIs have their
own impacts on patients in clinical treatment, and they
are known as the main cause of unexpected ADR [5].
As a result, there is an urgent need to detect DDIs to
alleviate life-threatening concerns, thereby facilitating
drug development and clinical treatment.

Recently, computational algorithms have been rapidly
developed to predict potential DDIs due to their promising performance in terms of efficiency [6], and they are
either similarity-based or network-based according to



their learning objectives. More specifically, following the
observation that similar drugs are more likely to interact
with each other [7, 8], similarity-based algorithms solve
the DDI prediction problem by calculating the similarity between drugs using their biological, or chemical
attribute information, including but not limited to fingerprints [9, 10], molecular structures [11], drug targets,
side effects and their combinations [12, 13].

Network-based computational algorithms consider
the DDI prediction problem from a global perspective,
and they normally make use of DDIs to compose
a DDI network [14, 15]. By doing so, the original
DDI prediction problem can be formulated as a link
prediction problem in the network context. To solve
it, most of network-based algorithms apply advanced
network representation learning models, which are
roughly categorized into three groups [16]: (i) Matrix
Factorization (MF)-based models [17–19], (ii) Random
Walk (RW)-based models [20] and (iii) Neural Network
(NN)-based models [21]. With these models, several



**Xiaorui Su** is a doctoral student of Xinjiang Technical Institute of Physics and Chemistry, Chinese Academy of Science, Urumqi, China. Her research interests
include machine learning, network representation learning, computational biology and bioinformatics.
**Lun Hu,** PhD, is a Professor of Xinjiang Technical Institute of Physics & Chemistry, Chinese Academy of Science, Urumqi, China. His research interests include
machine learning, big data analysis and its applications in bioinformatics.
**Zhuhong You,** PhD, is a Professor of School of Computer Science, Northwestern Polytechnical University, Xi’an, China. His research interests include neural
networks, intelligent information processing, sparse representation and its applications in bioinformatics.
**Pengwei Hu,** PhD, is a professor in Xinjiang Technical Institute of Physics & Chemistry, Chinese Academy of Science, Urumqi, China. His research interests include
machine learning, big data analysis and its applications in bioinformatics.
**Bowei Zhao** is a doctoral student of Xinjiang Technical Institute of Physics and Chemistry, Chinese Academy of Science, Urumqi, China. His research interests
include machine learning, complex networks analysis, graph neural network and their applications in bioinformatics.
**Received:** December 22, 2021. **Revised:** March 2, 2022. **Accepted:** March 27, 2022
© The Author(s) 2022. Published by Oxford University Press. All rights reserved. For Permissions, please email: journals.permissions@oup.com


2 | _Su_ et al.


attempts have been made to not only successfully
address the DDI prediction problem, but also accurately
identify other drug–related interactions, such as drug–
target interactions (DTIs) [22, 23]. In addition, some NNbased computational algorithms solve DDIs problem
based on biomedical literature. For example, the method
proposed by [24] constructed a hierarchical NN-based
models by integrating the shortest dependency path with
the sentence sequence for DDI extraction task.

Compared with similarity-based computational algorithms, network-based ones are superior in handling
graph-structured data and learning global representations from DDI networks. Recently, due to the ability
of providing more detailed information about node
attributes and link types, knowledge graphs (KGs) have
been received more attention [25]. However, only a
few network-based algorithms have been specifically
proposed to predict DDIs based on KG. In particular,
Karim et al. [26] obtain the embedding of drugs using a
traditional KG embedding method, i.e. ComplEx [27], and
then predict the potential DDIs with a convolutionalLSTM network. The main disadvantage of ComplEx
is that it learns only the first-order structures for
each drug while ignoring higher-order connectivity
patterns. As a new attempt in this direction, KGNN

[28] proposes a KG neural network to predict potential
DDIs by only considering triple facts in a given KG.
Though promising, they are concentrated on only making
use of drug-related triple facts without considering
drug attributes. As has been pointed out by [29–31],
node attributes are of great significance to present an
accurate analysis to complex networks. Hence, there is a
necessity for us to additionally integrate drug attributes
into the representation learning process for improved
performance of DDI prediction.

In this work, an attention-based KG representation
learning framework, namely DDKG, is proposed by simultaneously taking into account both drug attributes and
triple facts in KG such that potential DDIs can be effectively identified in an end-to-end manner. Given a KG
where the Simplified Molecular Input Line Entry System (SMILES) [32] sequences of drugs are considered as
their attributes, DDKG first integrates these attributes
into the learning process by initializing drug embeddings with their features extracted from corresponding
SMILES sequences with an encoder–decoder layer. It then
propagates the first-order neighboring information to
the receptive fields of drug nodes through top-ranked
network paths, which are determined by the attention
weights computed on both neighboring node embeddings and triple facts. After propagation, an aggregation
process is adopted by DDKG to recursively gather such
neighboring information, thus allowing DDKG to accurately learn the global representations of drug nodes.
Last, for a query pair of drugs, DDKG estimates the interaction probability by simply multiplying their respective
representations. By doing so, the DDI prediction task
can be completed by DDKG in an end-to-end manner.



The main contributions of our work are summarized as

follows:


 - We develop a novel attention-based KG representation learning framework, i.e. DDKG, to fully utilize the
information of biomedical KGs for improved performance of DDI prediction.

 - We construct an encoder–decoder layer to learn
the initial embeddings of drug nodes from their
attributes in the KG, and then simultaneously
consider both neighboring node embeddings and
triple facts to compute the attention weights, which
are used to select top-ranked network paths for
learning the global representations of drug nodes.

 - We have conducted extensive experiments on
two practical biomedical KGs constructed from
benchmark datasets, and the experimental results
have demonstrated the promising performance of
DDKG by comparing it with several state-of-the-art
algorithms.


Details of DDKG


Given a biomedical KG, the proposed framework, i.e.
DDKG, discovers potential DDIs in an end-to-end manner,
and its overall pipeline is illustrated in Figure 1. In particular, DDKG consists of four main steps: (i) KG construction, which generates a comprehensive KG by integrating
the SMILES sequences of drugs and their triple facts
with other entities, such as proteins and diseases; (ii)
drug embedding initialization, which builds an encoder–
decoder layer to learn the initial embeddings of drug
nodes from their corresponding attributes in the KG; (iii)
drug representation learning, which aims to learn the
global representations of drugs by simultaneously considering both neighboring node embeddings and triple
facts; (iv) DDI prediction, which estimates the interaction
probability for pairwise drugs with their respective representations.


**Notations and problem formulation**


In this section, we first introduce the mathematical notations used in our work, and then formulate the DDI
prediction problem with an formal definition of KG.

**Notations.** Throughout this paper, we denote vectors
with lowercase boldface letters ( _e.g._ **e** ∈ R _[d]_, **v** ∈ R _[d]_ ),
matrices by uppercase boldface letters ( _e.g._ **E** ∈ R _[m]_ ×
_n_ ), normal upper characters for hyper-parameters ( _e.g._
_N_, _L_ ), lower characters for scalars or drugs ( _e.g. d_ for
the dimension of entity embedding, _d_ : for drug nodes),
mathematical formal scripts for functions ( _e.g._ _F_, _L_, _A_ ),
and calligraphic scripts for sets ( _e.g._ _G_, _N_, _S_ ).

**KG Definition.** A KG, denoted as _G_, is composed of
a set of triple facts denoted by { _(h_, _r_, _t)_ | _h_, _t_ ∈ _E_, _r_ ∈
_R_ }, where _E_ is the set of molecule entities and _R_ is
the set of relationships. More specifically, a triple fact
_(h_, _r_, _t)_ is a statement interpreted as that a relationship _r_
holds between molecule entities _h_ and _t_ . For example, the


_Attention-based Knowledge Graph Representation Learning_ | 3


Figure 1. An illustration for the overall pipeline of DDKG, which takes Drug 1 as an example to describe the third step of DDKG.



triple fact (Azithromycin, Acquired metabolic disease,
Ciprofloxacin) states the fact that when Azithromycin
and Ciprofloxacin are taken together, it will cause the
Acquired metabolic disease due to their interaction [33].
Moreover, to provide more information about drugs, we
include their SMILES sequences and regard this kind of
information as the attributes of drugs in _G_ . In this regard,
a more comprehensive version of KG, denoted as _G_ [�], is
introduced to distinguish it from _G_, and we have _G_ [�] =
{ _G_, _S_ } = { _E_, _R_, _S_ }, where _S_ is a set of SMILES sequences
for all drug nodes in _G_ . One should note that in order
to avoid introducing any prior information about DDIs
during training, _G_ [�] does not contain any explicit triple
facts of DDIs [28].

**Problem Formulation.** Given two arbitrary drugs, i.e. _d_ _i_
and _d_ _j_, in _G_ [�], DDKG targets to predict whether they are
interact or not. To this end, the main task of DDKG is to
learn a prediction function _F_ for estimating � _y_ _i_, _j_, which is
the probability of _d_ _i_ to interact with _d_ _i_, as:


� _y_ _i_, _j_ = _F_ _(i_, _j_ | _G_ [�], _Θ)_ (1)


where _i_ and _j_ are the subscripts of drug nodes, and _Θ_
denotes a set of trainable parameters involved in _F_ .


**Drug embedding initialization**

When processing the attribute information of drugs in _G_ �, an encoder–decoder layer is constructed to convert

complex irregular-structured SMILES sequences into
real-valued embeddings. To do so, it consists of three
processes including Hash Process, Encoder Process and
Decoder Process,

**Hash Process.** Given an arbitrary drug _d_ _i_ in _G_ [�], its
attribute is the SMILES sequence composed by different
kinds of atoms (e.g. C, H, O) and the connectors between
them. Assuming that _S_ _i_ is the SMILES sequence of _d_ _i_, we
first apply a hash function to obtain a machine understandable vector **e** _i_ ∈ R _[m]_ . Since the hash function is
capable of mapping plaintext of any length to a fixedlength string, DDKG can take the SMILES sequences with



different lengths as input. The hash process is formulated by:


**e** _i_ = hash _(_ [ _c_ 1, _c_ 2, _c_ 3, ..., _c_ _n_ ] _)_ (2)


where _c_ : represents an atom in _S_ _i_ and _n_ is the length of
_S_ _i_ .

**Encoder Process.** In this process, we construct the
encoder layer by stacking Long Short-Term Memory
(LSTM) units [34] in the bi-direction way, namely Bi-LSTM

[35], to capture the attribute information of drugs in a
comprehensive manner. By doing so, each unit at the
current time knows not only the information of previous
units, but also that of subsequent units.

Assuming that **e** ∈ R _[m]_ is the drug embedding obtained
in the hash process, _T_ is the number of LSTM units in
encoder layer and _t_ is the current time, we first reshape
the drug embedding **e** ∈ R _[m]_ into **E** ∈ R _[T]_ [×] _[d]_, where _d_
denotes the latent dimension of the encoder–decoder
layer. Then, the output of a LSTM unit **h** _[(][t][)]_ is computed
according to the output of its previous unit **h** _[(][t]_ [−][1] _[)]_, the cell
state of its previous unit **c** _[(][t]_ [−][1] _[)]_, and the input of its current
unit **e** _[(][t][)]_, which is formulated by:


**h** _[(][t][)]_ = LSTM _(_ **h** _[(][t]_ [−][1] _[)]_, **c** _[(][t]_ [−][1] _[)]_, **e** _[(][t][)]_ _)_ (3)


where **h** _[(][t][)]_, **h** _[(][t]_ [−][1] _[)]_ ∈ R _[T]_, **c** _[(][t]_ [−][1] _[)]_ ∈ R _[T]_, and **e** _[(][t][)]_ = **E** [ _t_ ]. In specific,
we first compute the information received in current cell�

**c** _[(][t][)]_ ∈ R _[m]_, which is formulated by:


�
**c** _[(][t][)]_ = tanh _(_ **W** _c_       - [ **h** _[(][t]_ [−][1] _[)]_, **e** _[(][t][)]_ ] + **b** _c_ _)_ (4)


where **W** _c_ ∈ R _[(][d]_ [+] _[T][)]_ [×] _[T]_ and **b** _c_ ∈ R _[T]_ .

However, during this process, LSTM designs three gates
(input gate, forget gate and output gate) to control the
transmission of information by the weights computed on
**e** _[(][t][)]_ and **h** _[(][t]_ [−][1] _[)]_, as:


**i** _[(][t][)]_ = _σ(_ **W** _i_            - [ **h** _[(][t]_ [−][1] _[)]_, **e** _[(][t][)]_ ] + **b** _i_ _)_ (5)


**f** _[(][t][)]_ = _σ(_ **W** _f_           - [ **h** _[(][t]_ [−][1] _[)]_, **e** _[(][t][)]_ ] + **b** _f_ _)_ (6)


4 | _Su_ et al.



**o** _[(][t][)]_ = _σ(_ **W** _o_      - [ **h** _[(][t]_ [−][1] _[)]_, **e** _[(][t][)]_ ] + **b** _o_ _)_ (7)


where **i** _[(][t][)]_, **f** _[(][t][)]_, **o** _[(][t][)]_ ∈ R _[T]_, **b** : ∈ R _[T]_, and **W** : ∈ R _[(][d]_ [+] _[T][)]_ [×] _[T]_ . Then,
the final cell state **c** _[(][t][)]_ is updated by the weights of forget
and input gates, and the output of **h** _[(][t][)]_ is computed by the
weights of output gate and cell state. Their update rules
are given as follows:


**c** _[(][t][)]_ = **f** _[(][t][)]_ × **c** _[(][t]_ [−][1] _[)]_ + **i** _[(][t][)]_ × **c** [�] _[(][t][)]_ (8)


**h** _[(][t][)]_ = **o** _[(][t][)]_ × tanh _(_ **c** _[(][t][)]_ _)_ (9)


It should be noted that **W** : and **b** : are the trainable
parameters shared by all LSTM units.

In Bi-LSTM, **h** _[(][t][)]_ obtained by (3) is the forward output
and **h** ′ _(t)_ is defined as the reverse output. Hence, the
output of the encoder process, i.e. **H** [�], is obtained by
concatenating the forward and reverse outputs of LSTM
at the time _T_, and we have **H** [�] = [ **h** _[(][T][)]_ ; **h** ′ _(T)_ ].
**Decoder Process.** Different from the encoder layer, the
decoder layer is designed by stacking _T_ LSTM units. In
particular, once the input **H** [�] is received, the decoder layer
learns the drug embeddings that preserve the attribute
features denoted as � **e** ∈ R _[d]_ by (4)–(9).


**Drug representation learning**

After obtaining the initial embeddings of drug nodes, a
representation learning layer is adopted to recursively
propagate and aggregate the first-order neighboring
information along top-ranked network paths such that
the global representations of drug nodes can be more
precisely learned. This layer is composed of three
components: (i) Neighborhood Sampling, (ii) Information
Propagation and (iii) Information Aggregation.

**Neighborhood Sampling.** In order to maintain a constant computational footprint for each batch [36] and
improve the efficiency of DDKG, we uniformly build a
fixed-size set of neighbors for each drug node. Each
neighboring node is considered as a possible receptive
field to extend a particular network path, whose significance is measured by the attention weights of triple facts
on it. In particular, given an arbitrary node _v_ ∈ _E_ and
the set of its triple facts _N_ _(v)_ = { _(v_, _r_, _t)_ | _v_, _t_ ∈ _E_, _r_ ∈ _R_ },
the attention weights of these facts are computed on
the embeddings of involved nodes and their respective
semantic relationships as:


_α(v_, _r_, _t)_ = **e** _r_          - **e** _t_ (10)


where **e** _r_ and **e** _t_ are obtained by _glorot normal_ [37]. As the
decay factor on the propagation along the fact _(v_, _r_, _t)_,
_α(v_, _r_, _t)_ indicates the importance of the relationship _r_
held between _v_ and _t_ . Assuming that _N_ [�] _(v)_ is the set of
receptive fields of _v_, we compose it by selecting top- _N_
neighboring nodes in the descending order of _α(v_, _r_, _t)_ .

**Information Propagation.** For an arbitrary drug node
_d_ _i_, the amount of its first-order neighboring information



Figure 2. An illustration of the representation learning layer. The yellow
nodes are the receptive fields of red nodes, while blue nodes are the
receptive fields of neighboring yellow nodes.


is estimated by the linear combination of its receptive
fields in _N_ [�] _(d_ _i_ _)_ as below:



**e** _N_ � _(di)_ [=] � _α(d_ _i_, _r_, _t)_ **e** _t_ (11)

_t_ ∈ _N_ [�] _(di)_



where _α(d_ _i_, _r_, _t)_ is the attention weight of _(d_ _i_, _r_, _t)_ and **e** _t_ ∈
R _[d]_ is the embedding of _t_ .

**Information Aggregation.** Once obtaining the amount
of neighboring information propagated from _d_ _i_ with
(11), we combine **e** _N_ � _(di)_ [with the initial embedding of] _[ d]_ _i_ [,]
denoted as � **e** _di_, to update the global representation of _d_ _i_,

i.e., **e** _[(]_ [1] _[)]_
_di_ [, according to (][12][).]


**e** _[(]_ _di_ [1] _[)]_ [=] _[ A]_ _[ (]_ **[e]** _N_ [�] _(di)_ [,][�] **[e]** _di_ _[)]_ (12)


In the above equation, **e** _[(]_ [1] _[)]_ ∈ R _[d]_ and _A_ is the aggre_di_

gation function. An obvious advantage of doing so is
that both drug attributes and triple facts in _G_ [�] are simultaneous taken into account during the representation
learning.

In order to learn the global representations for drug
nodes, we further construct the representation layer by
stacking more propagation layers to enlarge the length of
top-ranked network paths, and then aggregate the firstorder neighboring information propagated from selected
neighbors as shown in Figure 2. More specifically, assuming that there are _L_ propagation layers in total, we recursively formulate the representation of _d_ _i_ at the _l_ -th layer

as:



**e** _[(][l][)]_ **e** _[(]_ � _[l]_ [−][1] _[)]_
_di_ [=] _[ A]_ � _N_ _(di)_ [,] **[ e]** _[(]_ _di_ _[l]_ [−][1] _[)]_



(13)
�



where **e** _[(]_ [0] _[)]_
_di_ [=][ �] **[e]** _[d]_ _i_ [, and the amount of neighboring informa-]

tion propagated from all receptive fields for _d_ _i_ in the _l_ -th
layer is defined as:



**e** _[(]_ _N_ � _[l]_ [−] _(_ [1] _di_ _[)]_ _)_ [=] � _α(d_ _i_, _r_, _t)_ **e** _[(]_ _t_ _[l]_ [−][1] _[)]_ (14)

_t_ ∈ _N_ [�] _(di)_


**Table 1.** The statistics of two datasets used in the experiments


**KEGG-drug** **OGB-biokg**


#Drug 1925 10533
#Drug Interaction 56983 1195972
#Entity 129910 93773
#Relation Type 167 51
#KG Triplets 362870 5088434
#Density 0.0043% 0.12%


2∗ _Entity_
Density is defined as _KG_ ∼ _Triple_ [2] [.]


where **e** _[(]_ _t_ _[l]_ [−][1] _[)]_ is the representation of _t_ generated from
the previous layer. With (14), the first-order neighboring
information originated from all the receptive fields of _d_ _i_
at each layer can thus be memorized [38].


**DDI prediction**

In our work, the task of DDI prediction is solved as a
binary classification problem. Hence, given a set of drugs
_D_ and the ground-truth DDIs, we train DDKG by minimizing the following binary cross-entropy loss function [39]:



� �

_L_ _(Θ)_ = − � _y_ _i_, _j_ log _(y_ _i_, _j_ _)_ + _(_ 1 − _y_ _i_, _j_ _)_ log _(_ 1 − _y_ _i_, _j_ _)_ (15)

_i_, _j_ ∈ _D_, _i_ ̸= _j_



where � _y_ _i_, _j_ = _F_ _(_ **e** _[(]_ _di_ _[L][)]_ [,] **[ e]** _[(]_ _dj_ _[L][)]_ _[)]_ [,] _[ y]_ _[i]_ [,] _[j]_ [ is the binary value indicating]

the existence of a DDI between _d_ _i_ and _d_ _j_, and _Θ_ denotes
the set of trainable parameters.

For a query pair of drugs, i.e. ( _d_ _i_, _d_ _j_ ), their global representations, i.e. **e** _[(][L][)]_
_di_ [and] **[ e]** _[(]_ _dj_ _[L][)]_ [, can be obtained with (][13][)]

and (14). To predict DDIs in an end-to-end manner, we
specifically design the scoring function _F_ based on the
inner product of **e** _[(][L][)]_
_di_ [and] **[ e]** _[(]_ _dj_ _[L][)]_ [, and the definition of] _[ F]_ [ is]

given as follows:



_F_ _(_ **e** _[(][L][)]_ **e** _[(][L][)]_
_di_ [,] **[ e]** _[(]_ _dj_ _[L][)]_ _[)]_ [ =] _[ σ]_ � _di_ [·] **[ e]** _[(]_ _dj_ _[L][)]_



(16)
�



_Attention-based Knowledge Graph Representation Learning_ | 5


**Baseline algorithms**

For the purpose of demonstrating the effectiveness of
DDKG, several state-of-the-art algorithms proposed for
DDI prediction are included as the baseline algorithms
as follows and their performances are also evaluated in
the experiments.


 - **Laplacian** [17]: It is a representative MF-based model,
which solves network representation learning task
by factorizing the matrix of input data into lower
dimensional matrices.

 - **GraRep** [18]: GraRep is also a typical MF-based model
that learns the low-dimensional representation by
matrix factorization.

 - **NEDTP** [46]: It applies RW to extract the topology
information of drug node in the combination of
drug similarity network and heterogeneous network
and classify the drug pairs with Gradient Boosting
Decision Tree [47].

 - **LINE** [48]: It is a network representation learning
method based on neural network, which learns
the final representation by designing two kinds of
proximities and optimizing them simultaneously.

 - **SDNE** [49]: It can be regarded as an extension of LINE,
and it is also the first method to apply deep learning
to network representation learning by using automatic encoder[50].

 - **KGNN** [28]: It is constructed based on the combination of knowledge graph and neural network, capturing the structure of knowledge graph by mining the
relations.

 - **EEG-DTI** [51]: It learns drug representations using a
spatial-based GCN model[36] and predicts drug pairs
by the learned features.

 - **GCN-DTI** [52]: It first uses a traditional GCN[53] to
learn features for drugs and then adopts a deep
neural network to predict the final labels for drug
pairs.

 - **ComplEx** [27]: It measures plausibility of facts by
matching latent semantics of entities and relations
embodied in their vector space representations.

 - **TransE** [54]: It measures the plausibility of a fact as
the distance between the two entities, usually after a
translation carried out by the relation.


In order to facilitate the following discussions, we
divide the baseline algorithms into two groups according to their models: (i) Network Embedding (NE)-based
algorithms, including Laplacian, GraRep, NEDTP, LINE
and SDNE; (ii) GNN-based algorithms, including KGNN,
EEG-DTI and GCN-DTI; (iii) KG-based algorithms, including ComplEx and TransE. The main difference between
these three groups is that NE-based algorithms solve the
DDI prediction problem by using conventional network
representation learning methods, such as MF and RW,
GNN-based algorithms apply GNN or its variants, such
as GCN and spatial-based GCN, to predict DDIs, while KGbased algorithms embed both entities and relations into



In (16), _σ_ is implemented as the sigmoid _(_  - _)_ activation
function [40], which is widely adopted to address binary
classification problems [41, 42].


Experiments
**Datasets**


In order to evaluate the performance of DDKG, two
datasets with different sizes are collected and they
are KEGG-drug[43] and OGB-biokg[44]. Regarding the
attributes of drugs, the SMILES sequences of drugs
are downloaded from DrugBank as of version 5.1.7

[45], which is a popular and widely used database
providing a variety of drug information. The detailed
statistics of KEGG-drug and OGB-biokg are summarized
in Table 1.


6 | _Su_ et al.


Figure 3. AUC and AUPR curves obtained by DDKG and baseline algorithms.


continuous vector spaces, so as to simplify the manipulation while preserving the inherent structure of the KG

[55].


**Experiment settings**

DDKG is implemented with Tensorflow [56] on a working
machine equipped with Intel Core I7 2.6 GHz and 16 GB
RAM. The above baseline algorithms are also deployed on
the same machine, and their parameters are set by the
values recommended in their original work. Besides, the
embedding size is fixed as 32 ( _d_ = 32) for all algorithms
used in comparison. As for the parameters of DDKG, we
set _N_ = 4 and _L_ = 2, and the reason will be analyzed in the
section of Parameter Sensitivity Analysis. Moreover, we
randomly divide all approved DDIs as positive samples
into training, validation and test sets in a 8:1:1 ratio, and
then the same number of negative samples are randomly
selected from the complement set of positive samples in
all phases.


**Results and analysis**

In this section, we first evaluate proposed model and all
baseline models with 5-fold cross validation (CV) and
four evaluation metrics are used to quantitatively measure the performances of all algorithms on two datasets,
and they are Accuracy (Acc.), F1 Score, Area Under
Curve (AUC) and Area Under Precision-Recall (AUPR). In
Table 2, we present the average values and their standard
deviations for each metric under a five-fold cross
validation scheme.The AUC and AUPR curves obtained

by each algorithm is also depicted in Figure 3. In addition,
we also use some rank-based metrics such as MAP

(Mean Average Precision), MRR (Mean Reciprocal Rank),
HIT@5 and HIT@10, to demonstrate the performance
of proposed model comprehensively, Figure 4 shows
the rand-based results achieved by DDKG and other
baseline models.


**Comparison with NE-based Algorithms.** According
to Table 2, it can be observed that DDKG consistently



Figure 4. Rank-based results achieved by DDKG and baseline algorithms.


yields the best performance on both two datasets
when compared with NE-based algorithms. The main
reason for this is that the performance of NE-based
algorithms are constrained by their ability in modeling
and integrating extra information. In other words, NEbased algorithms only consider the topology information of biomedical networks while ignoring the rich
attribute information in biomedical entities and their

relationships, while DDKG integrates such information
into the representation learning process, which in return
results in a promising performance. On the other hand,
compared with NE-based algorithms, DDKG has the
advantage in making use of the SMILES sequences of
drugs for the initialization of drug embeddings, which
also contributes to the improvement of performance.
Moreover, considering the standard deviations achieved
by NE-based algorithms, we also note that the robustness of NE-based algorithms is worse than that of
DDKG, as their standard deviations are larger by 2–
3 times than that of DDKG on both KEGG-drug and
OGB-biokg.

According to Table 2, it can be observed that DDKG
consistently yields the best performance on both two
datasets when compared with NE-based algorithms. In
a way, the poor performance of NE-based algorithms are
constrained by their ability in modeling and integrating
extra information, since they only consider the topology
information of biomedical networks while ignoring
the rich attribute information in biomedical entities

and their relationships, but DDKG integrates such
information into the representation learning process. In
addition, DDKG has the advantage in making use of the
SMILES sequences of drugs for the initialization of drug
embeddings, which also contributes to the improvement
of performance. In another way, it can be observed
that the experimental results achieved by DDKG are
almost 15% higher than those NE-based baselines, which
indicates that DDKG is good at handling sparse networks.
The possible reason is DDKG is able to enhance the


_Attention-based Knowledge Graph Representation Learning_ | 7


**Table 2.** Experimental results of DDKG and baseline models on KEGG-drug and OGB-biokg datasets


**Datasets** **Types** **Methods** **Acc.** **F1 Score** **AUC** **AUPR**


KEGG-drug NE-based Laplacian 0.7955±0.0050 0.8008±0.0049 0.8634±0.0047 0.8587±0.0055
GraRep 0.7744±0.0011 0.7796±0.0014 0.8499±0.0011 0.8406±0.0022

NEDTP 0.7516±0.0041 0.7556±0.0044 0.8148±0.0049 0.8067±0.0078

LINE 0.7504±0.0076 0.7541±0.0073 0.8155±0.0068 0.8079±0.0061

SDNE 0.7400±0.0014 0.7427±0.0013 0.8043±0.0014 0.7916±0.0026

GNN-based KGNN 0.8655±0.0013 0.8710±0.0023 0.9268±0.0044 0.8988±0.0026

EEG-DTI 0.8831±0.0012 0.8881±0.0029 0.9362±0.0013 0.9067±0.0016

GCN-DTI 0.9031±0.0014 0.9058±0.0025 0.9546±0.0016 0.9357±0.0018

KG-based ComplEx 0.8797±0.0017 0.8849±0.0013 0.9379±0.0014 0.9145±0.0015

TransE 0.8688±0.0012 0.8724±0.0011 0.9284±0.0014 0.9016±0.0016

**DDKG** **0.9098** ± **0.0007** **0.9119** ± **0.0008** **0.9606** ± **0.0012** **0.9466** ± **0.0014**


OGB-biokg NE-based Laplacian 0.7630±0.0004 0.7634±0.0003 0.7893±0.0002 0.7902±0.0002
GraRep 0.7635±0.0006 0.7639±0.0006 0.7892±0.0004 0.7902±0.0004

NEDTP 0.7825±0.0004 0.7825±0.0003 0.7946±0.0002 0.7962±0.0001

LINE 0.8203±0.0003 0.8303±0.0003 0.8456±0.0005 0.8469±0.0004

SDNE 0.8248±0.0006 0.8250±0.0006 0.8432±0.0004 0.8446±0.0004

GNN-based KGNN 0.7289±0.0018 0.7331±0.0012 0.7649±0.0012 0.7326±0.0011

EEG-DTI 0.7904±0.0010 0.8102±0.0008 0.7862±0.0009 0.7346±0.0009

GCN-DTI 0.8882±0.0008 0.8898±0.0008 0.9384±0.0006 0.9181±0.0007

KG-based ComplEx 0.7490±0.0015 0.7392±0.0010 0.7998±0.0010 0.7774±0.0008

TransE 0.7224±0.0010 0.7265±0.0010 0.7511±0.0009 0.6986±0.0011

**DDKG** **0.9953** ± **0.0002** **0.9953** ± **0.0002** **0.9996** ± **0.0005** **0.9994** ± **0.0004**


Best results are bolded.



network density by neighbor sampling when applied on
sparse network, so as to increase the expression ability
of proposed model. Moreover, considering the standard
deviations achieved by NE-based algorithms, we also
note that the robustness of NE-based algorithms is worse
than that of DDKG, as their standard deviations are larger
by 2–3 times than that of DDKG on both KEGG-drug and
OGB-biokg.

Regarding the results obtained by NE-based algorithms, we find that the performances of these algorithms vary greatly. In particular, Laplacian outperforms
the other NE-based algorithms on KEGG-drug, but it
achieves the worst performance on OGB-biokg. One
possible reason for this phenomenon is that the MF
model adopted by Laplacian aims to project the high
dimensional matrix into lower dimensional matrices,
and such operation is more appropriate for sparse or lowdensity networks, such as KEGG-drug. On the contrary,
for NE-based algorithms designed on neural network
models, such as LINE and SDNE, we note that they
achieve a satisfactory performance on OGB-biokg, which
is denser. The reasons are two-fold. On the one side,
both LINE and SDNE are capable of learning deeper
representations and capturing the connections in a highdimensional space via deep learning. On the other side,
taking LINE as an example, the node representations
are learned by designing first-order and second-order
proximities, which are optimized simultaneously [57].
The second-order proximity is defined to measure the
similarity between pairwise nodes by their common
neighborhoods [58]. However, some nodes in the KG of
KEGG-drug may have no such a second-order proximity
due to the sparsity of KEGG-drug, thus resulting in an



incomplete learning of topology information. In addition
to the above algorithms, NEDTP that adopts RW to
learn representations yields better results on both two
datasets, but it is more stable when applied to largescale datasets, which is also observed among the other
NE-based algorithms.

**Comparison** **with** **GNN-based** **Algorithms.** When
comparing DDKG with GNN-based algorithm, we note
from Table 2 that DDKG also yields a better performance
than GNN-based algorithm across all datasets. The
main reason for the unsatisfactory performance of
GNN-based algorithms is that they are incapable of
identifying and filtering out the noises [38]. Moreover,
these GNN-based algorithms are not able to process
the drug attributes in the KG. Although KGNN, EEG-DTI
and DDKG adopt the network structure of spatial-based
GCN for learning drug representations, DDKG proposes
a different sampling strategy that allows it to select
neighboring nodes along top-ranked network paths, and
thus achieves a superior performances on both two
datasets. Obviously, using such a sampling strategy not
only alleviates the influence of random sampling, but
also avoids the noises generated from considering all
neighboring nodes. Most importantly, all of these GNNbased algorithms only consider the triple facts during
the representation learning process while ignoring the
attribute information of drugs. The superior performance
of DDKG could be a strong indicator to verify the
significance of incorporating the SMILES sequences of
drugs to construct _G_ [�] .
There are also several points worth noting. First, GCNDTI consistently performs better than KGNN and EEGDTI on all datasets. The reason is that GCN-DTI adopts


8 | _Su_ et al.


a traditional GNN to learn node representations from
the whole network, while KGNN and EEG-DTI that use
a spatial-based GCN only learn them from a part of
network, and some important structural information
could thus be missed. Second, though KGNN and EEGDTI outperform NE-based algorithms on the KEGGdrug dataset, their performance on the OGB-biokg
dataset is fair, and particularly KGNN achieves the worst
performance. Although it is easier for KGNN and EEG-DTI
to capture the structural characteristics from sparser
networks, these spatial-based features are not sufficient
to represent the structural information of a large-scale
network with high-density.

**Comparison with KG-based Algorithms.** DDKG apparently performs better than both ComplEx and TransE
on all the datasets, and the reasons are two-fold. First,
though DDKG and KG-based models tale both entities
and their relations into account, ComplEx and TransE
conduct the representation learning by only using the
1-hop information of KGs, whereas DDKG is capable of
capturing the deep structure of KGs by increasing its
layers to consider higher-order hop information. Second,
DDKG is also integrated with extra information such as
drug SMILES sequences, which in return improved the
performance of DDKG as indicated by ablation study.


**Results and discussion on realistic scenarios**


As mentioned above, we use traditional 5-fold CV to
evaluate DDKG and baseline models, which the samples
are partitioned into equal sized 5-subsets where one
subset is used as a test set and the other sets are used to

train model. Although traditional 5-fold CV is capable of
avoiding the model suffering from over-fitting to a certain
extent, it may lead to optimistic results [59]. To make
realistic evaluation of DDI prediction task, we further
evaluate DDKG and other baseline models on two real
istic scenarios proposed by [60, 61]: (i) drug-wise disjoint
CV (DW-CV) and (ii) pairwise disjoint CV (PW-CV), and
compare the results with that achieved by Traditional CV.
Table 3 and Figure 5 report the results on traditional CV
and other two realistic scenarios.


In particular, for the KEGG-drug dataset, the overall
accuracy of DDKG is better than all baseline models,
as it achieves the best performance across most metrics under three CV schemes. A possible reason for that
phenomenon was that DDKG was able to capture deep
structural of KGs by integrating extra information. In
addition, DW-CV is mainly designed to evaluate the ability of modeling dynamic nodes by predicting the interactions between unseen drug nodes and existing drug
nodes in a given KG with the trained model. Though
the performance of all models under the DW-CV scheme
is not as good as they obtained under the other two
schemes, DDKG still achieves a better performance when
compared with each of baseline models, thus indicating its robustness in predicting drug-drug interactions
to a certain extent. Lastly, the PW-CV scheme aims to
verify the ability in processing unseen nodes, as the



drug pairs in the testing set are composed of drugs
whose interactions are unknown by following the PWCV scheme. The superior performance of DDKG is also
a strong indicator that the proposed model is preferred
over all baseline models in discovering novel drug-drug
associations mainly due to its ability in processing extra
information, such as drug SMILES sequences.

Besides, we also note the promising performance of
DDKG obtained on the OGB-biokg dataset across all CV
schemes when using the rank-based metrics for evaluation. Considering the difference in the performance of
DDKG between the OGB-biokg and KEGG-drug datasets,
a conclusion could be made that the prediction results
of DDKG are more accurate especially when applied to
large dense KGs.


**Ablation study**

To investigate the effects of embedding initialization and
neighborhood sampling used in the encoder-decoder and
representation learning layers respectively, an in-depth
ablation study has been performed. Moreover, additional
experiments have been conducted to analyze how different aggregation functions affect the performance of
DDKG. It should be noted that DDKG adopts _A_ _concat_ =
_σ(_ **W** _c_ - _(_ **e** _[(]_ _di_ _[l]_ [−][1] _[)]_ || **e** _[(]_ _N_ � _[l]_ [−] _(_ [1] _di_ _[)]_ _)_ _[)]_ [ +] **[ b]** _[c]_ _[)]_ [ as default aggregation function.]

In the experiments, four variants of DDKG are developed
and their brief descriptions are given as below. Experimental results of these variants are presented in Table 4.


 - **DDKG-E** : It only considers the triple facts in KG by
removing the encoder-decoder layer. We use _glorot_
_normal_ to initialize the drug embeddings.

 - **DDKG-L:** It adopts LSTM as encoder–decoder layer
and the other parts are all the same as DDKG.

 - **DDKG-A** : When compared with DDKG, DDKG-A samples the neighbors for propagation and aggregation in
a random way. The rest part of DDKG-A is the same
as DDKG.

 - **DDKG** _sum_ : The difference between DDKG _sum_ and
DDKG is that DDKG _sum_ uses _A_ _sum_ as its aggregation
function, which is defined as _A_ _sum_ = _σ(_ **W** _s_   - _(_ **e** _[(]_ _N_ � _[l]_ [−] _(_ [1] _di_ _[)]_ _)_ [+]

**e** _[(][l]_ [−][1] _[)]_ _)_ + **b** _s_ _)_ by following [62].
_di_

 - **DDKG** _neigh_ : A different aggregation function, i.e.,

_A_ _neigh_, is used in this variant, and it aggregates the
information in _A_ _neigh_ = _σ(_ **W** _n_   - **e** _[(]_ _N_ � _[l]_ [−] _(_ [1] _di_ _[)]_ _)_ [+] **[b]** _[n]_ _[)]_ [ by following]

[28].


In the aggregation functions of DDKG _sum_ and DDKG _neigh_,
**W** : and **b** : are the trainable parameters.

**Effect of Drug Embedding Initialization.** When the
encoder–decoder layer is removed, the performance of
DDKG decreases to different extent for each metrics, thus
indicating the importance of integrating the attribute
information of drugs into the initialization of drug
embeddings. According to the results shown in Table 4,
the consideration of drug embedding initialization
contributes more to the OGB-biokg dataset,as it improves
the Acc. of DDKG by 14.27% on OGB-biokg, whereas only


_Attention-based Knowledge Graph Representation Learning_ | 9


**Table 3.** Experimental results of DW-CV and PW-CV


**Datasets** **Methods** **Traditional CV** **DW-CV** **PW-CV**


**Acc.** **F1 Score** **AUC** **AUPR** **Acc.** **F1 Score** **AUC** **AUPR** **Acc.** **F1 Score** **AUC** **AUPR**


KEGG-drug Laplacian 0.7955 0.8008 0.8634 0.8587 0.5869 0.5133 0.6644 0.6955 0.7291 0.7067 0.7936 0.7792
GraRep 0.7744 0.7796 0.8499 0.8406 0.6463 0.5600 0.7011 0.6538 0.7125 0.6945 0.7796 0.7623

NEDTP 0.7516 0.7556 0.8148 0.8067 0.5458 0.4955 0.5655 0.5930 0.6507 0.6120 0.7049 0.6553

LINE 0.7504 0.7541 0.8155 0.8079 0.5380 0.3525 0.5099 0.4618 0.6375 0.5903 0.6961 0.6934

SDNE 0.7400 0.7427 0.8043 0.7916 0.5237 0.3777 0.5164 0.5404 0.6399 0.5991 0.6868 0.6755

KGNN 0.8655 0.8710 0.9268 0.8988 0.5306 0.5537 0.5426 0.5552 0.7276 0.7230 0.8065 0.7844

EEG-DTI 0.8831 0.8881 0.9362 0.9067 0.5124 0.3416 0.5265 0.5045 0.6744 0.6830 0.7425 0.7552

GCN-DTI 0.9031 0.9058 0.9546 0.9357 0.6873 0.7106 0.7489 0.7577 0.7021 0.6173 0.7989 0.7701

ComplEx 0.8797 0.8849 0.9379 0.9145 0.5306 0.5537 0.5426 0.5552 0.7276 0.7230 0.8065 0.7844

TransE 0.8688 0.8724 0.9284 0.9016 0.5558 0.5172 0.5282 0.5124 0.7101 0.7026 0.7808 0.7689

DDKG **0.9098** **0.9119** **0.9606** **0.9466** **0.7168** **0.7108** **0.7900** **0.7718** **0.7391** **0.7426** **0.8104** **0.7899**


OGB-biokg Laplacian 0.7630 0.7634 0.7893 0.7902 0.5503 0.4159 0.6182 0.6063 0.7085 0.5535 0.8519 0.8447
GraRep 0.7635 0.7639 0.7892 0.7902 0.5613 0.5045 0.6469 0.6155 0.6862 0.5443 0.8483 0.8540

NEDTP 0.7825 0.7825 0.7946 0.7962 0.5458 0.4955 0.5818 0.5903 0.6989 0.5591 0.8584 **0.8914**

LINE 0.8203 0.8303 0.8456 0.8469 0.6276 0.6983 0.7299 0.7161 0.6979 0.7051 0.8882 0.8900

SDNE 0.8248 0.8250 0.8432 0.8446 0.6622 0.6890 0.7564 0.7202 0.6712 0.7010 0.8696 0.8730

KGNN 0.7289 0.7331 0.7649 0.7326 0.7578 0.7753 0.8666 0.8861 0.8098 0.7735 0.8841 0.8367

EEG-DTI 0.7904 0.8102 0.7862 0.7346 0.6788 0.6283 0.8607 0.8587 0.7737 0.6988 0.7987 0.7240

GCN-DTI 0.8882 0.8898 0.9384 0.9181 0.6756 0.6998 0.8743 0.8751 0.7482 0.7854 0.7947 0.7204

ComplEx 0.7490 0.7392 0.7998 0.7774 0.6578 0.6753 0.8666 **0.8861** 0.6275 0.6490 0.7240 0.6889

TransE 0.7224 0.7265 0.7511 0.6986 0.6481 0.6076 0.7841 0.8367 0.6098 0.6735 0.6881 0.6218

DDKG **0.9953** **0.9953** **0.9996** **0.9994** **0.8743** **0.8724** **0.9383** 0.8097 **0.8725** **0.8790** **0.9381** 0.8595


Best results are bolded.


Figure 5. Results in various CV schemes on KEGG and OGB-biokg Datasets.


**Table 4.** Experimental results of ablation study on the KEGG-drug and OGB-biokg datasets


**Datasets** **Methods** **Acc.** **F1 Score** **AUC** **AUPR**


KEGG-drug DDKG-E 0.8797±0.0009 0.8849±0.0014 0.9380±0.0032 0.9145±0.0024

DDKG-L 0.8869±0.0009 0.8913±0.0011 0.9429±0.0024 0.9189±0.0020

DDKG-A 0.8839±0.0010 0.8898±0.0018 0.9404±0.0029 0.9162±0.0019

DDKG _sum_ 0.9048±0.0013 0.9068±0.0021 0.9559±0.0015 0.9390±0.0015
DDKG _neigh_ 0.9025±0.0010 0.9048±0.0013 0.9557±0.0018 0.9385±0.0014
**DDKG** **0.9098** ± **0.0007** **0.9119** ± **0.0008** **0.9606** ± **0.0012** **0.9466** ± **0.0014**


OGB-biokg DDKG-E 0.8526±0.0012 0.8564±0.0010 0.9142±0.0008 0.8827±0.0008

DDKG-L 0.8877±0.0010 0.8927±0.0010 0.9429±0.0006 0.9228±0.0006

DDKG-A 0.9604±0.0008 0.9669±0.0006 0.9685±0.0006 0.9679±0.0006

DDKG _sum_ 0.9925±0.0006 0.9924±0.0006 0.9992±0.0008 0.9988±0.0005
DDKG _neigh_ 0.9894±0.0009 0.9894±0.0009 0.9989±0.0008 0.9985±0.0007
**DDKG** **0.9953** ± **0.0002** **0.9953** ± **0.0002** **0.9996** ± **0.0005** **0.9994** ± **0.0004**


Best results are bolded.


10 | _Su_ et al.


3.01% on KEGG-drug. A possible reason for this is that
the global topology information of drug nodes may be
more similar in a large-scale KG with a dense structure.
Moreover, it can be observed that the performance of
DDKG decreases slightly when using LSTM to build
the encoder–decoder layer. A possible reason for this is
that the model with Bi-LSTM is capable of learning the
sequence information of drug SMILES comprehensively.
Therefore, integrating the attribute information of
drugs can enrich the drug representations to a certain
extent, and so as to improve the performance of DDKG
considerably.

**Effect of Neighborhood Sampling.** Comparing the
experimental results obtained by DDKG-A and DDKG,
it is observed that DDKG achieves a better performances
on all the datasets when adopting the neighborhood
sampling strategy, which makes use of the attention
mechanism to effectively filter out noisy nodes and
enhances the expressive ability of drug representations
to a certain extent. Moreover, the neighborhood sampling
strategy can improve the robustness of DDKG, as lower
standard deviations are obtained by DDKG among all
metrics. In this regard, the significance of neighborhood
sampling can be verified.

**Effect of The Aggregation Function.** In our work,
the representations of drug nodes at the _l_ -th layer are
updated by concatenating their representations obtained
at the previous layer and neighboring node embeddings.
To evaluate the effectiveness of such an aggregation
function, the performance of DDKG is compared with
two variants, i.e. DDKG _sum_ and DDKG _neigh_ . From the experimental results given in Table 4, several observations
can be made. First, compared with the effects of drug
embedding initialization and neighborhood sampling,
the aggregation function has the least impact on the
performance of DDKG. Second, updating node embeddings only by the embeddings of neighboring nodes is
not able to learn more distinguishing characteristics,
as DDKG _neigh_ achieves the worst performance across
all datasets and its robustness is also the worst. Last,
though DDKG _sum_ performs better than DDKG _neigh_, it is
still not an elegant solution, as it simply treats **e** _[(]_ _N_ � _[l]_ [−] _(_ [1] _di_ _[)]_ _)_
and **e** _[(][l]_ [−][1] _[)]_ without any difference, while DDKG adopts a
_di_

more reasonable aggregation function to enhance the
associations between them.


Parameter sensitivity analysis


As mentioned above, the number of selected neighboring
nodes, i.e. _N_ and the number of propagation layers, i.e.
_L_, are two hyper-parameters that need to be tuned for a
better performance of DDKG. In particular, _N_ controls the
scale of receptive fields to compose _N_ [�] _(v)_, while _L_ determines the depth of information aggregation. To analyze
the sensitivity of DDKG to the change of _N_ and _L_, we
have conducted a series of experiments by varying the
values of _N_ and _L_ from the sets {1, 2, 3, 4, 5, 6} and {1, 2,
3, 4}, respectively. Given different combinations of _N_ and



Figure 6. (A) The performance of DDKG in terms of Acc. given different
values of _N_ and _L_ . (B) The performance of DDKG in terms of AUC given
different values of _N_ and _L_ .


Figure 7. (A) The change in CPU time taken by DDKG given different
values of _N_ and _L_ . (B) The change in epoch number taken by DDKG given
different values of _N_ and _L_ .


_L_, the performance of DDKG in terms of Acc. and AUC is
reported in Figure 6.

When the value of _N_ increases, we note that the performance of DDKG is also gradually improved and achieves
its best when _N_ is set as 4 or 5. The reason why small
values of _N_ lead to an unsatisfactory performance of
DDKG is that a small value of _N_ is incapable of capturing
sufficient neighboring nodes during information aggregation. The performance of DDKG starts to decrease
when _N_ is set as 6 due to the existence of unexpected
noises. Moreover, though DDKG is capable of achieving a
better performance with a larger value of _N_, its convergence speed is negatively affected, as DDKG takes more
time with a larger _N_ for training according to Figure 7.
Hence, the recommended value of _N_ is 4, with which
DDKG is more efficient while maintaining a comparable

accuracy.

As shown in Figure 6 and Figure 7, DDKG obtains its
best performance when _L_ is set as 2 or 3. Since there are a
total of _N_ _[L]_ nodes involved in the representation learning
for a particular drug node, DDKG is prone to be affected
by the noisy data resulted from the increase of _L_ . It is
for this reason that the performance of DDKG degrades
when _L >_ 3. Moreover, due to the exponential increase of
involved nodes, we also note that DDKG trained with a
larger _L_ takes more CPU time and epoch number to reach
a convergence. As a result, the value of _L_ is recommended
as 2.


Conclusion


In this work, an attention-based representation learning
framework, i.e. DDKG, is proposed to identify potential
DDIs from biomedical KGs in an end-to-end manner. By
taking the SMILES sequences of drugs as their attributes
in a given KG, DDKG constructs an encoder-decoder layer


to obtain the initial embeddings of drugs from their
attribute information. Then, it learns the representations
of drugs by recursively propagating and aggregating the
embedding information along top-ranked network paths,
which are determined by triple facts and neighboring
node embeddings. Finally, the probability of being interacting for pairwise drugs is estimated by simply multiplying their respective representations. Extensive experiments on two golden standard biomedical KG datasets
have demonstrated the promising accuracy of DDKG,
as it outperforms several state-of-the-art DDI prediction
algorithms in terms of several evaluation metrics.

Regarding future work, there is still room for further
improving the performance of DDKG. To begin with,
DDKG only learns the initial embeddings for drug nodes,
and we would like to combine with other factorization

models like [63, 64] to initialize embeddings for all nodes
in a KG so as to better model asymmetric relations [65,
66]. Moreover, the network paths selected to propagate
the first-order neighboring information are determined
by the attention weights of triple facts, and it is difficult
for DDKG to obtain a global optimal solution. Hence,
we intend to adopt a multihop attention mechanism

[67] or conjoint attentions [68] to address this problem.
Last, we are interested in applying DDKG to predict
other kinds of associations in bioinformatics, such as
protein–protein interactions [69, 70] and drug–disease
associations [71].


**Key Points**


  - We develop a novel attention-based KG representation
learning framework, i.e. DDKG, to fully utilize the information of biomedical KGs for improved performance of
DDI prediction.

  - We construct an encoder–decoder layer to learn the
initial embeddings of drug nodes from their attributes
in the KG, and then simultaneously consider both neighboring node embeddings and triple facts to compute the
attention weights, which are used to select top-ranked
network paths for learning the global representations of
drug nodes.

  - We have conducted extensive experiments on two
practical biomedical KGs constructed from benchmark
datasets, and the experimental results have demonstrated the promising performance of DDKG by comparing it with several state-of-the-art algorithms.


Data availability


The dataset and source code can be freely downloaded
from https://github.com/Blair1213/DDKG.


Author contributions statement


X.S., L.H. and P.H. conceived the experiments and
conducted the experiments, Z.Y. and B.Z. analyzed the
results.



_Attention-based Knowledge Graph Representation Learning_ | 11


Acknowledgments


The authors would like to thank colleagues and the
anonymous reviewers who have provided valuable feedback to help improve the paper.


Funding


The Natural Science Foundation of Xinjiang Uygur
Autonomous Region under grant 2021D01D05, in part
by the Pioneer Hundred Talents Program of Chinese
Academy of Sciences, in part by the National Natural Science Foundation of China under grant 62172355, in part
by the Awardee of the NSFC Excellent Young Scholars
Program under grant 61722212, in part by the Science and
Technology Innovation 2030-New Generation Artificial
Intelligence Major Project, under grant 2018AAA0100100
and the Tianshan youth - Excellent Youth under grant
2019Q029.


References


1. Giacomini KM, Krauss RM, Roden DM, _et al._ When good drugs go
bad. _Nature_ 2007; **446** (7139):975–7.
2. Percha B, Altman RB. Informatics confronts drug–drug interactions. _Trends Pharmacol Sci_ 2013; **34** (3):178–84.
3. Aronson JK. Classifying drug interactions. _Br J Clin Pharmacol_
2004; **58** (4):343.
4. Finkel R, Clark MA, Cubeddu LX. _Pharmacology_ . Lippincott
Williams & Wilkins, 2009.

5. Ralph Edwards I, Aronson JK. Adverse drug reactions: definitions, diagnosis, and management. _The lancet_ 2000; **356** (9237):

1255–9.

6. Yang Q, Zhang Y, Deng Y, _et al._ A comprehensive review of
computational methods for drug-drug interaction detection.
_IEEE/ACM Trans Comput Biol Bioinform_ 2021.
7. Yamanishi Y, Araki M, Gutteridge A, _et al._ Prediction of drug–
target interaction networks from the integration of chemical
and genomic spaces. _Bioinformatics_ 2008; **24** (13):i232–40.
8. Li P, Huang C, Yingxue F, _et al._ Large-scale exploration and
analysis of drug combinations. _Bioinformatics_ 2015; **31** (12):2007–

16.

9. Vilar S, Uriarte E, Santana L, _et al._ Detection of drug-drug interactions by modeling interaction profile fingerprints. _PloS one_
2013; **8** (3):e58321.
10. Rogers D, Hahn M. Extended-connectivity fingerprints. _J Chem Inf_
_Model_ 2010; **50** (5):742–54.
11. Deng Y, Yang Q, Xu X, _et al._ Meta-ddie: predicting drug–
drug interaction events with few-shot learning. _Brief Bioinform_
2022; **23** (1):bbab514.
12. Zhang W, Jing K, Huang F, _et al._ Sflln: a sparse feature learning ensemble method with linear neighborhood regularization for predicting drug–drug interactions. _Inform Sci_ 2019; **497** :

189–201.

13. Deng Y, Xinran X, Qiu Y, _et al._ A multimodal deep learning framework for predicting drug–drug interaction events. _Bioinformatics_
2020; **36** (15):4316–22.
14. Zhang D, Yin J, Zhu X, _et al._ Network representation learning: a
survey. _IEEE transactions on Big Data_ 2018; **6** (1):3–28.
15. Zhang Y, Qiu Y, Cui Y, _et al._ Predicting drug-drug interactions
using multi-modal deep auto-encoders based network embedding and positive-unlabeled learning. _Methods_ 2020; **179** :37–46.


12 | _Su_ et al.


16. Xiaorui S, You Z, Wang L, _et al._ Sane: a sequence combined attentive network embedding model for covid-19 drug repositioning.
_Appl Soft Comput_ 2021; **111** :107831.
17. Belkin M, Niyogi P. Laplacian eigenmaps for dimensionality
reduction and data representation. _Neural_ _Comput_
2003; **15** (6):1373–96.
18. Shaosheng Cao, Wei Lu, and Qiongkai Xu. Grarep: Learning
graph representations with global structural information. In
_Proceedings of the 24th ACM international on conference on information_
_and knowledge management_, pages 891–900, 2015.
19. Zhang W, Chen Y, Li D, _et al._ Manifold regularized matrix factorization for drug-drug interaction prediction. _J Biomed Inform_

2018; **88** :90–7.

20. Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. Deepwalk:
Online learning of social representations. In _Proceedings of the_
_20th ACM SIGKDD international conference on Knowledge discovery_
_and data mining_, pages 701–10, 2014.
21. Shengzhong Zhang, Zengfeng Huang, Haicang Zhou, and Ziang
Zhou. Sce: Scalable network embedding from sparsest cut. In
_Proceedings of the 26th ACM SIGKDD International Conference on_
_Knowledge Discovery & Data Mining_, pages 257–65, 2020.
22. Chang S, Tong J, Zhu Y, _et al._ Network embedding in biomedical
data science. _Brief Bioinform_ 2020; **21** (1):182–97.
23. Pengwei H, Huang Y-a, Mei J, _et al._ Learning from low-rank multimodal representations for predicting disease-drug associations.
_BMC Med Inform Decis Mak_ 2021; **21** (1):1–13.
24. Zhang Y, Zheng W, Lin H, _et al._ Drug–drug interaction extraction
via hierarchical rnns on sequence and shortest dependency
paths. _Bioinformatics_ 2018; **34** (5):828–35.
25. Jay Pujara, Hui Miao, Lise Getoor, and William Cohen. Knowledge
graph identification. In _International Semantic Web Conference_,
pages 542–57. Springer, 2013.
26. Md Rezaul Karim, Michael Cochez, Joao Bosco Jares, Mamtaz Uddin, Oya Beyan, and Stefan Decker. Drug-drug interaction prediction based on knowledge graph embeddings and
convolutional-lstm network. In _Proceedings of the 10th ACM_
_international conference on bioinformatics, computational biology and_
_health informatics_, pages 113–23, 2019.
27. Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier,
and Guillaume Bouchard. Complex embeddings for simple link
prediction. In _International conference on machine learning_, pages

2071–80. PMLR, 2016.

28. Lin X, Quan Z, Wang Z-J, _et al._ Kgnn: knowledge graph neural network for drug-drug interaction prediction. _IJCAI_ 2020; **380** :2739–

45.

29. Lun H, Pan X, Tan Z, _et al._ A fast fuzzy clustering algorithm for
complex networks via a generalized momentum method. _IEEE_
_Trans Fuzzy Syst_ 2021.
30. Lun H, Pan X, Yan H, _et al._ Exploiting higher-order patterns for
community detection in attributed graphs. _Integrated Computer-_
_Aided Engineering_ 2021; **28** (2):207–18.
31. Lun H, Chan KCC, Yuan X, _et al._ A variational bayesian
framework for cluster analysis in a complex network. _IEEE_
_Transactions on Knowledge and Data Engineering_ 2019; **32** (11):

2115–28.

32. Toropov AA, Toropova AP, Mukhamedzhanoval DV, _et al._ Simplified molecular input line entry system (smiles) as an alternative
for constructing quantitative structure-property relationships
(qspr). 2005.
33. Kolumbic Lakos A, Skerk V, Malekovic G, _et al._ A switch ther
apy protocol with intravenous azithromycin and ciprofloxacin



combination for severe, relapsing chronic bacterial prostatitis: a prospective non-comparative pilot study. _J Chemother_
2011; **23** (6):350–3.
34. Greff K, Srivastava RK, Koutník J, _et al._ Lstm: a search space
odyssey. _IEEE transactions on neural networks and learning systems_
2016; **28** (10):2222–32.
35. Huang Z, Xu W, Yu K. Bidirectional lstm-crf models for sequence
tagging _arXiv preprint arXiv:1508.01991_ . 2015.
36. William L Hamilton, Rex Ying, and Jure Leskovec. Inductive
representation learning on large graphs. In _Proceedings of the 31st_
_International Conference on Neural Information Processing Systems_,

pages 1025–35, 2017.
37. Eugene Vorontsov, Chiheb Trabelsi, Samuel Kadoury, and Chris
Pal. On orthogonality and learning recurrent networks with
long term dependencies. In _International Conference on Machine_
_Learning_, pages 3570–8. PMLR, 2017.
38. Xiang Wang, Xiangnan He, Yixin Cao, Meng Liu, and Tat-Seng
Chua. Kgat: Knowledge graph attention network for recommendation. In _Proceedings of the 25th ACM SIGKDD International_
_Conference on Knowledge Discovery & Data Mining_, pages 950–8,

2019.

39. Zhirui Liao, Ronghui You, Xiaodi Huang, Xiaojun Yao, Tao
Huang, and Shanfeng Zhu. Deepdock: Enhancing ligandprotein interaction prediction by a combination of ligand
and structure information. In _2019 IEEE International Confer-_
_ence on Bioinformatics and Biomedicine (BIBM)_, pages 311–7. IEEE,

2019.

40. Han J, Moraga C. The influence of the sigmoid function parameters on the speed of backpropagation learning. In: _Inter-_
_national workshop on artificial neural networks_ . Springer, 1995,

195–201.

41. Daqi G, Yan J. Classification methodologies of multilayer perceptrons with sigmoid activation functions. _Pattern Recognition_
2005; **38** (10):1469–82.
42. Sharma R, Shrivastava S, Singh SK, _et_ _al._ Aniamppred:
artificial intelligence guided discovery of novel antimicrobial peptides in animal kingdom. _Brief Bioinform_ 2021; **22** (6):

bbab242.

43. Kanehisa M, Goto S. Kegg: Kyoto encyclopedia of genes and
genomes. _Nucleic Acids Res_ 2000; **28** (1):27–30.
44. Weihua Hu, Fey M, Zitnik M, Dong Y, Hongyu Ren, Bowen Liu,
Michele Catasta, and Jure Leskovec. Open graph benchmark:
datasets for machine learning on graphs. _Advances in neural_
_information processing systems_ 2020; **33** :22118–22133.
45. Wishart DS, Feunang YD, Guo AC, _et al._ Drugbank 5.0: a major
update to the drugbank database for 2018. _Nucleic Acids Res_
2018; **46** (D1):D1074–82.
46. An Q, Liang Y. A heterogeneous network embedding framework
for predicting similarity-based drug-target interactions. _Brief_
_Bioinform_ 2021; **22** (6):bbab275.
47. Ke G, Meng Q, Finley T, _et al._ Lightgbm: a highly efficient gradient
boosting decision tree. _Advances in neural information processing_
_systems_ 2017; **30** :3146–54.
48. Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, and
Qiaozhu Mei. Line: Large-scale information network embedding.
In _Proceedings of the 24th international conference on world wide web_,

pages 1067–77, 2015.
49. Daixin Wang, Peng Cui, and Wenwu Zhu. Structural deep
network embedding. In _Proceedings of the 22nd ACM SIGKDD inter-_
_national conference on Knowledge discovery and data mining_, pages

1225–34, 2016.


50. Wu L, Wang D, Song K, _et al._ Dual-view hypergraph neural
networks for attributed graph learning. _Knowledge-Based Systems,_
_page_ 2021; **107185** .
51. Peng J, Wang Y, Guan J, _et al._ An end-to-end heterogeneous
graph representation learning-based framework for drug–target
interaction prediction. _Brief Bioinform_ 2021.
52. Zhao T, Yang H, Valsdottir LR, _et al._ Identifying drug–target
interactions based on graph convolutional network and deep
neural network. _Brief Bioinform_ 2021; **22** (2):2141–50.
53. Hongyang Gao, Zhengyang Wang, and Shuiwang Ji. Large-scale
learnable graph convolutional networks. In _Proceedings of the 24th_
_ACM SIGKDD International Conference on Knowledge Discovery &_
_Data Mining_, pages 1416–24, 2018.
54. Bordes A, Usunier N, Garcia-Duran A, _et al._ Translating embeddings for modeling multi-relational data. _Advances in neural infor-_
_mation processing systems_ 2013; **26** .
55. Wang Q, Mao Z, Wang B, _et al._ Knowledge graph embedding:
a survey of approaches and applications. _IEEE Transactions on_
_Knowledge and Data Engineering_ 2017; **29** (12):2724–43.
56. Martín Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen,
Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat,
Geoffrey Irving, Michael Isard, _et al._ Tensorflow: A system for
large-scale machine learning. In _12th USENIX symposium on oper-_
_ating systems design and implementation (OSDI 16)_, pages 265–83,

2016.

57. Xie Y, Gong M, Wang S, _et al._ Sim2vec: node similarity preserving
network embedding. _Inform Sci_ 2019; **495** :37–51.
58. Xiao Wang, Peng Cui, Jing Wang, Jian Pei, Wenwu Zhu, and
Shiqiang Yang. Community preserving network embedding. In
_Thirty-first AAAI conference on artificial intelligence_, 2017.
59. Park Y, Marcotte EM. Flaws in evaluation schemes for
pair-input computational predictions. _Nat Methods_ 2012; **9** (12):

1134–6.

60. Guney E. Revisiting cross-validation of drug similarity based
classifiers using paired data. _Genomics and Computational Biology_
2018; **4** (1):e100047–7.



_Attention-based Knowledge Graph Representation Learning_ | 13


61. Celebi R, Uyar H, Yasar E, _et al._ Evaluation of knowledge graph
embedding approaches for drug-drug interaction prediction in
realistic settings. _BMC bioinformatics_ 2019; **20** (1):1–14.
62. Kipf TN, Welling M. Semi-supervised classification with
graph convolutional networks _arXiv preprint arXiv:1609.02907_ .

2016.

63. Tiantian He L, Bai, and Yew-Soon Ong. Vicinal vertex allocation
for matrix factorization in networks. _IEEE Transactions on Cyber-_

_netics_ 2021.

64. He T, Liu Y, Ko TH, _et al._ Contextual correlation preserving multiview featured graph clustering. _IEEE transactions on cybernetics_
2019; **50** (10):4318–31.
65. Michael Schlichtkrull, Thomas N Kipf, Peter Bloem, Rianne
Van Den Berg, Ivan Titov, and Max Welling. Modeling relational
data with graph convolutional networks. In _European semantic_
_web conference_, pages 593–607. Springer, 2018.
66. Xiao-Rui S, Lun H, You Z-H, _et al._ A deep learning method for
repurposing antiviral drugs against new viruses via multi-view
nonnegative matrix factorization and its application to sars-cov2. _Brief Bioinform_ 2021.
67. Guangtao Wang, Rex Ying, Jing Huang, and Jure Leskovec.
Multi-hop attention graph neural networks. In _Proceedings of the_
_Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-_

_21_, pages 3089–96, 2021.
68. He T, Ong Y, Lu B. Learning conjoint attentions for graph neural
nets. _Advances in Neural Information Processing Systems_ 2021; **34** .
69. Lun H, Yang S, Luo X, _et al._ A distributed framework for
large-scale protein-protein interaction data analysis and prediction using mapreduce. _IEEE/CAA Journal of Automatica Sinica_
2021; **9** (1):160–72.
70. Lun H, Wang X, Huang Y-A, _et al._ A survey on computational
models for predicting protein–protein interactions. _Brief Bioin-_
_form_ 2021.
71. Zhao B-W, Lun H, You Z-H, _et al._ Hingrl: predicting drug–disease
associations with graph representation learning on heterogeneous information networks. _Brief Bioinform_ 2021.


