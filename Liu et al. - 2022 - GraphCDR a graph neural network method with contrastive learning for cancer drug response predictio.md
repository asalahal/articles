_Briefings in Bioinformatics,_ 23(1), 2022, 1–9


**[https://doi.org/10.1093/bib/bbab457](https://doi.org/10.1093/bib/bbab457)**
Problem Solving Protocol

# **GraphCDR: a graph neural network method** **with contrastive learning for cancer drug** **response prediction**

## Xuan Liu, Congzhi Song, Feng Huang, Haitao Fu, Wenjie Xiao and Wen Zhang


Corresponding author. Wen Zhang, College of Informatics, Huazhong Agricultural University, Wuhan 430070, China. E-mail: zhangwen@mail.hzau.edu.cn


Abstract


Predicting the response of a cancer cell line to a therapeutic drug is an important topic in modern oncology that can help
personalized treatment for cancers. Although numerous machine learning methods have been developed for cancer drug
response (CDR) prediction, integrating diverse information about cancer cell lines, drugs and their known responses still
remains a great challenge. In this paper, we propose a graph neural network method with contrastive learning for CDR
prediction. GraphCDR constructs a graph neural network based on multi-omics profiles of cancer cell lines, the chemical
structure of drugs and known cancer cell line-drug responses for CDR prediction, while a contrastive learning task is
presented as a regularizer within a multi-task learning paradigm to enhance the generalization ability. In the computational
experiments, GraphCDR outperforms state-of-the-art methods under different experimental configurations, and the
ablation study reveals the key components of GraphCDR: biological features, known cancer cell line-drug responses and
contrastive learning are important for the high-accuracy CDR prediction. The experimental analyses imply the predictive
power of GraphCDR and its potential value in guiding anti-cancer drug selection.


**Key words:** Cancer drug response prediction; Graph neural network; Contrastive learning; Multi-omics; Drug structure



Introduction


Cancer is one of the most intractable diseases that cause mil
lions of deaths each year over the world. Drug discovery plays a
crucial role in cancer therapy and precision medicine.Traditional
methods of anti-cancer drug discovery are mainly based on _in_
_vivo_ animal experiments and _in vitro_ drug screening, but these
methods are expensive and laborious [1]. Recent advances in
pharmacogenomics have developed several databases, such as
Cancer Cell Line Encyclopedia (CCLE) [2] and Genomics of Drug
Sensitivity in Cancer (GDSC) [3], which provide genome-wide
data about cancer cell lines and drug responses against these cell



lines. These valuable resources enable researchers to investigate
the drug response mechanism in cancer therapy and have been
extensively utilized to establish machine learning methods for
cancer drug response (CDR) prediction.
Over decades, a number of machine learning methods have
been proposed for CDR prediction. The matrix factorizationbased (MF) methods reconstruct the known CDRs by the product
of decomposed factors that are usually constrained by side
information of cancer cell lines and drugs [4–6]. Networkbased methods construct networks with bio-entities (cancer
cell lines, drugs, etc.) and their associations, then formulate the



**Xuan Liu** is a PhD candidate in the College of Informatics at Huazhong Agricultural University.
**Congzhi Song** is a postgraduate in the College of Informatics at Huazhong Agricultural University.
**Feng Huang** is a PhD candidate in the College of Informatics at Huazhong Agricultural University.
**Haitao Fu** Haitao Fu is a PhD candidate in the College of Informatics at Huazhong Agricultural University.
**Wenjie Xiao** is a student in the Information School at University of Washington.
**Wen Zhang** is a professor in the College of Informatics at Huazhong Agricultural University.
**Submitted:** 10 July 2021; **Received (in revised form):** 25 September 2021


© The Author(s) 2021. Published by Oxford University Press. All rights reserved. For Permissions, please email: journals.permissions@oup.com



1


2 _Liu_ et al.


original problem as a link prediction task solved by using random
walk [7, 8], information flow [9, 10] or subset connection [11].
Classification-based methods learn representation vectors from
cancer cell line-drug responses to train classifiers with different
models, such as support vector machine [12], random forest [13,
14], logistic regression [15] and integration model [16]. For the
CDR prediction, deep learning-based methods learn the latent
representation of cancer cell lines and drugs from biochemical
data through different network architectures, such as multilayer perception [17, 18], convolutional neural networks [19, 20]
and recurrent neural networks [21].
Although the previous methods have led to significant
progress in CDR prediction, there is still room for improvement.
In recent years, graph neural networks (GNNs [22]), which apply
deep learning to graphs, have shown good performance in lots
of bioinformatics problems [23–25], and also motivated us to
develop GNN-based CDR prediction models. Since the annotated
cancer cell line-drug responses are scarce [3], the generalization
capability of models are restricted. Self-supervised learning has
emerged as a powerful technique for generating pseudo-label
data from the data itself to relieve the data scarcity. Contrastive
learning is a class of self-supervised methods, which aims to
learn discriminative representations by maximizing agreement/disagreement between the similar/dissimilar instances [26–28].
Moreover, previous works [29, 30] indicate that the biochemical
information (e.g. multi-omics data of cancer cell lines and
SMILES structure of drugs) is helpful for CDR prediction. Thus,
incorporating the biochemical information into the GNN with
contrastive learning can learn more meaningful representation
and boost the performance of CDR prediction.
In this study, we propose a GNN method with contrastive
learning, namely GraphCDR, for CDR prediction. GraphCDR constructs a GNN framework to integrate the biochemical information of cancer cell lines and drugs as well as their known
responses. First, multi-omics representations of cancer cell lines
learned via DNNs and the molecular graph representations of
drugs learned via GNNs are taken as attributes of nodes in a CDR
graph, which treats cancer cell lines, drugs as nodes and their
sensitive responses as edges. Second, we employ a GNN encoder
to learn the latent embedding of cancer cell lines and drugs from
the CDR graph for prediction. Further, a contrastive learning
task is designed to improve discriminative expressiveness of
the GNN encoder and generalize the prediction, which contrasts
the embeddings from the CDR graph and the graph constructed
based on resistant responses. All contributions are summarized
as follows:


 - GraphCDR integrates the biochemical features of cancer
cell lines and drugs as well as known cancer cell line-drug
responses under a GNN framework, which leverages diverse
information to boost the performance of CDR prediction.

  - By taking the domain knowledge into account, a contrastive
learning task is designed and incorporated into GraphCDR
as a regularizer to enhance the generalization ability.

 - In the absence of known responses, GraphCDR can also
utilize the biochemical information of cancer cell lines/
drugs for CDR prediction, which ensures inductive predictive capability when given new cell lines/drugs.


Datasets


**Multi-omics data for cancer cell lines.** CCLE [2] provides genomic,
transcriptomic and epigenomic profiles for more than 1000
cancer cell lines. Following DeepCDR [20], we downloaded



genomic mutation, gene expression and DNA methylation
[by using DeMap portal (https://depmap.org/). Specifically, 34](https://depmap.org/)
673 unique mutation positions within the related genes (697
genes from COSMIC Cancer Gene Census [31]) were collected
as genomic mutation data. The gene expression data were
obtained by the log-normalized TPM value of gene expression.
The DNA methylation data were directly obtained from the
processed Bisulfite sequencing data of promoter 1kb upstream
TSS region. The three omics data (i.e. genomic, transcriptomic
and epigenomic) of a cancer cell line can be represented as 34
673-dimensional, 697-dimensional and 808-dimensional feature
vectors, respectively.
**Molecular graph data for drugs.** PubChem [32] provides validated chemical structure information for 19 million unique
compounds. We downloaded the SMILES strings of all drugs
from PubChem. By leveraging the ConvMolFeaturizer method in
DeepChem library [ 33], the SMILES string of each drug can be
compiled into a molecular graph where the nodes and edges
denote chemical atoms and bonds, respectively. The attribute of
each atom node in a drug is represented as a 75-dimensional
feature vector, described in [34].
**Cancer cell line-drug responses** . In this study, we collected
the IC 50 values (natural log-transformed) from GDSC database
for measuring responses between cancer cell lines and drugs.
We binarized IC 50 values according to the threshold of each drug
provided by the reported maximum screening concentration

[3]. Furthermore, we removed cell lines that lacked any type of
omics data and drugs that shared the same Compound ID (CID)
in PubChem. Finally, we compiled a dataset containing 11 591
sensitive responses and 88 981 resistant responses across 561
cell lines and 222 drugs. Among all the 561 × 222 = 124 542
responses, approximately 19.2% (23 970) of responses (i.e. IC 50
values) were missing/unknown. Beyond that, we downloaded
the activity area value of responses from CCLE database, and
categorized the responses by setting a threshold (sensitive if
the z-score normalized active area _>_ 0.8; otherwise resistant)
according to [35]. We finally created another dataset with 7307
responses (1375 sensitive ones and 5932 resistant ones) between
317 cell lines and 24 drugs.


Methods


**GNN encoder**


GNN is a feed-forward neural network specifically designed
for directly processing graph to generate node representations.
Given a graph _G_ = ( _**X**_, _**A**_ ), where _**X**_, _**A**_ are, respectively, node
attributes and the adjacency matrix, the GNN encoder _Ω_ takes _G_
as input and outputs node embeddings _**H**_ by repeated aggregation over local node neighborhoods. Such neighborhood aggregation can be abstracted as:


_**h**_ ( _ik_ ) = _φ_ [(] _[k]_ [)] [ �] _**h**_ ( _ik_ −1), _f_ [(] _[k]_ [)] [ ��] _**h**_ ( _jk_ −1) : ∀ _j_ ∈ _N_ ( _i_ )��� (1)


Here, _**h**_ ( _ik_ ) = _**H**_ [(] _[k]_ [)] [ _i_, :] is the embedding for node _i_ at the _k_ -th layer
and _**H**_ [(0)] = _**X**_ ; _φ_ [(] _[k]_ [)] is a combination function, _N_ ( _i_ ) denotes a set of
nodes adjacent to _i_ in _**A**_, and _f_ [(] _[k]_ [)] is an aggregation function.


**GraphCDR**


As shown in Figure1, the framework of GraphCDR includes three
following modules: (i) the node representation module extracts


_GraphCDR_ 3


**Figure 1.** Overview of GraphCDR framework. 1 ⃝ Node representation module extracts representations from biochemical features of cancer cell lines/drugs via DNN

layers { _f_ _g_, _f_ _t_, _f_ _e_, _f_ _c_ }/GNN encoder _Ω_ _d_, then takes them as node attributes of the CDR graph _G_ _r_ (cell lines and drugs represent nodes, and their sensitive responses represent
edges). 2 ⃝ CDR prediction module employs a GNN encoder _Ω_ _r_ to learn node-level embeddings _**H**_ _r_ over _G_ _r_ for the CDR prediction task. 3 ⃝ Contrastive learning module
learns corrupted node-level embeddings _**H**_ [�] _r_ from a designed corrupted graph _G_ [�] _r_ (using resistant responses) through _Ω_ _r_, then constructs a contrastive learning task to
enhance the generalization ability of the prediction model by contrasting _**H**_ _r_ and _**H**_ [�] _r_ .



cancer cell line/drug representations from the multi-omics profiles/the molecular graph via DNN layers/GNN encoder; (ii) the
CDR prediction module formulates the known cancer cell linedrug responses as a CDR graph, and takes the preceding representations as the input attributes of nodes (i.e. cancer cell lines
and drugs), and learns the latent embedding of nodes through a
GNN encoder to predict novel CDRs; (iii) the contrastive learning
module presents a contrastive learning task based on the CDR
graph and its corrupted graph, and incorporate the task into
GraphCDR as a regularizer.


_**Node representation module**_


**Cancer cell line representation.** Following previous study [20],
omics-specific neural network layers are designed to integrate
multi-omics information so as to obtain the representation for
each cancer cell line. Here, we resort to the late-integration
fashion in which each neural network layer will first learn a representation of a specific omics feature and then be concatenated
together. Given a cancer cell line with its multi-omics features
(i.e. a genomic feature vector _**c**_ _g_, a transcriptomic feature vector
_**c**_ _t_ and an epigenomic feature vector _**c**_ _e_ ), we can encode it into an
_F_ -dimensional representation by:


_**c**_ = _f_ _c_ �� _f_ _g_ ( _**c**_ _g_ )|| _f_ _t_ ( _**c**_ _t_ )|| _f_ _e_ ( _**c**_ _e_ )�� (2)


where, _**c**_ ∈ R _[F]_ is the representation of a cancer cell line, || is a
vector concatenation operator and { _f_ _g_, _f_ _t_, _f_ _e_, _f_ _c_ } are diverse neural
network layers for feature transformation. Given a set of cancer
cell lines _C_ = { _c_ _i_ } _Ni_ = _C_ 1 [, we finally obtain representations] _**[ C]**_ [ ∈] [R] _[N][C]_ [×] _[F]_

of all cancer cell lines for the follow-up modeling.



**Drug representation.** As described before, a drug can serve as
a molecular graph where nodes represent atoms and edges are
chemical bonds. We denote graph _G_ _d_ = ( _**X**_ _d_, _**A**_ _d_ ) as the molecular
graph for drug _d_ . _**X**_ _d_ ∈ R _[N]_ _[d]_ [×] _[F]_ _[d]_ is a matrix that records the attribute
vectors ( _F_ _d_ = 75) of all atoms and _**A**_ _d_ ∈ R _[N]_ _[d]_ [×] _[N]_ _[d]_ is an adjacency
matrix representing the bonds, where _N_ _d_ is the number of atoms
in the molecular graph of drug _d_ . Here, we apply a GNN encoder
_Ω_ _d_ to capture the latent representation of atom nodes, denoted
by _**H**_ _d_ ∈ R _[N]_ _[d]_ [×] _[F]_, where _**h**_ **[ˆ]** _i_ = _**H**_ _d_ [ _i_, :] is the latent representation of
the node _i_ . As different drug molecular graphs have different
numbers of atom nodes, we apply a global max pooling (GMP)
layer over all nodes to produce a summary representation of the
entire graph as the representation for drug _d_ : _**d**_ ∈ R _[F]_ ← GMP _(_ _**H**_ _d_ _)_ .
Given a set of drugs _D_ = { _d_ _i_ } _Ni_ = _D_ 1 [, we utilize the GNN encoder] _[ Ω]_ _[d]_ [ to]
obtain representations _**D**_ ∈ R _[N][D]_ [×] _[F]_ of all drugs for the subsequent
modeling.


_**CDR prediction module**_


The cancer cell line-drug responses can be formulated as an
undirected heterogeneous graph (i.e. CDR graph) _G_ _r_ = _(_ _V_, _E_ _)_,
where _V_ represents the set of nodes that contain two disjoint
sets of entities (i.e. cancer cell lines _C_ and drugs _D_ ) and | _V_ | =
_N_ _C_ + _N_ _D_ ; _E_ ⊂ _V_ × _V_ denotes the set of edges representing cancer
cell line-drug responses (i.e. sensitive responses). The goal of the
CDR prediction is to learn a mapping function _Θ_ ( _ω_ ) : _E_ → [0, 1]
from edges to scores, where _ω_ is the learnable parameter of _Θ_,
such that we can determine the probability of cancer cell linedrug pairs having sensitive responses. _G_ _r_ can be further indicated
by an adjacency matrix _**A**_ _r_ ∈{1, 0} [|] _[V]_ [|×|] _[V]_ [|] and node attributes
_**X**_ _r_ ∈ R [|] _[V]_ [|×] _[F]_, where _**A**_ _r_ ( _c_, _d_ ) = 1 if cancer cell line _c_ is sensitive to



_**C**_
drug _d_ and _**A**_ _r_ ( _c_, _d_ ) = 0 otherwise, and _**X**_ _r_ = � _**D**_



� (i.e. the cell line


4 _Liu_ et al.


representations _**C**_ and drug representations _**D**_ that are learned
before, are taken as _**X**_ _r_ ).
In this paper, we employ a GNN encoder _Ω_ _r_ to the CDR graph
_G_ _r_, _Ω_ _r_ : ( _**X**_ _r_, _**A**_ _r_ ) → _**H**_ _r_ ∈ R [|] _[V]_ [|×] _[F]_ [′], which learns latent embeddings
of nodes. We denote _**h**_ **[⃗]** _c_ = _**H**_ _r_ [ _c_, :] and _**h**_ **[⃗]** _d_ = _**H**_ _r_ [ _d_, :], respectively, as
final embeddings for cancer cell line node _c_ and drug node _d_ .
For the CDR prediction, we utilize the final embeddings of
cancer cell line node _c_ and drug node _d_ to predict the probability
of their sensitive response ˆ _p_ _cd_ through a scoring function with
the inner product:


_p_ ˆ _cd_ = Sigmoid( _**h**_ **[⃗]** _c_ _**h**_ **[⃗]** _dT_ ) (3)


Then the loss of supervised CDR prediction task can be
formulated as:



_L_ _sup_ = − | _S_ [1] |



� ( _p_ _cd_ log ˆ _p_ _cd_ + �1 − _p_ _cd_ � log �1 −ˆ _p_ _cd_ �) (4)

( _c_, _d_ )∈ _S_



where _S_ is the training set of responses and _p_ _cd_ denotes true label
for the response between nodes _c_ and _d_ .


_**Contrastive learning module**_


Inspired by deep graph infomax (DGI) [26], we present a contrastive learning task that contrasts embeddings from the CDR
graph _G_ _r_ and its corrupted graph _G_ [�] _r_, to enhance the model’s
generalization ability. The procedure of the contrastive learning
is as follows.
We construct a corrupted CDR graph _G_ [�] _r_ = ( _**X**_ _r_, _**A**_ **[�]** _r_ ) based
on resistant responses between cancer cell lines and drugs
(obtained from the training set _S_ ). The reason behind this design
is intuitive: resistant responses naturally imply opposite information against the sensitive responses, and hence it is believed
that DGI can refine the embeddings learned from _G_ _r_ via maximizing dissimilarities between them and their counterparts (the
embeddings learned from _G_ [�] _r_ ), thereby making the prediction
model be more discriminative.

Following the vanilla DGI, we then obtain the corrupted node
embeddings _**H**_ [�] _r_ from the corrupted CDR graph through the same
GNN encoder _Ω_ _r_ : ( _**X**_ _**r**_, _**A**_ **[�]** _**r**_ ) → _**H**_ [�] _r_ ∈ R [|] _[V]_ [|×] _[F]_ [′] . The objective of our
contrastive learning task is formulated as:



1
_L_ _cl_ 1 = − 2| _V_ |



�� _v_ ∈ _V_



log _Γ_ ( _**h**_ **[⃗]** _v_, _**s**_ ) + �

_v_ ∈ _V_ _v_ ∈ _V_



**⃗**

� log(1 − _Γ_ ( _**h**_ [�] _v_, _**s**_ ))


_v_ ∈ _V_



�



(5)



where _**s**_ is the graph-level embedding obtained by a readout
function, _R_ : _**H**_ _r_ ∈ R [|] _[V]_ [|×] _[F]_ [′] → _**s**_ ∈ R _[F]_ [′], and _Γ_ (·, ·) is the contrastive

_T_
discriminator constructed by a simple bilinear function _σ_ ( _**h**_ **[⃗]** _W_ _**s**_ )
that estimates similarities between the node-level embeddings
and the graph-level embedding. _W_ is a learnable scoring matrix
and _σ_ is the logistic sigmoid nonlinearity.
Finally, different from the vanilla DGI, we extend the contrastive learning mechanism from another perspective: maximizing disagreements between node-level embeddings _**H**_ _r_ and
the corrupted graph-level embedding � _**s**_ = _R_ ( _**H**_ [�] _r_ ), which can be
formulated as:



1
_L_ _cl_ 2 = − 2| _V_ |



�� _v_ ∈ _V_



log _Γ_ ( _**h**_ **⃗** [�] _v_,� _**s**_ ) + �

_v_ ∈ _V_ _v_ ∈ _V_



�

� log(1 − _Γ_ ( _**h**_ **[⃗]** _v_, _**s**_ ))


_v_ ∈ _V_



�



(6)



_**Optimization**_


To implement the CDR prediction task and the contrastive learning task simultaneously, we optimize the following objective
function that combines Eq.4, Eq.5 and Eq.6:


_L_ = (1 − _α_ − _β_ ) _L_ sup + _α_ _L_ _cl_ 1 + _β_ _L_ _cl_ 2 (7)


where _α_ and _β_ are hyper-parameters that balance the contributions of different tasks. The pseudo-codes of GraphCDR are
illustrated in Algorithm 1.


Experiments


**Model evaluation**


In this study, we take the GDSC dataset as the main dataset to
evaluate the performance of prediction models. We randomly
select 90% all known cell line-drug responses from the GDSC
dataset to compile the cross-validation set, and use the remaining 10% responses as the independent test set, ensuring no
overlap between these two sets. The sensitive and resistant
responses are taken as positive and negative samples, respectively. Then, we consider the following two experimental configurations.


  - Cross-validation: the 5-fold cross-validation (5-CV) is implemented on the cross-validation set by randomly dividing
responses into five equal parts. The hyper-parameters of
GraphCDR are also set according to the 5-CV results and are
then used for other experiments.

  - Independent test: the prediction models are trained on the
cross-validation set and tested on the independent test set.
Furthermore, we also conduct an independent test on the


CCLE dataset, in which 90% and 10% responses are used as
the train set and the test set, respectively.


We evaluate the experimental results using two metrics:
the area under curve (AUC) and the area under the precisionrecall (AUPR). Besides, accuracy and f1-score are also taken into

account.


**Experimental settings**


In the node representation module for cancer cell line, we represent _f_ _g_, _f_ _t_ and _f_ _c_ by using three different fully connected layers. Considering the mutation positions are distributed linearly
along the chromosome, we take a 1D convolutional layer as _f_ _e_ . In
the node representation module for drug, we employ the graph
convolutional network (GCN) [22] layers as the GNN encoder _Ω_ _d_,
therefore, the representation of atom node _i_ can be updated by:



_j_



⎞

(8)
⎠



**ˆ** ( _k_ _d_ )
_**h**_



⎛



�
⎝ _j_ ∈ _N_ ( _i_ )



_i_ = ReLU [(] _[k]_ _[d]_ [)]



1  - **w** ( _dk_ _d_ )  - _**h**_ **[ˆ]** ( _jk_ _d_ −1)
~~√~~ ~~_q_~~ _i_ ~~_q_~~ _j_



_j_ ∈ _N_ ( _i_ )∪{ _i_ }



where _q_ _i_ = 1+| _N_ _i_ |, **w** _d_ is the weight matrix parameter, and _k_ _d_ = 3.
The representation dimension of a cell line/drug is fixed to 100
( _F_ = 100).
In the CDR prediction module, the GNN encoder _Ω_ _r_ is set to
a _k_ _r_ -layer ( _k_ _r_ = 1) GCN with the PReLU [36] function, and the
embedding of node _v_ ∈ _V_ can be formulated by:



**⃗** ( _k_ _r_ )
_**h**_



⎛



�
⎝ _u_ ∈ _N_ ( _v_ )



_v_ = PReLU [(] _[k]_ _[r]_ [)]



_u_



1  - **w** _r_ ( _k_ _r_ )  - _**h**_ **[⃗]** _u_ ( _k_ _r_ −1)
~~√~~ ~~_q_~~ _v_ ~~_q_~~ _u_



⎞



_GraphCDR_ 5


**Table 1.** Results of ablation study


Method AUC AUPR


GraphCDR **0.8496** **0.5237**
GraphCDR (w/o GO) 0.8430 0.5187
GraphCDR (w/o TO) 0.8437 0.5113
GraphCDR (w/o EO) 0.8466 0.5202
GraphCDR (w/o MG) 0.8317 0.4970
GraphCDR (w/o CL) 0.8428 0.5109
GraphCDR (w/o GNN) 0.7687 0.3719
GraphCDR (w CS) 0.8234 0.4436


lines and targets, and then employs an information flow
algorithm to predict CDRs.

  - **NRL2DRP [12]** integrates cancer cell lines and drugs with
the protein–protein interaction network, and learns the
cancer cell line representations from the network, and then
build support vector machine-based predictors for individual drugs.

  - **tCNNs [19]** is a CNN-based CDR prediction method that uses
SMILES sequences of drugs and genomic mutation data of
cancer cell lines.

  - **RefDNN [18]** is a deep neural network-based CDR prediction
method that utilizes gene expression profiles of cancer cell
lines and molecular structure similarity profiles of drugs.

  - **DeepCDR [20]** is a hybrid GCN-based CDR predictor which
integrates multi-omics profiles of cell lines and chemical
structures of drugs.


**The cross-validation results.** We conduct the 5-CV to evaluate

the performances of GraphCDR and baselines on the crossvalidation set of GDSC, as described in Section Model evaluation.
As shown in Figure2A, GraphCDR outperforms all the baselines,
and exceeds two best baselines: DeepCDR and NRL2DRP by
2.02% and 3.27%, respectively, in AUC scores, and 4.39% and
8.25%, respectively, in AUPR scores. By using CCLE and GDSC
annotation information, we group the cross-validation result of
GraphCDR according to the tissue and target pathway types,
and then calculate the AUC and AUPR scores of GraphCDR
on each group for specific analysis. The results of 24 tissues
and 23 target pathways are shown in Figure2B. The results
show that GraphCDR performs differently on different tissues
and different target pathways. GraphCDR achieves AUC scores
higher than 0.8 on 11 tissues and 16 target pathways, respectively. The cross-validation results demonstrate the superiority
of GraphCDR when performing on CDR prediction task.
**The independent test results.** We conduct the independent test to further assess the performances of prediction
models on the GDSC and CCLE datasets, as described in
Section Model evaluation. Since the hyper-parameter tuning
and model training are irrelevant to the independent test set,
the independent test can better measure generalization ability
of GraphCDR to unseen data. As shown in Figure2C, GraphCDR
achieves a higher AUC and AUPR scores of 0.8426 and 0.5148
than baselines on the GDSC dataset. On the CCLE dataset, the
AUC and AUPR scores of GraphCDR are, respectively, 0.9563 and
0.8877, which are still superior to baselines (results are provided
in Supplementary Table S3). The independent test results show
the high generalization ability of GraphCDR.
The results of other metrics in the cross-validation and inde
pendent test are provided in Supplementary Tables S3–S5.



(9)
⎠



_u_ ∈ _N_ ( _v_ )∪{ _v_ }



The embedded dimension of node _v_ is fixed to 256 ( _F_ [′] = 256). We



design an attentive readout function as _R_ : _**s**_ = [�]



( _k_ _r_ )
_v_ ∈ _V_ _[a]_ _[v]_ [ ·] **[ ⃗]** _**[h]**_ _v_



: _**s**_ = _v_ ∈ _V_ _[a]_ _[v]_ [ ·] _v_ [,] _[ a]_ _[v]_

denotes the attention score of node _v_ :



(0)
exp _f_ _a_ ([ _**h**_ **[⃗]** _v_
�
_a_ _v_ =



(0) ( _k_ _r_ )

_v_ [∥] _**[h]**_ **[⃗]** _v_



( _k_ _r_ )

_v_ []][)]
�



(0) ( _k_ _r_ )

_v_ [∥] _**[h]**_ **[⃗]** _v_



(0)
_v_ ∈ _V_ [exp] _f_ _a_ ([ _**h**_ **[⃗]** _v_
~~�~~



( _k_ _r_ ) (10)

_v_ []][)]
~~�~~



� _v_



where _f_ _a_ is a fully connected layer for mapping embedding
to a real number. In Eq.3, we concatenate cell line/drug node
embeddings of different GCN layers as the final embedding: _**h**_ [⃗] _c_ =



(0)

[ _**h**_ [⃗] _c_



(0) _c_ [||⃗] _**[h]**_ _c_ ( _k_ _r_ )



( _k_ _r_ ) (0)

_c_ ]/ _**h**_ [⃗] _d_ = [ _**h**_ [⃗] _d_



(0) _d_ [||⃗] _**[h]**_ _d_ ( _k_ _r_ )




[ _**h**_ _c_ _**[h]**_ _c_ ]/ _**h**_ _d_ = [ _**h**_ _d_ _**[h]**_ _d_ []][. Furthermore, we employ Adam with a]

learning rate of 0.001 as the optimizer.
In addition to the above empirical settings, several hyperparameters in GraphCDR need to be tuned: the coefficients _α_ and
_β_ in the objective function. Here, we choose both _α_ and _β_ from
{0.00, 0.05, ..., 0.50} for the hyper-parameter optimization under
the cross-validation on the GDSC dataset. According to the result
of hyper-parameter optimization (Supplementary Table S1), we
fix both _α_ and _β_ to 0.3 because they produced the best AUC and
AUPR scores. More details are presented in Supplementary Table

S2.



**Method comparison**


We compare our method with the following baselines.


  - **Random Forest [13]** is a random forest (RF)-based CDR predictor that utilizes oncogene mutational spectrum of cancer
cell lines and fingerprint chemical descriptors of drugs.

  - **HNMDRP [10]** constructs a heterogeneous network from
multiple sub-networks related to drugs, proteins, cancer cell


6 _Liu_ et al.


**Figure 2.** The performance of GraphCDR with different experimental configurations on the GDSC dataset. (A) The performance of GraphCDR and baselines on the
cross-validation. (B) The performance of GraphCDR across 24 tissues (left) and 23 target pathways (right), respectively. (C) The receiver operating characteristic (ROC)
and precision−recall (PR) curve of GraphCDR and baselines on the independent test. (D) The AUC (up) and AUPR (down) scores of GraphCDR and baselines in Inductive

capability study.



**Inductive capability study**


To evaluate the inductive capability of GraphCDR on new cell
lines/drugs, a challenging experiment is conducted on the GDSC
dataset. We randomly split entities (cell lines/drugs/both types)
into five equal parts. In each fold, one part of entities is used
to simulate new ones. The prediction model is trained on the
remaining four parts of entities and their related responses, and
then makes predictions for the new ones.
We consider several baselines with inductive capability (i.e.
DeepCDR, tCNNs, RefDNN and RF), and compare GraphCDR
with them under three inductive configuration experiments
(cell lines, drugs and both types), and the results of all models
are shown in Figure2D. In the inductive capability study for
cell lines, GraphCDR outperforms baselines by achieving a
higher AUC score and AUPR score. In the inductive capability
study for drugs, the performances of all methods drop slightly,
because cell lines may share similar genetic information while
chemical structures of drugs can be diversified, but GraphCDR
also produces the best results. It is worth mentioning that
the inductive capability study for both types (cell lines and
drugs) is a more strict evaluation, and the performances of
baselines decreased significantly, but GraphCDR still achieves
comparable AUC scores and higher AUPR scores when compared
with baselines.

The above studies show that GraphCDR outperforms the
state-of-the-art inductive CDR prediction methods and has good
adaptability in inductive learning.


**Ablation study**


To further investigate the importance of components, multiomic data of cancer cell lines, molecular graph data of drugs, the
GNN framework and the contrastive learning task, we design the
following variants of GraphCDR:




  - **GraphCDR** **without** **genomics** (w/o GO) removes the
genomics omics data.

  - **GraphCDR without transcriptomics** (w/o TO) removes the
transcriptomics omics data.

  - **GraphCDR without epigenomics** (w/o EO) removes the
epigenomics omics data.

  - **GraphCDR without molecular graph representation** (w/o
MG) uses randomized drug representations instead of drug
representations learned from molecular graphs.

  - **GraphCDR without contrastive learning task** (w/o CL)
removes the contrastive learning task.

  - **GraphCDR without the GNN framework** (w/o GNN) removes
the CDR graph and GNN encoder _Ω_ _r_, and directly uses cell
line representations _**C**_ and drug representations _**D**_ through
the inner product for prediction.

  - **GraphCDR with the contrastive strategy from the vanilla**
**DGI** (w CS) adopts the corrupted CDR graph with row-wise
shuffled node attributes.


The ablation study is conducted on the GDSC dataset. We
randomly split all responses into five equal parts to implement
5-CV, and the results are shown in Table 1. When removing omics
data, the AUC scores of GraphCDR variants (w/o GO, w/o TO and
w/o EO) range from 0.8430 to 0.8466, and the AUPR scores range
from 0.5113 to 0.5202, indicating the usefulness of all individual
omics profiles. The results of the variant (w/o MG) demonstrate
that the prediction performance is significantly boosted with the
drug molecular topology, compared to removing it. As expected,
the performances of the variant (w/o CL) show that the contrastive learning task makes a contribution to the prediction.
The experiment on the variant (w/o GNN) demonstrates that
our GNN framework does enhance performance and performs
better than directly using biochemical representations. Besides,
our contrastive strategy is superior to the vanilla DGI (i.e. the


_GraphCDR_ 7


**Table 2.** Top 10 predicted cancer cell lines for two drugs


Drug Rank Cancer cell line PMID


Dasatinib 1 EFM-192A N/A

2 TT2609-C02 N/A

3 HSC-2 N/A

**4** **MCF7** 22306341

5 8505C N/A

6 SNG-M N/A

7 SW-1710 N/A

**8** **786-0** 26984511

**9** **CAL-12T** 22649091

10 LCLC-103H N/A


GSK690693 1 GA-10 N/A

2 HGC-27 N/A

**3** **RCH-ACV** 19064730

4 IGROV1 N/A

5 NCI-H929 N/A

6 RH-18 N/A

7 NCI-H1650 N/A

**8** **JeKo-1** 32120074

**9** **HCC202** 26181325

**10** **MOLT-16** 19064730



node attribute shuffling) according to the results of the variant
(w CS).
In general, GraphCDR leverages the biological information,
the GNN framework and the contrastive learning for the highaccuracy CDR prediction, while the removal of components will
undermine the predictive capacity.


**Case study**


In this section, we conduct case studies to verify whether
GraphCDR could predict novel cancer cell line-drug responses.
We train the GraphCDR model with all known cancer cell linedrug responses in the GDSC dataset, and then predict novel
ones (i.e. unknown responses in the GDSC dataset, described
in Section Datasets). The prediction results are provided in
Supplementary Table S6.
Here, we take two clinically approved drugs Dasatinib and
GSK690693 for analysis. The top 10 cancer cell lines that have
responses with two drugs predicted by GraphCDR are illustrated
in Table 2. Three out of 10 predicted cell lines could be proved
to be sensitive to Dasatinib. For breast cancer cell line MCF7, its
response against Dasatinib was identified as moderately sensitive [37]. Dasatinib showed a significant dose-dependent influence on early non-metastatic 786-0 ccRCC cell lines, increasing
their apoptosis while decreasing proliferation [38]. According
to Sen et al.’s trials [39], they classified the response of lung
cancer cell line CAL-12T to Dasatinib as sensitive. For GSK690693,

four cell lines could be confirmed to be sensitive. GSK690693

restrained the proliferation of cells from both T-cell and B/pre-B-cell origin within the ALL cell panel, with RCH-ACV and
MOLT-16 found to be sensitive to GSK690693 [40]. Liu et al. [41]
discovered that GSK690693 is effective in inhibiting the proliferation of MCL cell line JeKo-1. In Korkola et al.’s study [42], they
measured responses for GSK690693 in breast cancer cell line
HCC202, which was identified to be sensitive according to the
threshold [3].
Therefore, the case studies demonstrate that GraphCDR
could help to find out the novel cancer cell line-drug responses.



Conclusion


In this work, we present a GNN-based method with contrastive
learning namely GraphCDR for CDR prediction. GraphCDR integrates multi-source information of bio-entities under a GNN
frame. Moreover, by leveraging information derived from data
itself, GraphCDR designs a contrastive learning task as a regularizer within a multi-task learning paradigm to boost the prediction performance. GraphCDR outperforms state-of-the-art CDR
prediction models under various experimental configurations.
In future work, we provide two directions for improving CDR
prediction: (i) Resistant responses have an inherently different
semantic meaning as compared to sensitive responses, and it
motivates us to employ the signed GCN model [43] on the signed
graph (i.e. the CDR graph having both sensitive and resistant
responses). (ii) We obtain the node embeddings from the CDR
graph having only cell lines and drugs, and diverse biological
association information (e.g. drug–target interactions, different
similarity networks of bio-entities) has not been well exploited.
Incorporating more associations for CDR prediction deserves
consideration.


**Key Points**


   - GraphCDR integrates the biochemical features of cancer cell lines and drugs as well as known cancer cell
line-drug responses under a graph neural network
framework, which leverages diverse information to
boost the performance of CDR prediction.

   - By taking the domain knowledge into account, a contrastive learning task is designed and incorporated
into GraphCDR as a regularizer to enhance the generalization ability.

   - In the absence of known responses, GraphCDR can
also utilize the biochemical information of cancer

cell lines/drugs for CDR prediction, which ensures
inductive predictive capability when given new cell
lines/drugs.


8 _Liu_ et al.


Supplementary data


[Supplementary data are available online at https://academi](https://academic.oup.com/bib/article-lookup/doi/10.1093/bib/bbab457#supplementary-data)
[c.oup.com/bib.](https://academic.oup.com/bib)


Acknowledgments


We thank anonymous reviewers for valuable suggestions.


Funding


This work was supported by the National Natural Science
Foundation of China (62072206, 61772381); Huazhong
Agricultural University Scientific & Technological Selfinnovation Foundation; Fundamental Research Funds for
the Central Universities (2662021JC008). The funders have
no role in study design, data collection, data analysis, data
interpretation or writing of the manuscript.


Data availability


The data sets and source code can be freely downloaded
from [https://github.com/BioMedicalBigDataMiningLab/Gra](https://github.com/BioMedicalBigDataMiningLab/GraphCDR)
[phCDR.](https://github.com/BioMedicalBigDataMiningLab/GraphCDR)


References


1. Li K, Du Y, Li L, _et al._ Bioinformatics approaches for anticancer drug discovery. _Curr Drug Targets_ 2020; **21** (1):3–17.
2. Barretina J, Caponigro G, Stransky N, _et al._ The cancer cell
line Encyclopedia enables predictive modelling of anticancer
drug sensitivity. _Nature_ 2012; **483** (7391):603–7.
3. Iorio F, Knijnenburg TA, Vis DJ, _et al._ A landscape of pharmacogenomic interactions in cancer. _Cell_ 2016; **166** (3):740–54.
4. Ammad-Ud-Din M, Khan SA, Malani D, _et al._ Drug response
prediction by inferring pathway-response associations with
kernelized Bayesian matrix factorization. _Bioinformatics_
2016; **32** (17):i455–63.
5. Suphavilai C, Bertrand D, Nagarajan N. Predicting cancer
drug response using a recommender system. _Bioinformatics_
2018; **34** (22):3907–14.
6. Wang L, Li X, Zhang L, _et al._ Improved anticancer drug
response prediction in cell lines using matrix factorization
with similarity regularization. _BMC Cancer_ 2017; **17** (1):1–12.
7. Stanfield Z, Co¸skun M, Koyutürk M. Drug response prediction as a link prediction problem. _Sci Rep_ 2017; **7** (1):1–13.
8. Turki T, Wei Z. A link prediction approach to cancer drug
sensitivity prediction. _BMC Syst Biol_ 2017; **11** (5):1–14.
9. Zhang N, Wang H, Fang Y, _et al._ Predicting anticancer drug
responses using a dual-layer integrated cell line-drug network model. _PLoS Comput Biol_ 2015; **11** (9):e1004498.
10. Zhang F, Wang M, Xi J, _et al._ A novel heterogeneous networkbased method for drug response prediction in cancer cell
lines. _Sci Rep_ 2018; **8** (1):1–9.
11. Meybodi FY, Eslahchi C. Predicting anti-cancer drug
response by finding optimal subset of drugs. _Bioinformatics_
[2021; btab466. doi: 10.1093/bioinformatics/btab466.](https://doi.org/10.1093/bioinformatics/btab466)
12. Yang J, Li A, Li Y, _et al._ A novel approach for drug response
prediction in cancer cell lines via network representation
learning. _Bioinformatics_ 2019; **35** (9):1527–35.
13. Lind AP, Anderson PC. Predicting drug activity against
cancer cells by random forest models based on minimal



genomic information and chemical properties. _PLoS One_
2019; **14** (7):e0219774.
14. Su R,Liu X,Wei L, _et al._ Deep-Resp-Forest: a deep forest model
to predict anti-cancer drug response. _Methods_ 2019; **166** :91–

102.

15. Yu L, Zhou D, Gao L, _et al._ Prediction of drug response in
multilayer networks based on fusion of multiomics data.
_Methods_ 2021; **192** :85–92.
16. Gerdes H, Casado P, Dokal A, _et al._ Drug ranking using
machine learning systematically predicts the efficacy of
anti-cancer drugs. _Nat Commun_ 2021; **12** (1):1–15.
17. Li M, Wang Y, Zheng R, _et al._ DeepDSC: a deep learning
method to predict drug sensitivity of cancer cell lines.
_IEEE/ACM Trans Comput Biol Bioinform_ 2021; **18** (2):575–582.
18. Choi J, Park S, Ahn J. RefDNN: a reference drug based neural
network for more accurate prediction of anticancer drug
resistance. _Sci Rep_ 2020; **10** (1):1–11.
19. Liu P, Li H, Li S, _et al._ Improving prediction of phenotypic
drug response on cancer cell lines using deep convolutional
network. _BMC Bioinformatics_ 2019; **20** (1):1–14.
20. Liu Q, Hu Z, Jiang R, _et al._ DeepCDR: a hybrid graph convolutional network for predicting cancer drug response.
_Bioinformatics_ 2020; **36** (Supplement_2):i911–8.
21. Li Q, Huang J, Zhu H, _et al._ Prediction of Cancer Drug Effectiveness Based on Multi-Fusion Deep Learning Model. In:
_Annual Computing and Communication Workshop and Confer-_
_ence (CCWC)_ . Las Vegas, NV, USA: IEEE, 2020:0634–0639. doi:
[10.1109/CCWC47524.2020.9031163](https://doi.org/10.1109/CCWC47524.2020.9031163)

22. Xu K, Hu W, Leskovec J, _et al._ How powerful are graph
neural networks? In: _International Conference on Learning Rep-_
_resentations (ICLR)._ New Orleans, Louisiana, United States:
OpenReview.net, 2019.
23. Li J, Zhang S, Liu T, _et al._ Neural inductive matrix completion with graph convolutional networks for miRNA-disease
association prediction. _Bioinformatics_ 2020; **36** (8):2538–46.
24. Yu Z, Huang F, Zhao X, _et al._ Predicting drug–disease
associations through layer attention graph convolutional
network. _Briefings in Bioinformatics_ 2021; **22** (4):bbaa243. doi:
[10.1093/bib/bbaa243.](https://doi.org/10.1093/bib/bbaa243)

25. Nyamabo AK, Yu H, Shi JY. SSI–DDI: substructure–
substructure interactions for drug–drug interaction
prediction. _Briefings_ _in_ _Bioinformatics_ 2021;bbab133. doi:
[10.1093/bib/bbab133.](https://doi.org/10.1093/bib/bbab133)

26. Velickovic P, Fedus W, Hamilton WL, _et al._ Deep Graph
Infomax. In: _nternational Conference on Learning Representa-_
_tions (ICLR)_ . New Orleans, Louisiana, United States: OpenReview.net, 2019.
27. Chen T, Kornblith S, Norouzi M, _et al._ A simple framework for contrastive learning of visual representations. In:
_International Conference on Machine Learning (ICML)_ . Virtual
Conference: PMLR, 2020; **119** :1597–607.
28. Qiu J, Chen Q, Dong Y, _et al._ Gcc: Graph contrastive coding for
graph neural network pre-training. In: _International Confer-_
_ence on Knowledge Discovery & Data Mining (KDD)_ . Virtual Event
[CA USA: ACM, 2020, 1150–60. doi: 10.1145/3394486.3403168.](https://doi.org/10.1145/3394486.3403168)
29. Kearnes SM, McCloskey K, Berndl M, _et al._ Molecular graph
convolutions: moving beyond fingerprints. _J Comput Aided_
_Mol Des_ 2016; **30** (8):595–608.
30. Sharifi-Noghabi H, Zolotareva O, Collins CC, _et al._ MOLI:
multi-omics late integration with deep neural networks for
drug response prediction. _Bioinformatics_ 2019; **35** (14):i501–9.
31. Sondka Z, Bamford S, Cole CG, _et al._ The COSMIC cancer gene
census: describing genetic dysfunction across all human
cancers. _Nat Rev Cancer_ 2018; **18** (11):696–705.


32. Kim S, Chen J, Cheng T, _et al._ PubChem 2019 update:
improved access to chemical data. _Nucleic_ _Acids_ _Res_
2019; **47** (D1):D1102–9.
33. Duvenaud D, Maclaurin D, Aguilera-Iparraguirre J, _et al._
Convolutional networks on graphs for learning molecular
fingerprints In: _Conference on Neural Information Processing_
_Systems (NeurIPS)_ . Montreal Canada: Curran Associates, Inc.,
2015; **2** :2224–2232.
34. Ramsundar B, Eastman P, Walters P, _et al. Deep learning for the_
_life sciences: applying deep learning to genomics, microscopy, drug_
_discovery, and more_ . Sebastopol, California: O’Reilly Media,
Inc., 2019.
35. Dong Z, Zhang N, Li C, _et al._ Anticancer drug sensitivity prediction in cell lines from baseline gene expression
through recursive feature selection. _BMC Cancer_ 2015; **15** (1):

1–12.

36. He K, Zhang X, Ren S, _et al._ Delving deep into rectifiers:
Surpassing human-level performance on imagenet classification. In: _International conference on computer vision (ICCV)_ .
Santiago, Chile: IEEE, 2015, 1026–34.
37. Park BJ, Whichard ZL, Corey SJ. Dasatinib synergizes with
both cytotoxic and signal transduction inhibitors in heterogeneous breast cancer cell lines–lessons for design of



_GraphCDR_ 9


combination targeted therapy. _Cancer Lett_ 2012; **320** (1):104–

10.

38. Roseweir AK, Qayyum T, Lim Z, _et al._ Nuclear expression of
Lyn, a Src family kinase member, is associated with poor
prognosis in renal cancer patients. _BMC Cancer_ 2016; **16** (1):1–

10.

39. Sen B, Peng S, Tang X, _et al._ Kinase impaired BRAF mutations
confer lung cancer sensitivity to Dasatinib. _Sci Transl Med_
2012; **4** [(136):136ra70. doi: 10.1126/scitranslmed.3003513.](https://doi.org/10.1126/scitranslmed.3003513)
40. Levy DS, Kahana JA, Kumar R. AKT inhibitor, GSK690693,
induces growth inhibition and apoptosis in acute lymphoblastic leukemia cell lines. _Blood, The Journal of the Amer-_
_ican Society of Hematology_ 2009; **113** (8):1723–9.
41. Liu Y, Zhang Z, Ran F, _et al._ Extensive investigation of benzylic N-containing substituents on the pyrrolopyrimidine
skeleton as Akt inhibitors with potent anticancer activity.
_Bioorg Chem_ 2020; **97** :103671.
42. Korkola JE, Collisson EA, Heiser L, _et al._ Decoupling of the PI3K
pathway via mutation necessitates combinatorial treatment
in HER2+ breast cancer. _PLoS One_ 2015; **10** (7):e0133219.
43. Derr T, Ma Y, Tang J. Signed graph convolutional networks.
In: _International Conference on Data Mining (ICDM)_ . IEEE, 2018,
[929–34. doi: 10.1109/ICDM.2018.00113.](https://doi.org/10.1109/ICDM.2018.00113)


