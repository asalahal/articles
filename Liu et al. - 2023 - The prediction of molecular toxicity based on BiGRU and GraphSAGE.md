[Computers in Biology and Medicine 153 (2023) 106524](https://doi.org/10.1016/j.compbiomed.2022.106524)


Contents lists available at ScienceDirect

# Computers in Biology and Medicine


[journal homepage: www.elsevier.com/locate/compbiomed](https://www.elsevier.com/locate/compbiomed)

## The prediction of molecular toxicity based on BiGRU and GraphSAGE


Jianping Liu [a], Xiujuan Lei [a] [,] [*], Yuchen Zhang [a], Yi Pan [b ]


a _School of Computer Science, Shaanxi Normal University, Xi’an, 710119, China_
b _Faculty of Computer Science and Control Engineering, Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences, Shenzhen, 518055, China_



A R T I C L E I N F O


_Keywords:_
Drug discovery
Toxicity properties prediction

SMILES

Graph

Tox21


**1. Introduction**



A B S T R A C T


The prediction of molecules toxicity properties plays an crucial role in the realm of the drug discovery, since it
can swiftly screen out the expected drug moleculars. The conventional method for predicting toxicity is to use
some in vivo or in vitro biological experiments in the laboratory, which can easily pose a threat significant time
and financial waste and even ethical issues. Therefore, using computational approaches to predict molecular
toxicity has become a common strategy in modern drug discovery. In this article, we propose a novel model
named MTBG, which primarily makes use of both SMILES (Simplified molecular input line entry system) strings
and graph structures of molecules to extract drug molecular feature in the field of drug molecular toxicity
prediction. To verify the performance of the MTBG model, we opt the Tox21 dataset and several widely used
baseline models. Experimental results demonstrate that our model can perform better than these baseline models.



The prediction of molecular properties is one of the most critical
tasks in drug discovery. Accurately predicting the properties of drug
molecules enables rapid screening of drug candidates, saving a lot of
time and money. During the drug candidate screening stage, pharma­
cokinetic properties (ADMET) are widely concerned [1]. ADMET is a
comprehensive study of the five properties of drug absorption, distri­
bution, metabolism, excretion and toxicity [2]. The ADMET property
evaluation method in the early stage of drug discovery can effectively
solve the problem of species differences, significantly improve the suc­
cess rate of drug discovery, and reduce the cost of drug discovery. It
takes more than 10 years and $200 million to bring an FDA drug to
market [3,4]. Drug safety is a main reason for such high costs, ac­
counting for 96% of drug failures [5]. Drug toxicity and side effects are a
major practiced problem in the later stages of drug discovery [6–9].
Therefore, the prediction of drug molecular toxicity is of great signifi­
cance in the drug discovery stage, and it should be implemented as soon
as possible to avoid high-cost consumption.
Traditionally, the study of drug toxicity is often carried out in the
laboratory by some in vivo or in vitro biological experiments [10].
Although these biological experiments are very reliable, these tech­
niques are inefficient and expensive, and sometimes even use some
animals, causing some ethical problems. Accordingly the Quantitative
Structure-Property/Activity Relationship (QSPR/QSAR) [11,12]


 - Corresponding author.
_E-mail address:_ [xjlei@snnu.edu.cn (X. Lei).](mailto:xjlei@snnu.edu.cn)



method has gradually replaced biological experiments in the field of
drug toxicity research. QSPR/QSAR mainly uses statistical methods and
molecular structure parameters to study the relationship between the
structure of compounds and various physical and chemical properties of
molecules and biological activities. In recent years, Heuristics Method
(HM) [13], Multivarate Linear Regression (MLR) [14], Artificial Neural
Networks (ANN) [15], Support Vector Machines (SVM) [16], Projection
Pursuit Regression (PPR) [17] and other methods had been used to build
QSPR/QSAR models. QSPR/QSAR models rely heavily on molecular
characterization in the field of molecular property prediction, so mo­
lecular expression is widely used in the realm of molecular toxicity
prediction [18].
Traditional molecular characterization methods rely on experts to
handcraft a set of rules to encode relevant structural information or
physicochemical properties of molecules into fixed-length vectors. Mo­
lecular fingerprints [19,20] and molecular descriptors [21] are two
typical expressions of molecular features. Thereinto, molecular finger­
print, an abstract expression of a molecule, converts the molecular into a
series of bit vectors, which can provide certain help for the prediction of
molecular properties. However, due to the sparseness of the encoding
itself, it is difficult to obtain molecular-specific features in predicting
molecular toxicity. Molecular descriptors are obtained by researchers
through professional observation or manual extraction. It is a measure of
molecular properties in a certain aspect, which can be either the physical
and chemical properties of the molecule, or a numerical index deduced



[https://doi.org/10.1016/j.compbiomed.2022.106524](https://doi.org/10.1016/j.compbiomed.2022.106524)
Received 21 November 2022; Received in revised form 10 December 2022; Accepted 31 December 2022

Available online 3 January 2023
0010-4825/© 2022 Published by Elsevier Ltd.


_J. Liu et al._ _Computers in Biology and Medicine 153 (2023) 106524_



by various algorithms based on the molecular structure. Molecular de­
scriptors can reduce properties irrelevant to property prediction to a
certain extent, but in the process of acquisition, due to manual methods,
it is prone to bias. In general, molecular fingerprints and molecular
descriptors tend to produce some unnecessary errors in molecular
toxicity prediction.
So far, deep learning has rapid development, causing extensive
exertion not only in fields like natural language processing, computer
vision, and artificial intelligence,but also in various other fields [22–26].
With the continuous development of deep learning, molecular repre­
sentations have also appeared in some expressions that are different
from molecular fingerprints and molecular descriptors. For example, the
one-dimensional sequence-based Simplified Molecular Input Line Entry
Specification (SMILES) expression method [27] and the
two-dimensional molecular graph-based representation method [28,29]
are widely used. It is very important to predict the properties of mo­
leculars or the toxicity of moleculars by using different representations
of molecules through deep learning. Some researchers have also paid
attention to this problem.
BiGRU neural network was often used in the task of text sentiment
classification [30–32]. Lin et al. [33] regarded the SMILES form of a
molecule as a sentence in a text, and used the BiGRU neural network to
propose a novel molecular representation learning framework to predict
the properties of molecules. Peng et al. [34] leveraged the context
structure of SMILES strings and the biochemical properties of the mol­
ecules themselves from another perspective and used deep learning to
predict the toxicity of drug molecules. Zhang et al. [35] proposed a
self-supervised learning method for molecular-related property predic­
tion using the local information transfer mechanism of graph neural
networks. Zhang et al. [36] used the graph neural network GraphSage to
conduct related research on drugs in drug repositioning, and provided a
new idea of graph neural network for molecular property prediction.
Guo et al. [37] also provided a new method in the field of molecular
characterization by means of using the recombination fusion of SMILES
strings and molecular graph structures.
This article seeks to combine the benefits of the above techniques to
propose a new methodology to predict molecular toxicity in the field of
drug development, particularly because of the outstanding accom­
plishments listed above. This innovative method mainly uses the mo­
lecular graph representation and the SMILES representation to predict
the drug molecular toxicity, which can not only use the context infor­
mation of the molecular SMILES string, but also use the structural in­
formation of the molecule graphs. The SMILES strings are firstly one-hot

[38] encoded when utilizing the SMILES strings of the molecular. Then
the one-hot encoding is transferred into an embedding matrix by uti­
lizing the SMILES string. The obtained embedding vector is used as the
input of BiGRU to train it, and finally the context feature vector _b_ _n_ of the
SMILES string is obtained through the pooling layer. On the other hand,
the graph neural network GraphSAGE [39] is mainly used when utilizing
the molecular graph structure. The neighbor vertices of each vertex are
sampled, and then the information contained in the neighbor vertices is
aggregated according to the aggregation function to obtain the molec­
ular graph structure feature vector _d_ _g_ . Finally, the global feature vector _y_
of the molecule is acquired through a fusion layer to perform the task of
drug molecule toxicity prediction. In this article, the main contributions
are as follows.


(1) We propose a molecular toxicity prediction model named MTBG
using molecular SMILES strings and molecular graph structures.
BiGRU and GraphSAGE are used to obtain the information of
SMILES strings and molecular graphs, respectively, and the
binding layer is used to integrate molecular feature information
to predict the toxicity of drug molecules.
(2) Extensive experiments are conducted on Tox21 dataset to
demonstrate the performance of the MTBG model. MTBG model



**Table 1**

Molecules and SMILES string representation.


Molecular SMILES


**CH3CH3** CC

**CH3COOH** CC(=O)O

C6H6 c **1ccccc1**

**C3H7NO2** N[C@H](C)C(=O)O


**Table 2**

Molecules and graph representation.


Molecular Graph


**CH3CH3**

**CH3COOH**


C6H6


**C3H7NO2**


achieved superior performance compared to widely used deep
learning models.


The rest of this article is organized as follows: Section 2 presents
molecular representation. Section 3 details our method. Section 4 ana­
lyzes the experimental results and analysis. Section 5 makes a conclusion
of this article.


**2. Molecular representation**


In recent years, the SMILES form of molecules has been widely used
for the prediction of relevant properties of molecules [40]. SMILES
strings are commonly used to represent and store molecular data in­
formation, taking the form of single-line text composed of molecular
symbols. SMILES is a one-dimensional representation of sequence con­
sisting of letters and numbers called ASCII. Table 1 shows some exam­
ples of simple chemical molecules and the SMILES string representation.
Compared with other representation structures, smiles can conserve a
significant amount of storage space as a language structure rather than a
computer data one.
With the development of molecular characterization, molecular
graphs are gradually applied in the study of the properties of drug
molecules. Molecular graph representation is the representation of
molecules in the form of a graph, which is a two-dimensional repre­
sentation. Table 2 shows some examples of simple chemical molecules
and molecules graph representation. In the representation of molecular
graphs, the connectivity relationship between atoms is represented by
the graph G=( _V_, _E_ ), where V = { _V_ _i_ } represents the set of atomic feature
vectors, _V_ _i_ represents the _i-th_ atomic feature vector, E = { _E_ _ij_ } represents
the set of edge eigenvectors between atoms, and _E_ _ij_ represents the edge
eigenvectors between the _i-th_ atom and the _j-th_ atom.


**Fig. 1.** Representation of SMILES strings and graphs.



2


_J. Liu et al._ _Computers in Biology and Medicine 153 (2023) 106524_


**Fig. 2.** The framework of the MTBG.



In general, SMILES strings and molecular maps of the same molecule
can be converted to each other using some chemistry toolkits. Even if the
molecular structure is the same, it is often represented by various
SMILES strings on account of the differences in the rules of some
chemical toolkits, as shown in Fig. 1. At the same time, some studies
believe that SMILES often cannot capture the spatial connectivity of
atoms in molecules, so the spatial structure information of some mole­
cules is often missing in the prediction of molecular toxicity, resulting in
inaccurate predictions. Molecular graph representations are increas­
ingly widely used for the prediction of molecular properties [41]. Jiang
and Li et al. [28,42]selected molecular graph representations in drug
discovery to predict the relevant properties of molecules and achieved
good results, further demonstrating the importance of molecular graphs
in molecular property prediction.



**3. Methods**


The MTBG model mainly utilized the context feature information of
the SMILES string of the molecule and the feature information of the
molecular graph structure. We present the overall framework of the
MTBG model as depicted in Fig. 2. The whole MTBG model is mainly
composed of three parts: extracting context feature information from
SMILES includes Fig. 2(a) and (b), extracting molecular graph structure
feature information from graph includes Fig. 2(c) and (d) and prediction
part includes Fig. 2(e) and (f).
Briefly, on the one hand, the molecular data in the SMILES string
format was converted into a set of sample vectors, and then these sample
vectors were constructed into atomic matrices as the input of the BiGRU
neural network according to some rules. After that, the BiGRU network
was used to train the input matrix to obtain the context feature vector _b_ _n_



3


_J. Liu et al._ _Computers in Biology and Medicine 153 (2023) 106524_



of the molecule. On the other hand, RDKit [43] served to convert the
SMILES string into a molecular graph structure, and then GraphSAGE
was intended for train the molecular graph structure feature vector _d_ _g_ .
Finally, the feature vectors _b_ _n_ and _d_ _g_ was integrated into an overall
molecular feature vector _y_ _m_ by a fusion layer, which was then processed
by a classifier for the prediction task of drug molecular toxicity.


_3.1. Features extraction from SMILES strings_


This part mainly used Word2Vec [44] and BiGRU to extract feature
information from SMILES strings.


_3.1.1. Feature transformation by Word2Vec_
In natural language processing (NLP), the sentence was divided into
multiple words so as to process a specific sentence. In our research, in
order to capture the relevant feature relationship of the SMILES string
context, we regarded the SMILES string as a molecular process in NLP,
and treated each symbol or letter in the SMILES string as a word in NLP.
We treated a SMILES string "O=(NC)OC – [–] CC" as a sentence _S_ _drug_ . Firstly,
we separated each SMILES symbol or letter in several, and converted
each individual to one-hot encoding. The one-hot encoding was then
turned into a specific embedding layer by using Word2Vec as shown in
Fig. 2(a).So the SMILES string _S_ _drug_ = { _s_ _1_, _s_ _2_, …, _s_ _n_ } that _n_ is the length of
the string was converted into one-hot encoding. Therefore, we could get
the feature vector _a_ _v_ of each symbol or letter through hidden layer by
calculating Equation (1). We got the representation of each symbol or
letter according to Equation (2).


_a_ _v_ = _W_ _f_ _1_ _s_ _[T]_ _n_ (1)


_x_ _n_ = _W_ _f_ _2_ _a_ _v_ (2)


Here _W_ _f1_ and _W_ _f2_ ∈ R [n][×][v ] were the randomly weight matrix. In the
follow-up phase of the study, we used a recurrent neural network long
short-term memory [45] to integrate the vector of each symbol or letter
together to obtain the vector of embedded moleculars _X_ _n_ = { _x_ _1_, _x_ _2_, … …,
_x_ _n_ }.
The main reason for using LSTM is that the length of the drug mol­
ecules we use is not equal, so we can better use LSTM to concatenate
their features. In this paper, we use word2vec to convert words to obtain
the characteristics of each symbol or letter, and use a similar concate­
nation method to connect each symbol or letter to obtain the charac­
teristics of the entire drug molecule.


_3.1.2. Feature extraction by BiGRU_
Since each SMILES string has a different length, we chose GRU [46]
to handle variable-length sequences. Its main idea is to freely capture
widely spaced dependencies in time series data. Let _X_ _n_ = { _x_ _1_, _x_ _2_, … …, _x_ _n_
} be the input sequence. The input vector _x_ _t_ is the input at time _t_ . _h_ _t_ is the
hidden state of each GRU unit at time _t_ . At the same time, the hidden

state _h_ _t-1_ at time _t-1_ and [̃] _h_ is the candidate hidden state at time _t_ . The
update gate and reset gate can be calculated as follows:

⎧ _r_ _t_ = _σ_ ( _x_ _t_ _W_ _xr_ + _h_ _t_ − 1 _W_ _hr_ )

_z_ _t_ = _σ_ ( _x_ _t_ _W_ _xz_ + _h_ _t_ − 1 _W_ _hz_ )

(3)

⎪⎨ _h_ _t_ = (1 − _z_ _t_ ) ⊙ _h_ _t_ − 1 + _z_ _t_ ⊙ _h_ [̃] _t_



function, which can transform the data to a value in the range of 0–1 to
serve as a gating signal. ⊙ is bitwise multiplication; _W_ _xr_, _W_ _hr_, _W_ _xz_, _W_ _hz_,
_W_ _hx_, _W_ _hh_ represent the corresponding weight coefficients.
When dealing with textual information, the unidirectionality of the
GRU makes it impossible to encode from the back to the front, which is
likely to cause information loss between some SMILES string structures.
Since most of the prediction of drug properties requires more attention
to the structural interaction information between atoms, the BiGRU
network is selected in this article. As depicted in Fig. 2(b), BiGRU neural
network is composed of two GRUs with unidirectional and opposite
directions. The state of the BiGRU is jointly determined by the states of
the two GRUs. At each moment, the input will provide two GRUs in
opposite directions at the same time, and the output is jointly deter­
mined by the two unidirectional GRUs. The specific calculation of the
BiGRU hidden state is as follows:
⎧⎪⎪⎨ →=←= _hh_ _tt_ _GRU GRU_ (( _xx_ _tt_ _,, h h_ ← **̅** _tt_ → −− **̅** 11 [)][)] (4)



⎪⎪⎩



→= _h_ _t_ _GRU_ _x_ _t_ _, h_ **̅** _t_ → − 1

←= _h_ _t_ _GRU_ ( _x_ _t_ _, h_ ← _t_ − **̅** 1
(



_b_ _t_ = _W_ _f_ 3 _h_ _t_



←

_h_ →+ _t_ _W_ _f_ 4 _h_ _t_



**̅** → [)]


_t_ − 1


← **̅** [)]


_t_ − 1



1 [)] (4)


←



_r_ _t_ = _σ_ ( _x_ _t_ _W_ _xr_ + _h_ _t_ − 1 _W_ _hr_ )


_z_ _t_ = _σ_ ( _x_ _t_ _W_ _xz_ + _h_ _t_ − 1 _W_ _hz_ )



Here the GRU() function represents a nonlinear transformation of the
input word vector, encoding the word vector into the corresponding
GRU hidden layer state; _W_ _f3_ and _W_ _f4_ ∈ R [n][×][v ] respectively represent the


→

weights corresponding to the forward hidden layer state _h_ _t_ and the


←̅

reverse hidden layer state _h_ _t_ corresponding to the bidirectional GRU at

time _t_ . The feature vector information _b_ _n_ of the SMILES string was ob­
tained by pooling of the vetor _b_ _t_ .


_3.2. Features extraction from graph strings by GraphSAGE_


GraphSAGE was a new learning model proposed by Hamilton [39].
The GraphSAGE method learns that the new node embedding changes
according to the change of the neighbor relationship of the node. Its core
idea was to sample nodes and aggregate nodes as shown in Fig. 2 (c) and
(d). In this article, in the graph structure represented by drug molecu­
lars, each node represents an atom, and each edge represents a chemical
bond between two atoms. In the molecular graph, we denoted the
feature _e_ _v_ of each node _v_ in drug molecular as a batch sample set, and the
feature vector information included node degree, centrality, and the
type of node edge including single bond, double bond and triple bond,
which were the combined information of node features.

Assuming that the entire drug molecular structure had a _K-layer_
batch set, an inside-outside random sampling method was used for
sampling. For the first-order neighbors, the neighbor nodes were
sampled, that was, the second-order neighbors. For the second-order
neighbors, the sampling was the third-order matrix, and so on, until
the _K_ layer sampling was completed. The batch size of each layer is _n_ _k_ . _β_
represented a batch sample set. Sampling their neighbor nodes which
was a section of neighbors. The neighbor nodes of the _k-th_ layer were
defined as ℵ _k_ . ℵ _k_ (v) represented the sampling set of nodes around the
node v of the _k-th_ layer.
The aggregation method is just the opposite of the sampling method.
From the outside to the inside, the aggregation method was used to
aggregate the features of adjacent nodes to iteratively update the rep­
resentation of nodes as shown in Fig. 2(d). Commonly used aggregation
methods include maximum aggregation and average aggregation. After
_K_ iterations, the node _h_ _[k]_ _v_ [represents the ] _[k-th ]_ [layer node as Equation ][(5)][– ]
(8):



(3)



⎪⎩



_h_ _t_ = (1 − _z_ _t_ ) ⊙ _h_ _t_ − 1 + _z_ _t_ ⊙ _h_ [̃] _t_



̃
_h_ _t_ = tanh( _x_ _t_ _W_ _hx_ + ( _r_ _t_ ⊙ _h_ _t_ − 1 ) _W_ _hh_ )



Here the reset gate and update gate at time _t_ are _r_ _t_ and _z_ _t_ . _σ_ is the sigmoid



4


_J. Liu et al._ _Computers in Biology and Medicine 153 (2023) 106524_


**Fig. 3.** Center node aggregation.



MeanPooling aggregator:



(5)
))



_h_ _[k]_ _v_ [←] _[σ]_



(W _[k]_ _uv_ [⋅CONCAT] ( _h_ _[k]_ _v_ [−] _[1]_ _, h_ _[k]_ ℵ( _v_ )



_h_ _[k]_ ℵ( _v_ ) [←MEAN] _[k]_



( { _h_ _[k]_ _u_ [−] _[1]_ _,_ ∀ _u_ ∈ℵ( _v_ )}) (6)



MaxPooling aggregator:



(7)
))



_3.3. Prediction part_


This prediction part was mainly composed of fusion layer and fully
connected layer.
For each medicinal chemical molecule, we used a connection layer to
combine the textual context information feature vector _b_ _n_ captured from
the SMILES string with the structural feature vector _d_ _g_ captured from the
molecular graph to obtain a binding vector _y_ _m_ as in Equation (9):


_y_ _m_ = _W_ f6 _b_ _[T]_ _n_ [+] _[ W]_ [f7] _[d]_ _g_ _[T]_ (9)


Here _W_ _f6_ and _W_ _f7_ ∈ _ℝ_ _[n]_ [×] _[g ]_ were the weight matrix. The obtained combined
feature vector was then input into a fully connected layer.
Input the fused feature vector _y_ _m_ into the full connection, and finally
we used the softmax function to get the probability _p_ _i_ of the positive
class. The fully connected layer was a special feedforward neural
network, which was generally placed at the end of the network. Each
node of each layer was connected to all nodes of the previous layer, as
shown in Fig. 2(f), which was used to classify the previously extracted
features. In this article, we firstly set a threshold (T = 0.5), when the
output value of an output node of the fully connected layer was greater
than the threshold, we considered the class corresponding to the output
node to be a positive sample, otherwise, it was a negative sample.


_3.4. Algorithm_


The algorithm of MTBG is as the **Algorithm 1** .


_3.5. Optimization_


The loss function is often used in neural networks to calculate the

difference between the predicted result and the true value. As the pre­
diction task of molecular toxicity is mainly a typical binary classification
prediction task, we chose the cross-entropy function as the objective
function. In order to prevent the problem of overfitting, we also use the
weight decay method:



_h_ _[k]_ _v_ [←] _[σ]_



(W _[k]_ _uv_ [⋅CONCAT] ( _h_ _[k]_ _v_ [−] _[1]_ _, h_ _[k]_ ℵ( _v_ )



_h_ _[k]_ ℵ( _v_ ) [←MAX] _[k]_



( { _h_ _[k]_ _u_ [−] _[1]_ _,_ ∀ _u_ ∈ℵ( _v_ )}) (8)



Here _k_ was the network layer, and also represented the number of hops
of adjacent points that each vertex could aggregate. And σ () was a nonlinear activation function. _h_ _[k]_ _u_ [, ][∀] _[u]_ [∈ℵ][(] _[v]_ [) represented the embedding of ]
the neighbor node _u_ of the node _v_ at the _k-1_ layer. _h_ _[k]_ ℵ( _v_ ) [represented the ]

feature representation of the fusion of all neighbor nodes of node _v_ in the
_k-th_ layer. _h_ _[k]_ _v_ [, ][∀] _[v ]_ [∈] _[V ]_ [represented the feature representation of node ] _[v ]_ [at ]
layer _k_ ; ℵ( _v_ ) was defined as a uniform take of a fixed size from the set,
_W_ _uv_ _[k]_ [represented the weight coefficient of the edge between node ] _[u ]_ [and ]
node _v_ .
For a specific central node feature _V_ _0_ as shown in Fig. 3, we firstly
passed all its neighbor nodes through a non-linear transformation, and
then obtained the neighbor node feature vector _V_ [′] _neighbor_ of the central
node through a pooling operation. The neighbor node features _V_ [′]

_neighbor_ and the central node features _V_ _0_ are respectively integrated into
the aggregated features _V_ [′′] _0_ of the central node after the operations of
the weights _W_ _f4_ and _W_ _f5_ _._ Finally, all the molecular node features are
connected together order of letters in SMILES to form the feature vector
of the entire molecular.

The feature vector representation of each atom in the molecule was
produced after randomly sampling and aggregator, and the features of
all atoms were then combined to produce the global feature vector _d_ _g_ of
the molecular.



5


_J. Liu et al._ _Computers in Biology and Medicine 153 (2023) 106524_


**Fig. 4.** The number of compounds about each dataset.


6


_J. Liu et al._ _Computers in Biology and Medicine 153 (2023) 106524_



_m_
_Loss_ = ∑

_i_



−[ _y_ _i_ × log( _p_ _i_ ) + (1 − _y_ _i_ ) × _log_ (1 − _p_ _i_ )] (10)



_4.1. Benchmark dataset_


The tox21 dataset was used in this article. The tox21 dataset origi­
nated from a compound prediction competition in 2014 and has since
been widely used as a standard dataset for evaluating toxicity prediction
models. The tox21 dataset includes 7831 compounds and consists of 12
different subtasks, each of which requires prediction of a different type
of toxicity. Seven of the 12 subtasks involved the nuclear receptor(NR)
signaling pathway, and five designed the stress response (SR) signaling
pathway.
Nuclear receptors are transcription factors inside cells that play an
important role in cell growth, development, differentiation and meta­
bolism. Woods et al. [47] have shown that nuclear receptors also feature
prominently in toxicology. The tox21 dataset mainly selects several
nuclear receptors involved in the endocrine system. NR-ER\NR-ER.LBD
and NR-AR\NR-AR.LBD are estrogen and androgen receptors, respec­
tively, and both participate in gene regulation mechanisms and regulate
the transcription of downstream genes.NR-Aromatase is a human
enzyme,a protein that speeds up chemical processes. The other two
nuclear receptors are NR-AhR and NR-PPAR-gamma, which are tran­
scription factors with aryl hydrocarbon receptors as ligands and perox­
isome proliferator-activated receptors that control intracellular
metabolism, respectively.
Toxicity can also cause cellular stress which in term can lead to
apoptosis [48]. SR-ARE is an important mechanism for cells to resist
oxidative stress and controls the transcription of antioxidant enzymes.
SR-HSE is involved in reacting to heat shocks as part of the cell’s internal
repair mechanisms. SR-ATAD5 is involved in the DNA damage response
and regulates the interaction between RAD9A and BCL2d, thereby
inducing DNA damage-induced apoptosis. SR-MMP is a key organelle
that promotes cellular energy conversion and participates in apoptosis.
SR-p53 is a tumor suppressor gene in humans that responds to various
other cellular stresses.


_4.2. Data preprocessing_


Firstly, we divided the data of each task set into positive and negative
samples and unlabeled data as shown in Fig. 4. SR-ARE is the most
balanced dataset, but has the largest number of Nan values. NR-AR is the
most least balanced dataset, but it has the least number of Nan val­
uesThe initial representation of the data is a string of smiles, and we first
remove unlabeled molecules from 12 different tasks. When inputting as
text, we use one-hot encoding, and when inputting as molecular graph,
we used RDKit chemistry software to convert the SMILES string into a
molecular graph. We randomly divided the training set, test set and
validation set five times according to the ratio of 8:1:1.


_4.3. Evaluation metrics_


The Tox21 data challenge and most existing methods only provide
AUC and ACC results. Therefore, we mainly use the AUC and ACC value
which the threshold here is 0.5 as the evaluation index. The area under

the curve (AUC) value is computed based on ROC curve. The ROC curve
is presented by plotting rat ( _TPR_ ) against the false positive rate ( _FPR_ ) at
various threshold settings. _TPR_, _FPR_ and ACC are defined as follows:


_TP_
_TPR_ = (11)
_TP_ + _FN_


_FP_
_FPR_ = (12)
_TN_ + _FP_


_TP_ + _TN_
_ACC_ = (13)
_TP_ + _TN_ + _FP_ + _FN_


Here _TP_ is the true positive, _TN_ is the true negative, _FN_ is false negative,
and _FP_ is the false positive.



here _i_ is the _i-th_ sample vector, _y_ _i_ represents the sample id label, the
positive class means toxicity as 1, the negative class means non-toxicity
as 0, and _p_ _i_ represents the sample _i_ predicts the probability of the posi­
tive class.


**Algorithm 1** . The algorithm of MBTG was used in the prediction of
drug molecular toxicity.


**4. Experimental results and analysis**


In this section, we carry out the datasets, preprocessing, baselines,
evaluation metrics and the experimental results.



7


_J. Liu et al._ _Computers in Biology and Medicine 153 (2023) 106524_


and the message-passing neural network model that can be directly
applied to chemical prediction tasks can learn molecular features
directly from molecular graphs without being affected by graph
isomorphism.
SLGCN: SLGCN is a supervised learning method using graph neural
networks, which solved the problem of predicting the toxicity of drug
molecules.

SSLGCN: SSLGCN is semi-supervised learning using the Mean
Teacher (MT) neural network algorithm. It was raised primarily as a
question for predicting molecular toxicity.


_4.5. Parametric analysis_



**Fig. 5.** The effect of aggregation node on MTBG performance.


_4.4. Baselines_


To establish baseline performance, we test several popular algo­
rithms, namely KNN [49], RF [50,51], decision tree (DT) [52], XGBoost

[53], MPNN [54], SLGCN and SSLGCN [55].
KNN: K-NearesNeighbor is one of the most mature and simplest
machine learning algorithms in theory. When making class decisions, it
is only related to a very small number of adjacent samples.
RF: Random forest is a highly accurate algorithm that produces an
internal unbiased estimate of the panchina error as the forest con­
struction progresses.
DT: A decision tree is an algorithm based on the probability of
occurrence in a known situation. It is often used in classification tasks
and is a type of supervised learning.
XGBoost: XGBoost has the characteristics of high efficiency, flexi­
bility and lightness. It mainly does some optimization tasks and has been
widely used in data mining, system recommendation and other fields.
MPNN: Message-passing neural network is a graph neural network,



For the GraphSAGE neural network, different aggregation methods
will have different effects on the performance of the MTBG method. We
chose the aggregation method of MEANPooling and MAXPooling to
compare the 12 datasets respectively. We find that the MEAN method
significantly outperforms the MAX aggregation method on all 12 data­
sets as shown in Fig. 5, This may explain that Mean aggregation is
approximately equivalent to the convolution propagation operation in
GCN. Therefore, in the end we chose the way of MEANPooling

aggregator.
In addition, the way of samples and aggregates molecules is done in
batches, so the size of the batch is very important for MTBG. If the batch
size is small, it will be difficult to converge, otherwise, if the batch size is
too large, it will take a lot of time. We tested the effect of different
batch_size on the MTBG method as shown in Fig. 6. The performance of
the method is the best when the batch_size is equal to 32.


_4.6. The result of baselines and MTBG_


We present the AUC and ACC values of all baselines and MTBG in
Table 3
in each task. We have repeated the experiments five times and
obtained the average results. The AUC value of baselines scores range
from 0.5332 to 0.8965 in all prediction tasks. And the AUC value of
MTBG scores range from 0.7380 to 0.8965. In addition, the ACC value of
the baselines scores range from 0.7401 to 0.9721. The AUC value of
MTBG scores range from 0.7688 to 0.8965. The ACC value of the MTBG
model scores range from 0.7688 to 0.8965. We can plainly see that the
AUC and ACC values obtained by the MTBG method are much higher
than other baseline methods except NR-AR. The reason that the value of



**Fig. 6.** The effect of Batch_size on MTBG performance.


8


_J. Liu et al._ _Computers in Biology and Medicine 153 (2023) 106524_


ACC of the graph neural network MPNN is slightly higher than that of
MTBG is that we believe that it is caused by the extremely unbalanced
distribution of positive and negative samples in the NR-AR dataset. So
we can think that our proposed MTBG method can have higher
performance.


_4.7. Ablation study_


For the purpose of forecasting chemical toxicity, our suggested
model gathers contextual feature information from SMILES strings as
well as feature information on the molecular graph structure. To
demonstrate that our proposed model can fully utilize this fusion feature
and the superiority of the model, we perform prediction work on 12
toxicity prediction tasks by extracting informative feature information
from the contextual feature information or molecular graph structure of
SMILES strings, respectively. Fig. 7 is a comparison diagram of the
ablation experiment. It can be seen from the figure that the AUC value of
the simultaneous extraction of SMILES string context feature informa­
tion is significantly better than the performance of the other two models.
It can also be concluded that the ACC value of the MTBG method is

higher than the other two, and the NR-PPAR-gamma prediction task has
the highest score. From this, we can argue that our model takes full
advantage of feature fusion and also proves that our model can
adequately predict the nature of molecular toxicity.


_4.8. Case study_


We randomly sampled 10 moleculars from the test set drug mole­
cules predicted to be toxic as Table 4. Six of them have been confirmed
as toxic drug molecules by other scholars, which can prove the validity
of our model.

The CH 2 N 2 -related compounds recorded in the journal published by
National Toxicology program [56] in 2021. In the case of mice, there is a
large amount of carcinogenic data,which proves that this molecule is
difficult to use in the development of new drugs as well. The C 17 H 26 C l NO
compound was confirmed by Li et al. [57] to inhibit Ache in 2021.
Although Ache inhibition has an important therapeutic mechanism for
Alzheimer’s disease, inhibition of Ache activity may also cause cholin­
ergic harm. Two substances, C 18 H 16 F 3 NO 4 and C 13 H 9 C l3 N 2 O, were
analyzed by Wei et al. [58] in the quantitative protein analysis of
mitochondrial toxic substances in the human cardiomyocyte system
further confirming the toxicity of the two substances to mitochondria.
Taking mice as an example, Erdogan et al. [59] studied the therapeutic
effect of C 13 H 12 O 2 -induced mouse vitiligo model, sudsequently con­
firming that C 13 H 12 O 2 has a promoting effect on mouse vitiligo. In
addition, according to the Hazardous Substances Data Bank, the adverse
reactions of C 13 H 12 O 2 to the human body are recorded. In the latest
study, Wisnewski et al. [60] confirmed the use of C 15 H 10 N 2 O 2 to have a
lethal effect on asthma.


**5. Conclusion and discussion**


In this article, we primarily propose an "end-to-end" model named
MTBG to forecast the toxic properties of molecules. In this model we can
capture not only the contextual feature information of molecular
SMILES strings, but also the structural feature information of molecular
graphs. And we performed toxicity prediction for 12 tasks on the tox21
dataset, the performance of our model was better than the seven base­
line models.

In this article, the idea of dual pathway is proposed in the prediction
of drug toxicity, which is a relatively new method and can provide a new
innovative idea for other bioinformatics computing. At the same time,
we can obtain a variety of new features of a molecule, which is more
conducive to the prediction of drug molecular toxicity. However, the
method in this article has some limitations. The method proposed in this
article cannot obtain the 3D structural feature of molecules. The method


9


_J. Liu et al._ _Computers in Biology and Medicine 153 (2023) 106524_


**Fig. 7.** Ablation experimental performance of MTBG.


**Table 4**

Toxicity of drug compounds.


Num Molecular SMILES 2D structure Evidence (PMID)

Formula


**1** CH 2 N 2 N#CN [33819212, National Toxicology program [56]](pmid:33819212)
(2021)


**2** C 17 H 26 ClNO CCc1ccc(C(=O)C(C)CN2CCCCC2)cc1.Cl [33844597,Li et al. [57] (2021)](pmid:33844597)


**3** C 18 H 16 F 3 NO 4 CO/C – [–] C(/C(=O)OC)c1ccccc1COc1cccc(C(F)(F)F)n1 [32733541,Wei](pmid:32733541) _en al._ [58] (2020)


**4** C 13 H 9 C l3 N 2 O O=C(Nc1ccc(Cl)cc1)Nc1ccc(Cl)c(Cl)c1 [32733541,Wei](pmid:32733541) _en al._ [58] (2020)


**5** C 20 H 27 N 5 O 5 S Cc1cc(C(=O)NCCc2ccc(S(=O)(=O)NC(=O)NN3CCCCCC3)cc2)no1 NULL


**6** C 13 H 12 O 2 Oc1ccccc1Cc1ccccc1O [35538739,Erdogan](pmid:35538739) _en al._ [59] (2022)



**7** C 21 H 22 N 4 O 6 S Cc1nc2ccc(CN(C)c3ccc(C(=O)N[C@@H](CCC(=O)O)C(=O)O)s3)cc2c

(=O)[nH]1



NULL



**8** C 15 H 10 N 2 O 2 O=C=Nc1ccc(Cc2ccc(N – [–] C – [–] O)cc2)cc1 [35028957, Wisnewski](pmid:35028957) _en al_ _[. ]_ [60] (2022)


**9** CNNaS N#C[S-].[Na+] NULL


**10** C8H19NO CC(C)N(CCO)C(C)C NULL


10


_J. Liu et al._ _Computers in Biology and Medicine 153 (2023) 106524_



in this article cannot be used in the research of 3D structural molecules

at present. Our subsequent work will pay more attention to the 3D
structural characteristics of molecules.


**Funding**


This work was supported by the National Natural Science Foundation
of China (62272288, 61972451, 61902230, U22A2041), the Shenzhen
Science and Technology Program (No. KQTD20200820113106007).


**Data availability statement**


The data of Tox21 and main code are located at [https://github.](https://github.com/jpliuhaha/jpliuhaha.git)
[com/jpliuhaha/jpliuhaha.git.](https://github.com/jpliuhaha/jpliuhaha.git)


**Declaration of competing interest**


We declare that we have no financial and personal relationships with
other people or organizations that can inappropriately influence our
work, there is no professional or other personal interest of any nature or
kind in any product, service and/or company that could be construed as
influencing the position presented in, or the review of, the manuscript
entitled, “The prediction of molecular toxicity based on BiGRU and

”
GraphSAGE .


**References**


[1] V. Venkatraman, F.P.- Admet, A compendium of fingerprint-based ADMET
[prediction models, J. Cheminf. 13 (2021) 75, https://doi.org/10.1186/s13321-](https://doi.org/10.1186/s13321-021-00557-5)
[021-00557-5.](https://doi.org/10.1186/s13321-021-00557-5)

[2] H. Zhang, L. Zhang, C. Gao, R. Yu, C. Kang, Pharmacophore screening, molecular
docking, ADMET prediction and MD simulations for identification of ALK and MEK
[potential dual inhibitors, J. Mol. Struct. 1245 (2021), 131066, https://doi.org/](https://doi.org/10.1016/j.molstruc.2021.131066)
[10.1016/j.molstruc.2021.131066.](https://doi.org/10.1016/j.molstruc.2021.131066)

[3] Y. Hua, X. Dai, Y. Xu, G. Xing, H. Liu, T. Lu, Y. Chen, Y. Zhang, Drug repositioning:
progress and challenges in drug discovery for various diseases, Eur. J. Med. Chem.
[234 (2022), 114239, https://doi.org/10.1016/j.ejmech.2022.114239.](https://doi.org/10.1016/j.ejmech.2022.114239)

[4] A.B. Deore, J.R. Dhumane, R. Wagh, R. Sonawane, The stages of drug discovery
[and development process, Asian J. Pharmaceut. Res. Dev. 7 (2019) 62–67, https://](https://doi.org/10.22270/ajprd.v7i6.616)
[doi.org/10.22270/ajprd.v7i6.616.](https://doi.org/10.22270/ajprd.v7i6.616)

[5] B. Shaker, S. Ahmad, J. Lee, C. Jung, D. Na, In silico methods and tools for drug
[discovery, Comput. Biol. Med. 137 (2021), 104851, https://doi.org/10.1016/j.](https://doi.org/10.1016/j.compbiomed.2021.104851)
[compbiomed.2021.104851.](https://doi.org/10.1016/j.compbiomed.2021.104851)

[6] O. Silakari, P.K. Singh, ADMET tools: prediction and assessment of chemical
ADMET properties of NCEs, in: O. Silakari, P.K. Singh (Eds.), Concepts and
Experimental Protocols of Modelling and Informatics in Drug Design, Academic
[Press, 2021, pp. 299–320, https://doi.org/10.1016/B978-0-12-820546-4.00014-3.](https://doi.org/10.1016/B978-0-12-820546-4.00014-3)

[7] W. Zhang, H. Zou, L. Luo, Q. Liu, W. Wu, W. Xiao, Predicting potential side effects
of drugs by recommender methods and ensemble learning, Neurocomputing 173
[(2016) 979–987, https://doi.org/10.1016/j.neucom.2015.08.054.](https://doi.org/10.1016/j.neucom.2015.08.054)

[8] W. Zhang, X. Yue, F. Liu, Y. Chen, S. Tu, X. Zhang, A unified frame of predicting
side effects of drugs by using linear neighborhood similarity, BMC Syst. Biol. 11
[(2017) 101, https://doi.org/10.1186/s12918-017-0477-2.](https://doi.org/10.1186/s12918-017-0477-2)

[9] W. Zhang, X. Liu, Y. Chen, W. Wu, W. Wang, X. Li, Feature-derived graph
regularized matrix factorization for predicting drug side effects, Neurocomputing
[287 (2018) 154–162, https://doi.org/10.1016/j.neucom.2018.01.085.](https://doi.org/10.1016/j.neucom.2018.01.085)

[10] V. Kumar, N. Sharma, S.S. Maitra, In vitro and in vivo toxicity assessment of
[nanoparticles, Int. Nano Lett. 7 (2017) 243–256, https://doi.org/10.1007/s40089-](https://doi.org/10.1007/s40089-017-0221-3)
[017-0221-3.](https://doi.org/10.1007/s40089-017-0221-3)

[11] K. Roy, S. Kar, R.N. Das, QSAR/QSPR modeling: introduction, in: A Primer on
QSAR/QSPR Modeling, Springer International Publishing, Cham, 2015, pp. 1–36,
[https://doi.org/10.1007/978-3-319-17281-1_1.](https://doi.org/10.1007/978-3-319-17281-1_1)

[12] A.A. Toropov, A.P. Toropova, QSPR/QSAR: state-of-art, weirdness, the future,
[Molecules 25 (2020) 1292, https://doi.org/10.3390/molecules25061292.](https://doi.org/10.3390/molecules25061292)

[13] J. Shi, F. Luan, H. Zhang, M. Liu, Q. Guo, Z. Hu, B. Fan, QSPR study of fluorescence
wavelengths (λex/λem) based on the heuristic method and radial basis function
[neural networks, QSAR Comb. Sci. 25 (2006) 147–155, https://doi.org/10.1002/](https://doi.org/10.1002/qsar.200510142)
[qsar.200510142.](https://doi.org/10.1002/qsar.200510142)

[14] K. Varmuza, P. Filzmoser, M. Dehmer, Multivariate linear QSPR/QSAR models:
rigorous evaluation of variable selection for PLS, Comput. Struct. Biotechnol. J. 5

[[15] N. Monta(2013), e201302007, nez-Godínez, A.C. Martínez-Olguín, O. Deeb, R. Gardu˜](https://doi.org/10.5936/csbj.201302007) [https://doi.org/10.5936/csbj.201302007. no-Ju˜](https://doi.org/10.5936/csbj.201302007) ´arez,
G. Ramírez-Galicia, QSAR/QSPR as an application of artificial neural networks, in:
H. Cartwright (Ed.), Artificial Neural Networks, Springer New York, New York, NY,
[2015, pp. 319–333, https://doi.org/10.1007/978-1-4939-2239-0_19.](https://doi.org/10.1007/978-1-4939-2239-0_19)

[16] K. Samghani, M. HosseinFatemi, Developing a support vector machine based QSPR
model for prediction of half-life of some herbicides, Ecotoxicol. Environ. Saf. 129
[(2016) 10–15, https://doi.org/10.1016/j.ecoenv.2016.03.002.](https://doi.org/10.1016/j.ecoenv.2016.03.002)




[17] Y. Ren, J. Qin, H. Liu, X. Yao, M. Liu, QSPR study on the melting points of a diverse
set of potential ionic liquids by projection Pursuit regression, QSAR Comb. Sci. 28
[(2009) 1237–1244, https://doi.org/10.1002/qsar.200710073.](https://doi.org/10.1002/qsar.200710073)

[18] A. Sato, T. Miyao, S. Jasial, K. Funatsu, Comparing predictive ability of QSAR/
QSPR models using 2D and 3D molecular representations, J. Comput. Aided Mol.
[Des. 35 (2021) 179–193, https://doi.org/10.1007/s10822-020-00361-7.](https://doi.org/10.1007/s10822-020-00361-7)

[19] D. Rogers, M. Hahn, Extended-connectivity fingerprints, J. Chem. Inf. Model. 50
[(2010) 742–754, https://doi.org/10.1021/ci100050t.](https://doi.org/10.1021/ci100050t)

[[20] R.C. Glem, A. Bender, C.H. Arnby, L. Carlsson, S. Boyer, J. Smith, Circular](http://refhub.elsevier.com/S0010-4825(22)01232-X/sref20)
[fingerprints: flexible molecular descriptors with applications from physical](http://refhub.elsevier.com/S0010-4825(22)01232-X/sref20)
[chemistry to ADME, Idrugs 9 (2006) 199–204.](http://refhub.elsevier.com/S0010-4825(22)01232-X/sref20)

[21] V. Consonni, R. Todeschini, Molecular descriptors, in: T. Puzyn, J. Leszczynski, M.
T. Cronin (Eds.), Recent Advances in QSAR Studies, Springer Netherlands,
[Dordrecht, 2010, pp. 29–102, https://doi.org/10.1007/978-1-4020-9783-6_3.](https://doi.org/10.1007/978-1-4020-9783-6_3)

[22] N. O’Mahony, S. Campbell, A. Carvalho, S. Harapanahalli, G.V. Hernandez,
L. Krpalkova, D. Riordan, J. Walsh, Deep learning vs. Traditional computer vision,
in: K. Arai, S. Kapoor (Eds.), Advances in Computer Vision, Springer International
[Publishing, Cham, 2020, pp. 128–144, https://doi.org/10.1007/978-3-030-17795-](https://doi.org/10.1007/978-3-030-17795-9_10)
[9_10.](https://doi.org/10.1007/978-3-030-17795-9_10)

[23] W. Tang, J. Chen, Z. Wang, H. Xie, H. Hong, Deep learning for predicting toxicity of
chemicals: a mini review, J. Environ. Sci. Health, Part C. 36 (2018) 252–271,
[https://doi.org/10.1080/10590501.2018.1537563.](https://doi.org/10.1080/10590501.2018.1537563)

[24] I.H. Sarker, Deep learning: a comprehensive overview on techniques, taxonomy,
[applications and research directions, Sn Comput. Sci. 2 (2021) 420, https://doi.](https://doi.org/10.1007/s42979-021-00815-1)
[org/10.1007/s42979-021-00815-1.](https://doi.org/10.1007/s42979-021-00815-1)

[25] I. Prapas, B. Derakhshan, A.R. Mahdiraji, V. Markl, Continuous training and
deployment of deep learning models, Datenbank Spektrum 21 (2021) 203–212,
[https://doi.org/10.1007/s13222-021-00386-8.](https://doi.org/10.1007/s13222-021-00386-8)

[26] R. Gupta, D. Srivastava, M. Sahu, S. Tiwari, R.K. Ambasta, P. Kumar, Artificial
intelligence to deep learning: machine intelligence approach for drug discovery,
[Mol. Divers. 25 (2021) 1315–1360, https://doi.org/10.1007/s11030-021-10217-3.](https://doi.org/10.1007/s11030-021-10217-3)

[27] C.-K. Wu, X.-C. Zhang, Z.-J. Yang, A.-P. Lu, T.-J. Hou, D.-S. Cao, Learning to
SMILES: BAN-based strategies to improve latent representation learning from
[molecules, Briefings Bioinf. 22 (2021), bbab327, https://doi.org/10.1093/bib/](https://doi.org/10.1093/bib/bbab327)
[bbab327.](https://doi.org/10.1093/bib/bbab327)

[28] D. Jiang, Z. Wu, C.-Y. Hsieh, G. Chen, B. Liao, Z. Wang, C. Shen, D. Cao, J. Wu,
T. Hou, Could graph neural networks learn better molecular representation for
drug discovery? A comparison study of descriptor-based and graph-based models,
[J. Cheminf. 13 (2021) 12, https://doi.org/10.1186/s13321-020-00479-8.](https://doi.org/10.1186/s13321-020-00479-8)

[29] Y. Kwon, D. Lee, Y.-S. Choi, K. Shin, S. Kang, Compressed graph representation for
[scalable molecular graph generation, J. Cheminf. 12 (2020) 58, https://doi.org/](https://doi.org/10.1186/s13321-020-00463-2)
[10.1186/s13321-020-00463-2.](https://doi.org/10.1186/s13321-020-00463-2)

[30] J. Wu, K. Zheng, J. Sun, Text sentiment classification based on layered attention
network, in: Proceedings of the 2019 3rd High Performance Computing and Cluster
[Technologies Conference, ACM, Guangzhou China, 2019, pp. 162–166, https://](https://doi.org/10.1145/3341069.3342990)
[doi.org/10.1145/3341069.3342990.](https://doi.org/10.1145/3341069.3342990)

[31] T. Jinbao, K. Weiwei, C. Yidan, T. Qiaoxin, S. Chenyuan, L. Long, Text
classification method based on BiGRU-attention and CNN hybrid model, in: 2021
4th International Conference on Artificial Intelligence and Pattern Recognition,
[ACM, Xiamen China, 2021, pp. 614–622, https://doi.org/10.1145/](https://doi.org/10.1145/3488933.3488970)
[3488933.3488970.](https://doi.org/10.1145/3488933.3488970)

[32] A. Kenarang, M. Farahani, M. Manthouri, BiGRU attention capsule neural network
for Persian text classification, J. Ambient Intell. Hum. Comput. 13 (2022)
[3923–3933, https://doi.org/10.1007/s12652-022-03742-y.](https://doi.org/10.1007/s12652-022-03742-y)

[33] X. Lin, Z. Quan, Z.-J. Wang, H. Huang, X. Zeng, A novel molecular representation
with BiGRU neural networks for learning atom, Briefings Bioinf. 21 (2020)
[2099–2111, https://doi.org/10.1093/bib/bbz125.](https://doi.org/10.1093/bib/bbz125)

[34] Y. Peng, Z. Zhang, Q. Jiang, J. Guan, S. Zhou, in: TOP: towards Better Toxicity
Prediction by Deep Molecular Representation Learning, 2019, pp. 318–325,
[https://doi.org/10.1109/BIBM47256.2019.8983340.](https://doi.org/10.1109/BIBM47256.2019.8983340)

[35] X.-C. Zhang, C.-K. Wu, Z.-J. Yang, Z.-X. Wu, J.-C. Yi, C.-Y. Hsieh, T.-J. Hou, D.S. Cao, M.G.- Bert, Leveraging unsupervised atomic representation learning for
[molecular property prediction, Briefings Bioinf. 22 (2021), bbab152, https://doi.](https://doi.org/10.1093/bib/bbab152)
[org/10.1093/bib/bbab152.](https://doi.org/10.1093/bib/bbab152)

[36] Y. Zhang, X. Lei, Y. Pan, F.-X. Wu, Drug repositioning with GraphSAGE and
clustering constraints based on drug and disease networks, Front. Pharmacol. 13
[(2022), 872785, https://doi.org/10.3389/fphar.2022.872785.](https://doi.org/10.3389/fphar.2022.872785)

[37] Z. Guo, W. Yu, C. Zhang, M. Jiang, N.V. Chawla, GraSeq: graph and sequence
fusion learning for molecular property prediction, in: Proceedings of the 29th ACM
International Conference on Information & Knowledge Management, ACM, Virtual
[Event Ireland, 2020, pp. 435–443, https://doi.org/10.1145/3340531.3411981.](https://doi.org/10.1145/3340531.3411981)

[[38] A. Fawcett, Data science in 5 minutes: what is one hot encoding?. https://www.](https://www.educative.io/blog/one-hot-encoding)
[educative.io/blog/one-hot-encoding (Last visited October 23, 2022).](https://www.educative.io/blog/one-hot-encoding)

[39] W.L. Hamilton, R. Ying, J. Leskovec, Inductive representation learning on large
[graphs. https://doi.org/10.48550/ARXIV.1706.02216, 2017.](https://doi.org/10.48550/ARXIV.1706.02216)

[40] S.B. Segota, N. Andelic, I. Lorencin, J. Musulin, D. Stifanic, Z. Car, Preparation of
simplified molecular input line entry system notation datasets for use in
convolutional neural networks, in: 2021 IEEE 21st International Conference on
Bioinformatics and Bioengineering (BIBE), IEEE, Kragujevac, Serbia, 2021, pp. 1–6,
[https://doi.org/10.1109/BIBE52308.2021.9635320.](https://doi.org/10.1109/BIBE52308.2021.9635320)

[41] M. Meng, Z. Wei, Z. Li, M. Jiang, Y. Bian, Property prediction of molecules in graph
convolutional neural network expansion, in: 2019 IEEE 10th International
Conference on Software Engineering and Service Science (ICSESS), IEEE, Beijing,
[China, 2019, pp. 263–266, https://doi.org/10.1109/ICSESS47205.2019.9040723.](https://doi.org/10.1109/ICSESS47205.2019.9040723)

[42] Y. Li, P. Li, X. Yang, C.-Y. Hsieh, S. Zhang, X. Wang, R. Lu, H. Liu, X. Yao,
Introducing block design in graph neural networks for molecular properties



11


_J. Liu et al._ _Computers in Biology and Medicine 153 (2023) 106524_



[prediction, Chem. Eng. J. 414 (2021), 128817, https://doi.org/10.1016/j.](https://doi.org/10.1016/j.cej.2021.128817)
[cej.2021.128817.](https://doi.org/10.1016/j.cej.2021.128817)

[[43] Landrum G, RDKit: open-source cheminformatics,http://www.rdkit.org/(Last](http://www.rdkit.org/)
[visited October 23, 2022.), (n.d.). https://zenodo.org/record/10398#.Yw](https://zenodo.org/record/10398#.Ywl3uXFByUk)
[l3uXFByUk (accessed August 27, 2022).](https://zenodo.org/record/10398#.Ywl3uXFByUk)

[44] T. Mikolov, K. Chen, G. Corrado, J. Dean, Efficient estimation of word
[representations in vector space. https://doi.org/10.48550/arXiv.1301.3781, 2013.](https://doi.org/10.48550/arXiv.1301.3781)

[45] S. Hochreiter, J. Schmidhuber, Long short-term memory, Neural Comput. 9 (1997)
[1735–1780, https://doi.org/10.1162/neco.1997.9.8.1735.](https://doi.org/10.1162/neco.1997.9.8.1735)

[46] J. Chung, C. Gulcehre, K. Cho, Y. Bengio, Empirical evaluation of gated recurrent
[neural networks on sequence modeling. https://doi.org/10.48550/arXiv.1412.](https://doi.org/10.48550/arXiv.1412.3555)
[3555, 2014.](https://doi.org/10.48550/arXiv.1412.3555)

[47] C.G. Woods, J.P. Vanden Heuvel, I. Rusyn, Genomic profiling in nuclear receptor[mediated toxicity, Toxicol. Pathol. 35 (2007) 474–494, https://doi.org/10.1080/](https://doi.org/10.1080/01926230701311351)
[01926230701311351.](https://doi.org/10.1080/01926230701311351)

[48] T. Unterthiner, A. Mayr, G. Klambauer, S. Hochreiter, Toxicity prediction using
[deep learning. https://doi.org/10.48550/arXiv.1503.01445, 2015.](https://doi.org/10.48550/arXiv.1503.01445)

[49] T. Abeywickrama, M.A. Cheema, D. Taniar, k-nearest neighbors on road networks:
a journey in experimentation and in-memory implementation, Proc. VLDB Endow.
[9 (2016) 492–503, https://doi.org/10.14778/2904121.2904125.](https://doi.org/10.14778/2904121.2904125)

[50] Q.-Y. Zhang, J. Aires-de-Sousa, Random forest prediction of mutagenicity from
[empirical physicochemical descriptors, ChemInform 38 (2007), https://doi.org/](https://doi.org/10.1002/chin.200715208)
[10.1002/chin.200715208.](https://doi.org/10.1002/chin.200715208)

[51] V. Svetnik, A. Liaw, C. Tong, J.C. Culberson, R.P. Sheridan, B.P. Feuston, Random
forest: a classification and regression tool for compound classification and QSAR
[modeling, J. Chem. Inf. Comput. Sci. 43 (2003) 1947–1958, https://doi.org/](https://doi.org/10.1021/ci034160g)
[10.1021/ci034160g.](https://doi.org/10.1021/ci034160g)

[52] L. Breiman, J.H. Friedman, R.A. Olshen, C.J. Stone, Classification and Regression
[Trees, first ed., Routledge, 2017 https://doi.org/10.1201/9781315139470.](https://doi.org/10.1201/9781315139470)




[53] T. Chen, C. Guestrin, XGBoost: a scalable tree boosting system, in: Proceedings of
the 22nd ACM SIGKDD International Conference on Knowledge Discovery and
[Data Mining, ACM, San Francisco California USA, 2016, pp. 785–794, https://doi.](https://doi.org/10.1145/2939672.2939785)
[org/10.1145/2939672.2939785.](https://doi.org/10.1145/2939672.2939785)

[54] B. Tang, S.T. Kramer, M. Fang, Y. Qiu, Z. Wu, D. Xu, A self-attention based message
passing neural network for predicting molecular lipophilicity and aqueous
[solubility, J. Cheminf. 12 (2020) 15, https://doi.org/10.1186/s13321-020-0414-z.](https://doi.org/10.1186/s13321-020-0414-z)

[55] J. Chen, Y.-W. Si, C.-W. Un, S.W.I. Siu, Chemical toxicity prediction based on semisupervised learning and graph convolutional neural network, J. Cheminf. 13
[(2021) 93, https://doi.org/10.1186/s13321-021-00570-8.](https://doi.org/10.1186/s13321-021-00570-8)

[56] National Toxicology Program, Toxicity studies of trimethylsilyldiazomethane
administered by nose-only inhalation to Sprague Dawley (Hsd:Sprague Dawley SD)
[rats and B6C3F1/N mice, Toxic Rep. (2021), https://doi.org/10.22427/NTP-TOX-](https://doi.org/10.22427/NTP-TOX-101)
[101. NTP-TOX-101.](https://doi.org/10.22427/NTP-TOX-101)

[57] S. Li, J. Zhao, R. Huang, J. Travers, C. Klumpp-Thomas, W. Yu, A.D. MacKerell,
S. Sakamuru, M. Ooka, F. Xue, N.S. Sipes, J.-H. Hsieh, K. Ryan, A. Simeonov, M.
F. Santillo, M. Xia, Profiling the Tox21 chemical collection for acetylcholinesterase
[inhibition, Environ. Health Perspect. 129 (2021), 47008, https://doi.org/10.1289/](https://doi.org/10.1289/EHP6993)
[EHP6993.](https://doi.org/10.1289/EHP6993)

[58] Z. Wei, J. Zhao, J. Niebler, J.-J. Hao, B.A. Merrick, M. Xia, Quantitative proteomic
profiling of mitochondrial toxicants in a human cardiomyocyte cell line, Front.

[59] A. ErdoGenet. 11 (2020) 719, gan, H.S. Mutlu, S. Solako˘ [https://doi.org/10.3389/fgene.2020.00719glu, Autologously transplanted dermis-derived ˘](https://doi.org/10.3389/fgene.2020.00719) .
cells alleviated monobenzone-induced vitiligo in mouse, Exp. Dermatol. (2022),
[https://doi.org/10.1111/exd.14603.](https://doi.org/10.1111/exd.14603)

[60] A.V. Wisnewski, R. Cooney, M. Hodgson, K. Giese, J. Liu, C.A. Redlich, Severe
asthma and death in a worker using methylene diphenyl diisocyanate MDI asthma
[death, Am. J. Ind. Med. 65 (2022) 166–172, https://doi.org/10.1002/ajim.23323.](https://doi.org/10.1002/ajim.23323)



12


