_Bioinformatics_, 2024, **40(3)**, btae135
https://doi.org/10.1093/bioinformatics/btae135
Advance Access Publication Date: 6 March 2024

**Original Paper**

## Structural bioinformatics

# **iNGNN-DTI: prediction of drug–target interaction with** **interpretable nested graph neural network and pretrained** **molecule models**


**Yan Sun** **[1,2,3]** **, Yan Yi Li** **4** **, Carson K. Leung** **2** **, Pingzhao Hu** **1,2,3,4,5,6,7,** �


1 Department of Biochemistry, Western University, London, ON, N6G 2V4, Canada
2 Department of Computer Science, University of Manitoba, Winnipeg, MB, R3T 2N2, Canada
3 Department of Computer Science, Western University, London, ON, N6G 2V4, Canada
4 Division of Biostatistics, University of Toronto, Toronto, ON, M5T 3M7, Canada
5 Department of Oncology, Western University, London, ON, N6G 2V4, Canada
6 Department of Epidemiology and Biostatistics, Western University, London, ON, N6G 2V4, Canada
7 The Children’s Health Research Institute, Lawson Health Research Institute, London, ON, N6A 4V2, Canada

�Corresponding author. Department of Biochemistry, Western University, 1400 Western Road London, Ontario N6G 2V4, Canada. E-mail: phu49@uwo.ca (P.H.)

Associate Editor: Arne Elofsson


**Abstract**


**Motivation:** Drug–target interaction (DTI) prediction aims to identify interactions between drugs and protein targets. Deep learning can auto­
matically learn discriminative features from drug and protein target representations for DTI prediction, but challenges remain, making it an open
question. Existing approaches encode drugs and targets into features using deep learning models, but they often lack explanations for underly­
ing interactions. Moreover, limited labeled DTIs in the chemical space can hinder model generalization.


**Results:** We propose an interpretable nested graph neural network for DTI prediction (iNGNN-DTI) using pre-trained molecule and protein mod­
els. The analysis is conducted on graph data representing drugs and targets by using a specific type of nested graph neural network, in which
the target graphs are created based on 3D structures using Alphafold2. This architecture is highly expressive in capturing substructures of the
graph data. We use a cross-attention module to capture interaction information between the substructures of drugs and targets. To improve
feature representations, we integrate features learned by models that are pre-trained on large unlabeled small molecule and protein datasets,
respectively. We evaluate our model on three benchmark datasets, and it shows a consistent improvement on all baseline models in all data­
sets. We also run an experiment with previously unseen drugs or targets in the test set, and our model outperforms all of the baselines.
Furthermore, the iNGNN-DTI can provide more insights into the interaction by visualizing the weights learned by the cross-attention module.


**Availability and implementation:** [The source code of the algorithm is available at https://github.com/syan1992/iNGNN-DTI.](https://github.com/syan1992/iNGNN-DTI)



**1 Introduction**


Drugs typically refer to small molecules, while targets often
refer to macromolecules such as proteins. A drug may change
the function of a biological target when it binds to the target,
which is known as drug–target interaction (DTI) (Sachdev
and Gupta, 2019). The prediction of DTI plays a key role in
drug discovery. Since it is time-consuming and expensive to
identify the DTI pairs through biological assays, many
computer-assisted methods have been developed (Ou-Yang
_et al.,_ 2012). With the success of deep learning in various
areas, deep learning-based DTI prediction methods have ex­
ploded in recent years (Wen et al., 2017).

The base framework of deep learning methods for DTI
involves encoding molecules and targets separately through two
branches. Subsequently, the output features from these two
branches are concatenated and fed into a classifier constructed
using a fully connected network. Various deep learning models
are utilized for feature representation learning when different
raw representations for drugs and proteins are used (Bronstein



_et al._ 2017, Ozt [€] urk € _et al._ 2018, Huang _et al._ 2021a, 2021b,
Yang et al., 2021). DeepDTA (Ozt [€] urk € _et al._ 2018) uses a threelayer convolutional neural network (CNN) on the simplified
molecular-input line-entry system (SMILES) string of drug mol­
ecules and the amino acid sequence of protein targets. In con­
trast, MolTrans (Huang _et al._ 2021a, 2021b) utilizes the
transformer encoder, a module known for its excellent perfor­
mance in natural language processing (NLP) tasks, on both se­
quence data. However, the sequence data loses the structural
information of the molecule. To address this limitation,
approaches that directly learn representations from the molecu­
lar graph have been developed. The graph, constructed by nodes
representing atoms and edges representing bonds, offers a more
natural representation of molecules. Various graph neural net­
works (GNNs), such as graph convolutional networks (GCN)
and graph attention network (GAT), have been implemented in
DTI tasks (Jiang _et al._ 2020, Nguyen _et al._ 2021). A deeper re­
view of different deep learning approaches for compound–pro­
tein interaction prediction can be found in Jiang _et al._ (2020).



**Received:** 23 July 2023; **Revised:** 31 December 2023; **Editorial Decision:** 14 February 2024; **Accepted:** 5 March 2024
# The Author(s) 2024. Published by Oxford University Press.
This is an Open Access article distributed under the terms of the Creative Commons Attribution License (https://creativecommons.org/licenses/by/4.0/), which
permits unrestricted reuse, distribution, and reproduction in any medium, provided the original work is properly cited.


**2** Sun _et al._



However, most methods focus on encoding drugs with GNNs,
with limited attempts to encode protein targets using GNNs.
The challenge in representing proteins as graphs arises from the
fact that the structures of most proteins are unknown. The re­
cent breakthrough achieved by Alphafold2 (Jumper _et al.,_
2021) in predicting protein structures has eased this challenge,
allowing the incorporation of graph data for proteins in DTI
tasks. Our approach involves generating protein structures with
Alphafold2 and applying GNNs to both drugs and targets.

Interpretability is a challenge to the DTI task. Most methods
learn representations of drugs and proteins with deep learning
models, but lack the interpretation of interactions between the
substructures of the drugs and targets (Schenone _et al.,_ 2013).
ML-DTI adds a mutual learning module between the drug en­
coder and the target encoder (Yang _et al._ 2021). The mutual
learning module calculates attention on each atom of the drug.
MolTrans designs a pairwise interaction module after the en­
coder (Huang _et al._ 2021a, 2021b). It decomposes drugs and
proteins into substructures and calculates the interaction be­
tween every two substructures. DrugBAN develops a novel bi­
linear interaction module to capture the interaction local
structure (Bai _et al.,_ 2023). The methods are all realized using
the protein sequence data. However, the sequence data loses de­
tailed local structural information, which is important to the
binding site. Furthermore, the number of molecules in the DTI
datasets is tiny compared to the scale of the chemical space. The
problem may limit the application of the DTI model to the new
drugs or targets that are unseen in the training sets.

To address the above-mentioned limitations, we propose
to build an interpretable nested graph neural network for the
drug–target interaction prediction (iNGNN-DTI) architec­
ture based on the graph data of both drugs and proteins. We
suggest using a nested graph neural network (NGNN)
(Zhang and Li, 2021) as the fundamental feature extraction
model to enhance the expression of the substructure. This
improves the expression of the substructures because the rep­
resentation of each node in the NGNN is the result of pooling
the k-hop subgraph that surrounds it. After the encoder, we
use a cross-attention free transformer (cross-AFT) (Zhai
_et al._, 2021) module to calculate the interaction between each
pair of nodes from the drugs and targets. We combine the
features learned by two pre-trained models, Chemformer
(Irwin _et al.,_ 2022) for drugs and ESM (Rives _et al.,_ 2021) for
proteins, with the output of the NGNN in order to further
enhance the expression of the features.

The contributions of our work include: (i) the proposed
method utilizes the k-subgraph GNN extractor layer from
the NGNN to encode the graph data. This particular layer
has demonstrated superior performance compared to a stan­
dard GNN layer; (ii) the method proposes to enhance perfor­
mance by integrating the features of targets and drugs that
have been extracted using models pre-trained on other largescale unlabeled datasets; (iii) we propose an interaction
module on the graph neural network for the DTI task, and
the experiment results indicate that it performs better than
the vanilla GNN.


**2 Materials and methods**

2.1 Datasets

We evaluate our method on three popularly used benchmark
datasets: Davis (Davis _et al.,_ 2011), KIBA (He _et al.,_ 2017),
[and BIOSNAP (http://snap.stanford.edu/biodata) (Zitnik](http://snap.stanford.edu/biodata)



We set samples with _pK_ _d_ � 7 as binding samples. For the
KIBA dataset, the threshold is 12.1, and samples with a KIBA
score larger than the threshold are set as positive samples.
BIOSNAP collects high-throughput experiment results from
diverse resources. Only positive pairs are provided by the
BIOSNAP dataset, and we randomly sample unseen drug–
target pairs with the same number of positive pairs as in the
negative samples. The statistics of all datasets are shown
in Table 1.


2.2 Input data representation
Both drugs and targets can be represented in two different
forms: sequence data and graph data. This section outlines
the methods used to construct graph inputs for drugs and tar­
gets using their respective sequence data.


**2.2.1 Drug molecule representation**
The sequence data of the drug molecule is the SMILES string
in which each character represents atoms or bonds of the
molecule. To represent drug molecules as graphs, a common
approach is to convert their chemical notations into graph
representations. This involves mapping atoms to nodes and
chemical bonds to edges. We pass the SMILES string to
RDKit (Landrum, 2016) tool to generate the drug molecule
graph. We initialize the nodes of the drug molecule graph
with the same feature as DGraphDTA (Jiang _et al.,_ 2020).
The features are listed in Table 2 and the total dimension of

the atom feature is 78.


**2.2.2 Target representation**
Each character in the target sequence data corresponds to an
amino acid. The graph representations can be constructed by
considering the structural characteristics of the amino acids
and their interactions. Unlike the drug molecule, the genera­
tion of target molecule structures are complicated. The stateof-the-art method for predicting the structure of the protein
is Alphafold2 (Jumper _et al.,_ 2021). Alphafold2 takes the
amino acid sequence as input and then outputs the threedimensional structure of a protein. As shown in Fig. 1, we ob­
tain the structural information of targets in the Protein Data
Bank (PDB) file format from the output of Alphafold2, and
then we calculate the Euclidean distance between the _C_ a
atoms in amino acids to generate the distance map of a pro­
tein. We generate the binary contact map by setting a thresh­
old on the distance map. In this work, we set the threshold as
8 Å based on the suggestion made by Duarte _et al._ (2010).
Other two alternative thresholds are also considered (see
[Supplementary Table S1). The node feature of the protein](https://academic.oup.com/bioinformatics/article-lookup/doi/10.1093/bioinformatics/btae135#supplementary-data)



_et al.,_ 2018). The Davis and the KIBA datasets are both for
the kinase protein family. The Davis dataset indicates the ac­
tivity of the drug with the equilibrium dissociation constant
_K_ _d_ value. The value of the KIBA dataset is the combination
of dissociation constant _K_ _d_, inhibitory constant _K_ _i_ and the
half maximal inhibitory concentration _IC_ 50 . The two datasets
need preprocessing to generate the binary label, and we fol­
low the analysis done in previous works (He _et al._ 2017,
Huang _et al._ 2021a, 2021b). For the Davis dataset, the _K_ _d_
value is transformed into log space as _pK_ _d_ :



_K_ _d_
_pK_ _d_ ¼ � _log_ 10 _:_ (1)

�1 _e_ 9�


iNGNN-DTI **3**


**Table 1.** Summary of the datasets. [a]


**No. of Drugs** **No. of proteins** **No. of interactions** **No. of positives** **No. of negatives**


Davis 68 442 30 056 1506 28 550

KIBA 2068 229 118 254 22 729 95 525

BIOSNAP 4510 2180 27 428 13 817 13 611 [a]


a Some of the negative pairs were removed during preprocessing the data.


**Table 2.** Atom features in the drug molecule.


**Feature name** **Dim** **Description**


Atom element 44 One-hot encoding of atom element
Degree of the atom 11 One-hot encoding of the degree of atoms
Number of Hydrogens 11 Number of hydrogen bonds
Number of implicit Hydrogens 11 Number of implicit hydrogen bonds
Aromatic 1 The atom is aromatic or not


**Figure 1.** The process of the generation of protein graph with Alphafold2.



graph is the same as Jiang _et al._ (2020). The specific descrip­
tion of the feature is shown in Table 3.

Position Probability Matrix (PPM) is a simplified feature of
the Position-Specific Scoring Matrix (PSSM), which is widely
used to evaluate the similarity between protein sequences
(Cheol Jeong _et al.,_ 2010). The PPM is generated based on
the result of multiple sequence alignment (MSA). MSA aligns
a query protein with thousands of protein sequences and out­
puts an aligned sequence array (Edgar and Batzoglou, 2006).
The PPM calculates the occurrence probability of each



residue on each point. The calculation of the PPM can be rep­
resented as:



where _F_ 2 _N_ _[R]_ [�] _[K ]_ is the frequency matrix, _R_ is the number of
residues, and _K_ is the length of the protein sequence. Each
row corresponds to a type of residue. _F_ _r;k_ represents the



_M_ _r;k_ ¼



_F_ _r;k_ þ _[p]_ 4 (2)
_N_ þ _p_ _[;]_


**4** Sun _et al._



**Table 3.** Protein molecular graph feature initialization.


**Feature name** **Dim** **Description**



Residual symbol 21 One-hot encoding of the residue
PPM 21 Position Probability Matrix
Aliphatic 1 The residue is aliphatic or not
Aromatic 1 The residue is aromatic or not
Polar neutral 1 The residue is polar neutral or not
Acidic charged 1 The residue is acidic charged

or not
Basic charged 1 The residue is basic charged or not
Residue weight 1 The weight of the residue
Dissociation constant for 1 The negative of the logarithm of
the –COOH group the dissociation constant for the



Dissociation constant for 1 The negative of the logarithm of
the –COOH group the dissociation constant for the

–COOH group
Dissociation constant for 1 The negative of the logarithm of
the –NH3 group the dissociation constant for the



Dissociation constant for 1 The negative of the logarithm of
the –NH3 group the dissociation constant for the

–NH3 group
Dissociation constant for 1 The negative of the logarithm of
any other group in the dissociation constant for any
the molecule



Dissociation constant for 1 The negative of the logarithm of
any other group in the dissociation constant for any
the molecule other group in the molecule

The pH at the 1 The pH at the isoelectric point
isoelectric point



occurrence frequency of the residue _r_ at the position _k_ . _p_ is a
pseudocount. _N_ is the total number of aligned sequences.


2.3 Methods

Figure 2 illustrates the proposed model architecture, which
can be divided into three main components: Input, Encoding,
and Prediction. In the Input part, the graph data is generated
along with the input sequence data. The Encoding part com­
prises three models. The Chemformer (Irwin _et al.,_ 2022) and
ESM (Rives _et al.,_ 2021) models are pre-trained models used
for extracting features from the sequence data of drugs and
targets, respectively. Additionally, the GNN part utilizes a ksubgraph GNN extractor, as proposed in NGNN (Zhang
and Li 2021), which begins by generating k-hop subgraphs.
Therefore, each node is a representation of a subgraph. The
method creates the graph representation in a hierarchical
fashion, beginning with the generation of subgraph-level rep­
resentations and then constructing the representation for the
entire graph. To capture the interaction information between
drugs and targets, a cross-attention free transformer (crossAFT) module is connected to the GNN model. Finally, in the
Prediction part, all the extracted features are integrated and
fed into a multilayer perceptron (MLP) module for making
predictions. The new algorithm is named iNGNN-DTI.



1 The pH at the isoelectric point



Hydrophobicity of
residue (pH ¼ 2)



Hydrophobicity of 1 Hydrophobicity of
residue (pH ¼ 2) residue (pH ¼ 2)

Hydrophobicity of 1 Hydrophobicity of
residue (pH ¼ 7) residue (pH ¼ 7)



1 Hydrophobicity of
residue (pH ¼ 7)



**Figure 2.** Model architecture of iNGNN-DTI. The SMILES string and the protein amino acid sequence are transformed into the graph data in the Input
section. The Encoding section uses the k-subgraph GNN extractor to encode the graph data while using the Chemformer and ESM models to encode the
SMILES string and amino acid sequence, respectively. To learn the interaction information, the cross-AFT module connects to the GNN extractor. The
final representation of the drug and the protein is created by concatenating the output of the cross-AFT module with the global feature created by the
unsupervised model. In the end, the two representations are concatenated and passed to an MLP module for the prediction of the interaction.


iNGNN-DTI **5**



Further details about each module will be presented in the
next subsections.


**2.3.1 k-subgraph GNN extractor layer with virtual node**
To encode the graph data, we utilize the k-subgraph GNN
extractor, as introduced in NGNN ( Zhang and Li, 2021).
Traditional GNN models like GCN have limitations in
expressing substructures when the graph is nested rather than
a tree structure (Chen _et al.,_ 2022). The k-subgraph GNN ex­
tractor addresses this limitation by updating the representa­
tion of each node using the k-hop subgraph surrounding that
node, resulting in more powerful representations compared
to base GNNs. The implementation of the k-subgraph GNN
extractor involves incorporating a subgraph pooling layer on
top of the base GNNs. This pooling operation aggregates in­
formation from the k-hop subgraph and propagates it to up­
date the node representations.

Here, we denote the graph data as _G_ ¼ ð _V; E_ Þ, where _V_ is
the set of nodes and _E_ is the set of edges. An intermediate
node representation will be generated first with a base layer
based on GCN. The intermediate representation of the node
_u_ at the layer _l_ þ 1 is calculated as:



where _Q_, _K_, and _V_ are query, key, and value matrices gener­
ated by linear transformations: _Q_ ¼ _XW_ _[Q]_, _K_ ¼ _XW_ _[K ]_ and
_V_ ¼ _XW_ _[V]_ . _w_ 2 _R_ _[T]_ [�] _[T ]_ is the pair-wise position biases learned
during training. _t_ indicates the position. r _q_ is the sigmoid
function. The operation is revised to a cross-AFT in
this work.

We denote the outputs of the GNN model as _H_ drug 2
_R_ _[N]_ [drug] [�] _[d ]_ and _H_ prot 2 _R_ _[N]_ [prot] [�] _[d ]_ for the drug and target, respec­
tively. _N_ drug is the number of nodes in the drug and _N_ prot is
the number of nodes in the protein. The cross-AFT on the
drug branch can be calculated as:


_Q_ drug _; K_ prot _; V_ prot ¼ _H_ drug _W_ drug _[Q]_ _[;][ H]_ [prot] _[W]_ prot _[K]_ _[;][ H]_ [prot] _[W]_ prot _[V]_ _[;]_


(8)



classical transformers (Vaswani _et al._ 2017), AFT proposes a
transformer operation without scale dot product attention.
Given an input _X_, AFT operates as:



_Y_ _t_ ¼ r _q_ _Q_ ð _t_ Þ �



_T_
P _t_ [0] ¼1 [exp] _[ K]_ � _t_ [0] [ þ] _[ w]_ _t; t_ [0] � � _V_ _t_ 0
_T_ _;_ (7)
~~P~~ _t_ [0] ¼1 [exp] _[ K]_ ~~�~~ _t_ [0] [ þ] _[ w]_ _t; t_ [0] ~~�~~



_Y_ drug _[n]_ [¼][ r] _[q]_ _[ Q]_ � _[n]_ drug � �



_;_ (3)
!



_h_ ð _u_ _[l]_ [þ][1] Þ ¼ r X

_j_ 2N _u_ ð Þ



_W_ _[l]_ _h_ [ð Þ] _[l]_
_j_




_[ K]_ � _[n]_ prot [0] [þ] _[ w]_ _[n][;][n]_ [0] � � _V_ _[n]_ [0]



_N_ prot
P _n_ [0] ¼1



prot

_n_ [0] ¼1 [exp] _[ K]_ _[n]_ [0]



¼1 prot _[n]_ _[;][n]_ prot


_N_ prot
~~P~~ _n_ [0] ¼1 [exp] _[ K]_ _[n]_ [0] [þ] _[ w]_ _[n][;][n]_ [0]



prot

_n_ [0] ¼1 [exp] _[ K]_ _[n]_ [0]



_;_ (9)

_[ K]_ ~~�~~ _[n]_ prot [0] [þ] _[ w]_ _[n][;][n]_ [0] ~~�~~



where _h_ ð _u_ _[l]_ [þ][1] Þ 2 R _[d ]_ denotes the representation of the node _u_
and _d_ is the size of the dimension. rð�Þ is a nonlinear activa­
tion function. N ð _u_ Þ is the set for the neighborhoods of the
node _u_ . _W_ _[l]_ 2 R _[d]_ [�] _[d ]_ is the weight of the layer _l_ . The final rep­
resentation is obtained through performing a subgraph pool­
ing on each node following the base GNN layers. The
pooling operation is calculated as:



_h_ _u_ ¼ X

_k_ 2 _S u_ ð Þ



_h_ _k_ _;_ (4)



_Y_ drug ¼ _Y_ drug þ _H_ drug _;_ (10)


where � is the element-wise product. _w_ 2 _R_ _[N]_ [drug] [�] _[N]_ [prot ] repre­
sents the pair-wise bias which is learned during the training
phase. The cross-AFT process on the target branch involves
similar steps, where the query matrix is generated using
_H_ prot, and the key and value matrices are derived from _H_ drug .


**2.3.3 Merge features extracted from the pre-trained models**
Molecule features extracted by pre-trained models may in­
clude fundamental properties information. We apply
Chemformer (Irwin _et al.,_ 2022) for feature extraction of
drug molecules and ESM (Rives _et al.,_ 2021) for proteins. For
both of them, the input is the sequence data. Chemformer is a
BART (Lewis _et al._, 2020) model which is pre-trained on
_>_ 100 million SMILES strings. The length of the output fea­
ture is 512. ESM is a 34-layer transformer model, and it is
trained with 250 million amino acid sequences. The length of
the output feature is 768. Both features are mapped to 128D
with the fully connected (FC) layer.

We denote the feature of the SMILES string as _P_ drug and
the feature of the target as _P_ prot . The two features are fused
with the outputs of the interaction module with the fully con­
nected (FC) layer to generate the final representations:


_F_ drug ¼ _FC_ � � _Y_ drug � _P_ drug �� _;_ (11)


_F_ prot ¼ _FC_ � � _Y_ prot � _P_ prot �� _;_ (12)


where ½�� denotes the concatenation operation. The two final
representations are finally joined together and given to an
MLP module to predict the interaction:



where _S_ ð _u_ Þ is the set of nodes in the k-subgraph of node _u_ .

A virtual node is an additional node inserted into the
graph. Virtual edges are created to link the virtual node to all
other nodes in the graph (Gilmer _et al.,_ 2017). After the
update of the node representation, the virtual node collects
the information of all nodes and scatters it back to each node
before the calculation of the next layer. The virtual node
operation is calculated as:



_v_ ð _[l]_ [�] [1] Þ _W_ _l_ 0ð Þ [þ] X _h_ [ð Þ] _i_ _[l]_ _W_ 1 [ð Þ] _[l]_ _;_ (5)

� � _i_ 2 _V_ � �



_v_ [ð Þ] _[l]_ ¼ r _v_ ð _[l]_ [�] [1] Þ _W_ _l_ 0ð Þ [þ] X



_i_ 2 _V_



_h_ ð _u_ _[l]_ [þ][1] Þ ¼ _h_ ð _u_ _[l]_ [þ][1] Þ þ _v_ [ð Þ] _[l]_ _;_ (6)


where _v_ [ð Þ] _[l]_ represents the virtual node representation at layer
_l_, and _W_ 0 [ð Þ] _[l]_ _[;][ W]_ _[ l]_ 1 [ð Þ] [2 R] _[d]_ [�] _[d ]_ [are two learnable weight matrices.]


**2.3.2 Interaction module**

The interaction module is designed to capture the molecular
interactions between drugs and their target proteins,
highlighting key atoms in the drug molecule and showcasing
the attention patterns in the target during the interaction.
The interaction module is implemented with an attention-free
transformer (AFT) ( Zhai _et al._, 2021). Different from


**6** Sun _et al._



_output_ ¼ _MLP_ � � _F_ drug � _F_ prot �� (13)


2.4 Baseline models and model evaluation

For the evaluation of the proposed model performance, four
baseline models are used for comparison. DeepDTA (Ozt [€] urk €
_et al._ 2018), ML-DTI ( Yang _et al._, 2021) and Moltrans
(Huang _et al._ 2021a, 2021b) are three methods using se­
quence data as inputs. DeepDTA uses two 3-layer CNN mod­
els for both drugs and targets. Moltrans is a transformer
model with an interaction matrix after feature extraction.
ML-DTI designs an interaction module between the CNN
layers. However, it only calculates the attention on the drug
molecule. DGraphDTA applies GNN to extract features on
the drug molecule and the target protein (Jiang _et al._ 2020).
The original DGraphDTA uses PconsC4 (Michel _et al._ 2019)
to construct the protein structure, whereas our test will use
the Alphafold2 result. All methods are tested with the sug­
gested hyperparameters from the original studies.

Four widely used performance metrics for the DTI task are
used in this study: area under the receiver operating charac­
teristic (AUROC), area under precision-recall curve
(AUPRC), sensitivity and specificity. Each dataset is split into
train, validation, and test sets in the ratio 7:1:2. During eval­
uation, we conduct five independent runs on each dataset
and report the mean value and standard deviation of
the results.


**3 Results**

3.1 Performance of the DTI prediction
As shown in Table 4, our method achieves the highest perfor­
mance according to AUROC and AUPRC among all methods
on all three datasets. For the Davis dataset, we get improve­
ments in AUROC of 2.3% (from 0.910 to 0.931) and
AUPRC of 23.8% (from 0.382 to 0.473). For the BIOSNAP
dataset, the improvements are 2.3% (from 0.913 to 0.934)
and 2.4% (from 0.917 to 0.939) for AUROC and AUPRC,
respectively. The improvements on the KIBA dataset are
0.3% (from 0.912 to 0.915) for AUROC and 1.3% (from
0.743 to 0.753) for AUPRC.


**Table 4.** Model performance comparison.



3.2 Performance on the test set with the unseen

drugs and targets
In practice, it is common for new drug or targets to be discov­
ered while their interactions have not yet been determined.
To ensure DTI predictions in such cases, unseen input data is
extracted from the DAVIS dataset. We process the dataset
following the setting in MolTrans (Huang _et al._ 2021a,
2021b). During the split of the dataset, we randomly select
20% drugs or targets and set all related DTI samples as the
test set. The remaining samples are divided into training and
validation sets in a 7:1 ratio. All baselines are tested on the

unseen datasets.


The results are shown in Table 5. When the test datasets
do not contain any unseen samples, we observe that ML-DTI
exhibits the highest performance compared to the other three
baseline methods on the DAVIS dataset (Table 4). However,
when considering the scenario of unseen drugs, ML-DTI
demonstrates performance comparable to DeepDTA and
MolTrans. Furthermore, our proposed method exhibits
greater consistency in its performance. Specifically, we
achieve improvements in AUROC of 2.5% (from 0.744 to
0.763) and AUPRC of 32.5% (from 0.169 to 0.224) for the
unseen drug case. Similarly, for the unseen protein case, we
achieve improvements in AUROC of 3.1% (from 0.840 to
0.867) and AUPRC of 14.3% (from 0.259 to 0.296).


3.3 Interpretability analysis
To enhance the interpretability of the prediction results, we
visualize the information learned by the interaction module,
specifically the AFT module. Unlike explicit attention matri­
ces, the AFT module does not directly learn an attention ma­
trix. However, it can be interpreted as having an implicit
attention mechanism through its operations. The interpreta­
tion of the AFT module with implicit attention can be under­
stood as follows:


_Y_ drug _[n][;][i]_ [¼] _[<][ a]_ _[n]_ drug _[;][i]_ _[;][ V]_ prot _[i]_ _[>;]_ (14)



**Method** **AUROC** **AUPRC** **Sensitivity** **Specificity**


DAVIS dataset

DeepDTA 0 _:_ 89260 _:_ 0066 0 _:_ 37860 _:_ 0231 0 _:_ 85460 _:_ 0066 0 _:_ 79260 _:_ 0291
Moltrans 0 _:_ 89860 _:_ 0050 0 _:_ 37160 _:_ 0067 0 _:_ 86560 _:_ 0050 0 _:_ 78360 _:_ 0387

ML-DTI 0 _:_ 91060 _:_ 0034 0 _:_ 38160 _:_ 0247 0 _:_ 89560 _:_ 0034 0 _:_ 79560 _:_ 0183
DGraphDTA (Alphafold2) 0 _:_ 88560 _:_ 0099 0 _:_ 31660 _:_ 0447 0 _:_ 89460 _:_ 0099 0 _:_ 72460 _:_ 0467
**iNGNN-DTI** **0.931 ± 0.0027** **0.473 ± 0.0167** **0.922 ± 0.0155** **0.802 ± 0.0240**

KIBA dataset

DeepDTA 0 _:_ 91260 _:_ 0037 0 _:_ 74360 _:_ 0127 0 _:_ 88160 _:_ 0056 0 _:_ 78060 _:_ 0127
Moltrans 0 _:_ 89960 _:_ 0022 0 _:_ 69160 _:_ 0142 0 _:_ 87260 _:_ 0116 0 _:_ 76060 _:_ 0160

ML-DTI 0 _:_ 90960 _:_ 0020 0 _:_ 72760 _:_ 0108 0 _:_ 87860 _:_ 0111 0 _:_ 77960 _:_ 0113
DGraphDTA (Alphafold2) 0 _:_ 91160 _:_ 0004 0 _:_ 73960 _:_ 0043 0 _:_ 88160 _:_ 0183 **0.784 ± 0.0277**
**iNGNN-DTI** **0.915 ± 0.0016** **0.753 ± 0.0071** **0.888 ± 0.0107** 0 _:_ 77960 _:_ 0146

BIOSNAP dataset

DeepDTA 0 _:_ 89760 _:_ 0027 0 _:_ 90060 _:_ 0046 0 _:_ 85960 _:_ 0089 0 _:_ 78660 _:_ 0197
Moltrans 0 _:_ 88760 _:_ 0034 0 _:_ 88160 _:_ 0085 0 _:_ 82460 _:_ 0106 0 _:_ 80960 _:_ 0104

ML-DTI 0 _:_ 91160 _:_ 0053 0 _:_ 91160 _:_ 0112 0 _:_ 85160 _:_ 0054 0 _:_ 82860 _:_ 0215
DGraphDTA (Alphafold2) 0 _:_ 91360 _:_ 0022 0 _:_ 91760 _:_ 0024 0 _:_ 85860 _:_ 0175 0 _:_ 83160 _:_ 0151
**iNGNN-DTI** **0.934 ± 0.0021** **0.939 ± 0.0022** **0.872 ± 0.0189** **0.854 ± 0.0200**


The bolded numbers represent the best results.


iNGNN-DTI **7**


**Table 5.** Model performance on the test set with unseen drugs and proteins.


**Method** **unseen drugs** **unseen proteins**


**AUROC** **AUPRC** **AUROC** **AUPRC**


DeepDTA 0 _:_ 73660 _:_ 0550 0 _:_ 14560 _:_ 0300 0 _:_ 77060 _:_ 0594 0 _:_ 14560 _:_ 0300
Moltrans 0 _:_ 74460 _:_ 0540 0 _:_ 14460 _:_ 0370 0 _:_ 77860 _:_ 0538 0 _:_ 23160 _:_ 0534

ML-DTI 0 _:_ 73760 _:_ 0700 0 _:_ 16960 _:_ 0730 0 _:_ 84060 _:_ 0357 0 _:_ 25960 _:_ 0519
DGraphDTA 0 _:_ 71860 _:_ 0045 0 _:_ 16960 _:_ 0049 0 _:_ 78060 _:_ 0478 0 _:_ 16460 _:_ 0391
**iNGNN-DTI** **0.763 ± 0.0490** **0.224 ± 0.0640** **0.867 ± 0.0357** **0.296 ± 0.0534**


The bolded numbers represent the best results.


**Figure 3.** Visualization of the residues with the highest weight in each dimension on the target structure. The complete structure of the protein predicted
by AlphaFold2 is denoted by the green structures, while the purple structures represent the structure sourced from the Protein Data Bank (PDB).
Residues with the highest scores in each dimension are marked by pink dots.



_s:t: a_ _[n][;][ i]_
drug [¼]



r _q_ _Q_ � _[n]_ drug _[;][i]_ �exp _K_ � _[n]_ prot _[;][i]_ [þ] _[ w]_ _n_ �



_N_ prot _;_
~~P~~ _n_ [0] ¼1 [exp] _[ K]_ ~~�~~ prot _[n]_ [0] _[;][i]_ [þ] _[ w]_ _n;n_ [0] ~~�~~



(15)



with the highest weight scores in each dimension, indicated
by the pink dots in Fig. 3. As illustrated in Fig. 3, the green
structures represent the complete protein structure generated
from AlphaFold2 and are used as input into our model,
whereas the purple sections indicate the reference structure
obtained from the Protein Data Bank (PDB). The entire pro­
tein is notably longer than the complex segment, presenting
difficulties in accurately identifying the binding site, particu­
larly in our configuration, which lacks explicit binding site in­
formation. Among the labelled residues, we note that some
residues are located in the loop regions, possibly owing to the
frequent presence of such structures in proteins, and these
regions show minimal differences. However, a distinct clus­
tering of residues around the interaction regions is clearly ob­
servable. This pattern is especially prominent in the case of
the 4OTG (Fig. 3b), where the majority of residues are
densely located near the complex.



_i_ ¼ 1 _;_ 2 _;_ . . . _; d; n_ ¼ 1 _;_ 2 _;_ . . . _; N_ prot _:_


Here _i_ is the index of feature dimension and _a_ _[n][;][ i]_
drug [2] _[ R]_ _[N]_ [prot ]

denotes the attention vector for each dimension. Two atten­
tion matrices _A_ drug 2 _R_ _[N]_ [drug] [�] _[N]_ [prot] [�] _[d ]_ and _A_ prot 2 _R_ _[N]_ [prot] [�] _[N]_ [drug] [�] _[d ]_

are generated and these matrices are derived by considering
drugs and targets as queries, respectively. The following sub­
sections analyze the learned attention matrices through visu­
alization and virtual docking.


**3.3.1 Visualization analysis of the targets**
We randomly select 4 protein targets that have both the true
and predicted protein structures. We visualize the residues


**8** Sun _et al._



**3.3.2 Virtual docking analysis**
We further perform virtual docking tests by placing the grid
box across the area where most residues with higher scores
are clustered. Figure 4 displays the grid box utilized for the
structure 2YFX case. The virtual docking results based on
Autodock Vina (Eberhardt _et al._ 2021) are shown in Fig. 5.
The ligand in blue (the true target structure in purple) is
sourced from the PDB, while the green ligand is derived from
our virtual docking result. Both cases achieved moderate vir­
tual docking scores (−8.9 and −7.2 kcal/mol), and for the
2YFX case, our virtual docking position closely aligns with
the position provided by the PDB.


**3.3.3 Visualization analysis on the drug molecules**
We visualize atoms with the highest weight on the drug mole­
cule (Fig. 6). In the 1M17 complex, the target protein forms hy­
drogen bonds with N2 and N3 in the drug molecule, both
highlighted by our attention mechanism. Similarly, in 4OTG,
O2 and N3 engage in hydrogen bonds with the target protein,
with one of them emphasized by our attention mechanism.


**Figure 4.** Illustration of the grid box configuration used for virtual docking
on 2YFX.



3.4 Ablation study
There are two modifications on top of the basic GNN net­
works: the interaction module using the cross AFT and the
use of the features extracted by the unsupervised model. We
conduct the ablation study to learn the effectiveness of each
module using the Davis dataset.

As shown in Table 6, we observe that the cross-AFT mod­
ule enhances the AUPRC result, and the incorporation of fea­
tures learned by the unsupervised models proves beneficial in
improving the AUROC. Combining the two modules resulted
in an improvement compared to the results obtained from the
vanilla NGNN. The AUROC increased by 1.3% (from 0.919
to 0.931), while the AUPRC increased by 1.9% (from 0.464
to 0.473).


3.5 Coronavirus disease 2019 case study
Coronavirus disease 2019 (COVID-19) is caused by Severe
Acute Respiratory Syndrome (SARS-CoV-2) (Mohanty _et al._
2020). Numerous proteins have been discovered that have an
effect on the COVID-19 virus. Our objective in this analysis
is to explore the potential of repurposing drugs that interact
with angiotensin-converting enzyme 2 (ACE2), a receptor
protein found on the surface of cells (Kumar 2020). This pro­
tein serves as the receptor for SARS-CoV-2 to enter and infect
human cells.

We pre-train a model using the BIOSNAP dataset and pre­
dict the interaction between ACE2 and all 4510 drugs in the
BIOSNAP dataset. Due to the limited validations of most
drugs against the ACE2 target, our approach involves con­
ducting blind virtual docking to discover potential binding
sites for each drug from the whole protein structure for per­
formance evaluation utilizing the DockConv2 (Chen _et al._
2021). Additionally, we also performed virtual docking
analysis based on the grid box across the area where most
residues with higher scores are clustered following the meth­
odology outlined above. Figure 7 illustrates the blind virtual
docking scores for the generated top 50 molecules and top
50–100 molecules. The mean (median) virtual docking score
obtained from DockConv2 for the top 50 drugs is −6.84

−
( 7.00) kcal/mol, while for the top 50–100 drugs, the score is

− −
6.49 ( 6.21) kcal/mol. The top 50 drugs showed better per­
formance in terms of virtual docking scores.



**Figure 5.** Visualization of the virtual docking result. The ligand in blue is sourced from the PDB, while the green ligand is derived from our virtual
docking result.


iNGNN-DTI **9**


**Figure 6.** Visualization of the atoms with the highest weight in each dimension on the drug molecule. In each panel, known polar contacts are illustrated
with yellow dashed lines on the left side figure. On the right side, atoms with high scores are denoted by pink points, while atoms forming bonds with
the target are enclosed within yellow circles.


**Table 6.** Ablation test on the DAVIS dataset.


**No. of tests** **GNN** **NGNN** **cross AFT** **unsupervised model** **AUROC** **AUPRC**


1 � 0 _:_ 91660 _:_ 0046 0 _:_ 44760 _:_ 0181
2 � 0 _:_ 91960 _:_ 0037 0 _:_ 46460 _:_ 0357
3 � � 0 _:_ 92060 _:_ 0051 **0.478 ± 0.0222**
4 � � 0 _:_ 92760 _:_ 0052 0 _:_ 43160 _:_ 0103
5 � � � **0.931 ± 0.0027** 0 _:_ 47360 _:_ 0167


The bolded numbers represent the best results.



Among the top five drugs generated by our methods,
Pseudoephedrine and Ephedrine are found as inhibitors that
disrupt the interactions between ACE2 and the SARS-CoV-2
receptor-binding domain of spike protein (Yu _et al._ 2021,
Mei _et al._ 2021). In Yamamoto _et al._ (2022), a cell surface en­
try path of SARS-CoV-2 is identified which is sensitive to var­
ious metalloproteinase inhibitors. Marimastat, a
metalloproteinase inhibitor, is considered a promising candi­
date for inhibiting COVID-19 (Yamamoto _et al._ (2022)).
Furthermore, erythromycin, exhibiting the lowest binding af­

−
finity with a score of 14.5 kcal/mol, has been the subject of
other studies where it is recognized as a high-potential drug
for COVID-19 (Wang _et al._ 2022).

Figure 8a illustrates the setting of the grid box on the
ACE2 target. We run the virtual docking for two drugs



(DB00786 and DB00616) from the top 5 drug molecules in
our results. The virtual docking results are displayed in
Fig. 8b and c. The virtual docking scores generated by our
grid box are −7.1 kcal/mol and −7.6 kcal/mol. Additionally,
we observed that our virtual docking positions closely align
with the results produced by blind docking using
DockCov2 (Fig. 9).


**4 Conclusion**


Our work introduces a novel interpretable nested graph neu­
ral network for the prediction of DTIs. We utilize
AlphaFold2 to generate the molecular structures of the target
proteins and create graph based on the structures. The model
combines features extracted from 1D sequence data using


**10** Sun _et al._



pre-trained models and features obtained from the 2D molec­
ular graph learned by the GNN. To capture the interaction
information between drugs and targets, we incorporate an
attention-free transformer module.

Our proposed architecture achieves superior performance
compared to four baseline models across all test datasets. In a
case study focused on COVID-19, we repurpose five drugs,
and three of them have been identified as promising candi­
dates for COVID-19 treatment based on previous studies.
This showcases the potential of our approach in aiding drug
discovery tasks.



From the interpretability analysis of our proposed model,
we observe that the model can highlight certain atoms within
some drugs that participate in hydrogen bonds or watermediated bonds. However, we note that the attention weights
assigned to the protein target do not always exhibit a strong
correlation with the actual positions of the residues, which
indicates the need for further improvement in our fu­
ture work.

Future study can be performed to improve our algorithm.
First, deeper analyses can be made to examine the perfor­
mance of our algorithm to use both membrane and nonmem­
brane proteins when AlphaFold2 is used to predict their
structures, although our preliminary results show that
AlphaFold2 can also accurately predict the structures of most
[of the membrane proteins in our datasets (Supplementary](https://academic.oup.com/bioinformatics/article-lookup/doi/10.1093/bioinformatics/btae135#supplementary-data)
[Figs S1 and S2); Second, using AlphaFold2 to build the con­](https://academic.oup.com/bioinformatics/article-lookup/doi/10.1093/bioinformatics/btae135#supplementary-data)
tact map is computationally extensive, it will be interesting to
explore other approaches to construct the contact map from
[protein sequence (Supplementary Fig. S3).](https://academic.oup.com/bioinformatics/article-lookup/doi/10.1093/bioinformatics/btae135#supplementary-data)



**Supplementary data**


[Supplementary data are available at](https://academic.oup.com/bioinformatics/article-lookup/doi/10.1093/bioinformatics/btae135#supplementary-data) _Bioinformatics_ online.


**Conflict of interest**


**Figure 7.** Virtual docking scores for top 50 drugs, and top 50–100 drugs. None declared.


**Figure 8.** Visualization of grid box-based virtual docking results for drugs DB00786 and DB00616 on the ACE2 target.


iNGNN-DTI **11**


**Figure 9.** Comparison of blind docking results and our grid box-based virtual docking results. The blue ligand represents the outcome from blind docking,
while the green ligand represents the results from our grid box-based virtual docking analysis.



**Funding**


This work was supported in part by the Canada Research
Chairs Tier II Program [CRC-2021-00482), the Canadian
Institute of Health Research [PLL 185683] and the Natural
Sciences and Engineering Research Council of Canada

[RGPIN-2021–04072].


**References**


Bai P, Miljkovi�c F, John B _et al._ Interpretable bilinear attention network
with domain adaptation improves drug–target prediction. _Nat_
_Mach Intell_ 2023; **5** :126–36. [https://doi.org/10.1038/s42256-022-](https://doi.org/10.1038/s42256-022-00605-1)

[00605-1.](https://doi.org/10.1038/s42256-022-00605-1)
Bronstein MM, Bruna J, LeCun Y _et al._ Geometric deep learning: going
beyond Euclidean data. _IEEE Signal Process Mag_ 2017; **34** :18–42.
[https://doi.org/10.1109/MSP.2017.2693418.](https://doi.org/10.1109/MSP.2017.2693418)
Chen D, O’Bray L, Borgwardt K. Structure-aware transformer for
graph representation learning. In: _Proceedings of the 39th_
_International Conference on Machine Learning, Baltimore,_
_Maryland USA_ . PMLR. 2022, 3469–89. [https://proceedings.mlr.](https://proceedings.mlr.press/v162/chen22r.html)
[press/v162/chen22r.html.](https://proceedings.mlr.press/v162/chen22r.html)
Chen TF, Chang YC, Hsiao Y _et al._ DockCoV2: a drug database against
SARS-CoV-2. _Nucleic Acids Res_ 2021; **49** [:D1152–9. https://doi.org/](https://doi.org/10.1093/nar/gkaa861)
[10.1093/nar/gkaa861.](https://doi.org/10.1093/nar/gkaa861)
Cheol Jeong J, Lin X, Chen XW. On position-specific scoring matrix
for protein function prediction. _IEEE/ACM Trans Comput Biol_
_Bioinf_ 2010; **8** :308–15.
Davis MI, Hunt JP, Herrgard S _et al._ Comprehensive analysis of kinase
inhibitor selectivity. _Nat Biotechnol_ 2011; **29** :1046–51.
Duarte JM, Sathyapriya R, Stehr H _et al._ Optimal contact definition for
reconstruction of contact maps. _BMC Bioinformatics_ 2010; **11** :283.
[https://doi.org/10.1186/1471-2105-11-283.](https://doi.org/10.1186/1471-2105-11-283)
Eberhardt J, Santos-Martins D, Tillack AF _et al._ AutoDock vina 1.2.0:
new docking methods, expanded force field, and python bindings. _J_
_Chem Inf Model_ 2021; **61** :3891–8. [https://doi.org/10.1021/acs.](https://doi.org/10.1021/acs.jcim.1c00203)
[jcim.1c00203.](https://doi.org/10.1021/acs.jcim.1c00203)
Edgar RC, Batzoglou S. Multiple sequence alignment. _Curr Opin Struct_
_Biol_ 2006; **16** :368–73.
Gilmer J, Schoenholz SS, Riley PF _et al._ Neural message passing for
quantum chemistry. In: _Proceedings of the 34th International_
_Conference on Machine Learning, Sydney, Australia_ . PMLR. 2017,
[1263–72. https://proceedings.mlr.press/v70/gilmer17a.html.](https://proceedings.mlr.press/v70/gilmer17a.html)
He T, Heidemeyer M, Ban F _et al._ SimBoost: a read-across approach for
predicting drug–target binding affinities using gradient boosting
machines. _J Cheminform_ 2017; **9** :24–14.



Huang K, Fu T, Glass LM _et al._ DeepPurpose: a deep learning library
for drug–target interaction prediction. _Bioinformatics_ 2021a; **36** :
[5545–7. https://doi.org/10.1093/bioinformatics/btaa1005.](https://doi.org/10.1093/bioinformatics/btaa1005)
Huang K, Xiao C, Glass LM _et al._ MolTrans: molecular interaction
transformer for drug–target interaction prediction. _Bioinformatics_
2021b; **37** :830–6.
Irwin R, Dimitriadis S, He J _et al._ Chemformer: a pre-trained trans­

former for computational chemistry. _Mach Learn Sci Technol_ 2022;
**3** :015022.
Jiang M, Li Z, Zhang S _et al._ Drug–target affinity prediction using
graph neural network and contact maps. _RSC Adv_ 2020; **10** :
[20701–12. https://doi.org/10.1039/D0RA02297G.](https://doi.org/10.1039/D0RA02297G)
Jumper J, Evans R, Pritzel A _et al._ Highly accurate protein structure pre­

diction with AlphaFold. _Nature_ 2021; **596** :583–9. [https://doi.org/](https://doi.org/10.1038/s41586-021-03819-2)
[10.1038/s41586-021-03819-2.](https://doi.org/10.1038/s41586-021-03819-2)
Kumar S. _COVID-19: A Drug Repurposing and Biomarker_
_Identification by Using Comprehensive Gene-disease Associations_
_Through protein-protein Interaction Network Analysis_ . _Preprints,_
Basel, Switzerland: MDPI 2020, [https://doi.org/10.20944/pre](https://doi.org/10.20944/preprints202003.0440.v1)
[prints202003.0440.v1.](https://doi.org/10.20944/preprints202003.0440.v1)
Landrum G. _RDKit: Open-Source Cheminformatics Software_ . 2016.

[https://github.com/rdkit/rdkit/releases/tag/Release_2016_09_4.](https://github.com/rdkit/rdkit/releases/tag/Release_2016_09_4)
Lewis M, Liu Y, Goyal N _et al._ BART: Denoising Sequence-to-Sequence
Pre-training for Natural Language Generation, Translation, and
Comprehension. In: _Proceedings of the 58th Annual Meeting of the_
_Association for Computational Linguistics_ . 2020. Online. [https://](https://doi.org/10.18653/v1/2020.acl-main.703)
[doi.org/10.18653/v1/2020.acl-main.703.](https://doi.org/10.18653/v1/2020.acl-main.703)
Zitnik M, Rok Sosic SM, Leskovec J. BioSNAP Datasets: Stanford
Biomedical Network Dataset Collection. 2018. [http://snap.stan](http://snap.stanford.edu/biodata)
[ford.edu/biodata. Stanford, USA.](http://snap.stanford.edu/biodata)
Mei J, Zhou Y, Yang X _et al._ Active components in ephedra Sinica
Stapf disrupt the interaction between ACE2 and SARS-CoV-2 RBD:
potent COVID-19 therapeutic agents. _J Ethnopharmacol_ 2021; **278** :
[114303. https://doi.org/10.1016/j.jep.2021.114303.](https://doi.org/10.1016/j.jep.2021.114303)
Michel M, Men�endez Hurtado D, Elofsson A. PconsC4: fast, accurate
and hassle-free contact predictions. _Bioinformatics_ 2019; **35** :
[2677–9. https://doi.org/10.1093/bioinformatics/bty1036.](https://doi.org/10.1093/bioinformatics/bty1036)
Mohanty SK, Satapathy A, Naidu MM _et al._ Severe acute respiratory syn­

drome coronavirus-2 (SARS-CoV-2) and coronavirus disease 19
(COVID-19) – anatomic pathology perspective on current knowledge.
_Diagn_ _Pathol_ 2020; **15** :103. [https://doi.org/10.1186/s13000-020-](https://doi.org/10.1186/s13000-020-01017-8)
[01017-8.](https://doi.org/10.1186/s13000-020-01017-8)
Nguyen T, Le H, Quinn TP _et al._ Predicting drug–target binding affinity
with graph neural networks. _Bioinformatics_ 2021; **37** :1140–7.
Ou-Yang S, Lu J, Kong X _et al._ Computational drug discovery. _Acta_
_Pharmacol Sin_ 2012; **33** :1131–40. [https://doi.org/10.1038/aps.](https://doi.org/10.1038/aps.2012.109)
[2012.109.](https://doi.org/10.1038/aps.2012.109)


**12** Sun _et al._



Ozt€ urk H, € Ozg€ ur A, Ozkirimli E. DeepDTA: deep drug–target binding €
affinity prediction. _Bioinformatics_ 2018; **34** :i821–i829.
Rives A, Meier J, Sercu T _et al._ Biological structure and function emerge
from scaling unsupervised learning to 250 million protein sequen­
ces. _Proc Natl Acad Sci USA_ 2021; **118** :e2016239118.
Sachdev K, Gupta MK. A comprehensive review of feature b methods
for drug target interaction prediction. _J Biomed Inform_ 2019; **93** :
[103159. https://doi.org/10.1016/j.jbi.2019.103159.](https://doi.org/10.1016/j.jbi.2019.103159)
Schenone M, Dan�c�ık V, Wagner BK _et al._ Target identification and
mechanism of action in chemical biology and drug discovery. _Nat_
_Chem Biol_ 2013; **9** :232–40.
Vaswani A, Shazeer N, Parmar N _et al._ Attention is all you need. In:
_Advances in Neural Information Processing Systems, Long Beach,_
_CA, USA_ 2017; **30** .
Wang X, Chen Y, Shi H _et al._ Erythromycin estolate is a potent inhibi­

tor against HCoV-OC43 by directly inactivating the virus particle.
_Front Cell Infect Microbiol_ 2022; **12** :905248. [https://doi.org/10.](https://doi.org/10.3389/fcimb.2022.905248)
[3389/fcimb.2022.905248.](https://doi.org/10.3389/fcimb.2022.905248)



Wen M, Zhang Z, Niu S _et al._ Deep-learning-based drug-target interac­

tion prediction. _J Proteome Res_ 2017; **16** [:1401–9. 10.1021/acs.jpro­](https://doi.org/10.1021/acs.jproteome.6b00618)
[teome.6b00618. 28264154](https://doi.org/10.1021/acs.jproteome.6b00618)
Yamamoto M, Gohda JIN, Kobayashi A _et al._ Metalloproteinase-de­

pendent and TMPRSS2-independent cell surface entry pathway of
SARS-cov-2 requires the furin cleavage site and the S2 domain of
spike protein. _mBio_ 2022; **13** :e00519–22.
Yang Z, Zhong W, Zhao LU _et al._ ML-DTI: Mutual learning mechanism
for interpretable drug-target interaction prediction. _J Phys Chem Lett_
2021; **12** [:4247–61. 10.1021/acs.jpclett.1c00867.](https://doi.org/10.1021/acs.jpclett.1c00867)
Yu S, Chen YAO, Xiang Y _et al._ Pseudoephedrine and its derivatives an­

tagonize wild and mutated severe acute respiratory syndrome-cov-2
viruses through blocking virus invasion and antiinflammatory effect.
_Phytother Res_ 2021; **35** [:5847–60. 10.1002/ptr.7245.](https://doi.org/10.1002/ptr.7245)
Zhai S, Talbott W, Srivastava N _et al._ An attention free transformer.
_arXiv Preprint Arxiv:2105.14103_ 2021.
Zhang M, Li Pan. _Advances in neural information processing systems_ .
Curran Associates, Inc., 2021. **34**, 15734–15747.


