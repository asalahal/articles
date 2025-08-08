[Computers in Biology and Medicine 150 (2022) 106127](https://doi.org/10.1016/j.compbiomed.2022.106127)


Contents lists available at ScienceDirect

# Computers in Biology and Medicine


[journal homepage: www.elsevier.com/locate/compbiomed](https://www.elsevier.com/locate/compbiomed)

## REDDA: Integrating multiple biological relations to heterogeneous graph neural network for drug-disease association prediction


Yaowen Gu [a] [,] [1], Si Zheng [a] [,] [b] [,] [1], Qijin Yin [c], Rui Jiang [c], Jiao Li [a] [,] [* ]


a _Institute of Medical Information (IMI), Chinese Academy of Medical Sciences and Peking Union Medical College (CAMS & PUMC), Beijing, 100020, China_
bc _Ministry of Education Key Laboratory of Bioinformatics, Bioinformatics Division at the Beijing National Research Center for Information Science and Technology, Center Institute for Artificial Intelligence, Department of Computer Science and Technology, BNRist, Tsinghua University, Beijing, 100084, China_
_for Synthetic and Systems Biology, Department of Automation, Tsinghua University, Beijing, 100084, China_



A R T I C L E I N F O


_Keywords:_
Drug repositioning
Drug-disease association prediction
Heterogeneous graph neural network
Topological subnet Topological subnet


**1. Introduction**



A B S T R A C T


Computational drug repositioning is an effective way to find new indications for existing drugs, thus can
accelerate drug development and reduce experimental costs. Recently, various deep learning-based repurposing
methods have been established to identify the potential drug-disease associations (DDA). However, effective
utilization of the relations of biological entities to capture the biological interactions to enhance the drug-disease
association prediction is still challenging. To resolve the above problem, we proposed a heterogeneous graph
neural network called REDDA ( **R** elations- **E** nhanced **D** rug- **D** isease **A** ssociation prediction). Assembled with three
attention mechanisms, REDDA can sequentially learn drug/disease representations by a general heterogeneous
graph convolutional network-based node embedding block, a topological subnet embedding block, a graph
attention block, and a layer attention block. Performance comparisons on our proposed benchmark dataset show
that REDDA outperforms 8 advanced drug-disease association prediction methods, achieving relative improve­
ments of 0.76% on the area under the receiver operating characteristic curve (AUC) score and 13.92% on the
precision-recall curve (AUPR) score compared to the suboptimal method. On the other benchmark dataset,
REDDA also obtains relative improvements of 2.48% on the AUC score and 4.93% on the AUPR score. Specif­
ically, case studies also indicate that REDDA can give valid predictions for the discovery of -new indications for
drugs and new therapies for diseases. The overall results provide an inspiring potential for REDDA in the _in silico_
[drug development. The proposed benchmark dataset and source code are available in https://github.com/gu-yao](https://github.com/gu-yaowen/REDDA)

[wen/REDDA.](https://github.com/gu-yaowen/REDDA)



The traditional wet-experiment-guided drug discovery is a timeconsuming and high-risk process [1]. Recently, it has become increas­
ingly difficult to identify potential therapeutic entities with novel
chemical structures. The total cost of developing a new drug range from
3.2 to 27.0 billion dollars, take over 5.8–15.2 years and only achieve a
success rate of 6.2% [2–4]. Thus, computational methods with cheaper
and labor-saving solutions can accelerate the drug discovery, and have
attracted increasing interests for both pharmaceutical industry and ac­
ademic research communities [5,6]. For instance, there have been suc­
cessful applications in drug property prediction [7–9], drug-target
interaction assessment [8,10,11], and drug sensitivity prediction [12,


 - Corresponding author.
_E-mail address:_ [li.jiao@imicams.ac.cn (J. Li).](mailto:li.jiao@imicams.ac.cn)
1 The first two authors contribute equally to the paper.



13], etc. Computational drug repositioning methods focus on deter­
mining the new indications for drugs [14], thus reducing the unnec­
essary cost and improving success rate of drug development [15]. A
variety of promising applications show that the role of computational
drug repositioning in drug development cannot be ignored [16–20].
The existing computational drug repositioning methods can be
approximately divided into 4 categories [21]: classical machine learning
approaches, network propagation approaches, matrix factor­
ization/completion approaches, and deep learning approaches.
Classical machine learning approaches take known drug-disease as­
sociation pairs as positive labels to convert drug repositioning into a
binary classification problem and further adopt the drug and disease
information as input features to train machine learning classifiers. For



[https://doi.org/10.1016/j.compbiomed.2022.106127](https://doi.org/10.1016/j.compbiomed.2022.106127)
Received 18 June 2022; Received in revised form 27 July 2022; Accepted 18 September 2022

Available online 22 September 2022
0010-4825/© 2022 Elsevier Ltd. All rights reserved.


_Y. Gu et al._ _Computers in Biology and Medicine 150 (2022) 106127_



instance, Gao et al. proposed a Laplacian regularized least squares al­
gorithm combined with a similarity kernel fusion method to predict the
drug-disease association, which called DDA-SKF [22]. Network propa­
gation approaches construct drug-disease heterogeneous networks and
use network-based algorithms (e.g., random walk) to predict the
drug-disease association probabilities [6]. For example, Luo et al. pro­
posed a method called MBiRW, which used similarity measurers to
construct a drug-disease heterogeneous network and adopted the
bi-random walk algorithm to predict potential drug-disease associations

[23]. Matrix factorization/completion approaches model drug reposi­
tioning as a recommendation system, thus recommending new drug­
s/indications based on prior information such as known drug-disease
associations. Zhang et al. proposed a similarity constrained matrix
factorization method called SCMFDD for drug-disease association pre­
diction. It maps the drug and disease features to low-rank spaces for
solving constraint optimization [24]. Yang et al. took drug-disease as­
sociation prediction as a noisy matrix completion problem and devel­
oped a bounded nuclear norm regularization (BNNR) method for it [25].
Yang et al. proposed a multi-similarities for bilinear matrix factorization
(MSBMF) to extract effective drug and disease representations, which
could be used to infer missing drug-disease associations [26]. With
ongoing development in recent years, these types of computational drug
repositioning methods have gained competitive performance. However,
some crucial shortcomings still limit the achievement of higher accuracy
and the utilization of practical scenarios, such as the high dependence on
the quality of input features in machine learning approaches, the rep­
resentation bias for nodes with high degrees on heterogeneous network
in network propagation approaches, and the weak representation ability
of drug-disease associations caused by linear multiplication in matrix
factorization/completion approaches.
Deep learning approaches have been effectively applied in many
biological domains, such as gene regulatory representation [27,28],
single-cell omics analysis [29–32], drug efficacy prediction [33], etc. For
drug repositioning, deep learning approaches use neural networks to
model the interactions between drugs and diseases with high flexibility
and scalability, which have been widely used and proven to be highly
competitive compared to the above three approaches [34–41]. For
instance, Zeng et al. integrated 10 drug-disease-related networks and
trained a multimodal deep autoencoder on them to learn the high-order
representations for drug repositioning, which is called deepDR [37]; Yu
et al. established a graph convolutional network called LAGCN on a
heterogeneous drug-disease network [34]; Meng et al. proposed a
neighborhood and neighborhood interaction-based neural collaborative
filtering approach called DRWBNCF for drug repositioning [36]. Zhang
et al. designed a multi-scale topology learning method which integrated
multiple drug-disease heterogeneous network and adopted random walk
and attention mechanism for representation learning [42]. Xuan et al.
proposed a graph autoencoder architecture with scale-level attention
and convolutional neural network which called MGPred [43].
As sufficiently advanced methods for modeling drug-disease associ­
ations, these approaches have provided a series of attractive method­
ologies for deep learning-based drug repositioning, such as the
construction of the heterogeneous networks, the utilization of layer
attention mechanism, and bilinear dot decoder, etc. However, the drugdisease associations cannot be simply integrated as an isolated biological
system as the above studies have done, while ignoring other extensive

–
biological interactions, such as drug-protein, protein gene, genepathway, pathway-disease, etc. From our perspective, these external
biological relations can be assembled in the drug-disease heterogeneous
network and bring extra information for the simulations of drug thera­
peutic process, thus enhance the representation ability of the drug
repositioning model. Nevertheless, these concerns have not been studied
in depth.
To resolve these problems, we propose a benchmark dataset which
can construct to a heterogeneous network with 5 entities (drug, protein,
gene, pathway, and disease) and 10 relations (drug-drug, drug-protein,



protein-protein, gene-gene, gene-pathway, pathway-pathway, pathwaydisease, disease-disease, and drug-disease) for drug repositioning.
Furthermore, we also develop a promising drug repositioning method on
the heterogeneous network, which we called **R** elations- **E** nhanced **D** rug**D** isease **A** ssociation prediction (REDDA). The main contributions of this
work are summarized as follows:


- We propose a large-scale benchmark dataset for drug repositioning.
The benchmark contains 41,100 nodes and 1,008,258 edges with 5
biological entities (drug, protein, gene, pathway, and disease).

- We propose a deep learning-based method for drug repositioning,
namely REDDA. It takes the heterogeneous graph neural network as
the backbone and integrates 3 attention mechanisms to learn the
node representations of the heterogeneous network and topological
subnets.

- Comprehensive experiments demonstrate that REDDA outperforms
several state-of-the-art algorithms. Ablation experiments indicate
that the fusion of extra biological relations is beneficial for REDDA to
predict drug-disease associations. Attention visualization analysis
shows the importance of topological decomposition, graph-level
aggregation, and layer-level aggregation in REDDA.


**2. Materials and methods**


_2.1. Dataset_


As the existing benchmarks lack the biological entities and their re­
lations, we construct a drug-disease association benchmark, including 5
entities (drug, protein, gene, pathway, and disease) and 10 relations
(drug-drug, drug-protein, protein-protein, protein-gene, gene-gene,
gene-pathway, pathway-pathway, pathway-disease, disease-disease,
and drug-disease) as these biological entities and relations have been
proved to contribute to drug repositioning [44–46]. We assemble the
Fdataset [47], Cdataset [23], and additional data downloaded from
KEGG [48] as our drug-disease association data, which contains 894
drugs, 454 diseases, and 2704 confirmed drug-disease associations.
Then, we adopt the biological relation data from DrugBank [49], CTD

[50], KEGG [48], STRING [51], and UniProt [52], integrating them into
a large-scale benchmark including 41,100 nodes and 1,008,258 edges.
The descriptions of our proposed benchmark dataset are listed in
Table 1. The Venn diagrams and other details of our proposed bench­
mark dataset can be founded in Fig. S1 and Table S1.
To emphasize the robustness and stability of REDDA, we also reor­
ganize a public drug-disease association benchmark dataset called Bdataset for evaluation, which is used in Ref. [44] and originally pro­
posed in Ref. [24], including 269 drugs, 598 diseases, but 18,416


**Table 1**

Summary of our proposed drug-disease association benchmark dataset.


Dataset Data Type


Entities


Drugs Proteins Genes Pathways Diseases
Number 894 18,877 20,561 314 454
Resource DrugBank, KEGG, CTD KEGG, KEGG,
Reference Uniprot CTD Reference

Interactions

Drug-Drug Protein- Gene- Pathway- DiseaseProtein Gene Pathway Disease
Number 14,291 201,382 712,546 1669 7199

Resource Sim Comput STRING CTD KEGG Sim
Comput

Associations

Drug-Protein Protein- Gene- Pathway- DrugGene Pathway Disease Disease
Number 4397 18,545 25,995 19,530 2704
Resource DrugBank KEGG, CTD [′] CTD KEGG,
Uniprot Reference



2


_Y. Gu et al._ _Computers in Biology and Medicine 150 (2022) 106127_



drug-disease associations. Moreover, we collect the drug-protein, pro­
tein-disease associations, and protein-protein interactions data from
DrugBank, STRING, and CTD databases. The detailed descriptions of the
B-dataset are listed in Table 2.


_2.2. Construction of the heterogeneous network_


_2.2.1. Drug-drug similarities_
As previous research has proven the benefit of computing drug-drug
similarities as the drug-drug interaction probabilities in the drug-disease
heterogeneous network [34–36], we compute the chemical structure
similarities in our method by converting the SMILES sequences to
2048-bit ECFP4 fingerprints [53] and calculate the pair-wise similarities
of drug fingerprints using some similarity measurements. Given two
drugs _i_ and _j_, their fingerprints are represented as _x_ _i_ and _x_ _j_, the Jaccard
index can be calculated by:



_C_ _d_ ( _n_ ) = { 1 _max, if n_ {△ = _. dC_ _d_ ( _n_ ′



′ (4)
∈ _children of n_ } _, otherwise_



′
)| _n_



where _δ_ is a contribution factor. The overall semantic contribution of _d_ is

_DV_ ( _d_ ) = ∑ _C_ _d_ ( _n_ ). Then, the disease-disease similarity of _d_ _i_ and _d_ _j_ can

_n_ ∈ **N** ( _d_ )

be measured by the number of common ancestral nodes and the se­
mantic contribution proportion of these ancestral nodes in _DV_ (d _i_ ) and
_DV_ (d _j_ ), which can be formulated as:



∑



_S_ _ij_ _[D]_ [=]



_n_ ∈ **N** (d _i_ )∩ **N** ( d _j_ )( _C_ d _i_ ( _n_ ) + _C_ d _j_ ( _n_ ))



_DV_ (d _i_ ) + _DV_ ~~(~~ d _j_



(5)
~~)~~



(1)
~~⃒⃒~~



_S_ _ij_ _[R]_ [=]



⃒⃒ _x_ _i_ ∩ _x_ _j_



⃒⃒ _x_ _i_ ∩ _x_ _j_ ⃒⃒

~~⃒⃒~~ _x_ _i_ ∪ _x_ _j_ ~~⃒⃒~~



Similarly, the Tanimoto similarity can be calculated by:

_S_ _ij_ _[R]_ [=] _x_ _i_ [2] + _xx_ _ij_ [2] _x_ _j_ − _x_ _i_ _x_ _j_


The Dice index can be calculated by:



(2)



Similar to the operation on drug similarity, _S_ _[D]_ ∈{0 _,_ 1} are acquired
as the final disease-disease interactions by _top15_ filtering.


_2.2.3. Construction of heterogeneous network_
The heterogeneous network is constructed by our proposed bench­
marks. Given an entity type set **O** = {ℴ| _R, P, G, M, D_ } (representing drug,
protein, gene, pathway, and disease, respectively), then the heteroge­
neous network can be regarded as a heterogeneous graph **G** **[O]** = ( **V** **[O]** _,_

**E** **[O]** ), where **V** **[O]** = { **V** _[R]_ _,_ **V** _[P]_ _,_ **V** _[G]_ _,_ **V** _[M]_ _,_ **V** _[D]_ } is the node set with 5

types, and **E** **[O]** = { **E** _[R]_ [−] _[R]_ _,_ … _,_ **E** [ℴ] [1] [−] [ℴ] [2] _,_ … _,_ **E** _[D]_ [−] _[D]_ } is the edge set of **G** **[O]** .
The unweighted and undirected heterogeneous network can be divided
into an adjacency matrix _A_ **[O ]** and the node feature matrix _H_ **[O]** . Based on

the collected relations in our benchmark, the adjacency matrix _A_ **[O]** ∈

R [(] _[N]_ _[R]_ [+] _[N]_ _[P]_ [+] _[N]_ _[G]_ [+] _[N]_ _[M]_ [+] _[N]_ _[D]_ [)×(] _[N]_ _[R]_ [+] _[N]_ _[P]_ [+] _[N]_ _[G]_ [+] _[N]_ _[M]_ [+] _[N]_ _[D]_ [)] can be defined as:



⃒
⃒



⃒⃒ _x_ _i_ ∩ _x_ _j_
_S_ _ij_ _[R]_ [=] [2]



(3)
~~⃒⃒~~



| _x_ _i_ | + ~~⃒⃒~~ _x_ _j_



The model performance comparisons for different similarity mea­
surements can be shown in Table S2. Consequently, we adopt the
Tanimoto similarity as the optimal method for calculating drug-drug
similarities. Also, considering the computing complexity of the hetero­
geneous network, we simplify the continual drug-drug similarities to
binarization values. Given a drug _i_ and its similarities with another drug
_S_ _[R]_ _i_ [∈[][0] _[,]_ [ 1][]][, we convert the ] _[topk ]_ [similarities (here ] _[k ]_ [=][ 15) ] _[S]_ _[R]_ _ik_ [to 1, and the ]
last ones are converted to 0 which means no existing interaction. After
the conversion, the _S_ _[R]_ ∈{0 _,_ 1} are gained as the final drug-drug
interactions.


_2.2.2. Disease-disease similarities_

To adopt disease-disease interactions, medical subject headings
(MeSH) identifiers are used to calculate semantic similarities as inter­
action probabilities. Similar to previous research [34,35,44,54], the
MeSH identifier of a disease can be represented as a hierarchical directed
acyclic graph (DAG). Given a disease _d_, the DAG can be represented as
DAG( _d_ ) = ( **N** ( _d_ ) _,_ **E** ( _d_ )), where **N** ( _d_ ) denotes the set of nodes including
_d_ and its ancestral nodes, and **E** ( _d_ ) denotes the parent-child relation
links among **N** ( _d_ ). The semantic contribution of a node _n_ ∈ **N** ( _d_ ) for _d_
can be formulated as:


**Table 2**

Summary of B-dataset.


Dataset Data Type


Entities


Drugs Proteins Diseases

Number 269 6040 598

Resource DrugBank, Reference Uniprot Reference

Interactions

Drug-Drug Protein-Protein Disease-Disease
Number 72,361 592,926 357,604

Resource Sim Comput STRING Sim Comput

Associations

Drug-Protein Protein-Disease Drug-Disease
Number 2107 17,631 18,416
Resource DrugBank CTD Reference



⎡

⎢⎢⎢⎢⎢⎣



⎡



⎤


(6)
⎥⎥⎥⎥⎥⎦



_A_ **[O]** =



_S_ _[R]_ _A_ _[R]_ [−] _[P]_ 0 0 _A_ _[R]_ [−] _[D]_
( _A_ _[R]_ [−] _[P]_ [)] _[T]_ _A_ _[P]_ [−] _[P]_ _A_ _[P]_ [−] _[G]_ 0 0
0 ( _A_ _[P]_ [−] _[G]_ [)] _[T]_ _A_ _[G]_ [−] _[G]_ _A_ _[G]_ [−] _[M]_ 0
0 0 ( _A_ _[G]_ [−] _[M]_ [)] _[T]_ _A_ _[M]_ [−] _[M]_ _A_ _[M]_ [−] _[D]_
( _A_ _[R]_ [−] _[D]_ [)] _[T]_ 0 0 ( _A_ _[M]_ [−] _[D]_ [)] _[T]_ _S_ _[D]_



We also take the computed similarities _S_ _[R ]_ and _S_ _[D ]_ to represent the
chemical structure pairwise features and disease classification pairwise

features, so _H_ **[O]** ∈ R [(] _[N]_ _[R]_ [+] _[N]_ _[D]_ [)×(] _[N]_ _[D]_ [+] _[N]_ _[R]_ [)] can be initialized and represented as:



_H_ _[R]_
_H_ **[O]** =

[ _H_ _[D]_



_S_ _[R]_ 0
] = [ 0 _S_ _[D]_



(7)
]



_2.3. Backbone model and mechanism_


_2.3.1. Heterogeneous graph convolutional network (HeteroGCN)_
In REDDA, a HeteroGCN is used as the backbone model, which learns
the node representations in a series of homogeneous graphs and then
aggregates the neighbor nodes’ embedding with different node types to
accomplish heterogeneous graph learning and node embedding
updating.
Given a homogeneous graph **G** _[ϑ]_ [−] _[ϑ ]_ whose node type is _ϑ_, the node

representations _H_ [̂] _ϑ_ can be learned by a classic GCN in **G** _ϑ_ − _ϑ_ . The

computation in the GCN layer can be formulated as:



_H_ ̂



_ϑ_ = _GCN_ ( _A_ _ϑ_ − _ϑ_ _, H_ _ϑ_ _, W_ _ϑ_ ) (8)



where _A_ _[ϑ]_ [−] _[ϑ ]_ is the adjacency matrix of the _ϑ_ - _ϑ_ interaction network, _H_ _[ϑ ]_ is
the input node embedding, and _W_ _[ϑ ]_ is the trainable parameter matrix in
GCN. Specifically, the propagation of the GCN layer can be represented

as:



_GCN_ ( _A, H, W_ ) = _σ_ _D_ [−] 2 [1] _AD_ [−] 2 [1] _HW_ (9)

( )



∑
( _j_



where _D_ = diag



_A_ _ij_



)



is the degree matrix of **G** _[ϑ]_ [−] _[ϑ ]_ and _σ_ =



_j_



3


_Y. Gu et al._ _Computers in Biology and Medicine 150 (2022) 106127_



_x, x_ ≥ 0
{ _α_ _x, x <_ 0 [is the PReLU activation function. ]

After generating the homogeneous-graph-level node representations,
a sum aggregation method is traversed and executed on all nodes in the
heterogeneous graph to acquire the heterogeneous-graph-level node
representations. Given a node _i_, the aggregation method sums the em­
beddings of its neighbor nodes _j_ ∈ **N** [ℴ] _i_ [to calculate the final embedding ]
of _i_, which can be formulated as:



REDDA computation flow is described in Algorithm 1.


**Algorithm 1** . The REDDA algorithm.


_2.4.1. Node embedding block_
In REDDA, a node embedding block is adopted as the first module,
which includes a feature transformation layer and two heterogeneous

GCN (HeteroGCN) layers. First, taking the node feature matrix _H_ **[O]** ∈

R [(] _[N]_ _[R]_ [+] _[N]_ _[P]_ [+] _[N]_ _[G]_ [+] _[N]_ _[M]_ [+] _[N]_ _[D]_ [)×] _[K ]_ and heterogeneous network **G** **[O ]** as input, the
REDDA deploy a feature transformation layer to map node representa­
tions with different node types into the same hidden space. The feature
transformation layer can be formulated as:



_nj_ (10)



_H_ ̃ _i_ = ∑∑



_H_ ̂



_n_ ∈ **O** _j_ ∈ **N** _[n]_ _i_



_2.3.2. Attention mechanism_

Different blocks in REDDA focus on capturing different representa­
tions for identifying the biological relations. The attention mechanism
can aggregate multi-source representations and adjust their importance
weights dynamically. Therefore, we adopt the attention mechanism in
REDDA inspired by Ref. [55] to gather different representations. Given _K_
multi-source representations for node _i_, we first calculate the importance
weight _w_ _[k]_ _i_ [for each source ] _[k ]_ [by measuring the non-linear similarity of the ]
_h_ _[k]_ _i_ [and an attention vector ] _[q]_ [, which can be formulated as: ]



⎡

⎢⎢⎢⎢⎣



_Linear_ ( _H_ _[R]_ [)]

_Initialize_ ( **V** _[P]_ [)]

_Initialize_ ( **V** _[G]_ [)]

_Initialize_ ( **V** _[M]_ [)]

_Linear_ ( _H_ _[D]_ [)]



⎤


(14)
⎥⎥⎥⎥⎦



_H_ _[R]_ [(][0][)]

_H_ _[P]_ [(][0][)]

_H_ _[G]_ [(][0][)]

_H_ _[M]_ [(][0][)]

_H_ _[D]_ [(][0][)]



⎤


=
⎥⎥⎥⎥⎦



⎤



_H_ **[O]** [ (][0][)] =



⎡

⎢⎢⎢⎢⎣



_w_ _[k]_ _i_ [=] _K_ [1]



∑ _q_ _[T]_ Linear( _h_ _[k]_ _i_

_i_ ∈ _K_



) (11)



where the _Linear_ function is a linear layer for the feature dimension
alignment and _Initialize_ function is used to allocate initial features for
protein, gene, and pathway nodes.
For the node embedding updating, we use a HeteroGCN (described in
section 2.3.1) to propagate adjacent biological network information.
Given the _l_ -th layer, the output _H_ **[O]** [ (] _[l]_ [)] can be formulated as:



Then we normalize the _w_ _[k]_ _i_ [and acquired the attention coefficient ] _[α]_ _[k]_ _i_ [∈]
(0 _,_ 1) by a Softmax function:



_H_ ̃



**O** ( _l_ − 1) _, W_ **O** ( _l_ ) ) (15)



**O** ( _l_ )
= _HeteroGCN_ ( _A_ **[O]** _,_ _H_ [̂]



exp( _w_ _[k]_ _i_
_α_ _[k]_ _i_ [=] _K_



)



_K_ (12)
~~∑~~ _i_ =1 [exp][(] _[w]_ _i_ _[k]_ [)]



The final representation _H_ [̂] _i_ for node _i_ can be calculated by a weighted

sum:



_H_ ̂ _i_ = ∑ _K_

_k_ =1



_α_ _[k]_ _i_ _[H]_ _i_ _[k]_ (13)



_2.4.2. Topological subnet embedding block_
Although the node embedding block captures the global adjacency
relation among the heterogeneous network, the node representations
are biased because of the unbalanced number of different associations

(e.g., the number of pathway-disease associations is over 22 times the
number of drug-disease associations). Therefore, we propose a topo­
logical subnet embedding block to learn the node embedding guided by
different association information.

First, the topological subnet embedding block decomposes the het­
erogeneous network into 5 subnets that each of them include 2 node
types and 3 relation types (e.g., the subnet, which comprises drug-drug



_2.4. The architecture of REDDA_


In this section, we introduce the architecture of REDDA for drugdisease association prediction, which is shown in Fig. 1. The overall



**Fig. 1.** The architecture of REDDA.


4


_Y. Gu et al._ _Computers in Biology and Medicine 150 (2022) 106127_



interactions, disease-disease interactions, and drug-disease associa­
tions). In this way, the topological subnets can be constructed without
unavailable associations in our heterogeneous network such as the drugpathway associations. Given the heterogeneous network **G**, the topo­
logical decomposition can be formulated as:


**G** = { **G** _[R]_ [−] _[D]_ _,_ **G** _[R]_ [−] _[P]_ _,_ **G** _[P]_ [−] _[G]_ _,_ **G** _[G]_ [−] _[M]_ _,_ **G** _[M]_ [−] _[D]_ [}] (16)


Given 2 kinds of nodes _ϑ_ 1 and _ϑ_ 2, the adjacency matrix _A_ _[ϑ]_ [1] [−] _[ϑ]_ [2] ∈

R [(] _[N]_ _[ϑ]_ [1] [ +] _[N]_ _[ϑ]_ [2] [ )×(] _[N]_ _[ϑ]_ [2] [ +] _[N]_ _[ϑ]_ [1] [ )] of the subnet **G** _[ϑ]_ [1] [−] _[ϑ]_ [2 ] can be formulated as:



network can iteratively run on the whole graph. Given two adjoining
nodes _i_ and _j_, the attention weight _α_ _ij_ can be calculated as:



exp( LeakyReLU( _a_ _[T]_ [[] W _H_ _i_ W _H_ _j_
_α_ _ij_ =



exp( LeakyReLU( _a_ _[T]_ [[] W _H_ _i_ W _H_ _j_ ]))

~~∑~~ _k_ ∈ **N** [exp][(][LeakyReLU][(] _[a]_ _[T]_ [[][W] _[H]_ _[i]_ [W] _[H]_



(19)
_k_ ∈ **N** _i_ [exp][(][LeakyReLU][(] _[a]_ _[T]_ [[][W] _[H]_ _[i]_ [W] _[H]_ _[k]_ []))]



where **N** _i_ is the node set which is the neighbors of _i_ ; _H_ _i_ and _H_ _j_ are the

embeddings of node _i_ and _j_ ; W ∈ R [1][×] _[K ]_ and _a_ _[T]_ ∈ R [2][×] _[K ]_ are the learnable

parameter matrixes; and LeakyReLU = { _x_ 0 _,. x_ 001 ≥⋅ _x_ 0 _, x <_ 0 [is the activa­]

tion function.

To enhance the numerical stability of the graph attention network,
we use multi-head attention to generate the output of node embeddings.
Setting _K_ attention heads that are randomly initialized, the output node

embedding _H_ [̂] _i_ with multi-head attention can be formulated as:



_A_ _[ϑ]_ [1] [−] _[ϑ]_ [1] _A_ _[ϑ]_ [1] [−] _[ϑ]_ [2]
_A_ _[ϑ]_ [1] [−] _[ϑ]_ [2] =

[ ( _A_ _[ϑ]_ [1] [−] _[ϑ]_ [2] ) _[T]_ _A_ _[ϑ]_ [2] [−] _[ϑ]_ [2]



(17)
]



Then, each subnet is assembled with a HeteroGCN to learn subnetlevel node embedding. Consequently, each node _υ_ _[ϑ ]_ will acquire two
embedding. For node _ϑ_ 1, the node embedding is generated from the
subnets **G** _[ϑ]_ [1] [−] _[ϑ]_ [2 ] and **G** _[ϑ]_ [1] [−] _[ϑ]_ [3 ] (e.g., the drug embedding includes repre­
sentations from drug-disease subnet and drug-protein subnet). More­
over, the attention mechanism (described in section 2.3.3) is adopted to
dynamically aggregate node embedding, and the attention coefficients
represent the contributions of different subnets learned from REDDA.
The above computation can be formulated as:



(∑ _j_ ∈ **N** _i_



_H_ ̂ _i_ =



_K_
| _k_ =1 _[σ]_
⃒⃒⃒⃒⃒



_α_ _[k]_
_ij_ _[W]_ _[k]_ _[H]_ _[j]_



)



(20)



(18)
])



where _W_ _[k ]_ is the learnable parameter matrix of the _k_ -th attention heads,
and _σ_ is the PReLU activation function.


_2.4.4. Layer attention block_
To alleviate the over-smoothing of the model and connect the loworder/high-order representations, we use a layer attention block to
aggregate each embedding from the previous blocks by attention
mechanism (described in section 2.3.3). The final embedding of drug

nodes _H_ _[R]_ [(] _[Layer Attn]_ [)] ∈ R _[N]_ _[R]_ [×] _[K ]_ and disease nodes _H_ _[D]_ [(] _[Layer Attn]_ [)] ∈ R _[N]_ _[D]_ [×] _[K ]_ can

be formulated as:



_H_ ̂



_ϑ_ 1 = _Attention_ _HeteroGCN_ ( _A_ _[ϑ]_ [1] [−] _[ϑ]_ [2] _, H_ _[ϑ]_ [1] _, W_ _[ϑ]_ [1] [−] _[ϑ]_ [2] )
([ _HeteroGCN_ ( _A_ _[ϑ]_ [1] [−] _[ϑ]_ [3] _, H_ _[ϑ]_ [1] _, W_ _[ϑ]_ [1] [−] _[ϑ]_ [3] )



_2.4.3. Graph attention block_
After generating the subnet-level node embedding, we consider
reconstructing the whole biological network **G** and updating the node
embedding with a graph attention layer. In the graph attention block, **G**
is reconstructed as a homogeneous graph so that the graph attention



5


_Y. Gu et al._ _Computers in Biology and Medicine 150 (2022) 106127_



⎤

⎥⎥⎦



⎞

⎟⎟ (22)
⎠



⎛

⎜
⎜
⎝



⎤

⎥⎥⎦



⎤



⎞

⎟⎟ (21)
⎠



⎞



_H_ _[R]_ [(] _[Layer Attn]_ [)] = _Attention_



⎡

⎢⎢⎣



⎡



⎛

⎜
⎜
⎝



_H_ _[R]_ [(][0][)]

_H_ _[R]_ [(] _[NodeEmb]_ [)]

_H_ _[R]_ [(] _[SubnetEmb]_ [)]

_H_ _[R]_ [(] _[GraphAttn]_ [)]


_H_ _[D]_ [(][0][)]

_H_ _[D]_ [(] _[NodeEmb]_ [)]

_H_ _[D]_ [(] _[SubnetEmb]_ [)]

_H_ _[D]_ [(] _[GraphAttn]_ [)]



⎤



⎞



_H_ _[D]_ [(] _[Layer Attn]_ [)] = _Attention_



⎡

⎢⎢⎣



_2.4.5. Bilinear inner product decoder_
To reconstruct the drug-disease association matrix, we use an inner
product decoder with a linear layer, which is formulated as:


̂
_A_ = _f_ ( _H_ _[R]_ _, H_ _[D]_ [)] = _sigmoid_ ( _H_ _[R]_ _W_ ( _H_ _[D]_ [)] _[T]_ [)] (23)


where _W_ ∈ R _[K]_ [×] _[K ]_ is the trainable parameter matrix and _A_ [̂] is the recon­
structed drug-disease association matrix.


_2.5. Optimization_


We use a weighted cross-entropy loss function to balance the impact
of different categories and help REDDA focus on the known drug-disease
associations. Given _N_ _[R ]_ drugs and _N_ _[D ]_ diseases in a heterogeneous
network, the known/unknown drug-disease associations are labeled as
_S_ [+] and _S_ [−], then the loss function can be formulated as:



( _i,j_ )∈ _S_ [+]



_logA_ [̂] _ij_ + ∑

( _i,j_ )∈ _S_ [−]



1
_Loss_ = −
_N_ _[R]_ × _N_ _[D]_



(



)
)



( 1 − _logA_ [̂] _ij_



_γ_ ∑



(24)



experiments are shown in Figs. S2–S3.


**3. Results**


_3.1. Comparison with state-of-the-art approaches_


We compared REDDA with 8 drug-disease association prediction
methods to demonstrate the effectiveness of our model, including
SCMFDD [24], MBiRW [23], NIMGCN [57], HINGRL-Node2Vec-RF,
HINGRL-DeepWalk-RF [44], LAGCN [34], and DRWBNCF [36]. The
details for the construction of these baseline methods can be found in

supplementary materials.
The performance results of 10-fold cross-validations on our proposed
benchmark dataset are shown in Table 3 and Fig. 2, while the statistical
results of the difference between these performance metrics of REDDA to
those of 8 baseline methods are listed in Tables S3–S4. These results

indicated that our proposed REDDA outperformed the other 8 baseline
methods in terms of the majority of metrics including AUC, AUPR, F1Score, and Recall, on which REDDA achieved relative improvements
of 0.44%, 26.14%, 18.99%, and 26.69% compared to the suboptimal
methods, while the statistical results also proved the significance of
these improvements ( _p_ -values _<_ 0.001). Focusing on the accuracy,
specificity, and precision metrics, the performance of our method is still
comparable. It should also be noted that compared to the other
HeteroGCN-based methods, such as NIMGCN, LAGCN, and DRWBNCF,
our proposed REDDA obtained desirable improvements on the major
metrics (13.97% on AUC, 26.14% on AUPR and 18.99% on F1-Score
compared to the suboptimal method). These results tended to indicate
the effective use of HeteroGCN in REDDA. Compared to HINGRLNode2Vec-RF and HINGRL-DeepWalk-RF, which used extra biological
relations as well and can be regarded as the suboptimal methods among
the 8 models, our method still achieved noticeable relative improve­
ments (0.44% on AUC, 84.23% on AUPR, and 71.28% on F1-Score). We
also calculated the Recall@k metric adopted by some previous studies

[36,37], which means the ratio of correctly predicted drug-disease as­
sociations in the _Topk_ predictions. The Recall@10,000 curves shown in
Fig. 2 indicated that REDDA achieved the best Recall@10,000 values
compared to 8 methods (Top3 results: 66.68% in REDDA; 51.63% in
DRWBNCF; and 49.78% in HINGRL-Node2Vec-RF).
When facing a sparse biological network where the known drugdisease associations are insufficient, the methods need to learn from
numerous unknown drug-disease associations and external biological
information to maintain stability and robustness. As the previous study
mentioned [34], we randomly removed part of known drug-disease as­
sociations in our benchmark datasets at a ratio _r_ ∈{0% _,_ 5% _,_ 10% _,_ 15% _,_
20%}, and trained REDDA and other 5 models (NIMGCN,
HINRGL-Node2Vec-RF, HINGRL-DeepWalk-RF, LAGCN, and
DRWBNCF) on them. The AUC and AUPR results of 10-fold
cross-validations on these partially removed datasets are shown in
Fig. 3. The results showed that our proposed REDDA maintained the
state-of-the-art performance among the 6 methods. More inspiringly,
REDDA keeps stable AUPR scores while some other methods got sig­
nificant decreases (e.g., NIMCGCN, LAGCN, and DRWBNCF). These re­
sults demonstrated that REDDA was relatively insensitive to the
proportion of known drug-disease associations of the training set,
implying a potential of REDDA for sparse biological network-based drug
repositioning.
We also accomplished a performance comparison experiment on the
B-dataset to emphasize the solidarity of REDDA on public benchmark.
The performance results and curves of REDDA and 4 baseline methods
(NIMCGCN, HINGRL-DeepWalk-RF, LAGCN, and DRWBNCF) are shown
in Table 4 and Fig. 4, while the p-values of performance comparisons are
listed in Tables S5–S6. We observed that REDDA also performed best
among the 5 methods. In each metric comparison, REDDA respectively
achieved a relative improvement of 2.48%, 4.93%, 4.35%, 1.14%,
0.18%, 1.20%, and 8.06% compared to the suboptimal method



where _γ_ = [|] | _[S]_ _S_ [−][+] [|] | [is the balance weight, ] [|] _[S]_ [+] [|][ and ] [|] _[S]_ [−] [|][ are the number of ]

known/unknown drug-disease associations in the training set, and _A_ [̂] _ij_ is
the predicted probability of drug _i_ and disease _j_ .
Following previous studies [35], we use the Adam optimizer for
model optimization and initialize the trainable parameters in each layer
by Xavier [56]. Moreover, the dropout layer and batch normalization
layer are also adopted to inhibit overfitting. To enhance the stability of
the training process, a cyclic learning scheduler is also used to dynam­
ically adjust the learning rate during the training process.


_2.6. Experimental settings_


To estimate the performance of REDDA and other drug repositioning
algorithms, we execute 10-fold cross-validations on each model and
repeated them 10 times to decrease the random error caused by data
splitting.
Considering the label imbalance in drug-disease association predic­
tion which need to focus more on the predictions of confirmed associ­
ations, we use multiple metrics to evaluate the performance of REDDA
and other proposed models, including AUC, AUPR, F1-Score, Accuracy,
Recall, Specificity, and Precision which are calculated by the process
proposed by Ref. [34].


_2.7. Hyper-parameter setting_


For the model-related hyper-parameters in REDDA, our proposed
REDDA algorithm is assembled by a sequential architecture (HeteroGCN

- _>_ HeteroGCN- _>_ GAT) in the encoder and a fully connected layer in the
decoder. Each layer has 128 hidden units, a dropout rate of 0.4, and
batch normalization. For the training-related hyper-parameters in
REDDA, we set the maximum and minimum learning rates to 0.01 and
0.001 in the cyclic learning rate scheduler and the number of training
epochs to 4000.
We used a grid search to determine these above hyper-parameters
throughout the experiments. The results of the grid search



6


_Y. Gu et al._ _Computers in Biology and Medicine 150 (2022) 106127_


**Table 3**

Performance comparison of 9 methods on our proposed benchmark dataset.

Methods AUC AUPR F1-Score Accuracy Recall Specificity Precision


SCMFDD 0.783 ± 0.000 0.045 ± 0.000 0.094 ± 0.000 0.989 ± 0.000 0.136 ± 0.000 0.993 ± 0.000 0.072 ± 0.000

MBiRW 0.678 ± 0.001 0.017 ± 0.000 0.036 ± 0.000 0.922 ± 0.005 0.218 ± 0.013 0.020 ± 0.000 **0.927 ± 0.005**

DDA-SKF 0.806 ± 0.001 0.121 ± 0.003 0.165 ± 0.002 0.985 ± 0.000 0.226 ± 0.007 0.990 ± 0.001 0.130 ± 0.003

NIMGCN 0.668 ± 0.084 0.181 ± 0.034 0.260 ± 0.036 0.993 ± 0.000 0.197 ± 0.033 0.998 ± 0.001 0.388 ± 0.056

HINGRL-Node2Vec-RF 0.915 ± 0.002 0.241 ± 0.017 0.289 ± 0.008 0.992 ± 0.000 0.242 ± 0.015 0.997 ± 0.000 0.360 ± 0.024

HINGRL-DeepWalk-RF 0.918 ± 0.002 0.241 ± 0.010 0.283 ± 0.012 0.992 ± 0.001 0.248 ± 0.030 0.997 ± 0.001 0.341 ± 0.045

LAGCN 0.809 ± 0.010 0.247 ± 0.009 0.223 ± 0.019 0.983 ± 0.002 0.356 ± 0.012 0.988 ± 0.002 0.163 ± 0.019

DRWBNCF 0.790 ± 0.001 0.352 ± 0.003 0.416 ± 0.004 **0.994 ± 0.000** 0.347 ± 0.017 **0.998 ± 0.000** 0.524 ± 0.036

REDDA **0.922 ± 0.003** **0.444 ± 0.008** **0.495 ± 0.009** **0.994 ± 0.000** **0.451 ± 0.024** **0.998 ± 0.000** 0.552 ± 0.035


Footnote: The best results in each column are in **bold faces** and the second-best results are underlined.


**Fig. 2.** The AUC, AUPR, and Recall@10,000 curves of 9 methods on our proposed benchmark dataset.


**Fig. 3.** The AUC and AUPR scores of 6 methods on 5 partially removed datasets.



(LAGCN), while the statistical results also emphasized these remarkable
improvements. Moreover, the Recall@10,000 value of REDDA is
significantly outperforming those of 4 baseline methods (Top3 results:
35.93% in REDDA; 32.89% in LAGCN, and 32.25% in DRWBNCF).
These inspiring results implied that the high-order representations of
drug-disease associations can benefit from fusing biological relations (e.



g., drug-protein associations and pathway-disease associations). More­
over, the way we utilized this information in REDDA (topological subnet
embedding block and graph attention block) may also partially account
for the considerable performance improvement.



7


_Y. Gu et al._ _Computers in Biology and Medicine 150 (2022) 106127_


**Table 4**

Performance comparison of 5 methods on B-dataset.

Methods AUC AUPR F1-Score Accuracy Recall Specificity Precision


NIMCGCN 0.675 ± 0.007 0.238 ± 0.009 0.295 ± 0.007 0.764 ± 0.012 0.430 ± 0.013 0.807 ± 0.015 0.224 ± 0.009

HINGRL-DeepWalk-RF 0.809 ± 0.051 0.401 ± 0.101 0.439 ± 0.067 0.843 ± 0.032 0.529 ± 0.040 0.883 ± 0.031 0.379 ± 0.080

LAGCN 0.848 ± 0.023 0.521 ± 0.019 0.506 ± 0.012 0.874 ± 0.004 0.564 ± 0.019 0.914 ± 0.004 0.459 ± 0.012

DRWBNCF 0.848 ± 0.001 0.477 ± 0.003 0.490 ± 0.002 0.868 ± 0.005 0.555 ± 0.021 0.908 ± 0.009 0.440 ± 0.015

REDDA **0.869 ± 0.002** **0.548 ± 0.007** **0.528 ± 0.004** **0.884 ± 0.004** **0.565 ± 0.013** **0.925 ± 0.005** **0.496 ± 0.013**


Footnote: The best results in each column are in **bold faces** and the second-best results are underlined.


**Fig. 4.** Performance comparison of 5 methods on B-dataset.



_3.2. Ablation study_


To demonstrate the importance and necessity of the architecture of
REDDA, we designed a series of ablation experiments by proposing and
testing some variants of REDDA that can be regarded as simplified
REDDA. The details of these variants of REDDA are listed as follows:


- REDDA-Base: The REDDA only includes the node embedding block
and bilinear dot decoder.

- REDDA-w/o BR: Basic REDDA architecture, but the input is a drugdisease heterogeneous network (without protein, gene, and
pathway relations involved).

- REDDA-w/o TSE: The REDDA without the topological subnet
embedding block.

- REDDA-w/o GAT: The REDDA without the graph attention block.

- REDDA-w/o LA: The REDDA without the layer attention block.


The performance comparison results of REDDA and its 5 simplified
versions on our proposed benchmark dataset are listed in Table 5.
Compared to these simplified models, REDDA achieved the optimal AUC
score, AUPR score, and the suboptimal F1-Score, which can be regarded
as the best-performed model. Among the simplified models, the REDDABase and REDDA-w/o BR both under-performed than REDDA, indicating
that a well-designed model structure and an integration of biological
relations both can contribute to better performance in drug reposition­
ing tasks; Compared REDDA-w/o TSE and REDDA-w/o GAT to REDDA

**Table 5**
Performance comparison of REDDA and 5 simplified methods.


Methods AUC AUPR F1-Score


REDDA-Base 0.887 ± 0.004 0.357 ± 0.000 0.421 ± 0.004

REDDA-w/o BR 0.914 ± 0.000 0.425 ± 0.002 0.475 ± 0.002

REDDA-w/o TSE 0.889 ± 0.001 0.359 ± 0.016 0.418 ± 0.010

REDDA-w/o GAT 0.893 ± 0.001 0.441 ± 0.000 **0.500 ± 0.002**

REDDA-w/o LA 0.713 ± 0.007 0.021 ± 0.003 0.042 ± 0.007

REDDA **0.922 ± 0.003** **0.444 ± 0.008** 0.495 ± 0.009


Footnote: The best results in each column are in **bold faces** and the second-best

results are underlined.



Base, respectively, the performance results implied that the topological
subnet embedding block and graph attention block both can benefit the
model performance, which tended to efficiently utilize the biological
information. The REDDA-w/o LA performed worst due to the deep
neural network, which required the residual architecture like layer
attention block to resolve potential overfitting. These results demon­
strated the reasonability of the REDDA structure.


_3.3. Attention visualization analysis_


In REDDA, 3 attention mechanisms are adopted to enhance the
representation: subnet attention, layer attention, and graph attention
mechanisms. In this section, we used attention visualization analysis to
discuss these attention mechanisms.


_3.3.1. Topological subnet attention_
Subnet Attention is used for weighted aggregating the representa­
tions from different topological subnets. The auto-learned attention
coefficients in REDDA are shown in Fig. 5A. The results showed that
each topological subnet played a role in generating the node represen­
tations. Particularly, the integration of biological relations such as **G** _[R]_ [−] _[P ]_

and **G** _[M]_ [−] _[D]_, had profound impacts on learning drug and disease repre­
sentations in topological subnet embedding block, which got attention
coefficients rather than 0.6. It also cannot be ignored that the attention
coefficients of topological subnets for the same node are different but
not extreme (compared to the extremely unbalanced number of associ­
ations such as drug-disease and drug-protein associations), which ten­
ded to show that the topological subnet embedding block can partly
prevent the generation of biased node representations.


_3.3.2. Graph attention_
Graph attention coefficients represent the importance of the edges
which are connected to a certain node and other neighbor nodes. We
plotted the distributions of attention coefficient of each edge type for
each node type (shown in Fig. 5B), which indicated that the graph
attention block aggregated the node representations with varied atten­
tion coefficients. Moreover, in drug and disease nodes representations,



8


_Y. Gu et al._ _Computers in Biology and Medicine 150 (2022) 106127_


**Fig. 5. Attention visualization analysis of REDDA. A.** Attention coefficients for 5 topological subnets in the node representations in topological subnet embedding
block. **B.** **C.**
Attention coefficients for 10 edge types in the node representations in graph attention block. Attention coefficients for the neighbors of node DB00248 in
graph attention block. **D.** Attention coefficients for the node representations from 4 blocks in REDDA.



the drug-protein associations and pathway-disease associations also
occupied portions of attention coefficients, which indicated the inte­
gration of extra biological nodes and relations provided additional valid
information for the node representation learning.
We also selected a drug node DB00248 (generic name: cabergoline)
as a case and visualized the attention coefficients of its edges, which are
shown in Fig. 5C. In the neighbors of DB00248 which is used for the
treatment of hyperprolactinemia, REDDA gave more attention to the
fewer associated disease neighbors. Moreover, REDDA can also recog­
nize the most related nodes to DB00248, such as the target diseases
galactorrhea (MeSH ID: D005687) and prolactinoma (MeSH ID:
D015175).


_3.3.3. Layer attention_
We adopted the residual structure by aggregating representations
with a layer attention mechanism. The attention coefficients represented
the importance contributions of different blocks to the final drug/dis­
ease representations (shown in Fig. 5D). The results showed that the
proposed computational blocks in REDDA all occupied portions of
attention coefficients in the layer attention block. Compared to the loworder ones, the high-order representations in the topological subnet
embedding block and graph attention block tended to focus on different
views of nodes and provided considerable contributions to the final node
representations (the sum of attention coefficient ≈ 0.55).
The above all attention visualizations demonstrated that REDDA can

selectively identify the important information in the heterogeneous
network for drug-disease association predictions and generated node
representations by weighted aggregations on the subnet/graph/layer
levels with varied attentions.



_3.4. Case study_


To demonstrate the ability of REDDA for discovering new indications
and new therapies, we trained a REDDA model using all known drugdisease associations and predicted the unknown drug-disease associa­
tions for known drugs/diseases. As for verifications, we used two reli­
able public databases (ClinicalTrials and CTD) to emphasize the
accuracy of our predictions. Moreover, the experimental results reported
in references were also additional evidence to verify the ability to
discover new indications/therapies of MOODA.


_3.4.1. Discovery of new indications_
We selected two drugs to investigate the ability of REDDA in
discovering new indications. Except for the known drug-disease asso­
ciations, the Top10 positive predictions of these drugs are listed in
Table 6, while the details of predicted associated drugs are listed in
Table S6. Among the tested drugs, Ifosfamide (DrugBank ID: DB01181)
is an alkylating and immunosuppressive agent which is widely used as a
chemotherapeutics for tumor treatment, including the treatment of nonHodgkin’s lymphoma, small cell lung cell, etc. Among the Top10 posi­
tive predictions of REDDA, 9 of the associated diseases are tumors and
have been confirmed by databases. Dexamethasone (DrugBank ID:
DB01234) is a glucocorticoid that is used for the treatment of inflam­
matory conditions. In Tables 6 and 8 of the Top10 associated diseases
have been confirmed by reliable sources or clinical trials. These pieces of
evidence indicate that REDDA can learn from limited biological infor­
mation and identify confirmed indications that not have been collected
in the training dataset.
Furthermore, as for the discovery of new potential indications, one of



9


_Y. Gu et al._ _Computers in Biology and Medicine 150 (2022) 106127_



**Table 6**

The Top10 REDDA-predicted new indications.


Drug Disease MeSH ID Evidence Evidence
(Database) (PMID)


Dexamethasone Nasal Polyps D009298 CTD 24917907
Inflammatory D015212 ClinicalTrials/ 29109767
Bowel Diseases CTD


Rhinitis, Allergic D065631 CTD 17762268
Spondylarthritis D025241 CTD N/A

Otitis Media D010033 ClinicalTrials 24093464

Erythema D004892 N/A N/A

Multiforme


Pityriasis Rosea D017515 N/A N/A
Adrenal D000312 ClinicalTrials/ 14764770

Hyperplasia, CTD
Congenital

Brain Neoplasms D001932 ClinicalTrials/ 31346902

CTD


Crohn Disease D003424 ClinicalTrials/ 17635367

CTD


Ifosfamide Mycosis D009182 ClinicalTrials/ 3130316
Fungoides CTD

Colitis, Ulcerative D003093 CTD N/A
Lymphoma, D008224 ClinicalTrials/ 12736225

Follicular CTD


Lymphoma, T- D016411 ClinicalTrials/ 33147935
Cell, Peripheral CTD

Lymphoma, D020522 ClinicalTrials 20038221

Mantle-Cell


Tuberculosis D014376 N/A N/A

Burkitt D002051 ClinicalTrials/ 12181251

Lymphoma CTD

Hepatitis C D006526 CTD N/A
Hyperthyroidism D006980 CTD N/A
Carcinoma, Non- D002289 ClinicalTrials/ 10582135
Small-Cell Lung CTD


the predicted associated diseases of dexamethasone, erythema multi­
forme, is non-confirmed by databases. However, related studies have
mentioned that two rare skin diseases, called Stevens-Johnson Syn­
drome (SJS) and Toxic Epidermal Necrolysis (TEN), are recognized as
the descendants of erythema multiforme in MeSH categories, can be
treated by some corticosteroids including dexamethasone [58,59].
Meanwhile, associated with tissue damage and necroptosis, the Annexin
A1 receptor which is mediated by dexamethasone and exists in our
heterogeneous network, is also a candidate marker in SJS/TEN reported
by a previous study [60]. Overall, these indirect shreds of evidence
enhance the reliability of the new unconfirmed potential indication
discovered by REDDA.


**Table 7**

The Top10 REDDA-predicted new therapies.



_3.4.2. Discovery of new therapies_
We selected two diseases (breast neoplasms and brain neoplasms) to
investigate the ability of REDDA in discovering new therapies. Same as
the details in the 3.4.1 Section, the Top10 positive predictions results are
listed in Table 7 and the details are listed in Table S8. In the Top10
results of breast neoplasms (MeSH ID: D001943), 8 of the Top10 pre­
dicted drug-disease associations are experimentally confirmed by data­
bases. Among them, 5 of these associated drugs are hormones
(norethisterone, estrone, leuprolide, testosterone cypionate, and gona­
dorelin), which are highly related to breast neoplasms. Focusing on the
results of brain neoplasms (MeSH ID: D001932), 5 of the Top10 results
are confirmed by databases. Among them, 4 associated drugs are anti­
neoplastic agents and 5 are corticosteroids. These pieces of evidence
demonstrate that our proposed REDDA can also predict the confirmed
therapies for existing diseases. Meanwhile, comparing the associated
drugs of breast neoplasms to those of brain neoplasms, it was inspiring to
observe that REDDA tended to selectively generate the prediction based
on the characteristic of the disease. These results demonstrate that

REDDA has the potential to identify therapies by knowledge-aware
prediction.
For the discovery of new therapies which are not confirmed by da­
tabases, the predicted gonadorelin for breast neoplasms is a synthetic
gonadotropin-releasing hormone (GnRH) and also a GnRH agonist
(GnRHa). Some previous studies indicated that GnRHa can be used for
the treatment of early breast neoplasms, infertility caused by breast
neoplasms and metastatic male breast neoplasms [61–63], etc. These
shreds of evidence emphasize the capacity of gonadorelin for the
treatment of breast neoplasms, which have not been mentioned by
previous studies or related databases. Moreover, a predicted drug for
brain neoplasms, ranimustine has been used in the treatment of chronic
myelogenous leukemia, Peripheral T-cell lymphoma, and glioblastoma
multiforme [64–66]. Meanwhile, the Top3 drugs which are the most
similar to ranimustine in our heterogeneous network (carmustine,
lomustine, and streptozocin) can all be used for the treatment of brain
neoplasms, which may account for the high association probability of
ranimustine predicted by REDDA. Other 2 predicted associated drugs
(clobetasol propionate and etretinate) for brain neoplasms are also
confirmed for the treatment of other cancers or cancer-related compli­
cations [67,68], which provided potentials for the therapies of brain
neoplasms.
In a nutshell, the case studies prove that REDDA can learn from the
large-scale heterogeneous network to recognize the drug-disease asso­
ciations which are unknown in the training set but confirmed in the realworld. Meanwhile, REDDA can also give confident predictions to some



Disease Drug DrugBank ID Evidence (Database) Evidence (PMID)


Breast Neoplasms Zoledronic acid DB00399 ClinicalTrials/CTD 29082518
Norethisterone DB00717 N/A N/A

Estrone DB00655 CTD 12796390

Leuprolide DB00007 ClinicalTrials/CTD 29747931
Testosterone cypionate DB13943 ClinicalTrials 26160683
Metformin DB00331 ClinicalTrials/CTD 31164151

Gonadorelin DB00644 N/A 32006118

Prednisone DB00635 ClinicalTrials/CTD 27052658

Sirolimus DB00877 ClinicalTrials/CTD 32335491

Risedronic acid DB00884 ClinicalTrials/CTD 25792492

Brain Neoplasms Everolimus DB01590 ClinicalTrials/CTD 34224367
Flurandrenolide DB00846 N/A N/A

Ranimustine DB13832 N/A 28214639

Mitoxantrone DB01204 CTD 12450040

Betamethasone DB00443 CTD 869982
Difluprednate DB06781 N/A N/A
Etretinate DB00926 N/A N/A

Paclitaxel DB01229 ClinicalTrials/CTD 21692650

Clobetasol propionate DB01013 N/A N/A
Hydrocortisone DB00741 ClinicalTrials/CTD 34214336


10


_Y. Gu et al._ _Computers in Biology and Medicine 150 (2022) 106127_



unconfirmed but potential drug-disease associations, implying that
REDDA has a noticeable ability for discovering new indications/thera­
pies for existing drugs/diseases of REDDA.


**4. Conclusion**


In this study, we propose a benchmark dataset for drug-disease as­
sociation prediction. Much larger than a single drug-disease heteroge­
neous network, the constructed comprehensive heterogeneous network
contains 5 biological entities and 10 relations. Moreover, to enhance the
effectiveness of these extra biological relations on improving the per­
formance of computational drug repositioning, a graph learning method
(REDDA) on a heterogeneous network is proposed for predicting drugdisease associations, which is designed and constructed on the basis of
HeteroGCNs, attention mechanisms, and topological disintegrations.
Unlike to other methods which can only be used in isolated drug-disease
heterogeneous network, or calculating multi-view similarities using
extra biological relationships, our proposed REDDA can be directly
trained on a heterogeneous network with abundant biological relations,
which can enhance the representations of drugs and diseases, and
decrease the secondary transformation loss of biological information.
Extensive experiments emphasized the effectiveness of the fused extra
biological information and the superiority of REDDA compared to
various baseline methods.

Although REDDA has achieved desirable performance. Some limi­
tations still need to be in-depth considered. Firstly, more biological as­
sociations should be further concerned which are inaccessible in our

heterogeneous network, such as the protein-disease associations, drugmiRNA associations, miRNA-disease associations, etc. Secondly, for
the model structure, more advanced backbone heterogeneous neural
networks should be involved in REDDA where we only use the classical
HeteroGCN in this version. In the future, we will concentrate on inte­
grating larger benchmark datasets with multiple data sources to spur the
isolated drug repositioning to large-scale biological network aided drug
repositioning. Meanwhile, the model interpretability in the computa­
tional drug repositioning method is another concern of us.
In conclusion, our proposed REDDA can enhance the ability to pre­
dict drug-disease associations by using external biological information
and dynamically aggregating the drug/disease representations. With a
promising capacity, it may be used in computational drug repositioning
for accelerating the discovery of positive entities and reducing the cost
of time-consuming wet-lab experiments.


**Funding**


This work was supported by Chinese Academy of Medical Sciences
(Grant No. 2021-I2M-1–056), Fundamental Research Funds for the
Central Universities (Grant No. 3332022144), National Key R&D Pro­
gram of China (Grant No. 2016YFC0901901 and Grant No.
2017YFC0907503) and the National Natural Science Foundation of
China (Grant No. 81601573).


**Declaration of competing interest**


The authors declare that they have no known competing financial
interests or personal relationships that could have appeared to influence
the work reported in this paper.


**Acknowledgments**


The authors would like to thank all anonymous reviewers for their
constructive advice.


**Appendix A. Supplementary data**


[Supplementary data to this article can be found online at https://doi.](https://doi.org/10.1016/j.compbiomed.2022.106127)



[org/10.1016/j.compbiomed.2022.106127.](https://doi.org/10.1016/j.compbiomed.2022.106127)


**References**


[[1] H.C.S. Chan, H. Shan, T. Dahoun, et al., Advancing drug discovery via artificial](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref1)
[intelligence, Trends Pharmacol. Sci. 40 (2019) 592–604.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref1)

[[2] V. Prasad, S. Mailankody, Research and development spending to bring a single](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref2)
[cancer drug to market and revenues after approval, JAMA Intern. Med. 177 (2017)](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref2)
[1569–1575.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref2)

[[3] J.A. DiMasi, H.G. Grabowski, R.W. Hansen, Innovation in the pharmaceutical](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref3)
[industry: new estimates of R&D costs, J. Health Econ. 47 (2016) 20–33.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref3)

[[4] C.H. Wong, K.W. Siah, A.W. Lo, Estimation of clinical trial success rates and related](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref4)
[parameters, Biostatistics 20 (2019) 273–286.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref4)

[[5] J. Vamathevan, D. Clark, P. Czodrowski, et al., Applications of machine learning in](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref5)
[drug discovery and development, Nat. Rev. Drug Discov. 18 (2019) 463–477.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref5)

[[6] K. Yang, X. Zhao, D. Waxman, et al., Predicting drug-disease associations with](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref6)
[heterogeneous network embedding, Chaos 29 (2019), 123109.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref6)

[[7] Y. Gu, S. Zheng, J. Li, CurrMG: a curriculum learning approach for graph based](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref7)
[molecular property prediction, in: 2021 IEEE International Conference on](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref7)
[Bioinformatics and Biomedicine (BIBM), 2021, pp. 2686–2693.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref7)

[[8] Q. Ye, C.Y. Hsieh, Z. Yang, et al., A unified drug-target interaction prediction](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref8)
[framework based on knowledge graph and recommendation system, Nat. Commun.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref8)
[12 (2021) 6775.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref8)

[[9] Y. Gu, S. Zheng, Z. Xu, et al., An efficient curriculum learning-based strategy for](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref9)
[molecular graph learning, Briefings Bioinf. 23 (2022) bbac099.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref9)

[[10] W. Kong, X. Tu, W. Huang, et al., Prediction and optimization of NaV1. 7 sodium](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref10)
[channel inhibitors based on machine learning and simulated annealing, J. Chem.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref10)
[Inf. Model. 60 (2020) 2739–2753.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref10)

[[11] T. Li, X. Zhao, L. Li, Co-VAE: drug-target binding affinity prediction by co-](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref11)
[regularized variational autoencoders, IEEE Trans. Pattern Anal. Mach. Intell.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref11)
[(2021) 1, 1.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref11)

[[12] Q. Liu, Z. Hu, R. Jiang, et al., DeepCDR: a hybrid graph convolutional network for](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref12)
[predicting cancer drug response, Bioinformatics 36 (2020) i911–i918.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref12)

[[13] F. Zhang, M. Wang, J. Xi, et al., A novel heterogeneous network-based method for](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref13)
[drug response prediction in cancer cell lines, Sci. Rep. 8 (2018) 3355.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref13)

[[14] J. Li, S. Zheng, B. Chen, et al., A survey of current trends in computational drug](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref14)
[repositioning, Briefings Bioinf. 17 (2016) 2–12.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref14)

[[15] H. Xue, J. Li, H. Xie, et al., Review of drug repositioning approaches and resources,](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref15)
[Int. J. Biol. Sci. 14 (2018) 1232–1244.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref15)

[[16] K. Mohamed, N. Yazdanpanah, A. Saghazadeh, et al., Computational drug](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref16)
[discovery and repurposing for the treatment of COVID-19: a systematic review,](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref16)
[Bioorg. Chem. 106 (2021), 104490.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref16)

[[17] G. Fahimian, J. Zahiri, S.S. Arab, et al., RepCOOL: computational drug](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref17)
[repositioning via integrating heterogeneous biological networks, J. Transl. Med. 18](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref17)
[(2020) 375.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref17)

[[18] J.I. Traylor, H.E. Sheppard, V. Ravikumar, et al., Computational drug repositioning](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref18)
[identifies potentially active therapies for chordoma, Neurosurgery 88 (2021)](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref18)
[428–436.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref18)

[[19] L. Bai, M.K.D. Scott, E. Steinberg, et al., Computational drug repositioning of](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref19)
[atorvastatin for ulcerative colitis, J. Am. Med. Inf. Assoc. 28 (2021) 2325–2335.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref19)

[[20] C. Budak, V. Mençik, V. Gider, Determining similarities of COVID-19 - lung cancer](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref20)
[drugs and affinity binding mode analysis by graph neural network-based GEFA](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref20)
[method, J. Biomol. Struct. Dyn. (2021) 1–13.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref20)

[[21] H. Luo, M. Li, M. Yang, et al., Biomedical data and computational models for drug](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref21)
[repositioning: a comprehensive review, Briefings Bioinf. 22 (2021) 1604–1619.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref21)

[[22] C.-Q. Gao, Y.-K. Zhou, X.-H. Xin, et al., DDA-SKF, Predicting drug-disease](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref22)
[associations using similarity kernel fusion, Front. Pharmacol. 12 (2022) 784171,](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref22)
[784171.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref22)

[[23] H. Luo, J. Wang, M. Li, et al., Drug repositioning based on comprehensive](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref23)
[similarity measures and Bi-Random walk algorithm, Bioinformatics 32 (2016)](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref23)
[2664–2671.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref23)

[[24] W. Zhang, X. Yue, W. Lin, et al., Predicting drug-disease associations by using](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref24)
[similarity constrained matrix factorization, BMC Bioinf. 19 (2018) 233.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref24)

[[25] M. Yang, H. Luo, Y. Li, et al., Drug repositioning based on bounded nuclear norm](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref25)
[regularization, Bioinformatics 35 (2019) i455–i463.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref25)

[[26] M. Yang, G. Wu, Q. Zhao, et al., Computational drug repositioning based on multi-](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref26)
[similarities bilinear matrix factorization, Briefings Bioinf. 22 (2021) bbaa267.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref26)

[[27] Q. Cao, Z. Zhang, A.X. Fu, et al., A unified framework for integrative study of](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref27)
[heterogeneous gene regulatory mechanisms, Nat. Mach. Intell. 2 (2020) 447–456.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref27)

[[28] W. Zeng, J. Xin, R. Jiang, et al., Reusability report: compressing regulatory](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref28)
[networks to vectors for interpreting gene expression and genetic variants, Nat.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref28)
[Mach. Intell. 3 (2021) 576–580.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref28)

[[29] Q. Liu, S. Chen, R. Jiang, et al., Simultaneous deep generative modeling and](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref29)
[clustering of single cell genomic data, Nat. Mach. Intell. 3 (2021) 536–544.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref29)

[[30] X. Chen, S. Chen, S. Song, et al., Cell type annotation of single-cell chromatin](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref30)
[accessibility data via supervised Bayesian embedding, Nat. Mach. Intell. 4 (2022)](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref30)
[116–126.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref30)

[[31] P. Zeng, J. Wangwu, Z. Lin, Coupled co-clustering-based unsupervised transfer](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref31)
[learning for the integrative analysis of single-cell genomic data, Briefings Bioinf.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref31)
[(2021) 22.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref31)

[[32] X. Huang, Y. Huang, Cellsnp-lite: an efficient tool for genotyping single cells,](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref32)
[Bioinformatics 37 (23) (2021) 4569–4571.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref32)

[[33] J. Zhu, J. Wang, X. Wang, et al., Prediction of drug efficacy from transcriptional](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref33)
[profiles with deep learning, Nat. Biotechnol. 39 (2021) 1444–1452.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref33)



11


_Y. Gu et al._ _Computers in Biology and Medicine 150 (2022) 106127_




[[34] Z. Yu, F. Huang, X. Zhao, et al., Predicting drug-disease associations through layer](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref34)
[attention graph convolutional network, Briefings Bioinf. (2021) 22.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref34)

[[35] L. Cai, C. Lu, J. Xu, et al., Drug repositioning based on the heterogeneous](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref35)
[information fusion graph convolutional network, Briefings Bioinf. (2021) 22.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref35)

[[36] Y. Meng, C. Lu, M. Jin, et al., A weighted bilinear neural collaborative filtering](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref36)
[approach for drug repositioning, Briefings Bioinf. 23 (2) (2022), bbab581.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref36)

[[37] X. Zeng, S. Zhu, X. Liu, et al., deepDR: a network-based deep learning approach to](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref37)
[in silico drug repositioning, Bioinformatics 35 (2019) 5191–5198.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref37)

[[38] P. Xuan, Y. Ye, T. Zhang, et al., Convolutional Neural Network and Bidirectional](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref38)
[Long Short-Term Memory-Based Method for Predicting, vol. 8, Drug-Disease](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref38)
[Associations, 2019. Cells.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref38)

[[39] H. Liu, W. Zhang, Y. Song, et al., HNet-DNN: inferring new drug-disease](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref39)
[associations with deep neural network based on heterogeneous network features,](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref39)
[J. Chem. Inf. Model. 60 (2020) 2367–2376.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref39)

[[40] P. Xuan, L. Gao, N. Sheng, et al., Graph convolutional autoencoder and fully-](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref40)
[connected autoencoder with attention mechanism based method for predicting](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref40)
[drug-disease associations, IEEE J Biomed Health Inform 25 (2021) 1793–1804.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref40)

[[41] M. Cos¸kun, M. Koyutürk, Node similarity based graph convolution for link](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref41)
[prediction in biological networks, Bioinformatics 37 (2021) 4501–4508.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref41)

[[42] H. Zhang, H. Cui, T. Zhang, et al., Learning multi-scale heterogenous network](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref42)
[topologies and various pairwise attributes for drug–disease association prediction,](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref42)
[Briefings Bioinf. 23 (2022) bbac009.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref42)

[[43] P. Xuan, X. Meng, L. Gao, et al., Heterogeneous multi-scale neighbor topologies](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref43)
[enhanced drug–disease association prediction, Briefings Bioinf. 23 (2022)](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref43)
[bbac123.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref43)

[[44] B.W. Zhao, L. Hu, Z.H. You, et al., HINGRL: predicting drug-disease associations](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref44)
[with graph representation learning on heterogeneous information networks,](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref44)
[Briefings Bioinf. 23 (2022).](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref44)

[[45] J. Huang, J. Chen, B. Zhang, et al., Evaluation of gene-drug common module](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref45)
[identification methods using pharmacogenomics data, Briefings Bioinf. (2021) 22.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref45)

[[46] J. Wang, Z. Wu, Y. Peng, et al., Pathway-based drug repurposing with DPNetinfer:](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref46)
[a method to predict drug-pathway associations via network-based approaches,](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref46)
[J. Chem. Inf. Model. 61 (2021) 2475–2485.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref46)

[[47] A. Gottlieb, G.Y. Stein, E. Ruppin, et al., PREDICT: a method for inferring novel](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref47)
[drug indications with application to personalized medicine, Mol. Syst. Biol. 7](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref47)
[(2011) 496.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref47)

[[48] M. Kanehisa, S. Goto, KEGG: kyoto encyclopedia of genes and genomes, Nucleic](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref48)
[Acids Res. 28 (2000) 27–30.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref48)

[[49] D.S. Wishart, C. Knox, A.C. Guo, et al., DrugBank: a knowledgebase for drugs, drug](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref49)
[actions and drug targets, Nucleic Acids Res. 36 (2008) D901–D906.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref49)

[[50] A.P. Davis, C.J. Grondin, R.J. Johnson, et al., Comparative toxicogenomics](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref50)
[database (CTD): update 2021, Nucleic Acids Res. 49 (2021) D1138–d1143.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref50)

[[51] C. von Mering, M. Huynen, D. Jaeggi, et al., STRING: a database of predicted](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref51)
[functional associations between proteins, Nucleic Acids Res. 31 (2003) 258–261.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref51)




[[52] UniProt: the universal protein knowledgebase, Nucleic Acids Res. 45 (2017)](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref52)
[D158–d169.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref52)

[[53] D. Rogers, M. Hahn, Extended-connectivity fingerprints, J. Chem. Inf. Model. 50](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref53)
[(2010) 742–754.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref53)

[[54] D. Wang, J. Wang, M. Lu, et al., Inferring the human microRNA functional](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref54)
[similarity and functional network based on microRNA-associated diseases,](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref54)
[Bioinformatics 26 (2010) 1644–1650.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref54)

[[55] X. Wang, H. Ji, C. Shi, et al., Heterogeneous graph attention network, in: The](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref55)
[World Wide Web Conference, 2019, pp. 2022–2032.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref55)

[[56] X. Glorot, Y. Bengio, Understanding the difficulty of training deep feedforward](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref56)
[neural networks, in: Proceedings of the Thirteenth International Conference on](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref56)
[Artificial Intelligence and Statistics, 2010, pp. 249–256. JMLR Workshop and](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref56)
[Conference Proceedings.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref56)

[[57] J. Li, S. Zhang, T. Liu, et al., Neural inductive matrix completion with graph](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref57)
[convolutional networks for miRNA-disease association prediction, Bioinformatics](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref57)
[36 (2020) 2538–2546.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref57)

[[58] B.R. Del Pozzo-Magana, A. Lazo-Langner, B. Carleton, et al., A systematic review of](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref58)
[treatment of drug-induced Stevens-Johnson syndrome and toxic epidermal](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref58)
[necrolysis in children, J Popul Ther Clin Pharmacol 18 (2011) e121–e133.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref58)

[[59] S.H. Kardaun, M.F. Jonkman, Dexamethasone pulse therapy for Stevens-Johnson](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref59)
[syndrome/toxic epidermal necrolysis, Acta Derm. Venereol. 87 (2007) 144–148.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref59)

[[60] R. Abe, Immunological response in Stevens-Johnson syndrome and toxic epidermal](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref60)
[necrolysis, J. Dermatol. 42 (2015) 42–48.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref60)

[[61] N.A.J. Khan, M. Tirona, An updated review of epidemiology, risk factors, and](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref61)
[management of male breast cancer, Med. Oncol. 38 (2021) 39.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref61)

[[62] E. Silvestris, M. Dellino, P. Cafforio, et al., Breast cancer: an update on treatment-](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref62)
[related infertility, J. Cancer Res. Clin. Oncol. 146 (2020) 647–657.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref62)

[[63] D.A. Beyer, F. Amari, M. Thill, et al., Emerging gonadotropin-releasing hormone](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref63)
[agonists, Expet Opin. Emerg. Drugs 16 (2011) 323–340.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref63)

[[64] M. Gkotzamanidou, C.A. Papadimitriou, Peripheral T-cell lymphoma: the role of](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref64)
[hematopoietic stem cell transplantation, Crit. Rev. Oncol. Hematol. 89 (2014)](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref64)
[248–261.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref64)

[[65] M. Kameda, Y. Otani, T. Ichikawa, et al., Congenital glioblastoma with distinct](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref65)
[clinical and molecular characteristics: case reports and a literature review, World](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref65)
[Neurosurg 101 (2017) 817, e815-817.e814.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref65)

[[66] N. Kusaba, H. Yoshida, F. Ohkubo, et al., [Granulocyte-colony stimulating factor-](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref66)
[producing myeloma with clinical manifestations mimicking chronic neutrophilic](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref66)
[leukemia], Rinsho Ketsueki 45 (2004) 228–232.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref66)

[[67] A.G. Venkat, S. Arepalli, S. Sharma, et al., Local therapy for cancer therapy-](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref67)
[associated uveitis: a case series and review of the literature, Br. J. Ophthalmol. 104](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref67)
[(2020) 703.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref67)

[[68] K. Yonekura, K. Takeda, N. Kawakami, et al., Therapeutic efficacy of etretinate on](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref68)
[cutaneous-type Adult T-cell leukemia-lymphoma, Acta Derm. Venereol. 99 (2019)](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref68)
[774–776.](http://refhub.elsevier.com/S0010-4825(22)00835-6/sref68)



12


