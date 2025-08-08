[Computers in Biology and Medicine 150 (2022) 105992](https://doi.org/10.1016/j.compbiomed.2022.105992)


[Contents lists available at ScienceDirect](http://www.elsevier.com/locate/compbiomed)

# Computers in Biology and Medicine


[journal homepage: www.elsevier.com/locate/compbiomed](http://www.elsevier.com/locate/compbiomed)

## A computational approach to drug repurposing using graph neural networks


Siddhant Doshi [∗], Sundeep Prabhakar Chepuri


_Indian Institute of Science, Bangalore, 560012, India_



A R T I C L E I N F O


_Keywords:_
Computational pharmacology
Drug repurposing
Drug repositioning
Graph neural networks
Link prediction


**1. Introduction**



A B S T R A C T


Drug repurposing is an approach to identify new medical indications of approved drugs. This work presents
a graph neural network drug repurposing model, which we refer to as GDRnet, to efficiently screen a large
database of approved drugs and predict the possible treatment for novel diseases. We pose drug repurposing
as a link prediction problem in a multi-layered heterogeneous network with about 1.4 million edges capturing
complex interactions between nearly 42,000 nodes representing drugs, diseases, genes, and human anatomies.
GDRnet has an encoder–decoder architecture, which is trained in an end-to-end manner to generate scores for
drug–disease pairs under test. We demonstrate the efficacy of the proposed model on real datasets as compared
to other state-of-the-art baseline methods. For a majority of the diseases, GDRnet ranks the actual treatment
drug in the top 15. Furthermore, we apply GDRnet on a coronavirus disease (COVID-19) dataset and show
that many drugs from the predicted list are being studied for their efficacy against the disease.



Drug repurposing involves strategies to identify new medical indications of approved drugs. It includes identifying potential drugs from
a large database of clinically approved drugs and monitoring their _in_
_vivo_ efficacy and potency against novel diseases. Drug repurposing is a
low-risk strategy as drugs to be screened have already been approved
with less unknown harmful adverse effects and requires less financial
investment compared to discovering new drugs [1]. Some of the successful examples of repurposed drugs in the past are _Sildenafil_, which
was initially developed as an antihypertensive drug and later proved to
be effective also in treating erectile dysfunction [1] and _Rituximab_ that
was originally used against cancer was proved to be effective against
rheumatoid arthritis [1]. Even during the coronavirus disease 2019
(COVID-19) pandemic, caused by the novel severe acute respiratory
syndrome coronavirus ( _SARS-CoV2_ ), which has affected about 450 million people with more than six million deaths worldwide as of February
2022, drug repurposing has been proved very beneficial. Approved
drugs like _Remdesivir_ (a drug for treating Ebola virus disease), _Iver-_
_mectin_ (anthelmintic drug), _Dexamethasone_ (anti-inflammatory drugs)
are being studied for their efficacy against the disease [2–4].
Experimental and computational approaches are usually considered
for identifying the right candidate drugs, which is the most critical step
in drug repurposing. To identify the candidate drugs experimentally, a
variety of chromatographic and spectroscopic techniques are available
for target-based drug discovery. Phenotype screening is used as an
alternative to target-based drug discovery when the identity of the



specific drug target and its role in the disease is not known [1].
Recently, computational approaches for identifying the candidates for
drug repurposing are gaining popularity due to the availability of large
biological data. Efficient ways to handle big data have opened up many
opportunities in the field of pharmacology. For instance, [5] elaborates
several data-driven computational tools using machine learning (ML)
and deep learning (DL) techniques to integrate large volumes of heterogeneous data and solve problems in pharmacology such as drug-target
interaction prediction and drug–drug interaction prediction [6], to list
a few. Drug repurposing has been studied using computational methods such as signature matching methods, molecular docking, matrix
factorization-based, and network-based approaches [7–13]. However,
signature matching approaches and molecular docking approaches rely
highly on knowing profiles and exact structures of the target genes,
that may not be always available. The matrix factorization-based models find new drug–disease interactions by quantifying the similarity
between drugs and disease causative viruses using their molecular
sequences. However, these approaches are restricted to pairwise similarities and fail to capture the interactions at a global level [13].
The network proximity-based methods predict drugs for a disease by
calculating the network proximity scores between the target genes of
the drug and the target genes of the disease [9,10], but these methods
cannot easily account for the additional information in the network,
such as similarities between drugs or diseases. Recently, representation learning techniques (i.e., machine learning and deep learning)
have been gaining attention due to their accelerated and improved



∗ Corresponding author.
_E-mail addresses:_ [siddhant.doshi@outlook.com (S. Doshi), spchepuri@iisc.ac.in (S.P. Chepuri).](mailto:siddhant.doshi@outlook.com)


[https://doi.org/10.1016/j.compbiomed.2022.105992](https://doi.org/10.1016/j.compbiomed.2022.105992)
Received 16 April 2022; Received in revised form 5 August 2022; Accepted 14 August 2022

Available online 31 August 2022
0010-4825/© 2022 Elsevier Ltd. All rights reserved.


_S. Doshi and S.P. Chepuri_


benefits for drug repurposing over the traditional non-deep learning
methods [14,15]. Existing deep learning techniques for drug repurposing can be categorized into sequence-based methods and graph-based
methods [15]. The sequence-based methods use the molecular structural sequences of drugs and the virus genome sequence of diseases
to encode their respective entity-specific information [16]. However,
these methods are highly dependent on the availability of the sequence
information for each entity. Also, these approaches focus on the consecutive one- or two-dimensional correlation in a sequence, but do not
capture the interactions at a global level between different biological
entities. On the other hand, the graph-based approaches capture the
structural connectivity information between different biological entities
and provide more flexible framework for modeling complex biological
interactions between the underlying entities [11,12,17].

A natural and efficient way to capture complex interactions between different biological entities like drugs, genes, diseases, etc., is
to construct a graph with nodes representing entities and edges representing interactions between these entities, e.g., interactions between
drugs and genes or between drugs and diseases. The graph-based
methods such as the deepwalk-based, or graph neural networks, that
are capable of processing such graph structured biological data have
been proposed for drug repurposing [11,12,17]. The deepwalk-based
architecture [17] independently generates the structural information
(using the deepwalk algorithm) and the self entity information due to
which the entity and the relational correspondence is not well captured.
_Graph neural networks_ (GNNs) capture structural information in data
by accounting for interactions between various underlying entities
while processing data associated with them, thus producing meaningful low-dimensional embeddings for the entities that are useful for
downstream machine learning tasks. However, the existing GNN-based
models have a considerable computational overhead when processing
huge biological networks having interactions of high density. In this
work, we address this problem and focus on drug repurposing using
computationally-efficient GNNs. We provide a comparative analysis of
several graph-based architectures for drug repurposing and showcase
the benefits of having a dedicated model through our experiments on

real datasets.


_1.1. Main results and contributions_


We construct a four-layered heterogeneous graph explaining interactions between the four entities, namely, drugs, genes, diseases, and
anatomies in each layer. We propose a new dedicated GNN model
for drug repurposing, called GDRnet, which has an encoder–decoder
architecture. We formulate drug repurposing as a link prediction problem and train GDRnet to predict unknown links between the drug and
disease entities, where a link between a drug–disease entity suggests
that the drug treats the disease. Specifically, the encoder is based on the
scalable inceptive graph neural network (SIGN) architecture [18] for
generating the node embeddings of the entities. We propose a learnable
quadratic norm scoring function as a decoder to rank the predicted
drugs. The proposed norm scorer is particularly designed and tuned
for the drug repurposing task that learns correlations between the drug
and disease pairs. The main contributions and results are summarized

as follows.


  - We formulate drug repurposing as a link prediction problem and

propose a new dedicated GNN-based drug repurposing model. The
trainable encoder of GDRnet precomputes the neighborhood features beforehand, thus, is computationally efficient with reduced
training and inference time. The trainable decoder scores a drug–
disease pair based on the low-dimensional embeddings obtained

from the encoder. The encoder and decoder are trained in an

end-to-end manner.



_Computers in Biology and Medicine 150 (2022) 105992_


  - We validate GDRnet in terms of its link prediction accuracy and

how well it ranks the known treatment drug. For a majority of
diseases with known treatment in the test set, which were not
used while training, GDRnet ranks the approved treatment drugs
in the top 15. This suggests the efficacy of the proposed drug
repurposing model.

  - We perform an ablation study to show the importance of genes

and anatomy entities, which model the indirect interactions between the drug and the disease entities.

  - We provide a detailed computational runtime analysis of the

proposed GDRnet architecture against the existing GNN models.
We demonstrate the advantage of using SIGN as an encoder in
GDRnet through the performance gain achieved in terms of its
training and inference time.

  - We apply GDRnet for COVID-19 drug repurposing by including

the COVID-19 interactome information from [19] in the dataset.
Many of the drugs predicted by GDRnet for COVID-19 are being
studied for their efficacy against the disease.


The software to reproduce the results are available in the github
[repository: https://github.com/siddhant-doshi/GDRnet](https://github.com/siddhant-doshi/GDRnet)


**2. Multilayered drug repurposing graph**


In this section, we model the biological data as a multilayer graph to
capture the complex interactions between different biological entities.
We consider four entities that are relevant to the drug repurposing task.
The four entities are drugs (e.g., _Dexamethasone_, _Sirolimus_ ), diseases
(e.g., _Scabies_, _Asthma_ ), anatomies (e.g., _Bronchus_, _Trachea_ ), and genes [1]

(e.g., _DUSP11_, _PPP2R5E_ ). We form a four-layered heterogeneous graph
with these entities as layers; see the illustration in Fig. 1a.

In the multilayer graph, i.e., the interactome there are inter-layered
connections between the four layers and intra-layered connections
within each layer. The inter-layered connections are of different types.
The drug–disease links indicate treatment or palliation, i.e., a drug
treats or has a relieving effect on a disease. For example, interaction between _Ivermectin-Scabies_ (as seen in Fig. 1b) and _Simvastatin-_
_Hyperlipidemia_ (as seen in Fig. 1d) are of type treatment, whereas
_Atropine-Parkinson’s disease_ is of type palliation. The drug–gene and
disease–gene links are the direct gene targets of the compound and the
disease, respectively. _NR3C2_, _RHOA_, _DNMT1_ are some of the target
genes of the drug _Dexamethasone_ (see Fig. 1b) and _PPP1R3D_, _CAV3_
are target genes of the disease _Malaria_ . There are also indirect links
between target genes of a drug and a disease, referred to as the shared
target genes (see Fig. 1b). For example, genes like _ATF3_, _UPP1_, _CTSD_,
are the shared target genes of drug _Ivermectin_ and disease _Malaria_ .
The disease–anatomy and gene–anatomy connections indicate how the
diseases affect the anatomies and interactions between the genes and
anatomies. For example, _GNAI2_ and _HMGCR_ belong to the _cardiac_
_ventricle_ anatomy (see Fig. 1d); disease _Schizophrenia_ affects multiple
anatomies like the _central nervous system (CNS)_ and _optic tract_ .

The intra-layered drug–drug and disease–disease connections show
the similarity between a pair of drugs and diseases, respectively. The
gene–gene links describe the interaction between genes (e.g., epistasis,
complementation) and form the whole gene interactome network. The
anatomy information helps by focusing on the local interactions of
genes related to the same anatomy as the genes targeted by the new
disease. Some examples of the intra-layered connections are _Simvas-_
_tatin_ - _Lovastatin_ and _POLA2_ - _RAE1_ as seen in Fig. 1d. This comprehensive network serves as a backbone for our model, which predicts
the unknown inter-layered links between drugs and novel diseases by
leveraging the multi-layered graph-structured data.


1 All the genes are represented using the symbols according to the HUGO
gene nomenclature committee (HGNC) [20].



2


_S. Doshi and S.P. Chepuri_



_Computers in Biology and Medicine 150 (2022) 105992_



**Fig. 1.** Drug repurposing network. (a) Illustration of the four-layered heterogeneous graph with the inter-layer and the intra-layer connections. (b), (c) and (d) Subgraphs centered
around the drugs _Dexamethasone_, _Ivermectin_ and _Simvastatin_, respectively, illustrate shared target genes between these drugs and COVID-19 disease nodes (see description later on
in Section 4.8).



**3. Methods and models**


Graph neural networks (GNNs) have become very popular for processing and analyzing such graph-structured data in the last few years.
Compared to deep learning models such as convolutional neural networks (CNNs), GNNs offer extraordinary performance improvements
while dealing with graph-structured data commonly encountered in
social networks, biological networks, brain networks, and molecular
networks, to name a few. GNN models learn low-dimensional graph
representations or node embeddings that capture the nodal connectivity information useful for solving graph analysis tasks like node
prediction, graph classification, and link prediction. In this section, we
describe the proposed GDRnet architecture for drug repurposing, which
is formulated as a link prediction problem.


_3.1. Notation_



**𝐱** _𝑖_ [(] _[𝑘]_ [+1)] = _𝑔_ _𝑘_ ( **𝐱** _𝑖_ [(] _[𝑘]_ [)] _[, 𝑓]_ _[𝑘]_ ({ **𝐱** _𝑗_ [(] _[𝑘]_ [)] _[,]_ [ ∀] _[𝑗]_ [∈] [] _𝑣_ [(1)] _𝑖_



to as input features). Let us denote the input feature vector of node _𝑣_ _𝑖_
by **𝐱** _𝑖_ [(0)] ∈ R _[𝑑]_, which contains attributes of that node.


_3.2. Graph neural networks_


In most of the existing GNN architectures, the embedding of a node
is updated during training by sequentially aggregating information
from its 1-hop neighbor nodes, thereby accounting for local interactions
in the network. This is also referred to as a GNN layer. Several such
GNN layers are cascaded to capture interactions beyond the 1-hop
neighborhood. Specifically, by cascading _𝐾_ such layers, node features
from its _𝐾_ -hop neighborhood are captured. For example, in Fig. 1c,
the drug _Ivermectin_ is a 2-hop neighbor of the anatomy _Lung_ and is
connected via _STC2_ . Mathematically, the node feature vector updates
can be represented by the recursion



_,_ (1)
}))



Consider an undirected graph  = ( _,_ ) with a set of vertices  =
{ _𝑣_ 1 _, 𝑣_ 2 _,_ … _, 𝑣_ _𝑁_ } and edges _𝑒_ _𝑖𝑗_ ∈  denoting a connection between nodes
_𝑣_ _𝑖_ and _𝑣_ _𝑗_ . We represent a graph  using the adjacency matrix **𝐀** ∈
R _[𝑁]_ [×] _[𝑁]_, where the ( _𝑖, 𝑗_ )th entry of **𝐀**, denoted by _𝑎_ _𝑖𝑗_, is 1 if there exists
an edge between nodes _𝑣_ _𝑖_ and _𝑣_ _𝑗_, and _𝑧𝑒𝑟𝑜_ otherwise. To account for
the non-uniformity in the degrees of the nodes, we use the normalized
adjacency matrix denoted by **𝐀** _[̃]_ = **𝐃** [−] 2 [1] **𝐀𝐃** [−] 2 [1], where **𝐃** ∈ R _[𝑁]_ [×] _[𝑁]_ is the

diagonal degree matrix. Each node in the graph has attributes (referred



where **𝐱** _𝑖_ [(] _[𝑘]_ [)] ∈ R _[𝑑]_ _[𝑘]_ is the embedding for node _𝑣_ _𝑖_ at the _𝑘_ th layer

and  _𝑣_ [(] _𝑖_ _[𝑗]_ [)] represents a set of _𝑗_ -hop neighbor nodes of node _𝑣_ _𝑖_ . Local
aggregation function _𝑓_ _𝑘_ (⋅) combines the neighbor node features (during
the training) and _𝑔_ _𝑘_ (⋅) transforms it to obtain the updated feature
vector. Different choices of the aggregation function _𝑓_ _𝑘_ (⋅) and the
transformation function _𝑔_ _𝑘_ (⋅) lead to different GNN variants like the
graph convolutional networks (GCN) [21], GraphSAGE [22], and graph
attention networks (GAT) [23], to name a few. However, these GNN



3


_S. Doshi and S.P. Chepuri_



_Computers in Biology and Medicine 150 (2022) 105992_



**Fig. 2.** The GDRnet architecture.



models do not scale well on large and dense graphs as their computational cost depends on the number of nodes and edges in the graph. To
reduce the runtime computations, a scalable GNN architecture called
SIGN [18] has been proposed, where the neighborhood aggregations at
various depths (till _𝐾_ -hop) are precomputed (before training), and the
node embeddings are generated non-iteratively, unlike the GNN models
in Eq. (1). As the node features updates are performed beforehand
outside the training procedure, these GNN variants easily scale on large
graphs, such as the multi-layered drug repurposing graph, as they are
independent of the number of edges in the graph. The proposed GDRnet
architecture has an encoder–decoder architecture, wherein the encoder
is based on the SIGN architecture due to its computational advantages.
While SIGN has been used for node classification [18], we utilize it here
for link prediction, i.e., to predict links between drugs and diseases.
Next, we describe the proposed GDRnet architecture.


_3.3. The GDRnet architecture_


The proposed GNN architecture for drug repurposing has two main
components, namely, the encoder and decoder. The encoder generates
the node embeddings of all the nodes in the four-layer graph. The
decoder scores a drug–disease pair based on the embeddings. The
encoder and decoder networks are trained in an end-to-end manner.

Next, we describe these two components of the GDRnet architecture,
which is illustrated in Fig. 2.


_3.3.1. Encoder_

The GDRnet encoder produces low-dimensional node embeddings
based on the input features and nodal connectivity information. Recall
that the matrix **𝐀** _[̃]_ is the normalized adjacency matrix of the four-layered
graph . We use graph operators represented using matrices **𝐅** _𝑟_ = **𝐀** _[̃]_ _[𝑟]_,
_𝑟_ = 1 _,_ 2 _,_ …, to aggregate information in the graph. Here, **𝐀** _[̃]_ _[𝑟]_ denotes
the _𝑟_ th matrix power. By choosing **𝐅** _𝑟_ = **𝐀** _[̃]_ _[𝑟]_, we aggregate information
from the _𝑟_ -hop neighborhood. We assume that each node has its own
_𝑑_ -dimensional feature, which we collect in the matrix **𝐗** ∈ R _[𝑁]_ [×] _[𝑑]_ to
obtain the input feature matrix associated with the nodes of . We can
then represent the encoder as


**𝐙** = _𝜎_ 1 {[ **𝐗** _**𝜣**_ 0 ‖ **𝐅** 1 **𝐗** _**𝜣**_ 1 ‖ … ∥ **𝐅** _𝑟_ **𝐗** _**𝜣**_ _𝑟_ ]} and **𝐘** = _𝜎_ 2 { **𝐙𝐖** } _,_ (2)


where **𝐘** is the final node embedding matrix for the nodes in the graph
 and { _**𝜣**_ 0 _,_ … _,_ _**𝜣**_ _𝑟_ _,_ **𝐖** } are the learnable parameters. Here, ∥ represents
concatenation, _𝜎_ 1 {⋅} and _𝜎_ 2 {⋅} are the nonlinear tanh and leaky rectified linear unit (leaky ReLU) activation functions, respectively. The _̃_
matrix **𝐅** _𝑟_ **𝐗** = **𝐀** _[𝑟]_ **𝐗** aggregates node features from _𝑟_ -hop neighbors,
which can be related to the neighborhood aggregation performed at
the _𝑟_ th layer of GNN models that perform sequential neighborhood



aggregation as in Eq. (1). Fig. 2 shows the encoder architecture. The
main advantage of using SIGN over other models (e.g., GCN, GAT,
GraphSAGE) is that the matrix product **𝐅** _𝑟_ **𝐗** is independent of the
learnable parameters _**𝜣**_ _𝑟_ . Thus, this matrix product can be precomputed
before training the neural network model. Doing so reduces the computational complexity while incorporating information from the graph

structure.

In our experiments, we choose _𝑟_ = 2, i.e., the low-dimensional node
embeddings have information from 2-hop neighbors. Choosing _𝑟_ ≥ 3 is
found to be not useful for drug repurposing, as we aim to capture the
local information of the drug targets such that a drug node embedding
should retain information about its target genes and the shared genes
in its vicinity. For example, the 1-hop neighbors of _Dexamethasone_ as
shown in Fig. 1b, are the diseases it treats (e.g., _Asthma_ ), and the drugs
similar to _Dexamethasone_ (e.g., _Methylprednisolone_ ) and its target genes
(e.g., _DUSP11_, _RHOA_ ). The 2-hop neighbors are the anatomies of the
target genes (e.g., _Bronchus_ ), and the drugs that have similar effects
on the diseases (e.g., _Hydrocortisone_ and _Dexamethasone_ have similar
effects on _Asthma_ ). While updating the node for the embedding related
to _Dexamethasone_, it is important to retain this local information for the
drug repurposing task.


_3.3.2. Decoder_

For drug repurposing, we propose a score function based on a
general dot-product that takes as input the updated embeddings of
drugs and diseases and outputs a score based on which we decide
if a certain drug treats the disease. Fig. 2 illustrates the proposed
learnable decoder. The columns of the embedding matrix **𝐘** contain
the embeddings of all the nodes in the four-layer graph, including
the embeddings of the disease and drug nodes. Let us denote the
embeddings of the _𝑖_ th drug as **𝐲** _𝑐_ _𝑖_ ∈ R _[𝑙]_ and the embeddings of the _𝑗_ th
disease as **𝐲** _𝑑_ _𝑗_ ∈ R _[𝑙]_ . The proposed scoring function 𝚜𝚌𝚘𝚛𝚎(⋅) to infer
whether drug _𝑐_ _𝑖_ is a promising treatment for disease _𝑑_ _𝑗_ is defined as



where _𝜎_ {⋅} is the nonlinear sigmoid activation function and _**𝜱**_ ∈ R _[𝑙]_ [×] _[𝑙]_ is
a learnable co-efficient matrix. We interpret _𝑠_ _𝑖𝑗_ as the probability that
a link exists between drug _𝑐_ _𝑖_ and disease _𝑑_ _𝑗_ . The term **𝐲** _𝑐_ _[𝑇]_ _𝑖_ _**[𝜱]**_ **[𝐲]** _[𝑑]_ _𝑗_ [can be]
interpreted as a measure of correlation (induced by _**𝜱**_ ) between the
disease and drug node embeddings.


_3.3.3. Training loss_
The model is trained in a mini-batch setting in an end-to-end
fashion using stochastic gradient descent to minimize the weighted



_𝑠_ _𝑖𝑗_ = 𝚜𝚌𝚘𝚛𝚎 ( **𝐲** _𝑐_ _𝑖_ _,_ **𝐲** _𝑑_ _𝑗_



) = _𝜎_ { **𝐲** _𝑐_ _[𝑇]_ _𝑖_ _**[𝜱]**_ **[𝐲]** _[𝑑]_ _𝑗_



_,_ (3)
}



4


_S. Doshi and S.P. Chepuri_


**Table 1**

Multi-layered graph data. The value in each cell represents the number of links between
the respective layers. NC represents no connection.


Drugs 6486

Diseases 6113 543

Genes 76 250 123 609 474 526

Anatomies NC 3602 726 495 NC

Drugs Diseases Genes Anatomies


cross-entropy loss, where the loss function for the sample corresponding
to the drug–disease pair ( _𝑖, 𝑗_ ) is given by



))



𝓁( _𝑠_ _𝑖𝑗_ _, 𝑧_ _𝑖𝑗_ ) = _𝑤𝑧_ _𝑖𝑗_



1
log
( ( _𝜎_ ( _𝑠_ _𝑖𝑗_ )



1
+ [(] 1 − _𝑧_ _𝑖𝑗_ ) log ( 1 − _𝜎_ ( _𝑠_ _𝑖𝑗_ )



_,_ (4)
)



_Computers in Biology and Medicine 150 (2022) 105992_


then divided into the training and testing set with a 90%−10% split.
We train the network using mini-batch stochastic gradient descent by
grouping the training set in batches of size 512 and train them for
nearly 20 epochs. Due to the significant class imbalance, we oversample
the drug–disease links while creating batches, thus maintaining the
class ratio (ratio of the number of negative samples to the number of
positive samples) of 1.5 in each batch. The additional hyperparameters
are set as follows. The intermediate embedding dimensions are fixed
to 250, the batch size and the learning rate (set to 10 [−][4] ) are chosen
by performing a grid search over the hyperparameter space. Also, we
use the leaky rectified linear unit (Leaky-ReLU) as the intermediate
activation function. We use the Adam optimizer to perform the back
propagation and update the model parameters. The weight _𝑤_ on the
positive samples (cf. Eq. (4)) is also chosen to be the class imbalance
ratio of each batch, i.e., we fix _𝑤_ to be 1.5.


_4.3. Baselines_


We perform experiments on the state-of-the-art network-based drug
repurposing methods, the network-proximity based [9], which is based
on the Z-scores computed using the permutation test, the HINGRL [17]
method based on the autoencoder and deepwalk algorithm, and the
Bipartite-GCN method [31], which uses an attention-based GNN layer.
In addition, we also provide a comparison with three commonly used
GNN encoder architectures, namely, GCN [21], GraphSAGE [22], and
GAT [23] for the drug repurposing task, which we treat as a link prediction problem, and compare the classification performance with the
GDRnet architecture. Specifically, the encoder in GDRnet is replaced
with GCN, GraphSAGE, and GAT to evaluate the model performance.
Two blocks of these sequential models are cascaded to maintain consistency with _𝑟_ = 2 of the GDRnet architecture. We evaluate these models
on the test set, which contains known treatments for diseases that are
not shown to the model while training. To remain consistent, we use
the same initial embeddings for all the experiments.


_4.4. Classification performance_


We measure the classification abilities of a model through the
receiver operating characteristic (ROC) curve of the true positive rate
(TPR) versus the false positive rates (FPR) and the precision–recall
(PR) curve of the precision versus the recall. The area under the PR
curves (AUPRC) along with the area under the receiver operating
characteristics (AUROC), would give a comprehensive view of the
performance statistics of the encoders. Fig. 3a shows the ROC curves of
different GNN models. We can see that all the models have very similar
AUROC values. Also, all the AUPRC values, as shown in Fig. 3b are in
a similar range. As compared to the baseline precision of 0.03, which
is calculated as the ratio of the minority class in the data, we see a
significant gain in the AUPRC values. Fig. 4a provides an illustration
of two-dimensional embeddings (from GDRnet), using the t-distributed
stochastic neighbor embedding (t-SNE), where we observe that diseases that target certain anatomy or a drug that target certain gene
have nearby representations in the embedding space demonstrating the
expressive power of GDRnet.


_4.5. Ranking performance_


We evaluate GDRnet in terms of ranks of the actual treatment drug
in the predicted list for a disease from the testing set, where the rank is
computed by rank ordering the scores. Fig. 5 represents the histograms
of the ranks of the drug–disease pairs from the testing set for GraphSAGE, GCN, GAT, HINGRL, and Bipartite-GCN compared with GDRnet.
To get the histograms, we compute the ranks of the actual treatment
drugs for the diseases from the test set and plot the frequencies of those
ranks on the vertical axis corresponding to the ranks on the horizontal
axis. We see that GDRnet has a higher density of ranks in the top



where _𝑧_ _𝑖𝑗_ is the known training label associated with the score _𝑠_ _𝑖𝑗_
for the drug–disease pair ( _𝑐_ _𝑖_ _, 𝑑_ _𝑗_ ), _𝑧_ _𝑖𝑗_ = 1 indicates that drug _𝑖_ treats or
palliates disease _𝑗_, and _𝑧_ _𝑖𝑗_ = 0 otherwise. Here, _𝑤_ is the weight on the
positive samples that we choose to account for the huge class imbalance
in the dataset. During training, we include no-drug–disease links, which
give us the negative control for learning. For example, there is no
link between the drug–disease pair _Simvastatin-Scabies_, i.e., _Simvastatin_
is not known to treat or suppress the effects of _Scabies_ . The number
of no-drug–disease links is almost thirty times the number of positive
samples. To handle this class disparity, we explicitly use a weight _𝑤>_ 0
on the positive samples.


**4. Model evaluation and experiments**


In this section, we evaluate GDRnet and discuss the choice of
various hyper-parameters. The model is evaluated based on two performance measures. Firstly, we report the ability to classify the links
correctly, i.e., to predict the known treatments correctly for diseases in
the test set. Next, using the list of predicted drugs for the diseases in
the test set, we report the model’s ability to rank the actual treatment
drug as high as possible (the ranking is obtained by ordering the scores
in Eq. (3)). Finally, we also report prediction results for coronavirus
related diseases.


_4.1. Dataset_


We use information from the drug repurposing knowledge graph
(DRKG) [24] to form the multi-layered drug repurposing graph. DRKG
includes information about six drug databases, namely, Drugbank [25],
Hetionet [26], GNBR [27], STRING [28], IntAct [29], and DGIdb [30].
We construct a four-layered graph comprising the drug layer, disease
layer, gene layer, and anatomy layer. We extract the details about
these entities specifically from the Drugbank, Hetionet, and GNBR
databases. We leverage their generic set of low-dimensional embeddings that represent the graph nodes and edges in the Euclidean space
for training. The four-layered graph is composed of 8070 drugs, 4166
diseases, 29 848 genes, 400 anatomies, and a total of 1,417,624 links,
which include all the inter-layer and intra-layer connections (refer
Section 2 for the description of the multi-layered graph). Details about
the inter-layered and intra-layered links are given in Table 1.


_4.2. Experimental setup and model parameters_


The drug repurposing problem is formulated as a link prediction.
It can be viewed as a binary classification problem, wherein a positive
class represents the existence of a link between a drug and disease, and
otherwise represents a negative class. We have 6113 positive samples
(drug–disease links) in our dataset. To account for the negative class
samples, we randomly choose 200,000 no-drug–disease links (i.e., those
pairs with no link between these drugs and diseases). These links are



5


_S. Doshi and S.P. Chepuri_



_Computers in Biology and Medicine 150 (2022) 105992_



**Fig. 3.** Classification performance of GDRnet. (a) and (b) represent the receiver operating curves (ROC) and the precision–recall (PR) curves, respectively, depicting the classification
performance of different drug-repurposing models.


**Fig. 4.** Embedding visualization. (a) Two-dimensional t-SNE visualization of the high-dimensional embeddings generated by GDRnet for the nodes in the four-layered heterogeneous
graph. The left embedding plot shows the representation of all the nodes (around 42 000 nodes), which are colored according to their layer. The right plot focuses on the drugs
and diseases used for testing. (b) Embeddings of the COVID-19 disease nodes (27 SARS-CoV-2 proteins and 6 coronavirus related diseases) and the predicted drugs by GDRnet.
The drugs in the both the plots (a) and (b) are colored according to their first-level anatomical therapeutic chemical (ATC) categorization.



15 as compared to other models. This clearly illustrates that GDRnet
outperforms the other graph-based methods in terms of its ranking
abilities. In addition, we compute the network proximity scores [9] and
rank order the drugs based on network proximity scores to compare
with the GNN-based encoder models. These network proximity scores
are a measure of the shortest distance between drugs and diseases
through their target genes. They are computed as



min
_𝑞_ ∈ _𝑝_ ∈ _[𝑑]_ [(] _[𝑝, 𝑞]_ [)]



1
_𝑃_ =
_𝑖𝑗_ || + | |



∑
( _𝑝_ ∈



_𝑝_ ∑ ∈ min _𝑞_ ∈ _[𝑑]_ [(] _[𝑝, 𝑞]_ [) +] _𝑞_ ∑ ∈



)



sample drug–disease pairs from the test set that were not shown during
the training. We can see that the GDRnet and the other GNN variants
result in better ranks on the unseen diseases than the network proximity
measure, which is solely based on the gene interactome, by a huge
margin. Also, determining the network proximity scores is extremely
computationally expensive due to the calculation of Z-scores using the
permutation test. For the same reasons we leave off the histogram
analysis for the network proximity approach, which evidently through
the examples in Table 2, results in poor ranking performance. The
diseases on which we evaluate are not confined to a single anatomy
(e.g., _rectal neoplasms_ are associated to the _rectum_ anatomy, whereas
_pulmonary fibrosis_ is a _lung_ disease), nor do they indicate a similar
family of drugs for their treatment (e.g., _Fluorouracil_ is an antineoplastic
drug, and _Prednisone_ is an anti-inflammatory corticosteroid). For a
majority of the diseases in the test set, GDRnet ranks the treatment
drug in the top 15 (as seen in Table 2). In the case of _Leukemia_, other
antineoplastic drugs like _Hydroxyurea_ and _Methotrexate_ are ranked high
(in top 10) and its known treatment drug _Azacitidine_ is ranked 17.



_,_ (5)



where _𝑃_ _𝑖𝑗_ is a proximity score of drug _𝑐_ _𝑖_ and disease _𝑑_ _𝑗_ . Here,  is the
set of target genes of _𝑐_ _𝑖_,  is the set of target genes of _𝑑_ _𝑗_, and _𝑑_ ( _𝑝, 𝑞_ ) is
the shortest distance between a gene _𝑝_ ∈  and a gene _𝑞_ ∈  in the gene
interactome. We convert these into Z-scores using the permutation test
_𝑍_ _𝑖𝑗_ = ( _𝑃_ _𝑖𝑗_ − _𝜇_ )∕ _𝜔_, where _𝜇_ is the mean proximity score of the pair ( _𝑐_ _𝑖_ _, 𝑑_ _𝑗_ )
computed by randomly selecting subsets of genes with the same degree
distribution as that of  and  from the gene interactome, and _𝜔_ is the
standard deviation of the scores generated in the permutation test of
these randomly selected subsets. Table 2 provides the rankings of a few



6


_S. Doshi and S.P. Chepuri_



_Computers in Biology and Medicine 150 (2022) 105992_



**Fig. 5.** Ranking histograms. The ranking performance of GDRnet compared with other GNN variants, namely, (a) GraphSAGE (b) GCN and (c) GAT (d) HINGRL (e) Bipartite-GCN.


**Table 2**

Ranking. A few examples of the ranks of the actual treatment drugs for the diseases from the testing set. There are no associated genes with some of the disease in our database,
which makes it impossible to rank them using the network proximity based method. These are indicated as ‘‘Not computable’’. The best ranks are highlighted in **bold** .


Disease Treatment drug Ranks


GDRnet GraphSAGE GCN GAT Network proximity HINGRL Bipartite-GCN


_Encephalitis_ _Acyclovir_ **10** 35 35 295 5462 435 27
_Rectal neoplasms_ _Fluorouracil_ **9** 421 16 231 2831 205 117
_Pulmonary fibrosis_ _Prednisone_ 5 3 10 9 2072 **2** 9
_Atrioventricular block_ _Atropine_ **6** 79 8 14 4453 26 196
_Pellagra_ _Niacin_ **2** 56 497 484 Not computable 460 288
_Colic_ _Hyoscyamine_ **1** **1** 501 205 Not computable 39 101
_Leukemia_ _Azacitidine_ **17** 120 31 332 377 527 507


**Table 3**

Layer ablation study. The AUROC values for a link prediction task compared across different graph layers and different GNN models. Best performances are
indicated in **bold** .


Graph layers GDRnet GraphSAGE GCN GAT


Drugs, Diseases 0 _._ 61 ± 0 _._ 02 0 _._ 707 ± 0 _._ 02 0 _._ 692 ± 0 _._ 02 0 _._ 655 ± 0 _._ 01
Drugs, Diseases, Anatomies 0 _._ 652 ± 0 _._ 01 0 _._ 75 ± 0 _._ 01 0 _._ 728 ± 0 _._ 02 0 _._ 722 ± 0 _._ 01
Drugs, Diseases, Genes 0 _._ 845 ± 0 _._ 02 **𝟎** _._ **𝟖𝟖𝟏** ± **𝟎** _._ **𝟎𝟐** 0 _._ 833 ± 0 _._ 01 0 _._ 84 ± 0 _._ 01
Drugs, Diseases, Genes, Anatomies **𝟎** _._ **𝟖𝟓𝟓** ± **𝟎** _._ **𝟎𝟐** 0 _._ 874 ± 0 _._ 01 **𝟎** _._ **𝟖𝟒𝟐** ± **𝟎** _._ **𝟎𝟐** **𝟎** _._ **𝟖𝟔𝟑** ± **𝟎** _._ **𝟎𝟏**



_4.6. Layer ablation study_


To gain more insights on the importance of different entities,
namely, drugs, disease, genes, and anatomies for drug repurposing,
we perform an ablation study on the layers of the constructed graph.
We perform link prediction using considered GNN models on the
constructed graphs, starting with the only drug–disease two-layered
graph, followed by the individual addition of the gene and the anatomy
interactome, making it a three-layered graph, and eventually converting it to a four-layered graph by getting all the layers together. We
report the corresponding AUROC values in Table 3. We use the degree
information as the input features for these experiments to eliminate
any biases due to the pre-trained embeddings. As seen in Table 3, the
addition of the anatomy and the gene layer shows their importance
by giving a significant improvement in the classification performance,
demonstrating the significance of the indirect connections provided by
the anatomy and the gene layers between the drugs and diseases for
drug repurposing. Finally, when all the information from the four layers
used together, we see a clear boost in the performance.
In summary, GNNs perform better than the prior network-based
approaches in predicting the drugs for a disease. This also signifies



the importance of capturing the local interactions in complex biological networks. These interactions are not sufficiently captured by the
network proximity methods that restrict their focus only on the target
genes of a drug and a disease. The proposed GNN-based GDRnet architecture is computationally attractive and better ranks known treatment
drugs for diseases than the popular sequential GNN variants.


_4.7. Computational complexity_


The time complexity of GNNs that perform aggregation sequentially
like GCN, GraphSAGE, and GAT, is ( _𝐿𝑁𝑑_ [2] + _𝐿_ || _𝑑_ ) for a graph having
_𝑁_ nodes and || edges with _𝐿_ sequential aggregation iterations [32].
The intermediate embedding dimensions are assumed to be _𝑑_ . Here,
the term _𝑁𝑑_ [2] corresponds to the feature transformation, and || _𝑑_ is
the additional computations performed to identify the neighborhood
for local aggregation during the training. GDRnet benefits itself in terms
of the training and inference time due to its parallel framework by precomputing this neighborhood aggregations. This results in the runtime
to be independent of the number of edges in the graph, having a time
complexity of ( _𝐿𝑁𝑑_ [2] ), where _𝐿_ is the number of parallel branches.
Fig. 6 illustrates the dependence of GNNs on the number of edges.



7


_S. Doshi and S.P. Chepuri_


**Fig. 6.** Computational complexity. Time plot showing the dependence of GNN
architectures on the number of edges.


**Table 4**

Drugs predicted by GDRnet for COVID-19.


COVID-19 node Drugs predicted by GDRnet ranked in top 10


_SARS-CoV2-E_ _Ivermectin_, _Spironolactone_, _Sirolimus_
_SARS-CoV2-M_ _Ivermectin_, _Cyclosporine_, _Acyclovir_
_SARS-CoV2-N_ _Rubella virus vaccine_, _Sirolimus_, _Hydralazine_
_SARS-CoV2-spike_ _Crizanlizumab_, _Cyclosporine_, _Cidofovir_, _Nitazoxanide_
_CoV-NL63_ _Dexamethasone_, _Prednisolone_, _Celecoxib_


The time taken for a single epoch (forward pass) on a graph having
the same number of nodes as in the constructed multilayered graph in
Section 2 (approximately 42 000) are plotted on the vertical axis for
varying number of edges on the horizontal axis. GCN, GraphSAGE, and
GAT clearly depict their linear dependence on ||, whereas GDRnet
verifies its independence by having a constant time, irrespective of
the number of edges. The Bipartite-GCN architecture uses an attentionbased graph layer similar to GAT. Thus it has the same complexity
as the sequential based GNNs. It is not straightforward to compare
the forward pass time complexity incurred by network proximity and
HINGRL methods. HINGRL pipeline involves multiple algorithms that
are trained independently, like the autoencoder, followed by deepwalk,
and finally the random forests, that incur more time complexity as
observed during our numerical experiments. For the network proximity
method, due to the involvement of the permutation test, it is extremely
computationally expensive as well.


_4.8. COVID-19 drug repurposing_


Next, we focus on drug repurposing for the four known human
coronaviruses (HCoVs), namely, _SARS-CoV_, _MERS-CoV_, _CoV-229E_ and
_CoV-NL63_, and two non-human coronaviruses, namely _MHV_, and _IBV_ .
We consider interactions of these disease nodes with human genes.
There are 129 known links between these six disease nodes and gene
nodes in the dataset [24]. In addition, we consider all the 27 _SARS-_
_CoV-2_ proteins that include 4 structural proteins, namely, envelope
( _SARS-CoV2-E_ ), membrane ( _SARS-CoV2-M_ ), nucleocapsid ( _SARS-CoV2-_
_N_ ) and surface ( _SARS-CoV2-spike_ ), 15 non-structural proteins (nsp)
and 8 open reading frames (orf), and their 332 links connecting the
target human genes [19]. We refer to these 33 nodes (6 disease nodes
and 27 _SARS-CoV2_ proteins) as the COVID-19 nodes. In other words,
there are only disease–gene interactions available for these COVID-19
nodes. Some of the genes targeted by the COVID-19 nodes are shown
in Fig. 1 (b, c and d), which are also the target genes for the drugs
(e.g., _Dexamethasone_, _Ivermectin_, _Simvastatin_ ).



_Computers in Biology and Medicine 150 (2022) 105992_


We individually predict the drugs for all these 33 COVID-19 nodes
as each protein in _SARS-CoV-2_ targets a different set of genes in
humans. We select the top 10 ranked predicted drugs out of 8070
clinically approved drugs for each disease entity. Table 4 lists some
of the predicted drugs by GDRnet. A complete list of the predicted
[drugs with their scores and ranks is available in our repository at: https:](https://github.com/siddhant-doshi/GDRnet)
[//github.com/siddhant-doshi/GDRnet. Our predictions have corticos-](https://github.com/siddhant-doshi/GDRnet)
teroids like _Dexamethasone, Methylprednisolone_, antineoplastic drugs
like _Sirolimus, Anakinra_, anti-parasitic drugs like _Ivermectin, Nitazox-_
_anide_, non-steroidal anti-inflammatory drugs (NSAIDs) like _Ibuprofen,_
_Celecoxib_, ACE inhibitors and statin drugs like _Simvastatin, Atorvastatin_,
and some of the vaccines discovered previously for other diseases
like the _Rubella virus vaccine_ . Fig. 4b gives a two-dimensional t-SNE
representation of the embeddings of a few predicted drugs and the
COVID-19 disease nodes, where we can see that the representation
of the predicted drugs is in the vicinity of the disease nodes in the
embedding space.


**5. Conclusions and future work**


We proposed a GNN model for drug repurposing model, called
GDRnet, to predict drugs from a large database of approved drugs for
further studies. We leverage a biological network of drugs, diseases,
genes, and anatomies and cast the drug repurposing task as a link
prediction problem. The proposed GDRnet architecture has a computationally attractive encoder to generate low-dimensional embeddings of
the entities and a decoder that scores the drug–disease pairs. Through
numerical simulations on real data, we demonstrate the efficacy of the
proposed approach for drug repurposing. We also apply GDRnet on
COVID-19 data.


This work can be extended along several directions. Considering
the availability of substantial biological data, the inclusion of information like individual side effects of drugs, may further improve the
predictions. Considering the comorbidities of a patient would help
us analyze the biological process and gene interactions in the body
specific to an individual and accordingly prescribe the line of treatment.
Also, including the edge specific information such as type of drug
interactions could help us predicting a synergistic combination of drugs
for a disease.


**Declaration of competing interest**


The authors declare that they have no known competing financial interests or personal relationships that could have appeared to
influence the work reported in this paper.


**Acknowledgments**


S.P. Chepuri is supported in part by the Pratiskha Trust Young
Investigator Award, Indian Institute of Science, Bangalore, and the
SERB, India grant SRG/2019/000619, and S. Doshi is supported by
the Robert Bosch Center for Cyber Physical Systems, Indian Institute
of Science, Bangalore, Student Research Grant 2020-M-11. The authors
[thank the Deep Graph Learning team for making DRKG public at https:](https://github.com/gnn4dr/DRKG)
[//github.com/gnn4dr/DRKG.](https://github.com/gnn4dr/DRKG)


**References**


[[1] S. Pushpakom, F. Iorio, P.A. Eyers, K.J. Escott, S. Hopper, A. Wells, T. Doig, J.](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb1)

[Latimer, C. McNamee, A. Norris, Drug repurposing: progress. and challenges and](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb1)
[recommendations, Nat. Rev. Drug Discov. 18 (1) (2018) 41–58.](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb1)

[[2] J.H. Beigel, K.M. Tomashek, L.E. Dodd, A.K. Mehta, B.S. Zingman, E. Kalil, H.Y.](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb2)

[Chu, A. Luetkemeyer, S. Kline, D. Lopez de Castilla, Remdesivir for the treatment](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb2)
[of Covid-19, N. Engl. J. Med. 383 (19) (2020) 1813–1826.](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb2)

[[3] L. Caly, J.D. Druce, M.G. Catton, D.A. Jans, K.M. Wagstaff, The FDA-approved](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb3)

[drug ivermectin inhibits the replication of SARS-CoV-2 in vitro, Antivir. Res. 178](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb3)
[(2020) 104787.](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb3)



8


_S. Doshi and S.P. Chepuri_


[[4] T.R. Group, Dexamethasone in hospitalized patients with Covid-19, N. Engl. J.](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb4)

[Med. (2020).](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb4)

[[5] M. Zitnik, F. Nguyen, B. Wang, J. Leskovec, A. Goldenberg, M.M. Hoffman,](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb5)

[Machine learning for integrating data in biology and medicine: Principles,](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb5)
[practice, and opportunities, Inf. Fusion 50 (2019) 71–91.](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb5)

[[6] M. Zitnik, M. Agrawal, J. Leskovec, Modeling polypharmacy side effects with](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb6)

[graph convolutional networks, Bioinformatics 34 (13) (2018) i457–i466.](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb6)

[[7] S. Kaliamurthi, G. Selvaraj, C. Selvaraj, S.K. Singh, D.Q. Wei, G.H. Peslherbe,](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb7)

[Structure-based virtual screening reveals ibrutinib and zanubrutinib as potential](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb7)
[repurposed drugs against COVID-19, Int. J. Mol. Sci. 22 (13) (2021) 7071.](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb7)

[[8] A. Khan, S.S. Ali, M.T. Khan, S. Saleem, A. Ali, M. Suleman, Z. Babar, A. Shafiq,](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb8)

[M. Khan, D.Q. Wei, Combined drug repurposing and virtual screening strategies](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb8)
[with molecular dynamics simulation identified potent inhibitors for SARS-CoV-2](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb8)
[main protease (3CLpro), J. Biomol. Struct. Dyn. 39 (13) (2021) 4659–4670.](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb8)

[[9] F. Cheng, R.J. Desai, D.E. Handy, R. Wang, S. Schneeweiss, J. Barabási, Network-](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb9)

[based approach to prediction and population-based validation of in silico drug](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb9)
[repurposing, Nature Commun. 9 (1) (2018) 1–12.](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb9)

[[10] Y. Zhou, Y. Hou, J. Shen, Y. Huang, W. Martin, F. Cheng, Network-based drug](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb10)

[repurposing for novel coronavirus 2019-nCoV/SARS-CoV-2, Cell Discov. 6 (1)](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb10)
[(2020) 1–18.](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb10)

[[11] D.M. Gysi, I. Do Valle, M. Zitnik, A. Ameli, X. Gan, O. Varol, S.D. Ghiassian,](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb11)

[J.J. Patten, R.A. Dave, J. Loscalzo, A.L. Barabási, Network medicine framework](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb11)
[for identifying drug-repurposing opportunities for COVID-19, Proc. Natl. Acad.](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb11)
[Sci. 118 (19) (2021).](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb11)

[12] V.N. Ioannidis, D. Zheng, G. Karypis, Few-shot link prediction via graph neural

[networks for Covid-19 drug-repurposing, 2020, arxiv preprint arxiv:2007.10261.](http://arxiv.org/abs/2007.10261)

[[13] X. Su, L. Hu, Z. You, P. Hu, L. Wang, B. Zhao, A deep learning method](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb13)

[for repurposing antiviral drugs against new viruses via multi-view nonnegative](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb13)
[matrix factorization and its application to SARS-CoV-2, Bioinformatics 23 (1)](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb13)
[(2022) bbab526.](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb13)

[[14] F. Yang, Q. Zhang, X. Ji, Y. Zhang, W. Li, S. Peng, F. Xue, Machine learning](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb14)

[applications in drug repurposing, Interdiscip. Sci.: Comput. Life Sci. (2022) 1–7.](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb14)

[[15] X. Pan, X. Lin, D. Cao, X. Zeng, P.S. Yu, L. He, R. Nussinov, F. Cheng, Deep](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb15)

[learning for drug repurposing: Methods. and databases. and and applications.,](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb15)
[Wiley Interdiscip. Rev.: Comput. Mol. Sci. (2022) e1597.](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb15)

[[16] X. Su, Z. You, L. Wang, L. Hu, L. Wong, B. Ji, B. Zhao, SANE: a sequence](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb16)

[combined attentive network embedding model for COVID-19 drug repositioning,](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb16)
[Appl. Soft Comput. 111 (107831) (2021).](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb16)

[[17] B.W. Zhao, L. Hu, Z.H. You, L. Wang, X.R. Su, Hingrl: predicting drug–disease](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb17)

[associations with graph representation learning on heterogeneous information](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb17)
[networks., Brief. Bioinform. 23 (1) (2022) bbab515.](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb17)

[18] F. Frasca, E. Rossi, D. Eynard, B. Chamberlain, M. Bronstein, F. Monti, SIGN:

[Scalable inception graph neural networks, 2020, arxiv preprint arXiv:2004.](http://arxiv.org/abs/2004.11198)

[11198.](http://arxiv.org/abs/2004.11198)



_Computers in Biology and Medicine 150 (2022) 105992_


[[19] D.E. Gordon, G.M. Jang, M. Bouhaddou, J. Xu, K. Obernier, K.M. White, M.J.](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb19)

[O’Meara, V.V. Rezelj, J.Z. Guo, D.L. Swaney, T.A. Tummino, A SARS-CoV-2](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb19)
[protein interaction map reveals targets for drug repurposing, Nature 583 (2020)](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb19)

[459–468.](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb19)

[[20] S. Povey, R. Lovering, E. Bruford, M. Wright, M. Lush, H. Wain, The HUGO gene](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb20)

[nomenclature committee (HGNC), Hum. Genet. 109 (6) (2001) 678–680.](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb20)

[21] T.N. Kipf, M. Welling, Semi-supervised classification with graph convolu
tional networks, in: Proceedings of the International Conference on Learning
Representations, Toulon, France, 2017.

[[22] W.L. Hamilton, R. Ying, J. Leskovec, Inductive representation learning on large](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb22)

[graphs, in: Advances in Neural Information Processing Systems, California,](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb22)
[United States, 2017.](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb22)

[23] P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Lio, Y. Bengio, Graph

attention networks, in: Proceedings of the International Conference on Learning
Representations, Vancouver, Canada, 2018.

[24] V.N. Ioannidis, X. Song, S. Manchanda, M. Li, X. Pan, D. Zheng, X. Ning, X.

Zeng, G. Karypis, DRKG - drug repurposing knowledge graph for Covid-19, 2020,
[https://github.com/gnn4dr/DRKG/.](https://github.com/gnn4dr/DRKG/)

[[25] D.S. Wishart, Y.D. Feunang, A.C. Guo, E.J. Lo, A. Marcu, J.R. Grant, D. Sajed,](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb25)

[C. Li, Z. Sayeeda, N. Assempour, Drugbank 5.0: a major update to the drugbank](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb25)
[database for 2018, Nucleic Acids Res. 46 (D1) (2017) D1074–D1082.](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb25)

[[26] D.S. Himmelstein, A. Lizee, C. Hessle, L. Brueggeman, S.L. Chen, A. Hadley,](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb26)

[P. Khankhanian, S.E. Baranzini, Systematic integration of biomedical knowledge](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb26)
[prioritizes drugs for repurposing, Elife 6 (2017) e26726.](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb26)

[[27] B. Percha, R.B. Altman, A global network of biomedical relationships derived](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb27)

[from text, Bioinformatics 34 (15) (2018) 2614–2624.](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb27)

[[28] D. Szklarczyk, A. Gable, D. Lyon, A. Junge, S. Wyder, M. Huerta-Cepas, N.T.](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb28)

[Doncheva, J.H. Morris, P. Bork, L.J. Jensen, STRING v11: protein–protein asso-](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb28)
[ciation networks with increased coverage. and supporting functional discovery](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb28)
[in genome-wide experimental datasets, Nucleic Acids Res. 47 (D1) (2019)](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb28)

[D607–D613.](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb28)

[[29] S. Orchard, M. Ammari, B. Aranda, L. Breuza, L. Briganti, F. Broackes-Carter,](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb29)

[N.H. Campbell, G. Chavali, C. Chen, N. Del-Toro, M. Duesbury, The mIntAct](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb29)
[project—IntAct as a common curation platform for 11 molecular interaction](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb29)
[databases, Nucleic Acids Res. 42 (D1) (2014) D358–D363.](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb29)

[[30] C.K. C., W.A. H., F.Y. Y., S. Kiwala, A.C. Coffman, G. Spies, A. Wollam, S.N. C.,](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb30)

[G.O. L., G. M., GIdb 3.0: a redesign and expansion of the drug–gene interaction](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb30)
[database, Nucleic Acids Res. 46 (D1) (2018) D1068–D1073.](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb30)

[[31] Z. Wang, M. Zhou, C. Arnold, Toward heterogeneous information fusion: bipartite](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb31)

[graph convolutional networks for in silico drug repurposing, Bioinformatics 36](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb31)
[(Supplement1) (2020) i525–33.](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb31)

[[32] Z. Wu, S. Pan, F. Chen, G. Long, C. Zhang, S.Y. Philip, A comprehensive survey](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb32)

[on graph neural networks, IEEE Trans. Neural Netw. Learn. Syst. 32 (1) (2021)](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb32)

[4–24.](http://refhub.elsevier.com/S0010-4825(22)00717-X/sb32)



9


