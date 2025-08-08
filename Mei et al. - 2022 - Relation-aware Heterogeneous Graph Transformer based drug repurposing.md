[Expert Systems With Applications 190 (2022) 116165](https://doi.org/10.1016/j.eswa.2021.116165)


[Contents lists available at ScienceDirect](http://www.elsevier.com/locate/eswa)

# Expert Systems With Applications


[journal homepage: www.elsevier.com/locate/eswa](http://www.elsevier.com/locate/eswa)

## Relation-aware Heterogeneous Graph Transformer based drug repurposing


Xin Mei [a] [,] [1], Xiaoyan Cai [b] [,] [1], Libin Yang [a] [,][∗], Nanxin Wang [a]


a _School of Cyber Science and Technology, Northwestern Polytechnical University, Xi’an, Shaanxi 710129, China_
b _School of Automation, Northwestern Polytechnical University, Xi’an, Shaanxi 710129, China_



A R T I C L E I N F O


_Keywords:_
Drug repurposing
Graph neural network
Graph transformer
Link prediction
Heterogeneous network


**1. Introduction**



A B S T R A C T


Drug repurposing refers to discovery of new medical instructions for existing chemical drugs, which has
great pharmaceutical significance. Recently, large-scale biological datasets are increasingly available, and
many graph neural network (GNN) based methods for drug repurposing have been developed. These methods
often deem drug repurposing as a link prediction problem, which mines features of biological data to
identify drug–disease associations (i.e., drug–disease links). Due to heterogeneity of data, we need to deeply
explore heterogeneous information of biological network for drug repurposing. In this paper, we propose
a Relation-aware Heterogeneous Graph Transformer (RHGT) model to capture heterogeneous information
for drug repurposing. We first construct a drug–gene–disease interactive network-based on biological data,
and then propose a three-level network embedding model, which learns network embeddings at fine-grained
subtype-level, node-level and coarse-grained edge-level, respectively. The output of subtype-level is the input
of node-level and edge-level, and the output of node-level is the input of edge level. We get edge embeddings
at edge-level, which integrates edge type embeddings and node embeddings. We deem that in this way,
characteristics of drug–gene–disease interactive network can be captured more comprehensively. Finally, we
identify drug–disease associations (i.e., drug–disease links) based on the relationship between drug–gene edge
embeddings and gene–disease edge embeddings. Experimental results show that our model performs better than
other state-of-the-art graph neural network methods, which validates effectiveness of the proposed model.



Drug repurposing refers to discovery of new medical instructions for
existing chemical drugs (Ashburn & Thor, 2004). Since it takes about
10 to 17 years to develop a new drug, costs between $500 million
and $2 billion (DiMasi, Hansen, & Grabowski, 2003), and at least a
decade to bring a new drug to the market (Adams & Brantner, 2006;
Dickson & Gagnon, 2009; DiMasi et al., 2003; Yella, Yaddanapudi,
Wang, & Jegga, 2018). While the safety of existing drugs is well
documented and can therefore be significantly reduced in the cost of
clinical trials. Compared with new drug design and discovery, drug
repurposing has significant advantages in many aspects such as safety
factors, cost, and rapid results, so drug repurposing has great pharmaceutical significance. Researchers usually take experimental methods or
computational methods to identify suitable drug candidates, which is
the key to drug repurposing.



Studies of drug repurposing have attained a number of achievements, many researchers used computational methods to solve drug
repurposing problem (Lotfi Shahreza, Ghadiri, Mousavi, Varshosaz,
& Green, 2018). It can be classified into similarity-based methods,
network-based and machine learning methods, as well as graph neural
networks (GNN) based methods. Among them, similarity-based methods use common features among drugs, such as chemical structures,
target proteins, and side effects, to build the computational models.
Network-based and machine learning methods mine the structure information among different biological networks to discover drug–disease
associations (drug–disease links).
With the development of deep learning, GNN has achieved success in network representation learning. Some researchers began to
study using GNN to solve drug repurposing problem, they treated
drug repurposing as a link prediction problem, which mines features



[The code (and data) in this article has been certified as Reproducible by Code Ocean: (https://codeocean.com/). More information on the Reproducibility](https://codeocean.com/)
[Badge Initiative is available at https://www.elsevier.com/physical-sciences-and-engineering/computer-science/journals.](https://www.elsevier.com/physical-sciences-and-engineering/computer-science/journals)
∗ Corresponding author.
_E-mail addresses:_ [meixin@mail.nwpu.edu.cn (X. Mei), xiaoyanc@nwpu.edu.cn (X. Cai), libiny@nwpu.edu.cn (L. Yang),](mailto:meixin@mail.nwpu.edu.cn)
[nanxin_wang@mail.nwpu.edu.cn.com (N. Wang).](mailto:nanxin_wang@mail.nwpu.edu.cn.com)
1 These authors have contributed equally to this work.


[https://doi.org/10.1016/j.eswa.2021.116165](https://doi.org/10.1016/j.eswa.2021.116165)
Received 30 April 2021; Received in revised form 21 October 2021; Accepted 27 October 2021

Available online 16 November 2021
0957-4174/© 2021 Elsevier Ltd. All rights reserved.


_X. Mei et al._


of biological data to identify drug–disease associations (i.e., drug–
disease links). For example, Dr-COVID (Doshi & Chepuri, 2020) used
Scalable Initial Graph Neural Network (SIGN) to design an encoder to
generate embeddings for nodes in the network, but it treats all nodes
equally, ignoring types of nodes. BiFusion (Wang, Zhou and Arnold,
2020) is a bipartite graph convolution network model, it divides the
heterogeneous network into several bipartite graphs, which cannot
reflect integrity and global information of the heterogeneous network.

In order to further enhance drug repurposing performance, we
propose a Relation-aware Heterogeneous Graph Transformer (RHGT)
model to capture interaction information of drug, medicine and gene.
Different from existing models that only learn network embedding at
node-level, we learn network embeddings at fine-grained subtype-level,
node-level and coarse-grained edge-level respectively. At subtype-level,
we first divide edge types into multiple subtypes, and build a subtype co-occurrence network-based on co-occurrence relationship of the
subtypes, then learn edge type embeddings based on the constructed
subtype co-occurrence network. At node-level, we first construct a
heterogeneous network named drug–gene–disease network ( _𝐺_ ), which
contains three types of nodes, i.e., chemical drug nodes, disease nodes
and gene nodes. Then we use edge type embeddings obtained from
the subtype-level to learn node embeddings in heterogeneous network
_𝐺_ . At edge-level, we use edge type embeddings and node embeddings to generate edge embeddings in heterogeneous network _𝐺_ . For
subtype-level, we learn edge type embeddings, which are the input of
node-level and edge-level, and node embeddings obtained at node-level
are also the input of edge-level. For edge-level, we get edge embeddings, which integrate edge type embeddings and node embeddings.
We deem that in this way, characteristics of heterogeneous network _𝐺_
can be captured more comprehensively. Finally, we can identify drug–
disease associations (i.e., drug–disease links) based on relationships
between drug–gene edge embeddings and gene–disease edge embeddings. Experimental results prove effectiveness of our proposed RHGT
model.


Our major contributions can be summarized as follows:
(1) We propose a three-level network model, which learns network
embeddings at subtype-level, node-level and edge-level, respectively.
For subtype-level, we learn edge type embeddings, which are the
input of node-level and edge-level, and node embeddings obtained at
node level are also the input of edge-level. For edge-level, we get
edge embeddings, which integrate edge type embeddings and node
embeddings.

(2) A fine-grained method is developed to learn edge type embeddings, which divides edge types into multiple subtypes, builds a
network based on the co-occurrence relationship of subtypes, learns
embeddings for the subtypes, and obtain the embedding for each edge

type.

(3) We propose a relational drug repurposing model, which identifies drug–disease associations based on the relationship between drug–
gene edge embeddings and gene–disease edge embeddings.

The rest of the paper is organized as follows. Section 2 reviews
related work. Section 3 represents problem formulation and notations.
Section 4 presents our proposed model in detail. Section 5 presents the
experimental results and analysis. The paper is concluded in Section 6.


**2. Related work**


We briefly review computational approaches for drug repurposing
and related research on GNNs.


_2.1. Computational approaches_


Computational approaches include similarity-based methods,
network-based and machine learning methods, as well as graph neural
network (GNN) based methods. Among them, most similarity-based
methods are integrated methods (Gottlieb, Stein, Ruppin, & Sharan,



_Expert Systems With Applications 190 (2022) 116165_


2011; Li & Lu, 2012; Napolitano et al., 2013; Zhang, Agarwal, &
Obradovic, 2013). For instance, Gottlieb et al. (2011) proposed a
similarity prediction model PREDICT that integrates similarity between
drugs and similarity between diseases. Iorio et al. (2010) developed
a method to discover the similarity between drugs, which uses drugs
as nodes to build a network based on the similarity of consensus
response pairs. Network-based and machine learning methods (Cheng
et al., 2018; Guney, Menche, Vidal, & Barábasi, 2016; Luo et al.,
2016; Zeng et al., 2019) often create a biological network where nodes
represent drugs, diseases or genes, and edges represent interactions or
relationships between nodes. Cheng et al. (2018) quantify the proximity
of disease genes and drug targets in human protein–protein interactome
in the network to identify new drug–disease associations (drug–disease
links). Recently, researchers also began to study using GNN to solve
drug repurposing task. Doshi and Chepuri (2020) proposed a drug
repurposing model based on graph neural network (GNN), namely
Dr-COVID. They used Scalable Initial Graph Neural Network (SIGN)
to design an encoder to generate embeddings for nodes in the network. Wang, Zhou et al. (2020) proposed a bipartite graph convolution
network model named BiFusion, which realizes drug repurposing by
fusing heterogeneous information.


_2.2. Graph neural networks_


GNN has achieved success in processing tasks based on graph
data (Cheng & Zhao, 2014; Gordon et al., 2020; Pushpakom et al.,
2019). Kipf and Welling (2016) proposed graph convolutional networks
(GCN) which generalizes convolutional neural networks (CNN) on
graph-structured data. Hamilton, Ying, and Leskovec (2017) proposed
an improved inductive framework based on GCN, called GraphSAGE,
which leverages node feature to efficiently generate node embeddings
for previously unseen data. Veličković et al. (2017) proposed graph
attention networks (GAT), which leverages self-attentional layers to
address shortcomings of prior methods based on graph convolutions.

Besides, there have been several attempts to use GNN to learn
heterogeneous networks (Chousterman, Swirski, & Weber, 2017; Guney
et al., 2016; Jockusch et al., 2020; Zaim, Chong, Sankaranarayanan,
& Harky, 2020). Schlichtkrull et al. (2018) proposed the relational
graph convolutional networks (RGCN) to model heterogeneous networks. RGCN is a graph convolutions-based model, which keeps a
distinct linear projection weight for each edge type. Hu, Dong, Wang,
and Sun (2020) proposed an attention mechanism-based method called
heterogeneous graph transformer, which characterizes heterogeneous
attention over each edge to model heterogeneity.

Our proposed RHGT model is also used to learn heterogeneous
network embeddings. Compared with existing similar heterogeneous
network embedding models, our model captures characteristics of node
type and edge type, and learns network embeddings at subtype-level,
node-level and edge-level, respectively. For subtype-level, we learn
edge type embeddings, which are the input of node-level and edgelevel, and node embeddings obtained at node-level are also the input
of edge-level. For edge-level, we get edge embeddings, which integrate
edge type embeddings and node embeddings. We deem that in this way,
characteristics of heterogeneous network _𝐺_ can be captured more comprehensively. Table 1 shows characteristics of several heterogeneous
network embedding models.


**3. Problem formulation and notations**


In this section, we first introduce the process of relation-aware
drug repurposing, which treats drug repurposing as a link prediction problem. Then we introduce concepts in heterogeneous networks,
node-level meta relation and edge-level meta relation used in RHGT
model.



2


_X. Mei et al._


**Table 1**

Characteristics of heterogeneous network embedding models.


Models Type features Embedding level


BiFusion Node type only Node-level
RGCN Edge type only Node-level
HGT Node type and edge type Node-level
RHGT Node type and edge type Subtype-level, node-level
and edge-level


**Problem 1** ( _Relation-aware Drug Repurposing_ ) **.** In this paper, we introduce a relation-aware method for drug repurposing, which identifies
drug–disease associations (i.e., drug–disease links) based on the relationship between drug–gene edges and gene–disease edges. That is,
we first learn vector representations of drug–gene edges and gene–
disease edges, and then make predictions of drug–disease associations
(drug–disease links) based on the relationships between drug–gene
edge embeddings and gene–disease edge embeddings.


**Definition 1** ( _Heterogeneous Network_ ) **.** A heterogeneous network is
defined as a graph with multiple types of nodes and/or multiple types
of edges. In this paper, we denote the heterogeneous network as _𝐺_ =
( _𝑉, 𝐴, 𝐸, 𝑅_ ). _𝑣_ ∈ _𝑉_ represents a node and _𝑒_ ∈ _𝐸_ is a edge. The
type mapping functions for node and edge are: _𝜏_ ( _𝑣_ ) ∶ _𝑉_ → _𝐴_ and
_𝜙_ ( _𝑒_ ) ∶ _𝐸_ → _𝑅_, respectively, where _𝑉_ is a node set, _𝐴_ represents the
node type union (including drug, gene and disease), _𝐸_ is an edge set,
and _𝑅_ represents the edge type union.


As shown in Fig. 1, we first construct a heterogeneous network
named drug–gene–disease interactive network ( _𝐺_ ). There are three
types of nodes in network _𝐺_ : drugs, genes and diseases. There are
multiple interaction relationships between different types of nodes, and
the relationships are represented by edges in the network. There exists
only one type of edges between drugs and diseases, i.e., if there exists
an edge between a drug and a disease, it means that the drug can
be used to treat the disease. There are many types of edges between
drugs and genes, such as decreasing edge type, increasing edge type
and affecting edge type. Each edge type consists of several interactions.
For example, a decreasing edge type is composed of two interactions,
including decreasing reaction and decreasing expression. There also
exist many types of edges between diseases and genes.


**Definition 2** ( _Node-level Meta Relation_ ) **.** For an edge _𝑒_ = ( _𝑣_ _𝑠_ _, 𝑣_ _𝑡_ ) linked
from source node _𝑣_ _𝑠_ to target node _𝑣_ _𝑡_, its meta relation is denoted as
⟨ _𝜏_ ( _𝑣_ _𝑠_ ) _, 𝜙_ ( _𝑒_ ) _, 𝜏_ ( _𝑣_ _𝑡_ )⟩.


**Definition 3** ( _Edge-level Meta Relation_ ) **.** For edge _𝑒_ _𝑠_ = ( _𝑣_ _𝑖_ _, 𝑣_ ) and edge
_𝑒_ _𝑡_ = ( _𝑣, 𝑣_ _𝑗_ ), the meta relation is denoted as ⟨ _𝜙_ ( _𝑒_ _𝑠_ ) _, 𝜏_ ( _𝑣_ ) _, 𝜙_ ( _𝑒_ _𝑡_ )⟩, _𝑣_ is the
common connection node of _𝑒_ _𝑠_ and _𝑒_ _𝑡_ .


**4. Relation-aware** **Heterogeneous** **Graph** **Transformer** **(RHGT)**
**based drug repurposing**


Our proposed Relation-aware Heterogeneous Graph Transformer
(RHGT) model is mainly composed of three modules: subtype-level
network embedding module, node-level network embedding module
and edge-level network embedding module, as shown in Fig. 2. Each
edge type embedding learned by subtype-level network embedding
module will be input as the edge type feature in node-level network embedding module and edge-level network embedding module.
Node-level network embedding module learns node embeddings in the
constructed drug–gene–disease interactive network, and edge embeddings are generated by edge-level network embedding module. Finally,
we can get edge embeddings that integrate edge type embeddings
and node embeddings. The ternary relation embeddings are generated
based on edge embeddings, which can be applied to drug repurposing
task. The whole architecture of RHGT is shown in Fig. 3.



_Expert Systems With Applications 190 (2022) 116165_


**Fig. 1.** Drug–gene–disease interactive network. The dotted line is the edge between
the chemical drug and the disease that we need to predict, indicating the drug can be
used to treat the disease. The blue line is the edge between drug and its target gene,
and the red line is the edge between gene and its related disease. Even edges of the
same color have multiple types.


**Fig. 2.** Three modules in RHGT model.


_4.1. Subtype-level network embedding_


The subtype-level network embedding module uses a sub-structure
to mine deeper characteristics of edge types, and finally obtains edge
type embeddings. In the drug–gene–disease interactive network, each
edge type is composed of multiple subtypes. For example, there is an
edge between a chemical drug node and a gene node, its edge type is
composed of three subtypes (i.e., decreasing reaction, increasing transport, and affecting binding). Each subtype has unique characteristics,
different subtypes reflect different interaction relationships between
two nodes. We deem that there is an association between two subtypes
if they appear in one edge type. According to the association between
subtypes, we can learn the embedding for each subtype, and then
generate edge type embeddings through subtype embeddings.
In this subsection, we first construct a subtype interactive network
with subtypes as nodes. After that, we use an attention-based method
to learn the embedding for each subtype, and then generate edge type
embeddings based on subtype embeddings.
**Subtype co-occurrence network construction.** Inspired by Mei,
Cai, Yang, and Wang (2021), we structure edge types and construct
a subtype co-occurrence network. The constructed network is shown
in Fig. 4. Each node _𝑡_ _𝑖_ ∈ _𝑇_ in the network represents a subtype,
_𝑇_ = { _𝑡_ 1 _, 𝑡_ 2 _,_ ⋯ _, 𝑡_ _𝑝_ } represents the subtype set, there are _𝑝_ subtypes in
_𝑇_ . If two subtypes _𝑡_ 1, _𝑡_ 2 appear in the same edge type, it indicates
that _𝑡_ 1 and _𝑡_ 2 have co-occurrence relationship, and they are linked by
an undirected weighted edge, where edge weight _𝑤_ _𝑡_ _𝑖_ _,𝑡_ _𝑗_ indicates their
co-occurrence rate.



3


_X. Mei et al._



_Expert Systems With Applications 190 (2022) 116165_



**Fig. 3.** Architecture of Relation-aware Heterogeneous Graph Transformer. Taking ternary relation ( _𝑐_ − _𝑔_ − _𝑑_ ) as an example, the overall framework of the model is mainly divided
into three parts: subtype-level network embedding, node-level network embedding and edge-level network embedding. _𝑣_ _𝑠_ _𝑐_ ∈ _𝑁_ ( _𝑣_ _𝑐_ ), _𝑁_ ( _𝑣_ _𝑐_ ) is the neighbor node set of node _𝑣_ _𝑐_, _𝑒_ _𝑠_ _𝑐_ _,𝑐_
is the edge between node _𝑣_ _𝑠_ _𝑐_ and node _𝑣_ _𝑐_ . _𝑒_ _𝑠_ _𝑐_ _,𝑔_ ∈ _𝑁_ ( _𝑒_ _𝑐,𝑔_ ), _𝑣_ _𝑠_ _𝑐_ _,𝑔_ is the common node between edge _𝑒_ _𝑠_ _𝑐_ _,𝑔_ and edge _𝑒_ _𝑐,𝑔_ . [⨂] represents the Hadamard product.


coefficient _𝑒_ _𝑡_ _𝑖_ _,𝑡_ _𝑗_ can be defined as:


_𝑒_ _𝑡_ _𝑖_ _,𝑡_ _𝑗_ = _𝑎𝑡𝑡_ ( **𝐡** [(] _𝑡_ _𝑖_ _[𝑙]_ [−1)] _,_ **𝐡** [(] _𝑡_ _𝑗_ _[𝑙]_ [−1)] _, 𝑤_ _𝑡_ _𝑖_ _,𝑡_ _𝑗_ ) (2)


where _𝑎𝑡𝑡_ is a feed forward neural network, **𝐡** [(] _𝑡_ _𝑖_ _[𝑙]_ [−1)] ∈ _𝑅_ _[𝑑]_ is the representation for subtype node _𝑡_ _𝑖_ at the ( _𝑙_ −1)th layer, we initialize subtype
node embedding **𝐡** [(0)] _𝑡_ _𝑖_ randomly.
Then we normalize the coefficients using the _𝑠𝑜𝑓𝑡𝑚𝑎𝑥_ function:



**Fig. 4.** Constructed subtype co-occurrence network. Take two edge types as an example
to construct a subtype interactive network, _𝑡_ _𝑖_ represents different subtypes, if _𝑡_ _𝑖_ and _𝑡_ _𝑗_
appear in one edge type, _𝑡_ _𝑖_ and _𝑡_ _𝑗_ are linked, then we can get the sub-type co-occurrence
graph on the right.


We define a vector **𝐜** = [ _𝑐_ _𝑡_ 1 _,𝑡_ 2 _, 𝑐_ _𝑡_ 1 _,𝑡_ 3 _,_ ⋯ _, 𝑐_ _𝑡_ _𝑝_ −1 _,𝑡_ _𝑝_ ] to represent the
co-occurrence frequency of all edges. For any subtype _𝑡_ _𝑖_ and _𝑡_ _𝑗_, the cooccurrence frequency _𝑐_ _𝑡_ _𝑖_ _,𝑡_ _𝑗_ is obtained by counting the number of edge
types that _𝑡_ _𝑖_ and _𝑡_ _𝑗_ occur at the same time. We define the weight of the
edge between node _𝑡_ _𝑖_ and _𝑡_ _𝑗_ as:

_𝑤_ _𝑡_ _𝑖_ _,𝑡_ _𝑗_ = ‖ **𝐜** _𝑐_ _𝑡_ _𝑖_ ‖ _,𝑡_ _𝑗_ 2 (1)


where ‖ ‖ 2 is the second norm of the vector.


**Attention-based subtype embedding.** We obtain subtype embeddings based on Graph Attention Networks (GATs) (Veličković et al.,
2017), which leverages attention mechanism on the network. We use
an _𝐿_ _𝑡_ -layer GAT to learn the importance of the neighbor nodes of each
subtype node to the current subtype node in the subtype co-occurrence
network, and aggregate embeddings for the corresponding neighbor
nodes to get a subtype node embedding. We denote the output of the
_𝑙_ th layer as **𝐡** [(] _[𝑙]_ [)], which is also the input of the ( _𝑙_ + 1)th layer. Given
a subtype node pair ( _𝑡_ _𝑖_ _, 𝑡_ _𝑗_ ), the attention coefficient _𝑒_ _𝑡_ _𝑖_ _,𝑡_ _𝑗_ represents the
importance of subtype node _𝑡_ _𝑗_ to subtype node _𝑡_ _𝑖_ . When calculating the
attention coefficient, we add edge weight, i.e., _𝑤_ _𝑡_ _𝑖_ _,𝑡_ _𝑗_ . Then the attention



_𝛼_ _𝑡_ _𝑖_ _,𝑡_ _𝑗_ = _𝑠𝑜𝑓𝑡𝑚𝑎𝑥_ ( _𝑒_ _𝑡_ _𝑖_ _,𝑡_ _𝑗_ )

= _𝑒𝑥𝑝_ ( _𝜎_ ( _𝑤_ _𝑡_ _𝑖_ _,𝑡_ _𝑗_ ⋅ **𝐚** _[𝑇]_ ⋅ [ **𝐡** [(] _𝑡_ _𝑖_ _[𝑙]_ [−1)] ∥ **𝐡** [(] _𝑡_ _𝑗_ _[𝑙]_ [−1)] ]))
~~∑~~ _𝑘_ ∈ _𝑁_ _𝑖_ _[𝑒𝑥𝑝]_ [(] _[𝜎]_ [(] _[𝑤]_ _𝑡_ _𝑖_ _,𝑡_ _𝑗_ [⋅] **[𝐚]** _[𝑇]_ [⋅] [[] **[𝐡]** _𝑡_ [(] _𝑖_ _[𝑙]_ [−1)] ∥ **𝐡** [(] _𝑡_ _𝑘_ _[𝑙]_ [−1)] ]))



(3)



where **𝐚** _[𝑇]_ ∈ _𝑅_ [2] _[𝑑]_ is a trainable weight vector, _𝜎_ is the activation
function, and ∥ is the concatenation operation.

After that, we can aggregate neighbors’ features with the corresponding coefficients to get the vector representation for subtype node
_𝑡_ _𝑖_ at the _𝑙_ th layer as:


**𝐡** [(] _𝑡_ _𝑖_ _[𝑙]_ [)] [=] _[ 𝜎]_ [(] ∑ _𝛼_ _𝑡_ _𝑖_ _,𝑡_ _𝑗_ ⋅ **𝐡** [(] _𝑡_ _𝑗_ _[𝑙]_ [−1)] ) (4)

_𝑗_ ∈ _𝑁_ _𝑖_


where _𝑁_ _𝑖_ represents all neighbor nodes of _𝑡_ _𝑖_ .
Finally, we use representation for subtype node _𝑡_ _𝑖_ at the _𝐿_ _𝑡_ th layer
to serve as the embedding for subtype node _𝑡_ _𝑖_, i.e.:


**𝐡** _𝑡_ _𝑖_ = **𝐡** _𝑡_ _𝑖_ ( _𝐿_ _𝑡_ ) (5)


**Edge embedding generation.** For an edge e, its type is _𝜙_ ( _𝑒_ ) =
_𝑡_ 1 _, 𝑡_ 2 _,_ ⋯ _, 𝑡_ _𝑛_, the embedding for _𝜙_ ( _𝑒_ ) is computed as:



**𝐡** _𝜙_ ( _𝑒_ ) = _𝜎_ ( _𝑛_ [1] _𝑡_



∑ **𝐡** _𝑡_ _𝑖_ ) (6)

_𝑡_ _𝑖_ ∈ _𝜙_ ( _𝑒_ )



where _𝑛_ _𝑡_ represents the number of subtypes in _𝜙_ ( _𝑒_ ).
Once each edge type embedding **𝐡** _𝜙_ ( _𝑒_ ) is obtained, it will be used in
drug–gene–disease network embedding as the edge type feature.



4


_X. Mei et al._


_4.2. Node-level network embedding_


The node-level network embedding module uses edge type embeddings to learn node embeddings. It mines relationships between nodes
based on characteristics of drug–gene–disease interactive network. We
propose a graph transformer (GT) based module to learn heterogeneous
network embedding. GT consists of multi-head attention mechanism
and feed forward network. The attention function can be described as

mapping a query and a set of key–value pairs to an output, where the
query, keys, values and output are all vectors. The output is computed
as a weighted sum of the values, where the weight assigned to each
value is computed by a compatibility function of the query with the
corresponding key (Vaswani et al., 2017). Due to heterogeneity of
nodes in the network, different types of nodes have different feature
spaces. Therefore, when calculating the key, query and value, we design
a node type specific transformation matrix to share the parameters
while still maintaining the features of different node types.
We apply an _𝐿_ _𝑛_ -layer GT to learn node embeddings. For each node
pair _𝑒_ = ( _𝑣_ _𝑠_ _, 𝑣_ _𝑡_ ), the corresponding meta relation is ⟨ _𝜏_ ( _𝑣_ _𝑠_ ) _, 𝜙_ ( _𝑒_ ) _, 𝜏_ ( _𝑣_ _𝑡_ )⟩, a
brief introduction to the GT layer is as follows:

**𝐡** [(] _𝑣_ _[𝑙]_ _𝑡_ [)] [←] **[𝐀𝐠𝐠𝐫𝐞𝐠𝐚𝐭𝐞]** [∀] _𝑣𝑠_ [∈] _[𝑁]_ [(] _[𝑣]_ _𝑡_ [)] (7)

( **𝐀𝐭𝐭𝐞𝐧𝐭𝐢𝐨𝐧** ( _𝑣_ _𝑠_ _, 𝑒, 𝑣_ _𝑡_ ) ⋅ **𝐌𝐞𝐬𝐬𝐚𝐠𝐞** ( _𝑣_ _𝑠_ _, 𝑒, 𝑣_ _𝑡_ ))


Then we describe three operations of **Attention**, **Message** and **Aggre-**
**gate** in detail. We first calculate _ℎ_ _𝑛_ -head attention.:


**𝐀𝐭𝐭𝐞𝐧𝐭𝐢𝐨𝐧** ( _𝑣_ _𝑠_ _, 𝑒, 𝑣_ _𝑡_ ) = (8)

_𝑠𝑜𝑓𝑡𝑚𝑎𝑥_ ∀ _𝑣𝑠_ ∈ _𝑁_ ( _𝑣_ _𝑡_ ) (∥ _𝑖_ ∈[1 _,ℎ_ _𝑛_ ] _𝐴𝑇𝑇_ − _ℎ𝑒𝑎𝑑_ _[𝑖]_ ( _𝑣_ _𝑠_ _, 𝑒, 𝑣_ _𝑡_ ))


where _𝑁_ ( _𝑣_ _𝑡_ ) represents all the neighbors of _𝑣_ _𝑡_, _ℎ_ _𝑛_ is the number of
attention heads. The _𝑖_ th attention head is defined as follows:


_𝐴𝑇𝑇_ − _ℎ𝑒𝑎𝑑_ _[𝑖]_ ( _𝑣_ _𝑠_ _, 𝑒, 𝑣_ _𝑡_ ) =



_Expert Systems With Applications 190 (2022) 116165_


**Fig. 5.** Drug–gene–disease network-based line graph. (The gray circle represents the
edge in (a).)


weight, and the corresponding features from the neighbors **Message** are
averaged, and the updated vector is obtained as:

_̃_ **𝐡** [(] _𝑣_ _[𝑙]_ _𝑡_ [)] [=] _[ 𝜎]_ [(] _[𝑀𝑒𝑎𝑛]_ [∀] _𝑣𝑠_ [∈] _[𝑁]_ [(] _[𝑣]_ _𝑡_ [)] (14)

( **𝐀𝐭𝐭𝐞𝐧𝐭𝐢𝐨𝐧** ( _𝑣_ _𝑠_ _, 𝑒, 𝑣_ _𝑡_ ) ⋅ **𝐌𝐞𝐬𝐬𝐚𝐠𝐞** ( _𝑣_ _𝑠_ _, 𝑒, 𝑣_ _𝑡_ )))

**𝐡** [(] _𝑣_ _[𝑙]_ _𝑡_ [)] [=] _[ 𝜃]_ [⋅] [(] _[𝜎]_ [(] **[𝐖]** _𝜏_ _[𝑄]_ ( _[𝑖]_ _𝑣_ _𝑠_ ) _[̃]_ **[𝐡]** _𝑣_ [(] _[𝑙]_ _𝑡_ [)] [)) + (1 −] _[𝜃]_ [)] **[𝐡]** _𝑣_ [(] _[𝑙]_ _𝑡_ [−1)] (15)


where _𝜎_ is an activation function, _𝜃_ is a trainable parameter, _[̃]_ **𝐡** [(] _𝑣_ _[𝑙]_ _𝑡_ [)] [is the]
neighbor information aggregated at the _𝑙_ th layer. By stacking _𝐿_ _𝑛_ layers,

we get the final embedding for the target node _𝑣_ _𝑡_, i.e., **𝐡** _𝑣_ _𝑡_ = **𝐡** [(] _𝑣_ _[𝐿]_ _𝑡_ _[𝑛]_ [)] .


_4.3. Edge-level network embedding_


Previous neural network-based drug repurposing methods implicitly
use genes as a bridge to learn embeddings for drugs and diseases, and
then use drug embeddings and disease embeddings to make predictions. We argue that genes are crucial in drug–gene–disease interactive
network and should be explicitly used in prediction. For example, for
a ternary relation ( _𝑐_ − _𝑔_ − _𝑑_ ), the blocked expression of gene _𝑔_ will
lead to disease _𝑑_ . If drug _𝑐_ can promote expression of gene _𝑔_, then
we can infer that drug _𝑐_ can be used to treat disease _𝑑_ based on the
relationship of edge _𝑔_ − _𝑑_ and edge _𝑐_ − _𝑔_ . Therefore, we deem that
drug–disease associations (i.e., drug–disease links) should be identified
based on relationships between drug–gene edges and gene–disease
edges. And we use edge-level network embedding module to learn edge
embeddings in drug–gene–disease interactive network.
We first use edge type embeddings and node embeddings to initialize edge embeddings. For an edge _𝑒_ = ( _𝑣_ _𝑠_ _, 𝑣_ _𝑡_ ), it is initialized as:


**𝐡** [(0)] _𝑒_ = **𝐡** _𝑣_ _𝑠_ ◦ **𝐡** _𝜙_ ( _𝑒_ ) ◦ **𝐡** _𝑣_ _𝑡_ (16)


where ◦ refers to Hadamard product.
**Drug–gene–disease network-based Line Graph.** Since Line graph
(Gross & Yellen, 2005) is an edge-centric graph that represents the
adjacency between edges of graph _𝐺_ (i.e., drug–gene–disease network),
where each node of line graph represents an edge of _𝐺_ and two
nodes of line graph are adjacent if and only if their corresponding
edges share a common endpoint in _𝐺_ . We construct drug–gene–disease
network-based line graph _𝐺_ _𝐿_, as shown in Fig. 5.
Then we use the same method as in node-level network embedding
module to learn node embeddings in the constructed line graph _𝐺_ _𝐿_,
i.e., edge embeddings in the drug–gene–disease network. The difference
is that in this module, the input meta-relation is ⟨ _𝜙_ ( _𝑒_ _𝑠_ ) _, 𝜏_ ( _𝑣_ ) _, 𝜙_ ( _𝑒_ _𝑡_ )⟩, the
number of heads is _ℎ_ _𝑒_, and the matrices that capture meta-relation
features in **Attention** and **Message** are **𝐖** _[𝐴𝑇𝑇]_ ⟨ _𝜙_ ( _𝑒_ _𝑠_ ) _,𝜏_ ( _𝑣_ ) _,𝜙_ ( _𝑒_ _𝑡_ )⟩ [= [] **[𝐖]** _𝜏_ _[𝐾]_ ( _[𝑖]_ _𝑣_ ) **[𝐡]** _[𝜙]_ [(] _[𝑒]_ _𝑠_ [)] []][ ⋅]

[ **𝐖** _[𝑄]_ _𝜏_ Finally, we can get embeddings for nodes in the constructed line ( _[𝑖]_ _𝑣_ ) **[𝐡]** _[𝜙]_ [(] _[𝑒]_ _𝑡_ [)] []] _[𝑇]_ [and] **[ 𝐖]** _[𝑀𝑆𝐺]_ ⟨ _𝜙_ ( _𝑒_ _𝑠_ ) _,𝜏_ ( _𝑣_ )⟩ [= [] **[𝐖]** _𝜏_ _[𝑉]_ ( _[𝑖]_ _𝑣_ ) **[𝐡]** _[𝜙]_ [(] _[𝑒]_ _𝑠_ [)] []][⋅][[] **[𝐖]** _𝜏_ _[𝑉]_ ( _[𝑖]_ _𝑣_ ) **[𝐡]** _[𝜙]_ [(] _[𝑒]_ _𝑠_ [)] []] _[𝑇]_ [, respectively.]
graph, i.e., embeddings for the edges in the drug–gene–disease graph,
which are then fed into the relevant model to solve downstream tasks

such as drug repurposing.



1
([ **𝐖** _[𝐾]_ _𝜏_ ( _[𝑖]_ _𝑣_ _𝑠_ ) **[𝐡]** _𝑣_ [(] _[𝑙]_ _𝑠_ [−1)] ] _[𝑇]_ ⋅ **𝐖** _[𝐴𝑇𝑇]_ ⟨ _𝜏_ ( _𝑣_ _𝑠_ _[𝑖]_ ) _,𝜙_ ( _𝑒_ ) _,𝜏_ ( _𝑣_ _𝑡_ )⟩ [⋅] [[] **[𝐖]** _𝜏_ _[𝑄]_ ( _[𝑖]_ _𝑣_ _𝑡_ ) **[𝐡]** _𝑣_ [(] _[𝑙]_ _𝑡_ [−1)] ]) ⋅ ~~√~~ _𝑑_ _𝑛_



(9)



where **𝐖** _[𝐾]_ _𝜏_ ( _[𝑖]_ _𝑣_ _𝑠_ ) [∈] _[𝑅]_ _𝑑_ × _ℎ𝑛_ _[𝑑]_ is a _𝜏_ ( _𝑣_ _𝑠_ )-type specific _𝑖_ th key vector transforma
tion matrix,layer. Similarly,transformation matrix. **𝐡** [(] _𝑣_ _[𝑙]_ _𝑡_ [−1)] **𝐖** ∈ _[𝑄]_ _𝜏_ ( _[𝑖]_ _𝑅_ _𝑣_ _𝑡_ ) _[𝑑]_ [∈] is the embedding for the node _[𝑅]_ _𝑑_ × _ℎ𝑛_ _[𝑑]_ is a _𝜏_ ( _𝑣_ _𝑡_ )-type specific _𝑣 𝑖_ _𝑠_ th query vectorat the ( _𝑙_ −1)th

meta-relation features, so that the neighbors that depend on differentWhen calculating the attention, we use **𝐖** _[𝐴𝑇𝑇]_ ⟨ _𝜏_ ( _𝑣_ _𝑠_ _[𝑖]_ ) _,𝜙_ ( _𝑒_ ) _,𝜏_ ( _𝑣_ _𝑡_ )⟩ [to capture]
meta-relations make different contributions to the target nodes. Thus,
the model can capture the unique characteristics of each meta-relation

as:


**𝐖** _[𝐴𝑇𝑇]_ ⟨ _𝜏_ ( _𝑣_ _𝑠_ _[𝑖]_ ) _,𝜙_ ( _𝑒_ ) _,𝜏_ ( _𝑣_ _𝑡_ )⟩ [= [] **[𝐖]** _𝜏_ _[𝐾]_ ( _[𝑖]_ _𝑣_ _𝑠_ ) **[𝐡]** _[𝜙]_ [(] _[𝑒]_ [)] []][ ⋅] [[] **[𝐖]** _𝜏_ _[𝑄]_ ( _[𝑖]_ _𝑣_ _𝑡_ ) **[𝐡]** _[𝜙]_ [(] _[𝑒]_ [)] []] _[𝑇]_ (10)


We concatenate _ℎ_ _𝑛_ attention heads and get the attention coefficients
between the target node _𝑣_ _𝑡_ and all neighbor nodes, then use _𝑠𝑜𝑓𝑡𝑚𝑎𝑥_ to
normalize them to get **𝐀𝐭𝐭𝐞𝐧𝐭𝐢𝐨𝐧** ( _𝑣_ _𝑠_ _, 𝑒, 𝑣_ _𝑡_ ) for each node pair. Then we
use **Message** to extract message by using the neighbor features of the
target node _𝑣_ _𝑡_, the _ℎ_ _𝑛_ -head message is calculated by:


**𝐌𝐞𝐬𝐬𝐚𝐠𝐞** ( _𝑣_ _𝑠_ _, 𝑒, 𝑣_ _𝑡_ ) =∥ _𝑖_ ∈[1 _,ℎ_ _𝑛_ ] _𝑀𝑆𝐺_ − _ℎ𝑒𝑎𝑑_ _[𝑖]_ ( _𝑣_ _𝑠_ _, 𝑒, 𝑣_ _𝑡_ ) (11)


_𝑀𝑆𝐺_ − _ℎ𝑒𝑎𝑑_ _[𝑖]_ ( _𝑣_ _𝑠_ _, 𝑒, 𝑣_ _𝑡_ ) = **𝐖** _[𝑀𝑆𝐺]_ ⟨ _𝜏_ ( _𝑣_ _𝑠_ ) _,𝜙_ _[𝑖]_ ( _𝑒_ )⟩ [⋅] [[] **[𝐖]** _𝜏_ _[𝑉]_ ( _[𝑖]_ _𝑣_ _𝑠_ ) **[𝐡]** _𝑣_ [(] _[𝑙]_ _𝑠_ [−1)] ] (12)


**𝐖** _[𝑀𝑆𝐺]_ ⟨ _𝜏_ ( _𝑣_ _𝑠_ ) _,𝜙_ _[𝑖]_ ( _𝑒_ )⟩ [= [] **[𝐖]** _𝜏_ _[𝑉]_ ( _[𝑖]_ _𝑣_ _𝑠_ ) **[𝐡]** _[𝜙]_ [(] _[𝑒]_ [)] []][ ⋅] [[] **[𝐖]** _𝜏_ _[𝑉]_ ( _[𝑖]_ _𝑣_ _𝑠_ ) **[𝐡]** _[𝜙]_ [(] _[𝑒]_ [)] []] _[𝑇]_ (13)



Similar to the processing of **Attention**, we use a meta-relation



_𝑑_ × _[𝑑]_
matrix to capture the meta-relation features, **𝐖** _[𝑉]_ _𝜏_ ( _[𝑖]_ _𝑣_ _𝑠_ ) ∈ _𝑅_



matrix to capture the meta-relation features, **𝐖** _𝜏_ ( _[𝑖]_ _𝑣_ _𝑠_ ) ∈ _𝑅_ _ℎ𝑛_ is a

_𝜏_ ( _𝑣_ _𝑠_ )-type specific _𝑖_ th value vector transformation matrix. Finally, we
concatenate message heads and get **𝐌𝐞𝐬𝐬𝐚𝐠𝐞** ( _𝑣_ _𝑠_ _, 𝑒, 𝑣_ _𝑡_ ) for each node pair.
**Aggregation.** After calculating **Attention** and **Message** of each
node pair, we need to aggregate information from the neighbors into
the target nodes. Then the attention coefficient **Attention** is used as the



5


_X. Mei et al._


**Table 2**

Statistics of the datasets.


Datasets TTD CTD


Nodes(drug) 3866 66
Nodes(gene) 638 2017
Nodes(disease) 341 4165
Edges(drug-gene) 4139 2944
Edges(drug-disease) 3481 33 566
Edge types(drug-gene) 148 267
Edge types(gene-disease) 26 1


_4.4. Drug repurposing_


After we obtain the embedding for each edge, we can identify
associations between drugs and diseases (drug–disease links) according
to relationships between the drug–gene edge and the gene–disease
edge. We generate an embedding for each ternary relation ( _𝑐_ − _𝑔_ − _𝑑_ ),
and predict whether drug _𝑐_ can be used to treat disease _𝑑_ by judging
the label of the ternary relation. The corresponding label is set to be 1 if
the ternary relation is true, otherwise 0. For a ternary relation ( _𝑐_ − _𝑔_ − _𝑑_ ),
its embedding can be obtained as:


**𝐡** _𝑐,𝑔,𝑑_ = **𝐡** _𝑒_ _𝑐,𝑔_ ◦ **𝐡** _𝑒_ _𝑔,𝑑_ (17)


where **𝐡** _𝑒_ _𝑐,𝑔_ and **𝐡** _𝑒_ _𝑔,𝑑_ are two edge embeddings obtained through GT
based edge embedding approach. Then we feed **𝐡** _𝑐,𝑔,𝑑_ into a binary
classifier to predict whether drug _𝑐_ could be used to treat disease _𝑑_ .
The loss function is defined as the binary cross-entropy error over all
labeled ternary relation as:

_𝐿_ = − _𝑚_ [1] _[𝛴]_ _𝑖_ _[𝑚]_ =1 [[] **[𝐲]** _[𝑖]_ _[𝑙𝑜𝑔]_ **[𝐳]** _[𝑖]_ [+ (1 −] **[𝐲]** _[𝑖]_ [)] _[𝑙𝑜𝑔]_ [(1 −] **[𝐳]** _[𝑖]_ [)]] (18)


where **𝐲** _[𝑖]_ and **𝐳** _[𝑖]_ are the true labels and embeddings for ternary relations,
**𝐳** _[𝑖]_ = **𝐂** ⋅ **𝐡** _[𝑖]_, **𝐂** is the parameter of the binary classifier, **𝐡** _[𝑖]_ is the
embedding for ternary relation _𝑖_ . With the guide of labeled data, we
can optimize the model via back propagation and learn embeddings
for ternary relations.


**5. Experiment and evaluations**


In this section, we evaluate our proposed RHGT model on CTD
and TTD datasets. We also adjust the parameters to investigate the
sensitivity of the parameters.


_5.1. Datasets_


**CTD** [2] **(Comparative Toxicogenomics Database).** CTD is a robust,
publicly available database that aims to advance understanding about
how environmental exposures affect human health. It provides manually curated information about chemical–gene, chemical–disease and
gene–disease relationships. These data can be used to study correlation
between chemistry, genes, phenotypes, diseases and the environment,
which can promote people’s understanding of chemical drugs and
human health.


**TTD** [3] **(Therapeutic Target Database).** A database to provide information about the known and explored therapeutic protein and nucleic acid targets, the targeted disease, pathway information and the
corresponding drugs directed at each of these targets.

We construct two heterogeneous networks corresponding to the
two datasets respectively. Statistics of the two networks are listed in
Table 2.


2 [http://ctdbase.org/.](http://ctdbase.org/)
3 [http://db.idrblab.net/ttd/.](http://db.idrblab.net/ttd/)



_Expert Systems With Applications 190 (2022) 116165_


_5.2. Evaluation metrics_


We evaluate experimental performance with three criteria: area
under the receiver operating characteristic curve (AUROC), Precision
and F1 Score, which are widely used for drug indication prediction
tasks (Patel & Guo, 2021; Wang et al., 2020; Wang, Zhou et al., 2020).

**AUROC.** In AUROC, the Receiver Operating Characteristics is the
probability curve, and Area Under Curve is the degree of separability.
AUROC is used to evaluate the binary classification model, it reflects
quality of the model with intuitive values. ROC curve is plotted with
True Positive Rate (TPR) against the False Positive Rate (FPR).


_𝑇𝑟𝑢𝑒𝑃𝑜𝑠𝑖𝑡𝑖𝑣𝑒_
_𝑇𝑃𝑅_ = (19)
_𝑇𝑟𝑢𝑒𝑃𝑜𝑠𝑖𝑡𝑖𝑣𝑒_ + _𝐹𝑎𝑙𝑠𝑒𝑁𝑒𝑔𝑎𝑡𝑖𝑣𝑒_


_𝐹𝑎𝑙𝑠𝑒𝑃𝑜𝑠𝑖𝑡𝑖𝑣𝑒_
_𝐹𝑃𝑅_ = (20)
_𝐹𝑎𝑙𝑠𝑒𝑃𝑜𝑠𝑖𝑡𝑖𝑣𝑒_ + _𝑇𝑟𝑢𝑒𝑁𝑒𝑔𝑎𝑡𝑖𝑣𝑒_


where True Positive is the number of node pairs correctly predicted as
linked, True Negative is the number of node pairs correctly predicted
as not linked, False Positive is the number of node pairs incorrectly
predicted as linked, and False Negative is the number of node pairs
incorrectly predicted as not linked.

**Precision.** Precision refers to proportion of the correct positive class
in the sample predicted to be the positive class. The higher the value
of Precision, the better the prediction performance.


_𝑇𝑟𝑢𝑒𝑃𝑜𝑠𝑖𝑡𝑖𝑣𝑒_
_𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛_ = (21)
_𝑇𝑟𝑢𝑒𝑃𝑜𝑠𝑖𝑡𝑖𝑣𝑒_ + _𝐹𝑎𝑙𝑠𝑒𝑃𝑜𝑠𝑖𝑡𝑖𝑣𝑒_


**F1 Score.** F1 Score is the harmonic mean of Precision and Recall.

Recall is the proportion of correctly predicted node pairs among all the
actual linked node pairs.


_𝑇𝑟𝑢𝑒𝑃𝑜𝑠𝑖𝑡𝑖𝑣𝑒_
_𝑅𝑒𝑐𝑎𝑙𝑙_ = (22)
_𝑇𝑟𝑢𝑒𝑃𝑜𝑠𝑖𝑡𝑖𝑣𝑒_ + _𝐹𝑎𝑙𝑠𝑒𝑁𝑒𝑔𝑎𝑡𝑖𝑣𝑒_


_𝐹_ 1 _𝑆𝑐𝑜𝑟𝑒_ = [2 ×] _[ 𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛]_ [×] _[ 𝑅𝑒𝑐𝑎𝑙𝑙]_ (23)

_𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛_ + _𝑅𝑒𝑐𝑎𝑙𝑙_


_5.3. Baselines_


To test effectiveness of the proposed RHGT model, we compare
it with other state-of-the-art graph neural network methods including
heterogeneous graph neural network methods and homogeneous graph
neural network methods for drug repurposing, which we treat it as a
link prediction problem. Besides, we also compare our RHGT model
with some classic knowledge graph(KG) embedding models, which are
widely used to compute scores of knowledge triples. In drug repurposing task, a triple(c,g,d) consists of a drug, a gene and a disease. Drugs
and diseases represent entities, genes represent relations.

∙ TransE (Bordes, Usunier, Garcia-Duran, Weston, & Yakhnenko,
2013): It is a translational distance-based model, and it is widely used
in link prediction.

∙ DisMult (Yang, Yih, He, Gao, & Deng, 2014): It matches the latent
semantics in the embedding space to compute scores of triples.

∙ ComplEx (Trouillon, Welbl, Riedel, Gaussier, & Bouchard, 2016):
It is an extension of DistMult which embeds entities and relations into

the complex space.

∙ ConvE (Dettmers, Minervini, Stenetorp, & Riedel, 2018): It is a
popular convolutional network-based model for link prediction.

∙ GCN (Kipf & Welling, 2016)(Graph Convolutional Networks): It is
a method designed for homogeneous graph. GCN averages the neighbor’s embedding followed by linear projection.

∙ GraphSAGE (Hamilton et al., 2017)(Graph SAmple and aggreGatE): It is a method based on graph convolutional network which
designed for homogeneous graphs. GraphSAGE expands GCN into an
inductive learning task by aggregating feature information of neighbors
of the nodes, which can generalize unknown nodes.

∙ GAT (Veličković et al., 2017)(Graph Attention Networks): It is an
attention-based method which designed for homogeneous graphs. GAT
adopts multi-head additive attention on neighbors.



6


_X. Mei et al._


**Fig. 6.** Test AUROC of RHGT with different dimensions of the hidden layer on TTD.


**Table 3**

Parameter values.


Parameters TTD CTD


Train:Validation:Test 7:1:2 8:1:1

Positive:Negative 1:1

Hidden dimension 256

Final dimension 128

Dropout 0.2

Heads 8


∙ BiFusion (Veličković et al., 2017): It is a bipartite graph convolution network model for drug repurposing through heterogeneous
information fusion.

∙ RGCN (Schlichtkrull et al., 2018)(Relational Graph Convolutional
Networks): It is a method based on graph convolutional network which
designed for heterogeneous networks. RGCN keeps a different weight
for each relationship.
∙ HGT (Hu et al., 2020)(Heterogeneous graph transformer): It is a
method based on graph transformer which designed for heterogeneous
networks. It uses parameters related to node and edge type to maintain
the type features, and to characterize the attention on each edge in the
heterogeneous network.


_5.4. Implementation details_


For the two datasets, we perform negative sampling based on the
original heterogeneous network, and the ratio of positive and negative
samples is 1:1. The ratio of training set, validation set, and test set is set
to 7:1:2 in TTD and 8:1:1 in CTD. We use Deepwalk (Perozzi, Al-Rfou, &
Skiena, 2014) to initialize node representations, initialize the remaining
parameters randomly, and use Adam to optimize the model. We use
Dropout (Goodfellow, Bengio, & Courville, 2016; Phaisangittisagul,
2016) and early stopping to avoid overfitting. The Dropout value is set
to 0.2. For each model, we train it for 200 epochs and stop training if
the validation loss does not decrease for 10 consecutive epochs.
For all comparison methods, we use the same parameter settings.
We set the hidden dimension to 256 and the final embedding dimension
to 128. For all methods based on multi-head attention, we set the

number of heads as 8. Parameter values are summarized in Table 3.


_5.5. Parameter settings_


In this section, we investigate sensitivity of parameters.
∙ Dimension of the hidden layer. Fig. 6 shows effect with different
dimensions of the hidden layer on TTD. We can see that as the hidden
layer embedding dimension increases, the AUROC gradually rises first.



_Expert Systems With Applications 190 (2022) 116165_


**Fig. 7.** Test AUROC of RHGT with different dimensions of the final layer on TTD.


**Table 4**

Results of variants of our model on TTD test set.


Metrics RHGT RHGT _𝑛𝑑_ RHGT _𝑒𝑔_


AUROC **0.7342** 0.7205 0.7105

F1 Score **0.7611** 0.7553 0.7487

Precision **0.7543** 0.7347 0.7239


When the dimension is 256, the AUROC reaches the maximum, and
when the dimension increases to 512, the AUROC slowly decreases.
Therefore, we set the hidden layer embedding dimension of the model
to 256. When the dimension changes between 16 and 512, the fluctuation of the test AUROC is small, which shows that the current prediction
task is not sensitive to the hidden layer dimension.
∙ Dimension of the final embedding. Fig. 7 shows effect with different dimensions of the final edge embedding on TTD. We can find
that the test AUROC of RHGT increases as the final edge embedding
dimension increases and reaches the maximum when the dimension is

128. When the dimensionality continues to increase, the test AUROC
decreases, which may be due to overfitting. Therefore, we set the final
edge embedding dimension of the RHGT model to 128. When the
dimension changes from 16 to 512, the test AUROC fluctuates greatly,
which shows that the current prediction task is sensitive to the final
edge embedding dimension.
Also, we obtain the above two values of parameters on CTD dataset
in a similar way, and the best result is the same as on TTD dataset.


_5.6. Model validation_


In order to verify the effectiveness of each module in RHGT model,
we test two variants of RHGT: RHGT _𝑛𝑑_ and RHGT _𝑒𝑔_ .
∙ RHGT _𝑛𝑑_ : It removes node-level network embedding module in
RHGT model.

∙ RHGT _𝑒𝑔_ : It removes edge-level network embedding module in
RHGT model.

∙ RHGT: The proposed relation-aware heterogeneous graph transformer model which employs GT based node embedding and GT based
edge embedding simultaneously.

Tables 4 and 5 show the results on the CTD and TTD datasets,
respectively. Among RHGT and its two variants, RHGT performs best on
the two datasets, followed by RHGT _𝑛𝑑_, and RHGT _𝑒𝑔_ performs worst. It
indicates the importance of learning network embeddings at node-level
and edge-level, and also shows that edge-level network embedding
module is more effective than node-level network embedding module.



7


_X. Mei et al._


**Table 5**

Results of variants of our model on CTD test set.


Metrics RHGT RHGT _𝑛𝑑_ RHGT _𝑒𝑔_


AUROC **0.7809** 0.7737 0.7734

F1 score **0.7754** 0.7687 0.7703

Precision **0.7957** 0.7866 0.7813


**Table 6**

Performance of different methods on TTD.


Methods AUROC F1 score Precision


TransE 0.5421 0.6064 0.5693

ComplEx 0.5773 0.6223 0.5942

Dismult 0.5803 0.6237 0.5995

ConvE 0.6154 0.6411 0.6404


GraphSAGE 0.6186 0.6535 0.6528

GCN 0.6357 0.6599 0.6738

GAT 0.6367 0.6605 0.6751


BiFusion 0.6519 0.6893 0.6762

RGCN 0.6631 0.7075 0.6833

HGT 0.6705 0.7046 0.6966

RHGT **0.7342** **0.7611** **0.7543**


_5.7. Comparison with baselines_


Tables 6 and 7 show performance of different state-of-the-art methods on CTD and TTD datasets, respectively. The first part of Tables 6
and 7 reports performance of several classic knowledge graph embedding methods, and the second part shows performance of homogeneous
graph neural network methods. The last part of the two tables compare our model against related heterogeneous graph neural network
methods. Our model achieves best performance on both datasets. In
the comparison methods, heterogeneous graph neural network methods including HGT and RGCN outperform homogeneous graph neural
network methods: GAT, GCN and GraphSAGE. The homogeneous graph
neural network methods treat all nodes equally, while the heterogeneous neural network methods use the type feature of nodes and
edges to better model the characteristics of heterogeneous networks.
HGT performs better than RGCN and BiFusion, it indicates that graph
transformer network based on attention mechanism performs better
than graph convolutional network when learning features of relations
in heterogeneous networks. Between the two methods based on graph
convolutional networks, RGCN performs better on TTD, while BiFusion
performs better on CTD, which may be attributed to the lack of edge
type information in CTD dataset. The three methods designed for
homogeneous graphs performed poorly. Among them, GAT, which uses
attention mechanism, outperforms the other two convolutional neural
network methods: GCN and GraphSAGE. Compared with several graph
neural network methods, the four popular KG embedding methods
perform poorly. Among them, the neural network-based ConvE performs best, followed by the two semantic matching based methods
DisMult and ComplEx, and the worst performance is the translation
distance-based method TransE.

In CTD dataset, only 66 drug nodes were selected randomly. As
can be seen from Table 2, the density of this graph was very large,
so the performance of all comparison methods on CTD was generally
better than that on TTD. However, the lack of gene–disease edge type
information in CTD could affect the performance of models that depend
on edge relationships.


_5.8. Proportions of the training data_


In order to evaluate effect of the labeled data size, we test the
proposed RHGT model and the other comparison methods with different proportions of training data. Fig. 8 shows test AUROC with
different proportions of the training data on TTD. We can see that as the



_Expert Systems With Applications 190 (2022) 116165_


**Table 7**

Performance of different methods on CTD.


Methods AUROC F1 score Precision


TransE 0.6503 0.6637 0.6587

ComplEx 0.6903 0.6845 0.6914

Dismult 0.7058 0.7132 0.7147

ConvE 0.7138 0.7161 0.7232


GraphSAGE 0.7284 0.7226 0.7388

GCN 0.7277 0.7182 0.7449

GAT 0.7353 0.7323 0.7414


BiFusion 0.7591 0.7534 0.7625

RGCN 0.7501 0.7474 0.7563

HGT 0.7655 0.7593 0.7702

RHGT **0.7809** **0.7754** **0.7957**


**Fig. 8.** Test AUROC of models with different proportions of training data on TTD.


proportion of training data increases, the test AUROC of each model is
on the rise. RHGT performs optimally with different proportions of the
training data. When the proportion of training data increases to 0.3,
the performance of RHGT is significantly better than other methods.
As the proportion of training data increases, this superiority becomes
more obvious. This shows that our model has a good performance at
any proportion of training data, and it can highlight the advantages of
our model when there are enough training data.


**6. Conclusion**


In this paper, we propose a Relation-aware Heterogeneous Graph
Transformer (RHGT) model for drug repurposing, which identifies
drug–disease associations (i.e., drug–disease links) based on relationships between drug–gene edges and gene–disease edges. In order to
capture heterogeneous characteristics of drug–gene–disease interactive
network more comprehensively, we propose a three-level network
structure. We learn edge type embeddings at fine-grained subtype-level,
node embeddings at node-level, and edge embeddings at coarse-grained
edge-level. The output of subtype-level is the input of node-level and
edge-level, and the output of node-level is the input of edge-level,
so as to transfer information level by level. Edge type embeddings
obtained at subtype-level are used when learning node embeddings and
edge embeddings, which determines indispensability of subtype-level.
Performance of variants of our model also demonstrates the importance
of node-level and edge-level.

In the future, we will explore using our RHGT model to combine more related information such as drug–drug associations and
gene–gene associations to achieve better results.



8


_X. Mei et al._


**CRediT authorship contribution statement**


**Xin Mei:** Software, Investigation, Writing – original draft. **Xiaoyan**
**Cai:** Conceptualization, Methodology, Writing – reviewing and editing. **Libin Yang:** Supervision, Funding acquisition. **Nanxin Wang:**
Software.


**Declaration of competing interest**


The authors declare that they have no known competing financial interests or personal relationships that could have appeared to
influence the work reported in this paper.


**Acknowledgments**


This work has been supported by the following grants: National Key
Research and Development Project of China (no. 2018YFB1402600),
National Natural Science Foundation of China (nos. 61872296,
61772429, U20B2065), MOE (Ministry of Education in China) Project
of Humanities and Social Sciences (no. 18YJC870001) and the Fundamental Research Funds for the Central Universities, China
(3102019DHKY04).


**References**


[Adams, C. P., & Brantner, V. V. (2006). Estimating the cost of new drug development:](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb1)

is it really $802 million? _[Health Affairs](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb1)_, _25_, 420–428.
[Ashburn, T. T., & Thor, K. B. (2004). Drug repositioning: identifying and developing](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb2)

new uses for existing drugs. _[Nature Reviews Drug Discovery](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb2)_, _3_, 673–683.
[Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J., & Yakhnenko, O. (2013).](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb3)

[Translating embeddings for modeling multi-relational data. In](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb3) _Advances in neural_
_[information processing systems 26](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb3)_ .
[Cheng, F., Desai, R. J., Handy, D. E., Wang, R., Schneeweiss, S., Barabasi, A. L., et al.](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb4)

[(2018). Network-based approach to prediction and population-based validation of](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb4)
in silico drug repurposing. _[Nature Communications](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb4)_, _9_, 1–12.
[Cheng, F., & Zhao, Z. (2014). Machine learning-based prediction of drug–drug in-](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb5)

[teractions by integrating drug phenotypic, therapeutic, chemical, and genomic](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb5)
properties. _[Journal of the American Medical Informatics Association](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb5)_, _21_, e278–e286.
[Chousterman, B. G., Swirski, F. K., & Weber, G. F. (2017). Cytokine storm and sepsis](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb6)

disease pathogenesis. In _[Seminars in immunopathology](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb6)_ (pp. 517–528). Springer.
[Dettmers, T., Minervini, P., Stenetorp, P., & Riedel, S. (2018). Convolutional 2d](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb7)

knowledge graph embeddings. In _[Thirty-second](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb7)_ _AAAI_ _conference_ _on_ _artificial_
_[intelligence](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb7)_ .
[Dickson, M., & Gagnon, J. P. (2009). The cost of new drug discovery and development.](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb8)

_[Discovery Medicine](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb8)_, _4_, 172–179.
[DiMasi, J. A., Hansen, R. W., & Grabowski, H. G. (2003). The price of innovation: new](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb9)

estimates of drug development costs. _[Journal of Health Economics](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb9)_, _22_, 151–185.
Doshi, S., & Chepuri, S. P. (2020). Dr-covid: Graph neural networks for sars-cov-2 drug

[repurposing. arXiv preprint arXiv:2012.02151.](http://arxiv.org/abs/2012.02151)
[Goodfellow, I., Bengio, Y., & Courville, A. (2016). Regularization for deep learning.](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb11)

_[Deep Learning](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb11)_, 216–261.
[Gordon, D. E., Jang, G. M., Bouhaddou, M., Xu, J., Obernier, K., White, K. M., et al.](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb12)

[(2020). A sars-cov-2 protein interaction map reveals targets for drug repurposing.](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb12)
_Nature_, _[583](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb12)_, 459–468.
[Gottlieb, A., Stein, G. Y., Ruppin, E., & Sharan, R. (2011). Predict: a method](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb13)

[for inferring novel drug indications with application to personalized medicine.](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb13)
_[Molecular Systems Biology](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb13)_, _7_, 496.
Gross, J. L., & Yellen, J. (2005). _[Graph theory and its applications](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb14)_ . CRC Press.
[Guney, E., Menche, J., Vidal, M., & Barábasi, A. L. (2016). Network-based in silico](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb15)

drug efficacy screening. _[Nature Communications](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb15)_, _7_, 1–13.



_Expert Systems With Applications 190 (2022) 116165_


Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on

[large graphs. arXiv preprint arXiv:1706.02216.](http://arxiv.org/abs/1706.02216)
Hu, Z., Dong, Y., Wang, K., & Sun, Y. (2020). Heterogeneous graph transformer. In

_Proceedings of the web conference 2020_ (pp. 2704–2710).
[Iorio, F., Bosotti, R., Scacheri, E., Belcastro, V., Mithbaokar, P., Ferriero, R., et](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb18)

[al. (2010). Discovery of drug mode of action and drug repositioning from](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb18)
transcriptional responses. _[Proceedings of the National Academy of Sciences](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb18)_, _107_,

[14621–14626.](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb18)

[Jockusch, S., Tao, C., Li, X., Anderson, T. K., Chien, M., Kumar, S., et al. (2020). A](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb19)

[library of nucleotide analogues terminate rna synthesis catalyzed by polymerases of](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb19)
[coronaviruses that cause sars and covid-19.](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb19) _Antiviral Research_, _180_, Article 104857.
Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph
[convolutional networks. arXiv preprint arXiv:1609.02907.](http://arxiv.org/abs/1609.02907)
[Li, J., & Lu, Z. (2012). A new method for computational drug repositioning using](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb21)

drug pairwise similarity. In _[2012 IEEE international conference on bioinformatics and](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb21)_
_biomedicine_ [(pp. 1–4). IEEE.](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb21)
[Lotfi Shahreza, M., Ghadiri, N., Mousavi, S. R., Varshosaz, J., & Green, J. R.](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb22)

[(2018). A review of network-based approaches to drug repositioning.](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb22) _Briefings in_
_Bioinformatics_, _[19](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb22)_, 878–892.
[Luo, H., Wang, J., Li, M., Luo, J., Peng, X., Wu, F. X., et al. (2016). Drug repositioning](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb23)

[based on comprehensive similarity measures and bi-random walk algorithm.](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb23)
_Bioinformatics_, _[32](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb23)_, 2664–2671.
[Mei, X., Cai, X., Yang, L., & Wang, N. (2021). Graph transformer networks based text](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb24)

representation. _[Neurocomputing](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb24)_, _463_, 91–100.
[Napolitano, F., Zhao, Y., Moreira, V. M., Tagliaferri, R., Kere, J., D’Amato, M., et al.](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb25)

[(2013). Drug repositioning: a machine-learning approach through data integration.](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb25)
_[Journal of Cheminformatics](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb25)_, _5_, 1–9.
Patel, R., & Guo, Y. (2021). Graph based link prediction between human phenotypes

[and genes. arXiv preprint arXiv:2105.11989.](http://arxiv.org/abs/2105.11989)
Perozzi, B., Al-Rfou, R., & Skiena, S. (2014). Deepwalk: Online learning of social

representations, In _Proceedings of the 20th ACM SIGKDD international conference on_
_Knowledge discovery and data mining_ (pp. 701–710).
[Phaisangittisagul, E. (2016). An analysis of the regularization between l2 and dropout in](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb28)

single hidden layer neural network. In _[2016 7th international conference on intelligent](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb28)_
_[systems, modelling and simulation (ISMS)](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb28)_ (pp. 174–179). IEEE.
[Pushpakom, S., Iorio, F., Eyers, P. A., Escott, K. J., Hopper, S., Wells, A., et al. (2019).](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb29)

[Drug repurposing: progress, challenges and recommendations.](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb29) _Nature Reviews Drug_
_Discovery_, _[18](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb29)_, 41–58.
[Schlichtkrull, M., Kipf, T. N., Bloem, P., Van Den Berg, R., Titov, I., & Welling, M.](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb30)

[(2018). Modeling relational data with graph convolutional networks. In](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb30) _European_
_semantic web conference_ [(pp. 593–607). Springer.](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb30)
[Trouillon, T., Welbl, J., Riedel, S., Gaussier, É., & Bouchard, G. (2016). Complex](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb31)

embeddings for simple link prediction. In _[International conference on machine learning](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb31)_
[(pp. 2071–2080). PMLR.](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb31)
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., et al.

[(2017). Attention is all you need. arXiv preprint arXiv:1706.03762.](http://arxiv.org/abs/1706.03762)
Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2017).

[Graph attention networks. arXiv preprint arXiv:1710.10903.](http://arxiv.org/abs/1710.10903)
[Wang, W., Lv, H., Zhao, Y., Liu, D., Wang, Y., & Zhang, Y. (2020). Dls: a link prediction](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb34)

[method based on network local structure for predicting drug-protein interactions.](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb34)
_[Frontiers in Bioengineering and Biotechnology](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb34)_, _8_, 330.
[Wang, Z., Zhou, M., & Arnold, C. (2020). Toward heterogeneous information fusion:](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb35)

[bipartite graph convolutional networks for in silico drug repurposing.](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb35) _Bioinformatics_,
_36_ [, i525–i533.](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb35)
Yang, B., Yih, W. t., He, X., Gao, J., & Deng, L. (2014). Embedding entities and relations

[for learning and inference in knowledge bases. arXiv preprint arXiv:1412.6575.](http://arxiv.org/abs/1412.6575)
[Yella, J. K., Yaddanapudi, S., Wang, Y., & Jegga, A. G. (2018). Changing trends in](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb37)

[computational drug repositioning.](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb37) _Pharmaceuticals_, _11_, 57.
[Zaim, S., Chong, J. H., Sankaranarayanan, V., & Harky, A. (2020). Covid-19 and](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb38)

multi-organ response. _[Current Problems in Cardiology](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb38)_, Article 100618.
[Zeng, X., Zhu, S., Liu, X., Zhou, Y., Nussinov, R., & Cheng, F. (2019). Deepdr: a network-](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb39)

[based deep learning approach to in silico drug repositioning.](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb39) _Bioinformatics_, _35_,

[5191–5198.](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb39)

[Zhang, P., Agarwal, P., & Obradovic, Z. (2013). Computational drug repositioning](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb40)

[by ranking and integrating multiple data sources. In](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb40) _Joint European conference on_
_[machine learning and knowledge discovery in databases](http://refhub.elsevier.com/S0957-4174(21)01487-1/sb40)_ (pp. 579–594). Springer.



9


