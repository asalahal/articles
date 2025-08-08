## Emerging Drug Interaction Prediction Enabled by Flow-based Graph Neural Network with Biomedical Network

Yongqi Zhang [a], Quanming Yao [∗][b], Ling Yue [b], Xian Wu [c], Ziheng
Zhang [c], Zhenxi Lin [c], and Yefeng Zheng [c]


a _4Paradigm Inc., Beijing, China_
b _Department of Electronic Engineering, Tsinghua University, Beijing, China_
c _Tencent Jarvis Lab, Shenzhen, China_


**Abstract**


Accurately predicting drug-drug interactions (DDI) for emerging
drugs, which offer possibilities for treating and alleviating diseases,
with computational methods can improve patient care and contribute
to efficient drug development. However, many existing computational
methods require large amounts of known DDI information, which is
scarce for emerging drugs. In this paper, we propose EmerGNN, a
graph neural network (GNN) that can effectively predict interactions for
emerging drugs by leveraging the rich information in biomedical networks.
EmerGNN learns pairwise representations of drugs by extracting the paths
between drug pairs, propagating information from one drug to the other,
and incorporating the relevant biomedical concepts on the paths. The
different edges on the biomedical network are weighted to indicate the
relevance for the target DDI prediction. Overall, EmerGNN has higher
accuracy than existing approaches in predicting interactions for emerging
drugs and can identify the most relevant information on the biomedical
network.


∗ Corresponding author: qyaoaa@tsinghua.edu.cn


1


### **1 Introduction**

Science advancements and regulatory changes have led to the development
of numerous emerging drugs worldwide, particularly for rare, severe, or lifethreatening illnesses [1, 2]. These drugs are novel substances with unknown
or unpredictable risks, as they have not been extensively regulated or used
before. For instance, although hundreds of COVID-19 drugs have been
developed, only six have been recommended by the FDA as of Oct 2023, such
as dexamethasone and hydrocortisone. Clinical deployment of new drugs is
cautious and slow, making it crucial to identify drug-drug interactions (DDIs)
for emerging drugs. To speed up the discovery of potential DDIs, computational
techniques, particularly machine learning approaches, have been developed [3–
6]. However, with limited clinical trial information, unexpected polypharmacy
or side effects can be severe and difficult to detect [7, 8].
Early DDI prediction methods used fingerprints [9] or hand-designed features

[4, 10] to indicate interactions based on drug properties. Although these
methods can work directly on emerging drugs in a cold-start setting [10, 11], they
can lack expressiveness and ignore the mutual information between drugs. DDI
facts can naturally be represented as a graph where nodes represent drugs and
edges represent interactions between a pair of drugs. Graph learning methods
can learn drug embeddings for prediction [12], but they rely on historical
interactions, thus cannot address the problem of scarce interaction data for
emerging drugs.
Incorporating large biomedical networks as side information for DDI
prediction is an alternative to learning solely from DDI interactions [5, 6, 13–17].
These biomedical networks, such as HetioNet [18], organize facts into a directed
multi-relational graph, recording relationships between biomedical concepts,
such as genes, diseases, and drugs. Tanvir et. al. used hand-designed metapaths from the biomedical network [5], while Karim et. al. learned embeddings
from the network and used a deep network to do DDI prediction [14]. Graph
neural networks [19, 20] can obtain expressive node embeddings by aggregating
topological structure and drug embeddings, but existing methods [6, 13, 15–
17] do not specially consider emerging drugs, leading to poor performance in
predicting DDIs for them.
Here, we propose to use large biomedical network to predict DDI for
emerging drugs by learning from the biomedical concepts connecting target
drugs pairs. Although emerging drugs may not have sufficient interactions in the
DDI network, they often share the same biochemical concepts used in the drug
development with existing drugs, such as targeted genes or diseases. Therefore,
we exploit related paths from the biomedical networks for given drug pairs.
However, properly utilizing these networks can be challenging as they are not
developed for emerging drugs, and the mismatch of objectives can lead machine
learning models to learn distracting knowledge.
To accurately and interpretable predict DDI for emerging drugs, we
introduce EmerGNN, a GNN method that learns pair-wise drug representations
by integrating the biomedical entities and relations connecting them. A flow

2


based GNN architecture extracts paths connecting drug pairs, traces from an
emerging drug to an existing drug, and integrates information of the biomedical
concepts along the paths. This approach utilizes shared information in both
biomedical and interaction networks. To extract relevant information, we weight
different types of relations on the biomedical network, and edges with larger
weights on the paths are helpful for interpretation. Compared with other GNNbased methods, EmerGNN propagates on the local subgraph around the drug
pair to be predicted and better discovers directional information flow within the
biomedical network. In summary, our main contributions are as follows:


  - Building upon a biomedical network, we develop an effective deep learning
method that predicts interactions for emerging drugs accurately.


  - We propose EmerGNN, a GNN-based method that learns pair-wise
representations of drug pairs to predict DDIs for emerging drugs by
integrating the relevant biomedical concepts connecting them.


  - Extensive experiments show that EmerGNN is effective in predicting
interactions for emerging drugs. The learned concepts on the biomedical
network are interpretable.


  - EmerGNN’s strong prediction ability has the potential to clinically
improve patient care and contribute to more efficient drug development

processes.

### **2 Results**


**EmerGNN: encoding pair-wise representations with flow-based GNN**
**for emerging drugs.** We focus on two DDI prediction task settings for
emerging drugs [10, 11, 21] (Fig. 1a, Method): S1 setting, determining the
interaction type between an emerging drug and an existing drug, and S2 setting,
determining the interaction type between two emerging drugs. To connect
emerging and existing drugs, we use a large biomedical network HetioNet [18],
which contains entities and relations related to biomedical concepts. We assume
that all the emerging drugs are connected to entities in the biomedical network,
allowing us to infer their properties from existing drugs and the biomedical
network.
Given the DDI network and biomedical network (Fig. 1a), we firstly integrate
the two networks to enable communication between existing and emerging drugs
connected by biomedical concepts, such as proteins, diseases or other drugs,
and then add inverse edges by introducing inverse types for each relation and
interaction type. The two steps generate an augmented network where the drugs
and biomedical entities can communicate better (Fig. 1b). For a target drug
pair to be predicted (for example an emerging drug _u_ and an existing drug
_v_ ), we extract all the paths with length no longer than _L_ between them, and
combine the paths to form a path-based subgraph _G_ _u,v_ _[L]_ [(Fig. 1c). The value of] _[ L]_
is a hyper-parameter to be tuned (Supplementary Table 2). A flow-based GNN


3


_g_ ( _·_ ; _**θ**_ ) with parameters _**θ**_ (Fig. 1d) is applied on _G_ _u,v_ _[L]_ [to trace drug features]
_**h**_ [0] _u,u_ [=] _**[ f]**_ _[u]_ [(like fingerprints) along the biomedical edges and integrate essential]
information along the path. In each iteration _ℓ_, the GNN flows to drug-specific
entities that are _ℓ_ -steps away from drug _u_ and ( _L −_ _ℓ_ )-steps away from drug
_v_ in the augmented network. An attention mechanism is applied on the edges
in _G_ _u,v_ _[L]_ [to adjust their importance. The GNN iterates] _[ L]_ [ steps to return the]

pair-wise representation _**h**_ [(] _u,v_ _[L]_ [)] [. Finally,] _**[ h]**_ [(] _u,v_ _[L]_ [)] [is fed to a linear classifier] _[ p]_ [(] _[·]_ [) to]
predict the interaction type between _u_ and _v_ (Fig. 1e).


**Comparison of EmerGNN to baseline methods in DDI prediction.**
Two public datasets DrugBank [22] and TWOSIDES [23] are used. The original
drug set is split into three parts with a ratio of 7:1:2 for training, validation,
and testing (Method). The drugs in validation and testing sets are considered
emerging drugs for validation and testing, respectively. For the DrugBank
dataset, there is at most one interaction type between any drug pair, and the
task is to predict the exact type in a multi-type classification setting. Macro F1score, Accuracy, and Cohen’s Kappa [24] are used as performance metrics, with
F1-score as the primary metric. For TWOSIDES dataset, there may be multiple
interaction types between a drug pair, and the task is to predict whether a pair
of drugs will have a certain interaction type under a binary classification setting.
PR-AUC, ROC-AUC and Accuracy are used to evaluate the performance, with
PR-AUC being the primary metric.
In S1 setting, _Emb_ type methods, particularly `MSTE`, poorly predict emerging
drugs because their embeddings are not updated during training. `KG-DDI`
performs better as it updates drug embeddings with information in the
biomedical network. For _DF_ methods, `CSMDDI` and `STNN-DDI` outperform MLP
on DrugBank dataset with their designed training schemes in a cold-start
setting, but they do not perform well on TWOSIDES with more interaction
types. `HIN-DDI` outperforms `MLP`, indicating that graph features from biomedical
network can benefit DDI prediction. Deep _GNN_ -based methods may not
perform better than _DF_ methods on DrugBank since the GNN-based methods
may not well capture the crucial property of similarity for emerging drug
prediction (Fig. 3). `CompGCN`, `Decagon` and `KGNN` perform comparably due to
their similar GNN architecture design. `SumGNN` constrains message passing in
the enclosing subgraph between drug pairs, making information more focused.
`DeepLGF` is the best GNN-based baseline by fusing information from multiple
sources, taking the advantage of both drug features and graph features. `EmerGNN`
significantly outperforms all compared methods as indicated by the small pvalues under two-sided t-testing of statistical significance. First, by learning
paths between emerging and existing drugs, it can capture the graph features,
whose importance has been verified by the _GF_ method `HIN-DDI` . Second,
different from `CompGCN`, `Decagon`, `KGNN`, and `DeepLGF`, the importance of edges
can be weighted such that it can implicitly learn the similarity properties
(Fig. 3). Third, with the designed path-based subgraph and flow-based GNN
architecture, EmerGNN captures more relevant information from the biomedical


4


network, thus outperforming `CompGCN` and `SumGNN` (Supplementary Figure 4) as
well.

We evaluate the top-performing models in each type in the more challenging
S2 setting (Table 1), where both drugs are new with sparser information. While
`KG-DDI` and `DeepLGF` performed well in S1 setting, they struggled in S2 setting
since they need to learn representations of both drugs effectively. Conversely,
`CSMDDI` and `HIN-DDI` performed more consistently, with `CSMDDI` ranking second
on DrugBank and `HIN-DDI` ranking second on TWOSIDES. This may be due to
their simple models but effective features. In comparison, `EmerGNN` significantly
outperforms all the baselines under two-sided t-testing of statistical significance
by aggregating essential information from the biomedical network. Additionally,
we provide results for the S0 setting (Supplementary Table 3), which predicts
interactions between existing drugs. We thoroughly investigate why `EmerGNN`
has superior performance for DDI prediction in the following results.


**Analysis of the Computational complexity.** Since `EmerGNN` learns pairwise representations for each drug pair, the computation complexity is higher
than the other GNN-based methods. However, `EmerGNN` can achieve higher
accuracy than other GNN-based methods in just a few hours, and longer
training time has the potential to achieve even better performance (Fig. 2ab). Among the baseline GNN methods, `Decagon` is the most efficient as it only
uses information related to drug, protein and disease in the biomedical network.
`SumGNN` and `EmerGNN` are slower than `Decagon` and `DeepLGF` as they need to
learn specific subgraph representations for different drug pairs. Given that the
clinical development of a typical innovative drug usually takes years [25], the
computation time of `EmerGNN` is acceptable. We also compare the GPU memory
footprint (Fig. 2c) and the number of parameters (Fig. 2d) of these GNN-based
models. It is clear that `EmerGNN` is memory and parameter efficient. First, its
subgraphs for DDI prediction are much smaller than the biomedical network
(Supplementary Figure 1). Second, `EmerGNN` mainly relies on the biomedical
concepts instead of the drugs’ embeddings to do predictions, resulting in a
small number of parameters. In comparison, `DeepLGF` requires a large number
of model parameters to learn embeddings from the biomedical network.


**Analysis of drug interaction types in the learned subgraph.** `EmerGNN`
uses attention weights to measure the importance of edges in the subgraph for
predicting DDI of the emerging drugs. Here, we analyze what is captured by the
attention weights by checking correlations between predicted interaction types
with interactions and relations in the path-based subgraphs (Fig. 3).
We firstly analyze the correlations between the interaction type _i_ pred to be
predicted and interaction types obtained in the selected paths. The dominant
diagonal elements in the heatmap (Fig. 3a) suggests that when predicting a
target interaction _i_ pred for ( _u, v_ ), paths with larger attention weights in the
subgraph _G_ _u,v_ _[L]_ [are likely to go through another drug (for instance] _[ u]_ [1] [) that has]
interaction _i_ 1 = _i_ pred with the existing drug _v_ . We suppose that these drugs


5


like _u_ 1 may have similar properties as the emerging drug _u_ . To demonstrate
this point, we group these cases of drug pairs ( _u, u_ 1 ) as _Group 1_ and other pairs
( _u, u_ 2 ) with a random drug _u_ 2 as _Group 2_ . The distributions of drug fingerprints
similarities show that _Group 1_ has a larger quantity of highly similar drug pairs
( _>_ 0 _._ 5) than _Group 2_ (Fig. 3b), demonstrating the crucial role of similar drugs
in predicting DDIs for emerging drugs, and our method can implicitly search for
these drugs. Apart from the diagonal part, there exists strongly correlated pairs
of interactions. This happens, for example, when the emerging drug _u_ finds some
connections with another drug _u_ 3 whose intersection _i_ 3 with the existing drug
_v_ is correlated with _i_ pred . In these cases, we find strongly correlated pairs like
“increasing constipating activity” and “decreasing analgesic activity” (Fig. 3a,
Supplementary Table 5), verified by Liu et al. [26].
We then analyze the biomedical relation types in the selected paths by
visualizing correlations between the interaction to be predicted _i_ pred and
biomedical relation types in the selected paths. There are a few relation types
consistently selected when predicting different interaction types (Fig. 3c). In
particular, the most frequent relation type is the drug resembling relation
CrC, which again verifies the importance of similar drugs for emerging drug
prediction. Other frequently selected types are related to diseases (CrD), genes
(CbG), pharmacologic classes (PCiC) and side effects (CsSE). To analyze their
importance, we compare the performance of `EmerGNN` with the full biomedical
network, and networks with only top-1, top-3, or top-5 attended relations (the
middle part of Fig. 3d). As a comparison, we randomly sample 10%, 30%
and 50% edges from the biomedical performance and show the performance
(the right part of Fig. 3d). Keeping the top-1, top-3 and top-5 relations in
biomedical network can all lead to comparable performance as using a full
network. However, the performance substantially deteriorates when edges are
randomly dropped. These experiments show that EmerGNN selects important
and relevant relations in the biomedical network for DDI prediction.


**Case study on drug-pairs.** We present cases of selected paths from
subgraphs by selecting top ten paths between _u_ and _v_ based on the average
of edges’ attention weights on each path (Fig. 4a-b). In the first case,
there are interpretable paths supporting the target prediction (Supplementary
Table 6). For example, there are paths connecting the two drugs through
the binding protein Gene::1565 (CYP2D6), which is a P450 enzyme that
plays a key role in drug metabolism [27]. Another path finds a similar
drug DB00424 (Hyoscyamine) of DB00757 (Dolasetron) through the resemble
relation (CrC), and concludes that DB06204 (Tapentadol) may potentially
decrease the analgesic activity of DB00757 (Dolasetron) due to the correlation
between constipating and analgesic activities (Fig. 3a). In the second case,
we make similar observations (Supplementary Table 6). In particular, a
path finds a similar drug DB00421 (Spironolactone) of DB00598 (Labetalol),
which may decrease the vasoconstricting activity of DB00610 (Metaraminol),
providing a hint that Labetalol may also decrease the vasoconstricting activity


6


of Metaraminol. Compared with the original subgraphs _G_ _u,v_ _[L]_ [which have tens of]
thousands of edges (Supplementary Figure 1), the learned subgraphs are much
smaller and more relevant to the target prediction. More examples with detailed
interpretations on the paths can support that EmerGNN finds important paths
that indicate relevant interaction types and biomedical entities for emerging
drug prediction (Supplementary Figure 5).
Next, we visualize the drug pair representations obtained by `CompGCN`,
`SumGNN` and `EmerGNN` (Fig. 4c-e). As shown, the drug pairs with the same
interaction are more densely gathered in `EmerGNN` than `CompGCN` and `SumGNN` .
This means that the drug pair representations of `EmerGNN` can better separate
the different interaction types. As a result, `EmerGNN` is able to learn better
representations than the other GNN methods, like `CompGCN` and `SumGNN` .


**Ablation studies.** We compare the performance of top-performing models
according to the frequency of interaction types to analyze the different models’
ability (Fig. 5a). `EmerGNN` outperforms the baselines in all frequencies. For the
high frequency relations (1% _∼_ 20%), all the methods, except for `KG-DDI`, have

_∼_
good performance. For extremely low frequency relations (81% 100%), all the
methods work poorly. The performance of all methods deteriorates in general
for relations with a lower frequency. However, the relative performance gain
of `EmerGNN` tends to be larger, especially in the range of 61% _∼_ 80%. These
results indicate `EmerGNN` ’s strengths in generalization and ability to extract
essential information from the biomedical network for predicting rare drugs and
interaction types.
The main experiments (Table 1) study the scenario of emerging drugs
without any interaction to existing drugs. In practice, we may have a few
known interactions between the emerging and existing drugs, often obtained
from limited clinical trials. Hence, we analyze how different models perform
if adding a few interactions for each emerging drug (Fig. 5b). We can see
that the performance of shallow models such as `CSMDDI` and `HIN-DNN` does not
change much since the features they use are unchanged. However, methods
learning drug embeddings, such as `KG-DDI` and `DeepLGF`, enjoy more substantial
improvement when additional knowledge is provided. In comparison, `EmerGNN`
has increased performance with more interactions added and is still the best
over all the compared methods.
The value of _L_ determines the maximum number of hops of neighboring
entities that the GNN-based models can visit. We study the impact of changing
the length _L_ for these methods (Fig. 5c). The performance of `Decagon` and
`DeepLGF` gets worse when _L_ gets larger. Considering that `Decagon` and `DeepLGF`
work on the full biomedical network, too many irrelevant information will be
involved in the representation learning, leading to worse performance. `DeepLGF`
runs out-of-memory when _L ≥_ 3. For `SumGNN` and `EmerGNN`, _L_ = 1 performs
the worst as the information is hard to be passed from the emerging drug
to the existing drug. `SumGNN` can leverage the drug features for prediction,
thus outperforms `Decagon` . In comparison, `EmerGNN` benefits much from the


7


relevant information on the biomedical network when _L_ increases from 1 to 3.

However, the performance will decrease when _L >_ 3. Intuitively, the pathbased subgraph will contain too much irrelevant information when the length
gets longer, increasing the learning difficulty. Hence, a moderate number of path
length with _L_ = 3 is optimal for `EmerGNN`, considering both the effectiveness and
computation efficiency.
We conduct experiments to analyze the main techniques in designing in
`EmerGNN` (Fig. 5d). First, we evaluate the performance of using undirected
edges without introducing the inverse edges (denoted as Undirected edges
w.o. inverse). It is clear that using undirected edges has negative effect
as the directional information on the biomedical network is lost. Then, we
design a variant, that learns a subgraph representation as `SumGNN` upon _G_ _u,v_ _[L]_
(denoted as Subgraph representation), and another variant that only learns
on the uni-directional computing (Method) from direction _u_ to _v_ without
considering the direction from _v_ to _u_ (denoted as Uni-directional pair-wise
representation). Comparing subgraph representation with uni-directional pairwise representation, we observe that the flow-based GNN architecture is more
effective than the GNN used in `SumGNN` . Even though uni-directional pair-wise
representation can achieve better performance compared with all the baselines in
S1 setting (Table 1), learning bi-directional representations can help to further
improve the prediction ability by balancing the bi-directional communications
between drugs.

### **3 Discussion**


Predicting drug-drug interactions (DDI) for emerging drugs is a crucial issue
in biomedical computational science as it offers possibilities for treating and
alleviating diseases. Despite recent advances in DDI prediction accuracy
through the use of deep neural networks [5, 13, 14, 17, 19, 20], these methods
require large amount of known DDI information, which is often scarce for
emerging drugs. Additionally, some approaches designed for DDI prediction
only leverage shallow features, limiting their expressiveness in this task.
One limitation of EmerGNN is that the emerging drug to be predicted should
be included in the biomedical network. Building connections between emerging
drug and existing drug through molecular formula or property may help address
this issue. Although we demonstrate the effectiveness of EmerGNN for DDI
prediction in this paper, EmerGNN is a general approach that can be applied
to other biomedical applications, such as predicting protein-protein interaction,
drug-target interaction and disease-gene interaction. We anticipate that the
paths attended by EmerGNN can enhance the accuracy and interpretability of
these predictions. We hope that our open-sourced EmerGNN can serve as a
strong deep learning tool to advance biomedicine and healthcare, by enabling
practitioners to exploit the rich knowledge in existing large biomedical networks
for low-data scenarios.


8


### **4 Methods**

To predict interactions between emerging drugs and existing drugs, it is
important to leverage relevant information in the biomedical network. Our
framework contains four main components: (i) constructing an augmented
network by integrating the DDI network with the biomedical network and
adding inverse edges; (ii) extracting all the paths with length no longer than _L_
from _u_ to _v_ to construct a path-based subgraph _G_ _u,v_ _[L]_ [; (iii) encoding pair-wise]

subgraph representation _**h**_ [(] _u,v_ _[L]_ [)] [by a flow-based GNN with attention mechanism]
such that the information can flow from _u_ over the important entities and edges
in _G_ _u,v_ _[L]_ [to] _[ v]_ [; (iv) predicting the interaction type based on the bi-directional]
pair-wise subgraph representations. The overall framework is shown in Fig. 1.


**Augmented network** Given the DDI network _N_ D = _{_ ( _u, i, v_ ) : _u, v ∈V_ D _, i ∈_
_R_ I _}_ and the biomedical network _N_ B = _{_ ( _h, r, t_ ) : _h, t ∈V_ B _, r ∈R_ B _}_ ( _N_ D is
specified as _N_ D-train / _N_ D-valid / _N_ D-test in the training/validation/testing stages,
respectively, so does _N_ B ), we integrate the two networks into


_N_ _[′]_ = _N_ D _∪N_ B = �( _e, r, e_ _[′]_ ) : _e, e_ _[′]_ = _V_ _[′]_ _, r ∈R_ _[′]_ [�] _,_


with _V_ _[′]_ = _V_ D _∪V_ B and _R_ _[′]_ = _R_ I _∪R_ B . The integrated network _N_ _[′]_ connects the
existing and emerging drugs by concepts in the biomedical network. Since the
relation types are directed, we follow the common practices in knowledge graph
learning [6, 28] to add inverse types. Specifically, we add _r_ inv for each _r ∈R_ _[′]_

and create a set of inverse types _R_ _[′]_ inv [, which subsequently leads to an inverse]
network
_N_ inv _[′]_ [=] �( _e_ _[′]_ _, r_ inv _, e_ ) : ( _e, r, e_ _[′]_ ) _∈N_ _[′]_ [�] _._


Note that the inverse relations will not influence the information in the original
biomedical network since we can transform any inverse edge ( _e_ _[′]_ _, r_ ~~i~~ nv _, e_ ) back
to the original edge ( _e, r, e_ _[′]_ ). Semantically, the inverse relations can be regarded
as a kind of active voice vs. passive voice in linguistics, for instance _includes_ ~~_i_~~ _nv_
can be regarded as _being included_ and _causes_ ~~_i_~~ _nv_ can be regarded as _being_
_caused_ . By adding the inverse edges, the paths can be smoothly organized in
_r_ 1 _r_ 2
single directions. For example, a path _a_ _−→_ _b_ _←−_ _c_ can be transformed to
_r_ 1 _r_ 2 inv
_a_ _−→_ _b_ _−_ ~~_−−_~~ _−→_ _c_, which is more computational friendly.
After the above two steps, we obtain the augmented network


_N_ = _N_ _[′]_ _∪N_ inv _[′]_ [=] �( _e, r, e_ _[′]_ ) : _e, e_ _[′]_ _∈V, r ∈R_ � _,_


with entity set _V_ = _V_ _[′]_ = _V_ D _∪V_ B and relation set _R_ = _R_ _[′]_ _∪R_ _[′]_ inv [.]


**Path-based subgraph formulation** Inspired by the path-based methods
in knowledge graph learning [29, 30], we are motivated to extract the paths
connecting existing and emerging drugs, and predict the interaction type based
on the paths.


9


Given a drug pair ( _u, v_ ) to be predicted, we extract the set _P_ _u,v_ _[L]_ [of all the]
paths with length up to _L_ . Each path in _P_ _u,v_ _[L]_ [has the form]


_r_ 1 _r_ 2 _r_ _L_
_e_ 0 _−→_ _e_ 1 _−→· · ·_ _−→_ _e_ _L_ _,_


with _e_ 0 = _u_, _e_ _L_ = _v_ and ( _e_ _i−_ 1, _r_ _i_, _e_ _i_ ) _∈N_ _, i_ = 1 _, . . ., L_ . The intermediate
entities _e_ 1 _, . . ., e_ _L−_ 1 _∈V_ can be drugs, genes, diseases, side-effects, symptoms,
pharmacologic class, etc., and _r_ 1 _, . . ., r_ _L_ _∈R_ are the interactions or relations
between the biomedical entities. In order preserve the local structures, we merge
the paths in _P_ _u,v_ _[L]_ [to a subgraph] _[ G]_ _u,v_ _[L]_ [such that the same entities are merged to]
a single node. The detailed steps of path extraction and subgraph generation
are provided in Supplementary Section 1.
Different from the subgraph structures used for link prediction on general
graphs [6, 31, 32], the edges in _G_ _u,v_ _[L]_ [are pointed away from] _[ u]_ [ and towards] _[ v]_ [. Our]
objective is to learn a GNN _g_ ( _·_ ) with parameters _**θ**_ that predicts DDI between
_u_ and _v_ based on the path-based subgraph _G_ _u,v_ _[L]_ [, that is]


DDI( _u, v_ ) = _g_ � _G_ _u,v_ _[L]_ [;] _**[ θ]**_ � _._ (1)


The link prediction problem on the DDI network is then transformed as a whole
graph learning problem.


**Flow-based GNN architecture** Given _G_ _u,v_ _[L]_ [, we would like to integrate]
essential information in it to predict the target interaction type. Note that
the edges in _G_ _u,v_ _[L]_ [are from the paths] _[ P]_ _u,v_ _[L]_ [connecting from] _[ u]_ [ to] _[ v]_ [. We aim to]
design a special GNN architecture that the information can flow from drug _u_
to _v_, via integrating entities and relations in _G_ _u,v_ _[L]_ [.]
Denote _V_ _[ℓ]_
_u,v_ _[, ℓ]_ [= 0] _[, . . ., L]_ [, as the set of entities that can be visited in the] _[ ℓ]_ [-th]
flow step from _u_ (like the four ellipses in _g_ ( _G_ _u,v_ _[L]_ [;] _**[ θ]**_ [) in Fig. 1). In particular, we]
have _V_ _u,v_ [0] [=] _[ {][u][}]_ [ as the starting point and] _[ V]_ _u,v_ _[L]_ [=] _[ {][v][}]_ [ as the ending point. In the]
_ℓ_ -th iteration, the visible entities in _V_ _u,v_ _[ℓ]_ [contains entities that are] _[ ℓ]_ [-steps away]
from drug _u_ and are ( _L −_ _ℓ_ )-steps away from drug _v_ in the augmented network
_N_ . We use the fingerprint features [9] of drug _u_ as the input representation
of _u_, namely _**h**_ [(0)] _u,u_ [=] _**[ f]**_ _u_ [. Then, we conduct message flow for] _[ L]_ [ steps with the]
function



_**h**_ [(] _u,e_ _[ℓ]_ [)] [=] _[δ]_ � _**W**_ [(] _[ℓ]_ [)] [�] _e_ _[′]_ _∈V_ _u,v_ _[ℓ][−]_ [1]



� _**h**_ [(] _u,e_ _[ℓ][−]_ [1)] + _ϕ_ ( _**h**_ [(] _u,e_ _[ℓ][−]_ _[′]_ [1)] _[,]_ _**[ h]**_ _r_ [(] _[ℓ]_ [)] [)] � [�] _,_ (2)



for entities _e ∈V_ _[ℓ]_
_u,v_ [, where] _**[ W]**_ [ (] _[ℓ]_ [)] _[ ∈]_ [R] _[d][×][d]_ [ is a learnable weighting matrix for]

step _ℓ_ ; _**h**_ [(] _u,e_ _[ℓ][−]_ _[′]_ [1)] is the pair-wise representation of entity _e_ _[′]_ _∈V_ _u,v_ _[ℓ][−]_ [1] [;] _[ r]_ [ is the relation]

type between _e_ _[′]_ and _e_ ; _**h**_ [(] _r_ _[ℓ]_ [)] _∈_ R _[d]_ is the learnable representation with dimension
_d_ of _r_ in the _ℓ_ -th step; and _ϕ_ ( _·, ·_ ) : (R _[d]_ _,_ R _[d]_ ) _→_ R _[d]_ is the function combining the
two vectors; and _δ_ ( _·_ ) is the activation function ReLU [33].
Since the biomedical network is not specially designed for the DDI prediction
task, we need to control the importance of different edges in _G_ _u,v_ _[L]_ [. We use a]


10


drug-dependent attention weight for function _ϕ_ ( _·, ·_ ). Specifically, we design the
message function for each edge ( _e_ _[′]_ _, r, e_ ) during the _l_ -th propagation step as


_ϕ_ ( _**h**_ [(] _u,e_ _[ℓ][−]_ _[′]_ [1)] _[,]_ _**[ h]**_ _r_ [(] _[ℓ]_ [)] [) =] _[ α]_ _r_ [(] _[ℓ]_ [)] _·_ � _**h**_ [(] _u,e_ _[ℓ][−]_ _[′]_ [1)] _⊙_ _**h**_ [(] _r_ _[ℓ]_ [)] � _,_ (3)


where _⊙_ is an element-wise dot product of vectors and _α_ _r_ [(] _[ℓ]_ [)] is the attention
weight controlling the importance of messages. We design the attention weight
depending on the edges’ relation type as


_α_ _r_ [(] _[ℓ]_ [)] = _σ_ ( _**w**_ _r_ [(] _[ℓ]_ [)] [)] _[⊤]_ [[] _**[f]**_ _[u]_ [;] _**[ f]**_ _[v]_ []] _,_
� �


where the relation weight _**w**_ _r_ [(] _[ℓ]_ [)] _∈_ R [2] _[d]_ is multiplied with the fingerprints

[ _**f**_ _u_ ; _**f**_ _v_ ] _∈_ R [2] _[d]_ of drugs to be predicted and _σ_ ( _·_ ) is a sigmoid function returning
a value in (0 _,_ 1).
After iterating for _L_ steps, we can obtain the representation _**h**_ [(] _u,v_ _[L]_ [)] [that]
encodes the important paths up to length _L_ between drugs _u_ and _v_ .


**Objective and training** In practice, the interaction types can be symmetric,
for example #Drug1 and #Drug2 may have the side effect of headache if
used together, or asymmetric, for example #Drug1 may decrease the analgesic
activities of #Drug2. Besides, the emerging drug can appear in either the source
(drug _u_ ) or target (drug _v_ ). We extract the reverse subgraph _G_ _v,u_ _[L]_ [and encode]
it with the same parameters in Equation (2) to obtain the reverse pair-wise
representation _**h**_ [(] _v,u_ _[L]_ [)] [. Then the bi-directional representations are concatenated]
to predict the interaction type with


_**l**_ ( _u, v_ ) = _**W**_ rel � _**h**_ [(] _u,v_ _[L]_ [)] [;] _**[ h]**_ _v,u_ [(] _[L]_ [)] � _._ (4)


Here, the transformation matrix _**W**_ rel _∈_ R _[|R]_ [I] _[|×]_ [2] _[d]_ is used to map the pair-wise
representations into prediction logits _**l**_ ( _u, v_ ) of the _|R_ I _|_ interaction types. The
_i_ -th logit _l_ _i_ ( _u, v_ ) indicates the plausibility of interaction type _i_ being predicted.
The full algorithm and implementation details of Equation (4) are provided in
Supplementary Section 1.
Since we have two kinds of tasks that are multi-class (on the DrugBank
dataset) and multi-label (on the TWOSIDES dataset) interaction predictions,
the training objectives are different.
For DrugBank, there exists at most one interaction type between two drugs.
Given two drugs _u_ and _v_, once we obtain the prediction logits _**l**_ ( _u, v_ ) of different
interaction types, we use a softmax function to compute the probability of each
interaction type, namely



exp � _l_ _i_ ( _u, v_ )�
_I_ _i_ ( _u, v_ ) =



_j∈R_ I [exp] ~~�~~ _l_ _j_ ( _u, v_ ) ~~�~~ _._



~~�~~



11


Denote _**y**_ ( _u, v_ ) _∈_ R _[|R]_ [I] _[|]_ as the ground-truth indicator of target interaction type,
where _y_ _i_ ( _u, v_ ) = 1 if ( _u, i, v_ ) _∈N_ D, otherwise zero. We minimize the following
cross-entropy loss to train the model parameters


_L_ DB = _−_ � _y_ _i_ ( _u, v_ ) log _I_ _i_ ( _u, v_ ) _._ (5)

( _u,i,v_ ) _∈N_ D-train


For TWOSIDES, there may be multiple interactions between two drugs.
The objective is to predict whether there is an interaction _p_ between two drugs.
Given two drugs _u_, _v_ and the prediction logits _**l**_ ( _u, v_ ), we use the sigmoid
function


1
_I_ _i_ ( _u, v_ ) = 1 + exp( _−l_ _i_ ( _u, v_ )) _[,]_


to compute the probability of interaction type _i_ . Different with the multi-class
task in DrugBank, we use the binary cross entropy loss



_L_ TS = _−_ �



�log � _I_ _i_ ( _u, v_ )� + �



( _u,i,v_ ) _∈N_ D-train



+ � log �1 _−_ _I_ _i_ ( _u_ _[′]_ _, v_ _[′]_ )� [�] _,_ (6)

( _u_ _[′]_ _,v_ _[′]_ ) _∈N_ _i_



where _N_ _i_ is the set of drug pairs that do not have the interaction type _i_ .
We use stochastic gradient optimizer Adam [34] to optimize the model
parameters


_**θ**_ = � _**W**_ rel _,_ � _**W**_ [(] _[ℓ]_ [)] _,_ _**h**_ [(] _r_ _[ℓ]_ [)] _[,]_ _**[ w]**_ _r_ [(] _[ℓ]_ [)] _[}]_ _[ℓ]_ [=1] _[,...,L,r][∈R]_ � _,_


by minimizing loss function in Equation (5) for the DrugBank dataset or
Equation (6) for the TWOSIDES dataset.


**Drug-drug interaction network.** Following [6, 13], we use two benchmark
datasets, DrugBank [22] and TWOSIDES [23], as the interaction network _N_ D
(Supplementary Table 1). When predicting DDIs for emerging drugs, namely
the S1 and S2 settings, we randomly split _V_ D into three disjoint sets with _V_ D =
_V_ D-train _∪V_ D-valid _∪V_ D-test and _V_ D-train _∩V_ D-valid _∩V_ D-test = _∅_, where _V_ D-train is the
set of existing drugs used for training, _V_ D-valid is the set of emerging drugs for
validation, and _V_ D-test is the set of emerging drugs for testing. The interaction
network for training is defined as _N_ D-train = _{_ ( _u, i, v_ ) _∈N_ D : _u, v_ _∈V_ D-train _}_ .
In the S1 setting, we set


  - _N_ D-valid = _{_ ( _u, i, v_ ) _∈N_ D : _u ∈V_ D-train _, v ∈V_ D-valid _} ∪{_ ( _u, i, v_ ) _∈N_ D : _u ∈_
_V_ D-valid _, v_ _∈V_ D-train _}_ as validation samples;


  - _N_ D-test = _{_ ( _u, i, v_ ) _∈N_ D : _u_ _∈_ ( _V_ D-train _∪V_ D-valid ) _, v ∈V_ D-test _} ∪{_ ( _u, i, v_ ) _∈_
_N_ D : _u_ _∈D_ D-test _, v_ _∈_ ( _V_ D-train _∪V_ D-valid ) _}_ as testing samples.


In the S2 setting, we set


  - _N_ D-valid = _{_ ( _u, i, v_ ) _∈N_ D : _u, v_ _∈V_ D-valid _}_ as validation samples; and


12


  - _N_ D-test = _{_ ( _u, i, v_ ) _∈N_ D : _u, v_ _∈V_ D-test _}_ as testing samples.


We follow [6] to randomly sample one negative sample for each ( _u, i, v_ ) _∈_
_N_ D-valid _∪N_ D-test to form the negative set _N_ _i_ for TWOSIDES dataset in the
evaluation phase. Specifically, if _u_ is an emerging drug, we randomly sample
an existing drug _v_ _[′]_ _∈V_ D-train and make sure that the new interaction does not
exist, namely ( _u, i, v_ _[′]_ ) _/∈N_ D ; if _v_ is an emerging drug, we randomly sample an
existing drug _u_ _[′]_ _∈V_ D-train and make sure that the new interaction does not
exist, namely ( _u_ _[′]_ _, i, v_ ) _/∈N_ D .


**Biomedical network.** In this work, same as the DDI network, we use
different variants of the biomedical network _N_ B for training, validation and
testing. The well-built biomedical network HetioNet [18] is used here. Denote
_V_ B _, R_ B _, N_ B as the set of entities, relations and edges, respectively, in the full
biomedical network. When predicting interactions between existing drugs in
the S0 setting, all the edges in _N_ B are used for training, validation and testing.
When predicting interactions between emerging drugs and existing drugs (S1
and S2 setting), we use different parts of the biomedical networks.
In order to guarantee that the emerging drugs are connected with some
existing drugs through the biomedical entities, we constrain the split of drugs
to satisfy the conditions _V_ D-valid _⊂V_ B and _V_ D-test _⊂V_ B . Meanwhile, we also
guarantee that the emerging drugs will not be seen in the biomedical network
during training. To achieve this goal, the edges for training are in the set
_N_ B-train = _{_ ( _h, r, t_ ) _∈N_ B : _h, t /∈_ ( _V_ D-valid _∪V_ D-test ) _}_ ; the edges for validation
are in the set _N_ B-valid = _{_ ( _h, r, t_ ) _∈N_ B : _h, t /∈V_ D-test _}_ ; and the testing network
is the original network, namely _N_ B-test = _N_ B .

In addition, we plot the size distribution (measured by the number of edges
in _G_ _u,v_ _[L]_ [) as histograms (Supplementary Figure 1). We observe that both datasets]
follow long-tail distributions. Many subgraphs have tens of thousands of edges
on DrugBank, while hundreds of thousands of edges on TWOSIDES since
the DDI network is denser. Comparing with the augmented networks, whose
sizes are 3,657,114 for DrugBank and 3,567,059 for TWOSIDES, the sizes of
subgraphs are quite small.


**Evaluation metrics.** As pointed by [6], there is at most one interaction
between a pair of drugs in the DrugBank dataset [22]. Hence, we evaluate
the performance in a multi-class setting, which estimates whether the model
can correctly predict the interaction type for a pair of drugs. We consider the
following metrics:


  - 1 2 _P_ _i_ _·R_ _i_
F1(macro) = _∥I_ _D_ _∥_ � _i∈I_ _D_ _P_ _i_ + _R_ _i_ [, where] _[ P]_ _[i]_ [ and] _[ R]_ _[i]_ [ are the precision and]

recall for the interaction type _i_, respectively. The macro F1 aggregates
the fractions over different interaction types.


  - Accuracy: the percentage of correctly predicted interaction type compared
with the ground-truth interaction type.


13


   - Cohen’s Kappa [24]:(accuracy) and _A_ _e_ is the probability of randomly seeing each class. _κ_ = _A_ 1 _p_ _−−AA_ _ee_ [, where] _[ A]_ _[p]_ [ is the observed agreement]


In the TWOSIDES dataset [23], there may be multiple interactions between
a pair of drugs, such as anaemia, nausea and pain. Hence, we model and
evaluate the performance in a multi-label setting, where each type of side effect
is modeled as a binary classification problem. Following [13, 23], we sample
one negative drug pair for each ( _u, i, v_ ) _∈N_ D-test and evaluate the binary
classification performance with the following metrics:


  - ROC-AUC: the area under curve of receiver operating characteristics,
measured by [�] _[n]_ _k_ =1 [TP] _[k]_ [∆FP] _[k]_ [, where (TP] _[k]_ _[,]_ [ FP] _[k]_ [) is the true-positive and]
false-positive of the _k_ -th operating point.


  - PR-AUC: the area under curve of precision-recall, measured by [�] _[n]_ _k_ =1 _[P]_ _[k]_ [∆] _[R]_ _[k]_ [,]
where ( _P_ _k_ _, R_ _k_ ) is the precision and recall of the _k_ -th operating point.


  - Accuracy: the average precision of drug pairs for each side effect.

### **Data availability**


Source data for Figures 2-5 is available with this manuscript. The resplit dataset

[35] of DrugBank, TWOSIDES and HetioNet for S1 and S2 settings is public
available at `[https://doi.org/10.5281/zenodo.10016715](https://doi.org/10.5281/zenodo.10016715)` .

### **Code availability**


The code for EmerGNN [36] is available at `[https://github.com/LARS-research/](https://github.com/LARS-research/EmerGNN)`

`[EmerGNN](https://github.com/LARS-research/EmerGNN)` .

### **Acknowledgments**


This project was supported by the National Natural Science Foundation of
China (No. 92270106) and CCF-Tencent Open Research Fund.

### **Author Contributions Statement**


Y. Zhang contributes to idea development, algorithm implementation, experimental design, result analysis, and paper writing. Q. Yao contributes to
idea development, experimental design, result analysis, and paper writing. L.
Yue contributes to algorithm implementation and result analysis. Y. Zheng
contributes to result analysis and paper writing. All authors read, edited, and
approved the paper.


14


### **Competing Interests Statement**

The authors declare no competing interests.


15


### **Tables**

Table 1: Performance of EmerGNN compared other DDI prediction methods.
Four types of DDI prediction methods are compared: (i) methods that use
drug features of target drug pairs ( _DF_ ) [4, 9, 11, 21]; (ii) methods that use
graph features in the biomedical network ( _GF_ ) [5]; (iii) methods that learn
drug embeddings ( _Emb_ ) [12, 14]; and (iv) methods that model with GNNs
( _GNN_ ) [6, 13, 16, 17, 28].

**S1 Setting** : DDI prediction between emerging drug and existing drug [1] .

|Datasets|DrugBank|TWOSIDES|
|---|---|---|
|Type Methods|F1-Score Accuracy<br>Kappa|PR-AUC ROC-AUC Accuracy|
|DF<br>MLP [9]<br>Similarity [4]<br>CSMDDI [11]<br>STNN-DDI [21]|21.1_±_0.8<br>46.6_±_2.1<br>33.4_±_2.5<br>43.0_±_5.0<br>51.3_±_3.5<br>44.8_±_3.8<br>45.5_±_1.8<br>62.6_±_2.8<br>55.0_±_3.2<br>39.7_±_1.8<br>56.7_±_2.6<br>46.5_±_3.4|81.5_±_1.5<br>81.2_±_1.9<br>76.0_±_2.1<br>56.2_±_0.5<br>55.7_±_0.6<br>53.9_±_0.4<br>73.2_±_2.6<br>74.2_±_2.9<br>69.9_±_2.2<br>68.9_±_2.0<br>68.3_±_2.6<br>65.3_±_1.8|
|GF<br>HIN-DDI* [5]|37.3_±_2.9<br>58.9_±_1.4<br>47.6_±_1.8|81.9_±_0.6<br>83.8_±_0.9<br>79.3_±_1.1|
|Emb<br>MSTE [12]<br>KG-DDI* [14]|7.0_±_0.7<br>51.4_±_1.8<br>37.4_±_2.2<br>26.1_±_0.9<br>46.7_±_1.9<br>35.2_±_2.5|64.1_±_1.1<br>62.3_±_1.1<br>58.7_±_0.7<br>79.1_±_0.9<br>77.7_±_1.0<br>60.2_±_2.2|
|GNN CompGCN* [28]<br>Decagon* [13]<br>KGNN* [16]<br>SumGNN* [6]<br>DeepLGF* [17]<br>**EmerGNN***|26.8_±_2.2<br>48.7_±_3.0<br>37.6_±_2.8<br>24.3_±_4.5<br>47.4_±_4.9<br>35.8_±_5.9<br>23.1_±_3.4<br>51.4_±_1.9<br>40.3_±_2.7<br>35.0_±_4.3<br>48.8_±_8.2<br>41.1_±_4.7<br>39.7_±_2.3<br>60.7_±_2.4<br>51.0_±_2.6<br>**62.0**_±_2.0** 68.6**_±_3.7** 62.4**_±_4.3|80.3_±_3.2<br>79.4_±_4.0<br>71.4_±_3.1<br>79.0_±_2.0<br>78.5_±_2.3<br>69.7_±_2.4<br>78.5_±_0.5<br>79.8_±_0.6<br>72.3_±_0.7<br>80.3_±_1.1<br>81.4_±_1.0<br>73.0_±_1.4<br>81.4_±_2.1<br>82.2_±_2.6<br>72.8_±_2.8<br>** 90.6**_±_0.7<br>**91.5**_±_1.0<br>**84.6**_±_0.7|
|**p-value**|8.9E-7<br>0.02<br>0.02|1.6E-6<br>6.0E-8<br>3.5E-5|



**S2 Setting** : DDI prediction between two emerging drugs.

|Datasets|DrugBank|TWOSIDES|
|---|---|---|
|Type Methods|F1-Score Accuracy<br>Kappa|PR-AUC ROC-AUC Accuracy|
|DF<br>CSMDDI [11]|19.8_±_3.1<br>37.3_±_4.8<br>22.0_±_4.9|55.8_±_4.9<br>57.0_±_6.1<br>55.1_±_5.2|
|GF<br>HIN-DDI* [5]|8.8_±_1.0<br>27.6_±_2.4<br>13.8_±_2.4|64.8_±_2.3<br>58.5_±_1.6<br>59.8_±_1.4|
|Emb<br>KG-DDI* [14]|1.1_±_0_._1<br>32.2_±_3.6<br>0.0_±_0.0|53.9_±_3.9<br>47.0_±_5.5<br>50.0_±_0.0|
|GNN DeepLGF* [17]<br>**EmerGNN***|4.8_±_1.9<br>31.9_±_3.7<br>8.2_±_2.3<br>**25.0**_±_2.8** 46.3**_±_3.6** 31.9**_±_3.8|59.4_±_8.7<br>54.7_±_5.9<br>54.0_±_6.2<br>**81.4**_±_7.4<br>**79.6**_±_7.9<br>**73.0**_±_8.2|
|**p-value**|0.02<br>0.01<br>0.01|1.4E-3<br>3.9E-4<br>7.8E-3|



1 All of the methods are run for five times on the five-fold datasets with mean value and
standard deviation reported on the testing data. The evaluation metrics are presented in
percentage (%) with the larger value indicating better performance. The boldface numbers
indicate the best values, while the underlined numbers indicate the second best. p-values are
computed under two-sided t-testing of EmerGNN over the second best baselines. Methods
leveraging a biomedical network are indicated by star *.


16


### **Figures**



**a** **Emerging** drugs **Known** DDI 𝒩 ' **b**



**Emerging** drugs



**Known** DDI 𝒩 '



**Augmented network** 𝒩











































prediction



**f**





Figure 1: **Overview of EmerGNN. (a)** Problem formulation: Given a DDI
network _N_ D of existing drugs and a large biomedical network _N_ B providing side
information for the drugs, the task is to predict the interaction type between
an emerging drug (like _u_ in dark blue) and an existing drug (like _v_ in purple)
in the S1 setting, or interaction type between two emerging drugs (like _u_ in
dark blue and _w_ in light blue) in the S2 setting. **(b)** Augmented network _N_ :
The DDI network and biomedical network are integrated and edges with inverse
types are incorporated to obtain an augmented network _N_ . The augmentation
brings better communication among drugs and entities in both interaction and
biomedical networks. **(c)** Path-based subgraph: Given a drug pair ( _u, v_ ) to be
predicted, all the paths from _u_ to _v_ with length no larger than _L_ are extracted
to construct a path-based subgraph _G_ _u,v_ _[L]_ [.] **(d)** Flow-based GNN _g_ ( _·_ ; _**θ**_ ) with
parameters _**θ**_ : the network flows the initial drug features _**h**_ [0] _u,u_ [=] _**[ f]**_ _[u]_ [over]
essential information in _G_ _u,v_ _[L]_ [for] _[ L]_ [ steps.] It uses different attention weights
_α_ ’s to weight the importance of different edges. After _L_ steps, a pair-wise
representation _**h**_ _[L]_ _u,v_ [between] _[ u]_ [ and] _[ v]_ [ is obtained as the subgraph representation]
of _G_ _u,v_ _[L]_ [.] **[ (e)]** [ Interaction predictor] _[ p]_ [(] _[·]_ [): a simple linear classifier] _[ p]_ [(] _**[h]**_ _[L]_ _u,v_ [) outputs]
a distribution _**I**_ ( _u, v_ ), where each dimension indicates an interaction type _i ∈R_ I
between _u_ and _v_ . **(f)** Legends: The different relation and interaction types are
indicated by arrows with different colors. Edges with inverse types are indicated
by dashed arrows with corresponding color. The icons represent biomedical
concepts including drugs, genes and diseases.


17


**a**





**b**









|60<br>50<br>40<br>30<br>20<br>10<br>0|Col2|Col3|90<br>85<br>(%)<br>80 PR-AUC<br>75<br>Testing<br>70<br>Decagon<br>SumGNN 65<br>DeepLGF<br>EmerGNN 60|Col5|Col6|Col7|Decagon<br>SumGNN<br>DeepLGF<br>EmerGNN|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|0<br>10<br>20<br>30<br>40<br>50<br>60<br><br><br>||||||||||
|0<br>10<br>20<br>30<br>40<br>50<br>60<br><br><br>||||||||||
|0<br>10<br>20<br>30<br>40<br>50<br>60<br><br><br>||||||||||
|0<br>10<br>20<br>30<br>40<br>50<br>60<br><br><br>||||||||||
|0<br>10<br>20<br>30<br>40<br>50<br>60<br><br><br>|||Deca||||||Decag|
|0<br>10<br>20<br>30<br>40<br>50<br>60<br><br><br>|||~~SumG~~<br>Deep<br>Emer||||||~~SumG~~<br>DeepL<br>EmerG|


**c** **d**













Figure 2: Complexity analysis of different GNN-based methods in the S1 setting.
**(a)** Comparison of training curves on DrugBank dataset. **(b)** Comparison of
training curves on TWOSIDES dataset. **(c)** Comparison of GPU memory
footprint usage on the two datasets in MB. **(d)** Comparison of number of
trainable model parameters on the two datasets.


18


**a**





~~(#5,#85)~~


~~(#49,#18)~~


~~(#52,#39)~~





**b**







**c** **d**



65%


55%


45%


35%























(Supplementary Figure 3). **(a)** Heatmap of correlation between interaction
_i_ pred to be predicted and interaction types _i_ in the selected paths. Yellow
circles indicate the three highlighted interaction pairs outside the diagonal
(Supplementary Table 5). **(b)** The histogram distribution of fingerprint
similarities in _Group 1_ (a drug _u_ with another drug _u_ 1, which connected to
_v_ with interaction type _i_ pred ) and _Group 2_ (a drug _u_ with a random drug
_u_ 2 ). **(c)** Heatmap of correlation between interaction _i_ pred to be predicted and
biomedical relations _r_ in the selected paths. **(d)** Performance of modified
biomedical networks with selected relations. Leftmost is the performance of
EmerGNN with full biomedical network. The middle three parts are EmerGNN
with top-1 (CrC, with 0.4% edges), top-3 (CrC, CbG, CsSE, with 9.3% edges),
top-5 (CrC, CtD, CvG, PCiC, CsSE, with 9.4% edges) attended relations in
the biomedical network. The right three parts are EmerGNN with randomly
sampled 10%, 30%, 50% edges from the biomedical network.


19


**a**



**b**







































**c** **d** **e**





















Figure 4: Visualization of drug pairs. **(a-b)** Two cases of subgraphs containing
top ten paths according to the average of edges’ attention weights on each
path (explanations in Supplementary Table 6). The drug pairs to be predicted
are highlighted as stars; dashed lines mean reverse types; CrC, CbG, CtD are
biomedical relations; #39, #5, #85 are interaction types; “other types” in
gray edges mean the interaction types aside from the given ones. In the first
case, DB06204 (Tapentadol) in blue star is an existing drug, and DB00757
(Dolasetron) in red star is an emerging drug. The target interaction type is
“#Drug1 may decrease the analgesic activities of #Drug2” (#52). In the second
case, DB00598 (Labetalol) in blue star is an emerging drug, and DB00610
(Metaraminol) in red star is an existing drug. The target interaction type
is “#Drug1 may decrease the vasoconstricting activities of #Drug2” (#5).
**(c-e)** t-SNE visualization [37] of the representations learned for drug pairs.
As CompGCN embeds each entity separately, we concatenate embeddings
of the two drugs’ representations for a given drug pair. SumGNN encodes
the enclosing subgraphs of ( _u, v_ ) for interaction prediction, thus we take the
representation of enclosing subgraph as the drug pair representation. The drug
pair representation of EmerGNN is directly given by _**h**_ [(] _u,v_ _[L]_ [)] [.] Since there are
too many interaction types and drug pairs in _N_ D-test, eight interaction types
and sixty-four drug pairs are randomly sampled for each interaction type. The
legends in these figures specify the IDs of the interaction type to be predicted;
each dot denotes a DDI sample ( _u, i, v_ ); the different colors in dots indicate the
interaction type _i_ that the drug pairs ( _u, v_ ) have.


20


**a**


**c**





**b**


**d**










|Col1|Col2|Col3|
|---|---|---|
||||
||||
|||C<br>H<br>K<br>D<br>E|








|60%|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|0<br>20%<br>40%<br>|||||
|0<br>20%<br>40%<br>|||||
|0<br>20%<br>40%<br>|2<br>3<br>|2<br>3<br>|2<br>3<br>|4<br>Deca<br>SumG<br>Deep<br>Emer|



Figure 5: Ablation studies on the DrugBank dataset. **(a)** Performance
comparison of interaction groups based on interaction frequency. The five groups
are formed by grouping the interaction types based on their frequency in the
dataset, and the average macro F1 performance is shown in each group. **(b)**
Performance comparison of adding interaction edges for emerging drugs into
the training set _N_ D-train . Specifically, 1/3/5 interaction edges in the testing set
_N_ D-test are randomly sampled for each emerging drug in _V_ D-test, and moved to
the training set _N_ D-train . **(c)** Performance comparison of GNN-based methods
by varying the depth _L_ . Specifically, _L_ means the number of GNN layers in
`Decagon` and `DeepLGF`, the depth of enclosing subgraph in `SumGNN`, and the depth
of path-based subgraph in `EmerGNN` . **(d)** Performance comparison of different
technique designing in EmerGNN (Supplementary Table 7).


21


### **References**


[1] Xian Su, Haixue Wang, Nan Zhao, Tao Wang, and Yimin Cui. Trends in
innovative drug development in China. Nature Reviews. Drug Discovery,
2022.


[2] Heidi Ledford. Hundreds of COVID trials could provide a deluge of new
drugs. Nature, pages 25–27, 2022.


[3] Bethany Percha and Russ B Altman. Informatics confronts drug-drug
interactions. Trends in Pharmacological Sciences, 34(3):178–184, 2013.


[4] Santiago Vilar, Eugenio Uriarte, Lourdes Santana, Tal Lorberbaum, George
Hripcsak, Carol Friedman, and Nicholas P Tatonetti. Similarity-based
modeling in large-scale prediction of drug-drug interactions. Nature
Protocols, 9(9):2147–2163, 2014.


[5] Farhan Tanvir, Muhammad Ifte Khairul Islam, and Esra Akbas.
Predicting drug-drug interactions using meta-path based similarities. In
IEEE Conference on Computational Intelligence in Bioinformatics and
Computational Biology, pages 1–8. IEEE, 2021.


[6] Yue Yu, Kexin Huang, Chao Zhang, Lucas M Glass, Jimeng Sun, and
Cao Xiao. SumGNN: multi-typed drug interaction prediction via efficient
knowledge graph summarization. Bioinformatics, 37(18):2988–2995, 2021.


[7] Louis Letinier, Sebastien Cossin, Yohann Mansiaux, Mickael Arnaud,
Francesco Salvo, Julien Bezin, Frantz Thiessard, and Antoine Pariente.
Risk of drug-drug interactions in out-hospital drug dispensings in France:
Results from the drug-drug interaction prevalence study. Frontiers in
Pharmacology, 10:265, 2019.


[8] Huaqiao Jiang, Yanhua Lin, Weifang Ren, Zhonghong Fang, Yujuan Liu,
Xiaofang Tan, Xiaoqun Lv, and Ning Zhang. Adverse drug reactions and
correlations with drug-drug interactions: A retrospective study of reports
from 2011 to 2020. Frontiers in Pharmacology, 13, 2022.


[9] David Rogers and Mathew Hahn. Extended-connectivity fingerprints.
Journal of Chemical Information and Modeling, 50(5):742–754, 2010.


[10] Pieter Dewulf, Michiel Stock, and Bernard De Baets. Cold-start problems
in data-driven prediction of drug-drug interaction effects. Pharmaceuticals,
14(5):429, 2021.


[11] Zun Liu, Xing-Nan Wang, Hui Yu, Jian-Yu Shi, and Wen-Min Dong.
Predict multi-type drug-drug interactions in cold start scenario. BMC
Bioinformatics, 23(1):75, 2022.


22


[12] Junfeng Yao, Wen Sun, Zhongquan Jian, Qingqiang Wu, and Xiaoli Wang.
Effective knowledge graph embeddings based on multidirectional semantics
relations for polypharmacy side effects prediction. Bioinformatics, 38(8):
2315–2322, 2022.


[13] Marinka Zitnik, Monica Agrawal, and Jure Leskovec. Modeling polypharmacy side effects with graph convolutional networks. Bioinformatics, 34
(13):i457–i466, 2018.


[14] Md Rezaul Karim, Michael Cochez, Joao Bosco Jares, Mamtaz Uddin,
Oya Beyan, and Stefan Decker. Drug-drug interaction prediction based
on knowledge graph embeddings and convolutional-LSTM network. In
Proceedings of the 10th ACM International Conference on Bioinformatics,
Cmputational Biology and Health Informatics, pages 113–123, 2019.


[15] Kexin Huang, Cao Xiao, Lucas M Glass, Marinka Zitnik, and Jimeng
Sun. SkipGNN: predicting molecular interactions with skip-graph networks.
Scientific Reports, 10(1):1–16, 2020.


[16] Xuan Lin, Zhe Quan, Zhi-Jie Wang, Tengfei Ma, and Xiangxiang
Zeng. KGNN: Knowledge graph neural network for drug-drug interaction
prediction. In Proceedings of the Twenty-Ninth International Conference
on International Joint Conferences on Artificial Intelligence, volume 380,
pages 2739–2745, 2020.


[17] Zhong-Hao Ren, Zhu-Hong You, Chang-Qing Yu, Li-Ping Li, Yong-Jian
Guan, Lu-Xiang Guo, and Jie Pan. A biomedical knowledge graph-based
method for drug-drug interactions prediction through combining local and
global features with deep neural networks. Briefings in Bioinformatics, 23
(5):bbac363, 2022.


[18] Daniel Scott Himmelstein, Antoine Lizee, Christine Hessler, Leo
Brueggeman, Sabrina L Chen, Dexter Hadley, Ari Green, Pouya
Khankhanian, and Sergio E Baranzini. Systematic integration of biomedical
knowledge prioritizes drugs for repurposing. Elife, 6:e26726, 2017.


[19] Thomas N Kipf and Max Welling. Semi-supervised classification with
graph convolutional networks. In International Conference on Learning
Representations, 2016.


[20] Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and
George E Dahl. Neural message passing for quantum chemistry. In
International Conference on Machine Learning, pages 1263–1272. PMLR,
2017.


[21] Hui Yu, ShiYu Zhao, and JianYu Shi. STNN-DDI: a substructure-aware
tensor neural network to predict drug-drug interactions. Briefings in
Bioinformatics, 23(4):bbac209, 2022.


23


[22] David S Wishart, Yannick D Feunang, An C Guo, Elvis J Lo, Ana Marcu,
Jason R Grant, Tanvir Sajed, Daniel Johnson, Carin Li, Zinat Sayeeda,
et al. DrugBank 5.0: a major update to the DrugBank database for 2018.
Nucleic Acids Research, 46(D1):D1074–D1082, 2018.


[23] Nicholas P Tatonetti, Patrick P Ye, Roxana Daneshjou, and Russ B
Altman. Data-driven prediction of drug effects and interactions. Science
translational medicine, 4(125):125ra31–125ra31, 2012.


[24] Jacob Cohen. A coefficient of agreement for nominal scales. Educational

and Psychological Measurement, 20(1):37–46, 1960.


[25] Dean G Brown, Heike J Wobst, Abhijeet Kapoor, Leslie A Kenna, and
Noel Southall. Clinical development times for innovative drugs. Nature
reviews. Drug discovery, 21(11):793–794, 2021.


[26] Maywin Liu and Eric Wittbrodt. Low-dose oral naloxone reverses opioidinduced constipation and analgesia. Journal of Pain and Symptom
Management, 23(1):48–53, 2002.


[27] Ronald W Estabrook. A passion for P450s (remembrances of the
early history of research on cytochrome P450). Drug Metabolism and
Disposition, 31(12):1461–1473, 2003.


[28] Shikhar Vashishth, Soumya Sanyal, Vikram Nitin, and Partha Talukdar.
Composition-based multi-relational graph convolutional networks. In
International Conference on Learning Representations, 2019.


[29] Ni Lao, Tom Mitchell, and William Cohen. Random walk inference and
learning in a large scale knowledge base. In Proceedings of the 2011
Conference on Empirical Methods in Natural Language Processing, pages
529–539, 2011.


[30] Wenhan Xiong, Thien Hoang, and William Yang Wang. DeepPath:
A reinforcement learning method for knowledge graph reasoning. In
Proceedings of the 2017 Conference on Empirical Methods in Natural
Language Processing, pages 564–573, 2017.


[31] Muhan Zhang and Yixin Chen. Link prediction based on graph neural
networks. In Proceedings of the 32nd International Conference on Neural
Information Processing Systems, pages 5171–5181, 2018.


[32] Komal Teru, Etienne Denis, and Will Hamilton. Inductive relation
prediction by subgraph reasoning. In International Conference on Machine
Learning, pages 9448–9457. PMLR, 2020.


[33] Vinod Nair and Geoffrey E Hinton. Rectified linear units improve restricted
Boltzmann machines. In Proceedings of the 27th International Conference
on Machine Learning, pages 807–814, 2010.


24


[34] D. P Kingma and J. Ba. Adam: A method for stochastic optimization.
Technical report, arXiv:1412.6980, 2014.


[35] Yongqi Zhang, Ling Yue, and Quanming Yao. EmerGNN ~~D~~ DI ~~d~~ ata,
October 2023.


[36] Yongqi Zhang, Ling Yue, and Quanming Yao. EmerGNN, 10 2023. URL
`[https://github.com/LARS-research/EmerGNN](https://github.com/LARS-research/EmerGNN)` .


[37] Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using tSNE. Journal of Machine Learning Research, 9(11), 2008.


[38] James Bergstra, Brent Komer, Chris Eliasmith, Dan Yamins, and David D
Cox. Hyperopt: A Python library for model selection and hyperparameter
optimization. Computational Science & Discovery, 8(1):014008, 2015.


[39] Raheel Chaudhry, Julia H Miao, and Afzal Rehman. Physiology,
cardiovascular. In StatPearls [Internet]. StatPearls Publishing, 2022.


[40] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful
are graph neural networks? In International Conference on Learning
Representations, 2018.


25


### **Supplementary Section 1 Algorithms**

**Algorithm for EmerGNN.** In this part, we show the full algorithm and
some implementation details of EmerGNN. Given the augmented network _N_
and the drug pairs ( _u, v_ ), it will be time consuming to explicitly extract all the
paths connecting _u_ and _v_ with length _≤_ _L_ . In practice, we implicitly encode
the pair-wise representations with Algorithm 1.


**Algorithm 1** EmerGNN: pair-wise representation learning with flow-based
GNN.
**Require:** ( _u, v_ ) _, L, δ, σ, {_ _**W**_ [(] _[ℓ]_ [)] _,_ _**w**_ [(] _[ℓ]_ [)] _}_ _ℓ_ =1 _...L_ _}_ .
_{_ ( _u, v_ ): drug pair; _L_ : the depth of path-based subgraph; _δ_ : activation
function; _σ_ : sigmoid function; _{_ _**W**_ [(] _[ℓ]_ [)] _,_ _**w**_ [(] _[ℓ]_ [)] _}_ _ℓ_ =1 _...L_ _}_ : learnable parameters. _}_



1: initialize the _u →_ _v_ pair-wise representation as _**h**_ [0] _u,e_ [=] _**[ f]**_ _[u]_ [if] _[ e]_ [ =] _[ u]_ [, otherwise]
_**h**_ [0] _u,e_ [=] **[ 0]** [;]

2: initialize the _v →_ _u_ pair-wise representation as _**h**_ [0] _v,e_ [=] _**[ f]**_ _[v]_ [if] _[ e]_ [ =] _[ v]_ [, otherwise]
_**h**_ [0] _v,e_ [=] **[ 0]** [;]
3: **for** _ℓ_ _←_ 1 to _L_ **do**
4: **for** _e ∈V_ D **do** _{_ This loop can work with matrix operations in parallel. _}_
5: message for _u →_ _v_ :
_**h**_ [(] _u,e_ _[ℓ]_ [)] [=] _[ δ]_ � _**W**_ [(] _[ℓ]_ [)] [ �] ( _e_ _[′]_ _,r,e_ ) _∈N_ D _[σ]_ �( _**w**_ _r_ [(] _[ℓ]_ [)] [)] _[⊤]_ [[] _**[f]**_ _u_ [;] _**[ f]**_ _v_ []] � _·_ � _**h**_ [(] _u,e_ _[ℓ][−]_ _[′]_ [1)] _⊙_ _**h**_ [(] _r_ _[ℓ]_ [)] ��;



6: message for _v →_ _u_ :
_**h**_ [(] _v,e_ _[ℓ]_ [)] [=] _[ δ]_ � _**W**_ [(] _[ℓ]_ [)] [ �] ( _e_ _[′]_ _,r,e_ ) _∈N_ D _[σ]_ �( _**w**_ _r_ [(] _[ℓ]_ [)] [)] _[⊤]_ [[] _**[f]**_ _u_ [;] _**[ f]**_ _v_ []] � _·_ � _**h**_ [(] _v,e_ _[ℓ][−]_ _[′]_ [1)] _⊙_ _**h**_ [(] _r_ _[ℓ]_ [)] ��;



7: **end for**

8: **end for**
9: **Return** _**W**_ rel [ _**h**_ [(] _u,v_ _[L]_ [)] [;] _**[ h]**_ [(] _v,u_ _[L]_ [)] [].]


Take the direction _u →_ _v_ as an example. We initialize the representation
_**h**_ [0] _u,e_ [=] _**[ f]**_ _[u]_ [if] _[ e]_ [ =] _[ u]_ [, otherwise] _**[ h]**_ [0] _u,e_ [=] **[ 0]** [ and the messages are computed based]

on a dot product operator _**h**_ [(] _u,e_ _[ℓ][−]_ _[′]_ [1)] _⊙_ _**h**_ [(] _r_ _[ℓ]_ [)] [, then the representations of all entities]
with length longer than _ℓ_ away from _u_ will be **0** in the _ℓ_ -th step. In the end,
only the entities with length _≤_ _L_ will have valid representations. In addition,
since we return _**h**_ [(] _u,v_ _[L]_ [)] [for specific entity] _[ v]_ [, only the entities with length less than]
_L −_ _ℓ_ away from _v_ can contribute to _**h**_ [(] _u,v_ _[L]_ [)] [in the] _[ ℓ]_ [-th step. In this way, we]
implicitly encode relevant entities and relations in the biomedical network from
_u_ to _v_ . We provide a graphical illustration (Supplementary Figure 2) of the
implicit encoding procedure as follows.


- When _ℓ_ = 0, only _**h**_ [0] _u,u_ [is initialized with the non-zero features] _**[ f]**_ _[u]_ [(in black)]
and other entities are initialized as **0** (in gray).


- During the _ℓ_ -th iteration, the representations are flowed from _u_ to the _ℓ_ -th
hop neighbors of _u_ in the _ℓ_ -th step (like the formulas in black, representing
the representing a node in corresponding layer).


26


- At the last step _ℓ_ = _L_, _**h**_ _[L]_ _u,v_ [is used as the subgraph representation. We use]
boxes to indicate the representations participated in the computation of _**h**_ _[L]_ _u,v_
in each step.


- As shown, the entities in each step are identical to the entities in the left
bottom figure, implicitly encoding the subgraph representation.


**Algorithm for path extraction.** Given a drug pair ( _u, v_ ), we use beam
search to find the top _B_ = 5 paths in the direction from _u_ to _v_ and top _B_ =
5 paths from _v_ to _u_ . Take the direction from _u_ to _v_ as an example. We
provide the path extraction procedure in Algorithm 2. We provide three kinds
of lists: openList, recording the top _K_ entities in each step; closeList, recording
the accumulated scores of entities visited in each step; pathList, recording the
searched paths at each step. In lines 3-4, we obtain the sets of entities visited
in the _ℓ_ -th step _V_ _u,v_ [(] _[ℓ]_ [)] [through bi-directional bread-first-search. For each step, we]
compute the accumulated scores of entities _e ∈V_ _u,v_ _[ℓ]_ [by summing the attention]

score _α_ _r_ [(] _[ℓ]_ [)] in lines 7-8, and record the scores to the clostList. Then we pick up
edges with top- _B_ scores, and add them to openList and pathList for next step
computation in lines 11-13. After _L_ steps, we aggregate the selected paths in
pathList[1], _. . ._, pathList[ _L_ ] to obtain the top- _B_ paths from _u_ to _v_ . The same
steps are conducted to obtain the top- _B_ paths from _v_ to _u_ .


**Algorithm 2** Path extractor
**Require:** ( _u, v_ ) _, L, B_ _{B_ : the number of top paths in each direction. _}_
1: initialize openList[0] _←_ _u_ ;

2: set _V_ _u,v_ [(0)] [=] _[ {][u][}][,][ V]_ _u,v_ [(] _[L]_ [)] [=] _[ {][v][}]_ [;]

3: obtain the set _V_ _u,v_ [(] _[ℓ]_ [)] [=] _[ {][e]_ [ :] _[ d]_ [(] _[e, u]_ [) =] _[ ℓ, d]_ [(] _[e, v]_ [) =] _[ L][ −]_ _[ℓ][}][, ℓ]_ [= 1] _[, . . ., L]_ [ with]
bread-first-search;
4: **for** _ℓ_ _←_ 1 to _L_ **do**

5: set closeList[ _ℓ_ ] _←∅_, pathList[ _ℓ_ ] _←∅_ ;
6: **for** each edge in _{_ ( _e_ _[′]_ _, r, e_ ) : _e_ _[′]_ _∈_ openList[ _ℓ_ _−_ 1] _, e ∈V_ _u,v_ _[ℓ]_ _[}]_ **[ do]**

7: compute the attention weights _α_ _r_ [(] _[ℓ]_ [)] [;]

8: compute score( _u, e_ _[′]_ _, e_ ) = score( _u, e_ ) + _α_ _r_ [(] _[ℓ]_ [)] [;]
9: closeList[ _ℓ_ ].add(( _e_, score( _u, e_ _[′]_ _, e_ )));
10: **end for**

11: **for** ( _u, e_ _[′]_ _, e_ ) _∈_ top _B_ (clostList[ _ℓ_ ]) **do**
12: openList[ _ℓ_ ].add( _e_ ), pathList[ _ℓ_ ].add(( _e_ _[′]_ _, r, e_ ));
13: **end for**

14: **end for**
15: **Return:** join(pathList[1] _. . ._ pathList[ _L_ ]).


**Comparison of EmerGNN with other deep learning methods for link**
**prediction.** The general pipeline for GNN-based link prediction contains
three parts: subgraph extraction, node labeling, and GNN learning. Take


27


SEAL [31] as an example. It firstly extracts enclosing subgraph, which contains
the intersection of _L_ -hop neighbors of ( _u, v_ ), between drug pairs. In order to
distinguish the nodes on the subgraph, SEAL then labels the nodes based on
their shortest path distance to both nodes _u_ and _v_ . Finally, a GNN aggregates
the representations of node labels for _L_ steps, integrates the representations of all
the nodes, and predicts the interaction based on the integrated representation.
On subgraph extraction, we use union of paths to form a path-based
subgraph instead of the enclosing subgraph, as we need to integrate the entities
and relations on the augmented network while propagating from drug _u_ to _v_ .
Node labeling is difficult to be extended to heterogeneous graph, and a simple
extension from homogeneous graph may not lead to good performance [32].
In comparison, our designed flow-based GNN avoids the labeling problem by
propagating from _u_ to _v_ step-by-step. Benefiting by the propagation manner,
we do not need use an extra pooling layer [28, 31, 32] and just use the pair-wise
representation _**h**_ [(] _u,v_ _[L]_ [)] [in the last step to encode the path-based subgraph] _[ G]_ _u,v_ _[L]_ [.]
The above benefits are demonstrated in Fig. 5c (main text) and we provide
more analysis in Supplementary Figure 4.

### **Supplementary Section 2 Additional Results**


**Implementation of baselines.** We summarize the details of how baseline
methods are implemented for the DDI prediction tasks.


- `MLP` [9]. For each drug, there is a fingerprint vector with 1024 dimensions
generated based on the drug’s SMILES attributes, which stndw for Simplified
Molecular Input Line Entry System. Given a pair of drugs _u_ and _v_, the
fingerprints _**f**_ _u_ and _**f**_ _v_ are firstly fed into an MLP with 3 layers, respectively.
Then the representations are concatenated to compute the prediction logits
with _**l**_ ( _u, v_ ) = _**W**_ rel � _**h**_ [(] _u,v_ _[L]_ [)] [;] _**[ h]**_ [(] _v,u_ _[L]_ [)] �.


- `Similarity` [4]. We generate four fingerprints based on the SMILES
representation for each drug. For a given pair of drugs, we compute
the similarity features between this drug pair and a known set of DDIs.
Specifically, we compare the 16 pairwise similarity features composed of the
fingerprints of each drug pair, and select the maximum similarity value as
the similarity feature for the current drug pair. Subsequently, we input these
features into a random forest model to predict the DDIs.


- `CSMDDI` [11]. CSMDDI uses a RESCAL-based method to obtain embedding
representations of drugs and DDI types. It then utilizes partial least squares
regression to learn a mapping function to bridge the drug attributes to their
embeddings to predict DDIs. Finally, a random forest classifier is trained as
the predictor, and the output of the random forest classifier provides the final
prediction score for the interaction between two drugs. The implementation
follows `[https://github.com/itsosy/csmddi](https://github.com/itsosy/csmddi)` .


28


- `STNN-DDI` [21]. STNN-DDI learns a substructure×substructure×interaction
tensor, which characterizes a substructure-substructure interaction (SSI)
space, expanded by a series of rank-one tensors. According to a list of
predefined substructures with PubChem fingerprint, two given drugs are
embedded into this SSI space. A neural network is then constructed to
discern the types of interactions triggered by the drugs and the likelihood
of triggering a particular type of interaction. The implementation follows
`[https://github.com/zsy-9/STNN-DDI](https://github.com/zsy-9/STNN-DDI)` .


- `HIN-DDI` [5]. We constructs a heterogeneous information network (HIN)
that integrates a biomedical network with DDIs. Within this network,
we defined 48 distinct meta-paths, representing sequences of node types
(including compounds, genes, and diseases) that connect nodes in the HIN.
For each meta-path, a series of topological features, such as path count, was
generated. Subsequently, these features were normalized and inputted into a
random forest model for DDI prediction.


- `MSTE` [12]. MSTE learns DDI with knowledge graph embedding technique and
models the interactions as triplets in the KG. Specifically, for each interaction
( _u, i, v_ ) _∈N_ D, there are learnable embedding vectors _**e**_ _u_ _,_ _**e**_ _v_ _∈_ R _[d]_ for the
drugs _u_ and _v_, respectively, and _**i**_ _∈_ R _[d]_ for interaction type _i_ . MSTE then
computes a score _s_ ( _u, i, v_ ) = _∥_ sin( _**i**_ _·_ _**e**_ _v_ ) _·_ _**e**_ _u_ +sin( _**e**_ _u_ _·_ _**e**_ _v_ ) _·_ _**i**_ _−_ sin( _**e**_ _u_ _·_ _**i**_ ) _·_ _**e**_ _v_ _∥_ 1 _/_ 2,
which is then used as a negative logit for the prediction of interaction type
_i_ . The dimension _d_ is a hyper-parameter tuned among _{_ 32, 64, 128 _}_ . The
implementation follows `[https://github.com/galaxysunwen/MSTE-master](https://github.com/galaxysunwen/MSTE-master)` .


- `KG-DDI` [14]. KG-DDI uses a Conv-LSTM network on top of the embeddings
to compute the score of interaction triplets ( _u, i, v_ ) _∈N_ D as well as the
biomedical triplets ( _h, r, t_ ) _∈N_ B . Different from MSTE, KG-DDI firstly
optimizes the parameters on both the interaction triplets and biomedical
triplets, namely triplets in the augmented network, then fine-tunes on the
interaction triplets for final prediction. The implementation follows `[https:](https://github.com/rezacsedu/Drug-Drug-Interaction-Prediction)`
`[//github.com/rezacsedu/Drug-Drug-Interaction-Prediction](https://github.com/rezacsedu/Drug-Drug-Interaction-Prediction)` .


- `CompGCN` [28]. All the drugs, biomedical concepts, interactions and relations
have their own learnable embeddings. These embeddings are aggregated by
a graph neural network with 1 layer. The high-order embeddings _**h**_ _[L]_ _u_ _[,]_ _**[ h]**_ _[L]_ _v_ _[,]_ _**[ h]**_ _[L]_ _i_
are used to compute the score _s_ ( _u, i, v_ ) = _⟨_ _**h**_ _[L]_ _u_ _[,]_ _**[ h]**_ _[L]_ _v_ _[,]_ _**[ h]**_ _[L]_ _i_ _[⟩]_ [, which is then used as]
the logic of interaction type _i_ . The implementation follows `[https://github.](https://github.com/malllabiisc/CompGCN)`
`[com/malllabiisc/CompGCN](https://github.com/malllabiisc/CompGCN)` .


- `Decagon` [13]. Decagon is similar to CompGCN. The main difference is that
the input biomedical network only considers biomedical concepts of drugs,
genes and diseases, rather than the full biomedical network _N_ B .


- `KGNN` [16]. KGNN is built upon a GNN which propagates information and
learns node representations within the new knowledge graph. Considering
computational efficiency, KGNN employed neighbor sampling, with four


29


neighbors sampled per layer for a total of two layers. Subsequently, the learned
node representations were used to predict DDIs. The implementation follows
`[https://github.com/xzenglab/KGNN](https://github.com/xzenglab/KGNN)` .


- `SumGNN` [6]. SumGNN has three steps. First, we extract enclosing subgraphs
from the augmented network for all the drug pairs ( _u, v_ ) to be predicted.
Second, a node labeling trick is applied for all the enclosing subgraphs to
compute the node features. Then, a graph neural network computes the graph
representations of enclosing subgraphs, which are finally used to predict the
interaction. The implementation follows `[https://github.com/yueyu1030/](https://github.com/yueyu1030/SumGNN)`

`[SumGNN](https://github.com/yueyu1030/SumGNN)` .


- `DeepLGF` [17]. The DeepLGF model contains three parts. First, the SMILES
of drugs are used as sentences to encode the drugs’ chemical structure.
Second, a KG embedding model ComplEx is applied on the biomedical
network to get the global embedding information of drugs. Third, a relationalGNN is used to aggregate the representations from the biomedical network.
Finally, the three kinds of representations are fused with an MLP module for
the DDI prediction. Since there is no official code provided, we implement
this model based on CompGCN.


**Performance comparison of the S0 setting.** There are three basic settings
for the DDI prediction [10, 11, 21]: (S0) interaction between existing drugs; (S1)
interaction between emerging and existing drugs; and (S2) interaction between
emerging drugs.
`EmerGNN` has shown substantial advantage over the baseline methods
for emerging drug prediction in Table 1 (main text). We also compare
the performance in the S0 setting for prediction between existing drugs in
Supplementary Table 3, where the setting exactly follows [6]. Comparing the
two tables, we find that the emerging drug prediction task is much harder than
existing drug prediction as the accuracy values in the Table 1 (main text) are
much lower than those in Supplementary Table 3. Even though the shallow
models `MLP`, `Similarity`, `HIN-DDI` perform well in predicting DDIs for emerging
drugs, they are worse than the deep networks when predicting DDIs between
existing drugs. The embedding model `MSTE` performs very poorly for emerging
drugs but is the third best for existing drug prediction. The GNN-based
methods, especially `SumGNN`, also works well for predicting DDIs between existing
drugs. This demonstrates that drug embeddings and deep networks can be
helpful for drug interaction prediction if sufficient data are provided. `EmerGNN`,
even though specially designed for emerging drug prediction, still outperforms
the baselines with a large margin for predicting interactions between existing
drugs. These results again show the flexibility and strengths of `EmerGNN` on the
DDI prediction task.


**Path visualization.** We provide additional results for path visualization
between the case of S1 setting and S0 setting (Supplementary Figure 5).


30


Specifically, we choose examples with predicted interaction types #52, #5 and
#18 in Supplementary Table 5. We plot the interactions between emerging and
existing drugs in the left part, and the interactions between two existing drugs in
the right part. As shown in Supplementary Figure 5, relation type CrC plays an
important role during prediction, which is also reflected by the high correlations
in Fig. 3c (main text). For the interaction types on the subgraphs, we also
observe the correlations of interaction types, namely (#52, #39), (#5, #85)
and (#18, #49), which are identified in Supplementary Table 5. Comparing
the left part with right part, we observe that the biomedical entities, like
Gene::1565, Disease::DOID:10763 and Parmacologic Class::N000000102(9), play
the role to connect the emerging drug and existing drug. However, the prediction
of interactions between two existing drugs relies mainly on the DDI between
drugs. These results again verify the claim that EmerGNN is able to identify
and leverage the relevant entities and relations in the biomedical network.


31


### **Supplementary Tables**

Supplementary Table 1: Statistics of datasets used for predicting interactions
for DDI prediction. _V_ ’s represent the sets of nodes. _R_ ’s represent the sets of
interaction types. _N_ ’s represent the sets of edges.


**S0 setting:** prediction interactions between exiting drugs.


Statistics _|V_ D _|_ _|R_ I _|_ _|N_ D-train _|_ _|N_ D-valid _|_ _|N_ D-test _|_


Drugbank 1,710 86 134,641 19,224 38,419
TWOSIDES 604 200 177,568 24,887 49,656


**S1 and S2 settings:** predicting interactions for emerging drugs.






|Data seed||VD-train| |VD-valid| |VD-test|||RI|||ND-train||S1<br>|ND-valid| |ND-test||S2<br>|ND-valid| |ND-test||
|---|---|---|---|---|---|
|DrugBank<br>1<br>12<br>123<br>1234<br>12345|1,461<br>79<br>161<br>1,465<br>79<br>161<br>1,466<br>81<br>161<br>1,463<br>81<br>162<br>1,461<br>80<br>169|86<br>86<br>86<br>86<br>86|137,864<br>140,085<br>140,353<br>139,141<br>133,394|17,591<br>32,322<br>17,403<br>30,731<br>14,933<br>32,845<br>15,635<br>33,254<br>17,784<br>35,803|536<br>1,901<br>522<br>1,609<br>396<br>1,964<br>434<br>1,956<br>546<br>2,355|
|TWOSIDES<br>1<br>12<br>123<br>1234<br>12345|514<br>30<br>60<br>514<br>30<br>60<br>514<br>30<br>60<br>514<br>30<br>60<br>514<br>30<br>60|200<br>200<br>200<br>200<br>200|185,673<br>172,351<br>181,257<br>186,104<br>179,993|16,113<br>45,365<br>23,815<br>48,638<br>18,209<br>46,969<br>25,830<br>35,302<br>22,059<br>43,867|467<br>2,466<br>717<br>3,373<br>358<br>2,977<br>837<br>1,605<br>702<br>2,695|



**Biomedical network:** HetioNet [18] is used in this paper.





|Data Seed||V | |R | |N | |N | |N | |N |<br>B B B B-train B-valid B-test|
|---|---|
|DrugBank<br>1<br>12<br>123<br>1234<br>12345|34,124<br>23<br>1,690,693<br>1,656,037<br>1,666,317<br>1,690,693<br>34,124<br>23<br>1,690,693<br>1,658,075<br>1,668,273<br>1,690,693<br>34,124<br>23<br>1,690,693<br>1,657,489<br>1,667,685<br>1,690,693<br>34,124<br>23<br>1,690,693<br>1,657,400<br>1,668,685<br>1,690,693<br>34,124<br>23<br>1,690,693<br>1,656,603<br>1,668,091<br>1,690,693|
|TWOSIDES<br>1<br>12<br>123<br>1234<br>12345|34,124<br>23<br>1,690,693<br>1,671,519<br>1,678,548<br>1,690,693<br>34,124<br>23<br>1,690,693<br>1,669,693<br>1,676,696<br>1,690,693<br>34,124<br>23<br>1,690,693<br>1,672,632<br>1,678,335<br>1,690,693<br>34,124<br>23<br>1,690,693<br>1,671,617<br>1,678,528<br>1,690,693<br>34,124<br>23<br>1,690,693<br>1,672,288<br>1,678,776<br>1,690,693|


32


Supplementary Table 2: Hyper-parameters and their tuning ranges for hyperparameter selection. For all the baselines and the proposed EmerGNN, we
use the hyper-parameter optimization toolbox hyperopt [38] to search for the
optimum among 360 of hyper-parameter configurations. The objective of hyperparameter selection is to maximize the premier metric performance (F1-score in
DrugBank and PR-AUC in TWOSIDES) on the validation data. Adam [34] is
used as the optimizer to update the model parameters of EmerGNN.


Hyper-parameter Ranges


Learning rate _{_ 1 _×_ 10 _[−]_ [4] _,_ 3 _×_ 10 _[−]_ [4] _,_ 1 _×_ 10 _[−]_ [3] _,_ 3 _×_ 10 _[−]_ [3] _,_ 1 _×_ 10 _[−]_ [2] _}_
Weight decay rate _{_ 1 _×_ 10 _[−]_ [8] _,_ 1 _×_ 10 _[−]_ [6] _,_ 1 _×_ 10 _[−]_ [4] _,_ 1 _×_ 10 _[−]_ [2] _}_
Mini-batch size _{_ 32 _,_ 64 _,_ 128 _}_
Representation size _d_ _{_ 32 _,_ 64 _}_
Length of subgraphs _L_ _{_ 2 _,_ 3 _,_ 4 _}_


Supplementary Table 3: Comparison of different methods on the DDI prediction
between two existing drugs (S0 Setting). “DF” is short for “Drug Feature”;
“GF” is short for “Graph Feature”; “Emb” is short for “Embedding”; and
“GNN” is short for “Graph Neural Network”.
1

|Datasets (S0 setting)|DrugBank|TWOSIDES|
|---|---|---|
|Type<br>Methods|F1-Score<br>Accuracy<br>Kappa|PR-AUC<br>ROC-AUC<br>Accuracy|
|DF<br>MLP [9]<br>Similarity [4]|61.1_±_0.4<br>82.1_±_0.3<br>80.5_±_0.2<br>55.0_±_0.3<br>62.8_±_0.1<br>67.6_±_0.1|81.2_±_0.1<br>82.6_±_0.3<br>73.5_±_0.3<br>59.5_±_0.0<br>59.8_±_0.0<br>57.0_±_0.1|
|GF<br>HIN-DDI [5]|46.1_±_0.5<br>54.4_±_0.1<br>63.4_±_0.1|83.5_±_0.2<br>87.7_±_0.3<br>82.4_±_0.3|
|Emb<br>MSTE [12]<br>KG-DDI [14]|83.0_±_1.3<br>85.4_±_0.7<br>82.8_±_0.8<br>52.2_±_1.1<br>61.5_±_2.8<br>55.9_±_2.8|90.2_±_0.1<br>91.3_±_0.1<br>84.1_±_0.1<br>88.2_±_0.1<br>90.7_±_0.1<br>83.5_±_0.1|
|GNN<br>CompGCN [28]<br>Decagon [13]<br>KGNN [16]<br>SumGNN [6]<br>**EmerGNN**|74.3_±_1.2<br>78.8_±_0.9<br>75.0_±_1.1<br>57.4_±_0.3<br>87.2_±_0.3<br>86.1_±_0.1<br>74.0_±_0.1<br>90.9_±_0.2<br>89.6_±_0.2<br>86.9_±_0.4<br>92.7_±_0.1<br>90.7_±_0.1<br>**94.4**_±_0.7<br>**97.5**_±_0.1<br>**96.6**_±_0.8|90.6_±_0.3<br>92.3_±_0.3<br>84.8_±_0.3<br>90.6_±_0.1<br>91.7_±_0.1<br>82.1_±_0.5<br>90.8_±_0.2<br>92.8_±_0.1<br>86.1_±_0.1<br>93.4_±_0.1<br>94.9_±_0.2<br>88.8_±_0.2<br>**97.6**_±_0.1<br>**98.1**_±_0.1<br>**93.8**_±_0.2|
|**p-values**|4.5E-7<br>6.5E-13<br>6.7E-8|2.3E-8<br>5.1E-10<br>6.1E-7|



1 All of the methods are run for five times on the five-fold datasets with mean value and
standard deviation reported on the testing data. The evaluation metrics are presented in
percentage (%) with the larger value indicating better performance. The boldface numbers
indicate the best values, while the underlined numbers indicate the second best. p-values are
computed under two-sided t-testing of EmerGNN over the second best baselines.


33


Supplementary Table 4: Complexity analysis of different GNN-based methods
in the S1 setting in terms of GPU memory footprint and the number of model
parameters.

|Col1|DrugBank<br>GPU memory (MB) Model parameters|TWOSIDES<br>GPU memory (MB) Model parameters|
|---|---|---|
|Decagon<br>SumGNN<br>DeepLGF<br>EmerGNN|8,214<br>1,766,492<br>6,968<br>1,237,628<br>16,822<br>11,160,226<br>7,104<br>137,164|2,908<br>1,145,850<br>12,752<br>1,263,188<br>5,974<br>10,012,456<br>2,040<br>156,406|



Supplementary Table 5: Detailed explanation on the highlighted interaction
pairs (three yellow circles) in Fig. 3a (main text). The interaction IDs are from
the original DrugBank datasets.









|ID|Description|Exemplar drug-pairs|
|---|---|---|
|#52|#Drug1 may decrease the analgesic activities of #Drug2.|(Tapentadol, Dolasetron)|
|#39|#Drug1 may increase the constipating activities of #Drug2.|(Cyclopentolate, Ramosetron)|
|Evidence|Oral naloxone is efcacious in reversing opioid-induced constipation, but often causes the unwanted<br>side efect of analgesia reversal [26].|Oral naloxone is efcacious in reversing opioid-induced constipation, but often causes the unwanted<br>side efect of analgesia reversal [26].|
|#5|#Drug1 may decrease the vasoconstricting activities of #Drug2.|(Labetalol, Formoterol)|
|#85|#Drug1 may increase the tachycardic activities of #Drug2.|(Duloxetine, Droxidopa)|
|Evidence|This decrease in aferent signaling from the baroreceptor causes vasoconstriction and increased heart<br>rate (tachycardic) [39].|This decrease in aferent signaling from the baroreceptor causes vasoconstriction and increased heart<br>rate (tachycardic) [39].|
|#18|#Drug1 can cause an increase in the absorption of #Drug2<br>resulting in an increased serum concentration and potentially a<br>worsening of adverse efects.|(Ethanol, Levomilnacipran)|
|#49|The risk or severity of adverse efects can be increased when<br>#Drug1 is combined with #Drug2.|(Methyl salicylate, Triamcinolone)|
|Evidence|Both of the two interactions are related to worsening adverse efects when two drugs are combined<br>together.|Both of the two interactions are related to worsening adverse efects when two drugs are combined<br>together.|


34


Supplementary Table 6: Detailed explanation of selected paths from the
learned model. In these cases, we provide the target interaction sample to be
predicted, two important paths selected by our method, and the corresponding
explanations.
**Case 1 in Fig. 4a.**


**Target** : Tapentadol (DB06204) may decrease the analgesic activity of Dolasetron
(DB00757).


binds binds inv
**Path1** (0.6666): Tapentadol _−−−→_ CYP2D6 (P450) _−−−_ ~~_−−_~~ _−→_ Dolasetron
Explanation: Tapentadol can binds the P450 enzyme CYP2D6 (Gene::1565), which
is vital for the metabolism of many drugs like Dolasetron (Estabrook, 2003). In
addition, Binding of drug to plasma proteins is reversible, and changes in the ratio of
bound to unbound drug may lead to drug-drug interactions (Kneip et. al. 2008).


resembles #39: _↑_ constipating
**Path2** (0.8977): Dolasetron _−−−−−−→_ Hyoscyamine _−−−−−−−−−−−→_ Eluxadoline
_−−−_ #39 ~~_−_~~ ~~i~~ _−→_ nv Tapentadol
Explanation: Dolasetron is similar to drug Hyoscyamine (DB00424). Hyoscyamine
and Tapentadol can get some connection since they will both increase the
constipating activity of Eluxadoline (DB09272). As suggested by Liu and Wittbrodt
(2022), reversing opioid-induced constipation often causes the unwanted side effect of
analgesia reversal.


**Case 2 in Fig. 4a.**


**Target** : Labetalol (DB00598) may decrease the vasoconstricting activity of
Metaraminol (DB00610).


**Path1** (0.8274): Labetalol _−−−−−−→_ resembles Isoxsuprine _−−_ #8 ~~_−−_~~ ~~i~~ nv _→_ Dronabinol
#85: _↑_ tachycardic
_−−−−−−−−−−−→_ Metaraminol

Explanation: Lebetalol is similar to the drug Isoxsupirune (DB08941). Isoxsupirune
and Metaraminol can get some connection since Dronabinol (DB00470) will increase
the tachycardic activity of both of them. As suggested by Chaudhry et al (2022), the
decrease of vasoconstriction and the increase of tachycardic are often correlated.


**Path2** (0.8175): Metaraminol _−−−−−−−−−−−−−_ #5: _↓_ vasoconstricting ~~_−−_~~ ~~i~~ nv _→_ Spironolactone _−−−→_ treat hypertension
treat ~~i~~ nv
_−−−_ ~~_−−_~~ _→_ Labetalol

Explanation: Labetelol and Spironolactone (DB00421) get can some connections
since they treat the same disease hypertension (DOID:10763). As Spironolactone
may decrease the vasoconstricting activity of Metaraminol (indicated by the inverse
edge), we predict that Labetelol may also decrease the vasoconstricting activity of
Metaraminol.


35


Supplementary Table 7: Performance comparison of different technique
designing in EmerGNN on DrugBank dataset. “Undirected edges w.o. inverse”
means the variant that uses undirected edges instead of introducing the inverse
edges. “Subgraph representation” means the variant that learns a subgraph
representation as `SumGNN` upon _G_ _u,v_ _[L]_ [. “Uni-directional pair-wise representation”]
means the variant that only learns on the uni-directional computing (Method)
from direction _u_ to _v_ without considering the direction from _v_ to _u_ . The
performance of EmerGNN is provided in the last row as a reference.

|Variants of designing|F1-Score Accuracy Kappa|
|---|---|
|Undirected edges w.o. inverse<br>Subgraph representation<br>Uni-directional pair-wise representation|53.7_±_2.0<br>61.8_±_1.9<br>54.8_±_2.0<br>33.1_±_3.6<br>50.2_±_5.6<br>40.7_±_5.6<br>55.6_±_2.1<br>67.4_±_1.6<br>61.1_±_1.6|
|EmerGNN|62.0_±_2.0<br>68.6_±_3.7<br>62.4_±_4.3|



36


### **Supplementary Figures**























Supplementary Figure 1: Histograms of subgraph sizes of _G_ _u,v_ _[L]_ [(indicated by]
the number of edges) in the testing sets of two datasets when _L_ = 3. Median
values (8,444 for DrugBank and 59120 for TWOSIDES) are indicated by the
red dashed line.


37


**a**



**Augmented network**,



**c**



$ -#,# $ -#,! $ -#,/ $ -#,. ! $ -#,. " $ -#,. # $ -#,. $ $ -#,. % $ -#,. & $ 


$ -#,! $ -#,/ $ -#,. !



-#,. # $ -#,. $



-#,. $ $ -#,. %



-#,. % $ -#,. &



-#,. & $ -#,. '



 - ! " $ -#,# $ -#,! $ -#,/



-#,. ! $ -#,. "



-#,. " $ -#,. #



! "








|$#-<br>,#|Col2|
|---|---|
|||



$ &#,# $ &#,! $ &#,/ $ &#,. ! $ &#,. " $ &#,. # $ &#,. $ $ &#,. % $ &#,. & $ &



&#,. # $ &#,. $



&#,. $ $ &#,. %



&#,. % $ &#,. &



&#,. & $ &#,. '



$ &#,! $ &#,/ $ &#,. !



$ &#,/



&#,. ! $ &#,. "



&#,. " $ &#,. #




|$&<br>#,.!|Col2|
|---|---|
|||



$ '#,# $ '#,! $ '#,/ $ '#,. ! $ '#,. " $ '#,. # $ '#,. $ $ '#,. % $ '#,. & $ '



$ '#,! $ '#,/ $ '#,. !



$ '#,/



'#,. # $ '#,. $



'#,. $ $ '#,. %



'#,. % $ '#,. &



'#,. & $ '#,. '



'#,. ! $ '#,. "



'#,. " $ '#,. #



**b**



-(# #,!$ ; /)






























|$#'<br>,!|Col2|
|---|---|
|||


|$&<br>#,.#|Col2|
|---|---|
|||
|$#,.#<br>'|$#,.#<br>'|
|||


|$#'<br>,.%|Col2|
|---|---|
|||















/ / / representations of involved entities in 0/1/2/3-th step



Supplementary Figure 2: A graphical illustration of why the initialization step
together with the message propagation function can implicitly encode the visible
entities in each layer (step _ℓ_ ). **(a)** Symbolic representation of the augmented
network in the example of Fig. 1 (main text). Different colors in edges mean
different relation types (in Fig. 1a) and the dashed lines mean the inverse edges
with corresponding relation type. **(b)** Symbolic representation of the flow-based
GNN from _u_ to _v_ . The four circles in different colors indicate the involved
entities in different steps. **(c)** Representation flows according to the proposed
Algorithm 1 (gray symbols mean **0** vectors, and the relation types in lines are
omitted for simplicity). From top to bottom, we show how the representations
are activated and propagated in each step. The involved entities in each step
in (b) and (c) are identical to each other, indicating that our algorithm can
implicitly encoding the subgraph representation.


38


## ! 0 "

0123




### - %


### $

% [,]

### $ & [,]





4 #


4 %


### - &



Supplementary Figure 3: A graphical exampler of selected paths (the different
icons mean different drugs and genes). We show how the correlation matrices
in Fig. 3a and 3c (main text) are calculated based on this example. Given the
interaction triplet ( _u, i_ pred _, v_ ) to be predicted, we extract several paths (two in
this figure) through Algorithm 2. Here, we have two paths which contain some
relations _{r_ 1 _, . . ., r_ 4 _}_ in the biomedical network _N_ _B_ and interactions _{i_ 1 _, i_ 2 _}_ in
the interaction network _N_ _D_ . The co-occurrence times for each type _i ∈R_ I
and _r ∈R_ B are counted on the paths for different interaction triplets. For the
interaction types _i ∈R_ I or biomedical relation types _r ∈R_ B, we group their
counting values according to the to-be-predicted interaction _i_ pred and normalize
the values by dividing the frequency of _i_ pred in _N_ D-test .


Supplementary Figure 4: A detailed comparison between SumGNN and
EmerGNN in terms of subgraph structure, usage of node labeling, GNN
architecture design and the pooling mechanism. Different colors in the edges
indicate different relation types. The different circles mean different computing
steps in GNNs. **Subgraph** : The enclosing subgraph used in SumGNN contains
all the edges among entities within _L_ steps away from both _u_ and _v_ ; the pathbased subgraph only considers edges pointing from _u_ to _v_ or _v_ to _u_ . **Node**
**labeling** : SumGNN requires a node labeling procedure to compute the distance
of nodes on the subgraph to the target drugs _u_ and _v_ ; however, as the edges are
connected in the direction from _u_ to _v_, in EmerGNN, the distance information
can be reflected by the number of jumps, thus EmerGNN does not need a
node labeling procedure. **GNN architecture** : SumGNN uses the whole-graph
GNN as in [31, 32, 40] to propagate over the whole subgraph. EmerGNN uses
the flow-based GNN to propagate information from _u_ to _v_ step-by-step with a
better control of information flow. **Pooling** : In SumGNN, the representations
of all the entities in the subgraph should be pooled (for example concatenated)
for final interaction prediction; however, benefiting from the flowing pattern
of flow-based GNN, all the information can be ordered and integrated when
propagating from _u_ to _v_, thus EmerGNN only uses the final step representation
of _v_, namely _**h**_ [(] _u,v_ _[L]_ [)] [, for interaction prediction.]


40


### **a**


### **b**
































### **c d**






























### **e f**























Supplementary Figure 5: Visualization of learned paths between drug pairs.
Left: interactions between an existing drug and an emerging drug. Right:
interactions between two existing drugs. The dashed lines mean inverse types.
**(a,b)** DB00757 (Dolasetron) is an emerging drug. DB06204 (Tapentadol)
and DB00377 (Palonosetron) are existing drugs. The prediction interaction
type is #52 (#Drug1 may decrease the analgesic activities of #Drug2.).
**(c,d)** DB00598 (Labetalol) is an emerging drug. DB00610 (Metaraminol) and
DB00590 (Doxazosin) are existing drugs. The prediction interaction type is
#5 (#Drug1 may decrease the vasoconstricting activities of #Drug2.). **(e,f)**
DB08918 (Levomilnacipran) is an emerging drug. DB00898 (Ethanol) and
DB00864 (Tacrolimus) are existing drugs. The prediction interaction type is
#18 (#Drug1 can cause an increase in the absorption of #Drug2 resulting in an
increased serum concentration and potentially a worsening of adverse effects.).


41


