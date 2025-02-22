        DGDNN:DecoupledGraphDiffusionNeuralNetworkforStock
                                                MovementPrediction
                 Zinuo You1, Zijian Shi1, Hongbo Bo2, John Cartlidge1, Li Zhang3, Yan Ge1
                                                         1 University of Bristol
                                                        2 Newcastle University
                                                   3 Huawei Technologies Co., Ltd
    {zinuo.you, zijian.shi, john.cartlidge, yan.ge}@bristol.ac.uk, hongbo.bo@newcastle.ac.uk, zhangli391@huawei.com
Keywords:       Stock prediction, Graph neural network, Graph structure learning, Information propagation.
Abstract:          Forecastingfuturestocktrendsremainschallengingforacademiaandindustryduetostochasticinter-stockdy-
                 namics and hierarchical intra-stock dynamics influencing stock prices. In recent years, graph neural networks
                 have achieved remarkable performance in this problem by formulating multiple stocks as graph-structured
                 data. However, most of these approaches rely on artificially defined factors to construct static stock graphs,
                 which fail to capture the intrinsic interdependencies between stocks that rapidly evolve.  In addition, these
                 methods often ignore the hierarchical features of the stocks and lose distinctive information within.  In this
                 work, we propose a novel graph learning approach implemented without expert knowledge to address these
                 issues. First, our approach automatically constructs dynamic stock graphs by entropy-driven edge generation
                 from a signal processing perspective.  Then, we further learn task-optimal dependencies between stocks via
                 a generalized graph diffusion process on constructed stock graphs. Last, a decoupled representation learning
                 scheme is adopted to capture distinctive hierarchical intra-stock features.  Experimental results demonstrate
                 substantial improvements over state-of-the-art baselines on real-world datasets. Moreover, the ablation study
                 andsensitivitystudyfurtherillustratetheeffectivenessoftheproposedmethodinmodelingthetime-evolving
                 inter-stock and intra-stock dynamics.
1   INTRODUCTION                                                        extract temporal features from historical stock data
                                                                        and predict stock movements accordingly. However,
The stock market has long been an intensively dis-                      these methods assume independence between stocks,
cussedresearchtopicbyinvestorspursuingprofitable                        neglecting their rich connections.  In reality, stocks
trading opportunities and policymakers attempting to                    are often interrelated from which valuable informa-
gain market insights. Recent research advancements                      tion can be derived. These complicated relations be-
have primarily concentrated on exploring the poten-                     tween stocks are crucial for understanding the stock
tial of deep learning models, driven by their ability                   markets (Deng et al., 2019; Feng et al., 2019b; Feng
to model complex non-linear relationships (Bo et al.,                   et al., 2022).
2023) and automatically extract high-level features                          Tobridgethisgap,somedeeplearningmodelsat-
from raw data (Akita et al., 2016; Shi and Cartlidge,                   tempt to model the interconnections between stocks
2022).  These abilities further enable the capture of                   by integrating textual data (Sawhney et al., 2020),
intricate patterns in stock market data that traditional                such as tweets (Xu and Cohen, 2018) and news (Li
statistical methods might omit.   However, the effi-                    etal.,2020b). Nevertheless,thesemodelsheavilyrely
cient market theory (Malkiel, 2003) and the random                      on the quality of embedded extra information, result-
walk nature of stock prices make it challenging to                      ing in highly volatile performance.  Meanwhile, the
predict exact future prices with high accuracy (Adam                    transformer-based methods introduce different atten-
etal.,2016). Asaresult,researcheffortshaveshifted                       tionmechanismstocaptureinter-stockrelationsbased
towards the more robust task of anticipating stock                      on multiple time series (i.e., time series of stock indi-
movements (Jiang, 2021).                                                cators, such as open price, close price, highest price,
    Early works (Roondiwala et al., 2017; Bao et al.,                   lowest price, and trading volume) (Yoo et al., 2021;
2017) commonly adopt deep learning techniques to                        Ding et al., 2021).  Despite this advancement, these

methods often lack explicit modeling of temporal in-                 2023;Mantegna,1999). However,intheconventional
formationofthesetimeseries,suchastemporalorder                       GNN-based methods, representation learning is com-
and inter-series information (Zhu et al., 2021; Wen                  bined with the message-passing process between im-
et al., 2022).                                                       mediate neighbors in the Euclidean space.  As a re-
    Recently, Graph Neural Networks (GNNs) have                      sult, node representations become overly similar as
shown promising performance in analyzing various                     the message passes, severely distorting the distinc-
real-world networks or systems by formulating them                   tive individual node information (Huang et al., 2020;
as  graph-structured  data,  such  as  transaction  net-             Rusch et al., 2023; Liu et al., 2020). Hence, preserv-
works (Pareja et al., 2020), traffic networks (Wang                  ingthesehierarchicalintra-stockfeaturesisnecessary
et al., 2020), and communication networks (Li et al.,                for GNN-based methods in predicting stock move-
2020a).  Typically, these networks possess multiple                  ments.
entities interacting over time,  and time series data                    In this paper, we propose the Decoupled Graph
cancharacterizeeachentity. Analyzingstockmarkets                     Diffusion Neural Network (DGDNN) to address the
as complex networks is a natural choice, as previous                 abovementioned challenges.  Overall, we treat stock
works indicate (Liu and Arunkumar, 2019; Shahzad                     movement prediction as a temporal node classifica-
et al., 2018).   Moreover, various interactive mech-                 tion task and optimize the model toward identifying
anisms  (e.g.,  transmitters  and  receivers  (Shahzad               movements (classes) of stocks (nodes) on the next
et al., 2018)) that exist between stocks can be eas-                 trading day. The main contributions of this paper are
ily represented by edges (Cont and Bouchaud, 2000).                  summarised as follows:
Therefore,  these  properties  make  GNNs  powerful                    • We exploit the information entropy of nodes as
candidates  for  explicitly  grasping  inter-stock  rela-                 their pair-wise connectivities with ratios of node
tions and capturing intra-stock patterns with stock                       energy as weights, enabling the modeling of in-
graphs (Sawhney et al., 2021a; Xiang et al., 2022).                       trinsictime-varyingrelationsbetweenstocksfrom
    However, existing GNN-based models face two                           the view of information propagation.
fundamental challenges for stock movement predic-                      • We extend the layer-wise update rule of conven-
tion:  representing complicated time-evolving inter-                      tional GNNs to a decoupled graph diffusion pro-
stock dependencies and capturing hierarchical fea-                        cess.   This allows for learning the task-optimal
turesofstocks. First,specificgroupsofrelatedstocks                        graphtopologyandcapturingthehierarchicalfea-
areaffectedbyvariousfactors,whichchangestochas-                           tures of multiple stocks.
tically over time (Huynh et al., 2023).  Most graph-
based models (Kim et al.,  2019;  Ye et al.,  2021;                    • We conduct extensive experiments on real-world
Sawhney et al., 2021b) construct time-invariant stock                     stock datasets with 2,893 stocks from three mar-
graphs,whicharecontrarytothestochasticandtime-                            kets (NASDAQ, NYSE, and SSE). The experi-
evolving nature of the stock market (Adam et al.,                         mental results demonstrate that DGDNN signif-
2016).   For instance, inter-stock relations are com-                     icantly outperforms state-of-the-art baselines in
monly pre-determined by sector or firm-specific rela-                     predicting the next trading day movement, with
tionships(e.g.,belongingtothesameindustry(Sawh-                           improvementsof9.06%inclassificationaccuracy,
ney et al., 2021b) or sharing the same CEO (Kim                           0.09 in Matthew correlation coefficient, and 0.06
et al., 2019)). Besides, artificially defined graphs for                  in F1-Score.
specific tasks may not be versatile or applicable to
other tasks.  Sticking to rigid graphs risks introduc-
ingnoiseandtask-irrelevantpatternstomodels(Chen                      2   RELATEDWORK
et al., 2020). Therefore, generating appropriate stock
graphs and learning task-relevant topology remains a                 This section provides a brief overview of relevant
preliminaryyetcriticalpartofGNN-basedmethodsin                       studies.
predicting stock movements. Second, stocks possess
distinctive  hierarchical  features  (Mantegna,  1999;               2.1   GNN-basedMethodsforModeling
Sawhney et al., 2021b) that remain under-exploited
(e.g., overall market trends, group-specific dynamics,                       MultipleStocks
and individual trading patterns (Huynh et al., 2023)).               The  major  advantage  of  applying  GNNs  lies  in
Previous works indicate that these hierarchical intra-               their  graphical  structure,  which  allows  for  explic-
stock features could distinguish highly related stocks               itly modeling the relations between entities.  For in-
from different levels and be utilized for learning bet-              stance, STHAN-SR (Sawhney et al., 2021a), which
ter and more robust representations (Huynh et al.,
                                                                     is similar to the Graph Attention Neural Networks

(GATs) (Veliˇckovi´c et al., 2018), adopts a spatial-                2.3   DecoupledRepresentationLearning
temporal attention mechanism on a hypergraph with
industryandcorporateedgestocaptureinter-stockre-                     Variousnetworksorsystemsexhibituniquecharacter-
lations on the temporal domain and spatial domain.                   istics that are challenging to capture within the con-
HATS(Kimetal.,2019)predictsthestockmovement                          straints of Euclidean space, particularly when rely-
by a GAT-based method that the immediate neighbor                    ing on manually assumed prior knowledge (Huynh
nodesareselectivelyaggregatedwithlearnedweights                      et al., 2023; Sawhney et al., 2021b).  In addressing
on  manually  crafted  multi-relational  stock  graphs.              this challenge, DAGNN (Liu et al., 2020) offers the-
Moreover, HyperStockGAT (Sawhney et al., 2021b)                      oretical insights, emphasizing that the entanglement
leverages graph learning in hyperbolic space to cap-                 between representation transformation and message
turetheheterogeneityofnodedegreeandhierarchical                      propagation can hinder the performance of message-
nature of stocks on an industry-related stock graph.                 passing GNNs. SHADOW-GNN (Zeng et al., 2021),
This method illustrates that the node degree of stock                on the other hand,  concentrates on decoupling the
graphs is not evenly distributed.  Nonetheless, these                representation  learning  process  both  in  depth  and
methods directly correlate the stocks by empirical                   scope. By learning on multiple subgraphs with arbi-
assumptions or expert knowledge to construct static                  trarydepth,SHADOW-GNNpreservesthedistinctive
stockgraphs,contradictingthetime-varyingnatureof                     informationoflocalizedsubgraphsinsteadofglobally
the stock market.                                                    smoothing them into white noise. Another approach,
                                                                     MMP (Chen et al., 2022), transforms updated node
2.2   GraphTopologyLearning                                          messages into self-embedded representations. It then
                                                                     selectively aggregates these representations to form
To address the constraint of GNNs relying on the                     the final graph representation, deviating from the di-
quality  of  raw  graphs,  researchers  have  proposed               rect use of representations from the message-passing
graph structure learning to optimize raw graphs for                  process.
improved performance in downstream tasks.  These
methods can be broadly categorized into direct pa-
rameterizing  approaches  and  neural  network  ap-                  3   PRELIMINARY
proaches.  In the former category, methods treat the
adjacency matrix of the target graph as free param-                  In this section, we present the fundamental notations
eters to learn.  Pro-GNN, for instance, demonstrates                 used throughout this paper and details of the problem
that refined graphs can gain robustness by learning                  setting. Nodes represent stocks, node features repre-
perturbed raw graphs guided by critical properties of                sent their historical stock indicators, and edges repre-
raw graphs (Jin et al., 2020).   GLNN (Gao et al.,                   sent interconnections between stocks.
2020)integratessparsity,featuresmoothness,andini-
tial connectivity into an objective function to obtain               3.1   Notation
target graphs. In contrast, neural network approaches
employmorecomplexneuralnetworkstomodeledge                           Let Gt(V,Et)  represents  a  weighted  and  directed
weights based on node features and representations.                  graph on trading day t, whereV  is the set of nodes
For example, SLCNN utilizes two types of convolu-                    (stocks) {v1,...,vN}  with  the  number  of  nodes  as
tional neural networks to learn the graph structure at               |V|=   N, and Et is the set of edges (inter-stock re-
both the global and local levels (Zhang et al., 2020).               lations). LetAt∈ R N× N representstheadjacencyma-
GLCN integrates graph learning and convolutional                     trix and its entry (At)i,j represents an edge from vi
neural networks to discover the optimal graph struc-                 to vj.   The node feature matrix is denoted as Xt ∈
ture that best serves downstream tasks (Jiang et al.,                R N× (τM),whereM representsthenumberofstockin-
2019). Despitetheseadvancements,directparameter-                     dicators (i.e., open price, close price, highest price,
izingapproachesoftennecessitatecomplexandtime-                       lowest price, trading volume, etc), and τ represents
consuming alternating optimizations or bi-level opti-                thelengthofthehistoricallookbackwindow. Thefea-
mizations, and neural network approaches may over-                   turevectorofvi ontradingdayt isdenotedasxt,i. Let
look the unique characteristics of graph data or lose                ct,i represent the label of vi on trading day t, where
the positional information of nodes.                                 Ct∈ R N× 1 is the label matrix on trading dayt.

Fig.1: TheDGDNNframeworkconsistsofthreesteps: (1)constructingtherawstockgraphGt (seeSection4.1);(2)learning
the task-optimal graph topology by generalized graph diffusion (see Section 4.2); (3) applying a hierarchical decoupled
representation learning scheme (see Section 4.3).
3.2   ProblemSetting                                                   2018;Ferreretal.,2018). Additionally,stockmarkets
                                                                       exhibit  significant  node-degree  heterogeneity,  with
Since  we  are  predicting  future  trends  of  multiple               highly influential stocks having relatively large node
stocks by utilizing their corresponding stock indica-                  degrees (Sawhney et al., 2021b; Arora et al., 2006).
tors, we transform the regression task of predicting                       Consequently, we propose to model interdepen-
exact stock prices into a temporal node classifica-                    denciesbetweenstocksbytreatingthestockmarketas
tion task.  Similar to previous works on stock move-                   a communication network. Prior research (Yue et al.,
ment prediction (Kim et al., 2019; Xiang et al., 2022;                 2020) generates the asymmetric inter-stock relations
Sawhneyetal.,2021a;XuandCohen,2018;Lietal.,                            based on transfer entropy. Nonetheless, the complex
2021), we refer to this common and important task                      estimationprocessoftransferentropyand the limited
as next trading day stock trend classification.  Given                 considerationofedgeweightshampertheapproxima-
a set of stocks on the trading day t, the model learns                 tion of the intrinsic inter-stock connections.
from a historical lookback window of length τ (i.e.,                       To this end, we quantify the links between nodes
[t− τ+  1,t])andpredictstheirlabelsinthenexttimes-                     byutilizingtheinformationentropyasthedirectional
tamp (i.e., trading day t+  1). The mapping relation-                  connectivityandsignalenergyasitsintensity. Onthe
ship of this work is expressed as follows,                             one hand, if the information can propagate between
                   f(Gt(V,Et)) −→    Ct+ 1.                    (1)     entities within real-world systems, the uncertainty or
                                                                       randomness is reduced, resulting in a decrease in en-
Here, f(·) represents the proposed method DGDNN.                       tropyandanincreaseinpredictabilityatthereceiving
                                                                       entities (Jaynes, 1957; Csisz´ar et al., 2004).  On the
                                                                       other hand, the energy of the signals reflects their in-
4   METHODOLOGY                                                        tensity during propagation, which can influence the
                                                                       receivedinformationatthereceiver. Theentry(At)i,j
In this section, we detail the framework of the pro-                   is defined by,
posed DGDNN in depth, as depicted in Fig 1.                               (At)i,j=   E(xt,i)
                                                                                      E(xt,j)(eS(xt,i)+  S(xt,j)− S(xt,i,xt,j)−  1).    (2)
4.1   Entropy-DrivenEdgeGeneration                                     Here,E(·) denotesthesignalenergy,andS(·) denotes
                                                                       the information entropy.  The signal energy of vi is
Defining the graph structure is crucial for achieving                  obtained by,
reasonable performance for GNN-based approaches.
Intermsofstockgraphs,traditionalmethodsoftenes-                                                      τM− 1∑
tablishstaticrelationsbetweenstocksthroughhuman                                          E(xt,i)=          |xt,i[n]|2.                   (3)
labeling or natural language processing techniques.                                                   n= 0
However,recentpracticeshaveproventhatgenerating                        The information entropy of vi is obtained by,
dynamicrelationsbasedonhistoricalstockindicators                                      S(xt,i)=  − ∑     p(sj)lnp(sj),               (4)
is more effective (Li et al., 2021; Xiang et al., 2022).                                            j= 0
These indicators, as suggested by previous financial                   where{s0,...,sj} denotesthenon-repeatingsequence
studies (Dessaint et al., 2019; Cont and Bouchaud,                     ofxt,i and p(sj) representstheprobabilityofvaluesj.
2000; Liu and Arunkumar, 2019), can be treated as                      By definition, we can obtain p(sj) by,
noisy temporal signals.  Simultaneously, stocks can
be viewed as transmitters or receivers of informa-
tion signals, influencing other stocks (Shahzad et al.,                              p(sj)=  ∑ τM− 1n= 0  δ(sj−  xt,i[n])
                                                                                                          τM             .              (5)

Hereδ(·) denotes the Dirac delta function.
4.2   GeneralizedGraphDiffusion
However,  simply assuming constructed graphs are
perfect for performing specific tasks can lead to dis-
cordance between given graphs and task objectives,
resulting in sub-optimal model performance (Chen
et al., 2020).  Several methods have been proposed                     Fig. 2: The component-wise layout of hierarchical decou-
tomitigatethisissue,includingAdaEdge(Chenetal.,                        pled representation learning with inputXt,At.
2020)andDropEdge(Rongetal.,2019). Thesemeth-
ods demonstrate notable improvements in node clas-
sification tasks by adding or removing edges to per-                                     rl =  ∑  K− 1k= 0 θ l,kk
turbgraphtopologies,enablingmodelstocaptureand                                                  ∑  K− 1
leverage critical topological information.                                                        k= 0 θ l,k , rl >  0                   (7)
    Withthisinmind,weproposetoutilizeageneral-                         Here, large rl indicates the model explores more on
ized diffusion process on the constructed stock graph                  distant nodes and vice versa.
tolearnthetask-optimaltopology. Itenablesmoreef-
fectivecaptureoflong-rangedependenciesandglobal                        4.3   HierarchicalDecoupled
information on the graph by diffusing information                              RepresentationLearning
across larger neighborhoods (Klicpera et al., 2019).
    The following equation defines the generalized                     Theoretically, GNNs update nodes by continuously
graph diffusion at layerl,                                             aggregating direct one-hop neighbors, producing the
                    K− 1∑          K− 1∑                               final representation. However, this can lead to a high
             Ql =        θ l,kTl,k,     θ l,k=  1.             (6)     distortion of the learned representation.  It is proba-
                    k= 0            k= 0                               bly because the message-passing and representation
Here Ql denotes the diffusion matrix, K denotes the                    transformation do not essentially share a fixed neigh-
maximumdiffusionstep,θ l,k denotestheweightcoef-                       borhood in the Euclidean space (Liu et al., 2020; Xu
ficients, andTl,k denotes the column-stochastic tran-                  et al., 2018; Chen et al., 2020).  To address this is-
sition matrix.  Specifically, generalized graph diffu-                 sue, decoupled GNNs have been proposed (Liu et al.,
sion transforms the given graph structure into a new                   2020; Xu et al., 2018), aiming to decouple these two
one while keeping node signals neither amplified nor                   processes and prevent the loss of distinctive local in-
reduced.  Consequently, the generalized graph diffu-                   formation in learned representation. Similarly, meth-
sion turns the information exchange solely between                     ods such as HyperStockGAT (Sawhney et al., 2021b)
adjacent connected nodes into broader unconnected                      have explored learning graph representations in hy-
areas of the graph.                                                    perbolic spaces with attention mechanisms to capture
    Notably, θ l,k and Tl,k can be determined in ad-                   temporal features of stocks at different levels.
vance (Klicpera et al., 2019).  For instance, we can                       Inspired by these methods, we adopt a hierarchi-
use the heat kernel or the personalized PageRank to                    caldecoupledrepresentationlearningstrategytocap-
defineθ l,k, and the random walk transition matrix or                  ture hierarchical intra-stock features.  Each layer in
symmetric transition matrix to defineTl,k. Although                    DGDNN comprises a Generalized Graph Diffusion
these  pre-defined  mappings  perform  well  in  some                  layerandaCatAttentionlayerinparallel,asdepicted
datasets (e.g., CORA, CiteSeer, and PubMed) with                       in Fig. 2. The layer-wise update rule is defined by,
time-invariant relations (Zhao et al., 2021), they are                              Hl =  σ  (Ql⊙  At)Hl− 1W0l        ,
not feasible for tasks that require considering chang-                              H′l =  σ  ζ(Hl||H′l− 1)W1l +  b1l   .            (8)
ing relationships.
    Therefore, we make θ l,k as trainable parameters,                  Here, H′l  denote the node representation of l−  th
Tl,k as trainable matrices, and K as a hyperparame-                    layer,σ(·) is the activation function,ζ(·) denotes the
ter.   Furthermore, we introduce a neighborhood ra-                    multi-headattention,||denotestheconcatenation,and
dius (Zhao et al., 2021) to control the effectiveness                  Wl denotes the layer-wise trainable weight matrix.
ofthegeneralizedgraphdiffusion. Theneighborhood
radius at layerl is expressed as,

                     4.4   ObjectiveFunction                                                  Perceptron is set to 3, the number of heads of Cat At-
                                                                                              tention layers is set to 3, the embedding dimension is
                     According to Eq. 1, Eq. 6, and Eq. 7, we formulated                      set to 128, and full batch training is selected.
                     the objective function of DGDNN as follows,
                                                                                              5.3   Baseline
                                     B− 1∑                               L− 1∑
                            J =   1B                                          rl              To evaluate the performance of the proposed model,
                                     t= 0LCE(Ct+ 1, f(Xt,At))− α         l= 0                 we compared DGDNN with the following baseline
                                     L− 1∑K− 1∑                                               approaches:
                                  +      (     θ l,k−  1).                                 (9)
                                     l= 0 k= 0                                                     Table 1: Statistics of NASDAQ, NYSE, and SSE.
                     Here, LCE(·)  denotes the cross-entropy loss, B de-
                     notes the batch size, L denotes the number of infor-
                     mation propagation layers, andα  denotes the weight
                     coefficient controlling the neighborhood radius.
                     5   EXPERIMENT
                     The experiments are conducted on 3x Nvidia Tesla                         5.3.1   RNN-basedBaseline
                     T4, CUDA version 11.2.  Datasets and source code
                     are available1.                                                            • DA-RNN  (Qin  et  al.,   2017):    A  dual-stage
                     5.1   Dataset                                                                 attention-based  RNN  model  with  an  encoder-
                                                                                                   decoder structure.  The encoder utilizes an atten-
                     Following previous works (Kim et al., 2019; Xiang                             tion mechanism to extract the input time-series
                     etal.,2022;Sawhneyetal.,2021a;Lietal.,2021),we                                feature, and the decoder utilizes a temporal atten-
                     evaluate DGDNN on three real-world datasets from                              tion mechanism to capture the long-range tempo-
                     twoUSstockmarkets(NASDAQandNYSE)andone                                        ral relationships among the encoded series.
                     Chinastockmarket(SSE).Wecollecthistoricalstock                             • Adv-ALSTM (Feng et al., 2019a):  An LSTM-
                     indicators from Yahoo Finance and Google Finance                              based model that leverages adversarial training to
                     for all the selected stocks. We choose the stocks that                        improve the generalization ability of the stochas-
                     span the S&P 500 and NASDAQ composite indices                                 ticityofpricedataandatemporalattentionmech-
                     for the NASDAQ dataset.  We select the stocks that                            anism to capture the long-term dependencies in
                     spantheDowJonesIndustrialAverage,S&P500,and                                   the price data.
                     NYSE composite indices for the NYSE dataset.  We
                     choose the stocks that compose the SSE 180 for the                       5.3.2   Transformer-basedBaseline
                     SSE dataset. The details of the three datasets are pre-
                     sented in Table 1.                                                         • HMG-TF  (Ding  et  al.,  2021):   A  transformer
                                                                                                   method for modeling long-term dependencies of
                     5.2   ModelSetting                                                            financial time series. The model proposes multi-
                                                                                                   scale Gaussian priors to enhance the locality, or-
                     Based on grid search, hyperparameters are selected                            thogonal regularization to avoid learning redun-
                     using sensitivity analysis over the validation period                         dant heads in multi-head attention, and trading
                     (see Section 5.6).  For NASDAQ, we set α  =   2.9×                            gap splitter to learn the hierarchical features of
                     10− 3, τ =  19, K =  9, and L=  8.  For NYSE, we set                          high-frequency data.
                     α =  2.7×  10− 3,τ =  22,K=  10,andL=  9. ForSSE,                          • DTML (Yoo et al., 2021): A multi-level context-
                     wesetα =  8.6×  10− 3,τ =  14,K=  3,andL=  5. The                             based transformer model learns the correlations
                     training epoch is set to 1200. Adam is the optimizer                          between stocks and temporal correlations in an
                     with a learning rate of 2×  10− 4 and a weight decay                          end-to-end way.
                     of 1.5×  10− 5.  The number of layers of Muti-Layer
                          1https://github.com/pixelhero98/DGDNN
                      NASDAQ           NYSE              SSE
    TrainPeriod    05/2016-06/2017  05/2016-06/2017  05/2016-06/2017
  ValidationPeriod 07/2017-12/2017  07/2017-12/2017  07/2017-12/2017
    TestPeriod     01/2018-12/2019  01/2018-12/2019  01/2018-12/2019
 #DaysTr:Val:Test     252:64:489        252:64:489        299:128:503
     #Stocks            1026              1737               130
 #StockIndicators        5                  5                  4
#Labelpertradingday      2                  2                  2

                      Table 2: ACC, MCC, and F1-Score of proposed DGDNN and other baselines on next trading day stock trend classification
                      over the test period. Bold numbers denote the best results.
                      5.3.3   GNN-basedBaseline                                                  5.5   EvaluationResult
                        • HATS (Kim et al., 2019):  A GNN-based model                            TheexperimentalresultsarepresentedinTable2. Our
                           with a hierarchical graph attention mechanism.                        modeloutperformsbaselinemethodsregardingACC,
                           It utilizes LSTM and GRU layers to extract the                        MCC, and F1-score over three datasets. Specifically,
                           temporal features as the node representation, and                     DGDNN exhibits average improvements of 10.78%
                           themessage-passingisachievedbyselectivelyag-                          inACC,0.13inMCC,and0.10inF1-Scorecompared
                           gregating the representation of directly adjacent                     to RNN-based baseline methods.  In comparison to
                           nodes according to their edge type at each level.                     Transformer-basedmethods,DGDNNshowsaverage
                        • STHAN-SR (Sawhney et al., 2021a):  A GNN-                              improvements of 7.78% in ACC, 0.07 in MCC, and
                           based model operated on a hypergraph with two                         0.05 in F1-Score. Furthermore, when contrasted with
                           types of hyperedges:  industrial hyperedges and                       GNN-based models, DGDNN achieves average im-
                           Wikidata corporate hyperedges.   The node fea-                        provementsof7.16%inACC,0.12inMCC,and0.07
                           turesaregeneratedbytemporalHawkesattention,                           in F1-Score.
                           and weights of hyperedges are generated by hy-                            Wecanmakethefollowingobservationsbasedon
                           pergraph attention.  The spatial hypergraph con-                      experimental results.   First, models such as Graph-
                           volutionachievesrepresentationandinformation-                         WaveNet,DTML,HMG-TF,DA-RNN,andDGDNN
                           spreading.                                                            that  obtain  the  interdependencies  between  entities
                        • GraphWaveNet  (Wu  et  al.,  2019):   A  spatial-                      during the learning process perform better in most of
                           temporalgraphmodelingmethodthatcapturesthe                            the metrics than those methods (HATS, STHAN-SR,
                           spatial-temporal dependencies between multiple                        HyperStockGAT,andAdv-ALSTM)withpre-defined
                           time series by combining graph convolution with                       relationships (e.g., industry and corporate edges) or
                           dilated casual convolution.                                           without considering dependencies between entities.
                                                                                                 Second,  regarding the GNN-based models,  Hyper-
                        • HyperStockGAT  (Sawhney  et  al.,  2021b):   A                         StockGAT and DGDNN, which learn the graph rep-
                           graph attention network utilizing the hyperbolic                      resentations in different latent spaces, perform bet-
                           graph  representation  learning  on  Riemannian                       ter than those (STHAN-SR and HATS) in Euclidean
                           manifolds to predict the rankings of stocks on the                    space.
                           next trading day based on profitability.                                  Fig. 3 presents visualizations of diffusion matri-
                                                                                                 ces across three consecutive trading days, with col-
                      5.4   EvaluationMetric                                                     ors representing normalized weights.  We make the
                                                                                                 following three observations. First, stocks from con-
                      Following approaches taken in previous works (Kim                          secutive trading days do not necessarily exhibit sim-
                      et al., 2019; Xiang et al., 2022; Deng et al., 2019;                       ilar patterns in terms of information diffusion.  The
                      Sawhney et al., 2021a; Sawhney et al., 2021b), F1-                         distributions of edge weights change rapidly between
                      Score, Matthews Correlation Coefficient (MCC), and                         Fig. 3a and Fig. 3b, and between Fig. 3e and Fig. 3f.
                      Classification Accuracy (ACC) are utilized to evalu-                       Second, shallow layers tend to disseminate informa-
                      ate the performance of the models.                                         tion across a broader neighborhood. A larger number
                                                                                                 ofentriesinthediffusionmatricesarenotzeroandare
                                                                                                 distributedacrossthematricesinFig3atoFig.3c. In
                                                                                                 contrast, deeper layers tend to focus on specific lo-
            Method                            NASDAQ                                NYSE                                 SSE
                                 ACC(%)         MCC         F1-Score  ACC(%)         MCC         F1-Score  ACC(%)         MCC         F1-Score
     DA-RNN(Qinetal.,2017)      57.59±0.36  0.05±1.47×10−3  0.56±0.0156.97±0.13  0.06±1.12×10−3  0.57±0.0256.19±0.23  0.04±1.24×10−3  0.52±0.02
  Adv-ALSTM(Fengetal.,2019a)    51.16±0.42  0.04±3.88×10−3  0.53±0.0253.42±0.30  0.05±2.30×10−3  0.53±0.0252.41±0.56  0.03±6.01×10−3  0.51±0.01
    HMG-TF(Dingetal.,2021)      57.18±0.17  0.11±1.64×10−3  0.59±0.0158.49±0.12  0.09±2.03×10−3  0.59±0.0258.88±0.20  0.12±1.71×10−3  0.59±0.01
     DTML(Yooetal.,2021)        58.27±0.79  0.07±2.75×10−3  0.58±0.0159.17±0.25  0.07±3.07×10−3  0.60±0.0159.25±0.38  0.11±4.79×10−3  0.59±0.02
      HATS(Kimetal.,2019)       51.43±0.49  0.01±5.66×10−3  0.48±0.0152.05±0.82  0.02±7.42×10−3  0.50±0.0353.72±0.59  0.02±3.80×10−3  0.49±0.01
 STHAN-SR(Sawhneyetal.,2021a)   55.18±0.34  0.03±4.11×10−3  0.56±0.0154.24±0.50  0.01±5.73×10−3  0.58±0.0255.01±0.11  0.03±3.09×10−3  0.57±0.01
   GraphWaveNet(Wuetal.,2019)   59.57±0.27  0.07±2.12×10−3  0.60±0.0258.11±0.66  0.05±2.21×10−3  0.59±0.0260.78±0.23  0.06±1.93×10−3  0.57±0.01
HyperStockGAT(Sawhneyetal.,2021b)58.23±0.68  0.06±1.23×10−3  0.59±0.0259.34±0.19  0.04±5.73×10−3  0.61±0.0257.36±0.10  0.09±1.21×10−3  0.58±0.02
           DGDNN                65.07±0.25  0.20±2.33×10−3  0.63±0.0166.16±0.14  0.14±1.67×10−3  0.65±0.0164.30±0.32  0.19±4.33×10−3  0.64±0.02

               (a)Q0,t−  2                                  (b)Q0,t−  1                                     (c)Q0,t
             (d)QL− 1,t−  2                                (e)QL− 1,t−  1                                  (f)QL− 1,t
 Fig. 3: Example normalized color maps of diffusion matrices from different layers on the NYSE dataset. t=  03/06/2016.
cal areas. The entries with larger absolute values are                5.6   HyperparameterSensitivity
more centralized in Fig. 3d to Fig. 3f).  Third, even
though the initial patterns from consecutive test trad-               In this section, we explore the sensitivity of two im-
ingdaysaresimilar(asshowninFig.3bandFig.3c),                          portanthyperparameters: thehistoricallookbackwin-
differencesinlocalstructuresresultindistinctivepat-                   dow sizeτ and the maximum diffusion step K. These
terns as the layers deepen (Fig. 3e and Fig. 3f), i.e.,               hyperparameters directly affect the model’s ability to
the weights of edges can show similar distributions                   modeltherelationsbetweenmultiplestocks. Thesen-
globally,butlocalareasexhibitdifferentpatterns. For                   sitivity results ofτ and K are shown in Fig. 5. Based
instance,inFig.3f,somedarkblueclustersaredistin-                      on the sensitivity results, DGDNN consistently per-
guished from light blue clusters in shape and weight,                 forms better on the three datasets when the histori-
which might be crucial local graph structures.                        callookbackwindowsizeτ∈ [14,24]. Thiscoincides
    These results suggest that the complex relation-                  with the 20-day (i.e., monthly) professional financial
ships between stocks are not static but evolve rapidly                strategies (Adam et al., 2016).  Moreover, the opti-
over time, and the domain knowledge does not suf-                     mal K of DGDNN varies considerably with differ-
ficiently describe the intrinsic interdependencies be-                ent datasets.  On the one hand, the model’s perfor-
tween multiple entities.  The manually crafted fixed                  mance generally improves as K grows on the NAS-
stock  graph  assumes  that  the  stocks  of  the  same               DAQ dataset and the NYSE dataset, achieving the
classareconnected(Livingston,1977),neglectingthe                      optimal when K∈{ 9,10}.  On the other hand, the
possibility that stocks change to different classes as                model’s performance on the SSE dataset reaches the
time changes.  Besides, some stocks are more criti-                   peak when K=  3 and retains a slightly worse perfor-
cal than others in exhibiting the hierarchical nature                 mance as K grows. Intuitively, the stock graph of the
of intra-stock dynamics (Mantegna, 1999; Sawhney                      SSE dataset is smaller than the NASDAQ dataset and
et al., 2021b), which is hard to capture in Euclidean                 the NYSE dataset, resulting in a smallerK.
space by directly aggregating representations as the
message-passing process does.

Fig.4: Resultsoftheablationstudy. Blue: P1denotesentropy-drivenedgegeneration,P2denotesgeneralizedgraphdiffusion,
and P3 denotes hierarchical decoupled representation learning. Gray dot line: best baseline accuracy.
                                                                      data2 (Feng et al., 2019b).  We observe that apply-
                                                                      ing the industry and corporate relationships leads to
                                                                      anaverageperformancereductionofclassificationac-
                                                                      curacy by 9.23%, reiterating the importance of con-
                                                                      sidering temporally evolving dependencies between
                                                                      stocks. Moreover, when testing on the NYSE dataset
                                                                      and the SSE dataset, the degradation of model per-
                                                                      formance is slightly smaller than on the NASDAQ
                                                                      dataset.  According to financial studies (Jiang et al.,
                                                                      2011; Schwert, 2002), the NASDAQ market tends to
                                                                      be more unstable than the other two.  This might in-
                                                                      dicate that the injection of expert knowledge works
                                                                      better in less noisy and more stable markets.
                                                                      5.7.2   GeneralizedGraphDiffusion
                                                                      We explore the impact of utilizing the generalized
                                                                      graph diffusion process. Results of the ablation study
                                                                      show that DGDNN performs worse without general-
                                                                      ized graph diffusion on all datasets, with classifica-
                                                                      tion accuracy reduced by 10.43% on average.  This
                                                                      indicates that the generalized graph diffusion facil-
                                                                      itates  information  exchange  better  than  immediate
Fig. 5: Sensitivity study of the historical lookback window           neighbors with invariant structures.   While the per-
lengthτ and the maximum diffusion step K over validation              formance degradation on the SSE dataset is about
period.                                                               38%oftheperformancedegradationontheNASDAQ
                                                                      dataset and the NYSE dataset.  Since the size of the
5.7   AblationStudy                                                   stock graphs (130 stocks) of the SSE dataset is much
                                                                      smaller than the other two (1026 stocks and 1737
TheproposedDGDNNconsistsofthreecriticalcom-                           stocks), the graph diffusion process has limited im-
ponents: entropy-driven edge generation, generalized                  provements through utilizing larger neighborhoods.
graphdiffusion,andhierarchicaldecoupledrepresen-                      5.7.3   HierarchicalDecoupledRepresentation
tation learning. We further verify the effectiveness of                        Learning
each component by removing it from DGDNN. The
ablation study results are shown in Fig. 4.                           The ablation experiments demonstrate that the model
5.7.1   Entropy-drivenEdgeGeneration                                  coupling the two processes deteriorates with a reduc-
                                                                      tion of classification accuracy by 9.40% on the NAS-
To demonstrate the effectiveness of constructing dy-                  DAQdataset,8.55%ontheNYSEdataset,and5.23%
namicrelationsfromthestocksignals,wereplacethe                        on the SSE dataset. This observation empirically val-
entropy-driven edge generation with the commonly                      idates that a decoupled GNN can better capture the
adopted industry-corporate stock graph using Wiki-                        2https://www.wikidata.org/wiki/Wikidata:
                                                                      List of properties

hierarchical characteristic of stocks. Meanwhile, this                    Arora,N.,Narayanan,B.,andPaul,S.(2006). Financialin-
suggests that the representation transformation is not                          fluences and scale-free networks.  In Computational
necessarily aligned with information propagation in                             Science–ICCS 2006:  6th International Conference,
Euclidean space.  It is because different graphs ex-                            Reading, UK, Proceedings, pages 16–23. Springer.
hibit various types of inter-entities patterns and intra-                 Bao, W., Yue, J., and Rao, Y. (2017).   A deep learning
entities features, which do not always follow the as-                           framework for financial time series using stacked au-
sumptionofsmoothednodefeatures(Liuetal.,2020;                                   toencoders and long-short term memory.   PloS one,
Xu et al., 2018; Li et al., 2018).                                              12(7):e0180944.
                                                                          Bo, H., Wu, Y., You, Z., McConville, R., Hong, J., and Liu,
                                                                                W. (2023).  What will make misinformation spread:
                                                                                An xai perspective. In World Conference on Explain-
6   CONCLUSION                                                                  able Artificial Intelligence, pages 321–337. Springer.
                                                                          Chen, D., Lin, Y., Li, W., Li, P., Zhou, J., and Sun, X.
In this paper, we propose DGDNN, a novel graph                                  (2020).  Measuring and relieving the over-smoothing
learning approach for predicting the future trends of                           problem for graph neural networks from the topolog-
multiple stocks based on their historical indicators.                           ical view.  In Proceedings of the AAAI conference on
                                                                                artificial intelligence, volume 34, pages 3438–3445.
Traditionally, stock graphs are crafted based on do-                      Chen, J., Liu, W., and Pu, J. (2022).  Memory-based mes-
main knowledge (e.g., firm-specific and industrial re-                          sage passing:  Decoupling the message for propaga-
lations) or generated by alternative information (e.g.,                         tionfromdiscrimination. InICASSP2022-2022IEEE
news and reports).  To make stock graphs appropri-                              International Conference on Acoustics, Speech and
ately represent complex time-variant inter-stock re-                            Signal Processing, pages 4033–4037. IEEE.
lations, we dynamically generate raw stock graphs                         Cont, R. and Bouchaud, J.-P. (2000).  Herd behavior and
from a signal processing view considering financial                             aggregate fluctuations in financial markets.  Macroe-
theories of stock markets.  Then, we propose lever-                             conomic dynamics, 4(2):170–196.
aging the generalized graph diffusion process to opti-                    Csisz´ar, I., Shields, P. C., et al. (2004). Information theory
mize the topologies of raw stock graphs. Eventually,                            andstatistics: Atutorial. FoundationsandTrends®in
                                                                                Communications and Information Theory, 1(4):417–
the decoupled representation learning scheme cap-                               528.
turesandpreservesthehierarchicalfeaturesofstocks,                         Deng, S., Zhang, N., Zhang, W., Chen, J., Pan, J. Z., and
which are often overlooked in prior works.  The ex-                             Chen, H. (2019).  Knowledge-driven stock trend pre-
perimentalresultsdemonstrateperformanceimprove-                                 diction and explanation via temporal convolutional
ments of the proposed DGDNN over baseline meth-                                 network.   In Companion Proceedings of The 2019
ods.  The ablation study results prove the effective-                           World Wide Web Conference, pages 678–685.
ness of each module in DGDNN. Besides financial                           Dessaint,  O.,  Foucault,  T.,  Fr´esard,  L.,  and Matray,  A.
applications,theproposedmethodcanbeeasilytrans-                                 (2019). Noisy stock prices and corporate investment.
ferred to tasks that involve multiple entities exhibit-                         The Review of Financial Studies, 32(7):2625–2672.
ing interdependent and time-evolving features.  One                       Ding, Q., Wu, S., Sun, H., Guo, J., and Guo, J. (2021). Hi-
limitation of DGDNN is that it generates an overall                             erarchical multi-scale gaussian transformer for stock
                                                                                movement prediction.  In Proceedings of the Twenty-
dynamic relationship from multiple stock indicators                             NinthInternationalConferenceonInternationalJoint
withoutsufficientlyconsideringtheinterplaybetween                               Conferences on Artificial Intelligence, pages 4640–
them. Notwithstandingthepromisingresults,weplan                                 4646.
tolearnmulti-relationaldynamicstockgraphsandal-                           Feng, F., Chen, H., He, X., Ding, J., Sun, M., and Chua,
lowinformationtobefurtherdiffusedacrossdifferent                                T.-S. (2019a). Enhancing stock movement prediction
relational stock graphs in future work.                                         with adversarial training. InIJCAI.
                                                                          Feng, F., He, X., Wang, X., Luo, C., Liu, Y., and Chua, T.-
                                                                                S.(2019b). Temporalrelationalrankingforstockpre-
                                                                                diction.  ACM Transactions on Information Systems,
REFERENCES                                                                      37(2):1–30.
                                                                          Feng, S., Xu, C., Zuo, Y., Chen, G., Lin, F., and XiaHou,
Adam, K., Marcet, A., and Nicolini, J. P. (2016).   Stock                       J. (2022).   Relation-aware dynamic attributed graph
      marketvolatilityandlearning. TheJournaloffinance,                         attentionnetworkforstocksrecommendation. Pattern
      71(1):33–82.                                                              Recognition, 121:108119.
Akita, R., Yoshihara, A., Matsubara, T., and Uehara, K.                   Ferrer, R., Shahzad, S. J. H., L´opez, R., and Jare˜no, F.
      (2016).  Deep learning for stock prediction using nu-                     (2018).  Time and frequency dynamics of connected-
      merical and textual information.  In 2016 IEEE/ACIS                       ness between renewable energy stocks and crude oil
      15th International Conference on Computer and In-                         prices. Energy Economics, 76:1–20.
      formation Science, pages 1–6. IEEE.

Gao, X., Hu, W., and Guo, Z. (2020). Exploring structure-                           ence on international joint conferences on artificial
      adaptive graph learning for robust semi-supervised                            intelligence, pages 4541–4547.
      classification.  In 2020 ieee international conference                 Liu, C. and Arunkumar, N. (2019).   Risk prediction and
      on multimedia and expo, pages 1–6. IEEE.                                      evaluation of transnational transmission of financial
Huang,W.,Rong,Y.,Xu,T.,Sun,F.,andHuang,J.(2020).                                    crisisbasedoncomplexnetwork. ClusterComputing,
      Tackling over-smoothing for general graph convolu-                            22:4307–4313.
      tional networks. arXiv preprint arXiv:2008.09864.                      Liu, M., Gao, H., and Ji, S. (2020). Towards deeper graph
Huynh,  T. T.,  Nguyen,  M. H.,  Nguyen,  T. T.,  Nguyen,                           neural networks.   In Proceedings of the 26th ACM
      P. L., Weidlich, M., Nguyen, Q. V. H., and Aberer,                            SIGKDD international conference on knowledge dis-
      K.(2023). Efficientintegrationofmulti-orderdynam-                             covery & data mining, pages 338–348.
      ics and internal dynamics in stock movement predic-                    Livingston, M. (1977).   Industry movements of common
      tion.  In Proceedings of the Sixteenth ACM Interna-                           stocks. The Journal of Finance, 32(3):861–874.
      tional Conference on Web Search and Data Mining,                       Malkiel, B. G. (2003).   The efficient market hypothesis
      pages 850–858.                                                                and its critics.    Journal of economic perspectives,
Jaynes, E. T. (1957). Information theory and statistical me-                        17(1):59–82.
      chanics. Physical review, 106(4):620.                                  Mantegna, R. N. (1999).   Hierarchical structure in finan-
Jiang,   B.,   Zhang,   Z.,   Lin,   D.,   Tang,   J.,   and  Luo,                  cial  markets.    The  European  Physical  Journal  B-
      B.  (2019).     Semi-supervised  learning  with  graph                        Condensed  Matter  and  Complex  Systems,  11:193–
      learning-convolutional networks.   In Proceedings of                          197.
      the IEEE/CVF conference on computer vision and                         Pareja,A.,Domeniconi,G.,Chen,J.,Ma,T.,Suzumura,T.,
      pattern recognition, pages 11313–11320.                                       Kanezashi, H., Kaler, T., Schardl, T., and Leiserson,
Jiang, C. X., Kim, J.-C., and Wood, R. A. (2011). A com-                            C. (2020).  Evolvegcn: Evolving graph convolutional
      parison of volatility and bid–ask spread for nasdaq                           networks for dynamic graphs.  In Proceedings of the
      and nyse after decimalization.   Applied Economics,                           AAAIconferenceonartificialintelligence,volume34,
      43(10):1227–1239.                                                             pages 5363–5370.
Jiang, W. (2021).  Applications of deep learning in stock                    Qin, Y., Song, D., Cheng, H., Cheng, W., Jiang, G., and
      market prediction:  recent progress.   Expert Systems                         Cottrell, G. W. (2017).  A dual-stage attention-based
      with Applications, 184:115537.                                                recurrentneuralnetworkfortimeseriesprediction. In
Jin, W., Ma, Y., Liu, X., Tang, X., Wang, S., and Tang,                             Proceedings of the 26th International Joint Confer-
      J. (2020).  Graph structure learning for robust graph                         ence on Artificial Intelligence, pages 2627–2633.
      neural networks.   In Proceedings of the 26th ACM                      Rong, Y., Huang, W., Xu, T., and Huang, J. (2019). Drope-
      SIGKDD international conference on knowledge dis-                             dge: Towards deep graph convolutional networks on
      covery & data mining, pages 66–74.                                            nodeclassification. arXivpreprintarXiv:1907.10903.
Kim, R., So, C. H., Jeong, M., Lee, S., Kim, J., and Kang,                   Roondiwala, M., Patel, H., and Varma, S. (2017). Predict-
      J. (2019).  Hats:  A hierarchical graph attention net-                        ing stock prices using lstm.  International Journal of
      work for stock movement prediction.  arXiv preprint                           Science and Research, 6(4):1754–1756.
      arXiv:1908.07999.                                                      Rusch, T. K., Bronstein, M. M., and Mishra, S. (2023).  A
Klicpera, J., Weißenberger, S., and G¨unnemann, S. (2019).                          survey on oversmoothing in graph neural networks.
      Diffusion improves graph learning. In Proceedings of                          arXiv preprint arXiv:2303.10993.
      the33rdInternationalConferenceonNeuralInforma-                         Sawhney, R., Agarwal, S., Wadhwa, A., Derr, T., and Shah,
      tion Processing Systems, pages 13366–13378.                                   R. R. (2021a). Stock selection via spatiotemporal hy-
Li,  Q.,  Gama,  F.,  Ribeiro,  A.,  and Prorok,  A. (2020a).                       pergraph attention network:  A learning to rank ap-
      Graph neural networks for decentralized multi-robot                           proach.   In Proceedings of the AAAI Conference on
      path planning. In 2020 IEEE/RSJ International Con-                            Artificial Intelligence, volume 35, pages 497–504.
      ference on Intelligent Robots and Systems (IROS),                      Sawhney,  R.,  Agarwal,  S.,  Wadhwa,  A.,  and Shah,  R.
      pages 11785–11792. IEEE.                                                      (2020).  Deep attentive learning for stock movement
Li,  Q.,  Han,  Z.,  and  Wu,  X.-M.  (2018).     Deeper  in-                       prediction from social media text and company cor-
      sights into graph convolutional networks for semi-                            relations.  In Proceedings of the 2020 Conference on
      supervised learning. In Proceedings of the AAAI con-                          Empirical Methods in Natural Language Processing,
      ference on artificial intelligence, volume 32.                                pages 8415–8426.
Li, Q., Tan, J., Wang, J., and Chen, H. (2020b).  A mul-                     Sawhney,  R.,  Agarwal,  S.,  Wadhwa,  A.,  and Shah,  R.
      timodal event-driven lstm model for stock prediction                          (2021b). Exploringthescale-freenatureofstockmar-
      using online news. IEEE Transactions on Knowledge                             kets: Hyperbolic graph learning for algorithmic trad-
      and Data Engineering, 33(10):3323–3337.                                       ing. InProceedingsoftheWebConference,pages11–
Li, W., Bao, R., Harimoto, K., Chen, D., Xu, J., and Su,                            22.
      Q. (2021).   Modeling the stock relation with graph                    Schwert, G. W. (2002). Stock volatility in the new millen-
      network for overnight stock movement prediction. In                           nium:  how wacky is nasdaq?   Journal of Monetary
      Proceedings of the twenty-ninth international confer-                         Economics, 49(1):3–26.

Shahzad, S. J. H., Hernandez, J. A., Rehman, M. U., Al-                             from tweets and historical prices.  In Proceedings of
      Yahyaee, K. H., and Zakaria, M. (2018).   A global                            the 56th Annual Meeting of the Association for Com-
      network topology of stock markets: Transmitters and                           putational Linguistics, pages 1970–1979.
      receivers of spillover effects.   Physica A: Statistical               Ye,  J.,  Zhao,  J.,  Ye,  K.,  and  Xu,  C.  (2021).    Multi-
      Mechanics and its Applications, 492:2136–2153.                                graph convolutional network for relationship-driven
Shi, Z. and Cartlidge, J. (2022).   State dependent paral-                          stock  movement  prediction.     In  2020  25th  Inter-
      lel neural hawkes process for limit order book event                          national Conference on Pattern Recognition,  pages
      stream prediction and simulation.  In Proceedings of                          6702–6709. IEEE.
      the 28th ACM SIGKDD Conference on Knowledge                            Yoo, J., Soun, Y., Park, Y.-c., and Kang, U. (2021). Accu-
      Discovery and Data Mining, pages 1607–1615.                                   rate multivariate stock movement prediction via data-
Veliˇckovi´c,P.,Cucurull,G.,Casanova,A.,Romero,A.,Li`o,                             axis transformer with multi-level contexts.   In Pro-
      P., and Bengio, Y. (2018).  Graph attention networks.                         ceedings of the 27th ACM SIGKDD Conference on
      In International Conference on Learning Representa-                           Knowledge Discovery & Data Mining, pages 2037–
      tions.                                                                        2045.
Wang, X., Ma, Y., Wang, Y., Jin, W., Wang, X., Tang, J.,                     Yue, P., Fan, Y., Batten, J. A., and Zhou, W.-X. (2020).
      Jia, C., and Yu, J. (2020).  Traffic flow prediction via                      Information transfer between stock market sectors:
      spatialtemporalgraphneuralnetwork. InProceedings                              A comparison between the usa and china.  Entropy,
      of the web conference, pages 1082–1092.                                       22(2):194.
Wen, Q., Zhou, T., Zhang, C., Chen, W., Ma, Z., Yan, J.,                     Zeng, H., Zhang, M., Xia, Y., Srivastava, A., Malevich, A.,
      and Sun, L. (2022).  Transformers in time series:  A                          Kannan,R.,Prasanna,V.,Jin,L.,andChen,R.(2021).
      survey. arXiv preprint arXiv:2202.07125.                                      Decoupling the depth and scope of graph neural net-
Wu, Z., Pan, S., Long, G., Jiang, J., and Zhang, C. (2019).                         works.   Advances in Neural Information Processing
      Graph wavenet for deep spatial-temporal graph mod-                            Systems, 34:19665–19679.
      eling.  In Proceedings of the 28th International Joint                 Zhang, Q., Chang, J., Meng, G., Xiang, S., and Pan, C.
      Conference  on  Artificial  Intelligence,  pages  1907–                       (2020).  Spatio-temporal graph structure learning for
      1913.                                                                         traffic forecasting.  In Proceedings of the AAAI con-
Xiang, S., Cheng, D., Shang, C., Zhang, Y., and Liang,                              ference on artificial intelligence, volume 34, pages
      Y. (2022). Temporal and heterogeneous graph neural                            1177–1185.
      network for financial time series prediction.  In Pro-                 Zhao, J., Dong, Y., Ding, M., Kharlamov, E., and Tang, J.
      ceedingsofthe31stACMInternationalConferenceon                                 (2021). Adaptive diffusion in graph neural networks.
      Information&KnowledgeManagement,pages3584–                                    Advances in Neural Information Processing Systems,
      3593.                                                                         34:23321–23333.
Xu, K., Li, C., Tian, Y., Sonobe, T., Kawarabayashi, K.-                     Zhu,  Y.,  Xu,  W.,  Zhang,  J.,  Du,  Y.,  Zhang,  J.,  Liu,
      i., and Jegelka, S. (2018). Representation learning on                        Q.,   Yang,   C.,   and  Wu,   S.  (2021).      A  survey
      graphswithjumpingknowledgenetworks. InInterna-                                on  graph  structure  learning:   Progress  and  oppor-
      tional conference on machine learning, pages 5453–                            tunities.  arXiv:2103.03036 https://doi.org/10.48550/
      5462. PMLR.                                                                   arXiv.2103.03036.
Xu,Y.andCohen,S.B.(2018). Stockmovementprediction

