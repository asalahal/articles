             GraphConvolutionalNeuralNetworksforWeb-Scale
                                                   RecommenderSystems
                               Rex Ying∗†, Ruining He∗, Kaifeng Chen∗†, Pong Eksombatchai∗,
                                                  William L. Hamilton†, Jure Leskovec∗†
                                                            ∗Pinterest,†Stanford University
                                {rhe,kaifengchen,pong}@pinterest.com,{rexying,wleif,jure}@stanford.edu
ABSTRACT                                                                            [9,12].Therepresentationslearnedusingdeepmodelscanbeused
Recentadvancementsindeepneuralnetworksforgraph-structured                           tocomplement,orevenreplace,traditionalrecommendationalgo-
data have led to state-of-the-art performance on recommender                        rithmslikecollaborativefiltering.andtheselearnedrepresentations
systembenchmarks.However,makingthesemethodspracticaland                             have high utility because they can be re-used in various recom-
scalabletoweb-scalerecommendationtaskswithbillionsofitems                           mendation tasks. For example, item embeddings learned using a
and hundreds of millions of users remains a challenge.                              deep model can be used for item-item recommendation and also to
   Here we describe a large-scale deep recommendation engine                        recommendedthemedcollections(e.g.,playlists,or“feed”content).
that we developed and deployed at Pinterest. We develop a data-                        Recent years have seen significant developments in this space—
efficientGraphConvolutionalNetwork(GCN)algorithmPinSage,                            especially thedevelopment ofnewdeeplearning methodsthat are
which combines efficient random walks and graph convolutions                        capable of learning on graph-structured data, which is fundamen-
togenerateembeddingsofnodes(i.e.,items)thatincorporateboth                          talforrecommendationapplications(e.g.,toexploituser-to-item
graphstructureaswellasnodefeatureinformation.Comparedto                             interaction graphs as well as social graphs) [6, 19, 21, 24, 29, 30].
priorGCNapproaches,wedevelopanovelmethodbasedonhighly                                  Most prominent among these recent advancements is the suc-
efficient random walks to structure the convolutions and design a                   cess of deep learning architectures known as Graph Convolutional
novel training strategy that relies on harder-and-harder training                   Networks (GCNs) [19, 21, 24, 29]. The core idea behind GCNs is
examples to improve robustness and convergence of the model.                        to learn how to iteratively aggregate feature information from lo-
   WedeployPinSageatPinterest andtrainiton7.5billionexam-                           cal graph neighborhoods using neural networks (Figure 1). Here a
ples on a graph with 3 billionnodes representing pins and boards,                   single“convolution”operationtransforms andaggregatesfeature
and18billion edges.Accordingtoofflinemetrics,userstudiesand                         information from a node’s one-hop graph neighborhood, and by
A/Btests,PinSagegenerateshigher-qualityrecommendationsthan                          stackingmultiplesuchconvolutionsinformationcanbepropagated
comparable deep learning and graph-based alternatives. To our                       across far reaches of a graph. Unlike purely content-based deep
knowledge, this is the largest application of deep graph embed-                     models (e.g., recurrent neural networks [3]), GCNs leverage both
dings to date and paves the way for a new generation of web-scale                   contentinformationaswellasgraphstructure.GCN-basedmethods
recommendersystemsbasedongraphconvolutionalarchitectures.                           have set a new standard on countless recommender system bench-
ACMReferenceFormat:                                                                 marks(see[19]forasurvey).However,thesegainsonbenchmark
Rex Ying∗†, Ruining He∗, Kaifeng Chen∗†, Pong Eksombatchai∗, William L.             tasks have yet to be translated to gains in real-world production
Hamilton†, Jure Leskovec∗†. 2018. Graph Convolutional Neural Networks               environments.
forWeb-Scale RecommenderSystems.In KDD ’18: The 24th ACM SIGKDD                        The main challenge is to scale both the training as well as in-
International Conference on Knowledge Discovery & Data Mining, August               ference ofGCN-based node embeddings to graphswith billions of
19–23, 2018, London, United Kingdom.  ACM,New York,NY, USA,10 pages.                nodes and tens of billions of edges. Scaling up GCNs is difficult
https://doi.org/10.1145/3219819.3219890                                             because many of the core assumptions underlying their design are
                                                                                    violated when working in a big data environment. For example,
1   INTRODUCTION                                                                    all existing GCN-based recommender systems require operating
Deep learning methods have an increasingly critical role in rec-                    on the full graph Laplacian during training—an assumption that
ommender system applications, being used to learn useful low-                       isinfeasiblewhentheunderlyinggraphhasbillionsofnodesand
dimensionalembeddingsofimages,text,andevenindividualusers                           whose structure is constantly evolving.
Permission to make digital or hard copies of all or part of this work for personal orPresentwork.Herewepresentahighly-scalableGCNframework
classroom use isgranted without fee providedthat copies are notmade or distributed  thatwehavedevelopedanddeployedinproductionatPinterest.Our
forprofitorcommercialadvantageandthatcopiesbearthisnoticeandthefullcitation
onthefirstpage.CopyrightsforcomponentsofthisworkownedbyothersthanACM                framework,arandom-walk-basedGCNnamedPinSage,operates
mustbehonored.Abstractingwithcreditispermitted.Tocopyotherwise,orrepublish,         on a massive graph with 3 billion nodes and 18 billion edges—a
topostonserversortoredistributetolists,requirespriorspecificpermissionand/ora
fee. Request permissions from permissions@acm.org.                                  graph that is 10,000×larger than typical applications of GCNs.
KDD ’18, August 19–23, 2018, London, United Kingdom                                 PinSageleveragesseveralkeyinsightsto drasticallyimprove the
© 2018 Association for Computing Machinery.                                         scalability of GCNs:
ACM ISBN 978-1-4503-5552-0/18/08...$15.00
https://doi.org/10.1145/3219819.3219890

Figure1:Overviewofourmodelarchitectureusingdepth-2convolutions(bestviewedincolor).Left:Asmallexampleinput
graph.Right:The2-layerneuralnetworkthatcomputestheembeddingh(2)A   ofnodeAusingtheprevious-layerrepresentation,
h(1)A  ,ofnodeAandthatofitsneighborhoodN(A)(nodesB,C,D).(However,thenotionofneighborhoodisgeneralandnotall
neighborsneedtobeincluded(Section3.2).)Bottom:Theneuralnetworksthatcomputeembeddingsofeachnodeoftheinput
graph.While neuralnetworksdifferfrom nodetonodetheyall sharethesameset ofparameters(i.e.,the parametersofthe
convolve(1)andconvolve(2)functions;Algorithm1).Boxeswiththesameshadingpatternsshareparameters;γdenotesan
importancepoolingfunction;andthinrectangularboxesdenotedensely-connectedmulti-layerneuralnetworks.
•On-the-fly convolutions: Traditional GCN algorithms per-                     •Constructing convolutions via random walks: Taking full
  form graph convolutions by multiplying feature matrices by                     neighborhoods of nodesto perform convolutions (Fig.1) would
  powersofthefullgraphLaplacian.Incontrast,ourPinSagealgo-                       result in huge computation graphs, so we resort to sampling.
  rithmperformsefficient,localizedconvolutionsbysamplingthe                      However,randomsamplingissuboptimal,andwedevelopanew
  neighborhood around a node and dynamically constructing a                      technique using short random walks to sample the computa-
  computation graph from this sampled neighborhood. These dy-                    tion graph. An additional benefit is that each node now has an
  namically constructed computation graphs (Fig. 1) specify how                  importancescore,whichweuseinthepooling/aggregationstep.
  toperformalocalizedconvolutionaroundaparticularnode,and                     •Importancepooling:Acorecomponentofgraphconvolutions
  alleviatetheneedtooperateontheentiregraphduringtraining.                       is the aggregation of feature information from local neighbor-
•Producer-consumerminibatchconstruction: Wedevelopa                              hoodsinthegraph.Weintroduceamethodtoweightheimpor-
  producer-consumerarchitectureforconstructingminibatches                        tanceof nodefeaturesin thisaggregationbased uponrandom-
  that ensures maximal GPU utilization during model training. A                  walk similaritymeasures, leadingto a 46%performance gainin
  large-memory, CPU-bound producer efficiently samples node                      offline evaluation metrics.
  network neighborhoods and fetches the necessary features to                 •Curriculumtraining:Wedesignacurriculumtrainingscheme,
  definelocalconvolutions,whileaGPU-boundTensorFlowmodel                         wherethealgorithmis fedharder-and-harderexamplesduring
  consumes these pre-defined computation graphs to efficiently                   training, resulting in a 12% performance gain.
  run stochastic gradient decent.                                                We have deployed PinSage for a variety of recommendation
•Efficient MapReduce inference:Given a fully-trained GCN                      tasksatPinterest,apopularcontentdiscoveryandcurationappli-
  model,wedesignanefficientMapReducepipelinethatcandis-                       cation where users interactwith pins, whichare visualbookmarks
  tribute thetrained modelto generateembeddingsfor billionsof                 toonlinecontent(e.g.,recipestheywanttocook,orclothesthey
  nodes, while minimizing repeated computations.                              wanttopurchase).Usersorganizethesepinsintoboards,whichcon-
Inadditiontothesefundamentaladvancementsinscalability,we                      taincollections ofsimilar pins.Altogether,Pinterestis theworld’s
also introduce new training techniques and algorithmic innova-                largest user-curated graph of images, with over 2 billion unique
tions.Theseinnovationsimprovethequalityoftherepresentations                   pins collected into over 1 billion boards.
learnedbyPinSage,leadingsignificantperformancegainsindown-                       Throughextensiveofflinemetrics,controlleduserstudies,and
stream recommender system tasks:                                              A/B tests, we show that our approach achieves state-of-the-art

performance compared to other scalable deep content-based rec-                      graph neighborhoods in a producer-consumer architecture. We
ommendation algorithms, in both an item-item recommendation                         also introduce a number of new training techniques to improve
task (i.e., related-pin recommendation), as well as a “homefeed”                    performance and a MapReduce inference pipeline to scale up to
recommendation task. In offline ranking metrics we improve over                     graphs with billions of nodes.
the best performing baseline by more than 40%, in head-to-head                         Lastly,also note that graph embedding methods like node2vec
human evaluations our recommendations are preferred about 60%                       [17] and DeepWalk [26] cannot be applied here. First, these are
ofthetime,andtheA/Btestsshow 30%to 100%improvementsin                               unsupervised methods.Second, they cannot include node feature
user engagement across various settings.                                            information. Third, they directly learn embeddings of nodes and
   To our knowledge, this is the largest-ever application of deep                   thusthenumberofmodelparametersislinearwiththesizeofthe
graph embeddings and paves the way for new generation of rec-                       graph, which is prohibitive for our setting.
ommendationsystems basedongraph convolutional architectures.
                                                                                    3   METHOD
2   RELATEDWORK                                                                     Inthissection,wedescribethetechnicaldetailsofthePinSagearchi-
Our work builds upon a number of recent advancements in deep                        tectureandtraining,aswellasaMapReducepipelinetoefficiently
learning methods for graph-structured data.                                         generate embeddings using a trained PinSage model.
   The notion of neural networks for graph data was first outlined                     Thekeycomputationalworkhorseofourapproachisthenotion
in Gori et al. (2005) [15] and further elaborated on in Scarselli et                of localized graph convolutions.1 To generate the embedding for
al. (2009)[27]. However, theseinitial approachesto deeplearning                     a node (i.e., an item), we apply multiple convolutional modules
on graphs required running expensive neural “message-passing”                       that aggregate feature information (e.g., visual, textual features)
algorithms to convergence and were prohibitively expensive on                       fromthenode’slocalgraphneighborhood(Figure1).Eachmodule
large graphs. Some limitations were addressed by Gated Graph                        learns how to aggregateinformation from a small graph neighbor-
SequenceNeuralNetworks[22]—whichemploysmodernrecurrent                              hood, and by stacking multiple such modules, our approach can
neural architectures—but the approach remains computationally                       gain information about the local network topology. Importantly,
expensiveandhasmainlybeenusedongraphswith < 10,000nodes.                            parameters of these localized convolutional modules are shared
   More recently, there has been a surge of methods that rely on                    acrossallnodes,makingtheparametercomplexityofourapproach
the notion of “graph convolutions” or Graph Convolutional Net-                      independent of the input graph size.
works(GCNs).ThisapproachoriginatedwiththeworkofBrunaet                              3.1   ProblemSetup
al.(2013),whichdevelopedaversionofgraphconvolutionsbased                            Pinterest is a content discovery application where users interact
onspectralgraphthery[7].Followingonthiswork,anumberof                               withpins,whicharevisualbookmarkstoonlinecontent(e.g.,recipes
authorsproposedimprovements,extensions,andapproximations                            theywanttocook,orclothestheywanttopurchase).Usersorganize
ofthesespectralconvolutions[6,10,11,13,18,21,24,29,31],lead-                        these pins into boards, which contain collections of pins that the
ing to new state-of-the-art results on benchmarks such as node                      user deems to be thematically related. Altogether, the Pinterest
classification,linkprediction,aswellasrecommendersystemtasks                        graphcontains2billionpins,1billionboards,andover18billion
(e.g.,theMovieLensbenchmark[24]).Theseapproacheshavecon-                            edges (i.e., memberships of pins to their corresponding boards).
sistently outperformed techniques based upon matrix factorization                      Our task is to generate high-quality embeddings or representa-
or random walks (e.g., node2vec [17] and DeepWalk [26]), and                        tionsofpinsthatcanbeusedforrecommendation(e.g.,vianearest-
their successhas ledto asurge of interestin applyingGCN-based                       neighbor lookup for related pin recommendation, or for use in a
methodstoapplicationsrangingfromrecommendersystems[24]to                            downstream re-ranking system). In order to learn these embed-
drugdesign[20,31].Hamiltonetal.(2017b)[19]andBronsteinetal.                         dings, we model the Pinterest environment as a bipartite graph
(2017)[6]providecomprehensivesurveysofrecentadvancements.                           consisting of nodes in two disjoint sets,I(containing pins) and
   However, despite the successes of GCN algorithms, no previous                    C(containing boards). Note, however, that our approach is also
works have managedto apply them to production-scaledata with                        naturallygeneralizable, withIbeingviewed asa setofitems and
billions of nodes and edges—a limitation that is primarily due to                   Cas a set of user-defined contexts or collections.
the fact that traditional GCN methods require operating on the                         In addition to the graph structure, we also assume that the
entire graph Laplacian during training. Here we fill this gap and                   pins/itemsu∈Iare associated with real-valued attributes,xu∈
show that GCNs can be scaled to operate in a production-scale
recommendersystemsettinginvolvingbillionsofnodes/items.Our                          R d . In general, these attributes may specify metadata or content
work also demonstrates the substantial impact that GCNs have on                     information about an item, and in the case of Pinterest, we have
recommendation performance in a real-world environment.                             that pins are associated with both rich text and image features.
   Intermsofalgorithmdesign,ourworkismostcloselyrelatedto                           Our goal is to leverage both these input attributes as well as the
Hamiltonetal.(2017a)’sGraphSAGEalgorithm[18]andtheclosely                           structure of the bipartite graph to generate high-quality embed-
related follow-up work of Chen et al. (2018) [8]. GraphSAGE is                      dings.These embeddingsare thenusedfor recommendersystem
an inductive variant of GCNs that we modify to avoid operating                      1Following a numberof recent works (e.g., [13, 20]) we use theterm “convolutional”
on the entire graph Laplacian. We fundamentally improve upon                        to refer to a module that aggregates information from a local graph region and to
GraphSAGE by removing the limitation that the whole graph be                        denote the fact that parameters are shared between spatially distinct applications of
storedinGPUmemory,usinglow-latencyrandomwalkstosample                               this module; however, the architecture we employ does not directly approximate a
                                                                                    spectral graph convolution (though they are intimately related) [6].

candidategenerationvianearestneighborlookup(i.e.,givenapin,                       visitcountofnodesvisitedbytherandomwalk[14].2 Theneigh-
find related pins) or as features in machine learning systems for                 borhood ofu is then defined as the topT nodes with the highest
ranking the candidates.                                                           normalized visit counts with respect to nodeu.
   For notational convenience and generality, when we describe                       The advantages of this importance-based neighborhood defi-
thePinSage algorithm,we simplyrefer tothe nodesetof thefull                       nition are two-fold. First, selecting a fixed number of nodes to
graphwithV=I∪Canddonotexplicitlydistinguishbetween                                aggregate from allows us to control the memory footprint of the
pin and board nodes (unless strictly necessary), using the more                   algorithm during training [18]. Second, it allows Algorithm 1 to
general term “node” whenever possible.                                            takeintoaccounttheimportanceofneighborswhenaggregating
                                                                                  the vector representations of neighbors. In particular, we imple-
3.2   ModelArchitecture                                                           mentγ inAlgorithm1asaweighted-mean,withweightsdefined
We use localized convolutional modules to generate embeddings                     accordingto theL1 normalizedvisitcounts. Werefer tothisnew
fornodes.Westart withinput nodefeaturesand thenlearn neural                       approach as importance pooling.
networksthattransform andaggregatefeaturesover thegraphto                         Stackingconvolutions.Eachtimeweapplytheconvolveopera-
compute the node embeddings (Figure 1).                                           tion (Algorithm 1) we get a new representationfor a node, and we
Forwardpropagationalgorithm.Weconsiderthetaskofgener-                             canstackmultiplesuchconvolutionsontopofeachotherinorder
ating an embedding,zu  for a nodeu, which depends on the node’s                   togainmoreinformationaboutthelocalgraphstructurearound
input features and the graph structure around this node.                          nodeu.Inparticular,weusemultiplelayersofconvolutions,where
                                                                                  the inputs to the convolutions at layerk depend on the representa-
                                                                                  tionsoutput fromlayerk−1(Figure 1) andwhere theinitial (i.e.,
 Algorithm1:convolve                                                             “layer0”)representationsareequaltotheinputnodefeatures.Note
   Input   :Current embeddingzu  for nodeu; set of neighbor                       that the model parameters in Algorithm 1 (Q, q, W, and w) are
             embeddings{zv|v∈N(u)}, set of neighbor weights                       shared across the nodes but differ between layers.
             α ; symmetric vector functionγ(·)                                       Algorithm2detailshowstackedconvolutionsgenerateembed-
   Output:New embeddingznewu      for nodeu                                       dings fora minibatchset of nodes,M. We first computethe neigh-
1 nu←γ({ReLU(Qhv +  q)|v∈N(u)},α);                                                borhoodsofeachnodeandthenapplyK convolutionaliterations
                                                                                  to generate the layer-K representations of the target nodes. The
2 znewu ← ReLU(W·concat(zu ,nu)+  w);                                             output of the final convolutional layer is then fed through a fully-
3 znewu ← znewu /∥znewu ∥2                                                        connectedneuralnetworktogeneratethefinaloutputembeddings
                                                                                  zu ,∀u∈M.
   The core of our PinSage algorithm is a localized convolution                      The full set of parameters of our model which we then learn
operation, where we learn how to aggregate information fromu’s                    is: the weight and bias parameters for each convolutional layer
neighborhood(Figure1).ThisprocedureisdetailedinAlgorithm1                         (Q(k),q(k),W(k),w(k),∀k∈{1,..., K})aswellastheparametersof
convolve.The basicidea isthatwe transformtherepresentations                       the finaldense neuralnetwork layer,G1,G2,andg. Theoutput di-
zv,∀v∈N(u)ofu’s neighbors through a dense neural network                          mensionofLine1inAlgorithm1(i.e.,thecolumn-spacedimension
and then apply a aggregator/pooling fuction (e.g., a element-wise                 ofQ) is set to bem at all layers. For simplicity, we set the output
meanorweightedsum,denotedasγ)ontheresultingsetofvectors                           dimensionof allconvolutionallayers (i.e.,the outputatLine3 of
(Line 1). This aggregation step provides a vector representation,                 Algorithm 1) tobe equal, andwe denote this sizeparameter byd.
nu , ofu’s local neighborhood,N(u). We then concatenate the ag-                   Thefinaloutputdimensionofthemodel(afterapplyingline18of
gregated neighborhood vectornu  withu’s current representation                    Algorithm 2) is also set to bed.
hu  andtransformtheconcatenatedvectorthroughanotherdense                          3.3   ModelTraining
neural network layer (Line 2). Empirically we observe significant                 WetrainPinSageinasupervisedfashionusingamax-marginrank-
performancegains whenusingconcatenationoperation insteadof                        ing loss. In this setup, we assume that we have access to a set of
the average operation as in [21]. Additionally, the normalization in              labeled pairs of itemsL, where the pairs in the set,(q,i)∈L, are
Line3makestrainingmorestable,anditismoreefficienttoperform                        assumed to be related—  i.e., we assumethat if(q,i)∈Lthen itemi
approximate nearestneighbor searchfor normalizedembeddings                        is agood recommendation candidatefor query itemq. Thegoal of
(Section 3.5). The outputof the algorithm is a representation ofu                 thetrainingphaseistooptimizethePinSageparameterssothatthe
thatincorporatesbothinformationaboutitself anditslocalgraph                       output embeddings of pairs(q,i)∈Lin the labeled set are close
neighborhood.
Importance-basedneighborhoods.Animportantinnovationin                             together.
ourapproachishowwedefinenodeneighborhoodsN(u),i.e.,how                               Wefirstdescribeourmargin-basedlossfunctionindetail.Follow-
we select the set of neighbors to convolve over in Algorithm 1.                   ingthis,wegiveanoverviewofseveraltechniqueswedeveloped
Whereasprevious GCNapproachessimplyexaminek-hopgraph                              that lead to the computation efficiency and fast convergence rate
neighborhoods, in PinSage we define importance-based neighbor-                    of PinSage, allowing us to train on billion node graphs and billions
hoods,wheretheneighborhoodofanodeuisdefinedastheT nodes                           trainingexamples.Andfinally,wedescribeourcurriculum-training
that exert themost influence on nodeu. Concretely, we simulate                    2In the limit of infinite simulations, the normalized counts approximate the Personal-
randomwalksstartingfromnodeu andcomputetheL1-normalized                           ized PageRank scores with respect tou .

 Algorithm2:minibatch                                                                WeusetechniquessimilartothoseproposedbyGoyal et al.[16]
   Input   :Set of nodesM⊂V; depth parameterK;                                    toensurefastconvergenceandmaintaintrainingandgeneralization
             neighborhood functionN:V→2V                                          accuracy when dealing with large batch sizes. We use a gradual
   Output:Embeddingszu ,∀u∈M                                                      warmup procedure that increases learning rate from small to a
   /*Sampling neighborhoods of minibatch nodes.       */                          peak value in the first epoch according to the linear scaling rule.
                                                                                  Afterwards the learning rate is decreased exponentially.
1S(K)←M;                                                                          Producer-consumerminibatchconstruction. Duringtraining,
2 fork =  K,..., 1do                                                              the adjacency list and the feature matrix for billions of nodes are
3 S(k−1)←S(k);                                                                    placedinCPUmemoryduetotheirlargesize.However,duringthe
4       foru∈S(k)do                                                               convolve step of PinSage, each GPU process needs access to the
5   S(k−1)←S(k−1)∪N(u);                                                           neighborhood and feature information of nodes in the neighbor-
6       end                                                                       hood.AccessingthedatainCPUmemoryfromGPUisnotefficient.
7 end                                                                             To solve this problem, we use a re-indexing technique to create a
   /*Generating embeddings                                                 */     sub-graphG′=(V′,E′)containingnodesandtheirneighborhood,
                                                                                  whichwillbeinvolvedinthecomputationofthecurrentminibatch.
8 h(0)u ← xu ,∀u∈S(0);                                                            A small feature matrix containing only node features relevant to
9 fork =  1,..., K do                                                             computation of the current minibatch is also extracted such that
10       foru∈S(k)don                   o                                         theorderisconsistentwiththeindexofnodesinG′.Theadjacency
11   H←             h(k−1)v        ,∀v∈N(u);                                      listofG′andthesmallfeaturematrixarefedintoGPUsatthestart
                                                                                  of each minibatch iteration, so that no communication between
12             h(k)u ← convolve(k) h(k−1)u         ,H                             the GPU and CPU is needed during the convolve step, greatly
13       end                                                                      improving GPU utilization.
14 end                                                                               ThetrainingprocedurehasalternatingusageofCPUsandGPUs.
15 foru∈Mdo                                                                       ThemodelcomputationsareinGPUs,whereasextractingfeatures,
                                                                                  re-indexing,andnegativesamplingarecomputedonCPUs.Inad-
16       zu← G2·ReLU       G1h(K)u    +  g
                                                                                  dition to parallelizing GPU computation with multi-tower training,
17 end                                                                            and CPU computation using OpenMP [25], we design a producer-
                                                                                  consumerpattern torunGPUcomputationat thecurrentiteration
                                                                                  and CPU computation at the next iteration in parallel. This further
scheme, which improves the overall quality of the recommenda-                     reduces the training time by almost a half.
tions.                                                                            Samplingnegativeitems. Negative samplingis usedin our loss
Lossfunction.Inordertotrain theparametersofthemodel,we                            function (Equation 1) as an approximation of the normalization
use a max-margin-based loss function. The basic idea is that we                   factorofedgelikelihood[23].Toimproveefficiencywhentraining
wanttomaximizetheinnerproductofpositiveexamples, i.e.,the                         withlargebatchsizes,wesampleasetof500negativeitemstobe
embedding of the query item and the corresponding related item.                   shared by all training examplesin each minibatch. This drastically
At the same time we want to ensure that the inner product of                      saves the number of embeddings that need to be computed during
negativeexamples—  i.e.,theinnerproductbetweentheembedding                        eachtrainingstep,comparedtorunningnegativesamplingforeach
ofthequeryitemandanunrelateditem—issmallerthanthatofthe                           node independently. Empirically, we do not observe a difference
positivesamplebysomepre-defined margin.Thelossfunctionfor                         between the performance of the two sampling schemes.
a single pair of node embeddings(zq ,zi):(q,i)∈Lis thus                              In the simplest case, we could just uniformly sample negative
        JG(zq zi)=  E n k∼P n(q)max{0,zq·zn k−zq·zi + ∆},       (1)               examplesfromtheentiresetofitems.However,ensuringthatthe
                                                                                  innerproduct ofthe positive example(pair ofitems(q,i))is larger
wherePn(q)denotesthedistributionofnegativeexamplesforitem                         thanthatoftheqandeachofthe500negativeitemsistoo“easy”and
q,and∆  denotesthemarginhyper-parameter.Weshallexplainthe                         does not provide fine enough “resolution” for the system to learn.
sampling of negative samples below.                                               In particular, our recommendation algorithm should be capable
Multi-GPUtrainingwithlargeminibatches.Tomakefull use                              of finding 1,000 most relevant items to q among the catalog of
of multipleGPUs ona singlemachine fortraining, werun thefor-                      over 2 billion items. In other words, our model should be able to
ward and backward propagation in a multi-tower fashion. With                      distinguish/identify 1 item out of 2 million items. But with 500
multipleGPUs,wefirstdivideeachminibatch(Figure1bottom)into                        random negative items, the model’s resolution is only 1 out of
equal-sizedportions.EachGPUtakesoneportionoftheminibatch                          500.Thus,ifwesample500randomnegativeitemsoutof2billion
andperformsthecomputationsusingthesamesetofparameters.Af-                         items, the chanceof any of these itemsbeing even slightly related
ter backward propagation, the gradients for each parameter across                 to the query item is small. Therefore, with large probability the
allGPUsareaggregatedtogether,andasinglestepofsynchronous                          learningwillnotmakegoodparameterupdatesandwillnotbeable
SGD is performed. Due to the need to train on extremely large                     to differentiate slightly related items from the very related ones.
numberofexamples(onthescaleofbillions),werunoursystem                                To solve the above problem, for each positive training example
with large batch sizes, ranging from 512 to 4096.                                 (i.e., item pair(q,i)), we add “hard” negative examples,  i.e., items

                                                                                 Note that our approach avoids redundant computations and that
                                                                                 thelatentvectorforeachnodeiscomputedonlyonce.Aftertheem-
                                                                                 beddings of the boards are obtained, we use two more MapReduce
                                                                                 jobstocomputethesecond-layerembeddingsofpins,inasimilar
                                                                                 fashion as above, and this process can be iterated as necessary (up
                                                                                 toK convolutional layers).3
Figure 2: Random negative examples and hard negative ex-                         3.5   Efficientnearest-neighborlookups
amples. Notice that the hard negative example is signifi-                        TheembeddingsgeneratedbyPinSagecanbeusedforawiderange
cantlymoresimilartothequery,thantherandomnegative                                of downstream recommendation tasks, and in many settings we
example,thoughnotassimilarasthepositiveexample.                                  can directly use these embeddings to make recommendations by
                                                                                 performing nearest-neighbor lookups in the learned embedding
                                                                                 space. Thatis, given a queryitemq, thewe can recommend items
                                                                                 whoseembeddingsaretheK-nearestneighborsofthequeryitem’s
that are somewhat related to the query itemq, but not as related                 embedding. Approximate KNN can be obtained efficiently via lo-
as the positive itemi. We call these “hard negative items”. They                 calitysensitivehashing[2].Afterthehashfunctioniscomputed,
are generated by ranking items in a graph according to their Per-                retrievalofitemscanbeimplementedwithatwo-levelretrievalpro-
sonalizedPageRankscoreswithrespecttoqueryitemq [14].Items                        cess based onthe WeakAND operator [5]. Giventhat the PinSage
ranked at 2000-5000 are randomly sampled as hard negative items.                 modelistrainedofflineandallnodeembeddingsarecomputedvia
As illustrated in Figure 2, the hard negative examples are more                  MapReduce and saved in a database, the efficient nearest-neighbor
similartothequerythanrandomnegativeexamples,andarethus                           lookup operation enables the system to serve recommendations in
challenging for the model to rank, forcing the model to learn to                 an online fashion,
distinguish items at a finer granularity.
   Using hard negative items throughout the training procedure                   4   EXPERIMENTS
doubles the number of epochs needed for the training to con-                     To demonstrate the efficiency of PinSage and the quality of the
verge.Tohelpwithconvergence,wedevelopacurriculumtraining                         embeddings it generates, we conduct a comprehensive suite of
scheme[4].Inthefirstepochoftraining,nohardnegativeitemsare                       experimentsontheentirePinterestobjectgraph,includingoffline
used, sothat thealgorithm quicklyfinds an areain theparameter                    experiments, production A/B tests as well as user studies.
space where the loss is relatively small. We then add hard negative
items in subsequent epochs, focusing the model to learn how to                   4.1   ExperimentalSetup
distinguishhighlyrelatedpinsfromonlyslightlyrelatedones.At                       We evaluate the embeddings generated by PinSage in two tasks:
epochn ofthetraining,weaddn−1hardnegativeitemstotheset                           recommending related pins and recommending pins in a user’s
of negative items for each item.                                                 home/newsfeed.To recommendrelated pins,we selecttheK near-
3.4   NodeEmbeddingsviaMapReduce                                                 estneighborstothequerypinintheembeddingspace.Weevaluate
                                                                                 performance on this related-pinrecommendation task using both
Afterthemodelistrained,itisstillchallengingtodirectlyapplythe                    offline ranking measures as well as a controlled user study. For the
trainedmodeltogenerateembeddingsforallitems,includingthose                       homefeedrecommendationtask,weselectthepinsthatareclosest
thatwerenotseenduringtraining.Naivelycomputingembeddings                         intheembeddingspacetooneofthemostrecentlypinneditemsby
fornodesusingAlgorithm2leadstorepeatedcomputationscaused                         the user. We evaluate performance of a fully-deployed production
by the overlap betweenK-hop neighborhoods of nodes. As illus-                    systemon thistask usingA/B teststo measurethe overallimpact
tratedinFigure1,manynodesarerepeatedlycomputedatmultiple                         on user engagement.
layerswhen generatingtheembeddingsfor differenttargetnodes.                      Training details and data preparation. We define the set,L,
Toensureefficient inference,wedevelop aMapReduceapproach                         of positive training examples (Equation (1)) using historical user
that runs model inference without repeated computations.                         engagementdata.Inparticular,weusehistoricaluserengagement
   Weobservethatinferenceofnodeembeddingsverynicelylends                         datatoidentifypairsofpins(q,i),whereauserinteractedwithpin
itselftoMapReducecomputationalmodel.Figure3detailsthedata                        i immediatelyaftersheinteractedwithpinq.Weuseallotherpins
flowonthebipartitepin-to-boardPinterestgraph,whereweassume                       as negative items (and sample them as described in Section 3.3).
theinput(i.e.,“layer-0”)nodesarepins/items(andthelayer-1nodes                    Overall, we use 1.2 billion pairsof positive training examples(in
are boards/contexts). The MapReduce pipeline has two key parts:                  addition to 500 negative examples per batch and 6 hard negative
   (1) One MapReduce job is used to project all pins to a low-                   examplesperpin).Thusintotalweuse7.5billiontrainingexamples.
       dimensionallatentspace,wheretheaggregationoperation                           Since PinSage can efficiently generate embeddings for unseen
       will be performed (Algorithm 1, Line 1).                                  data, we only train on a subset of the Pinterest graph and then
   (2) AnotherMapReducejobisthenusedtojointheresultingpin                        generate embeddings for the entire graph using the MapReduce
       representationswiththeidsoftheboardstheyoccurin,and                       pipelinedescribed inSection3.4. Inparticular,fortraining weuse
       theboardembeddingiscomputedbypoolingthefeaturesof                         3Note that since we assume that only pins (and not boards) have features, we must
       its (sampled) neighbors.                                                  use an even number of convolutional layers.

Figure3:NodeembeddingdataflowtocomputethefirstlayerrepresentationusingMapReduce.Thesecondlayercomputation
followsthesamepipeline,exceptthattheinputsarefirstlayerrepresentations,ratherthanrawitemfeatures.
arandomlysampledsubgraphoftheentiregraph,containing 20%                                state-of-the-artat Pinterestfor certainrecommendationtasks
of all boards(and all the pinstouched by those boards)and 70%of                        [14] and thus an informative baseline.
the labeled examples.During hyperparameter tuning, a remaining                     The visual and annotation embeddings are state-of-the-art deep
10% of the labeled examples are used. And, when testing, we run                    learningcontent-basedsystemscurrentlydeployedatPinterestto
inference on the entire graph to compute embeddings for all 2                      generate representations of pins. Note that we do not compare
billion pins, and the remaining 20% of the labeled examples are                    against other deep learning baselines from the literature simply
usedtotesttherecommendationperformanceofourPinSageinthe                            due to the scale of our problem. We also do not consider non-deep
offlineevaluations.Notethattrainingonasubsetofthefullgraph                         learningapproachesforgeneratingitem/contentembeddings,since
drasticallydecreasedtrainingtime,withanegligibleimpactonfinal                      otherworkshavealreadyprovenstate-of-the-artperformanceof
performance. In total, the full datasets for training and evaluation               deep learning approaches for generatingsuch embeddings [9, 12,
are approximately 18TB in size with the full output embeddings                     24].
being 4TB.                                                                            We also conduct ablation studies and consider several variants
Features used for learning. Each pin at Pinterest is associated                    of PinSage when evaluating performance:
withanimageandasetoftextualannotations(title,description).To                         •max-poolingusestheelement-wisemaxasasymmetricaggre-
generate feature representationxq  for each pinq, we concatenate                       gation function(i.e.,γ =  max) without hard negative samples;
visualembeddings(4,096dimensions),textualannotationembed-                            •mean-pooling uses the element-wise mean as a symmetric
dings(256 dimensions),and thelogdegree ofthe node/pininthe                             aggregation function (i.e.,γ =  mean);
graph. The visual embeddings are the 6-th fully connected layer of                   •mean-pooling-xentisthesameasmean-poolingbutusesthe
a classificationnetwork usingthe VGG-16architecture [28]. Tex-                         cross-entropy loss introduced in [18].
tualannotationembeddingsaretrainedusingaWord2Vec-based                               •mean-pooling-hardisthesameasmean-pooling,exceptthat
model [23], where the context of an annotation consists of other                       itincorporateshardnegativesamplesasdetailedinSection3.3.
annotations that are associated with each pin.                                       •PinSageusesalloptimizationspresentedinthispaper,includ-
Baselinesforcomparison.WeevaluatetheperformanceofPin-                                  ing the use of importance pooling in the convolution step.
Sage against the following state-of-the-art content-based, graph-                  Themax-poolingandcross-entropysettingsareextensionsofthe
based and deep learning baselines that generate embeddings of                      best-performing GCN model from Hamilton et al. [18]—other vari-
pins:                                                                              ants(e.g.,basedonKipfetal.[21])performedsignificantlyworse
(1) Visual embeddings (Visual): Uses nearest neighbors of deep                     indevelopmenttestsandareomittedforbrevity.4 Foralltheabove
    visualembeddingsforrecommendations.Thevisualfeatures                           variants, we usedK =  2, hidden dimension sizem =  2048, and set
    are described above.                                                           the embedding dimensiond to be 1024.
(2) Annotation embeddings (Annotation): Recommends based                           Computationresources. Training of PinSage is implemented in
    onnearestneighborsintermsofannotationembeddings.The                            TensorFlow [1] and run on a single machine with 32 cores and
    annotation embeddings are described above.                                     16 Tesla K80 GPUs. To ensure fast fetching of item’s visual and
(3) Combined embeddings (Combined): Recommends based on                            annotation features, we store them in main memory, together with
    concatenating visual and annotation embeddings, and using                      the graph, using Linux HugePages to increase the size of virtual
    a 2-layermulti-layer perceptron to computeembeddings that                      memorypagesfrom4KBto2MB.Thetotalamountofmemoryused
    capture both visual and annotation features.                                   in training is500GB. Our MapReduce inferencepipeline is run on
(4) Graph-basedmethod(Pixie):Thisrandom-walk-basedmethod                           a Hadoop2 cluster with 378 d2.8xlarge Amazon AWS nodes.
    [14]usesbiasedrandomwalkstogeneraterankingscoresby
    simulating random walks starting at query pinq. Items with
    top K scores are retrieved as recommendations. While this                      4Note that the recent GCN-based recommender systems of Monti et al. [24] and Berg
    approach does not generate pin embeddings, it is currently the                 et al. [29] are not directly comparable because they cannot scale to the Pinterest size
                                                                                   data.

                      Method             Hit-rate    MRR
                       Visual               17%       0.23
                    Annotation              14%       0.19
                     Combined               27%       0.37
                    max-pooling             39%       0.37
                   mean-pooling             41%       0.51
                mean-pooling-xent           29%       0.35
                mean-pooling-hard           46%       0.56
                      PinSage              67%        0.59
Table 1: Hit-rate and MRR for PinSage and content-based
deep learning baselines. Overall, PinSage gives 150% im-
provement in hit rate and 60% improvement in MRR over
thebestbaseline.5
4.2   OfflineEvaluation                                                            Figure 4: Probability density of pairwise cosine similarity
                                                                                   for visual embeddings, annotation embeddings, and Pin-
Toevaluate performanceon therelatedpin recommendationtask,                         Sageembeddings.
wedefinethenotionofhit-rate.Foreachpositivepairofpins(q,i)
in the test set, we useq as a query pin and then compute its top                   plots thedistribution ofcosine similaritiesbetween pairsof items
K nearest neighbors NNq  from a sample of 5 million test pins. We                  usingannotation,visual,andPinSageembeddings.Thisdistribution
then define the hit-rate as the fraction of queriesq where i was                   of cosine similarity between random pairs of items demonstrates
ranked among the topK of the test sample (i.e., wherei∈NNq ).                      theeffectivenessofPinSage,whichhasthemostspreadoutdistribu-
Thismetricdirectlymeasurestheprobabilitythatrecommendations                        tion.Inparticular,thekurtosisofthecosinesimilaritiesofPinSage
madebythealgorithmcontaintheitemsrelatedtothequerypinq.                            embeddingsis 0.43,comparedto 2.49forannotationembeddings
In our experimentsK is set to be 500.                                              and 1.20for visual embeddings.
   We also evaluate the methods using Mean Reciprocal Rank                            Anotherimportantadvantageofhavingsuchawide-spreadin
(MRR), which takes into account of the rank of the item j among                    the embeddings is that it reduces the collision probability of the
recommended items for query itemq:                                                 subsequentLSHalgorithm,thusincreasingtheefficiencyofserving
                                  Õ           1                                    the nearest neighbor pins during recommendation.
                     MRR =   1n          Ri,q/100.                         (2)
                                (q  ,i)∈L                                          4.3   UserStudies
Due to the large pool of candidates (more than 2 billion), we use a                WealsoinvestigatetheeffectivenessofPinSagebyperforminghead-
scaledversionoftheMRRinEquation(2),whereRi,q  istherank                            to-headcomparisonbetweendifferentlearnedrepresentations.In
ofitemi amongrecommendeditemsforqueryq,andn isthetotal                             theuserstudy,auserispresentedwithanimageofthequerypin,
numberoflabeleditempairs.Thescalingfactor 100ensuresthat,                          togetherwithtwopinsretrievedbytwodifferentrecommendation
forexample,thedifferencebetweenrankat1,000andrankat2,000                           algorithms. The user is then asked to choose which of the two
is still noticeable, instead of being very close to 0.                             candidatepinsismorerelatedtothequerypin.Usersareinstructed
   Table 1 compares the performance of the various approaches                      to find various correlations between the recommended items and
using the hit rate as well as the MRR.5 PinSage with our new                       thequeryitem,inaspectssuchasvisualappearance,objectcategory
importance-poolingaggregationandhardnegativeexamplesachieves                       and personal identity. If both recommended items seem equally
the best performance at 67% hit-rate and 0.59 MRR, outperforming                   related,usershavetheoptiontochoose“equal”.Ifnoconsensusis
the top baseline by 40% absolute (150% relative) in terms of the                   reachedamong 2/3ofuserswhoratethesamequestion,wedeem
hitrateandalso22%absolute(60%relative)intermsofMRR.We                              the result as inconclusive.
also observe that combining visual and textual information works                      Table 2 shows the results of the head-to-head comparison be-
muchbetterthanusingeitheronealone(60%improvementofthe                              tweenPinSageandthe4baselines.Amongitemsforwhichtheuser
combined approach over visual/annotation only).                                    has an opinion of which is more related, around 60% of the pre-
Embedding similarity distribution. Another indication of the                       ferreditemsarerecommendedbyPinSage.Figure5givesexamples
effectiveness of the learned embeddings is that the distances be-                  ofrecommendationsandillustratesstrengthsandweaknessesofthe
tweenrandom pairsof itemembeddingsare widelydistributed. If                        differentmethods.Theimagetotheleftrepresentsthequeryitem.
allitemsareataboutthesamedistance(i.e.,thedistancesaretightly                      Each row to the right corresponds to the top recommendations
clustered)thentheembeddingspacedoesnothaveenough“resolu-                           made by the visual embedding baseline, annotation embedding
tion” to distinguish between items of different relevance. Figure 4                baseline, Pixie, and PinSage. Although visual embeddings gener-
5NotethatwedonotincludethePixiebaselineintheseofflinecomparisonsbecausethe         ally predict categoriesand visual similarity well, theyoccasionally
Pixiealgorithmrunsin productionand is“generating” labeledpairs(q  , j)forus—  i.e.,make large mistakes in terms of image semantics. In this example,
thelabeledpairsareobtainedfromhistoricaluserengagementdatainwhichthePixie          visualinformationconfusedplantswithfood,andtreeloggingwith
algorithm was used as the recommender system. Therefore, the recommended item j    warphotos,duetosimilarimagestyleandappearance.Thegraph-
isalwaysintherecommendationsmadebythePixiealgorithm.However,wecompare
to the Pixie algorithm using human evaluations in Section 4.3.                     basedPixiemethod,whichusesthegraphofpin-to-boardrelations,

         Methods              Win       Lose     Draw     Fraction of wins
   PinSage vs. Visual         28.4%    21.9%     49.7%          56.5%
   PinSage vs. Annot.         36.9%    14.0%     49.1%          72.5%
 PinSage vs. Combined         22.6%    15.1%     57.5%          60.0%
    PinSage vs. Pixie         32.5%    19.6%     46.4%          62.4%
Table 2: Head-to-head comparison of which image is more
relevanttotherecommendedqueryimage.
correctly understandsthat the categoryof query is “plants”and it
recommends items in that general category. However, it does not
find the most relevant items. Combining both visual/textual and
graphinformation,PinSageisabletofindrelevantitemsthatare
both visually and topically similar to the query item.
   In addition, we visualize the embedding space by randomly
choosing 1000 items and compute the 2D t-SNE coordinates from
thePinSageembedding,asshowninFigure6.6 Weobservethatthe
proximity of the item embeddings corresponds well with the simi-
larityofcontent,andthatitemsofthesamecategoryareembedded
into the same part of the space. Note that items that are visually
different but have the same theme are also close to each other
in the embedding space, as seen by the items depicting different
fashion-related items on the bottom side of the plot.
4.4   ProductionA/BTest
Lastly, we also report on the production A/B test experiments,
whichcomparedtheperformanceofPinSagetootherdeeplearning
content-based recommender systems at Pinterest on the task of
homefeedrecommendations.Weevaluatetheperformancebyob-
serving thelift in userengagement. Themetric of interestis repin
rate,whichmeasuresthepercentageofhomefeedrecommendations                          Figure5:ExamplesofPinterestpinsrecommendedbydiffer-
thathave been savedbytheusers. Ausersavingapin toaboard                           ent algorithms. The image to the left is the query pin. Rec-
is a high-value action that signifies deep engagement of the user.                ommendeditemstotherightarecomputedusingVisualem-
It means that a given pin presented to a user at a given time was                 beddings, Annotation embeddings, graph-based Pixie, and
relevant enoughfor the user to savethat pin to one oftheir boards                 PinSage.
so that they can retrieve it later.
   We find that PinSage consistently recommends pins that are                     thetrainingsetsizedidnotseem tohelp),reducingtheruntimeby
morelikelytobere-pinnedbytheuserthanthealternativemethods.                        a factor of 6 compared to training on the full graph.
Depending on the particular setting, we observe 10-30% improve-                      Table3showsthetheeffectofbatchsizeoftheminibatchSGD
ments in repin rate over the Annotation and Visual embedding                      on the runtime of PinSage training procedure, using the mean-
based recommendations.                                                            pooling-hardvariant.Forvaryingbatchsizes,thetableshows:(1)
4.5   TrainingandInferenceRuntimeAnalysis                                         the computation time, in milliseconds, for each minibatch, when
                                                                                  varyingbatchsize;(2)thenumberofiterationsneededforthemodel
One advantage of GCNs is that they can be made inductive [19]:                    to converge; and (3) the total estimated time for the training proce-
at the inference (i.e., embedding generation) step, we are able to                dure. Experiments show that a batch size of 2048 makes training
compute embeddings for items that were not in the training set.                   most efficient.
This allows us to train on a subgraph to obtain model parameters,                    When training the PinSage variant with importance pooling,
and thenmake embed nodesthat havenot been observedduring                          another trade-off comes from choosing the size of neighborhood
training.Alsonotethatitiseasytocomputeembeddingsofnew                             T. Table 3 shows the runtime and performance of PinSage when
nodes that get added into the graph over time. This means that                    T =  10, 20and 50.WeobserveadiminishingreturnasT increases,
recommendations can be made on the full (and constantly grow-                     and findthata two-layerGCN withneighborhood size50 can best
ing) graph. Experiments on development data demonstrated that                     capture the neighborhood information of nodes, whilestill being
training ona subgraphcontaining 300 million itemscould achieve                    computationally efficient.
thebestperformanceintermsofhit-rate(i.e.,furtherincreasesin                          After training completes, due to the highly efficient MapReduce
                                                                                  inferencepipeline, thewhole inferenceprocedureto generateem-
6Some items are overlapped and are not visible.                                   beddings for 3 billion items can finish in less than 24 hours.

                                                                                                     ZitaoLiufor providing datausedbyPixie[14],andVitaliy Kulikov
                                                                                                     for help in nearest neighbor query of the item embeddings.
                                                                                                     REFERENCES
                                                                                                      [1]  M. Abadi, A. Agarwal, P. Barham, E. Brevdo, Z. Chen, C. Citro, G. S. Corrado, A.
                                                                                                          Davis, J.Dean, M. Devin,et al.2016. Tensorflow: Large-scale machinelearning
                                                                                                          on heterogeneous distributed systems. arXiv preprint arXiv:1603.04467 (2016).
                                                                                                      [2]  A.AndoniandP.Indyk.2006. Near-optimalhashingalgorithmsforapproximate
                                                                                                          nearest neighbor in high dimensions. In FOCS.
                                                                                                      [3]  T.Bansal,D.Belanger,andA.McCallum.2016. AsktheGRU:Multi-tasklearning
                                                                                                          for deep text recommendations. In RecSys. ACM.
                                                                                                      [4]  Y. Bengio, J.Louradour, R. Collobert,and J. Weston. 2009. Curriculum learning.
                                                                                                          In ICML.
                                                                                                      [5]  A. Z. Broder, D. Carmel, M. Herscovici, A. Soffer, and J. Zien. 2003.  Efficient
                                                                                                          query evaluation using a two-level retrieval process. In CIKM.
                                                                                                      [6]  M. M. Bronstein, J. Bruna, Y. LeCun, A. Szlam, and P. Vandergheynst. 2017.
                                                                                                          Geometricdeep learning:Going beyondeuclidean data. IEEE Signal Processing
                                                                                                          Magazine 34, 4 (2017).
                                                                                                      [7]  J. Bruna, W. Zaremba, A. Szlam, and Y. LeCun. 2014.  Spectral networks and
                                                                                                          locally connected networks on graphs. In ICLR.
                                                                                                      [8]  J. Chen, T. Ma, and C. Xiao. 2018. FastGCN: Fast Learning with Graph Convolu-
                                                                                                          tional Networks via Importance Sampling. ICLR (2018).
                                                                                                      [9]  P. Covington,J. Adams, andE. Sargin.2016. Deep neuralnetworks for youtube
  Figure6:t-SNEplotofitemembeddingsin2dimensions.                                                         recommendations. In RecSys. ACM.
                                                                                                     [10]  H. Dai, B. Dai, and L. Song. 2016. Discriminative Embeddings of Latent Variable
                                                                                                          Models for Structured Data. In ICML.
   Batch size        Per iteration (ms)          # iterations        Total time (h)                  [11]  M. Defferrard, X. Bresson, and P. Vandergheynst. 2016. Convolutional neural
        512                    590                    390k                  63.9                          networks on graphs with fast localized spectral filtering. InNIPS.
       1024                    870                    220k                  53.2                     [12]  A. Van den Oord, S. Dieleman, and B. Schrauwen. 2013.  Deep content-based
                                                                                                          music recommendation. In NIPS.
       2048                   1350                    130k                  48.8                     [13]  D.Duvenaud,D.Maclaurin,J.Iparraguirre,R.Bombarell,T.Hirzel,A.Aspuru-
       4096                   2240                    100k                  68.4                          Guzik, and R. P. Adams. 2015. Convolutional networks on graphs for learning
                                                                                                          molecular fingerprints. InNIPS.
   Table3:Runtimecomparisonsfordifferentbatchsizes.                                                  [14]  C. Eksombatchai, P. Jindal,J. Z. Liu, Y. Liu,R. Sharma, C. Sugnet, M. Ulrich, and
                                                                                                          J. Leskovec. 2018. Pixie: A System for Recommending 3+ Billion Items to 200+
           # neighbors         Hit-rate       MRR       Training time (h)                                 Million Users in Real-Time. WWW (2018).
                                                                                                     [15]  M.Gori,G.Monfardini,andF.Scarselli.2005. Anewmodelforlearningingraph
                 10               60%         0.51                 20                                     domains. In IEEE International Joint Conference on Neural Networks.
                 20               63%         0.54                 33                                [16]  P.Goyal,P.Dollár,R.Girshick,P.Noordhuis,L.Wesolowski,A.Kyrola,A.Tulloch,
                                                                                                          Y.Jia, and K.He. 2017. Accurate, LargeMinibatch SGD:Training ImageNet in1
                 50               67%         0.59                 78                                     Hour. arXiv preprint arXiv:1706.02677 (2017).
   Table4:Performancetradeoffsforimportancepooling.                                                  [17]  A.GroverandJ.Leskovec.2016. node2vec:Scalablefeaturelearningfornetworks.
                                                                                                          In KDD.
                                                                                                     [18]  W.L.Hamilton,R.Ying,andJ.Leskovec.2017. InductiveRepresentationLearning
5   CONCLUSION                                                                                            on Large Graphs. In NIPS.
                                                                                                     [19]  W. L. Hamilton, R. Ying, and J. Leskovec. 2017.  Representation Learning on
WeproposedPinSage,arandom-walkgraphconvolutionalnetwork                                                   Graphs: Methods and Applications. IEEE Data Engineering Bulletin (2017).
(GCN).PinSageisahighly-scalableGCNalgorithmcapableoflearn-                                           [20]  S. Kearnes, K. McCloskey, M. Berndl, V. Pande, and P. Riley. 2016.  Molecular
                                                                                                          graph convolutions: moving beyond fingerprints. CAMD 30, 8.
ing embeddings for nodes in web-scale graphscontaining billions                                      [21]  T. N. Kipf and M. Welling. 2017.   Semi-supervised classification with graph
ofobjects.Inadditiontonewtechniquesthatensurescalability,we                                               convolutional networks. In ICLR.
introducedtheuseofimportancepoolingandcurriculumtraining                                             [22]  Y. Li, D. Tarlow, M. Brockschmidt, and R. Zemel. 2015. Gated graph sequence
                                                                                                          neural networks. In ICLR.
that drastically improved embedding performance. We deployed                                         [23]  T. Mikolov, I Sutskever, K. Chen, G. S. Corrado, and J. Dean. 2013. Distributed
PinSage at Pinterest and comprehensively evaluated the quality                                            representations of words and phrases and their compositionality. In NIPS.
                                                                                                     [24]  F. Monti, M. M. Bronstein, and X. Bresson. 2017. Geometric matrix completion
of the learned embeddings on a number of recommendation tasks,                                            with recurrent multi-graph neural networks. In NIPS.
with offline metrics, user studies and A/B tests all demonstrating                                   [25]  OpenMPArchitectureReviewBoard.2015. OpenMPApplicationProgramInter-
asubstantialimprovementinrecommendationperformance.Our                                                    face Version 4.5. (2015).
                                                                                                     [26]  B. Perozzi, R. Al-Rfou, and S. Skiena. 2014. DeepWalk: Online learning of social
workdemonstratestheimpactthatgraphconvolutionalmethods                                                    representations. In KDD.
can have in a production recommender system, and we believe                                          [27]  F. Scarselli, M. Gori, A.C. Tsoi, M. Hagenbuchner, and G. Monfardini. 2009. The
                                                                                                          graph neural network model. IEEE Transactions on Neural Networks 20, 1 (2009),
thatPinSage canbe furtherextendedin thefuture totackleother                                               61–80.
graph representation learning problems at large scale, including                                     [28]  K. Simonyan and A. Zisserman. 2014.  Very deep convolutional networks for
knowledge graph reasoning and graph clustering.                                                           large-scale image recognition. arXiv preprint arXiv:1409.1556 (2014).
                                                                                                     [29]  R.van denBerg, T. N.Kipf, andM.Welling.2017. Graph ConvolutionalMatrix
                                                                                                          Completion. arXiv preprint arXiv:1706.02263 (2017).
Acknowledgments                                                                                      [30]  J. You, R. Ying, X. Ren, W. L. Hamilton, and J. Leskovec. 2018.   GraphRNN:
TheauthorsacknowledgeRaymondHsu,AndreiCureleaandAli                                                       Generating Realistic Graphs using Deep Auto-regressive Models. ICML (2018).
                                                                                                     [31]  M. Zitnik, M. Agrawal, and J. Leskovec. 2018.  Modeling polypharmacy side
AltafforperformingvariousA/Btestsinproductionsystem,Jerry                                            effects with graph convolutional networks. Bioinformatics (2018).

