  PopularityPredictiononSocialPlatformswithCoupledGraph
                                                           NeuralNetworks
                         Qi Cao1,2, Huawei Shen1,2, Jinhua Gao1, Bingzheng Wei3, Xueqi Cheng1
                                   {caoqi,shenhuawei,gaojinhua,cxq}@ict.ac.cn,coltonwei@tencent.com
               1 CAS Key Laboratory of Network Data Science and Technology, Institute of Computing Technology,
                                                   Chinese Academy of Sciences, Beijing, China
                                         2 University of Chinese Academy of Sciences, Beijing, China
                                                                 3 WeChat, Tencent Inc
ABSTRACT
Predicting the popularity of online content on social platforms is
animportanttaskforbothresearchersandpractitioners.Previous
methodsmainlyleveragedemographics,temporalandstructural
patternsofearlyadoptersforpopularityprediction.However,most
existingmethodsarelesseffectivetopreciselycapturethecascad-
ing effectininformationdiffusion,inwhichearlyadopterstryto
activatepotentialusersalongtheunderlyingnetwork.Inthispaper,
we consider the problem of network-aware popularity prediction,
leveraging both early adopters and social networks for popularity
prediction. We propose to capture the cascading effect explicitly,
modeling the activation state of a target usergiven the activation
state and influenceof his/her neighbors. Toachievethis goal, we
proposeanovelmethod,namelyCoupledGNN,whichusestwocou-
pledgraphneuralnetworkstocapturetheinterplaybetweennode                             Figure 1: Example of cascading effect in information diffu-
activation states and the spread of influence. By stacking graph                    sion.Thelightorangecirclerepresentsthesubgraphofearly
neural network layers, our proposed method naturally captures                       adopters. The green, gray, and blue circles represent the
thecascading effectalong thenetwork ina successivemanner.Ex-                        neighborhoods of early adopters that are reachable within
periments conducted on both synthetic and real-world SinaWeibo                      one-hop,two-hops,andthree-hopsrespectively.
datasets demonstrate that our method significantly outperforms
the state-of-the-art methods for popularity prediction.
CCSCONCEPTS                                                                         1   INTRODUCTION
                                                                                   Withtheboomingofsocialmediaplatforms,e.g.,Twitter,Facebook,
• Human-centeredcomputing →    Social networks; Social media.                       Sina Weibo, Instagram, etc, the production and dissemination of
                                                                                    user-generated online content, which we refer to as a piece of
KEYWORDS                                                                            information, becomes extremely convenient and common in our
Popularity Prediction, Graph Neural Networks, Cascading Effect,                     life. Every day, there are tens of millions of information generated
Network-aware                                                                       on these platforms [1]. With such a vast amount of information,
                                                                                    predicting the popularity of pieces of information is valuable for
ACMReferenceFormat:                                                                 us to discover the hot information in advance and to help people
QiCao,HuaweiShen,JinhuaGao,BingzhengWei,XueqiCheng.2020.Popu-                       out of the dilemma of information explosion. However, due to the
larityPredictiononSocialPlatformswithCoupledGraphNeuralNetworks.                    opennessofsocialplatformsandthecascadingeffectofinformation
In The Thirteenth ACM International Conference on Web Search and Data               diffusion, it’s very challenging to accurately predict the popularity
Mining (WSDM ’20), February 3–7, 2020, Houston, TX, USA.  ACM,NewYork,              of online content.
NY, USA, 9 pages. https://doi.org/10.1145/3336191.3371834                              Inthepastdecade,aseriesofeffortshavebeendevotedtothepop-
                                                                                    ularitypredictionprobleminsocialnetworks,consideringthisprob-
                                                                                    lemeither asaregression [2, 16, 46]or classificationtask[17, 33].
Permission to make digital or hard copies of all or part of this work for personal orGenerally speaking, popularity prediction aims to predict future
classroom use isgranted without fee providedthat copies are notmade or distributed
forprofitorcommercialadvantageandthatcopiesbearthisnoticeandthefullcitation         popularity when observing early adopters at a specific observa-
onthefirstpage.CopyrightsforcomponentsofthisworkownedbyothersthanACM                tion time (see the light orange circle in Figure 1 as an example).
mustbehonored.Abstractingwithcreditispermitted.Tocopyotherwise,orrepublish,         Various hand-crafted features of early adopters are extracted to
topostonserversortoredistributetolists,requirespriorspecificpermissionand/ora
fee.Requestpermissions frompermissions@acm.org.                                     predict the future popularity, e.g., demographic features in user
WSDM ’20, February 3–7, 2020, Houston, TX, USA                                      profile, useractivity [6],user degree [14],density of thesubgraph
©2020Associationfor Computing Machinery.                                            ofearlyadopters[10,45],aswellassubstructure[30]andcommu-
ACMISBN978-1-4503-6822-3/20/02...$15.00
https://doi.org/10.1145/3336191.3371834                                             nity[4, 33].With thesuccess ofrepresentation learning methods,

end-to-enddeeprepresentationlearningmethodsarealsoproposed                          tweet/microblogs [6, 18], images [44], videos[23], recipes [24], and
toautomaticallylearntherepresentationofthesubgraphofearly                           academic papers [27].
adopters [16]. In addition, to further improve the prediction per-                     Generallyspeaking,existingmethodsforpopularityprediction
formance, temporal information [21, 27, 32] of the early adopters                   mainlyfocusonfourtypesofinformation,i.e.,content,temporal
andthecontentinformation[17,44]arefurtherutilized.Themeth-                          information, early adopters and network structure. For content
odsmentioned sofarmainly focuson thecharacteristicsof early                         information,hierarchicalattentionnetworks[17]oruser-guidedhi-
adoptersorthesubgraphofearlyadopters,ignoringthecascading                           erarchical attentionmechanisms [44] areproposed to characterize
effect(showninFigure1)ininformationdiffusionswhichisoneof                           the content features. For temporal information, heuristical tem-
thekeysto accuratelypredictfuturepopularityof onlinecontent                         poral features[21], timeseries models includingrecurrent neural
in social platforms.                                                                network [34] and temporal convolutional network [26], or point
   To further characterize the cascading effect, researchers have                   process method including reinforced Poisson processes [27] and
alsomadesomeattempts.Theyadoptsomestatistics,suchasthe                              Hawkes process[19, 23, 38, 46], areproposed to devoteto capture
average number of fans of users, to approximate the impact of                       the underlying laws or patterns behind the temporal information.
cascadingeffectineachgeneration[19,40,46].However,sincethey                            As for early adopters and network structure, which is also the
onlyadoptsimplestatisticsandregardlessoftheexplicitnetwork                          focusofthis paper,bothfeature-basedmethodsandrepresentation
structuregoverningthecascadingeffect,theyarelesseffectivefor                        learning methods are proposed. The designed effective features
popularity prediction.                                                              in the former one including node degree [14, 46], the number of
   In this paper, we focus on the network-aware popularity predic-                  nodes in the frontier graph [11], cascade density [10, 45], substruc-
tionproblem,leveragingbothearlyadoptersandnetworkstructure                          tures [30], community [4, 33] and so on. Unfortunately, the per-
forpredicting thepopularity ofonline contentonsocial platforms.                     formance of such methods heavily depends on the quality of the
To effectively capture the crucial cascading effect, we devote to                   hand-craftedfeatures,whicharegenerallyextractedheuristically.
applying graph neural networks to successively characterize the                     Toavoidtheaboveheuristicfeatureextractionprocess,attemptsof
activationstateofeachuser.Specifically,theactivationofatarget                       end-to-end deeprepresentationlearning fashion are proposedto
userisintrinsicallygovernedbytwokeycomponents,i.e.,thestate                         automaticallylearntheimpactrepresentationofearlyadoptersby
ofneighborsandthespreadofinfluence,alongsocialnetworks.As                           cascading effect [1, 16, 32].
a result,we proposeto model theiterative interplaybetween node                         However,theabovemethodsarelesseffectivetocapturethecas-
states and the spread of influence by two coupled graph neural                      cadingeffectininformationdiffusion,sincetheyneglecttheexplicit
networks. One graph neural network models the spread of inter-                      interactions between users along the underlying social network.
personalinfluence,gatedbytheactivationstateofusers.Theother                         In contrast, the method proposed in this paper effectively capture
graph neural network models the update of the activation state of                   suchcascadingeffectalongthenetworkstructurebycoupledgraph
each user via interpersonal influence from their neighbors. With                    neural networks.
the iterative aggregation mechanism of the neighborhood in graph
neuralnetworks,thecascadingeffectalongthenetworkstructureis                         2.2   DiffusionModels
naturally characterized. Note that, other information like temporal                 Modeling how information diffuse is of outstanding interest over
or content, if available, can be further included in the prediction                 the past decades. There are two classic diffusion models in this
model by representation fusion flexibly.                                            category, i.e., Independent Cascades (IC) model [7] and Linear
   We verify the effectiveness of our proposed coupled graph neu-                   Threshold (LT) model [9]. The diffusion process of these models
ral networks on both the synthetic data and real-world data in                      is both iteratively carried on a synchronous way along discrete
SinaWeibo.Experimentalresultsdemonstratethatourproposed                             timestepsfrominitialadopters.Thesynchronicityassumptionis
methodsignificantlyoutperformsallthestate-of-the-artmethods.                        furtherrelaxedbyproposingasynchronouscontinuous-timeexten-
For conveience of the reproduction of the results, we have made                     sions [8, 10]. Such diffusion models can well capture the cascading
the source code publicly available1.                                                effectalongnetworkbyiterativelymodelingthespecificactivation
2   RELATEDWORK                                                                     process.However,theygenerallyneedanextremelyhighnumber
In this section, we briefly review the research on the popularity                   of Monte-Carlo simulations to estimate the final influence spread,
prediction, traditional diffusion models, the development and ap-                   i.e., the popularity to be predicted. Such a prediction process is
plication of graph neural networks.                                                 time-consuming and limits its applicability to real scenarios.
                                                                                       Thedifference betweenour proposedmethod andworksof this
2.1   PopularityPrediction                                                          line is that we do not model the specific diffusion process, but
                                                                                    utilize the graph neural networks to directly model the influence
Popularity prediction aims to predict the future popularity of on-                  of cascading effect by neighborhood aggregation, which is more
linecontentwhenobservingearlyadopterswithintheobservation                           efficient and flexible.
time. Due to the openness of social platforms and the cascade
phenomenon of online content, future popularity results in huge                     2.3   GraphNeuralNetworks
variance and is challenging to predict. The predictability of par-                  InspiredbythehugesuccessofneuralnetworksinEuclideanspace,
ticulartypesofinformationhasbeenprovedtosomeextent,e.g.,                            recentlytherehasbeenasurgeofinterestingraphneuralnetwork
1https://github.com/CaoQi92/CoupledGNN.                                             approaches for representation learning of graphs [12, 31, 36, 37].

Figure 2: The framework of coupled graph neural networks for popularity prediction. s∗and r∗are the activation state and
influencerepresentationofuser∗respectively.
Graphneuralnetworks(GNNs)broadlyfollowarecursiveneighbor-                           between users, thisproblem aims topredict the finalpopularity of
hood aggregation fashion, where each node updates its representa-                   informationm, i.e.,nm∞.
tionbyaggregatingtherepresentationofitsneighborhood.AfterK                             Notethat,thenetwork-awarepopularitypredictionproblemem-
iterations of aggregation, the updated representation of each node                  phasizestheroleofthenetwork,i.e.,thereareinteractionsbetween
capturesboththestructuralandrepresentationinformationwithin                         early adoptersand potential active users,or amongthe potential
the node’sK-hop neighborhood [37].                                                  active users. It’s precisely because of this characteristic, making
   GNNs have been successfully applied to a lot of non-Euclidean                    thecaptureofthecascadingeffectalongthenetworkbecomesthe
domain problems, e.g., semi-supervised learning on graph [13, 35],                  key to accurately predict the future popularity of online content.
socialinfluenceprediction[22],correlatedtemporalsequencemod-
eling [25]. Among the above, the application to social influence                    3.2   GeneralFrameworkofGNNs
prediction,i.e.,DeepInf[22],isthemostrelatedonewithourwork.                         GNNs is an effective framework for representation learning of
However,sinceDeepInfmorefocusesonthepredictionofthemicro                            graphs.AsintroducedinSection2.3,manyvariantsofgraphneural
action status of a user on a fixed-sized local network, rather than                 networks have been proposed. They usually follow a neighbor-
the macro popularity prediction on the global diffusion network                     hood aggregation strategy, where the representation of a node is
studied in this paper, it performs not well for future popularity.                  updated by recursively aggregating the representation of its neigh-
   In this paper, we devote to utilizing GNNs to characterize the                   boring nodes. Formally, thek−th layerof agraph neuralnetwork
cascading effect in popularity prediction. To better adapt to the                   is generally formulated as in [37]:
scenarioofinformationdiffusion,wedesignanovelmodelcoupled
graph neuralnetworksto solvethe popularityprediction problem.                                    a(k  )v   =  AGGREGATE      h(k  )u    :u∈N(v)    ,              (1)
3   PRELIMINARIES
                                                                                                       h(k +1    )v      =  COMBINEh(k  )v   ,a(k  )v,                      (2)
Thissectiongivestheformaldefinitionofthepopularityprediction
problem studied in this paper and the general framework of GNNs.                    whereh(k  )v   is the feature vector of nodev at thek-th layer,N(v)
                                                                                    is the set of nodes which appear in the neighborhood of nodev.
3.1   ProblemDefinition                                                             The choice of the function AGGREGATE(∗) and COMBINE(∗) in
Supposing that we haveM pieces of information, the observed cas-                    GNNs is crucial.
cadeofinformationmisrecordedasthesetofearlyadopterswithin                              TherepresentationoftheentiregraphhGisobtainedbyaREAD-
the observation time windowT, i.e.,CmT   ={u1,u2,..., un mT}, where                 OUT function:
nmT   is the total number of adopted or active users of informationm                                hG=  READOUT            h(k +1    )v       :v∈G.                  (3)
withinthe observationtimewindowT.For example,theobserved
cascade inFigure 1is recorded asCmT   ={A,B}. Inaddition tothe                         READOUT(∗) canbeasimple functionsuchassummationora
observed cascades, given the underlying network which governing                     more sophisticated graph-level pooling function [39, 43].
the information diffusion, e.g., the following relationships in Sina                4   METHODS
Weibo,wecanformalizethepopularitypredictionproblemstudied
in this paper as:                                                                   In this section, we introduce the proposed coupled graph neural
   Network-awarePopularityPrediction. Given the observed                            network(CoupledGNN)fornetwork-awarepopularityprediction.
cascadesCmT   and the underlying networkG=    (V,E), whereV                         Wedesignthe CoupledGNNmodel tocapture the cascading effect
is the set of all users,E⊆V×Vis the set of all relationships                        widely observed in information diffusion over social networks.

4.1   FrameworkofCoupledGNN
Thecascadingeffectindicatesthattheactivationofoneuserwill
trigger its neighbors in a successive manner, forming an informa-
tioncascadeoversocialnetworks.Foratargetuser,whetherhe/she
couldbeactivatedisintrinsicallygovernedbytwokeycomponents,
i.e., the state of neighbors and the spread of influence, direct or in-
direct, over social networks. In this sense, the cascading effect is
intrinsically the iterative interplay between node states and the
spread of influence. Previous methods, e.g., independent cascade
model, assumes a fixed-yet-unknown interpersonal influence and                                Figure3:Mechanismsofstategraphneuralnetwork.
probes the interplay manifested as the cascading effect over the
social network via Monte-Carlo simulation.
   In thispaper, wepropose to usetwo coupled graphneural net-                                Then the COMBINE(∗) function used to update the activation
workstonaturallycapturethecascadingeffect,ormorespecifically,                             state of userv is defined as theweighted sum of the neighborhood
the interplay of node states and the spread of influence. One graph                       aggregation and the activation state of userv itself:
neural network, namely state graph neural network, is used to                                                      1,                 v∈CmT
model the activation state of nodes. The other graph neural net-                                    s(k +1    )v      =                                   ,          (6)
work,namelyinfluencegraphneuralnetwork,isusedtomodelthe                                                            σ   µ(k  )s   s(k  )v   + µ(k  )a   a(k  )v, v <CmT
spreadofinfluenceoversocialnetworks.Thetwographneuralnet-
worksarecoupledthroughtwogatingmechanisms.Theframework                                    whereµ(k  )s    ,µ(k  )a ∈R  are weight parameters,σ is a nonlinear ac-
of our coupled graph neural networks is shown in Figure 2.                                tivation function. The initial activation state of userv is defined
                                                                                          as                                (  1, v∈CmT
4.2   StateGraphNeuralNetwork                                                                                       s(0)v   =
The stategraph neural networkis to modelthe activation ofeach                                                                  0, v <CmT      .                              (7)
user during the cascading effect. Specifically, for a target user
v  <CmT  , since he/she is usually influenced by the active users                         4.3   InfluenceGraphNeuralNetwork
in the neighborhoodN(v), we apply a graph neural network to                               The influence graph neural network is to model the diffusion of
modeltheactivationstateofeachuser(showninFigure3).Each                                    interpersonal influence in the social network. Specifically, each
userisassociatedwithaone-dimensionalvaluesv,indicatingthe                                 userv is associated with an influence representationrv. Then the
activationstateofuserv.Besides,sincetheinterpersonalinfluence                             influence representation of active users further diffuses to other
between the pair of users is generally various, we model such het-                        usersalongwithnetworkstructure,implementedbyneighborhood
erogeneous influence weight by an influence gating mechanism,                             aggregationofgraphneuralnetworksandastategatingmechanism.
i.e.,                                                                                     The entire mechanism ofinfluence graph neuralnetwork isshown
        InfluGate     r (k  )u     ,r (k  )v=  β   (k  )[W (k  )r (k  )u ∥W (k  )r (k  )v  ],      (4)inFigure4.Specifically,theneighborhoodaggregationisdefined
                                                                                          as:                     X
wherer (k  )u ∈R h   (k  ) is theinfluence representation ofuseru at the                              b(k  )v   =        StateGate     s(k  )uα(k  )u vW (k  )r (k  )u     ,            (8)
k-th layer,W (k  )∈R h   (k +1    )×h   (k  ) is a weight matrix to transform                                  u∈N(v)
the influence representation from dimensionh(k  ) toh(k +1    ),β   (k  )∈                wherer (k  )
R 2h   (k +1    ) isaweightvector.Notethat,Equation4isjustoneinstance                              u ∈R h   (k  ) istheinfluencerepresentationofuseruat (k)-th
of the InfluGate(∗) function. We can also choose other types of                           layer,W(k  )∈R h   (k +1    )×h   (k  ) isaweightmatrixtotransformtheinflu-
functions which can reflect the influence gating between the pair                         ence representation from dimensionh(k  ) toh(k +1    ). StateGate(∗) is
of users.                                                                                 the state gating mechanism, implemented by a 3-layer MLP in this
   Afterobtainingtheheterogeneousinfluenceweightbyinfluence                               paper to reflect the nonlinear effect of state.α(k  )u v  is the attention
gating function, the aggregation of the expected influence that the                       weightfromuseru touserv,whereweadopttheformulationused
target userv receives from his/her neighborhood is:                                       in [31], i.e.,
                       X                                                                                     e(k  )
            a(k  )v   =       InfluGate     r (k  )u     ,r (k  )vs(k  )u    +  pv,            (5)            u v =  γ   (k  )[W (k  )r (k  )u ∥W (k  )r (k  )v  ],                     (9)
                     u∈N(v)                                                                                                                       u v )
whereN(v) istheneighborhoodofuserv,InfluGate (∗)s(k  )u    isthe                                       α(k  )u v =  softmax(e(k  )u v ) =          exp(e(k  )P,           (10)
expectedinfluenceconsideringtheactivationstates(k  )                                                                                    z∈N(v) exp(e(k  )zv  )
                                                               u    ofneighbor
u,pv∈R  is a self activation parameter to reflect the probability                         whereγ   (k  )∈R 2h   (k  ) is a weight vector.
thatuserv maybeactivatedbywaysoutoffollowingrelationships,                                   Thentheinfluencerepresentationofuserv at (k +  1)-thlayeris
e.g., offline communication or browsing the hot list in the front                         updated by
pages. Note that, Equation 5 is actually a specific design of the
AGGREGATE function as mentioned in Section 3.2.                                                           r (k +1    )v       = σζ(k  )r   W (k  )r (k  )v   +ζ(k  )b   b(k  )v,                (11)

                                                                                     theabovecomputationalcomplexityisbasedonthecomputationof
                                                                                     the whole network. To make it more efficient, we can also address
                                                                                     several mini-batch with R samples, which makes the algorithm
                                                                                     independent of the graph size and achieveO(R) complexity [25].
                                                                                     5   EXPERIMENTALSETUP
                                                                                    WecompareourCoupledGNNwithseveralstate-of-the-artmethods
                                                                                     ondifferentdatasetsundervariousevaluationmetrics.Thedetailed
                                                                                     experimental settings are introduced in this section.
 Figure4:Mechanismsofinfluencegraphneuralnetwork.                                    5.1   DataSets
                                                                                     Tothoroughly evaluatethe performanceof ourmethods,wecon-
whereζ(k  )r     ,ζ(k  )b ∈R  are weight parameters,σ is a nonlinear ac-             duct experimentson both thesynthetic data set anda real-world
tivation function. The initial influence representationr (0)v  of user               data set from Sina Weibo.
v used in this paper consists of two parts: node embeddings and                      5.1.1   Synthetic Data Set. The synthetic network is constructed
node features. We will discuss each part in detail in the section of                 by Kronecker generator [15], which can generate networks that
implementation details.                                                              have common structural properties of real networks, i.e., heavy
4.4   OutputLayer                                                                    tailsforbothin-andout-degreedistributions,smalldiameters.The
                                                                                     parameter matrix is set to be [0.9;0.5;0.5;0.1] and we retain the
AfterK layers of graph neural networks for both activation state                     largestconnectedcomponentasthefinalnetwork,containing1,086
andinfluencerepresentation,theoutputactivationprobabilityof                          nodes and4,038 edges.
each user in the network is s(K   )v ∈[0,1], i.e., the output of the                    Asforinformationcascades,wefirstsampletheseedsetofeach
lastlayerinthestategraphneuralnetwork.Thepopularitytobe                              cascade.Thesizeoftheseedsetissampledaccordingtothepower-
predicted is then obtained by a sum pooling mechanism over all                       law distributionwith parameter2.5, i.e.,p(n)∝n−2    .5 [5],and the
users in the network, i.e.,          X                                               node in each seed set is uniformly sampled. With a given seed set,
                              ˆnm∞=       s(K   )                                    thecommonlyusedICmodel[7]isappliedtogeneratethediffusion
                                           u                                       (12)data,wheretheactivationprobabilityfromnodeu tonodev isset
                                    u∈V                                              tobe1/dv anddv isthe in-degreeof nodev.The observationtime
   Asforthelossfunctiontobeoptimized,weconsiderthemean                               windowT is set to be 2 time steps in this scenario, i.e., we observe
relativesquareerror(MRSE)loss[2,29],whichisrobusttooutliers                          the diffusion process at time stept =  0 andt =  1.
as well as smooth and differentiable:                                                   In total, there are 108,600 information cascades are generated
                                     MX    ˆnm∞−nm∞!2                                andthecascadeswithlessthan3activeusersarefilteredout.Finally,
                   LMRSE =   1M               nm∞       ,                     (13)   27,218 information cascades are taken as our data. We randomly
                                   m  =1                                             sample 80% of the data as our train set, 10% as the validation set
whereM is thetotal number piecesof information,nm∞is thetrue                         and 10% as the test set.
final popularity of informationm.                                                    5.1.2   Sina Weibo Data Set. For real-world data, let’s turn our at-
   To avoid over-fitting and accelerate the process of convergence,                  tentiontotheinformationcascadesonSinaWeibo,oneofthemost
we also add a L2 and user-level cross entropy to the objective
function as regularization:                                                          popular social platform in China. The Sina Weibo data set used
                                                                                     in this paper is from [41, 42] and publicly available online2. The
                          L =  LMRSE +  LReg,                               (14)     network in this data set is the following network, reflecting the
whereLReg = ηP       p∈P∥p∥2 +λLuser,Pisthesetofparameters,                          following relationships between users. Note that, such following
ηandλarehyper-parameters.Luser   istheuser-levelcrossentropy,                        network is quite related to the retweet information cascades since
                  P Mm  =1      1P                                                   the posted messages by user B will appear in user A’s feed when
i.e.,Luser =    1M        |V|v∈V        s∞v logs(K   )v    +    (1−s∞v  )logs(K   )v,userA followsuser BinSina Weibo.The followingnetwork con-
s∞v  is the true final activation state of each user.                                tains 1.78 million users and 308 million following relationships
                                                                                     in total.300 thousand popular microblog retweet information cas-
4.5   ComputationalComplexity                                                        cadesoftheseusersareincluded.Toanalysistheretweetsbehavior
For the state graph neural network, the computational complex-                       of a specific group of users with corresponding the information
ity including the influence gating mechanism at k-th layer, i.e.,                    cascades,weconstructasubsetofusersandmessagesontheuser-
O(|V|h(k−1    )h(k  )+|E|h(k  )),andtheupdationofactivationstate,i.e.,               microblog bipartite graph. Specifically, we start with a randomly
O(|V|+|E|).Fortheinfluencegraphneuralnetwork,thecomputa-                             chosen user and then obtain all the messages with coverage≥η.
tionalcomplexityatk-thlayerisO(|V|+|V|h(k−1    )h(k  )+|E|h(k  )).                   The coverage is defined as the number of users in the chosen set
Sum up, the computational complexity of coupled graph neural                         normalized by the number of total users in the message. The users
networkisO(p|V|+ q|E|),wherep,qaresmallconstantassociated                            appearedintheobtainedmessagesarethenaddedintothechosen
withthehiddendimensionh(k  ) ateachlayer.It’sworthnotingthat                         2https://www.aminer.cn/Influencelocality.

user set. We repeated the above steps and obtain23,732 users with                   andsparse,wesetalearningratealonefortheparametersofthese
corresponding 149,53 information cascades. The largest compo-                       featuresandchoosefromΦ 1 ={10−5  ,5×10−5  ,10−4  ,..., 0.01}.For
nent of the following network between these users is regarded as                    theparametersofotherfeatures,wechoosethelearningratefrom
the final network, containing 23,681 users and 1,802,146 edges.                     Φ 2 ={0.0005,0.001,0.005,0.01}. Similarly, for DeepCas, the learn-
Informationcascadeswithlessthan5activeusersarefilteredout                           ingrateforuserembeddingsarechosenfromΦ 1,whilethelearning
and the remaining 3,228 pieces of information are taken as our                      rateforotherparametersarechosenfromΦ 2.Theuserembeddings
data. As for the observation time window, we set the observation                    forDeepCasisinitializedbyDeepWalk[20]whichwillbefurther
time windowT =  1 hour, 2 hours and 3hours respectively.                            optimized during the training process, while the user embeddings
                                                                                    forourCoupledGNNarealsoobtainedbyDeepWalkbutwithout
5.2   Baselines                                                                     furtherfine-tuning.Thedimensionoftheembeddingsisallsetto
Since this paper focuses on the network-aware popularity predic-                    be 32. The hidden units of RNN in DeepCas is set to be 32, and
tion without temporal information, we mainly consider methods                       the units of the first dense layer and the second dense layer in the
that utilizeearly adoptersand networkstructure asour baselines.                     outputpartare32and16respectively.AsforSEISMIC,weadoptthe
Existing methods for this problem are mainly classified into two                    setting ofparameters usedin [46], i.e.,setting the constantperiod
categories:feature-basedmethodsanddeeprepresentationlearning                        s0 to5minutesandpower-lawdecayparametersθ=  0.242.Besides,
methods.Wechoosethe state-of-the-artmethodineach category                           wechoosemeandegreen∗from{1,3,5,10,20,50,100}tominimize
asourbaselines.Besides,wealsoincludetherepresentativeattempt                        themRSE ofvalidation set.For ourCoupledGNN model,similar to
of capturing the cascading effect in Popularity prediction.                         baselines,thelearningrateforselfactivationparametersofallusers
5.2.1    Feature-based. Weextractalltheeffectivehand-craftedfea-                    arechosen fromΦ 1,and thelearning ratefor otherparameters are
tures that can be easily generalized across data sets [3, 6, 16, 28].               chosenfromΦ 2.Thecoefficientλinthelossfunction,whichbal-
The extracted features are conducted on three types of graphs: the                  ancestheweightoftheregularizationofuser-levelcrossentropy,
global graphG, the cascade graphдc , and the frontier graphдf .                     is set to be 0.5 in our experiments. The number of GNN layersK is
Specifically, the cascade graphдc  contains all early adopters and                  chosenfrom{2,3,4},andeachlayercontainsthesamenumberof
thecorresponding edgesbetweenthese users.Thefrontier graph                          hidden units as input. Following [22], the vertex features for our
дf  containsallusersintheone-hopneighborhoodofearlyadopters                         CoupledGNN contains coreness, pagerank, hub score, authority
andtheedgesbetweentheseneighboringusers.Asforfeatures,we                            score, eigenvector centrality, and clustering coefficient.
extractthemeanand90thpercentileofthedegreesofusers[14],                             5.4   EvaluationMetrics
the numberof leaf nodes, edgedensity [10, 45] inдc ; the number                     We adoptseveraldifferent evaluationmetrics tocomprehensively
of substructures [28, 30], including nodes, edges and triangles, and                demonstrate the performance of each method.
the number of communities and the corresponding coverage of
the partition of these communities [4, 33] in bothдc  andдf . In                    5.4.1   MeanRelativeSquareError(MRSE)[2,29]. Wetakethemean
addition, since the node identity is quite important for popular-                   relativesquareerrorlossalsoasourevaluationmetricforpopularity
ity prediction [16], here we also include the global node ids inG                   prediction.
as thestructurefeature.Once the cascadeis represented asa bag
of features, we feed them into a linear regression model with L2                    5.4.2    Median Relative Square Error (mRSE). Since SEISMIC is sen-
regularization.                                                                     sitive to outlier error, we also use median RSE as an evaluation
5.2.2   DeepCas[16]. DeepCasisthestate-of-the-artdeeprepresen-                      metric, which is defined as the 50th percentile of the distribution
tation learningmethod fornetwork-aware popularityprediction,                        of RSE over test data.
whichlearnstherepresentationofcascadegraphsinanend-to-end                           5.4.3   MeanAbsolutePercentageError(MAPE)[27,34]. Thismetric
manner. Specifically, it represents the cascade graph as a collection               measures the average deviation between the predicted and true
ofsequencesbyrandomwalks,andthenutilizestheembeddingsof                             popularity. The formulation is
nodes and recurrent neural networks to obtain the representation                                                          MX
of each sequence. Attention mechanisms are further applied to                                             MAPE =   1           |ˆnm∞−nm∞|
assemble the representation of the cascade graph from sequences.                                                      M  m  =1     nm∞        .                        (15)
5.2.3   SEISMIC [46]. SEISMIC is a representative method for at-                    5.4.4   Wrong Percentage Error (WroPerc). The wrong prediction
temptsofcapturingthecascadingeffect.Itisanimplementationof                          error is defined as the percentage of online contents that are incor-
Hawkes self-exciting point process andestimates or approximates                     rectly predicted for a given error toleranceϵ:
the impact of cascading effect in each generation by the average
number of fans of users.                                                                                                MX
                                                                                                    WroPerc =   1M          I[|ˆnm∞−nm∞|nm∞ ≥ϵ].                (16)
5.3   ImplementationDetails.                                                                                          m  =1
ForallbaselinesandourCoupledGNNmodel,thehyper-parameters                            We set the thresholdϵ=  0.5 in this paper.
are tuned to obtain the best results on the validation set. The L2-                     Notethat, amongall thesethree evaluationmetrics, thesmaller
coefficient is chosen from{10−8  ,10−7  ,..., 0.01,0.1}. For feature-               the value is, indicating the better the performance of the corre-
basedmethod,sincethe featuresof nodeids are high-dimensional                        sponding method.

                                                  Table1:PopularitypredictioninSinaWeibo
 Observation Time                         1 hour                                       2 hours                                       3 hours
 Evaluation Metric       MRSE       mRSE       MAPE      WroPerc       MRSE      mRSE       MAPE       WroPerc      MRSE       mRSE      MAPE       WroPerc
      SEISMIC               -       0.2112        -       48.63%         -       0.1347        -        34.59%         -       0.0823        -       27.15%
   Feature-based         0.2106     0.1254     0.3749     35.17%      0.1796     0.1041     0.3557      28.86%      0.1581     0.0804     0.3147     18.97%
      DeepCas            0.2077    0.0930      0.3633     30.00%      0.1650     0.0670     0.3134      20.55%      0.1365     0.0361     0.2813     17.24%
    CoupledGNN           0.1816     0.0946    0.3515      25.68%      0.1397     0.0519     0.2989     17.81%       0.1120    0.0333     0.2611      13.01%
     Table2:PopularitypredictioninsyntheticdataSet                                      Table3:CompareCoupledGNNwithSingle-GNN
    Evaluation Metric       MRSE      mRSE       MAPE       WroPerc                        Observation Time                   1 hour
         SEISMIC               -      0.2025        -        47.92%                        Evaluation Metric       MRSE      MAPE       WroPerc
      Feature-based        0.1225     0.0452     0.2718      16.90%                           Single-GCN          0.1964     0.3707      29.11%
         DeepCas           0.1199     0.0361     0.2657      16.82%                           Single-GAT          0.1999     0.3754      30.82%
      CoupledGNN           0.1101     0.0339     0.2517      14.71%                          CoupledGNN           0.1816     0.3515      25.68%
                                                                                           Observation Time                   2 hours
6   EXPERIMENTALRESULTS                                                                    Evaluation Metric       MRSE      MAPE       WroPerc
Inthissection,wefirstcompareourCoupledGNNwithbaselineson                                      Single-GCN          0.1595     0.3201      22.26%
thetargettask:popularityprediction.Besides,thesuperiorofthe                                   Single-GAT          0.1569     0.3199      20.55%
coupled structure in CoupledGNN over single-GNN is also demon-                               CoupledGNN           0.1397     0.2989      17.81%
strated. Finally, the effect of hyper-parameters or experimental                           Observation Time                   3 hours
settings is analyzed comprehensively.
                                                                                           Evaluation Metric       MRSE      MAPE       WroPerc
6.1   OverallPerformance                                                                      Single-GCN          0.1230     0.2653      16.10%
The experimental results for popularity prediction on both the                                Single-GAT          0.1222     0.2655      16.10%
synthetic data and real-world data in Sina Weibo are shown in                                CoupledGNN           0.1120     0.2611      13.01%
Table 2 and Table 1 respectively.
   ForSEISMIC,duetoitpredictsinfinitepopularityforsomepieces
of information,we onlyuse mRSEand WroPercas theevaluation                        The reason is that the longer the observation time is, the more
metrics for a fair comparison. From the experimental results, we                 information is available, making the prediction easier.
can see that SEISMIC performs not well on both data sets. Since it
onlyestimatestheimpactofthecascadingeffectineachgeneration                       6.2   CompareCoupledGNNwithSingle-GNN
bytheaveragenumberoffans,it’s easytodeviatefromcomplex
and real situations, thus having limited predictive power. As for                To further demonstrate the advantages of our CoupledGNN struc-
DeepCas,thedeeprepresentationlearningmethods,itdoesperform                       ture, wesimplify our method with twoversions: Single-GCNand
betterthanthefeature-basedmethods.Thisresultindicatesthatit’s                    Single-GAT. In both these two simplified versions, we concatenate
effectivetoautomaticallylearningtherepresentationofthecascade                    theactivationstateandinfluencerepresentationasonevectorasso-
graph through an end-to-end manner rather than heuristically                     ciatedwith eachuser.Thena singlegraphneuralnetwork,i.e., the
design hand-crafted features with prior knowledge.                               commonlyusedgraphconvolutionnetwork(GCN)[13]orgraphat-
   As forour CoupledGNN model,it outperforms allthe baselines                    tentionneuralnetworks(GAT)[31]areappliedtoiterativelyupdate
on both synthetic and real-world datasets, achieving more than                   thevectorassociatedwitheachuser.Thefinalpopularityisobtained
10% improvement over DeepCas in Sina Weibo under the MRSE.                       similarly as CoupledGNN, i.e.,applying a sum pooling mechanism
Theseresultsdemonstratethatit’seffectivetoutilizegraphneural                     over all users after transforming the vector of each user at the last
networkstocapturethecascadingeffectalongnetworkstructure                         layerintoone-dimensionalvalue.Otherhyper-parametersandthe
and to predict the popularity of online content on socialplatforms.              implementation details are the same as CoupledGNN.
Inotherwords,consideringtheinteractionsbetweenearlyadopters                         The experimental results are shown in Table 3. Single-GCN and
and the potential active users, as well as the interactions among                Single-GAT perform almost similarly, indicating that when model-
potential active users over network structure is useful to further               ing thefuturepopularity, thenormalized Laplacianmatrix usedin
improve the prediction performance for future popularity.                        GCNisalreadya goodreflectionofthe correlation betweenpairof
   As for the observation time in Sina Weibo (Figure 1), we can see              userswithalinkingedge.TheattentionmechanismadoptedinGAT
thatthe longertheobservationtime is, thesmallerthe errorsare                     won’t significantly improve the performance further. As for our
(MRSE, MAPE, WrongPerc).This is applicable to all the methods.                   CoupledGNN, it significantly improves the prediction performance

    Figure5:Theinfluenceofapartiallackofnetwork.
under all the evaluation metrics compared with the Single-GNN                     Figure 6:  The distribution of the shortest path length be-
methods.Theseresultsdemonstratethattheactivationstateand                          tweenactivatedusersaftertheobservationtimeandtheset
influence representation play different roles in the modeling of                  ofearlyadopterswithinobservationtime.
future popularity. Instead of mixing them up together, it’s effective
to modelthe activation stateand influencerepresentationby two
graphneuralnetworksrespectivelyandthencouplethembygating
mechanisms.                                                                       hand, Figure 6 shows the distribution of the length of the shortest
                                                                                  path from the set of early adopters within observation time to
6.3   ParameterAnalysis                                                           theactivatedusers after the observation time. Such distribution
We further analyze the effect of the coefficientλ, the number of                  reflects the range of cascading effect caused by the early adopters.
layersinCoupledGNNinthissubsection.Theinfluenceofapartial                         Wecanseethatalmostalltheactivatedusers,i.e.,99.76%,canbe
lack of network is also analyzed.                                                 coveredwithinthree-hopsintheneighborhoodofearlyadopters.
6.3.1    The coefficientλin loss function. We vary the coefficientλ               This means that the optimal number of layers obtained by our
from0.0,0.5,1.0,10.0 to20.0, and the corresponding mean relative                  methodscanexactlymatchthescopeofthecascadingeffectinthis
squarelossis0.1109,0.1101,0.1111,0.1111,0.1141.Inotherwords,                      case,whichprovides guidanceforsettingthehyper-parameterK
theMRSElossisfirstreducedwiththeincreasingofλ,indicating                          in other situations, as well as further supports the effectiveness of
thataddingtheuser-levelcrossentropylossisbeneficialformacro                       our proposed method.
popularity prediction. However, with the continuous increase of                   7   CONCLUSION
λ,the modelpaystoomuch attentiontothe user-levelprediction                        Inthispaper,wefocusontheproblemofnetwork-awarepopularity
whileignoringthemacropopularitypredictiontask,thusresulting                       prediction of online content on social platforms. How to capture
in less effective prediction performance.                                         thecascading effectisone ofthekeysto accuratelypredictfuture
6.3.2    Theinfluenceofapartiallackofnetwork. Consideringthat                     popularityandtacklethisproblem.Inspiredbythesuccessofgraph
our method is based on the given underlying network, we further                   neural networks on various non-Euclidean domains, we propose
construct an experiment with the partial lack of network to better                CoupledGNN to characterize the critical cascading effect along the
demonstratetheapplicabilityandgeneralityofourmethods.Specif-                      network structure. We devote to modeling the two crucial compo-
ically, we randomly dropout a certain percentage of edges in the                  nents in the cascading effect, i.e., the iterative interplay between
social network under the premise of network connectivity. Then                    nodeactivation statesand thespreadof influence,bytwo coupled
wetrainandtestthepredictionmodelbasedonsuchanincomplete                           graphneuralnetworks respectively.Specifically,one graphneural
network. Note that, not only our CoupledGNN are influenced by                     networkmodelstheinterpersonalinfluence,gatedbytheactivation
suchdropoutofnetworkedges,butalsothebaselinemethods.Here,                         stateofusers.Theothergraphneuralnetworkmodelstheactiva-
we compare our methods with the strong baseline, i.e., DeepCas,                   tionstate ofusersvia interpersonalinfluencefrom theirneighbors.
while varying the dropout of the network from 0%, 5%,10% to 20%.                  Theiterativeupdatemechanismofneighborhoodaggregationin
Figure5showsthatboththeperformanceofourCoupledGNNand                              GNNseffectivelycapturessuchacascadingeffectinpopularitypre-
DeepCas will be slightly degraded by the dropout of the network.                  dictionalongtheunderlyingnetwork.Theexperimentsconducted
But by comparison, our methods always significantly perform bet-                  onbothsyntheticandreal-worlddatavalidatetheeffectivenessof
ter than DeepCas. In conclusion, our CoupledGNN model is still                    ourproposedmethodforpopularityprediction.Asforfuturework,
suitableforpredicting thefuturepopularityofonlinecontent even                     we will devote to modeling the cascading effect along the network
when a small part of the network structure is lacking.                            when further given the specific adoption time of early adopters.
6.3.3    The number of layers in CoupledGNN. While we apply our                   ACKNOWLEDGMENTS
CoupledGNN to capture the cascading effect along the network,
it’sinterestingtofurtherfindoutwhetherthenumberoflayersK                          This work is funded by the National Natural Science Foundation
inour CoupledGNNcancorrespond tothe scopeofthe cascading                          ofChinaundergrantnumbers61425016,61433014,91746301,and
effect.WetaketheSinaWeiboasashowingcase.Ontheonehand,                             61472400. This work is supported by Beijing Academy of Artificial
asmentionedinSection5.2,thenumberoflayersK ischosenfrom                           Intelligence(BAAI).HuaweiShenisalsofundedbyK.C.WongEdu-
2,3,4.Whentheobservationtime windowis1 hour,weobtainthe                           cationFoundationandtheYouthInnovationPromotionAssociation
best performance on the validation set withK =  3. On the other                   of the Chinese Academy of Sciences.

REFERENCES                                                                                                  [24]  Satoshi Sanjo and Marie Katsurai. 2017.   Recipe Popularity Prediction with
 [1]  Qi Cao, Huawei Shen, Keting Cen, Wentao Ouyang, and Xueqi Cheng. 2017.                                      DeepVisual-SemanticFusion.In Proceedings of the 2017 ACM on Conference on
      DeepHawkes: Bridgingthe Gap BetweenPrediction and Understandingof Infor-                                    Information and Knowledge Management (CIKM ’17).2279–2282.
      mationCascades.In Proceedings of the 2017 ACM on Conference on Information                            [25]  Jin Shangand MingxuanSun. 2019. Geometric HawkesProcesses withGraph
      and Knowledge Management (CIKM ’17).1149–1158.                                                              Convolutional Recurrent Neural Networks. In Thirty-Three AAAI Conference on
 [2]  Qi Cao,Huawei Shen, Hao Gao,Jinhua Gao, andXueqi Cheng. 2017. Predicting                                    Artificial Intelligence (AAAI ’19).
      the Popularity of Online Content with Group-specific Models. InProceedings of                         [26]  Jiangli Shao, Huawei Shen, Qi Cao, and Xueqi Cheng. 2019. Temporal Convo-
      the 26th International Conference on World Wide Web Companion (WWW ’17).                                    lutional Networks for Popularity Prediction of Messages on Social Medias. In
      765–766.                                                                                                    China Conference on Information Retrieval.Springer,135–147.
 [3]  Justin Cheng, Lada Adamic, P. Alex Dow, Jon Michael Kleinberg, and Jure                               [27]  Huawei Shen, Dashun Wang, Chaoming Song, and Albert-László Barabási. 2014.
      Leskovec.2014. Can CascadesBePredicted?.In Proceedings of the 23rd Interna-                                 Modeling andPredicting Popularity Dynamicsvia Reinforced PoissonProcesses.
      tional Conference on World Wide Web (WWW ’14).925–936.                                                      In Proceedings of the Twenty-Eighth AAAI Conference on Artificial Intelligence
 [4]  AaronClauset,M.E.J.Newman,andCristopherMoore.2004.Findingcommunity                                          (AAAI’14).291–297.
      structure inverylargenetworks. Phys. Rev. E 70(Dec2004), 066111. Issue6.                              [28]  BenjaminShulman, AmitSharma, andDan Cosley. 2016. Predictabilityof popu-
 [5]  Nan Du, Yingyu Liang, Maria-Florina Balcan, and Le Song. 2014.  Influence                                   larity:Gapsbetweenpredictionandunderstanding.In Tenth International AAAI
      Function Learning in Information Diffusion Networks. InProceedings of the 31th                              Conference on Web and Social Media (ICWSM ’16).348–357.
      International Conference on Machine Learning (ICML’14). II–2016–II–2024.                              [29]  AlexandruTatar,MarceloDiasdeAmorim,SergeFdida,andPanayotisAntoniadis.
 [6]  Xiaofeng Gao, Zhenhao Cao, Sha Li, Bin Yao, Guihai Chen, and Shaojie Tang.                                  2014. A surveyon predicting the popularityof web content. Journal of Internet
      2019. TaxonomyandEvaluationforMicroblogPopularityPrediction. ACMTrans.                                      Services and Applications 5,1(13 Aug2014),8.
      Knowl. Discov. Data 13,2, Article15 (March 2019), 40 pages.                                           [30]  Johan Ugander, Lars Backstrom, Cameron Marlow, and Jon Kleinberg. 2012.
 [7]  JacobGoldenberg,BarakLibai,andEitanMuller.2001. Talkofthenetwork:A                                          Structural diversity in social contagion. Proceedings of the National Academy of
      complexsystemslookattheunderlyingprocessofword-of-mouth. Marketing                                          Sciences 109,16 (2012),5962–5966.
      letters 12,3 (2001),211–223.                                                                          [31]  Petar Veličković,Guillem Cucurull,Arantxa Casanova,Adriana Romero,Pietro
 [8]  ManuelGomez-Rodriguez,DavidBalduzzi,andBernhardSchölkopf.2011. Un-                                          Lio, andYoshuaBengio. 2018. Graphattention networks. In Proceedings of the
      covering the Temporal Dynamics of Diffusion Networks. InProceedings of the                                  7th International Conference on Learning Representations.
      28th International Conference on Machine Learning (ICML’11). 561–568.                                 [32]  Jia Wang, Vincent W Zheng, Zemin Liu, and Kevin Chen-Chuan Chang. 2017.
 [9]  Mark Granovetter. 1978.  Threshold Models of Collective Behavior.  Amer. J.                                 Topological recurrent neural network for diffusion prediction. In2017 IEEE Inter-
      Sociology 83,6 (1978), 1420–1443.                                                                           national Conference on Data Mining (ICDM).475–484.
[10]  Adrien Guille and Hakim Hacid. 2012.  A predictive model for the temporal                             [33]  LilianWeng,FilippoMenczer,andYong-YeolAhn.2014. Predictingsuccessful
      dynamicsofinformationdiffusioninonlinesocialnetworks.InProceedings of                                       memes using network and community structure. In Eighth international AAAI
      the 21st international conference on World Wide Web (WWW’12). 1145–1152.                                    conference on weblogs and social media (ICWSM ’14).535–544.
[11]  RuochengGuoandPauloShakarian.2016. AComparisonofMethodsforCascade                                     [34]  QitianWu,ChaoqiYang,HengruiZhang,XiaofengGao,PaulWeng,andGuihai
      Prediction. In Proceedings of the 2016 IEEE/ACM International Conference on                                 Chen. 2018.  Adversarial Training Model Unifying Feature Driven and Point
      Advances in Social Networks Analysis and Mining (ASONAM ’16).591–598.                                       Process Perspectives for Event Popularity Prediction. In Proceedings of the 27th
[12]  WilliamL.Hamilton,RexYing,andJureLeskovec.2017.InductiveRepresentation                                      ACMInternationalConferenceonInformationandKnowledgeManagement (CIKM
      LearningonLargeGraphs.In Proceedings of the 31st International Conference on                                ’18).517–526.
      Neural Information Processing Systems (NIPS ’17).1025–1035.                                           [35]  Bingbing Xu, HuaweiShen, Qi Cao, Keting Cen,and Xueqi Cheng. 2019. Graph
[13]  Thomas N Kipf and Max Welling. 2017.  Semi-Supervised Classification with                                   Convolutional Networks Using Heat Kernel for Semi-supervised Learning. In
      GraphConvolutionalNetworks.InProceedingsofthe6thInternationalConference                                     Proceedings of the 28th International Joint Conference on Artificial Intelligence
      on Learning Representations.                                                                                (IJCAI’19).1928–1934.
[14]  KristinaLermanandAramGalstyan.2008. AnalysisofSocialVotingPatternson                                  [36]  BingbingXu, HuaweiShen,Qi Cao,YunqiQiu,and XueqiCheng.2019. Graph
      Digg.In Proceedings of the First Workshop on Online Social Networks (WOSN ’08).                             Wavelet Neural Network. In Proceedings of the 8th International Conference on
      7–12.                                                                                                       Learning Representations.
[15]  Jure Leskovec, Deepayan Chakrabarti, Jon Kleinberg, Christos Faloutsos, and                           [37]  KeyuluXu,WeihuaHu,JureLeskovec,andStefanieJegelka.2019. HowPowerful
      ZoubinGhahramani. 2010. KroneckerGraphs:AnApproachtoModelingNet-                                            are Graph Neural Networks?. In Proceedings of the 8th International Conference
      works. J. Mach. Learn. Res.11 (March 2010), 985–1042.                                                       on Learning Representations.
[16]  ChengLi,JiaqiMa,XiaoxiaoGuo,andQiaozhuMei.2017. DeepCas:AnEnd-to-                                     [38]  Junchi Yan, Xin Liu, Liangliang Shi, Changsheng Li, and Hongyuan Zha. 2018.
      end Predictor of Information Cascades.In Proceedings of the 26th International                              ImprovingMaximum LikelihoodEstimation ofTemporal PointProcessvia Dis-
      Conference on World Wide Web (WWW ’17).577–586.                                                             criminative and Adversarial Learning. In Proceedings of the 27th International
[17]  Dongliang Liao, JinXu, Gongfu Li, Weijie Huang, WeiqingLiu, and Jing Li. 2019.                              Joint Conference on Artificial Intelligence (IJCAI’18).2948–2954.
      PopularityPredictiononOnlineArticleswithDeepFusionofTemporalProcess                                   [39]  Rex Ying, Jiaxuan You, Christopher Morris, Xiang Ren, William L. Hamilton,
      andContentFeatures.In Thirty-Three AAAI Conference on Artificial Intelligence                               and Jure Leskovec. 2018.  Hierarchical Graph Representation Learning with
      (AAAI ’19).200–207.                                                                                         Differentiable Pooling. In Proceedings of the 32rd International Conference on
[18]  TravisMartin,JakeM.Hofman,AmitSharma,AshtonAnderson,andDuncanJ.                                             Neural Information Processing Systems (NIPS’18).4805–4815.
      Watts. 2016.   Exploring Limits to Prediction in Complex Social Systems. In                           [40]  LinyunYu,PengCui,FeiWang,ChaomingSong,andShiqiangYang.2015. From
      Proceedings of the 25th International Conference on World Wide Web (WWW ’16).                               micro to macro: Uncovering and predicting information cascading process with
      683–694.                                                                                                    behavioraldynamics.In2015IEEEInternationalConferenceonDataMining(ICDM
[19]  Swapnil Mishra, Marian-Andrei Rizoiu, and Lexing Xie. 2016. Feature Driven                                  ’15).559–568.
      and Point Process Approaches for Popularity Prediction. In Proceedings of the                         [41]  Jing Zhang,Biao Liu,Jie Tang, TingChen, and JuanziLi. 2013. SocialInfluence
      25thACMInternationalonConferenceonInformationandKnowledgeManagement                                         Locality for Modeling Retweeting Behaviors. In Proceedings of the Twenty-Third
      (CIKM ’16). 1069–1078.                                                                                      International Joint Conference on Artificial Intelligence (IJCAI ’13).2761–2767.
[20]  BryanPerozzi,RamiAl-Rfou,andStevenSkiena.2014. DeepWalk:OnlineLearn-                                  [42]  JingZhang,JieTang,JuanziLi,YangLiu,andChunxiaoXing.2015. WhoInflu-
      ing of Social Representations. In Proceedings of the 20th ACM SIGKDD Interna-                               enced You? Predicting Retweet via Social Influence Locality. ACM Trans. Knowl.
      tional Conference on Knowledge Discovery and Data Mining (KDD ’14). 701–710.                                Discov. Data 9,3,Article25 (April2015),26 pages.
[21]  HenriquePinto,JussaraM.Almeida,andMarcosA.Gonçalves.2013. UsingEarly                                  [43]  Muhan Zhang, Zhicheng Cui, Marion Neumann, and Yixin Chen. 2018. An end-
      View Patterns to Predict the Popularity of Youtube Videos. In Proceedings of the                            to-enddeeplearningarchitectureforgraphclassification.InThirty-Second AAAI
      Sixth ACM International Conference on Web Search and Data Mining (WSDM ’13).                                Conference on Artificial Intelligence (AAAI ’18).
      365–374.                                                                                              [44]  Wei Zhang,WenWang,JunWang,andHongyuanZha.2018. User-guidedHier-
[22]  Jiezhong Qiu, Jian Tang, Hao Ma, Yuxiao Dong, Kuansan Wang, and Jie Tang.                                   archicalAttention Networkfor Multi-modalSocial ImagePopularity Prediction.
      2018. DeepInf: Social Influence Prediction with Deep Learning. InProceedings of                             In Proceedings of the 2018 World Wide Web Conference (WWW ’18).1277–1286.
      the24thACMSIGKDDInternationalConferenceonKnowledgeDiscoveryandData                                    [45]  Xiaoming Zhang, Zhoujun Li, Wenhan Chao, and Jiali Xia. 2014.  Popularity
      Mining (KDD ’18). 2110–2119.                                                                                PredictionofBurstEventinMicroblogging.InWeb-AgeInformationManagement.
[23]  Marian-AndreiRizoiu,LexingXie,ScottSanner,ManuelCebrian,HonglinYu,and                                       SpringerInternational Publishing, 484–487.
      Pascal Van Hentenryck. 2017. Expecting toBe HIP: Hawkes IntensityProcesses                            [46]  Qingyuan Zhao, Murat A. Erdogdu, Hera Y. He, Anand Rajaraman, and Jure
      forSocialMediaPopularity.InProceedingsofthe26thInternationalConferenceon                                    Leskovec. 2015. SEISMIC: A Self-Exciting Point Process Model for Predicting
      World Wide Web (WWW ’17).735–744.                                                                           TweetPopularity.InProceedingsofthe21thACMSIGKDDInternationalConference
                                                                                                                  on Knowledge Discovery and Data Mining (KDD ’15).1513–1522.

