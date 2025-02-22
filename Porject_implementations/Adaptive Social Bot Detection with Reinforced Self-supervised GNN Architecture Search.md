                                                                                                                                     1
RoSGAS: Adaptive Social Bot Detection with Reinforced
Self-Supervised GNN Architecture Search
YINGGUANG YANG, University of Science and Technology of China, China and Key Laboratory of
Cyberculture Content Cognition and Detection, Ministry of Culture and Tourism, China
RENYU YANG∗, University of Leeds, United Kingdom
YANGYANGLI† ,NationalEngineeringLaboratoryforPublicSafetyRiskPerceptionandControlbyBig
Data,ChinaandKeyLaboratoryofCybercultureContentCognitionandDetection,MinistryofCultureand
Tourism, China
KAICUI,UniversityofScienceandTechnologyofChina,ChinaandKeyLaboratoryofCybercultureContent
Cognition and Detection, Ministry of Culture and Tourism, China
ZHIQIN YANG and YUE WANG,Beihang University, China
JIE XU,University of Leeds, United Kingdom and Beihang University, China
HAIYONGXIE‡ ,UniversityofScienceandTechnologyofChina,ChinaandKeyLaboratoryofCyberculture
Content Cognition and Detection, Ministry of Culture and Tourism, China
Social bots are referred to as the automated accounts on social networks that make attempts to behave like
human.WhileGraphNeuralNetworks(GNNs)hasbeenmassivelyappliedtothefieldof socialbotdetection,
ahugeamountofdomainexpertiseandpriorknowledgeisheavilyengagedinthestate-of-theartapproaches
todesignadedicatedneuralnetworkarchitectureforaspecificclassificationtask.Involvingoversizednodes
andnetwork layers inthemodel design,however,usuallycauses theover-smoothingproblemand thelackof
embedding discrimination. In this paper, we proposeRoSGAS, a novel Reinforced and Self-supervised GNN
Architecture Search framework to adaptively pinpoint the most suitable multi-hop neighborhood and the
numberoflayersintheGNNarchitecture.Morespecifically,weconsiderthesocialbotdetectionproblemasa
user-centricsubgraphembeddingandclassificationtask.Weexploitheterogeneousinformationnetworkto
∗Co-first author with equal contribution
† Corresponding author
‡ Corresponding author
Authors’addresses:YingguangYang,dao@mail.ustc.edu.cn,SchoolofCyberScienceandTechnology,UniversityofScience
and Technology of China, 96 Jinzhai Road, Hefei, Anhui, China, 230026 and Key Laboratory of Cyberculture Content
CognitionandDetection,MinistryofCultureandTourism,China;RenyuYang,UniversityofLeeds,WoodhouseLane,Leeds,
WestYorkshire,UnitedKingdom,LS29JT,r.yang1@leeds.ac.uk;YangyangLi,NationalEngineeringLaboratoryforPublic
Safety Risk Perception and Control by Big Data, 11 Shuangyuan Road, Beijing, Beijing, China, liyangyang@cetc.com.cn
and Key Laboratory of Cyberculture Content Cognition and Detection, Ministry of Culture and Tourism, China; Kai Cui,
School of Cyber Science and Technology, University of Science and Technology of China, 96 Jinzhai Road, Hefei, Anhui,
China and Key Laboratory of Cyberculture Content Cognition and Detection, Ministry of Culture and Tourism, China,
kaicui@mail.ustc.edu.cn;ZhiqinYang;YueWang,BeihangUniversity,37XueyuanRoad,Beijing,Beijing,China,100191,
yangzqccc@buaa.edu.cn,zb2039111@buaa.edu.cn;JieXu,UniversityofLeeds,WoodhouseLane,Leeds,WestYorkshire,
UnitedKingdom,LS29JTandBeihangUniversity,37XueyuanRoad,Beijing,China,j.xu@leeds.ac.uk;HaiyongXie,Schoolof
CyberScienceandTechnology,UniversityofScienceandTechnologyofChina,96JinzhaiRoad,Hefei,Anhui,ChinaandKey
Laboratory ofCyberculture ContentCognition andDetection, Ministryof Cultureand Tourism, China,hxie@ustc.edu.cn.
Permissionto makedigital orhard copiesof allorpart ofthis workfor personalorclassroomuseis grantedwithout fee
providedthatcopiesarenotmadeordistributedforprofitorcommercialadvantageandthatcopiesbearthisnoticeandthe
fullcitationonthefirstpage.Copyrightsforcomponentsofthisworkownedbyothersthantheauthor(s)mustbehonored.
Abstracting withcredit ispermitted. To copyotherwise, orrepublish, topost on serversor toredistribute tolists, requires
prior specific permission and/or a fee. Request permissions from permissions@acm.org.
© 2022 Copyright held by the owner/author(s). Publication rights licensed to ACM.
1559-1131/2022/1-ART1 $15.00
https://doi.org/10.1145/3572403
                                                  ACM Trans. Web, Vol. 1, No. 1, Article 1. Publication date: January 2022.

1:2                                                                                                                                                                               Yang, et al.
presenttheuserconnectivityby leveraging accountmetadata,relationships,behavioralfeaturesandcontent
features.RoSGASuses amulti-agent deep reinforcement learning(RL) mechanism for navigating thesearch
ofoptimalneighborhoodandnetworklayerstolearnindividuallythesubgraphembeddingforeachtarget
user.A nearestneighbormechanism isdevelopedfor acceleratingthe RLtrainingprocess, andRoSGAScan
learnmorediscriminative subgraphembeddingwith theaid ofself-supervised learning.Experiments on5
TwitterdatasetsshowthatRoSGASoutperformsthestate-of-the-artapproachesintermsofaccuracy,training
efficiency and stability, and has better generalization when handling unseen samples.
CCSConcepts:• Computingmethodologies→Artificialintelligence;Machinelearningapproaches.
Additional Key Words and Phrases: Graph neural network, architecture search, reinforcement learning
ACMReferenceFormat:
YingguangYang,RenyuYang,YangyangLi,KaiCui,ZhiqinYang,YueWang,JieXu,andHaiyongXie.2022.
RoSGAS: Adaptive Social Bot Detection with Reinforced Self-Supervised GNN Architecture Search. ACM
Trans. Web 1, 1, Article 1 (January 2022), 32 pages. https://doi.org/10.1145/3572403
1   INTRODUCTION
Socialbots–theaccountsthatarecontrolledbyautomatedsoftwareandmockhumanbehaviours
[1] – widely exist on online social platforms such as Twitter, Facebook, Instagram, Weibo, etc. and
normallyhave maliciousattempts. Forexample, interestgroupsor individualscan usesocial bots
toinfluencethepoliticsandeconomy,e.g.,swayingpublicopinionsatscale,throughdisseminating
disinformation and on-purpose propaganda, and to steal personal privacy through malicious
websitesorphishing messages[47].Suchdeception andfraudcanreach outtoahuge community
and lead to cascading and devastating consequences.
   Socialbotshavebeenlongstudiedbutnotyetwell-resolvedduetothefastbotevolution[7].The
thirdgenerationofbotssince2016withdeepenedmixtureofhumanoperationsandautomatedbot
behaviorsmanagedtodisguisethemselvesandsurvivedplatform-leveldetectionusingtraditional
classifiers[7,9].Thecat-and-mousegamecontinues–whilenewwork-around,camouflageand
adversarial techniques evolve to maintain threats and escape from perception, a huge body of
detectionapproachesemergetodifferentiatethehiddenbehaviorsoftheemergingsocialsbotsfrom
legitimateusers.TherecentadvancementsinGraphNeuralNetworks(GNNs)[40]canhelptobetter
understand the implicit relationships between abnormal and legitimate users and thus improve
the detection efficacy. GNN-based approaches [6, 14, 15, 17, 31, 37, 50] formulate the detection
procedure as a node or graph classification problem. Heterogeneous graphs are constructed by
extractingtheaccounts’metadataandcontentinformationfromsocialnetworksbeforecalculating
numericalembeddingfornodesandgraphs.However,therearestillseveralinterrelatedproblems
to be addressed:
   GNNarchitecturedesignhasastrongdependenceupondomainknowledgeandmanualintervention.
In most of the existing works, the embedding results are inherently flat because the neighbor
aggregation ignores the difference between the graph structure pertaining to the target node and
the structures of other nodes. This will result in the lack of deterministic discrimination among
the final embeddings when the scale of the formed graph structure grows to be tremendous. To
addressthisissue,subgraphsareleveragedtoexplorethelocalsubstructuresthatmerelyinvolve
partialnodes,whichcanobtainnon-trivialcharacteristicsandpatterns[45].However,thesubgraph
neuralnetworkbasedapproachesheavilyrelyonexperiencesordomainknowledgeinthedesignof
rulesforextractingsubgraphsandofmodelarchitecturesformessageaggregation[3,29,35].This
manual interventionsubstantially impedesthe elaborateddesign ofa neuralnetwork modelthat
can adapt to the evolving changes of the newer social bots. Usingfixed and fine-grained subgraph
extraction rules is not sufficiently effective [9, 18].
ACM Trans. Web, Vol. 1, No. 1, Article 1. Publication date: January 2022.

RoSGAS: Adaptive Social Bot Detection with Reinforced Self-Supervised GNN Architecture Search                               1:3
  Over-assimilated embedding when aggregating a huge number of neighbors.Themostintractable
anddemandingtaskistoeffectivelyperceiveandpinpointthecamouflagesofthenew-generation
socialbots.Camouflagetechnologymainlycomprisestwodistinctcategories–featurecamouflage
and relation camouflage.Featurecamouflage is referred toas the procedure wherebots steal the
metadata of the benign accounts and transform into their own metadata. They also employ ad-
vancedgenerationtechnologytocreatecontentinformation[7].Apartfrommimicking features of
legitimateusers,relationcamouflagetechniquesfurtherhidethemaliciousbehaviorsbyindiscrim-
inatelyinteractingwithlegitimateusersandestablishingfriendshipswithactivebenignusers[57].
The interactions, particularly with the influencers, can considerably shield the bots from being
detected. It is thus critical to include sufficient heterogeneous nodes in the neighborhood when
extractingthesubgraphforthetargetusersothatcamouflagedbotscanbepickedup.Meanwhile,
itisalsoimportanttodifferentiatethesubgraphembeddingsofdifferenttargetuserswhilesimilar-
izing the embeddings within the same target user. However, the over-smoothing representation
problemwillmanifest[30,53]astheinvolvementofahugenumberofnodesintheGNNstendsto
over-assimilate the node numerical embedding when aggregating its neighbor information.
  Inadequate labeled samples. A large number of labeled samples are presumably acquired and
massivelyusedinthesupervisedmodeltraining.However,thisassumptioncanbehardlyachieved
in the real-world social bot detection. In fact, there are always very limited users with annotated
labels, or limited access to adequate and labelled samples [21]. This will hamper the precision of
supervised deep learning models and particularly lead to poor performance in identifying out-
of-sample entities, i.e., the new types of social bots out of the existing datasets or established
models.
  Toaddresstheseissues,thestate-of-the-artworks[20,27,52,62,63]adoptreinforcementlearning
(RL)tosearchtheGNNarchitecture.However,suchapproachessometimeslackgeneralizability;the
effectiveness ofdetermining theoptimal GNNarchitecture istightly boundto specific datasetsand
usuallyhavehugesearchspace,andhencelowefficiency.Theyarenotsuitedforsocialnetwork
networks where the graph structure follows a power-law distribution [4, 34]. In this scenario,
dense and sparse local graph structures co-exist and huge disparities manifest among different
users.Additionally,as onlytakingasmall fractionofthe labelledusersas theenvironmentstate,
the existing RL agents can hardly learn the state space in an accurate manner and will lead to
a slow convergence in the RL agents. Hence, it is highly imperative to personalize the selection
of subgraphs and GNN architecture of the model for each target user, without prior knowledge
and manual ruleextraction, and to devise automated andadaptive subgraph embedding to fitthe
ever-changing bot detection requirements.
  In this paper, we proposeRoSGAS, a subgraph-based scalable Reinforced and Self-supervised
GNNArchitectureSearchapproachtoadaptivelyextractthesubgraphwidthandsearchthemodel
architecture forbettersubgraph embedding,and tospeed upthe RLmodel convergencethrough
self-supervised learning.Specifically, we useHeterogeneousInformationNetwork (HIN)tomodel
theentitiesandrelationshipsinthesocialnetworksandusemeta-schemaandmeta-pathtodefine
the required relationship and type constraints of nodes and edges in the HIN, on the basis of
real-worldobservationsinsocialnetworkplatforms.Weformulatethesocialbotdetectionproblem
as a subgraph classificationproblem. We firstly propose a multi-agentreinforcement learning (RL)
mechanism for improving the subgraph embedding for target users. The RL agent can start to
learnthelocalstructureoftheinitial1-orderneighborsubgraphofagiventargetuserandhelpto
select themost appropriate numberof neighborhops as the optimalwidth of thesubgraph. The
RL agent is also elaborately devised to select the optimal number of model layers such that the
neural networks are well-suited for encoding the dedicated subgraphs with sufficient precision,
withoutintroducingoversizedarchitectureandcomputationoverhead.Wethenexploittheresidual
                                             ACM Trans. Web, Vol. 1, No. 1, Article 1. Publication date: January 2022.

1:4                                                                                                                                                                               Yang, et al.
   Fig. 1. (a) An example of HIN for social network. (b) Network schema of the HIN for social network.
structuretoretainthecharacteristicsofatargetuser asmuchaspossible,therebyovercomingthe
over-smooth problem on the occasion of message aggregations from ahuge number of neighbor
nodes.WhileusingRLtoautomatetheneighborselectionandmodelconstruction,weadditionally
developanearestneighbormechanismintheearlystageofRLforacceleratingthetrainingprocess
oftheproposedRLapproach.Aself-supervisedlearningmechanismisfurtherinvestigatedand
integratedwiththeRoSGASframeworktoovercomethedeficiencyofover-assimilatedembedding.
The self-supervised module canfacilitate more discriminativerepresentationvectors andhelp to
enhance the capability of expressing discrepancies among different target users. Experimental
results show thatRoSGAS outperforms the state-of-the-art approaches over five Twitter datasets
andcanachievecompetitiveeffectivenesswhencomparedwithhand-madedesignofthemodel
architecture.
   Particularly, the main contribution of this work are summarized as follows:
    •proposed for the first time a user-centric social bot detection framework based on Heteroge-
       neous Information Network and GNN without prior knowledge.
    •developedanadaptiveframeworkthatleveragesDeepQ-learningtooptimizethewidthof
       thesubgraphandtheGNNarchitecturelayersforthesubgraphcorrespondingtoeachtarget
       user.
    •investigated a nearest neighbor mechanism for accelerating the convergence of training
       process in RL agents.
    •proposed a self-supervised learning approach toenable homologous subgraphs have closer
       representation vectors whilst increasing the disparities of representation vectors among
       non-homologous subgraphs after information aggregation.
    •presented an explicit explanation for the model stability.
   Organization.Thispaperisstructuredasfollows:Section2outlinestheproblemformulation,and
Section3describesthetechnicaldetailsinvolvedinRoSGAS.Theexperimentalsetupisdescribed
inSection4andtheresultsoftheexperimentarediscussedinSection5.Morediscussionsaregiven
in Section 6. Section 7 presents the related work before we conclude the paper in Section 8.
ACM Trans. Web, Vol. 1, No. 1, Article 1. Publication date: January 2022.

RoSGAS: Adaptive Social Bot Detection with Reinforced Self-Supervised GNN Architecture Search                               1:5
                                      Fig. 2. The extracted meta-paths.
2   PROBLEM FORMULATION
In this section, we introduce HINs and information network representation before discussing the
scope of this work and formulating the target problem.
2.1   Preliminaries
Inthiswork,wefollowtheterminologiesusedintheworkof[13,24,36,42]todefineHeterogeneous
Information Network (HIN) embedding. The aim is to project different nodes in the HIN into a
low-dimensionalspacewhistpreservingtheheterogeneousstructureandrelationshipsbetween
heterogeneous nodes.
Definition 1. Heterogeneous Information Network.A heterogeneous information network
(HIN)denotedasG=G(V,E,F,R,𝜑,𝜙),whereVdenotesthenodesset,Edenotestheedgesset,
FdenotesthenodetypessetandRdenotestheedgetypesset.Inreal-worldsettings,theremay
bemultiple typesofnodes oredges,i.e.,|F|+|R|>  2.Each individualnode𝑖∈Visassociated
with a nodetype mapping function𝜑 :V→F; similarly, each individual edge𝑒∈Ehas an edge
type mapping function𝜙 :E→R.
  Inanutshell,real-lifeinformationnetworkshavedifferentstructuresconsistingofmulti-typed
entitiesandrelationships.Arelationshipisreferredtoasthelinkbetweenentitiesinanetworkor
graphsystem.Forexample,Fig.1(a)showsanexampleofsocialnetworkHINthatweconstructfor
Twitter. It comprises fivetypes of nodes (user, tweet, comment,entity and hashtag) and sixtypes
of relationships (write, follow, post, reply, retweet, and contain).
Definition2.NetworkSchema.Given a HING(V,E,F,R,𝜑,𝜙), the network schema for net-
workGcanbedenotedasT(F,R),adirectedgraphwiththenodetypesetFandedgetypesset
R. In simple words, HIN schema comprehensively depicts the node types and their relations in
an HIN, and provide a meta template to guide the exploration of node relationships and extract
subgraphsfromtheHIN.Fig.1(b)exemplifiesthenetworkschemathatcanreflectentitiesandtheir
interactions in a generic social network.
Definition3.Meta-path.GivenaMetaSchemaT(F,R),aMeta-PathMP,denotedasF1                                      R1−→
F2  R2−→...R𝑙−1−→F𝑙, isa pathonTthat connectsa pairof networknodes anddefines the composite
relationRwhich contains multi types of relationships.
  Inreality,ameta-pathdescribesthesemanticrelationshipbetweennodes.Miningsuchsemantic
relationshipisthecornerstoneofsubsequenttaskssuchasclassification,clustering,etc.Asshown
in Fig. 2, we extracted five most useful meta-paths from our defined meta-schema, based on
observations in social network platforms.
                                            ACM Trans. Web, Vol. 1, No. 1, Article 1. Publication date: January 2022.

1:6                                                                                                                                                                               Yang, et al.
2.2   Problem Statement
We consider the social bot detection problem as a subgraph classification task, instead of a node
classification task, in a semi-supervised learning manner.
Definition4.Semi-supervisedSubgraphClassification.Givenacollectionoftargetusers,the
subgraph pertaining to the𝑖-th target user can be defined asG𝑖 ={V𝑖,X𝑖,{E𝑟}|𝑅𝑟= 1,𝑦𝑖}.V𝑖is
the collection of nodes in the subgraph, including𝑣𝑖0, the target user itself, and the neighbors
{𝑣𝑖1,...,𝑣𝑖𝑛}within a few hops from the target user. These nodes are extracted from the entire
graphGandconsistofdifferenttypes.Eachnodeinitiallyhasa𝑑-dimensionalfeaturevectorand
X𝑖represents the vector set of all node embeddings, i.e.,X𝑖={𝑥𝑖0,...,𝑥𝑖𝑛}where each element
𝑥∈R . The edge in the subgraph can be represented as𝑒𝑟𝑖𝑚,𝑖𝑛  =(𝑣𝑖𝑚,𝑣𝑖𝑛)∈E𝑟, where𝑣𝑖𝑚   and𝑣𝑖𝑛
isconnectedthroughacertainrelationship𝑟∈{1,...,𝑅}.𝑦𝑖∈{0,1}representsabinarylabelof
the target user𝑣𝑖0; 0 indicates benign account while 1 represents a social bot. Once a dedicated
subgraph𝐺𝑖is extracted, the subsequent task is to conduct a subgraph binary classification. At
thecore ofthe adaptivearchitecturesearch istopinpoint thesubgraph width(𝑘)and theoptimal
value of model layers (𝑙) that constitute the whole bot detection model.
2.3   Research Questions
For achieving discriminative, cost-effective and explainable subgraph embedding, there are three
main research challenges facing the RL-based social bot detection:
   [Q1] How to determine the right size of the subgraph for an individual target user in
apersonalizedandcost-effectivemanner?Themajorissue withtheDNNconstructionis the
selection of neighbor hops and the model layers. In fact, two interrelated yet opposite factors may
affect the choice of a detection model. On the one hand, a larger number of hops and model layers
caninvolvemorenodes,includingbothbenignandmaliciousnodes,intheneighboraggregation.
This is beneficial for the detection quality since the hidden camouflages of social bots could be
identifiedbythehigher-ordersemanticembeddingenabledbythecontinuumofHIN-baseddata
engineering and DNN model training. However, excessive involvement will bring performance
issuesin termsof time-and computation-efficiency, and,more severely,lead tothe over-smooth
problemcommonly manifestedingraphsat scale[30].On theotherhand,a smallportionofthe
neighbors would overlook node information and lead to less informative node embeddings. To
resolvethisdilemma–balancingcompetitiveaccuracyandhighcomputationefficiency–whilst
addressing the assimilation within the neighborhood, it is critical to automatically pick up an
appropriatehopsofneighborsandtostackupjust-enoughneuralnetworklayerstobeassembledin
thedetectionmodel.Thisrequiresthereinforcementlearningprocesstoproperlydefinededicated
policies and optimize the setting of environment states and actions.
   [Q2]Howtoacceleratetheconvergenceofreinforcementlearning? Itisobservablethat
in the initial stage of training, the learning curve substantially fluctuates and this phenomenon
will slow down the training process and model convergence. This is because in the starting phase
the noisy data may take up a high portion of the limited memory buffer and thus misdirect to the
wrong optimization objective. To ensure the training efficiency, it is necessary to boost the action
exploration in RL agent and speed up the training stabilization.
   [Q3] How to more efficiently optimize the reinforcement learning agents in the face
oflimitedannotatedlabels? Data annotation is expensive and sometimes difficult in practical
problemsolving.IfonlyasmallportionofthelabelledusersareusedastheinputoftheRLagentas
environmentstate,thestatespacecannotbeaccuratelyandefficientlylearntwithinarequiredtime
frame.Thiswillconsequentlydelay theoptimizationofaRLagentandfurtherhaveacascading
ACM Trans. Web, Vol. 1, No. 1, Article 1. Publication date: January 2022.

           RoSGAS: Adaptive Social Bot Detection with Reinforced Self-Supervised GNN Architecture Search                               1:7
                                                        Table 1. Notations.
           impact on the multi-agent training. This issue therefore necessitates an self-supervised learning
           mechanism for optimizing the training effectiveness and efficiency.
           3   METHODOLOGY
           In this section, we will introduce how we design the social bot detection framework through
           adaptiveGNNarchitecturesearchwithreinforcementlearning.Wefirstintroducethebasicprocess
           ofsubgraph embedding(Section3.1).In responseto[Q1],we introduceareinforcementlearning
           enhanced architecture search mechanism (Section 3.2). To address [Q2], we propose a nearest
           neighbor mechanism (Section 3.3) for accelerating the convergence process of reinforcement
           learning.Theself-supervisedlearningapproachisdiscussed(Section3.4)totackle[Q3]beforewe
           presenthowtotackleparameterexplosionandoutlinetheholisticalgorithmworkflow(Section3.5
           and Section 3.6).
    Symbol      Definition
𝑈𝑖;𝑇𝑖;𝐶𝑖;𝐻𝑖3.1   OverviewUser;tweet;comment;hashtag
 G;G𝑖;V;EFig.3depictstheoverallarchitectureof RoSGAStoperformthesubgraphembedding.TheworkflowGraph;𝑖-thsubgraph;nodeset;edgeset
        F;Rmainlyconsistsofthreeparts:graphpreprocessingandconstruction,RL-basedsubgraphembeddingNodetype;relationtype
         𝜑;𝜙and the final-stage attention aggregation mechanism among graph nodes before feeding into theMappinganodetothetypeF;mappingaedgetothetypeR
      V𝑖;X𝑖finalclassifier fordetermining theexistence ofsocial bots.Toaid discussion,Table 1depictstheNodesetofsubgraphG𝑖;nodefeaturesetofsubgraphG𝑖
   𝑣𝑖0;𝑣𝑖𝑛 ;𝑦𝑖notations used throughout the paper.𝑖-thtargetuser;𝑛-thnodeinSubgraphG𝑖;labelof𝑖-thtargetuser
   𝑒𝑟𝑣𝑖𝑚,𝑣𝑖𝑛 ;𝑟 Anedgeconnecting𝑣𝑖𝑚,𝑣𝑖𝑛  througharelationship𝑟∈R
          𝑘;𝑙   ThenumberofneighborhopsandthenumberofmodellayersACM Trans. Web, Vol. 1, No. 1, Article 1. Publication date: January 2022.
   D;ℎ𝑗;𝑚𝑖      Thetargetuserset;theoriginalfeatureof𝑣𝑗;themodelfortargetuser𝑣𝑖0
         𝐿;L    Thelastlayernumberandtheerrorcorrectionparameter
     𝛼𝑘𝑖𝑗;𝑊𝑘    The𝑘-thattentioncoefficientandthe𝑘-thweightmatrix
        𝑧𝑖;𝜋    Theembeddingofthe𝑖-thsubgraphandthepolicynetwork
     𝑠𝑡;𝑎𝑡;𝑟𝑡   Thestate,choseaction,andtherewardattimestamp𝑡
         R,𝑏    Theadvantagefunctionmeasurementof(𝑠𝑡,𝑎𝑡)andthehistorywindowsize
         𝑄,𝛾    Thevalueofstate-actionpairandthefuturecumulativerewardsdiscountparameter
         𝐵;𝜏    Theobservationexperiencesetandthestate-actionpair
        𝑑𝜏,Θ    ThestatedistancemeasurementfunctionandtheparameterofRLagent
        𝛼0,𝛽    Theinitialweightanddecayrateof𝛼inthenearestneighbormechanism
            𝜆   TheweightparameteroftheGNNlossfunction
          G𝑘𝑖   AsubgraphGfor𝑖-thtargetuserwith𝑘hopneighbors
L𝑝𝑟𝑒𝑡𝑒𝑥𝑡𝑖,𝑔𝑖    Thelossofthe𝑖-thpretexttask,the𝑖-thstackedGNNencoder
    𝑦𝑝𝑟𝑒𝑡𝑒𝑥𝑡𝑖   the𝑦𝑝𝑟𝑒𝑡𝑒𝑥𝑡𝑖standsforthegroundtruthofsubgraphG𝑖acquiredbypretexttask
      G′𝑖,  ¯G𝑖 G′𝑖isapositivesampleforG𝑖.  ¯G𝑖isanegativesampleforG𝑖.
        𝑧′𝑖,¯𝑧𝑖 𝑧′𝑖istherepresentationofG′𝑖, ¯𝑧𝑖istherepresentationof  ¯G𝑖

1:8                                                                                                                                                                               Yang, et al.
                                    Fig. 3. The proposedRoSGASframework.
3.1.1   GraphConstruction. Initially,thefeatureextractionmoduletransformstheoriginalinfor-
mation into a heterogeneous graph. The edges between nodes in the heterogeneous graph are
established based on the account’s friend relationships and interactions in the social network
platform. Weretrieve themeta features anddescription features ofeach account asits initial node
feature in a similar way as [56]. Extra tweet features and entities are extracted by using NLPtool1
from the original tweets. The composition of features for each type of node may vary. For user
node,featuressuchas status, favorites, list,etc.areextractedfromusermetadata.For tweet node,
wenormallyextractthenumberofretweetsandthenumberofreplies,whilstembeddingthetweet
content into a 300-dimension vector and concatenating them together as their original features.
Similarly,weembedthecontentof hashtag and entity to300-dimensionvectors.Tosimplifythe
featureextractionandprocessing,thefeaturevectorofeachnodetypeissettobe326dimensions.
Those with insufficient dimensions are filled with zero. More details will be given in Section 4.3.
   To refine the heterogeneous graph under the given semantics, we further conduct a graph
pre-processingbyenforcingmeta-pathsupontheoriginalgraphandonlytheentitiesandedges
conformingthe givenmeta pathswill be retainedin thegraph structure.As shownin Fig.2, we
extracted five meta-paths that are widely-recognized in social graphs and represent most of the
typicalbehaviorsinthe meta-schemadefinedinSection2.1.Themain purposeistocutdownthe
information redundancies in the graph at scale and thus substantially improve the computational
efficiency. The transformed graph will be further used to extract subgraphs for the target users
beforefeedingthesubgraphsintothedown-streamingtasksincludingthenumericalembedding
and classification.
3.1.2   RL-basedParameterSearching. Theprimarygoaloftheparameterselectionistodetermine
the appropriate subgraph width (𝑘) and the model layers (𝑙) for each target user in the target user
collectionDwith𝑛users. In a nutshell, for each target user, we take it as the center node and first
extractafixedwidth(e.g.,1hop)subgraphG𝑖astheinitialsubgraph.Afterwards,weencodethe
subgraph into the embedding space and regard the embedding as the environment state before
feedingitintothereinforcementlearningagent.Tobemorespecific,theembeddingrepresentation
ofthe𝑖-thtarget usercanbeobtainedby usinganencoder (e.g.,averageencoding operation,sum
1https://github.com/explosion/spaCy
ACM Trans. Web, Vol. 1, No. 1, Article 1. Publication date: January 2022.

RoSGAS: Adaptive Social Bot Detection with Reinforced Self-Supervised GNN Architecture Search                               1:9
encoding operation, etc.):
                                           𝑒(G𝑖)={ℎ𝑗|𝑣𝑗∈V𝑖(G𝑖)}.                                                    (1)
   AtthecoreoftheembeddingimprovementistheRLagent.Thepreliminaryencodingresultwill
be fed into the policye𝜋1 ande𝜋2 in the RL agent, successively.e𝜋1 is responsible for selecting the
appropriate width ofsubgraphG′𝑖whilee𝜋2 is in chargeof pinpointing themost suitable number
of layers for constructing model𝑚𝑖for the target user𝑣𝑖0. The types of specific model layer can
beselectedfromthemostpopularmodelssuchasGCN[26],GAT[48]andGraphSAGE[23],etc.
Generally speaking, the goal of reinforcement learning is to maximize the expected accuracy:
E[RD({𝑚𝑖,G𝑖}|𝑛𝑖= 1)]onD:
                       e𝜋∗1,e𝜋∗2 =  argmax E[RD({𝜋1(𝑒(G𝑖);𝜃1),𝜋2(𝑒(G𝑖);𝜃2)}|𝑛𝑖= 1)].                         (2)
                                    𝜃1,𝜃2
More details about the learning procedure will be discussed in Section 3.2.
   To calibrate the subgraph embeddings, we can then aggregate the𝑘-hop neighbors and stack
modelswithvarious𝑙layers.However,theaggregationfromthestackedGNNmodelswouldblur
theoriginaldetectablefeaturesofthetargetuserinthesubgraphembedding.Tomitigatethisissue,
we apply the residual network to aggregate the target node’s input features and its corresponding
embedding delivered by the last layer of the model:
                                             ℎ(𝐿)𝑗  =𝐴𝐷𝐷(𝑥𝑖𝑗,ℎ(𝐿)𝑗),                                                       (3)
where𝐿is thelast layer of thestacked GNN model.Then we can applya pooling operation(e.g.,
average, sum, etc.) to integrate the subgraphG𝑖into a super node:
                                         𝑧(𝐿)𝑖   =𝑅𝐸𝐴𝐷𝑂𝑈𝑇({ℎ(𝐿)𝑗}𝑛′𝑗= 1).                                                 (4)
3.1.3   AttentionAggregationandClassification. Weadoptanattentionmechanismforintegrating
the influence of subgraphs belonging to the relevant neighbors into the final embedding:
                                                     𝐾∑︁ ∑︁
                                           𝑧𝑖=   1𝐾           𝛼𝑘𝑖𝑗Wk𝑧(𝐿)𝑗,                                                   (5)
                                                    𝑘= 1G𝑗∈G
where𝛼𝑖𝑗is the attention coefficient,W∈R dL×dl is the weight matrix and𝐾 is the number of
independent attention.𝑧𝑖is the final embedding for detecting if the target user is a social bot.
Eventually, the bot classifier digests the learned vector embeddings to learn a classification model
anddeterminesifagivensocialuserbehaviorsnormallyormaliciously.Generalpurposetechniques
including Random Forest, Logistic Regression, SVM, etc. can be adopted for implementing the
classifier.
3.2   Reinforced Searching Mechanism
In this subsection, we will introduce how to obtain the optimal policies  ˜𝜋∗1 and  ˜𝜋∗2 through the
searching mechanism. The learning procedure of the optimal ˜𝜋∗1 and ˜𝜋∗2 can be formulated as a
Markov Decision Process (MDP). An RL agent episodically interacts with the environment where
eachepisodelastsfor𝑇 steps.TheMDPincludesstatespace,actionspace,rewardandthetransition
probability that maps the current state and action into the next state.
StateSpace.Ineachtimestamp𝑡,thestate𝑠𝑡isdefinedastheembeddingofthesubgraphextracted
fromG.
Action Space. Since we need two policies to pinpoint the optimal width of subgraph and the
optimal number of model layers respectively, the action at timestep𝑡consists of dual sub-actions
(𝑎(1)𝑡,𝑎(2)𝑡). The RL agent integrated in our proposed framework RoSGAS performs an action
                                               ACM Trans. Web, Vol. 1, No. 1, Article 1. Publication date: January 2022.

1:10                                                                                                                                                                             Yang, et al.
𝑎(1)𝑡   to get the value of𝑘, and performs an action𝑎(2)𝑡   to get the value of𝑙. For instance,𝑎(1)𝑡   is
chosen at the timestep𝑡to re-extract the subgraph of the target user𝑣𝑖0. We then calculate the
number of reachable paths from target user𝑣𝑖0 to the other target users in this subgraph as the
connection strength. Forthosetarget usersthatare includedinthe collectionDyetexcludedfrom
this subgraphs, the connection value will be set to 0.𝐿1 normalization is performed upon these
valuesasthereachabilityprobabilitiesfromthetargetusertotheothertargetusers.Afterselecting
certainactions(𝑎(1)𝑡,𝑎(2)𝑡)atthetimestep𝑡,theRLenvironmentformsaprobabilitydistribution𝑃𝑖.
Transition. The probability𝑃𝑖serves as the state transition probability of the reinforcement
learningenvironment.Thesubgraphembeddingofanyothertargetuserisusedasthenextstate𝑠𝑡+1.
The whole trajectory of the proposed MDP can be described as(𝑠0,(𝑎(1)0 ,𝑎(2)0),𝑟0,...,𝑠𝑇−1,(𝑎(1)𝑇−1,
𝑎(2)𝑇−1),𝑟𝑇−1,𝑠𝑇).
Reward.Weneedtoevaluatewhetherthesearchmechanismisgoodenoughatthecurrenttimestep
𝑡.Inotherwords,itreflectsiftheparametersinthecurrentRLagentcanachievebetteraccuracy
than the parametersat the previoustimestep𝑡−1. Todo so, we firstly define a measure to flag the
improvement of model accuracy when compared with the previous timesteps:
                                                               Í𝑡−1
                            R(𝑠𝑡,𝑎𝑡)=(ACC(𝑠𝑡,𝑎𝑡)−                 𝑖=𝑡−𝑏ACC(𝑠𝑖,𝑎𝑖)
                                                                        𝑏−1   ),                                (6)
where𝑏is a hyperparameter that indicates the window size of historical results involved in the
comparison andACC(𝑠𝑖,𝑎𝑖)is theaccuracy ofsubgraph classificationon thevalidation set attheÍ 𝑡−1
timestep𝑖.     𝑖=𝑡−𝑏ACC(𝑠𝑖,𝑎𝑖)𝑏−1           reflectstheaverageaccuracyinthemostrecent𝑏timestepwindows.
The training RL agent continuously optimizes the parameters to enable a rising accuracy and
accordinglypositive rewards.Thiswill give risetothe cumulativerewardsin finitetimestepsand
eventually achieve the optimal policies.
   We use a binary reward𝑟𝑡combined with Eq.6 to navigate the training direction as follows:   1,    ifR(st,at)>R(st−1,at−1)
                              𝑟(𝑠𝑡,𝑎𝑡)=       −1,                            otherwise.                                     (7)
The valueis set to be1 if𝑎𝑡can increasetheRcompared withthat of the previous timestep𝑡−1;
otherwise it will be set -1.
Termination. State-action values can be approximated by the Bellman optimal equation:
                                                                                    ′).                                    (8)
                               𝑄∗(𝑠𝑡,𝑎𝑡)=𝑟(𝑠𝑡,𝑎𝑡)+𝛾 arg max
                                                                  𝑎′ 𝑄∗(𝑠𝑡+1,𝑎
Nevertheless,toimproveboth trainingspeed andstability,we willintroducean enhancedapproxi-
mation approach in Section 3.3. We exploit the𝜖-greedy policy to select the action𝑎𝑡with respect
to𝑄∗and obtain the policy function𝜋:
                                                (     random action, 𝑤.𝑝.𝜖
                              𝜋(𝑎𝑡|𝑠𝑡;𝑄∗)=        argmax
                                                      𝑎  𝑄∗(𝑠𝑡,𝑎),  otherwise.                                   (9)
3.3   Nearest Neighbor Mechanism for Accelerating Model Stabilization
Conventionally, at each time step𝑡, the RL agent employs its prediction network to determine
the value of state-action pairs for choosing the best action to maximize the future cumulative
rewards.Inspiredby[41],weappliedthenearestneighbormechanismforassistingandaccelerating
the training process of the RL agent. The intuition behind the scheme is that when the RL agent
observessimilarorthesamestate-actionpairs,theenvironmentishighlylikelytoproduceasimilar
reward value. In other words, the distance between state-action pairs can indicate their relative
ACM Trans. Web, Vol. 1, No. 1, Article 1. Publication date: January 2022.

RoSGAS: Adaptive Social Bot Detection with Reinforced Self-Supervised GNN Architecture Search                             1:11
rewardvalues.Therefore,weaimtofindoutthesimilarstate-actionpairsanddeterminethereward
of the current state-action pair by combining the reward estimated by the Q-network (i.e., the
predictionnetwork)andtherewardoftheexistingsimilarpairs.Thismeansthatthemodeltraining
benefits from both the RL environment and the prediction network, which can boost the action
exploration and accelerate the training stabilization.
   To look into and record the historical actions, we set up an observation experience set𝐵 =
{𝜏1,...,𝜏𝑛}; each element𝜏𝑖in𝐵 represents a pair of explored state𝑠𝑖and the corresponding
selected action𝑎𝑖, namely,(𝑠𝑖,𝑎𝑖). While recording the action-state pairs, we also record the
corresponding value labels{𝑄(𝜏𝑖)}⊆R . We employ the distance function𝑑𝜏– for example using
cosine tocalculate thesimilarity –to measure thedistance between theexploredstate-action pairs
andtheincomingstate-actionpairs.Weusethedistancetoascertainthenearestneighborofthe
state-actionpair tobeestimated from therecorded state-actionpairs. Subsequently,thevalue label
of the nearest neighbor can be used to estimate the value of the state-action pair:
                                     ˆ𝑄(𝜏)=  min{Q(𝜏i)+L·d𝜏(𝜏,𝜏i)}|ni= 1,                                          (10)
where  ˆ𝑄(𝜏)is the estimated value of𝜏, and L is a parameter to correct the estimated error.
   We combine  ˆ𝑄(𝜏)estimated by the nearest neighbor mechanism and𝑄(𝑠𝑡,𝑎𝑡;Θ𝑡𝑎𝑟𝑔𝑒𝑡)estimated
by the target network into a new estimated value  ˆ𝑄(𝑠𝑡,𝑎𝑡):
                   ˆ𝑄(𝑠𝑡,𝑎𝑡)=𝛼·ˆ𝑄(𝜏𝑡)+(1−𝛼)·(𝑟+𝛾max𝑎𝑡+1𝑄(𝑠𝑡+1,𝑎𝑡+1;Θ𝑡𝑎𝑟𝑔𝑒𝑡)),                 (11)
where𝛼 is an exponentially decaying weight parameter and  ˆ𝑄(𝜏𝑡)is the estimated value of
𝜏𝑡 =(𝑠𝑡,𝑎𝑡)by using the nearest neighbor search mechanism. In fact,𝛼 is used to assist the
RLoptimizationintheearlystage,andgraduallyreducetheeffectoftheproposednearestneighbor
mechanismwhen thetraining proceduremovesforward. To achieve this,weset𝛼=𝛼0·(1−𝛽)𝑘,
where𝛼0∈(0,1]is an initial weight;𝛽∈(0,1)is a decay rate and𝑘is the episode number. We
then define the RL loss function as:
                                   𝐿(Θ)=(ˆ𝑄(𝑠𝑡,𝑎𝑡)−𝑄(𝑠𝑡,𝑎𝑡;Θ𝑝𝑟𝑒𝑑))2,                                         (12)
whereΘ𝑝𝑟𝑒𝑑is the parameters of the DQN agent’s prediction network, and the Θ𝑡𝑎𝑟𝑔𝑒𝑡is the
parameters of the target network.
3.4   Self-supervised Learning
Tobetter differentiatethesubgraph representationsamong graphs,wepropose acontrastiveself-
supervisedlearningapproachtomaximizethedifferencebetweentwodistinctpatches,without
relying onthe human-annotated data samples.The task of self-supervised learning (alsoknown as
pretext task) is to minimize the distance between positive samples whilst maximizing the distance
betweennegativesamples.Inthecontextofthiswork,allsubstructures(e.g.,subgraphswith1-hop
neighbors or 2-hop neighbors) pertaining to the same user should have similar representation
vectors.Subgraphsbelongingtotwodistincttargetusersshouldhavediscriminativeembeddings.
   In general, given a subgraphG𝑖={V𝑖,X𝑖,{E𝑟}|𝑅𝑟= 1,𝑦𝑖}, the lossL𝑝𝑟𝑒𝑡𝑒𝑥𝑡for a self-supervised
learning task can be defined as follows:
                                 L𝑝𝑟𝑒𝑡𝑒𝑥𝑡(A𝑖,X𝑖,𝑔𝑖)=𝜙(𝑔𝑖(G𝑖),𝑦𝑝𝑟𝑒𝑡𝑒𝑥𝑡𝑖),                                      (13)
where𝑔𝑖is the stacked GNN encoder for the extracted subgraphG𝑖and the𝑦𝑝𝑟𝑒𝑡𝑒𝑥𝑡𝑖 stands for the
ground truth of subgraphG𝑖acquired by a specific self-supervised pretext task.
   Practically,thekeystepistogeneratepositivesamplesinourself-supervisedpretexttask.After
the RLalgorithm outputs thecustomized width value𝑘for thetarget node𝑣𝑖, werandom select a¯𝑘∈[1,𝐾]!=𝑘asthewidthofanewsubgraphG′𝑖,toserveanewpositivesampleoftheoriginal
                                              ACM Trans. Web, Vol. 1, No. 1, Article 1. Publication date: January 2022.

1:12                                                                                                                                                                             Yang, et al.
subgraphG𝑖. Meanwhile, to provide the negative-sampled counterparts, we randomly select the
target user𝑣𝑗from the target user setDand directly use the learnt value𝑘. We use the stacked
GNNencoder𝑔𝑖andthe proposed RLpipelinestoperformfeatureextractionandsummary,and
obtain the final subgraph embedding𝑧𝑖,𝑧′𝑖, ¯𝑧𝑖for the subgraphsG𝑖,G′𝑖,  ¯G𝑖, respectively. Then we
use themargin Triplet lossfor model optimizationto obtainhigh-quality representationsthat can
well distinguish the positive and negative samples. The loss function is defined as follows:
                                L𝑝𝑟𝑒𝑡𝑒𝑥𝑡𝑖 =−𝑚𝑎𝑥(𝜎(𝑧𝑖𝑧′𝑖)−𝜎(𝑧𝑖¯𝑧𝑖)+𝜖,0),                                    (14)
where𝜖isthemarginvalue.Thelossfunctionofthepretexttaskwillbeincorporatedintotheholistic
loss functionas the optimizationobjective of RoSGAS. Sincethere may exist manyoverlapping
nodesbetweendifferentsubgraphs,especiallyinalarge-scalesocialgraph,theadoptionoftheloss
function can effectively avoid excessive differentiation between positive and negative samples and
prevent from any performance degradation of representation.
3.5   Parameter Sharing and Embedding Buffer Mechanism
Thecustomizedmodelconstructionforeachindividualtargetuserwillleadtoasubstantialnumber
of training parameters. We use the following two schemes to alleviate this problem.
    •ParameterSharing:Wefirstdetermineamaximumbaselayernumber𝑘toinitializethemodel,
       and then repeatedly stack the whole or part of these layers according to𝑎𝑡output from the
       RLagentineachtimestep𝑡intheinitializationorder.Thiscanavoidtrainingalargenumber
       of model parameters.
    •Embedding Buffer:Webuffertheembeddingsoftherelevantsubgraphsasabatchtocarry
       out𝑎(2)𝑡   ineachtimesteptoreduceunnecessaryoperationsforembeddingpassing.Oncethe
       bufferspaceapproachesthespecifiedbatchsize,themodelre-constructionwillbetriggered
       byleveragingtheobtainednumberoflayersfrom𝑎(2)𝑡   andadoptingthebufferedembeddings.
       We cleanse the buffer space once the GNN model training terminates to ensure the buffer
       can be refilled in the later stage.
3.6   Put Them Together
Algorithm 1 summarizes the overall training process of the proposed RoSGAS including the
initializationofthesubgraphembeddingandthefollow-uparchitecturesearchviatheproposedRL
process. Specifically, We first construct the social graph according to our definition in Section 2.1
before initializing theGNN model with themax layers𝐿and the parametersof the two RLagents
(Line1).Atthetrainingstage,werandomlysampleatargetuserandembedits𝑘𝑖𝑛𝑖𝑡-hopsubgraph
astheinitialstate𝑠0 (Lines2-4).Afterwards,ateachtimestep,anactionpairwaschosentoindicate
thewidth𝑘ofthesubgraphandthenumber𝑙ofGNNlayersforthetargetuser(Line6).Thenwe
                          ′          ′
re-extract subgraphG𝑡, storeG𝑡and thevalue𝑙represented by𝑎(2)𝑡   into thebufferB(Lines 7-9).
                           ′
Once the number ofG𝑡reaches a threshold value𝐵𝐷, we stack the GNN model with𝑎(2)𝑡   layers,
andgeneratethepositiveandnegativepairforthe𝑏-thtargetuserinB.Thenwetrainthemodel
togetherwiththeself-supervisedlearningmechanismdescribedinSection3.4(Lines11-15).After
thestackedGNNmodeltraining,wevaliditonthevalidationdatasettogetthereward𝑟𝑡(Line18)
and store the corresponding transition into memoryM(1)andM(2)(Lines 20-21).
  To optimize the𝜋1, we fetch batches of transitions fromM(1). For a specific transition𝑇(1)𝑡   =
(𝑠𝑡,𝑎(1)𝑡,𝑠𝑡+1,𝑟𝑡),weusethe Q-networktoselectthebestaction𝑎(1)𝑡+1 forthestate𝑠𝑡+1 andusethe
target network to estimate the target value𝑄𝑡𝑎𝑟𝑔𝑒𝑡=𝑟+𝛾max𝑎𝑡+1𝑄(𝑠𝑡+1,𝑎𝑡+1;Θ𝑡𝑎𝑟𝑔𝑒𝑡). Then we
use nearest neighbor mechanism to search the nearest neighbor of state-action pair(𝑠𝑡,𝑎(1)𝑡)and
ACM Trans. Web, Vol. 1, No. 1, Article 1. Publication date: January 2022.

RoSGAS: Adaptive Social Bot Detection with Reinforced Self-Supervised GNN Architecture Search                             1:13
  Algorithm1: The overall process of RoSGAS
   Input: Themaxneighborhopnumber,layernumber:𝐾 and𝐿;initialneighborhopnumber
            𝑘𝑖𝑛𝑖𝑡,thebatchsizeofGNNandDQN:𝐵𝐺,𝐵𝐷,DQNtrainingstep𝑆,thetotaltraining
            epoch𝑇, epsilon value𝜖, the window size𝑏, the error correction parameters L, the
            initial decay parameter𝛼0, the decay rate𝛽,the full graphG, targeted node set𝑉.
 1 Initialize𝐿GNN layers, RL agent networks𝜋1,𝜋2;
 2 Initialize the memory bufferM(1),M(2), and the GNN bufferB;
 3 Sample a target node, extract𝑘𝑖𝑛𝑖𝑡-hop subgraphG0;
 4𝑠0 =𝑒(G0)via Eq. 1;
 5 for𝑡=  0,1,...,𝑇do
 6   𝑎𝑡=(𝑎(1)𝑡,𝑎(2)𝑡)via Eq. 9;
                                    ′          ′
 7       Re-extract subgraphG𝑡and𝑒(G𝑡)via Eq. 1;
                                                           ′
 8       Sample meta-path instances to get newG𝑡;
                 ′
 9       StoreG𝑡and𝑎(2)𝑡   into bufferB;
10       if size(B)>𝐵𝐷 then
11             Stack𝑎(2)𝑡   layers GNN ;
12            for𝑏=  1,...,𝐵𝐺 do
13                   Generate positive and negative pair for the𝑏-th target user inB;
14                   Train the stacked model on the buffer of action𝑎(2)𝑡   via Eq. 15;
15            end
16             Clear the buffer for𝑎(2)𝑡   inB;
17       end
18       Obtain𝑟𝑡on validation dataset via Eq. 6;
19       Jump to the next subgraphG𝑡+1 and𝑒(G𝑡+1);
20       Store the𝑇(1)𝑡   =(𝑠𝑡,𝑎(1)𝑡,𝑠𝑡+1,𝑟𝑡)intoM(1);
21       Store the𝑇(2)𝑡   =(𝑠𝑡,𝑎(2)𝑡,𝑠𝑡+1,𝑟𝑡)intoM(2);
22       for step=1,...,S do
23             Optimize𝜋1 and𝜋2 via Eq. 12 ;
24       end
25 end
26  Re-init GNN to train via Eq. 15 withe𝜋∗1,e𝜋∗2 ;
add itsreward valueupon the valueof𝑄𝑡𝑎𝑟𝑔𝑒𝑡to obtaina new targetvalue  ˆ𝑄(𝑠𝑡,𝑎𝑡). Thenwe can
optimizethe𝜋1 throughEq.12.Thismethodisalsoappliedtooptimize𝜋2 (Lines22-24).Eventually,
we retrain the GNN with the help of the trained policies𝜋1 and𝜋2 (Line 26). The final embedding
𝑧𝑖of eachtargeted userwill beproduced bythe lastattention layer andused forthe downstream
classification task.
   We combine the the loss function of GNN and the self-supervised lossL𝑝𝑟𝑒𝑡𝑒𝑥𝑡in Eq.13. The
lossLof RoSGAS is defined as follows:
                                𝑛∑︁
                         L=        (−log(𝑦𝑖·𝜎(𝑀𝐿𝑃(𝑧𝑖)))+L𝑝𝑟𝑒𝑡𝑒𝑥𝑡𝑖)+𝜆∥Θ∥2,                         (15)
                               𝑖= 1
                                                ACM Trans. Web, Vol. 1, No. 1, Article 1. Publication date: January 2022.

1:14                                                                                                                                                                             Yang, et al.
                Dataset               Nodes         Edges       Benign    Bots    Labels   Un-Labels
               Cresci-15            2,263,472   10,782,235     1,950     3,339     5,289       99.77%
                Varol-17            1,978,967    4,916,116      1,244      639      1,883       99.90%
              Vendor-19             3,208,255   11,479,317     1,893      569      2,462       99.92%
               Cresci-19              669,616      3,341,084       269        297       566         99.92%
        Botometer-Feedback            468,536      1,333,762       276         82        358         99.92%
                                           Table 2. Statistics of datasets.
where thefirst term represents the cross-entropyloss function and∥Θ∥2 is the𝐿2-norm ofGNN
modelparameters,𝜆isaweightingparameter.𝑀𝐿𝑃 reducetheembeddingdimensionof𝑧𝑖tothe
number of categories.
4   EXPERIMENTAL SETUP
4.1   Software and Hardware
We implement RoSGAS with Pytorch 1.8.0, Python 3.8.10, PyTorch Geometric [19] with sparse
matrixmultiplication.AllexperimentsareexecutedonaseverwithanNVIDIATeslaV100GPU,
2.20GHz Intel Xeon Gold 5220 CPUwith 64GB RAM. The operating system is Ubuntu 16.04.6. To
improvethe trainingefficiencyand avoidoverfitting, we employ themini-batch technique totrain
RoSGAS and other baselines.
4.2   Datasets
We build heterogeneous graph for experiments based upon five public datasets. The detailed
description of these datasets is as follows:
    •Cresci-15 [8] encompasses two types of benign accounts, including a) TFP, a mixture of
       accountsetfromresearchersandsocialmediaexpertsandjournalists,andb)E13,anaccount
       set consists of particularly active Italian Twitter users. Three types of social bots were
       collected from three different Twitter online markets, called FSF, INT, TWT.
    •Varol-17 [47] collects 14 million accounts of Twitter during three months in 2015. 3,000
       accountsaresampledandselectedaccordingtosomegivenrules.Theseaccountsarethen
       manually annotated into benign accounts and bot accounts.
    •Vendor-19 [55] include a collection of fake followers deriving from several companies. To
       create a mixture of benign and bot accounts, we mix Vendor-19 with Verified [56] that
       contains benign accounts only.
    •Cresci-19[32]containstheaccountsthatareassociatedwithItalianretweet,collectedbetween
       17-30 June 2018.
    •Botometer-Feedback [55] stems from social bot detector Botometer. The dataset contains
       manually-annotated accounts based on the feedback from Botometer.
   ThestatisticsofthesedatasetsareoutlinedinTable2.Thenumberofeachclassoflabellednodes
in each data set are basically balanced.
4.3   Feature Extraction
Theoriginaldatasetsabovemerelyincludethemetadatasuchasage,nickname,etc.andtheposted
tweetofthesocialaccount.Thisinformation,however,isinsufficienttoconstructtheheterogeneous
graph,requiredforthe effectivesubgraph embedding.These publiclyreleased datasetsoriginally
included the social accounts’ metadata (e.g., accounts’ age, nickname) and the account’s posted
ACM Trans. Web, Vol. 1, No. 1, Article 1. Publication date: January 2022.

RoSGAS: Adaptive Social Bot Detection with Reinforced Self-Supervised GNN Architecture Search                             1:15
tweet data. Since these public data are not enough to construct the heterogeneous graph we
designed,weusetwitterAPIstofurthercrawlandobtainthethemetadataandtweetdataofthe
friends and followers of the original accounts. We form these nodes into a huge heterogeneous
socialgraphviathemultiplerelationshipsaforementionedinFig.2.WethenusetheNLPtoolkit
spaCy to extract name entities and treat them as a type of node in the heterogeneous graph.
  Inaddition,weextractedtheoriginalfeaturevectorforeachtypeofnodeintheheterogeneous
graph and further explored the following information as additional features:
    •Account nodes: Weembedthe description fieldofthe accountmetadataintoa 300-dimension
      vectorthroughthepre-trainedlanguagemodelWord2Vec.Wealsoextractedsomefeatures
       suchasstatus,favorites,andlist fieldthatwouldbehelpfulforbotdetectionaccordingto[56].
      We count the number of followers and the number of friends as the key account features,
       becausethenumberoffollowersandfriendsarethemostrepresentativeofaTwitteruser,and
       usingthemcouldmoreefficientlyandaccuratelydescribeaTwitteruser.Inaddition,wedivide
       the numbers by the user account lifetime (i.e., the number of years since the account was
       created)toreflectthechangesduringthewholelife-cycleofanaccount.Wealsouseboolean
      value to flag the fields includingdefault_profile, verified and profile_use_background_image,
       count the length of screen_name, name and description fields and the number of digits in
       screen_name and name fields. We combine all these values as the original node features.
    •Tweet nodes: We embed the text of the original tweet by the pre-trained language model
      Word2Vec into a 300 dimension vector. Apart from the original node features, we further
       combineadditionalinformation–thenumberofretweets,thenumberofreplies,thenumber
       of favorites, the number of mentioning of the original tweet, and the number of hashtags
       and the number of URLs involved in the tweet.
    •Hashtag and entity nodes: wealsoembedthetextofhashtagandentityintoa300-dimension
      vector. We use zero to fill the blank holes if the number of dimension is less than 300.
  The graph constructed by each data set contains millions of edges and nodes, which greatly
increasesthedifficultyofthesocialbotdetectiontask.Noticeably,mostsamplesinthedatasetsare
unlabelled, i.e., more than 99% samples are not annotated.
4.4   Baselines and Variations
4.4.1   Baselines. To verifythe effectiveness of ourproposedRoSGAS, Wecomparewith various
semi-supervised learning baselines. Because these baselines will run on very large-scale graphs,
to ensure training and inference on limited computing resources, we use the PyG [19] which
calculationon sparsematrixmultiplication toimplementthese baselines.Thedetail aboutthese
baselines as described as follows:
    •Node2Vec[22]isbuiltuponDeepWalkandintroducestwoextrabiasedrandomwalkmeth-
       ods, BFS and DFS. Compared with the random walk without any guidance, Node2Vec sets
       differentbiasestoguidetheprocedureoftherandomwalkbetweenBFSandDFS,representing
       the structural equivalence and homophily at the same time.
    •GCN [26] is a representative of the spectral graph convolution method. It uses the first-
       orderapproximationoftheChebyshevpolynomialtofulfillanefficientgraphconvolution
       architecture. GCN can perform convolutional operation directly on graphs.
    •GAT [48] is a semi-supervised homogeneous graph model that utilizes attention mechanism
       for aggregating neighborhood information of graph nodes. GAT uses self-attention layers
       tocalculatetheimportanceofedgesandassigndifferentweightstodifferentnodesinthe
       neighborhood. GAT also employs a multi-head attention to stabilize the learning process of
       self-attention.
                                             ACM Trans. Web, Vol. 1, No. 1, Article 1. Publication date: January 2022.

1:16                                                                                                                                                                             Yang, et al.
    •GraphSAGE [23] is a representative non-spectrogram method. For each node, it samples
       neighborsindifferenthopsforthenodeandaggregatesthefeaturesoftheseneighborsto
       learntherepresentationforthenode.GraphSAGEimprovesthescalabilityandflexibilityof
       GNNs.
    •GraphSAINT [59] is an inductive learning method based on graph sampling. It samples
       subgraphsandperformsGCNonthemtoovercometheneighborexplosionproblemwhile
       ensuring unbiasedness and minimal variance.
    •SGC[51]isasimplifiedgraphconvolutionalneuralnetwork.Itreducestheexcessivecom-
       plexityofGCNsbyrepeatedlyremovingthenon-linearitybetweenGCNlayersandcollapsing
       the resultingfunction intoa single lineartransformation. Thiscan ensure competitiveper-
       formance when compared with GCN and significantly reduce the size of parameters.
    •ARMA[5]isanon-linearandtrainablegraphfilterthatgeneralizestheconvolutionallayers
       based on polynomial filters. ARMA can provide GNNs with enhanced modeling capability.
    •Policy-GNN [27]is ameta-policyframeworkthat adaptivelylearnsan aggregationpolicy
       to sample diverse iterations of aggregations for different nodes. It also leverages a buffer
       mechanism for batch training and a parameter sharing mechanism for diminishing the
       training cost.
4.4.2   Variants. WegenerateseveralvariantsofthefullRoSGASmodel,tomorecomprehensively
understandhoweachmoduleworksintheoveralllearningframeworkandbetterevaluatehoweach
moduleindividuallycontributetotheperformanceimprovement.RoSGASmainlycomprisesthree
modules: Reinforced Searching Mechanism, Nearest Neighbor Mechanism, and self-supervised
learning. We selectively enable or disable some parts of them to carry out the ablation study.
  The details of these variations are described as follows:
    •RoSGAS-K:Thisvariantonlyenablesthereinforcedsearching,withouttheaidofanyother
       modules, tofind outthe width(𝑘) ofthe subgraphfor everytarget user𝑣𝑖. Dueto thehuge
       scale of the constructed social graph, the search range will be limited to[1,2]to prevent
       the explosion of neighbors. Nevertheless, such search range can be flexibly customized to
       adapttoanyotherscenariosanddatasets.Inthecontextofthismodelvariant,thenumberof
       layers is fixed to be𝑙=  3.
    •RoSGAS-L:Thisvariantonlyswitchesonthereinforcedsearchingmechanismforpinpoint-
       ingthenumberofthelayers(𝑙)tostacktheGNNmodelforeverytargetuser𝑣𝑖.Thesearch
       rangewillbelimitedto[1,3]tosavecomputingresources.Thewidth𝑘ofthesubgraphis
       fixed𝑘=  2.
    •RoSGAS-KL:Thisvariantenablesboththesubgraphwidthsearchandthelayersearchin
       thereinforcedsearchingmechanismforeachtargetuser𝑣𝑖.Inthismodelvariant,thewidth
       is set to be𝑘∈[1,2]while the number of layers of the GNN model is set to be𝑙∈[1,3].
    •RoSGAS-KL-NN: This variant utilizes the reinforced search mechanism, together with
       the nearest neighbor mechanism. In the optimization process of the RL agent, the nearest
       neighbor module can stabilize the learning in the early stages of RL as soon as possible, and
       accelerate the model convergence.
    •RoSGAS contains all modules in the learning framework. The self-supervised learning
       mechanism is additionally supplemented uponRoSGAS-KL-NN.
4.5   Model Training
Weusethefollowingsetting:embeddingsize(64),batchsize(64),thebaselayerof RoSGAS(GCN),
learningrate(0.05),optimizer(Adam),L2regularizationweight(𝜆2 =  0.01),andthetrainingepochs
(30).AsaforementionedinSection4.4.2,wesettheaction(range)ofsearchingGNNlayersfrom
ACM Trans. Web, Vol. 1, No. 1, Article 1. Publication date: January 2022.

RoSGAS: Adaptive Social Bot Detection with Reinforced Self-Supervised GNN Architecture Search                             1:17
1 to 3 and the action (range) of subgraph width searching from 1 to 2 to prevent neighbors from
explodingfortheDQN[33].Theagenttrainingepisodesissettobe20andweconstruct5-layerof
MLP with 64,128, 256, 128, 64 hiddenunits. We use theaccuracy obtained from the validationset
to select thebest RL agent andcompare the performancewith the other modelsin the test set.As
for the nearest neighbor mechanism, we set the (L=  7), the initial𝛼0 =  0.5.
4.6   Evaluation Metrics
Asthenumber oflabelledbenignaccountsandmalicious accountsintheseveraldatasetsused is
well-balanced, we utilize Accuracy to indicate the overall performance of classifiers:
                                     𝐴𝑐𝑢𝑟𝑟𝑎𝑐𝑦=    𝑇𝑃+𝑇𝑁𝑇𝑃+𝐹𝑃+𝐹𝑁+𝑇𝑁,                                            (16)
where𝑇𝑃is True Positive,𝑇𝑁 is True Negative,𝐹𝑃is False Positive,𝐹𝑁is False Negative.
5   EXPERIMENT RESULTS
In this section, we conduct several experiments to evaluate RoSGAS. We mainly answers the
following questions:
    •Q1: How different algorithms perform in different scenarios, i.e., algorithm effectiveness
       (Section 5.1).
    •Q2: How each individual module of RoSGAS contributes to the overall effectiveness (Sec-
       tion 5.2).
    •Q3: How the RL search mechanism work in terms of effectiveness and explainability (Sec-
       tion 5.3).
    •Q4: How fast different algorithm can achieve, i.e., efficiency (Section 5.4) and how the RL
       algorithms can converge effectively (Section 5.5).
    •Q5:Howdifferentalgorithmsperformwhendealingwithpreviouslyunseendata,i.e.,gener-
       alization (Section 5.6).
    •Q6:Howtoexplorethedetectionresultandvisualizethehigh-dimensionaldata(Section5.7).
5.1   Overall Effectiveness
Inthissection,weconductexperimentstoevaluatetheaccuracyofthesocialbotdetectiontaskon
thefivepublicsocialbotdetectiondatasets.Wereportthebesttestresultsofbaselines,RoSGAS,and
thevariants. Weperformed10-foldcross-validationoneach dataset.AsshowninTable 3,RoSGAS
outperforms other baselines and different variants under all circumstances. This indicates the
feasibilityandapplicabilityof RoSGASinwiderrangesofsocialbotdetectionscenarios.Compared
withthebestresultsamongallthestate-of-the-arts,ourmethodcanachieve3.38%,9.55%,5.35%,
4.86%,1.73%accuracyimprovement,onthedatasetsofCresci-15,Varol-17,Vendor-19,Cresci-19
and Botometer-Feedback, respectively.
  In the baselines, Node2Vec is always among the worst performers in the majority of datasets.
ThisisbecauseNode2Veccontrolstherandomwalkprocessbysettingaprobabilityptoswitch
between the BFS and the DFS strategy. Node2Vec sometimes fails to obtain the similarity of
adjacent nodes in large-scale social graph with extremely complex structures, and does not make
good use of node features. GraphSAGE samples the information of neighbors for the aggregation
for each node. This design can not only reduce information redundancy but also increase the
generalizationabilityofthemodelthroughrandomness.However,theproposedrandomsampling
is not suitedfor super large-scale graph, and theinability to adapt to the changeofthe receptive
field drastically limit its performance. GCN multiplies the normalized adjacency matrix with
the feature matrix and then multiplies it with the trainable parameter matrix to achieve the
                                              ACM Trans. Web, Vol. 1, No. 1, Article 1. Publication date: January 2022.

           1:18                                                                                                                                                                             Yang, et al.
               Table 3. Comparison of the average accuracy of different methods for social bot detection (unit: %).
           convolutionoperationonthewholegraphdata.However,obtainingtheglobalrepresentationvector
           foranodethroughfull-graphconvolutionoperationswouldmassivelyreducethegeneralization
           performanceofthemodel.Meanwhile,theincreasesofreceptivefieldwillleadtoasoaringnumber
           of neighbors, and thus weaken the ability of feature representation. GAT shares weight matrix
           for node-wise feature transformation and then calculates the importance of a node to another
           neighbor node, before aggregating the output feature vector of the central node through the
           weightedproductandsummation.GATalsoexperiencestheexplosionofreceptivefieldandthe
           consequentincreaseoftheneighbornumber.UnlikeGraphSAGE,GraphSAINTsetsupasamplerto
           samplesubgraphsfromtheoriginalgraph.ItusesGCNforconvolutiononthesubgraphtoresolve
           neighbor explosion and sampling probability is set to ensure unbiasedness and minimum variance.
           However,extracting subgraphsinGraphSAINTisrandom andthuslimitthe precisionofsubgraph
           embedding. SGC simplifies the conventional GCNs by repeatedly removing the non-linearities
           between GCN layers and collapsing the resulting function into a single linear transformation.
           Namely, the nonlinearactivation functionbetweeneach layeris removedto obtaina linear model.
           Compared with GCN, SGC can achieve similar performance, with a slightly-reduced accuracy
           among all datasets. In addition, RoSGAS also outperforms Policy-GNN and ARMA since these
           counterparts merely make exclusive improvement on convolution layers. By comparison, our
           approach takes advantage of subgraph search and GNN architecture search, whilst leveraging
           self-supervisedlearningtoovercomethelimitationoflimitedlabelledsamples.Thesefunctionalities
           can fulfill better performance even only based upon the basic GCN layer.
           5.2   Ablation Study
           ThesecondhalfofTable3alsocomparestheperformanceofmultiplevariantstodemonstratehow
           to break down the performance gain into different optimization modules.
              WhileRoSGAS-𝐾 only uses the RL agent for subgraph width search, the achieved effectiveness
     Methodis competitive against the state-of-the-arts in some datasets, such as Varol-17, Cresci-19, andCresci-15       Varol-17       Vendor-19      Cresci-19     Botometer-Feedback
 Node2Vec[22]Botometer-Feedback.RoSGAS-𝐿thatonlyenablesthearchitecturesearchcanevenachievebetter73.02±0.91      61.43±0.8      76.13±2.61     76.65±4.97            75.68±0.81
GraphSAGE[23]accuracythanotherbaselinesinmostdatasets,exceptthevendor-19dataset.Theseobservations91.94±1.12     65.71±1.62     81.65±1.19     70.43±0.31            76.16±0.43
    GCN[26]ACM Trans. Web, Vol. 1, No. 1, Article 1. Publication date: January 2022.94.22±0.57     65.15±1.85     85.84±0.81     72.75±2.65            76.02±0.34
    GAT[48]             94.05±0.31     64.63±1.45     78.04±0.78     71.74±1.63            75.21±1.46
     SGC[51]            89.32±0.65     63.32±7.24     78.71±0.75     64.34±1.67            74.51±0.35
GraphSAINT[59]          80.93±0.92     64.89±1.26     77.60±2.51     78.55±1.14            75.51±0.59
Policy-GNN[27]          93.44±0.13     66.20±4.21     82.01±0.23    80.19±7.35           76.61±1.17
    ARMA[5]             94.71±0.43     65.43±3.24     83.44±1.94     78.46±1.52            75.77±0.71
   RoSGAS-𝐾             93.13±0.09     66.06±0.19    76.88±0.78    80.38±0.29            77.09±0.42
    RoSGAS-𝐿            97.82±0.10     67.42±0.37     77.40±0.45     82.80±0.16           77.25±0.90
   RoSGAS-𝐾𝐿            97.82±0.14     68.01±2.72     87.84±1.22     82.95±1.13           77.09±0.14
RoSGAS-𝐾𝐿-𝑁𝑁            97.84±0.05     75.11±0.07     90.52±0.03     84.88±0.01            77.93±0.14
     RoSGAS            98.09±0.36   75.75±0.31   91.19±0.35   85.05±0.21         78.34±0.25
       Gain                3.38↑         9.55↑         5.35↑         4.86↑               1.73↑

RoSGAS: Adaptive Social Bot Detection with Reinforced Self-Supervised GNN Architecture Search                             1:19
Fig.4. Per-nodepredictioneffectivenesswhenconductingarchitecturesearch.Theunderlinerepresentsthe
selected layer number by the RL-agent in theRoSGASfor each node.
arein line withthe enhancement provided by theRL capability.Intrinsically, putting thesearching
mechanisms together will bring synergetic benefits to the effectiveness. Particularly, for datasets
Vendor-19, the combination of width search and layer search can significantly make RoSGAS
outstandingfromtheothercounterparts,reaching87.84%accuracyonaverage.Wecanalsoobserve
thatthe contributionof layersearch totheperformance gainappears tobe moresignificantthan
thewidthsearch.Thisisbecausetheembeddingeffectivenessislessinsensitivetotheneighbor
selection,andthe numberofGNNlayerscanmore effectivelyaffecttheoveralleffectiveness.The
increased layer can make the nodes closer to the target node more frequently aggregated to form
theembeddingofthetargetnode.Likewise,nodesfarfromthetargetnodewillbelessinvolvedin
theembedding.Hence,anenhancedembeddingeffectivenesscanbegainedfromenablingGNN
layersearch.Thesefindingsindicatethenecessityofjointlysearchingthewidthofsubgraphsand
the number of GNN layers to unlock the full potential of performance optimization.
  As shown in the result of RoSGAS-KL-NN, integrating the nearest neighbor mechanism with
thebackbonesearchingmechanismcanfurtherenhancetheperformancegainstemmingfromthe
learningprocessintheRLagent.Mostnoticeably,theaccuracycanbesubstantiallyaugmentedfrom
68% to 75% by the nearest neighbor design when tackling Varol-17 dataset. Even if in some cases,
the improvement is not as significant as that in Varol-17, the similarity driven nearest neighbor
schemecanboosttheactionexplorationintheRLagentandthushelpwithaccuracypromotion.
Furthermore, we demonstrate an incremental performance gain from adopting the proposed self-
supervisedlearningmechanism.Acrossalldifferentdatasets,upto0.7%improvementcanbefurther
achieveddespitemarginalincrementobservedinsomedatasets.Furtherinvestigationwouldbe
requiredtobetterleveragelargequantitiesofunlabeleddataandenhancethegeneralizationability
of the self-supervised classifiers in different scenarios. This is beyond the core scope of this paper
and will be left for future study.
5.3   Effectiveness of the RL Mechanism
5.3.1   Parameter Searching Result by the RL Agent. To conduct an in-depth investigation in how
effectivenessoftheRLsearchmechanism,werandomlyselect20labelednodesfromtheCresci-15
dataset as the target users and examine the accuracy when adopting some given circumstances
of GNN layers. For example, we create 3 types of GNN models, by manually stacking 1 layer of
GCN, 2 layers of GCN, and 3 layers of GCN, respectively. We feed the same graph data used in
the previous testing into the three models and train 100 times. The trained models are used for
validatingif theRLcan obtainthemostproper numberoflayers fortheselected nodes.Foreach
nodein the20target users,wecountthe ratioofbeing correctlyclassifiedout ofthe100 runs.The
                                             ACM Trans. Web, Vol. 1, No. 1, Article 1. Publication date: January 2022.

1:20                                                                                                                                                                             Yang, et al.
           (a) The upper bound of width of subgraph          (b) The upper bound of the number of GNN layers.
      Fig. 5. Impact of search upper bound of subgraph width and layer number on the effectiveness
main purpose of this investigation is to examine if the RL-based mechanism can pick up the layer
with the highest classification ratio, to automatically enable the best model performance.
   Fig. 4 shows the ratio obtained for each target node when using different layers. Observably,
differentGCNlayershaveavaryingimpactonthecorrectpredictionofacertainnode.Forexample,
the prediction effectiveness of other nodes (e.g., node index 2, 4, 5, 11, 12, 13 and 18 ) will be
drasticallyaffectedbythenumberofGNNlayers.Wereasonthisphenomenonisbecausetherange
of GNN’s receptive field will gradually increase when the layer number ramps up; meanwhile,
higherorderneighborinformationwillbeinvolvedandaggregated,therebyhavingadirectimpact
onthedetectionaccuracy.Bycontrast,somenodes(e.g.,nodeindex6,8,9,15,16and19)canbe
betterpredictednomatterwhatinformationisaggregatedfromneighborswithdifferentorders.It
isthereforenecessarytoelaboratelyselectthenumberofGNNlayersforthesenodestoincrease
the probability of correct prediction of these nodes.
   We use underline, e.g.,0.92, to mark the final decision made by the RLagent when searching
themodellayer.TheproposedRLagentcanselecttheoptimallayerthatcandeliverthehighest
predictionratio.Thereisonly10%(2of20classificationtasks)mismatchbetweenthebestlayer
option and the choice made by the RL agent. This indicates the proposed approach can effectively
reduce the manual tuning whilst reaching the best effectiveness.
5.3.2   Impactofdifferentsearchrangesontheeffectiveness.Wedive intotheimpactofparameter
selectionontheoveralleffectivenessanddemonstratethesensitivitytosuchparameterchanges.To
doso,wefirstfixthemaximumsearchingboundofthenumberofsearchlayersofthegraphneural
network to be 3, whilst gradually increasing the searching bound for the width of the subgraph
from 1 to 3. As shown in Fig. 5(a), for all datasets without exception, all the model instances
experienceariseofaccuracywhenthesearchingrangeofwidthgrowsto2butaslightdropwhen
the range is further extended to 3. This is because the increase of width range will lead to a hugely
growing number of neighbors involved in the extracted subgraph. Particularly the scale of the
constructedgraphsinthelarge-scaledatasetsarenormallylargeandwillleadtotheexplosionof
neighbors,whichinturngiverisetothereducedqualityofgraphembeddingandloweraccuracy.
ACM Trans. Web, Vol. 1, No. 1, Article 1. Publication date: January 2022.

RoSGAS: Adaptive Social Bot Detection with Reinforced Self-Supervised GNN Architecture Search                             1:21
           Method            Cresci-15   Varol-17   Vendor-19   Cresci-19   Botometer-Feedback
             GCN               572.53        115.04         323.36          18.78                    10.21
             GAT               576.72        118.51         331.66          21.75                    11.64
             SGC               296.38        198.62         331.43         108.10                   39.97
        GraphSAINT             604.13        167.11         460.13          55.98                    12.56
        Policy-GNN             2076.15       578.21        1734.41        120.34                  109.21
            ARMA                59.79          51.37          84.89           18.29                    12.61
        RoSGAS-𝐾𝐿              587.53        174.12         354.11          55.87                    26.54
     RoSGAS-𝐾𝐿-𝑁𝑁              612.32        212.13         382.11         356.42                   29.22
           RoSGAS              809.88        245.62         469.20          429.6                    32.27
Table 4. The average time consumption (unit: second) of running each method 10 times on the datasets
Cresci-15,Varol-17(Varol),Vendor-19(Vendor),Cresci-19,Botometer-Feedback(Botometer-F),respectively.
Thisobservationindicatesthesearchrangeforextractingsubgraphsneedstobecarefullymodified
and adaptive to the scale of a given scenario.
   Giventhebestresultcanbestablyobtainedbyadopting[1,2]asthewidthrange,wethenfixthis
setting and varying the the search range of the number of model layers. We gradually ramp up the
upperboundoftherangfrom2to5.AsshowninFig.5(b),thereisnosignificantdisparitiesinthe
effectivenessamongdifferentoptions.Theeffectivenessisinsensitivetothechangeofmodellayers
despitesomenoticeable variations.Forexample,intheCresci-15 dataset,themodelaccuracywill
reachthepeakwhenchoosing4astheupperboundofthesearchingrangewhileforVendor-19the
accuracy peak will come when searching up to 3 layers. Nevertheless, the discrepancy in accuracy
is negligible and the proposed RL-based searching mechanism can more flexibly and adaptively
makethebestdecisioninensuringthebestmodelperformancewithoutincurringexcessivecostin
exploring additional GNN layers.
5.4   Efficiency
We primarily evaluate the efficiency by measuring the training time. As the reward signal needs to
be obtained from the validation dataset to train the RL agent before constructing GNN stack, we
breakdownthetimeconsumptionintotwoparts–RLandGNNmodeltraining.Itisworthnoting
that sparse matrix multiplication used in PyTorch Geometric can enable GNNs to be applied in
very large graphs and accelerate the model training.
   Table4presentstheaveragetimeconsumptionofrunningeachmodel10times.Overall,RoSGAS
canstrikeabalancebetweenthetrainingtimeandtheeffectiveness.AlthoughARMAandSGC
takelesstimetotraintheirmodelduetothesimplifiedGCNmodelthroughremovingnon-linearity
andcollapsingweightmatricesbetweenconsecutivelayers,theachievedaccuracyisfarlowerthan
RoSGASacrossalldatasets,particularlyonCresci-19wherethelabelsarescarcer.Policy-GNNtakes
thelongesttimeformodeltrainingsimplybecausetheGNNstackingandconvolutionoperation
will be carried out for all nodes in the entire graph.
   Most notably, the variantRoSGAS-𝐾𝐿which does notinclude the nearest neighbor mechanism
andtheself-supervisedlearningmechanismcanachievecompetitivetrainingefficiencycompared
with GCN, GAT, and GraphSAINT, only with a marginal time increase. The slight difference is
negligible when considering RoSGAS-𝐾𝐿needs to train both the RL agent and the GNN model
separately.ComparedwithRoSGAS-𝐾𝐿,RoSGAS-𝐾𝐿-𝑁𝑁 intrinsicallyneedsextratimetosearch
the set of state-action pairs that have been explored and ascertain the nearest neighbor to the
                                               ACM Trans. Web, Vol. 1, No. 1, Article 1. Publication date: January 2022.

1:22                                                                                                                                                                             Yang, et al.
                        (a) Cresci-15.                                               (b) Varol-17.
                        (c) Vendor-19.                                         (d) Botometer-Feedback.
                                 Fig. 6. The RL-agent training process of RoSGAS.
current state-actionpair,before modifyingthe expectationof therewardfor optimized network
parameters.RoSGAS additionally involves the self-supervised learning based on the target user
batch to extract the homologous subgraphs for additional forward propagation.
   Inaddition,thetimeconsumptionforCresci-15isfarlargerthanthatforCresci-19.Thereisa
linearly-increasedtimeconsumptionofGCNandGATwhenthegraphscalesoars.Infact,theGNN
trainingismoreefficientsinceweonlyneedtoperformconvolutionoperationsonthesampled
subgraph;bycontrast,eachiterationofGCNandGATwillhavetoperformaconvolutionoperation
onall nodesofan entiregraph.RoSGAS issolely relevantto theextractednumber ofsubgraphs
for detectingthe targetusers, andthus independent from thescale of theentiregraph. This clearly
showcases the inherent scalability and robustness of our sample and subgraph based mechanism
adopted inRoSGAS.
ACM Trans. Web, Vol. 1, No. 1, Article 1. Publication date: January 2022.

RoSGAS: Adaptive Social Bot Detection with Reinforced Self-Supervised GNN Architecture Search                             1:23
5.5   Stability
WealsocomparethestabilityoftheRL-basedarchitecturesearchbetweenRoSGASandPolicy-GNN.
Fig.6 demonstratesthedetailed trainingprocesswiththe RLagenton Cresci-15,Varol-17,Vendor-
19andBotometer-Feedback,respectively.Thedottedlinerepresentstheaccuracyobtainedinthe
validation set during the 150 episodes. Obviously,RoSGAS can promptly achieve high accuracy
under all the datasets and the mean accuracy can be achieved only after 15 RL agent training
episodes.Meanwhile,oncereachingthispoint,astablestateofNashequilibriumcanbesteadily
maintained without huge turbulence.
   By contrast, the accuracy of Policy-GNN is noticeably lower thanRoSGAS and lacks stability,
i.e., very obvious fluctuations manifest. The disparity mainly stems from the design of the state
transitionandthenearestneighbormechanism.RoSGASusestheembeddingoftheinitialsubgraph
as the state input to the RL agent, and jumps between the initial subgraphs as the state transition,
whilePolicy-GNNusesthenodeembeddingasthestateinputandjumpsbetweenthenodesasthe
state transition. Undoubtedly, the embedding of the initial subgraph as a state can better reflect
the local structure of a targeted node, resulting in a stronger representation ability, and hence
the enhanced stability. At the same time, during the RL agent training episodes, for a specific
state-actionpairof onetransition,thenearestneighbormechanism explores theexistingpairsand
exploitstherewardpertaining tothisnearestneighborfrom theenvironment.Therewardisused
asapartofthelabeltooptimizetheQ-network,whichgreatlyeliminatesthedifferencebetween
thetargetnetworkprediction andtheactualrewardinthe initialstage.Asaresult,the RLagents
can achieve higher accuracy with only a few training episodes.
   Interestingly, RoSGAS has different volatility across different datasets and the magnitude of
volatilityis positivelycorrelatedto theaveragecorrectrate. Thisindicatesthe nodefeaturedistri-
bution and graph structure pertaining to each individual dataset have a non-trivial impact on the
model training. The in-depth study will be left for future work.
5.6   Generalization
Inthe fieldofsocial botdetection, thecontinuousevolution ofsocialbots’ camouflagetechnology
hasbroughtgeneralizationchallengestothemodeldesign.Toidentifytheemergingattackmethods
ofsocialbotsandminediverseuserinformation,arobustdetectionmodelshouldhaveahighlevel
of generalization.In this subsection, weexamine the generalizationof RoSGAS and comparewith
other baselines.
5.6.1   Out-of-sampleValidationAccuracy. Apartfromthein-samplevalidationinprevioussubsec-
tions, the mosteffective way to demonstrate thestrong generalization is toperformout-of-sample
validation,i.e.,retainingsomeofthesampledataformodeltraining,andthenusingthemodelto
make predictions for unseen data and examine the accuracy. To do so, we select one dataset as the
trainingdataset andthen useother datasetsasthe testdataset. Wedivide thetraining setinto10
equal parts, take one part of the training set for training each time, and train each baseline for 30
epochs.Weusethetrainedmodelstopredictthelabelledaccountsinthetestdatasetstocalculate
the accuracy.
   A series of figures including Fig.7, Fig. 8, Fig. 9, Fig. 10 and Fig. 11show the prediction results
whenthetrainingisbaseduponCresci-15,Varol-17,Vendor-19,Cresci-19,andBotometer-Feedback,
respectively. It is obviously observable thatRoSGAS outperforms all other baselines in the vast
majority ofscenarios. Theonly exceptions are when themodel istrainedbased onVendor-19and
validatedonCresci-15dataset,orwhenthemodelistrainedbasedonCresci-15whilstvalidating
upon Vendor-19. In this two cases, the accuracy of RoSGAS is slightly lower than GCN. We reason
thisphenomenonis possiblybecauseGCNisless sensitivetothedisparity betweentwodatasets.
                                              ACM Trans. Web, Vol. 1, No. 1, Article 1. Publication date: January 2022.

1:24                                                                                                                                                                             Yang, et al.
                          Fig. 7. Train on the Cresci-15 and testing on different dataset
                         Fig. 8. Training on the Varol-17 and testing on different dataset
Unsurprisingly,allmodelshasareducedaccuracywhenconductingtheout-of-sampleprediction
as opposed to its in-sample accuracy, simply because of the potential overfitting in the in-sample
evaluations. The experiments carried out in this subsection generically showcase the robustness
and generalization of our approach when handling new data where different noise manifests.
5.6.2   Stability. We canalso observethe minimum performance fluctuation of RoSGAS, compared
with other baselines, when different datasets are used as test sets. On the contrary, many other
baselines such as GraphSAGE, GCN, and GraphSAINT have a noticeable fluctuation. For example,
asdemonstrated inFig.7, whentrainingon Cresci-15,theaccuracy ofGraphSAGEisonly 23.21%
and 26.63%when the trainedmodel is testedbased onVendor-19 andBotometer-F. However,the
accuracy can climb up to 47.8% if tesing on Cresci-19. Likewise, as shown in Fig. 8, while the
accuracy of the GraphSAINT model trained upon Varol-17 and tested upon Cresci-15 is merely
41.34%,acompetitiveaccuracycanbeobtainedwhenthemodelistestedonVendor-19(75.23%)or
Botometer-F (74.12%).
   Thestablegeneralizationstemsfromtheadaptabilityandrobustnessof RoSGAS.Infact,our
method only exploits some common features for the task of detecting social bot, without tightly
ACM Trans. Web, Vol. 1, No. 1, Article 1. Publication date: January 2022.

RoSGAS: Adaptive Social Bot Detection with Reinforced Self-Supervised GNN Architecture Search                             1:25
                              Fig. 9. Train on the Vendor-19 and testing on different dataset
                             Fig. 10. Train on the Cresci-19 and testing on different dataset
                      Fig. 11. Train on the Botometer-Feedback and testing on different dataset
                                                         ACM Trans. Web, Vol. 1, No. 1, Article 1. Publication date: January 2022.

1:26                                                                                                                                                                             Yang, et al.
couplingwith, ordependingupon exclusivefeatures ornumericalcharacteristics. Thiswillhelp
maintain an outstanding quality of detection even when the testset varies.
5.7   Case Study: Effectiveness of Representation Learning
Tofurtherunderstandthequalityofvectorrepresentation,wecomparetherepresentationsof RoS-
GAS with the baselines that can achieved good results in the accuracy evaluation, i.e., GraphSAGE,
GCN,GraphSAINTandARMA.Foreachindividualmodel,weclustertherepresentationresultsby
using𝑘-meanswith𝑘=  2andthencalculatethehomogeneityscore–ahigher-the-betterindicator
that measures how much the sample in a cluster are similar. The homogeneity is satisfied – the
value equals 1 – if all of its clusters contain only data points which are members of a single class.
 (a) GraphSAGE (Score:4.956×10−1)            (b) GraphSAGE (Score:2.250×10−1)            (c) GraphSAGE (Score:1.825×10−1)
     (d) GCN (Score:1.101×10−1)                  (e) GCN (Score:4.158×10−1)                 (f) GCN (Score:2.849×10−1)
 (g) GraphSAINT (Score:4.600×10−1)          (h) GraphSAINT (Score:1.799×10−1)           (i) GraphSAINT (Score:1.505×10−1)
ACM Trans. Web, Vol. 1, No. 1, Article 1. Publication date: January 2022.

RoSGAS: Adaptive Social Bot Detection with Reinforced Self-Supervised GNN Architecture Search                             1:27
    (j) ARMA (Score:9.448×10−2)            (k) ARMA (Score:3.223×10−1)             (l) ARMA (Score:1.847×10−1)
  (m) RoSGAS (Score:8.174×10−1)           (n) RoSGAS (Score:4.619×10−1)           (o) RoSGAS(Score:3.654×10−1)
Fig.12. 2Dt-SNEplotofrepresentationvectorsofusersproducedbyGraphSAGE,GCN,GraphSAINTARMA
and RoSGAS on the dataset of Cresci-15 (left column), Vendor-19 (middle column) and Cresci-19 (right
column), respectively. The corresponding homogeneity score is given in the bracket
   Fig. 12 visualizes the result of t-SNE dimensionality reduction on the representation vectors
of users. We evaluate each model based on three given datasets, Cresci-15, Vendor-19, Cresci-
19, respectively. For each baseline model, the results on the three datasets are placed in a row.
Numerically, thehomogeneity scoreof RoSGASis higherthan otherbaselines whenadopting all
datasets. For example, the score of RoSGAS is over 2 times higher than that of GraphSAGE on
average,indicatingthemostdistinguishablerepresentvectorscanbeobtainedbyourapproach.
Thevisualizationcompletelyalignswiththemeasurement.Observably,thebotsinRoSGAScanbe
far more easily differentiated from the benign users.
6   DISCUSSION
Significanceofsubgraph-basedandRL-guidedsolution.Thisworkistoadvancethedevelop-
mentandapplicationofGNNsinthefieldofsocialbotdetection.Theperformanceoffeatures-based,
statistics-based and deep learning based methods fades facing the evolving bot technologies. We
applied the GNNto leverage theinformation of the neighborhoodand relationship to counterthe
development of bots. This work is non-trivial when tackling massive datasets with hundreds of
thousandsorevenmillionsofnodes.Theexistingsolutionstocomplexmodelarchitecturesearch
isnotwell-suitedfortheproblemofsocialbotdetectionatscaleunderstudy–thedatadistribution
becomes more non-IID due to the evolution of the bots, which complicates the model design with
goodgeneralizationinpractice.Theextremely-largescaleofthesocialnetworkgraphalsoleads
                                              ACM Trans. Web, Vol. 1, No. 1, Article 1. Publication date: January 2022.

1:28                                                                                                                                                                             Yang, et al.
totremendouslydifferentuserstructuresandnecessitatesadaptivedetectionofsuchstructures
with reasonable computational costs. To this end, we proposed to search subgraphs to realize the
reductionofgraph scale,thesimplificationofmodel architectures,andtheimprovementsofthe
detection performance.
Necessity of using RL. The width (𝑘) of the subgraph cannot be a common hyperparameter
sharedbyallnodesandrequiresnode-by-nodepersonalization.Infact,socialbotstendtorandomly
follow benign accounts as a disguise. The selection of𝑘primarily derives from our behavioural
studies–Therearenoticeablyenoughbotsinthesubgraphwithin2th-ordersubgraphwhilebenign
nodes could be well recognized in the 1th order subgraph. Increasing the value would not incur
additionalperformancegainbutinvolveanoverwhelmingnumberofneighbors.Nevertheless,the
rangecanbeflexiblyconfiguredtoadapttoanyotherscenariosanddatasets.Thisnecessitatesthe
adoption of RL to facilitate the customized parameter search for each individual subgraph.
Dealing with real-time streaming scenarios. Critical challenges for on-the-fly bot detection
encompass the need of message delivery, distributed event store and the capability of handling
revolution anduncertainty of eventsand entitiesin thesocial networkplatform over time.This
isparticularlyintricatedue tothepresenceofnewbots.The accuracyofofflinelearningmodels
highlyreliesonthequalityandquantityofthedatasetthatisfedintothemodels.However,thiswill
betime-consumingandcostlyinreal-timescenariosandthemodelupdateisrequiredtomaintain
thehighstandardofmodelaccuracy.Inpractice,toimplementthestreamingpipeline,adistributed
crawlerneedstobedevelopedtocontinuouslyfetchsocialnetworkinformation.Thecollecteddata
isthenforwardedtoprocessingmodulesthroughdistributedeventstreamingsuch asKafka.More
online and incremental designs are desired to underpin the on-the-fly version of the current bot
detection framework.
7   RELATED WORK
Inthissection,wesummarizetherelatedliteratureandstate-of-the-artapproaches.Theexisting
literature can be roughly classified into three categories: GNN based approaches, subgraph and RL
based approaches, and self-supervised enhanced approaches.
7.1   GNN based Social Bot Detection
Earlysocialbotdetectionmainlyfocusesonmanuallyanalyze thecollecteddatatofinddiscrim-
inative features that can be used to detect the social bot. However, the detection features are
easy to imitate and escaped by social bots which are constantly evolving, and eventually become
invalid. Therecent boom ofGraph Neural Networks (GNNs) haspromoted the latestprogressin
socialbotdetection.The firstattemptto useGNN to detectsocial bots[2]is mainlybycombining
thegraphconvolutionalneuralnetworkwithmultilayerperceptionandbeliefpropagation.The
heterogeneous graph is constructed and original node features are extracted by the pre-trained
language model to get the final node embedding after aggregating by the R-GCN model [17], the
BotRGCN successfully surpass the performance of traditional detection methods on the newly
released dataset called TwiBot-20 [16]. The heterogeneous graph is also applied in [15], it also
proposes a relational graph transformer inspired from natural language processing to model the
influence between users and learn node representation for better social bot detection, and its
performance exceeds the BotRGCN. However, these node embedding and node classification based
methods perform convolution at the level of the entire graph, after stacking multiple layers of
GNN on the tremendous scale of social graph will cause the over-smoothing problem [30, 53].
Subgraphprovidesanewperspectivetosolvethisissue,[31]constructsaheterogeneousgraphand
reconstructssubgraphsbasedonmanually-definedheuristicrulesfordetectingmaliciousaccounts
ACM Trans. Web, Vol. 1, No. 1, Article 1. Publication date: January 2022.

RoSGAS: Adaptive Social Bot Detection with Reinforced Self-Supervised GNN Architecture Search                             1:29
inonlineplatforms.However,thiskindofmanual-basedmethodnotonlyconsumesenergytoset
the extraction rules but also cannot be easily generalized to the field of social bot detection.
7.2   Subgraph and RL based Approaches
Thescaleofthegraphconstructedfromthesocialnetworkistremendous.Performingconvolution
operation onthe entiregraph willnot onlyconsume computing resourcesbut alsolead toperfor-
mance degradation due to problems such as over-smoothing. The extraction of subgraphs either
requires domain-specific expert knowledge to set the heuristic rules [31, 58] or needs to design
motif for matching subgraphs [35, 54], which drastically limits the flexibility and capability of
generalization.Toaddressthisissue,weleverageReinforcementLearning(RL)toadaptivelyextract
subgraphs. Therehave been afewattempts tomarryRL andGNNs. DeepPath[52] establishesa
knowledgegraphembeddingandreasoningframeworkthatutilizestheRLagenttoascertainthe
reasoningpathsintheknowledgebase.RL-HGNN[62]devisesdifferentmeta-pathsforanynodein
aHINforadaptiveselectmeta-pathtolearnitseffectiverepresentations.CARE-GNN[14],RioGNN
[37] RTGNN [61] and FinEvent [38] all marry the RL and GNNs for dynamically optimizing the
similarity threshold to achieve the purpose of selecting more valuable neighbour nodes for the
aggregated nodes, to obtain more effective representation vectors for fraud detection or event
detection.Policy-GNN[27]utilizesRLto selectthenumberofGNNarchitecturelayersforaggre-
gatingthenodeembeddingvectorstoclassifynodes.GraphNAS[20]isamongthefirstattemptsto
combinetherecurrentneuralnetworkwithGNNs.ItcontinuouslygeneratesdescriptionsofGNN
architecture to find the optimal network architecture based on RL by maximizing the expected
accuracy. SimilarlytothearchitecturesearchinGraphNAS, Auto-GNN[63]additionallyproposes
aparametersharingmechanismforsharingtheparametersinhomogeneousarchitectureforreduc-
ing the computation cost. However, these methods are not combined with the subgraph method
andtheiroptimizationistightlycoupledwithspecificdatasets.Theyarenotsuitedfordetecting
graphs that follow a power-law distribution with huge disparities among different users [4, 34].
By contrast, ourapproach relies on subgraph embedding to achieve high detectioneffectiveness
without compromising time efficiency, and has strong generalization across multiple datasets.
7.3   Self-Supervised Learning Approaches on Graphs
Inrecentyears,self-supervisedlearninghashugelyadvancedasapromisingapproachtoovercome
thelimiteddataannotationandlimitedandtoenableatargetobjectiveachievedwithoutsupervision.
The technology has been investigated in a wide range of domains, such as natural language
processing [11, 60], computer vision [28, 46] and graph analysis [22, 25]. At the core of self-
supervisedlearningistodefineanannotation-freepretexttasktotrainanencoderforrepresentation
learning.Particularly forgraphanalysis,there areafew worksofliteratureabout designingself-
supervised tasks based on either edge attributes [10, 44] or node attributes [12]. However, the
inherent dependencies among different nodes in the topology hinder the appropriate design of
the pretext tasks. DGI [49] trains a node encoder to maximize the mutual information between
the node representations and the global graph representation. GMI [39] defines a pretext task
that maximizes the mutualinformation between the hidden representation of each nodeand the
originalfeaturesofits1-hopneighbors.InfoGraph[43]maximizesthemutualinformationbetween
thegraph embeddings andthesubstructure embeddings atdifferentscales tomore effectivelearn
graph embeddings. However, the aforementioned self-supervised learning approaches need to
takethe holisticgraph asthe input,whichis time-and resource-consumingand thusrestricts the
scalabilityonlarge-scalegraphs.Ourapproachaimstoobtainasubgraph-levelrepresentationto
ensure non-homologous subgraphs are discriminative while homologous subgraphs have similar
representation vectors.
                                             ACM Trans. Web, Vol. 1, No. 1, Article 1. Publication date: January 2022.

1:30                                                                                                                                                                             Yang, et al.
8   CONCLUSION
This paper studies a RL-enabled framework for GNN architecture search. The proposedRoSGAS
framework can adaptively ascertain the most suitable multi-hop neighborhood and the number of
layersintheGNNarchitecturewhenperformingthesubgraphembeddingforthesocialbotdetection
task.WeexploitHIN torepresentthe userconnectivityanduse multi-agentdeepRLmechanism
forsteering thekey parametersearch.Thesubgraph embeddingfor atargeted usercan be more
effectively learnt and used for the downstream classification with competitive accuracy whilst
maintaininghighcomputationefficiency.ExperimentsshowthatRoSGASoutperformsthestate-of-
the-artGNNmodelsintermsofaccuracy,trainingefficiencyandstability.RoSGAScanmorequickly
achieve,andcarryonwith,highaccuracyduringtheRLtraining,andhasstronggeneralization
andexplainability.Webelievethedata-centricsolutionguidedbybehaviouralcharacterizationand
reinforcementlearning–insteadofheavilycomplicatingthenetworkarchitectureitself–wouldbe
apromisingandinnovativedirection,whichisbothscientificallyandengineering-wisechallenging
in the field of bot detection. In the future, we plan to examine the impact of feature distribution
and graph structure on model training, and extendRoSGAS to underpin the streaming scenarios.
ACKNOWLEDGMENT
Wethankanonymousreviewersfortheprovidedhelpfulcommentsonearlierdraftsoftheman-
uscript. Zhiqin Yang and Yue Wang are supported by the National Key R&D Program of China
through grant2021YFB1714800, and S&TProgram ofHebei through grant20310101D. Yangyang
Li is supported by NSFC through grant U20B2053. Yingguang Yang, Kai Cui and Haiyong Xie are
supportedbyNationalKeyR&DProgramofChinathroughgrantSQ2021YFC3300088.RenyuYang
and JieXu are supported byUK EPSRC Grant(EP/T01461X/1),UK Turing PilotProject funded by
the Alan Turing Institute. Renyu Yang is also supported by the UK Alan Turing PDEA Scheme.
REFERENCES
 [1] NorahAbokhodair,DaisyYoo,andDavidWMcDonald.2015. Dissectingasocialbotnet:Growth,contentandinfluence
     in Twitter. In CSCW. 839–851.
 [2] SeyedAli Alhosseini,RaadBin Tareaf,PejmanNajafi,and ChristophMeinel.2019. Detectmeifyoucan:Spam bot
     detection using inductive representation learning. In WWW. 148–153.
 [3] EmilyAlsentzer,SamuelFinlayson,MichelleLi,andMarinkaZitnik.2020. SubgraphNeuralNetworks. NIPS 33(2020).
 [4] Albert-László Barabási and Réka Albert. 1999. Emergence of scaling in random networks. science 286, 5439 (1999),
     509–512.
 [5] Filippo Maria Bianchi, Daniele Grattarola, Lorenzo Livi, and Cesare Alippi. 2021.  Graph neural networks with
     convolutional arma filters. TPAMI (2021).
 [6] AdamBreuer,RoeeEilat, andUdiWeinsberg.2020. Friend orfaux:Graph-based earlydetectionof fakeaccountson
     social networks. In WWW. 1287–1297.
 [7]Stefano Cresci. 2020. A decade of social bot detection.      Commun. ACM 63, 10 (2020), 72–83.
 [8] StefanoCresci,RobertoDiPietro,MarinellaPetrocchi,AngeloSpognardi,andMaurizioTesconi.2015. Fameforsale:
     Efficient detection of fake Twitter followers. Decis. Support Syst.80 (2015), 56–71.
 [9] Stefano Cresci, Roberto Di Pietro, Marinella Petrocchi, Angelo Spognardi, and Maurizio Tesconi. 2017. The paradigm-
     shift of social spambots: Evidence, theories, and tools for the arms race. In WWW. 963–972.
[10]Quanyu Dai, Qiang Li, Jian Tang, and Dan Wang. 2018. Adversarial network embedding. In       AAAI, Vol. 32.
[11] JacobDevlin,Ming-WeiChang,KentonLee,andKristinaToutanova.2018. Bert:Pre-trainingofdeepbidirectional
     transformers for language understanding. In NAACL. 4171–4186.
[12] MingDing,JieTang,andJieZhang.2018. Semi-supervisedlearningongraphswithgenerativeadversarialnets.In
     CIKM. 913–922.
[13] Yuxiao Dong, Nitesh V Chawla, and Ananthram Swami. 2017. metapath2vec: Scalable representation learning for
     heterogeneous networks. In KDD. 135–144.
[14]Yingtong Dou,Zhiwei Liu,Li Sun, YutongDeng, Hao Peng,and PhilipS Yu. 2020. Enhancing graphneural network-
     based fraud detectors against camouflaged fraudsters. InCIKM. 315–324.
ACM Trans. Web, Vol. 1, No. 1, Article 1. Publication date: January 2022.

RoSGAS: Adaptive Social Bot Detection with Reinforced Self-Supervised GNN Architecture Search                             1:31
[15] Shangbin Feng, Zhaoxuan Tan, Rui Li, and Minnan Luo. 2022.  Heterogeneity-aware Twitter Bot Detection with
      Relational Graph Transformers. AAAI.
[16] ShangbinFeng,HerunWan,NingnanWang,JundongLi,andMinnanLuo.2021. TwiBot-20:AComprehensiveTwitter
      Bot Detection Benchmark. In CIKM. 4485–4494.
[17] Shangbin Feng,Herun Wan, NingnanWang, andMinnan Luo.2021. BotRGCN:Twitter botdetection withrelational
      graph convolutional networks. In Proceedings of the 2021 IEEE/ACM International Conference on Advances in Social
      Networks Analysis and Mining. 236–239.
[18] EmilioFerrara,OnurVarol,ClaytonDavis,FilippoMenczer,andAlessandroFlammini.2016. Theriseofsocialbots.
      Commun. ACM 59, 7 (2016), 96–104.
[19] MatthiasFeyandJanE.Lenssen.2019.FastGraphRepresentationLearningwithPyTorchGeometric.InICLRWorkshop.
[20] Yang Gao, Hong Yang, Peng Zhang, Chuan Zhou, and Yue Hu. 2020. Graph Neural Architecture Search.. In IJCAI,
     Vol. 20. 1403–1409.
[21] Zafar Gilani, Ekaterina Kochmar, and Jon Crowcroft. 2017. Classification of twitteraccounts into automated agents
      and human users. In ASONAM. 489–496.
[22]Aditya Grover and Jure Leskovec. 2016. node2vec: Scalable feature learning for networks. In       KDD. 855–864.
[23] WilliamLHamilton,RexYing,andJureLeskovec.2017. Inductiverepresentationlearningonlargegraphs.In NIPS.
     1025–1035.
[24] YimingHei,RenyuYang,HaoPeng,LihongWang,XiaolinXu,JianweiLiu,HongLiu,JieXu,andLichaoSun.2021.
      Hawk: Rapid android malware detection through heterogeneous graph attention networks. TNNLS (2021).
[25]Thomas N Kipf and Max Welling. 2016. Variational graph auto-encoders.        arXiv preprint arXiv:1611.07308 (2016).
[26]Thomas N Kipf and Max Welling. 2017. Semi-supervised classification with graph convolutional networks. In       ICLR.
[27] Kwei-HerngLai,DaochenZha,KaixiongZhou,andXiaHu.2020. Policy-gnn:Aggregationoptimizationforgraph
      neural networks. In KDD. 461–471.
[28] GustavLarsson,MichaelMaire,andGregoryShakhnarovich.2016. Learningrepresentationsforautomaticcolorization.
      In ECCV. Springer, 577–593.
[29] JohnBoazLee,RyanARossi,XiangnanKong,SungchulKim,EunyeeKoh,andAnupRao.2019. Graphconvolutional
      networks with motif-based attention. In CIKM. 499–508.
[30] Qimai Li, Zhichao Han, and Xiao-Ming Wu. 2018.  Deeper insights into graph convolutional networks for semi-
      supervised learning. In Thirty-Second AAAI conference on artificial intelligence.
[31] Ziqi Liu, Chaochao Chen, Xinxing Yang, Jun Zhou, Xiaolong Li, and Le Song. 2018. Heterogeneous graph neural
      networks for malicious account detection. In CIKM. 2077–2085.
[32] MicheleMazza,StefanoCresci,MarcoAvvenuti,WalterQuattrociocchi,andMaurizioTesconi.2019. Rtbust:Exploiting
      temporal patterns for botnet detection on twitter. In WebSci. 183–192.
[33] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves,
      MartinRiedmiller,AndreasKFidjeland,GeorgOstrovski,etal.2015. Human-levelcontrolthroughdeepreinforcement
      learning. Nature 518, 7540 (2015), 529–533.
[34] Lev Muchnik,SenPei, LucasCParra, SauloDS Reis,JoséS AndradeJr, ShlomoHavlin,and HernánAMakse.2013.
      Originsofpower-lawdegreedistributionintheheterogeneityofhumanactivityinsocialnetworks. Scientificreports3,
     1 (2013), 1–8.
[35] Hao Peng, Jianxin Li, Qiran Gong, Yuanxin Ning, Senzhang Wang, and Lifang He. 2020.  Motif-Matching Based
      Subgraph-Level Attentional Convolutional Network for Graph Classification. InAAAI, Vol. 34. 5387–5394.
[36] HaoPeng,Renyu Yang,ZhengWang, JianxinLi,LifangHe,PhilipYu,AlbertZomaya,andRaj Ranjan.2021. Lime:
      Low-cost incremental learning for dynamic heterogeneous information networks. IEEE Trans. Comput.(2021).
[37] HaoPeng,RuitongZhang,YingtongDou,RenyuYang,JingyiZhang,andPhilipS.Yu.2021. ReinforcedNeighborhood
      SelectionGuidedMulti-RelationalGraphNeuralNetworks. ACM Trans. Inf. Syst.40,4,Article69(dec2021),46pages.
[38] HaoPeng, ruitongZhang, ShaoningLi, Yuwei Cao,Shirui Pan,and PhilipS. Yu. 2022. Reinforced, Incrementaland
      Cross-lingual Event Detection From Social Messages. TPAMI (2022), 1–1.
[39] ZhenPeng, WenbingHuang,Minnan Luo, QinghuaZheng,YuRong,Tingyang Xu,andJunzhou Huang.2020. Graph
      representation learning via graphical mutual information maximization. In WWW. 259–270.
[40] FrancoScarselli,MarcoGori,AhChungTsoi,MarkusHagenbuchner,andGabrieleMonfardini.2008. Thegraphneural
      network model. IEEE transactions on neural networks 20, 1 (2008), 61–80.
[41] Junhong Shen and LinF Yang. 2021. TheoreticallyPrincipled Deep RL Acceleration via Nearest NeighborFunction
     Approximation. In AAAI, Vol. 35. 9558–9566.
[42] ChuanShi,YitongLi,JiaweiZhang,YizhouSun,andSYuPhilip.2016. Asurveyofheterogeneousinformationnetwork
      analysis. TKDE 29, 1 (2016), 17–37.
[43] Fan-Yun Sun, Jordan Hoffmann, Vikas Verma, and Jian Tang. 2020. Infograph: Unsupervised and semi-supervised
      graph-level representation learning via mutual information maximization. ICLR (2020).
                                                       ACM Trans. Web, Vol. 1, No. 1, Article 1. Publication date: January 2022.

1:32                                                                                                                                                                             Yang, et al.
[44] Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, and Qiaozhu Mei. 2015. Line: Large-scale information
      network embedding. In WWW. 1067–1077.
[45]Julian R Ullmann. 1976. An algorithm for subgraph isomorphism.        Journal of the ACM (JACM) 23, 1 (1976), 31–42.
[46] AäronvandenOord,NalKalchbrenner,LasseEspeholt,KorayKavukcuoglu,OriolVinyals,andAlexGraves.2016.
      Conditional Image Generation with PixelCNN Decoders. In NIPS. 4790–4798.
[47] Onur Varol, Emilio Ferrara, Clayton Davis, Filippo Menczer, and Alessandro Flammini. 2017.  Online human-bot
      interactions: Detection, estimation, and characterization. In Proceedings of the international AAAI conference on web
      and social media, Vol. 11.
[48] PetarVeličković, GuillemCucurull,Arantxa Casanova,Adriana Romero,PietroLio, andYoshuaBengio. 2018. Graph
      attention networks. In ICLR.
[49] Petar Velickovic, William Fedus, William L Hamilton, Pietro Liò, Yoshua Bengio, and R Devon Hjelm. 2019. Deep
      Graph Infomax. ICLR (Poster) 2, 3 (2019), 4.
[50] Jianyu Wang, Rui Wen, Chunming Wu, Yu Huang, and Jian Xion. 2019.   Fdgars: Fraudster detection via graph
      convolutional networks in online app review system. In WWW. 310–316.
[51] Felix Wu, Amauri Souza, Tianyi Zhang, Christopher Fifty, Tao Yu, and Kilian Weinberger. 2019. Simplifying graph
      convolutional networks. In ICML. 6861–6871.
[52] Wenhan Xiong, Thien Hoang, and William Yang Wang. 2017.  Deeppath: A reinforcement learning method for
      knowledge graph reasoning. In EMNLP. 564–573.
[53] Keyulu Xu, Chengtao Li, Yonglong Tian, Tomohiro Sonobe, Ken-ichi Kawarabayashi, and Stefanie Jegelka. 2018.
      Representation learning on graphs with jumping knowledge networks. In ICML. PMLR, 5453–5462.
[54] CarlYang, MengxiongLiu, VincentWZheng, andJiawei Han.2018. Node, motifandsubgraph: Leveragingnetwork
      functional blocks through structural convolution. In ASONAM. IEEE, 47–52.
[55] Kai-Cheng Yang, Onur Varol, Clayton A Davis, Emilio Ferrara, Alessandro Flammini, and Filippo Menczer. 2019.
      Arming the public with artificial intelligence to counter social bots. Comput. Hum. Behav.1, 1 (2019), 48–61.
[56] Kai-ChengYang,OnurVarol,Pik-Mai Hui,andFilippo Menczer. 2020. Scalableandgeneralizable socialbotdetection
      through data selection. In AAAI, Vol. 34. 1096–1103.
[57] XiaoyuYang,YuefeiLyu,TianTian,YifeiLiu,YudongLiu,andXiZhang.2020. RumorDetectiononSocialMediawith
      Graph Structured Adversarial Learning.. In IJCAI. 1417–1423.
[58] ZihaoYuan,QiYuan,andJiajingWu.2020. Phishingdetectiononethereumvialearningrepresentationoftransaction
      subgraphs. In BlockSys. Springer, 178–191.
[59] Hanqing Zeng, Hongkuan Zhou, Ajitesh Srivastava, Rajgopal Kannan, and Viktor Prasanna. 2020. Graphsaint: Graph
      sampling based inductive learning method. In ICLR.
[60] XingxingZhang,FuruWei,andMingZhou.2019. HIBERT:Documentlevelpre-trainingofhierarchicalbidirectional
      transformers for document summarization. In ACL. 5059–5069.
[61] XushengZhao,QiongDai,JiaWu,HaoPeng,MingshengLiu,XuBai,JianlongTang,andPhilipS.Yu.2022. Multi-view
      Tensor Graph Neural Networks Through Reinforced Aggregation. TKDE (2022), 1–1.
[62]ZhiqiangZhong,Cheng-TeLi,andJunPang.2020. ReinforcementLearningEnhancedHeterogeneousGraphNeural
      Network. arXiv preprint arXiv:2010.13735 (2020).
[63] KaixiongZhou, QingquanSong, Xiao Huang,and XiaHu. 2019. Auto-gnn: Neuralarchitecture searchof graphneural
      networks. arXiv preprint arXiv:1909.03184 (2019).
ACM Trans. Web, Vol. 1, No. 1, Article 1. Publication date: January 2022.

