               MedicalTriageChatbotDiagnosisImprovementvia
                Multi-relationalHyperbolicGraphNeuralNetwork
                   Zheng Liu∗                                         Xiaohan Li∗                                           Zeyu You
                 zliu212@uic.edu                                      xli241@uic.edu                                 zeyuyou@tencent.com
       University of Illinois at Chicago                   University of Illinois at Chicago                                  Tencent
                         USA                                                 USA                                                USA
                    Tao Yang                                             Wei Fan                                            Philip Yu
            tytaoyang@tencent.com                              davidwfan@tencent.com                                      psyu@uic.edu
                      Tencent                                             Tencent                              University of Illinois at Chicago
                         USA                                                USA                                                 USA
ABSTRACT                                                                           1   INTRODUCTION
Medical triage chatbot is widely used in pre-diagnosis by asking                   In the healthcare field specifically, a medical triage chatbot aims
symptom and medical history related questions. Information col-                    at collectingrelevantinformation such as symptomsand medical
lected from patients through an online chatbot system is often                     history through conversations, and then provides pre-diagnosis
incomplete and imprecise, and thus it’s essentially hard to achieve                and triaging. From the modeling perspective, a key challenge of
precise triaging. In this paper, we proposeMulti-relationalHy-                     triaging is to make full use of the patient features (i.e., medical
perbolicDiagnosisPredictor(MHDP)—anovelmulti-relational                            history,examsandsymptomsdescribedbythepatient)collected
hyperbolic graph neural network based approach, to build a dis-                    fromthechatbot conversations.Manyrelatedstudies[6,9,15,16,
easepredictivemodel.Morespecifically,inMHDP,wegeneratea                            18, 21, 22] leverage different deep learning architectures to extract
heterogeneous graph consisting of symptoms, patients and diag-                     unstructured information from healthcare-related datasets such as
nosesnodes,andthenderivenoderepresentationsbyaggregating                           Electronic Health Records (EHRs).
neighborhood information recursively in the hyperbolic space.Ex-                      However,theaboveapproachescanbarelyachievesatisfactory
perimentsconducted ontwo real-worlddatasetsdemonstrate that                        resultsonthemedicaltriagechatbotdatasetduetostrongfeature
theproposedMHDPapproachsurpassesstate-of-the-artbaselines.                         connections among samples. Some prevalent symptoms exist in
                                                                                   mostofthesamples(users)inthedataset,bringingchallengesfor
CCSCONCEPTS                                                                        thetriagingtask.Figure1showsanexampleofsuchcorrelations.To
• Information systems→ Web applications; • Applied com-                            obtainFigure1,wefirstconstructagraphcontainingallsymptoms
puting→Healthcareinformationsystems.                                               as nodes. Then,if two symptomsco-occur on morethan five users,
                                                                                   weaddanedgebetweenthem.Finally,weobservethenumberof
KEYWORDS                                                                           10mostcommonsymptoms’neighborsunderdifferentproximities.
                                                                                   Since there are only 3370 symptoms in the dataset, we can con-
Healthcare; Diagnosis prediction; Graph neural network                             clude two issues from the figure. First, The number of neighbors
                                                                                   increasesexponentiallywiththeproximity.Second,thesecommon
ACMReferenceFormat:                                                                symptomsare associatedwithalmost allsymptomsin thedataset.
Zheng Liu, Xiaohan Li, Zeyu You, TaoYang, WeiFan, and Philip Yu. 2021.             Therefore, the strong symptom connections in the dataset make
Medical Triage Chatbot Diagnosis Improvement via Multi-relational Hy-              thelearneduserrepresentationssimilarandbringdifficultiestothe
perbolic Graph Neural Network . In Proceedings of the 44th International           classification task.
ACMSIGIRConferenceonResearchandDevelopmentinInformationRetrieval                      Toaddressaboveissues,weproposeaMulti-relationalHyper-
(SIGIR ’21), July 11–15, 2021, Virtual Event, Canada.  ACM, New York, NY,
USA, 5 pages. https://doi.org/10.1145/3404835.3463095                              bolicDiagnosisPredictor (MHDP) to learn symptom, user and
                                                                                   diagnosis representations in the hyperbolic space via graph neural
                                                                                   network (GNN). First of all, we can regard the chatbot dataset as a
∗Both authorscontributedequally tothis research.                                   graph with three kinds of nodes: the symptom, user, and diagno-
                                                                                   sis nodes (while all nodes and edges are extracted from raw data
                                                                                   via engineering techniques). To this graph a vanilla GNN can be
Permission to make digital or hard copies of all or part of this work for personal orapplied directly in order to perform triaging (predicting the link
classroom use isgranted without fee providedthat copies are notmade or distributed betweenuseranddiagnosisnodes).Inspiredbytherecentprogress
forprofitorcommercialadvantageandthatcopiesbearthisnoticeandthefullcitation
onthefirstpage.CopyrightsforcomponentsofthisworkownedbyothersthanACM               of hyperbolic graph neural networks [4, 5, 14, 26], we introduce
mustbehonored.Abstractingwithcreditispermitted.Tocopyotherwise,orrepublish,        a method to learn hyperbolic node representations to deal with
topostonserversortoredistributetolists,requirespriorspecificpermissionand/ora      issuesshowninFigure1.Morespecifically,asthenormofapointa
fee.Requestpermissions frompermissions@acm.org.
SIGIR ’21, July 11–15, 2021, Virtual Event, Canada.                                hyperbolicspacegrows,thedistanceofthepointtotheotherpoint
©2021Associationfor Computing Machinery.                                           grows exponentially. This nature provides us with the opportunity
ACMISBN978-1-4503-8037-9/21/07...$15.00                                            of learning separated embeddings and make accurate predictions.
https://doi.org/10.1145/3404835.3463095

                                                                               3   PRELIMINARIES
                                                                                  Graph Neural Network. Theforwardpropagationprocedureofa
                                                                               GNNongraph𝐺(V,E)istoupdatetherepresentationofeachnode
                                                                               𝑣𝑖∈Vthrough neighboring nodes. Suppose for each node𝑖there
                                                                               isainitialnoderepresentation𝒉(0)𝑖   astheinputofthemodel.Then,
                                                                               eachhiddenlayerofGNNlearnsthenoderepresentation𝒉(𝑙)𝑖  from
                                                                               the previous hidden layer𝒉(𝑙−1)𝑖      by aggregating the neighboring
                                                                               nodes as follows:
                                                                                            𝒉(𝑙)𝑖←  𝑨𝒈𝒈𝒓                  𝑗 )},𝒉(𝑙−1)𝑖 ),             (1)
Figure1:Thenumberofneighborswithdifferentordersof                                                    ∀𝑗∈N(𝑖)({𝑻𝒓𝒂𝒏𝒔 (𝒉(𝑙−1)
proximitiesfor10commonsymptomsinthechatbotdataset.                             whereN(𝑖)means the set of neighbors of node𝑖in the graph.
We construct the graph with more than 3000 symptoms.                           Thefeaturetransformationoperation𝑇𝑟𝑎𝑛𝑠(·)istypicallyalinear
This figure shows that common symptoms are related to                          transformation(e.g.,     𝑾(𝑙)𝒉(𝑙−1)𝑗     )andtheneighborhoodaggregation
mostnodesandcanstronglyinfluencethewholegraph.                                 function𝐴𝑔𝑔𝑟(·)sumsneighboring information upandappliesan
                                                                               activation function (e.g.,  𝜎(Í(·))).
Furthermore, MHDP considers multiple kinds of nodes and rela-                     Hyperbolic Manifold. A manifold is a topological space that lo-
tions in the graph, which helps to separate node representations               callyresemblesEuclideanspaceneareachpoint,andthehyperbolic
from different kinds. In summary, our contributions are as follows:            spaceisthecomplete,simplyconnectedRiemannianmanifoldwith
    •To the best of our knowledge, this is the first paper intro-              constantnegativecurvature.WeuseH𝑑ℎ,𝐾ℎ todenoteahyperbolic
       ducing hyperbolic graph neural networks to the medical                  manifold with dimension𝑑ℎ and negative curvature−1/𝐾ℎ. For
       triaging domain. We demonstrate the existence and influ-                each point𝒙  on thehyperbolic manifold,wecan finda Euclidean
       enceofcommonsymptomsinthehealthcare-relateddataset                      (tangent) spaceT𝑥H𝑑ℎ,𝐾ℎ that best approximates the hyperbolic
       and try to eliminate this effect.                                       manifold around𝒙 . Thelogarithmic mapping𝑙𝑜𝑔(·)and theexpo-
    •WeproposeMHDP,amulti-relationalgraphneuralnetwork                         nentialmapping𝑙𝑜𝑔(·)areusedtoconvertvectorsfromH𝑑ℎ,𝐾ℎ to
       tolearnuseranddiagnosisrepresentationsinthehyperbolic                   T𝑥H𝑑ℎ,𝐾ℎ andviceversa,andthedistancefunctiononaRiemannian
       spaces to make medical triage.                                          manifold is defined based on the geodesics function.
    •Experimentsontworeal-worlddatasets(oneofthemispub-                           Here we take the Poincaré ball model as an example. For var-
       lic available) demonstrate that MHDP outperforms many                   ious Riemannian manifolds, we use a Poincaré ball(B,𝑔B𝒙)in
       state-of-the-art methods especially when there are a few                𝑑ℎ dimension whose curvature is -1, where the ball is defined as
       features that are dominant in the dataset.                              B  ={𝒙∈R𝑑ℎ|∥𝑥∥<  1}, and the metric tensor𝑔B𝒙  =𝜆𝒙 𝑰𝑑 with a
                                                                               conformal factor𝜆𝒙  =        21−∥𝒙∥2. Forsuch a hyperbolic manifold,the
2   RELATEDWORKS                                                               distance of two nodes𝒙 ,𝒚∈R𝑑ℎ is
2.1   HyperbolicGraphRepresentationLearning                                              𝑑B(𝒙,𝒚)=𝑎𝑟𝑐𝑜𝑠ℎ(1+2  ∥𝒙−𝒚∥2
Many early studies try to build the connection between artificial                                                   (1−∥𝒙∥2)(1−∥𝒚∥2)),         (2)
neuralnetworks andhyperbolicmanifolds [1,2].Nickel etal.[20]                   where∥·∥denotes the Euclidean norm. For any point𝒙∈B , the
proposethefirstgraphembeddingmethodonaPoincaréball(avari-                      exponentialfunctionmapsapointinEuclideanspacetoapointin
antofhyperbolicspaces)duetoitsadvantageofmodelingcomplex                       hyperbolic space𝑒𝑥𝑝𝒙  :T𝒙 B→ B , and the logarithmic function
tree-likestructures.Later, manystudies[5,14,27]integratehyper-                 maps a point in hyperbolic space back into the Euclidean space
bolic geometry with the graph neural network architecture. Re-                 𝑙𝑜𝑔𝒙  :B→T𝒙 B  as:
cently,thesehyperbolicmethodsarewidelyappliedonmanyareas,
including healthcare [3], recommender system [25, 26], knowledge                               𝑒𝑥𝑝𝐾ℎ𝒙(𝒗)=  𝒙⊕(𝑡𝑎𝑛ℎ(𝜆𝒙∥𝒗∥
graph [4, 13] and language processing [28].                                                                                 2 )𝒗∥𝒗∥),                 (3)
2.2   Graph-basedMethodsonHealthcare                                                     𝑙𝑜𝑔𝐾ℎ𝒙(𝒚)=   2𝜆𝒙𝑎𝑟𝑐𝑡𝑎𝑛ℎ(∥−𝒙⊕𝒚∥)−𝒙⊕𝒚∥−𝒙⊕𝒚∥.         (4)
GraphNeural Networks(GNNs) arewidelyusedinthe healthcare                       where⊕is the Möbius addition for any 𝒙,𝒚∈B  that 𝒙⊕𝒚  =
domain to incorporate unstructured information. GRAM [7] and                    (1+2 ⟨𝒙,𝒚⟩+∥𝒚∥2)𝒙+(1−∥𝒙∥2)𝒚
KAME[19]useknowledgegraphandattentionmechanismtopro-                                 1+2 ⟨𝒙,𝒚⟩+∥𝒙∥2∥𝒚∥2        ,while𝒗 ≠  0isthetangentvectorand
cess external hierarchical ontologies. MiME [8] and GCT [10] are               𝒚  ≠  0 is a point inB.
GNNsdesignedespeciallyforElectronicHealthRecords(EHRs).Liu                     4   THEPROBLEMANDTHESOLUTION
etal.[16]deployheterogeneousgraphstorepresentthehealthcare                     4.1   ProblemDefinition
data. Since graphs can represent more relations and interactions,
theseapproachesareflexibleandcancapturethecomplexstructure                     We are interested in analyzing the relation between a set of𝑀
of health-related data.                                                        symptomsS={𝑆𝑠}𝑀𝑠=1   andthecorrespondingpotentialdiagnosis

                                                        Figure2:TheframeworkofMHDP.
in a set of𝐾 diseasesD ={𝐷𝑑}𝐾𝑑=1  . Given the dataset with𝑁                       4.2.2    HyperbolicAggregation. Inspiredby[5,14],weintroducethe
recordsdenotedas{{𝑆𝑖𝑗}𝑀𝑖𝑗=1,{𝐷𝑖𝑘}𝐾𝑖𝑘=1}𝑁𝑖=1   whereeach{𝑆𝑖𝑗}𝑀𝑖𝑗=1⊂                hyperbolicaggregationfunctionsusedinMHDP.Inthefirststep,
Sis a subset of𝑀𝑖< 𝑀  symptoms and each{𝐷𝑖𝑘}𝐾𝑖𝑘=1⊂Dis a                           weset𝒐 ={p     𝐾ℎ,0,···,0}∈H𝑑ℎ,𝐾ℎ astheorigininthehyperbolic
subset of𝐾𝑖< 𝐾 diseases, the goal is to build a model to fit the                  manifoldH𝑑ℎ,𝐾ℎ, and we map the input features of all nodes to the
trainingdataandpredictthemostpossiblediagnosisgivenasubset                        hyperbolic space centered at 𝒐 by𝑒𝑥𝑝𝐾ℎ𝒐(·)as shown in Figure 2.
ofsymptomsforanewtestrecord.In addition,we denote𝑠,𝑢,and                             Then,weleveragethefollowingequationsasthetransformation
𝑑as the symptom, user, and diagnosis node, respectively.                          and aggregation functions used in Eqs. (5), (6), and (7):
4.2   SolutionApproach                                                                         𝑻𝒓𝒂𝒏𝒔 (𝑙)𝑡1→𝑡2(𝒉)=𝑒𝑥𝑝𝐾ℎ𝒐(𝑾(𝑙)𝑡1→𝑡2𝑙𝑜𝑔𝐾ℎ𝒐(𝒉)),            (8)
                                                                                                                                     Õ
HereweproposeamodelnamedMHDPwithitsarchitectureshown                                 𝑨𝒈𝒈𝒓 N(𝒉)=𝑒𝑥𝑝𝐾ℎ𝒐(𝜎(𝑙𝑜𝑔𝐾ℎ𝒐(𝑒𝑥𝑝𝐾ℎ𝒉(1|N|                𝑙𝑜𝑔𝐾ℎ𝒉(𝒉𝑗))))),
in Figure 2. We first construct a multi-relational graph based on                                                                    𝑗∈N
the chatbot data with three types of nodes (user, symptom and                                                                                             (9)
diagnosis) along with two types of relations (the user-symptom                    where𝒉  is the hyperbolic node representations andNis the set
𝑢−𝑠relationandtheuser-diagnosis𝑢−𝑑relation)asshownonthe                           of its neighbor nodes. In the transformation step, we first map
left-handsideinFigure2.Here,wedefineN(𝑛,𝑡)asallretrieved                          the node embedding to the tangent (Euclidean) space, conduct
neighbors of node𝑛 with relation type𝑡. We then use a multi-                      linear transformation with weight𝑾(𝑙)𝑡1→𝑡2, then map it back to the
relational hyperbolic graph neural network to leverage both the                   hyperbolic space with canter 𝒐. Then, in the aggregation step, we
neighborhoodinformationofthegraphandthecharacteristicsof                          mapallneighbornodeembeddingstothetangentspacecenteredat
hyperbolic space. The multi-relational aggregation procedure and                  𝒉 (thenodewhoseneighborsweareaggregating),andthenconduct
the hyperbolic graph aggregation procedure are described in the                   aggregation and non-linear operations on this space.
following.                                                                        4.2.3    Modeltraining. Afterlearningthehyperbolicembeddingsof
4.2.1    Multi-relational Aggregation. The challenge of our model                 eachnodeinthegraph,wetrytooptimizethemodelbyminimizing
is to learn representations for user nodes that contain a complex                 the distance among positive samples and maximizing the distance
relationshipwithboththesymptomanddiagnosisnode.Hence,we                           among sub-sampled negative samples. The positive samples for
consider aggregating the neighborhood information based on the                    eachuser𝑢meansitspositivediagnosis𝐷+𝑢 ={𝑑+𝑢1,𝑑+𝑢2,···,𝑑+𝑢𝑁+},
relationtypes.Tobespecific,MHDPaggregatesuserembeddings                           where each diagnosis𝑑𝑢𝑖is present, and negative samples are
fromboth𝑢−𝑠relationand𝑢−𝑑relationsinthegraph.Specifically,                        𝐷−𝑢 ={𝑑−𝑢1,𝑑−𝑢2,···,𝑑−𝑢𝑁−}thataresubsampledfromallnon-linked
the aggregation functions are as follows:                                         diagnosis nodes with a user𝑢. Hence, the objective is defined as
        𝒉(𝑙)𝑠←      𝑨𝒈𝒈𝒓                                                                      Õ     Õ                         Õ
                                                 𝑢′)},𝒉(𝑙−1)𝑠 ),     (5)              𝑚𝑖𝑛                   𝑑B(ℎ𝐿𝑢,ℎ𝐿𝑑+𝑢)−           𝑑B(ℎ𝐿𝑢,ℎ𝐿𝑑−𝑢),     (10)
                 ∀𝑢′∈N(𝑠,‘𝑢−𝑠′)({𝑻𝒓𝒂𝒏𝒔 𝑢→𝑠(𝒉(𝑙−1)                                              𝑢   ∀𝑑+𝑢∈𝐷+𝑢                 ∀𝑑−𝑢∈𝐷−𝑢
𝒉(𝑙)𝑢←      𝑨𝒈𝒈𝒓        ({𝑻𝒓𝒂𝒏𝒔𝑠→𝑢(𝒉(𝑙−1)𝑠′)},{𝑻𝒓𝒂𝒏𝒔𝑑→𝑢(𝒉(𝑙−1)𝑑′)},𝒉(𝑙−1)𝑢 ),     whereℎ𝐿𝑢 is the last layer node embedding of a user𝑢,ℎ𝐿𝑑+is the
         ∀𝑑′N(𝑢,‘𝑢−𝑑′),                                                           last layer node embedding of a positive disease𝑑+𝑢, andℎ𝐿𝑑−is the
         ∀𝑠′∈N(𝑢,‘𝑢−𝑠′)
                                                                        (6)       last layer node embedding of a negative disease𝑑−𝑢.
       𝒉(𝑙)𝑑←       𝑨𝒈𝒈𝒓                         𝑢′)},𝒉(𝑙−1)𝑑 ),     (7)          5   EXPERIMENTS
                ∀𝑢′∈N(𝑑,‘𝑢−𝑑′)({𝑻𝒓𝒂𝒏𝒔 𝑢→𝑑(𝒉(𝑙−1)
whereN(𝑢,‘𝑢−𝑠′)denotes the set of neighbor nodes of𝑢whose                         We conduct experiments on two datasets to present the perfor-
typeissymptom.Here𝐴𝑔𝑔𝑟(·)and𝑇𝑟𝑎𝑛𝑠(·)areaggregationand                             mance of our proposed MHDP and baseline methods. One is the
transformation functions.𝑇𝑟𝑎𝑛𝑠𝑡1→𝑡2 is used before aggregating                    onlinemedicaltriagechatbotdatasetcollectedfromTencentMedi-
informationfromnodetype𝑡1 tonodetype𝑡2.Toencodethenode                            pediaservice,which containsmorethantwomillionrecords. The
representation in the hyperbolic graph, we utilize the following                  otherdatasetisconstructedfromtheMIMIC-IIIdataset[12],which
hyperbolic operations.                                                            isapublic-availablecomprehensiveEHRdatasetassociatedwith

         Table1:TheperformanceofMHDPcomparedwithbaselinesindifferentembeddingdimensions(d=10or200).
                                        Dataset             MF-BPR    GraphSAGE    Multi-relationalGCN    Dipole    HGCN                 MHDP
                                ChatBot       Recall@5       0.3555          0.4803                    0.5481                0.4108     0.74270.7734
                    𝑑ℎ=10                       MRR          0.0928          0.2365                    0.3039                0.1445     0.39610.4181
                               MIMIC-III     Recall@10       0.1275          0.3996                    0.4190                0.3625     0.58990.6273
                                                MRR          0.1327          0.2645                    0.3172                0.2729     0.40370.4254
                                ChatBot       Recall@5       0.6428          0.7623                    0.7879                0.6996     0.78420.8004
                  𝑑ℎ=  200                      MRR          0.2744          0.3466                    0.3760                0.2988     0.43060.4566
                               MIMIC-III     Recall@10       0.3515          0.5489                    0.5625                0.5518   0.59170.5811
                                                MRR          0.2657          0.4216                    0.4506                0.3922   0.47830.4716
       Table2:Statisticsofdatasetsinourexperiment.                                    MHDPoutperformsallbaselinesonthechatbotdatasetandMIMIC-
                                                                                      III dataset when𝑑ℎ =  10. It also provides competitive results on
                              Chatbot                      MIMIC-III                  MIMIC-III dataset when dimension𝑑ℎ =  200. The results show
                      #ofnode   Avg.degree     #ofnode   Avg.degree                   that our method can effectively predict the ground truth diagno-
         Symptom/       3,370           4.75      423           33.45                 siswith top-5rankings onchatbotdata (roughly77%)and top-10
          LabTest
        User/Patient  2,285,226          –       24,803            –                  rankingsonMIMIC-IIIdata(roughly63%).Forbothdatasets,when
         Diagnosis       307              1       108             5.9                 theembeddingdimensionislow(𝑑ℎ=  10),MHDPoutperformsthe
                                                                                      GraphSAGE by 30%, and all other models such as Multi-relational
                                                                                      GCN and dipole in terms of MRR and recall.Eventhough the per-
overfortythousandpatients.TomimictheEHRdatasetasachatbox                              formance improvement on𝑑ℎ =  200 is not as significant as it is
dataset represented in Figure2, we use methods in[16] to prepro-                      on𝑑ℎ =  10, the model still shows promising results. Apart from
cessthedatasetandextract423labtestitemsassymptomsand108                               MHDP, HGCN achieves the second-best performance among all
diagnosesaslabelsfrom24,803admissionrecords.Statisticsofboth                          baselines,whichindicatestheeffectivenessoflearninghyperbolic
datasets after preprocessing are shown in Figure 2.                                   embeddings.However,HGCNfailstomodelvariousrelationsand
   Implementation Details. WethendeploytheMHDPwiththree                               thusperforms worsethan MHDP.Multi-relationalGCN iscapable
layersofhyperbolicaggregatorsonPoincarémanifoldswithdimen-                            oflearningtherelationinformation,butlearningembeddingsinthe
sion 10and 200.We usegrid searchwith cross validationto select                        Euclideanspacecausesadistortionwhenthedimensionislow.Our
thebestparameters.Theratiooftraining,testing,andvalidationis                          proposed MHDP outperforms all baselines because it can capture
set to 7:2:1.                                                                         relation betweensymptom and user anddiagnosis and user, in the
                                                                                      meanwhile, it learn embeddings in the hyperbolic space.
5.1   Baselines&EvaluationMetric                                                      6   CONCLUSIONANDFUTUREWORK
In this paper, we compare MHDP with the following baselines:                          In this paper, we explored the challenges of medical triaging tasks.
     •Matrix Factorization with BPR loss (MF-BPR)[23]. A basic                       We demonstrated the strong connection between features in the
        method to learn embeddings for symptoms and diagnoses.                        medical chatbot dataset and devised MHDP, a graph neural net-
     •GraphSAGE [11]. An inductive GNN with Mean aggrega-                            work architecture to learn user representations in the hyperbolic
        tortolearnrepresentationsforhomogeneousgraphsinEu-                            spaces. By learning hyperbolic embeddings for different types of
        clidean space.                                                                nodes, MHDP can accurately predict the link between users and
     •Multi-relationalGCN[24].AGNNthatcanleveragerelation                             diagnoses, especially when the network we leverage exhibits a
        information to multi-relational graphs.                                       scale-free structure.
     •Dipole [17]. A predictive model designed for EHR based on                          Inthefuture,wewillfurtherfocusonthelatentstructurewithin
        bidirectional RNN.                                                            differentkindsofhealth-relateddatasuchasthechatbotdata,EHRs,
     •HGCN[5].AGNN learnsrepresentationsinthehyperbolic                               andmedicalimages.Previously,studiesontheseareassimplyregard
        spaces which is designed for homogeneous graphs.                              triaging as a classification task, and design model architectures
   We compare the performance of MHDP and baselines based on                          merely accordingto the data structure. However,our study shows
Recall@5, Recall@10, and Mean Reciprocal Rank (MRR) metrics,                          that different datasets have different latent structures and even
which measures the quality of the single highest-ranked relevant                      distinctdatadistribution.Therefore,agoodmodelneedstoexplore
items.                                                                                the latent nature of the dataset and capture the spirit of the data.
5.2   Performance                                                                     7   ACKNOWLEDGEMENT
Table 1 demonstrates the performance of the proposed MHDP and                        This work is supported in part by NSF under grants III-1763325,
the compared other baselines. We conduct experiments on both                          III-1909323,andSaTC-1930941,andNationalKeyR&DProgramof
the chatbot dataset and the EHR dataset using hyperbolic space                        China 2018YFC0117000.
dimension𝑑ℎ=  10and𝑑ℎ=  200.AsshowninTable1,ourproposed

REFERENCES                                                                                                [14]  Qi Liu, Maximilian Nickel, and Douwe Kiela. 2019.  Hyperbolic graph neural
 [1]  George A Anastassiou. 2011. Multivariate hyperbolic tangent neural network                                 networks. NeurIPS (2019).
      approximation. Computers&MathematicswithApplications61,4(2011),809–821.                             [15]  ZhengLiu,XiaohanLi,HaoPeng,LifangHe,andSYuPhilip.2020.Heterogeneous
 [2]  Sven Buchholz and Gerald Sommer. 2000. A hyperbolic multilayer perceptron.                                 Similarity Graph Neural Network on Electronic Health Records. In 2020 IEEE
      In Proceedings of the IEEE-INNS-ENNS International Joint Conference on Neural                              International Conference on Big Data (Big Data).IEEE,1196–1205.
      Networks. IJCNN 2000. Neural Computing: New Challenges and Perspectives for                         [16]  ZhengLiu,XiaohanLi,HaoPeng,LifangHe,andPhilipSYu.2021.Heterogeneous
      the New Millennium, Vol. 2.IEEE,129–133.                                                                   Similarity Graph Neural Network on ElectronicHealth Records. arXiv preprint
 [3]  PengfeiCao,YuboChen,KangLiu,JunZhao,ShengpingLiu,andWeifengChong.                                          arXiv:2101.06800 (2021).
      2020. HyperCore:HyperbolicandCo-graphRepresentation forAutomaticICD                                 [17]  Fenglong Ma,RadhaChitta,JingZhou,QuanzengYou, TongSun,and JingGao.
      Coding.InProceedingsofthe58thAnnualMeetingoftheAssociationforComputa-                                      2017. Dipole:Diagnosispredictioninhealthcareviaattention-basedbidirectional
      tional Linguistics. 3105–3114.                                                                             recurrentneural networks.In KDD.1903–1911.
 [4]  InesChami,Adva Wolf, Da-ChengJuan,FredericSala,Sujith Ravi,andChristo-                              [18]  Fenglong Ma, Quanzeng You, Houping Xiao, Radha Chitta, Jing Zhou, and Jing
      pherRé.2020. Low-dimensionalhyperbolicknowledgegraphembeddings. arXiv                                      Gao. 2018. Kame: Knowledge-based attention model for diagnosis prediction in
      preprint arXiv:2005.00545 (2020).                                                                          healthcare.InProceedingsofthe27thACMInternationalConferenceonInformation
 [5]  Ines Chami, Rex Ying, Christopher Ré, and Jure Leskovec. 2019.  Hyperbolic                                 and Knowledge Management.743–752.
      graph convolutional neural networks. Advances in neural information processing                      [19]  Fenglong Ma, Quanzeng You, Houping Xiao, Radha Chitta, Jing Zhou, and Jing
      systems 32(2019),4869.                                                                                     Gao. 2018. KAME: Knowledge-based Attention Model for Diagnosis Prediction
 [6]  EdwardChoi,MohammadTahaBahadori,LeSong,WalterFStewart,andJimeng                                            inHealthcare.In CIKM.743–752.
      Sun. 2017. GRAM: graph-based attention model for healthcare representation                          [20]  Maximilian Nickel and Douwe Kiela. 2017. Poincaré embeddings for learning
      learning. In Proceedings of the 23rd ACM SIGKDD international conference on                                hierarchicalrepresentations.In Proceedings of the 31st International Conference
      knowledge discovery and data mining. 787–795.                                                              on Neural Information Processing Systems.6341–6350.
 [7]  EdwardChoi,MohammadTahaBahadori,LeSong,WalterF.Stewart,andJimeng                                    [21]  Zhi Qiao, XianWu,Shen Ge, and Wei Fan.2019. Mnn:multimodal attentional
      Sun.2017. GRAM:Graph-basedAttentionModelforHealthcareRepresentation                                        neural networksfordiagnosis prediction. IJCAI 1(2019),A1.
      Learning.In KDD. 787–795.                                                                           [22]  ZhiQiao,ZhenZhang,XianWu,ShenGe,andWeiFan.2020. MHM:Multi-modal
 [8]  Edward Choi, Cao Xiao, Walter F. Stewart, and Jimeng Sun. 2018. MiME: Multi-                               ClinicalDatabasedHierarchicalMulti-labelDiagnosisPrediction.In Proceedings
      levelMedicalEmbeddingofElectronicHealthRecordsforPredictiveHealthcare.                                     of the 43rd International ACM SIGIR Conference on Research and Development in
      In NIPS 2018.4552–4562.                                                                                    Information Retrieval.1841–1844.
 [9]  Edward Choi, Zhen Xu, Yujia Li, Michael Dusenberry, Gerardo Flores, Emily                           [23]  SteffenRendle,ChristophFreudenthaler,ZenoGantner,andLarsSchmidt-Thieme.
      Xue, and Andrew Dai. 2020.   Learning the graphical structure of electronic                                2012. BPR:Bayesianpersonalizedrankingfromimplicitfeedback. arXiv preprint
      health records with graph convolutional transformer. In Proceedings of the AAAI                            arXiv:1205.2618 (2012).
      Conference on Artificial Intelligence, Vol. 34. 606–613.                                            [24]  Michael Schlichtkrull, ThomasN Kipf, PeterBloem, Rianne VanDen Berg, Ivan
[10]  EdwardChoi,ZhenXu,YujiaLi,MichaelWDusenberry,GerardoFlores,Emily                                           Titov,andMaxWelling.2018. Modelingrelationaldatawithgraphconvolutional
      Xue,andAndrewMDai.2020. LearningtheGraphicalStructureofElectronic                                          networks.In European semantic web conference.Springer,593–607.
      Health Records with Graph Convolutional Transformer. In Proceedings of the                          [25]  Jianing Sun, Zhaoyue Cheng, Saba Zuberi, Felipe Pérez, and Maksims Volkovs.
      Thirty-Second AAAI Conference on Artificial Intelligence. AAAI Press.                                      2021.HGCF:HyperbolicGraphConvolutionNetworksforCollaborativeFiltering.
[11]  WilliamLHamilton,RexYing,andJureLeskovec.2017. Inductiverepresentation                                     (2021).
      learningonlargegraphs.In NIPS. 1025–1035.                                                           [26]  LucasVinhTran,YiTay,ShuaiZhang,GaoCong,andXiaoliLi.2020. HyperML:a
[12]  Alistair EW Johnson, Tom J Pollard, LuShen, H Lehman Li-Wei,Mengling Feng,                                 boostingmetriclearningapproachinhyperbolicspaceforrecommendersystems.
      Mohammad Ghassemi,Benjamin Moody, PeterSzolovits, Leo AnthonyCeli, and                                     InProceedingsofthe13thInternationalConferenceonWebSearchandDataMining.
      RogerGMark.2016. MIMIC-III,afreelyaccessiblecriticalcaredatabase. Scientific                               609–617.
      data 3,1 (2016),1–9.                                                                                [27]  Yiding Zhang, Xiao Wang, Xunqiang Jiang, Chuan Shi, and Yanfang Ye. 2019.
[13]  ProdromosKolyvakis,AlexandrosKalousis,andDimitrisKiritsis.2019. Hyperkg:                                   Hyperbolicgraphattentionnetwork. arXiv preprint arXiv:1912.03046 (2019).
      Hyperbolicknowledgegraphembeddingsforknowledgebasecompletion. arXiv                                 [28]  Yudong Zhu, Di Zhou, Jinghui Xiao, Xin Jiang, Xiao Chen, and Qun Liu. 2020.
      preprint arXiv:1908.04895 (2019).                                                                          HyperText: Endowing FastText with Hyperbolic Geometry.   arXiv preprint
                                                                                                                 arXiv:2010.16143 (2020).

