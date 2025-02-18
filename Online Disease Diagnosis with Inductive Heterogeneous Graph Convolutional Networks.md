 OnlineDiseaseDiagnosiswithInductiveHeterogeneousGraph
                                                ConvolutionalNetworks
                 Zifeng Wang                                            Rui Wen                                           Xi Chen∗
      wangzf18@mails.tsinghua.edu.cn                             ruiwen@tencent.com                              jasonxchen@tencent.com
           TBSI, Tsinghua University                              Tencent Jarvis Lab                                 Tencent Jarvis Lab
                       China                                              China                                              China
                   Shilei Cao                                    Shao-Lun Huang                                         Buyue Qian
            eliasslcao@tencent.com                      shaolun.huang@sz.tsinghua.edu.cn                          qianbuyue@xjtu.edu.cn
               Tencent Jarvis Lab                            TBSI, Tsinghua University                           Xi’an Jiaotong University
                       China                                              China                                              China
                                                                    Yefeng Zheng
                                                             yefengzheng@tencent.com
                                                                  Tencent Jarvis Lab
                                                                          China
ABSTRACT                                                                          ACMReferenceFormat:
WeproposeaHealthcareGraphConvolutionalNetwork(HealGCN)                            Zifeng Wang, Rui Wen, Xi Chen, Shilei Cao, Shao-Lun Huang, Buyue Qian,
to offer disease self-diagnosis service for online users based on                 andYefengZheng.2021.OnlineDiseaseDiagnosiswithInductiveHeteroge-
Electronic Healthcare Records (EHRs). Two main challenges are                     neousGraphConvolutionalNetworks.In Proceedings of the Web Conference
focusedinthispaperforonlinediseasediagnosis:(1)servingcold-                       2021 (WWW ’21), April 19–23, 2021, Ljubljana, Slovenia.  ACM, New York,
                                                                                  NY, USA, 10 pages. https://doi.org/10.1145/3442381.3449795
startusersviagraphconvolutionalnetworksand(2)handlingscarce
clinical description via a symptom retrieval system. To this end,                 1   INTRODUCTION
we first organize the EHR data into a heterogeneous graph that is                 ElectronicHealth Records(EHRs)aredocumented informationon
capableofmodelingcomplexinteractionsamongusers,symptoms                           many clinical events that occur during a patient’s stay and visit
anddiseases,andtailorthegraphrepresentationlearningtowards                        in the hospital. Recently, the advancement of machine learning
diseasediagnosiswith aninductivelearning paradigm.Then,we                         sheds light on building an alternative substitute “robot doctor”,
build a disease self-diagnosis system with a corresponding EHR                    which benefits from massive EHR data and empirically learns a
Graph-basedSymptomRetrievalSystem(GraphRet)thatcansearch                          diseasediagnosismodelbasedonthegrowingcollectionsofclinical
andprovidealistofrelevantalternativesymptomsbytracingthe                          observations.Automaticdisease diagnosiscanbenefit alot inthe
predefinedmeta-paths.GraphRet helpsenrichtheseedsymptom                           currentmedicinesystem.Forexample,primarycareproviders,who
set through the EHR graph when confronting users with scarce                      areresponsibleforcoordinatingpatientcareamongspecialistsand
descriptions, hence yield better diagnosis accuracy. At last, we                  other care levels, often confront a mix of diseases which should
validate the superiority ofour modelon alarge-scale EHRdataset.                   belong to various departments. In this circumstance, they could
CCSCONCEPTS                                                                       refer to the automatic online disease diagnosis results as useful
                                                                                  advice and decide which level of healthcare service patients would
• Computing methodologies→ Natural language processing; •                         need.Moreover,userscanalsogetaccesstothisonlinediagnosis
Information systems→ Data mining; Data mining; Web ap-                            system for getting advice based on their chief complaints, then
plications.                                                                       decide which hospitaland department they would liketo go for a
                                                                                  visit. In a nutshell, an advanced automatic online diagnosis system
KEYWORDS                                                                          can boost the efficiency of the existing medicine system.
diseasediagnosis,graphneuralnetwork,onlinehealthcareservice                          SomedocumentedEHRswererecentlymadepubliclyavailable,
                                                                                  e.g.,theMIMIC-III[23]andCPRD[18],whichencourageasurgeof
                                                                                  research in developing automatic clinical expert systems. Existing
∗Corresponding author: jasonxchen@tencent.com                                     works in utilizing EHRs for disease diagnosis include three gen-
                                                                                  res:featureengineeringassociatedwithaclassifierfortheaimof
                                                                                  predictingdiseaseoutcomes[4,13,35,39];unsupervisedrepresenta-
This paper is published under the Creative Commons Attribution 4.0 International  tionlearningbasedonrawnotes[6,8,28]bymodelingthesemantic
(CC-BY 4.0) license. Authors reserve their rights to disseminate the work on theirrelations between diseases and symptoms for downstream tasks;
personal and corporate Web sites with the appropriate attribution.
WWW ’21, April 19–23, 2021, Ljubljana, Slovenia                                   and graph-based methods [20, 21] by modeling the EHR data with
© 2021 IW3C2 (International World Wide Web Conference Committee), published       graphs, then attempting to tailor embedding learning specifically
under Creative Commons CC-BY 4.0 License.                                         fordiseasediagnosis.Inthiswork,wefollowtheideaofmodeling
ACM ISBN 978-1-4503-8312-7/21/04.
https://doi.org/10.1145/3442381.3449795                                           EHR data into Heterogeneous Information Network (HIN) [16].

WWW ’21, April 19–23, 2021, Ljubljana, Slovenia                                                       Zifeng Wang, Rui Wen, Xi Chen, Shilei Cao, Shao-Lun Huang, Buyue Qian, and Yefeng Zheng
Figure 1: Demonstration of (a) inductive learning that a pa-
tientu∗needstobeincorporatedintothegraphand(b)guid-
ingself-diagnosisbyexploringpossiblesymptomsthrough
meta-path.Here,u,sandd representtheuser,symptom,and                                Figure2:Theflowchartoftheautomaticdiagnosissystem.
diseasenode,respectively,inaHeterogeneousInformation
Network(HIN).                                                                         The scarcity of clinical descriptions is another obstacle in de-
                                                                                   veloping an accurate self-diagnosis system for practical use, but
Weperformgraphrepresentationlearningwithalinkprediction                            haslongbeenignoredintheliterature.Notesbyphysicians,which
paradigmbyGraphConvolutionalNetworks(GCNs)[27].HINis                               synthetically reflect patients’ physical conditions, are core ingre-
inherently suitable for modeling the complex interactions among                    dients of former models [20, 21]. However, without professional
users,symptomsanddiseasesintheEHRdata.Itallowstoanalyze                            knowledge in medicine, an ordinary patient cannot provide ac-
the diagnosis results conveniently by tracing the links between                    curate descriptions about symptoms but could merely present a
nodes of symptoms and diseases through an interpretable way.                       colloquial description (such as high temperature and feel sick) to
   It should be noted that our work emphasizes ondiseasediag-                      thetargetsymptom,whichmightbedistinctivelydifferentinthe
nosisratherthanindividualdiseaseriskprediction, suchthat                           clinicaldefinition. Inthiscircumstance, weleveragean nameden-
it differs from most previous deep learning based methods, e.g.,                   tityrecognition(NER)systemtoextractkeysymptomsfromuser
DoctorAI[5],RETAIN[7],Dipole[30],etc.Theyleveragerecurrent                         inputs,thenguideuserstooffer more symptomsbyexploitingthe
neural network (RNN) to model sequential EHR data by examining                     learned HIN, as shown in Fig. 1(b). According to the meta-path
each user’s historical visits for predicting his/her future disease                symptom-disease-symptom, we can find out that symptoms3 is
risk. Nonetheless, all of them are transductive and unable to deal                 a two-hop neighbor ofu∗, starting froms1 which is directly con-
with cold-start users, i.e., users do not have historical records of               nectedtou∗.Asfarasweknow,fewworkshavebeendevotedtoit
hospital visits.                                                                   in the literature.
   Unfortunately, in online self-diagnosis, we often have little                      In summary, our main contributions are highlighted as:
prior knowledge about individuals compared with those in the                       •WeproposeHealGCN,aninductiveheterogeneousGCN-based
publicMIMIC-IIIorCPRDdatasets.Usersaccesstoaweb-baseddi-                              disease diagnosis model, towards serving cold-start users by
agnosissystemwhereneitherpersonalinformationiscollectednor                            mining complex interactions in the EHR data.
their previous logs are found. The system should make a decision                   •Webuildasyntheticdiseaseself-diagnosissystembasedonHeal-
purely based on questioning and answering with newcomers. It                          GCN, medical NER system, and symptom retrieval system for
couldbeexpectedinformationprovidedbyasingleuserisfarfrom                              serving online users.
enough to make an accurate and confident diagnosis. Such that,                     •We verify the effectiveness of the proposed HealGCN and the
exploiting the HIN to get information from the neighbors should                       correspondingsymptomretrievalsystemonreal-worldEHRdata
benefit a lot. However, recent graph based disease risk prediction                    and in online A/B test.
methods [19– 21] all fall short in transductive learning paradigm,
whichrelyonuser’shospitalvisitsinvolvingthedetailedclinical                        2   RELATEDWORK
notes written by physicians, physical examination, etc., thus not                  A series of works have been devoted to disease diagnosis by ma-
applicable to online self-diagnosis.                                               chine learning. Traditional wisdom focused on feature engineering
   To deal with the above-mentioned challenge, we propose to                       withdomainknowledge,alongwithstatisticalclassificationmodels
employ inductive learning [15]. Anewpatient’s embedding isgen-                     fordiseaseprediction[29,37,41,44].Thesemethodsoftenrequire
erated by inductive graph convolutional operation tracing the pre-                 intensive labor indata preprocessing and professional knowledge
defined meta-path. Thisprocess encodes the node information by                     todesigndiscriminativehandcraftedfeatures.Inspiredbytheemer-
exploitingthehigh-orderconnectivity.AsshowninFig.1(a),anew                         genceofpowerfultechniquesinnaturallanguageprocessing(NLP),
patient nodeu∗is incorporated in the existing EHR graph by its                     e.g.,Word2Vec[31],manyfollowupswereproposedtomineclini-
connection with two symptom nodess1 ands2. We can generate                         cal notes through an unsupervised word representation learning
embeddingofu∗byanaggregationoperationfromitstwoneigh-                              scheme,inordertosupportthedownstreamdiseasepredictiontask
bors, e.g., taking average of the embeddings ofs1 ands2. Moreover,                 engaged with a multi-class classification model [6, 10, 13, 33]. Re-
we can involve neighbors ofs1 ands2 into aggregation, e.g., users                  centadvancementinpretrainedlanguagemodels,e.g.,BERT[11],
u1 andu2 who are so-called two-hop neighbors ofu∗with respect                      further fueled research in formulating EHR data parsing as text
to the meta-path user-symptom-user.                                                processing[28].However,theseself-supervisedlearningmethods

Online Disease Diagnosis with Inductive Heterogeneous Graph Convolutional Networks                                                               WWW ’21, April 19–23, 2021, Ljubljana, Slovenia
onlylearngeneralembeddings,whichareunnecessarilyoptimalfor                        3.2   ProblemSetup
diseasediagnosis.Besides,theyignoretherichstructuralinforma-                      DifferentfrommanypreviousGCNworksthattrytomodeldisease
tionofEHRdata,e.g.,theinteractionsamongusers,symptomsand                          diagnosis as a node classification task, we formulate the disease
diseases,whichshouldbeusefultobuildaccurateandinterpretable                       diagnosis process as link prediction between user nodesU and
diagnosis systems.                                                                disease nodesD. Similar to the standard collaborative filtering
   In order to utilize rich semantics contained in EHR data, Het-                 scheme,findingthemostpossiblylinkeddiseased withtheuseru
eoMed[20]proposedHIN-basedgraphneuralnetworkstargeting                            amounts to solving the following problem
fordiseasediagnosis,followedbyworksinadoptingGraphConvo-                                               d =  argmax
lutionalTransformer(GCT)[9]andattentionGCN[21].However,                                                       d′∈D ˆyu  ,d′=  q⊤u qd′,                           (1)
thesemethods arenotcapable ofhandlingcold-start usersbecause                      whereqistheembeddingofanode.Here, ˆyu  ,d  isthepredictedscore
theymodelpatientembeddingsbasedonabundanthistoricalclini-                         thatmeasureshowpossibleuseru isaffectedbydiseased,hencewe
calevents,thusbeingnotabletohandlepatientswithouthistorical                       candeploytop-K disease“recommendations”inourself-diagnosis
visits. In this work, we leverage aninductive heterogeneous GCN                   system for users, based on the ranked scores.
to resolve this challenge.                                                           However,wedonotdirectly assignanembeddingqu  foreach
   Therewerealotofresearchworksonthevectorizationofclinical                       user; instead, we only maintain a trainable embedding matrix of
concepts, including patients [12, 43, 45], doctors [2] and medical                symptoms and diseases:
notes [14, 40], for clinical information retrieval. However, many
ofthemrelyondeeparchitecturesforextractingembeddingsfor                                       Q=[qd  1,...,qd|D|,    qs1,...,qs|S|].             (2)
pair-wise comparison, which are not easy to scale up to a large                                       |           {z            }|          {z           }
number of alternative symptoms, or are not targeting specifically                                    diseases embeddings symptoms embeddings
for symptom-level retrieval. Oncontrast, we exploit meta-path in                  The main reason is that we have limited knowledge about indi-
the EHR graph with GCNs for generating symptom embeddings,                        viduals in online self-diagnosis. For instance, most of users are
whicharethenexploitedforsymptomretrievalintheself-diagnosis                       cold-start, and the number of users can be very large compared
question and answer (Q&A) system. Our retrieval system takes                      withsymptomsanddiseases.Inthisscenario,maintainingembed-
diseaseasintermediatenodetobridgesymptoms,whichconstraints                        dingsofuserscausesdifficultyinoptimizationonthosetremendous
thesearchtomorerelevantsymptomsandincreasestheretrieval                           numberofparameters.Instead,weproposetorepresentusersby
diversity.                                                                        the linked symptoms. By exploiting the meta-path methods, we
                                                                                  minetheusers’positionandthestructureoftheirneighborhoodto
3   METHOD                                                                        build the embeddings, which reduces the optimization complexity
The overall flowchart of the proposed automatic diagnosis system                  significantly.
is shown by Fig. 3. It encompasses three main steps: question and
answer, inference for diagnosis, and diagnosis results display. In                3.3   Meta-PathGuidedNeighborSampling
this section, we present technical details of the inference compo-                In order to fulfill the potential of the EHR graph for representation
nent. Thefirst Q&Acomponent willbe introducedin§4. We first                       learning,unlikeHosseinietal.[20]thatobtainsembeddingsbyag-
introducehowtobuildanHINbasedontheEHRdataandadaptthe                              gregationonone-hopneighbors,weexploithighorderconnections
graphical reasoning for disease diagnosis. After that, we elaborate               bymeta-path.Weincludetwometa-pathsinourmodel:Disease-
onthedetailsofneighborsamplingthroughthemeta-path,andthe                          Symptom-Disease(DSD)andUser-Symptom-User(USU).Techni-
embedding propagation and aggregation process, on the basis of                    cally, given a meta-pathρ, we define thei-hop neighborhood of a
messagefunctions.Atlast,wepresenttheideaofemployingacon-                          nodev asNiρ(v). For instance, in Fig. 1(a), when exploiting neigh-
trastivelossfunctionforoptimizationandutilizinghardnegative                       borsofu∗inlinewiththeUSUmeta-path,wecanobtainneighbors
example mining for long-tailed distribution.                                      asN1USU(u∗)={s1,s2},N2USU(u∗)={u1,u2}andN3USU(u∗)={s3};
                                                                                  whenhandlingd1 bytheDSDmeta-pathinFig.1(b),wecanobtain
3.1   BuildingtheEHRHINGraph                                                      N1DSD(d1)={s1,s2}andN2DSD(d1)={d2,d3}.Inpractice,toreduce
A homogeneous graph is denoted byG=(V,E)that consists of                          computationalburden,weoftenrestrictthemaximumnumberof
two elements: nodev∈Vand edgee =(v,v′)∈E. By contrast,                            neighborsinmeta-pathguidedsampling,i.e.,uniformlysampling
ourgraph builtonEHR datais heterogeneous,i.e., thereare three                     fromtheneighborsiftheneighbornumberexceedsthethreshold.
classes of nodes, representing symptom s, useru, and disease d,                   For example, if we set the maximum neighbor number for each3USU(u)|}
respectively,thusthreetypesofedges:(u,s),(u,d)and(d,s).These                      node as 5 in USU, then max{|N                 =  5×5×5 =  125. As we
threeedgetypesrepresentcommoninteractionsintheEHRdata.In                          considerhighorderinteractionsbymeta-path,itrealizesreasoning
particular,edges(u,s)and(u,d)existwhenuseru reportssymptom                        by involving the rich structural information in the graph. In the
sandisaffectedbydiseased,and(d,s)reflectsthatdiseasedappears                      sequel,wewilldiscusshowtodesignthemessageconstructionand
togetherwithsymptoms,accordingtoobservationsintheEHRdata.                         passing functions for it.
ThenodesetofourEHRgraphisacombinationofusers,symptoms                             3.4   EmbeddingPropagation&Aggregation
anddiseases,asV={U,S,D}.Besides,theneighborhoodofnode
v isN(v)={v′∈V|(v,v′)∈E}, from which we will propose                              Wenextelaborateontheembeddingpropagationmechanisminour
the meta-path guided reasoning in the sequel.                                     framework, which encompasses two main components: message

WWW ’21, April 19–23, 2021, Ljubljana, Slovenia                                                       Zifeng Wang, Rui Wen, Xi Chen, Shilei Cao, Shao-Lun Huang, Buyue Qian, and Yefeng Zheng
                                     Figure3:OverallflowchartoftheproposedHealGCNframework.
construction and message passing.Then,weextendthis technique
to high order embedding propagation.
   Fig.3illustratestheoverallflowchartoftheembeddingpropaga-
tionprocess, throughboth theUSUand DSDmeta-paths. Itcanbe
identifiedthatGCNfollowsalayer-wisepropagationmanner,with
layer0fromthecurrentnodeandgraduallyincreasingalongthe
meta-path. Taking thedisease-symptom (DS) graphas an example,
ineachlayer,newembeddingofanodev isestablishedonmessages
fromitsone-hopneighborsv′∈N(v),aswellasaprojectionfrom
itsembedding.Specifically,thisprocessonalayerl canbewritten
as
                                             Õ
             qlv = ϕ©­mlv←v +       1|N(v)|         mlv←v′ª®,             (3)
                     «                    v′∈N(v)           ¬
whereϕ(·)is an activation function, e.g., tanh, and the message
construction ofmv←v andmv←v′are                                                    Figure 4: The online self-diagnosis system: (a) a user in-
                                                                                   putstheinitialdescriptions;(b)theguidancesystemrecom-
                 mlv←v =  Wl1qv,                                                   mends several alternative symptoms for user to select; (c)
                mlv←v′=  Wl1ql+ 1v′ + Wl2(qv⊙ql+ 1v′).                   (4)       after several rounds of Q&A, the system collects sufficient
                                                                                   informationandmakesadiagnosis.
Here,W1 is responsible for projectingqv andqv′into the same
space,andtheelement-wiseproduct⊙measuressimilaritybetween                          propagationgoestotheoutputlayerl =  0,ityieldsthefinalembed-
theqv andqv′. In the DS graph, the propagation follows the rule                    ding of the diseaseq∗d  :=  q0d  and the userq∗u  :=  q0u , which will be
mentionedinEqs.(3)and(4),asthemessagesinvolvems←d ,md←s,
md←d  andms←s.Specifically,theinitialembeddingofdiseaseis                          utilized for score estimation similar to Eq. (1), i.e., ˆyu  ,d  =  q∗⊤u   q∗d .
qLd  :=  qd , which comes from the parametric embedding matrixQ.
   Themessagepropagationisslightlydifferentintheuser-symptom                       3.5   Optimization
(US)graphbecausewedonothavetrainableembeddings forusers.                           WeadopttheBayesianPersonalizedRanking(BPR)loss[36]forthe
Instead, userembeddingonly depends on the message passing and                      model optimization, which is contrastive, such that it encourages
aggregationfromsymptomnodes.Inotherwords,auser’sembed-                             thelearnedembeddingsinformativefordiscriminatingpositiveand
dingisgeneratedbytheneighboringsymptoms.Thisdesignallows                           negative interactions. Specifically, the BPR loss is
inductive learning that aims for coping with cold-start users. In                                         Õ
particular, when the message is targeted to users,ql−1u     is                                LBPR =              −logσ(ˆyu  ,d−ˆyu  ,d′)+λ∥Θ∥22,          (6)
                                         Õ                                                             (u  ,d  ,d′)∈Ω
                     ql−1u    =       1|N(u)|  mlu←s                               where Ω   ={(u,d,d′)|(u,d)∈Ω + ,(u,d′)∈Ω−}represents
                                       s∈N(u)                            (5)       the pairwise training set; Ω +  and Ω−are the observed and un-
                          where   mlu←s =  Wl1qs.                                  observed interaction sets, respectively;σ(·)is the sigmoid func-
                                                                                   tion,σ  : R7→(0,1); Θ   denotes the trainable parameters, i.e.,
Since we do not define the initial embeddings for users, no prior                  Θ  ={Q,{Wl1,Wl2}Ll= 1}; andλis a hyperparameter that controls
information of them is required during this process. When the                      the imposedℓ2-regularization. During training, we apply message

Online Disease Diagnosis with Inductive Heterogeneous Graph Convolutional Networks                                                               WWW ’21, April 19–23, 2021, Ljubljana, Slovenia
dropout[38]toalleviateoverfitting.Specifically,wesetaprobabil-
ity ofp to randomly drop the elements in messages in Eq. (4).
   Moreover,negativesamplingisindispensableforoptimizingthe
BPR loss. Since the distribution of disease is highly long-tailed,
insteadof uniformlysampling negativeexamplesfromtheentire
setofdiseases,weconductanonlinehardnegativeexamplemining
similar to [42], i.e., hard examples are dynamically generated at
theendofeachepoch.Tothisend,foreachpositiveuser-disease
pair(u,d)∈Ω + , we look for a negative example(u,d′)∈Ω−by
rankingdiseasesaccordingtotheirsimilaritywithrespecttothe
positive diseased, and pick the top rankedd′as a hard example.
Thesimilaritybetweendiseaseshereisdefinedbycosinesimilarity
of embeddings
                      Sim(d,d′)=       q⊤d qd′∥qd∥2∥qd′∥2.                           (7)
Thesehardnegativeexamplesaremore challengingforthemodel                            Figure 5: An example of the raw clinical note and the ex-
to rank, thus increasingits capability in discriminating diseases at               tractedentitiesfromit.
a fine granularity.
                                                                                   Algorithm1 Disease Diagnosis with GraphRet and HealGCN.
4   SELF-DIAGNOSISSYSTEM                                                           Require: PretrainedgraphembeddingsQ;TheEHRgraphG;The
We develop an online disease self-diagnosis system, as its front                        GraphRet systemR(·);
page shownin Fig. 4.Unlike learning andpredicting on EHRdata,                        1: Get embeddings of all diseasesQ∗d← ForwardDSD(D;G,Q);
information offeredby users in onlinesystem is usually very lim-                     2: Receive the seed symptom setS0 from the user;
ited:whenengagedinthissystem,auseroftendescribeshisorher                             3: for epocht =  1→T do
feelings ina non-professional andcolloquialform.This ambiguous                       4:       Retrieve symptoms by GraphRet ˜S← R(St−1,G);
and limited information causes extreme uncertainty for decision                      5:        ˆS← UserSelect(˜S);
making.Todealwithit,wefirstbuildanNERsystemforextracting                             6:       Update symptom setSt←St−1∪ˆS;
entitiespertinenttosymptomsfromtherawuserdialog.Thenthe                              7:       Get the user embeddingq∗u← ForwardUSU(ST ;G,Q);
extractedsymptomsaremappedto a standard symptom set which                            8:       Do inference by  ˆy←Q∗d q∗u∈R|D|;
is alignedwith symptom nodes inthe built HIN. We furtherbuild                        9:       Exit the loop if the diagnosis confidence is high enough;
a guiding system to lead users to provide more information. The                     10: endfor
systemdisplaysseveralsymptomspertinenttotheinitialdescrip-
tion, from which the user picks the most relevant ones. It could
be expected that after several rounds of dialog, it collects enough                   Basedonthislabeledcorpus,wetrainaBiLSTM-CRF[22]model
symptomsoftheuserandisconfidenttomakethefinaldiagnosis.                            for NER task. In addition, another challenge we identify is that
   Inthissection, wepresenttheimplementation oftheNERsys-                          there are multiple extracted symptoms corresponding to similar
temandthenspecificallyintroducethesymptomretrievalcompo-                           orsamesymptom.Itcommonlyappearsinself-diagnosisbecause
nent that is responsible for retrieving and displaying alternative                 usertendstoinputcolloquialdescriptions,andthesemanticscould
symptoms for users, termed as EHR Graph-based Retrieval Sys-                       be very diverse. For example, a user may input “have got a run”
tem (GraphRet). After that, we present the overall flowchart of our                but this phrase is not aligned to any symptom node in the built
self-diagnosis system.                                                             HIN (the closest phrase in medical terms should be “diarrhea”).
                                                                                   In order to proceed user dialog as accurate as possible, we build
4.1   DialogProcessing                                                             a standard symptom set that contains many common equivalent
Differentfrommanygeneralnamedentityrecognition(NER)tasks                           phrases. This BiLSTM-CRF based NER system hence could extract
onpublicdatasets,wehaveChinesetextswhichhavemanyspecific                           asrichaspossibleinformationfromtherawdialog,whichsupports
medicalconceptswhichrarelyappearincommondialogue.Whatis                            more accurate diagnosis.
worse,therehasbeenbyfarnoopendatasetsaboutChineseclinical                          4.2   SymptomRetrieval
notes.Theseproblemscausedialogprocessinginweb-basedself-
diagnosisdifficulttoberealizedbasedonthecurrenttechniques.                         Symptomretrieval isthe bedrock ofthe diagnosissystemfor pro-
Accountingforthischallenge,Wetrytocollectandlabelmorethan                          vidingalternativesymptomsformakingfinaldiseasediagnosis.We
50 thousand medical corpus via crowdsourcing. In particular, in                    have developed HealGCN where the graph representation learning
each sentence, 14 types of entities are labeled, including disease,                is tailored towards disease diagnosis. However,the learned embed-
symptom, body part, negative words, etc. An example of labeled                     dings might be suboptimal for symptom retrieval. In this scenario,
sentenceisshowninFig.5,wheremodifieraresplitfromsymptom                            wewould liketoperform furtherrepresentationlearning forhigh
as the independent entities and negative words are specified as a                  qualitysymptomretrieval,byconsideringco-occurrenceofsymp-
indicator to no such symptom.                                                      toms in the bipartite disease-symptom (DS) graph. In detail, our

WWW ’21, April 19–23, 2021, Ljubljana, Slovenia                                                       Zifeng Wang, Rui Wen, Xi Chen, Shilei Cao, Shao-Lun Huang, Buyue Qian, and Yefeng Zheng
task is to generate embeddings of symptoms that can be utilized                    andgeneratealldiseaseembeddingsthroughDSD.Thesystemthen
fornearest-neighborlookupfortoprelatedsymptoms.Tothisend,                          checks if the predicted score of(u,d)calculated by Eq. (1) exceeds
we apply GraphRet on the DS graph with its symptom and disease                     the confidencethreshold. Predicted probability of eachdisease can
embeddingsinitializedbythepretrainedHealGCN.Specifically,we                        be obtained by a softmax function. If not, the system turns to re-
use a max-margin based loss [42]                                                   questGraphRetformoresymptoms.Itusuallyhappenswhenusers
         LMM =  E s−∼P n(s)max{0,q∗⊤s  q∗s +−q∗⊤s  q∗s−+ ∆},         (8)           only offer a limited number ofsymptoms, or their descriptions are
                                                                                   ambiguous.After severalroundsofQ&A, thesystemexpandsthe
where(s,s+)isapairofrelatedsymptoms;s−isanunrelatedsymp-                           symptomset andbecomesmore confidenttomake thefinaldiag-
tom to s sampled from a negative distribution Pn(s); and ∆  is a                   nosis. Our online inference system empowered by the symptom
marginhyperparameter.Wekeep∆  =  1.0inourexperiments.The                           retrieval works as Algorithm 1 shows.
graph convolutional operation works on the DS graph similar to
§3 but follows the meta-path symptom-disease-symptom (SDS),                        5   EXPERIMENTS
to obtain the final symptom embeddingq∗s. Minimizing the max-                      In this section, we evaluate HealGCN on the real-world EHR data
margin loss encourages the related items to be close, while the                    collectedfromhospitalsoffline.Then,wecompareourmethodwith
unrelateditems tobedistant intheembeddingspace.Duringtrain-                        several baselines. Specifically, we aim at answering five research
ing, for a sampled symptom s, we find the positive symptoms+                       questions below:
that co-occurs most frequently with s, while the rest symptoms
that co-occur less frequently thans+  are all alternative negative                 •RQ1:DoesHealGCNleadtoimprovementinaccuracyofdisease
ones.Additionally,weleveragecurriculumlearning[1]toenhance                            diagnosis?
granularityoflearnedembeddings.Intheearlytrainingphase,neg-                        •RQ2:Does highorder neighborhood of HealGCN contributeto
ativesymptomsareuniformlysampled,whilewegraduallyinvolve                              better performance?
more hard negativesymptoms, i.e.,s−co-occurring withs slightly                     •RQ3:Howmuch dometa-paths USUandDSD benefitthe infer-
less thans+ , in the subsequent epochs.                                               ence?
   Supposeweextractasymptoms fromtheaNamedEntityRecog-                             •RQ4:Does GraphRet lead to good retrieval results?
nition (NER) system from the user’s description, which is called                   •RQ5:HowdoesHealGCN+GraphRetperforminonlineA/Btest?
a seed symptom. And the objective of symptom retrieval system
is to search for the related symptoms. In this work, we propose a                  5.1   Datasets
hierarchicallookupframeworkbasedonthegeneratedsymptom                              Wecollectmanyelectronicclinicalnotesfromseveralhospitalsand
embedding q∗s  through the DS graph. For example, we can take                      build the EHR data used in the offline experiments. In each case,
relatedsymptomsfromthetwo-hopneighborsoftheseedsymptom                             there are department name, chief complaint, clinical note written
s0 through theSDS meta-path, i.e.,theN2SDS(s0), wherewe define                     by clinicians, and diagnosis resultsby clinicians. Examples of this
disease d∈N1SDS(s0)as the intermediate node. This formulation                      dataset are shown in Table 1. Raw notes are written in Chinese,
takestheco-occurrencebetweensymptomanddiseaseintoaccount,                          we here translate them into English for reading. In detail, from
and allows us to reallocate attention over symptoms conditioned                    eachnote,weextractthebasicpersonalcharacteristicsofpatients,
on the intermediate diseases. Technically, considering that there                  likegender andage. Thesymptomnodes constituteofentities ex-
are m diseases linked to the seed symptom s0, we can compute                       tracted by NER system mentioned in§4.1. In this dataset, most
thenormalizedpoint-wisemutualinformation(nPMI)[3]between                           patients have only one visit, whichexaggerates the diagnosis diffi-
symptoms and the connected diseases by                                             culty substantially. Besides, each note is associated with a disease
                                                  p(s0)p(di))                      diagnosis made by physicians, with the corresponding symptom
            hi :=  npmi(s0;di)=   log(p(s0,di)/−logp(s0,di)     ,            (9)   description by patients. We view the diagnosis made by physicians
where the mutual informationhi denotes the importance of dis-                      asthegroundtruthforlearning.ThestatisticsoftheEHRgraphare
easedi to s0. After that, we transform the score into probability                  shown in Table 2 and Table 3.
distribution                                                                       5.2   ExperimentalProtocol
                 pi =  softmax(hi)=     eh iÍ                                      5.2.1    Evaluation Metric. We follow the standard evaluation ap-
                                           j eh j∈[0,1].                  (10)     proachinrecommendation.Duringthetestphase,themodelneeds
Forexample,ifthetargetnumberofretrievedsymptomsisk,we                              topredictthe probabilityforauserto beaffectedbyeachdisease,
will pickk×pi related symptoms fromN(di), the neighborhood                         thenonlythosediseasesthatuserreallyhasaretakenaspositiveex-
ofdi. After that, the related symptoms can be ranked with respect                  amples.Weadoptwidely-acceptedmetricRecall@k,nDCG@kand
totheir cosine similaritytotheseedsymptomwithhigh efficiency.                      P@1, in terms of the top-k ranking results based on the predicted
                                                                                   scores, to evaluate the model’s performance.
4.3   SystemOverview                                                               5.2.2    Baseline. Asaforementioned,thepersonalcharacteristicsas
Fig.2illustratestheflowchartofthediseasediagnosissystem.When                       wellashistoricalvisitsareinaccessibleinonlineself-diagnosistask,
a user inputs the initial descriptions, the system turns to request                hence previous transductive learning disease diagnosis methods
GraphRet for relevant symptoms. After that, the seed symptoms                      areNOT applicable, especially those RNN-likemodelsthat depend
accompaniedwith theclicked symptomsareadopted forinference                         on sequential EHR data, e.g., Doctor AI [5], RETAIN [7], Dipole
byHealGCN,wherewegeneratetheuserembeddingthroughUSU                                [30], etc.Therefore, we only pickbaselines which areinductive or

Online Disease Diagnosis with Inductive Heterogeneous Graph Convolutional Networks                                                               WWW ’21, April 19–23, 2021, Ljubljana, Slovenia
Table1:ExamplesoftheclinicalnotesincollectedEHRdata.RawnotesareinChinese,wetranslatethemintoEnglishhere.
  Table2:StatisticsofnodesandedgesinourEHRgraph.                                          NeuMF[17]:Itisaneuralcollaborativefiltering model, which
       Node           Counts       Edge                            Counts             uses hidden layers above the user and item embeddings, in order
       Disease              146    User-Disease               136,478                 to mine the nonlinear feature interactions. This method, as well
       User            135,356     Disease-Symptom       229,373                      as the MF above, serves as representative traditional collaborative
       Symptom    146,871          User-Symptom         1,213,475                     filtering approaches for comparison.
                                                                                          GBDT[24]:Thegradientboostingdecisiontree(GBDT)model
Table 3: Statistics of the most frequent symptoms and dis-                            is popularin many industrialapplications. We useWord2Vec[32]
easesintheEHRdata.                                                                    tolearnthewordembeddingsofsymptoms,thentakeaverage of
                                                                                      eachuser’ssymptomembeddingsastheinputoftheGBDTmodel.
   Disease                   Counts        Symptom               Counts                   TextCNN[25]:Weformulatediseasediagnosisasatextclassifi-
   Hypertension           12,155           Fever                       20,321         cationtaskbasedonTextCNN.Specifically,wemodelthesymptom
   URTIa                        11,258     Stomachache           11,806               descriptionsofauseras“word”embeddings,followedbymultiple
   Pregnancyb                 9,470        Vomiting                   9,173           convolutionallayersanddenselayerstargetedtotextclassification.
   Influenza                    9,083      Anhelation                8,214            We includeitto compareourmethod withtheNLP-based disease
   Gastritis                     6,674     Runny nose               7,031             diagnosis approaches.
   Diabetes                     4,317      Headache                  6,937                Med2Vec [6]: It is a medical embedding method that learns
   Rhinitis                      2,882     Coughing                  6,420            embeddingsofmedicalcodesandvisitsbasedonaskip-grammodel
a  Upper Respiratory Tract Infections;                                                similar to Word2Vec [32]. Different from the original Med2Vec,
b  Pregnancy is not a disease while it does appear in clinical                        we here train it on the clinical notes to get the code (symptom)
                                                                                      representation.Withthepretrainedsymptomembeddings,weadd
diagnosis as a health event, we list it here as a “disease” for                       a multi-layer perceptron to get the final prediction on disease.
convenience;                                                                              GraphSAGE[15]:ItisaninductiveGCNbutignoresthehetero-
                                                                                      geneityoftheEHRgraph.Weperformitforcomparingourmethod
easy to be adapted to the inductive learning manner. Considering                      with homogeneous GCN in terms of disease diagnosis.
this, we compare our HealGCN with the following baselines:                                HealGCN-local:ThisisareducedversionofourHealGCNthat
   MF [36] : This is a classical collaborative filtering method that                  only considers the one-hop neighbors in embedding propagation
triestoprojectthediscreteindexofuseranditemintothesamereal-                           with the EHR graph. We aim to find out how the high order neigh-
valued vector space, then measures the similarity of user and item                    borscontributetothefinalresultsbycomparingitwithHealGCN.
embeddingsforpredictingscores.FortheEHRdata,weonlymodel                                   HealGCN-USU & DSD: Similar to HealGCN-local, these two
the   symptom   and   disease   embeddings,   and   the   user’s                      modelsarereducedfromHealGCNbyonlyinvolvingtheUSUor
embedding is aggregated by symptoms. The BPR loss is adopted                          DSDmeta-path.Theyareusedtomeasuringthecontributionfrom
for its optimization.                                                                 different meta-paths.

WWW ’21, April 19–23, 2021, Ljubljana, Slovenia                                                       Zifeng Wang, Rui Wen, Xi Chen, Shilei Cao, Shao-Lun Huang, Buyue Qian, and Yefeng Zheng
Table4:DiseasediagnosisaccuracyofMF[36],NeuMF[17],GBDT[24],TextCNN[25],Med2Vec[6],GraphSAGE[15]andour
HealGCN,inanofflineexperiment,wherethebestonesareinbold.
                                Precision@1       Recall@3       nDCG@3       Recall@5       nDCG@5       Recall@10       nDCG@10
            MF                                0.4459            0.6816             0.5836            0.7733             0.6214              0.8630               0.6506
            NeuMF                         0.5089            0.7213             0.6334            0.8019             0.6668              0.8773               0.6913
            GBDT                           0.4990            0.7054             0.6204            0.7657             0.6454              0.8368               0.6683
            TextCNN                      0.5291            0.7333             0.6491            0.8026             0.6778              0.8734               0.7009
            Med2Vec                      0.5218            0.7346             0.6464            0.8103             0.6777              0.8779               0.6999
            GraphSAGE                 0.5228            0.7393             0.6504            0.8133             0.6809              0.8872               0.7051
            HealGCN                     0.5507          0.7620           0.6750           0.8339           0.7046            0.9002            0.7263
              Table5:Resultsoftheablationstudy.                                      thatthelocalmodelonlyinvolvestheone-hopneighborsinmessage
                                                                                     passingandaggregation,itdoesnotexploithighorderneighborsas
                           Precision@1    Recall@3    nDCG@3                         well as the node’s position in the graph. Therefore, the local model
      HealGCN-local              0.5040         0.7263         0.6399                performs the worst among all.
      HealGCN-DSD              0.5225         0.7479         0.6552                     Furthermore,bothDSDandUSUmodelsperformworsethanfull
      HealGCN-USU               0.5495         0.7595         0.6725                 HealGCNingeneral, andbetterthan thelocalmodel.In particular,
      HealGCN                      0.5507        0.7620         0.6750               weidentifythattheUSUmodelismuchbetterthantheDSDmodel,
                                                                                     whichindicatestheimportanceoftakingsymptomco-occurrence
5.2.3    Hyperparameters. WeimplementallmodelsonPyTorch[34].                         throughUSUintoaccounttoencodeusersintobetterembeddings.
The size of symptom and disease embeddings is fixed at 64. The                       In summary, by combining both DSD and USU meta-paths, our
Adam optimizer [26] is used for optimization of all methods. We                      HealGCN achieves the best result.
applyagridsearchforoptimalhyperparameters:learningratein                             5.5   RQ4:RetrievalSystem
{0.05,0.01,0.005,0.001}, batch size in{128,256,512,1024,2048},
andweightdecayin{10−5,10−4,10−3}.IntermsoftheHealGCN                                 WecompareourGraphRetwithMed2VecandPMIinasimulated
model, the maximum numbers of neighbors in the USU and DSD                           online test. In Med2Vec and PMI, we directly compute cosine simi-
meta-pathsare{5,5,5}and{20,5},respectively.Wesplitthefull                            larity and normalized PMI (npmi) [3] between all symptoms and
datainto thetraining, validationandtest setsby 7 : 1 : 2.During                      theinitial one,respectively. Weevaluate theretrieval accuracyby
data processing, we only involve the symptoms appearing in the                       countinghow manyretrievedsymptoms are alignedwith thetrue
training set for preventing data leakage. During training, we evalu-                 symptoms of users, as shwon in Table 7. It can be identified that
ate the model’s nDCG@5 on the validation set for early stopping.                     GraphRetsignificantlyoutperformsthebaselines.Welistseveral
                                                                                     casesinTable6.Itcanbeobservedthatuserspickmoreretrieved
5.3   RQ1:OverallComparison                                                          symptomsfromGraphRet,thustheexpandedsymptomsetleadsto
Inthissection,wecomparetheoverallperformanceofourHealGCN                             accuratepredictionbyHealGCN.Besides,GraphRetcanyieldmore
and theselected baselines, asshown in Table 4.In this experiment,                    diversesymptomsthanPMI,whichbenefitsinreducingdiagnosis
allsymptoms extractedfromtheclinical notesareused fordisease                         mistakes.
diagnosis,withoutretrievedsymptomsfromGraphRet.Itcanbeob-                               Moreover,weevaluatehowmuchtheretrievedsymptomsbenefit
servedthatourHealGCNconsistentlyoutperformsotherbaselines,                           indiseasediagnosisaccuracy,resultsareillustratedinTable8.Note
withimprovementaround5%overthebestbaselineswithrespectto                             thatdifferent fromtheexperiment donein§5.3,here weuseonly
all metrics. MFonly takesdirect multiplicationof userand disease                     onesymptomastheseedsymptom,andtheDirectmethodperforms
embeddings,whileNeuMFperformsbetterbyleveraginghidden                                inference only on the seed symptom. In GraphRet, inference is
layersandexploitinghighorderinteractionsbetweenembeddings.                           performed involving all retrieved symptoms. It can be identified
This demonstrates the effectiveness of non-linear interactions be-                   thatretrievalsystemssignificantlybenefittheinferenceaccuracy,
tweenuser anddiseaseembeddings. TextCNN performsrelatively                           which verifies the effectiveness of retrieval system for improving
well,butdoesnotinvolveconnectivitybetweenuseranddisease                              diagnosis accuracy.
considering the graphical structure. Med2Vec takes semantics of
symptomsintoconsiderationbyutilizingaWord2Vec-likemodel;                             5.6   RQ5:OnlineA/BTest
however, it ignores rich interactions in the EHR graph and does                      We evaluate the proposed HealGCN+GraphRet framework on a
notperformverywell.GraphSAGEleveragesgraphstructure,but                              real-worldonlineself-diagnosis serviceplatform,whichcurrently
ignorestheinteractiontypesandperformsworsethanourmethod.                             servesaround60thousandrequestsperday,inanonlineA/Btest
5.4   RQ2&3:AblationStudy                                                            for one month. On this platform, users are made diagnosis and
                                                                                     then guided to make an appointment with a right physician in a
Weperformablationexperimentstoexploreinfluencesofdifferent                           right department,which covers over500 hospitals rightnow.The
componentsofourHealGCN,withresultsshowninTable5.Recall                               basemodelthathasalreadydevelopedfordemoonthisplatform

Online Disease Diagnosis with Inductive Heterogeneous Graph Convolutional Networks                                                               WWW ’21, April 19–23, 2021, Ljubljana, Slovenia
       Table6:Examplesofhowretrievedsymptomsleadtofinaldiagnosis.Thesymptomspickedbyusersareinbold.
        PMI  [3]
        Seed Symptom        Retrieved Symptoms                                            Top-1 Prediction                    Groundtruth
        Belching            Stomachache; Nausea; Regular bowel movement;                  Gastrointestinal dysfunction        Gastroenteritis
                            Weight loss; Vomiting
        Chest pain          Muscleache;     Chesttightness; Cold; Tinnitus; Hy-           Coronary artery disease             Pulmonary bronchitis
                            pertension;
        Weight loss         Constipation;Abdominalbloating;Highbloodpres-                 Irritable bowel syndrome            Hyperthyroidism
                            sure; Abdomen ache; Navel ache
        Irritability        Insomnia;        Tension;       Mental       exhaustion;      Depressive disorder                 Schizophrenia
                            Depressedmood; Shivering
        GraphRet
        Seed Symptom        Retrieved Symptoms                                            Top-1 Prediction                    Groundtruth
        Belching            Diarrhea;   Darkstool;    Bloodystool; Anorexia; He-          Gastroenteritis                     Gastroenteritis
                            matemesis
        Chest pain          Chesttightness;     Coughing;Palpitation;      Dyspnoea;      Pulmonary bronchitis                Pulmonary bronchitis
                            Dizziness
        Weight loss         Fatigue;     Drymouth;   Palpitation;        Impatience;      Hyperthyroidism                     Hyperthyroidism
                            Strongappetite
        Irritability        Depressedmood;    Alcoholism;            Phobia;    Hand      Schizophrenia                       Schizophrenia
                            tremor;   Fidget
Table  7:  Rec@20,  Rec@50  and  Rec@100  achieved  by                              Table 9: Online evaluation results of the proposed method
GraphRetandcomparedbaselinesonsymptomretrieval.                                     with the existing base model (TextCNN+PMI). Here, UP,
                                                                                    UD and UO denote user click ratio regarding the recom-
                           Recall@20    Recall@50    Recall@100                     mended physician, disease and appointment order, respec-
                                                                                    tively.ACCindicatesaccuracyofthosewhosendbackfeed-
     PMI [3]                         0.0786           0.1063             0.1298     backaftercometoseeadoctor.
     Med2Vec [6]                0.1253           0.1799             0.2332
     GraphRet (ours)            0.2798         0.3953           0.4855                                                UP        UD        UO      ACC
Table 8: P@1, Rec@5 and nDCG@5 achieved by HealGCN                                         Base Model                   74.4%    20.6%    4.8%    35.5%
supportedbyGraphRet,comparedwiththeDirectmethod.                                           HealGCN+GraphRet    86.6%    26.8%    6.9%    42.3%
                       Precision@1    Recall@5    nDCG@5
         Direct                    0.1623         0.3794         0.2729             observedfromTable9thattheproposedmethodoutperformsthe
         GraphRet              0.2719        0.5230        0.4053                   base model significantly, which shows our model has advantage in
                                                                                    diseasediagnosis,thusattractinguserstomakefurtherexploration.
constitute of TextCNN for disease diagnosis and PMI for symptom                     And our new system indeed obtains better accuracy in practice.
retrieval.
   In the A/B test, we split requests into two buckets: one for the                 6   CONCLUSION
base model and other for the proposed method. Several indica-                       In this paper, we proposed HealGCN for disease diagnosis and
tors are adopted: user-physician (UP) click ratio, user-disease (UD)                GraphRetforsymptomretrievalaimingathandlingtwomainchal-
click ratio and user-order(UO) click ratio, because our systemdis-                  lengesinonlineself-diagnosis:cold-startusersandambiguousde-
playsnotonlydiseasediagnosis,butalsodepartmentandphysician                          scriptions.HealGCNadoptsaninductivelearningparadigm,thus
recommendations. UP and UD means how many users click the                           applicable for users without historical hospital visits. GraphRet
recommended physiciansand diseases for detailedexploration, re-                     serves for symptom retrieval in Q&A in self-diagnosis by apply-
spectively. UO means how many users make appointment to the                         inggraphconvolutionoperationstoabipartitedisease-symptom
recommended departments for further consultation. These indica-                     graphtogeneratesymptomembeddingsforretrieval.Theoffline
torsare usefulproxyformeasuring diagnosisaccuracy.Moreover,                         evaluation and online test verified the superiority of HealGCN in
we send questionaire to users for feedback about their diagnosis                    diagnosisaccuracy.ItalsoprovedthatGraphRetbenefitsinprovid-
results after they come to see a doctor, then we obtain an accuracy                 ingarichsupportivesymptomsetforuserstoselect,resultingin
ofthe diagnosisresults,namelyACC,from thefeedback.Itcan be                          more accurate diagnosis.

WWW ’21, April 19–23, 2021, Ljubljana, Slovenia                                                       Zifeng Wang, Rui Wen, Xi Chen, Shilei Cao, Shao-Lun Huang, Buyue Qian, and Yefeng Zheng
ACKNOWLEDGMENTS                                                                                                   In Proceedings of the 27th ACM International Conference on Information and
The research of Shao-Lun Huang was supported in part by the                                                       Knowledge Management. 763–772.
                                                                                                           [21]  Anahita Hosseini,TylerDavis, and Majid Sarrafzadeh.2019. Hierarchical target-
NaturalScienceFoundationofChinaunderGrant61807021,inpart                                                          attentive diagnosis prediction in heterogeneous information networks. In Inter-
by the Shenzhen Science and Technology Program under Grant                                                        national Conference on Data Mining Workshops. 949–957.
                                                                                                           [22]  Zhiheng Huang, Wei Xu, and Kai Yu. 2015. Bidirectional LSTM-CRF models for
KQTD20170810150821146, and in part by the Innovation and En-                                                      sequence tagging. arXiv preprint arXiv:1508.01991 (2015).
trepreneurshipProjectforOverseasHigh-LevelTalentsofShenzhen                                                [23]  Alistair EW. Johnson, Tom J. Pollard, Lu Shen, H. Lehman Li-wei, Mengling
under Grant KQJSCX20180327144037831.                                                                              Feng, Mohammad Ghassemi, Benjamin Moody, Peter Szolovits, Leo Anthony
                                                                                                                  Celi,andRogerG.Mark.2016. MIMIC-III,afreelyaccessiblecriticalcaredatabase.
                                                                                                                  Scientific Data3 (2016).
REFERENCES                                                                                                 [24]  Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma,
                                                                                                                  QiweiYe,andTie-YanLiu.2017. Lightgbm:Ahighlyefficientgradientboosting
 [1]  Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. 2009.                                   decision tree.In Advances in Neural Information Processing Systems.3146–3154.
      Curriculum learning.In Proceedings of the 26th Annual International Conference                       [25]  YoonKim.2014. Convolutionalneuralnetworksforsentenceclassification. arXiv
      on Machine Learning. 41–48.                                                                                 preprint arXiv:1408.5882 (2014).
 [2]  Siddharth Biswal, Cao Xiao, Lucas M. Glass, Elizabeth Milkovits, and Jimeng                          [26]  Diederik P. Kingma and Jimmy Ba. 2014. Adam: A method for stochastic opti-
      Sun.2019. Doctor2Vec:Dynamicdoctorrepresentationlearningforclinicaltrial                                    mization. arXiv preprint arXiv:1412.6980 (2014).
      recruitment. arXiv preprint arXiv:1911.10395 (2019).                                                 [27]  Thomas N. Kipf and Max Welling. 2016.  Semi-supervised classification with
 [3]  Gerlof Bouma. 2009. Normalized (pointwise) mutual information in collocation                                graph convolutional networks. arXiv preprint arXiv:1609.02907 (2016).
      extraction.  Proceedings of German Society for Computational Linguistics and                         [28]  Yikuan Li, Shishir Rao, Jose Roberto Ayala Solares, Abdelaali Hassaine, Dex-
      Language Technology, 31–40.                                                                                 ter Canoy, Yajie Zhu, Kazem Rahimi, and Gholamreza Salimi-Khorshidi. 2019.
 [4]  Zhengping Che and Yan Liu. 2017. Deep learning solutions to computational                                   BEHRT:Transformerforelectronichealthrecords.arXivpreprintarXiv:1907.09538
      phenotyping in health care. In IEEE International Conference on Data Mining                                 (2019).
      Workshops. 1100–1109.                                                                                [29]  Rong-Ho Lin. 2009. An intelligent model for liver disease diagnosis. Artificial
 [5]  Edward Choi, Mohammad Taha Bahadori, Andy Schuetz, Walter F Stewart, and                                    Intelligence in Medicine 47 (2009), 53–62.
      Jimeng Sun. 2016.  Doctor AI: Predicting clinical events via recurrent neural                        [30]Fenglong Ma,RadhaChitta, JingZhou,QuanzengYou, TongSun,and JingGao.
      networks. In Machine Learning for Healthcare Conference. 301–318.                                           2017. Dipole:Diagnosispredictioninhealthcareviaattention-basedbidirectional
 [6]  EdwardChoi,MohammadTahaBahadori,ElizabethSearles,CatherineCoffey,                                           recurrentneuralnetworks.In Proceedings of the 23rd ACM SIGKDD international
      MichaelThompson,JamesBost,JavierTejedor-Sojo,andJimengSun.2016. Multi-                                      conference on knowledge discovery and data mining. 1903–1911.
      layer representation learning for medical concepts. In Proceedings of the 22nd                       [31]  Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013.   Efficient
      ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.                                 estimationofwordrepresentationsinvectorspace. arXivpreprintarXiv:1301.3781
      1495–1504.                                                                                                  (2013).
 [7]  Edward Choi, Mohammad Taha Bahadori, Jimeng Sun, Joshua Kulas, Andy                                  [32]  Tomas Mikolov, IlyaSutskever,Kai Chen,Greg S.Corrado, andJeff Dean.2013.
      Schuetz,andWalterStewart.2016. RETAIN:Aninterpretablepredictivemodel                                        Distributed representations of words and phrases and their compositionality. In
      for healthcare using reverse time attention mechanism. In Advances in Neural                                Advances in Neural Information Processing Systems. 3111–3119.
      Information Processing Systems. 3504–3512.                                                           [33]  JamesMullenbach,SarahWiegreffe,JonDuke,JimengSun,andJacobEisenstein.
 [8]  EdwardChoi,CaoXiao,WalterStewart,andJimengSun.2018. MiME:Multilevel                                         2018. Explainable prediction of medical codes from clinical text. arXiv preprint
      medical embedding of electronic health records for predictive healthcare. In                                arXiv:1802.05695 (2018).
      Advances in Neural Information Processing Systems. 4547–4557.                                        [34]  AdamPaszke,SamGross,FranciscoMassa,AdamLerer,JamesBradbury,Gregory
 [9]  EdwardChoi,ZhenXu,YujiaLi,MichaelW.Dusenberry,GerardoFlores,Yuan                                            Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al.
      Xue,andAndrew M.Dai.2019. Graphconvolutionaltransformer: Learningthe                                        2019. PyTorch: An imperative style, high-performance deep learning library. In
      graphical structure of electronic health records. arXiv preprint arXiv:1906.04716                           Advances in Neural Information Processing Systems. 8024–8035.
      (2019).                                                                                              [35]  SanjayPurushotham,ChuizhengMeng,ZhengpingChe,andYanLiu.2017.Bench-
[10]  Youngduck Choi, Chill Yi-I Chiu, and David Sontag. 2016.    Learning low-                                   markofdeeplearningmodelsonlargehealthcareMIMICdatasets. arXivpreprint
      dimensionalrepresentationsofmedicalconcepts. AMIASummitsonTranslational                                     arXiv:1710.08531 (2017).
      Science Proceedings (2016), 41–50.                                                                   [36]  SteffenRendle,ChristophFreudenthaler,ZenoGantner,andLarsSchmidt-Thieme.
[11]  JacobDevlin,Ming-WeiChang,KentonLee,andKristinaToutanova.2019. BERT:                                        2012. BPR: Bayesian personalized rankingfromimplicit feedback. arXiv preprint
      Pre-training ofdeepbidirectionaltransformersfor languageunderstanding. In                                   arXiv:1205.2618 (2012).
      Proceedings of the Conference of the Association for Computational Linguistics.                      [37]  Jyoti Soni, Ujma Ansari, Dipesh Sharma, and Sunita Soni. 2011.   Predictive
      4171–4186.                                                                                                  data mining for medical diagnosis: An overview of heart disease prediction.
[12]  Dmitriy Dligach and Timothy Miller. 2018.  Learning patient representations                                 International Journal of Computer Applications 17 (2011), 43–48.
      from text. arXiv preprint arXiv:1805.02096 (2018).                                                   [38]  Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, and Tat-Seng Chua. 2019.
[13]  Wael Farhan, Zhimu Wang, Yingxiang Huang, Shuang Wang, Fei Wang, and                                        Neuralgraphcollaborativefiltering.InProceedingsofthe42ndInternationalACM
      XiaoqianJiang.2016. Apredictivemodelformedicaleventsbasedoncontextual                                       SIGIRConferenceonResearchandDevelopmentinInformationRetrieval.165–174.
      embedding of temporal sequences. JMIR Medical Informatics 4, 4 (2016), e39.                          [39]  Zifeng Wang, Yifan Yang, Rui Wen, Xi Chen, Shao-Lun Huang, and Yefeng
[14]  Ferenc Galkó and Carsten Eickhoff. 2018. Biomedical question answering via                                  Zheng. 2021. Lifelong Learning based Disease Diagnosis on Clinical Notes. In
      weightedneuralnetworkpassageretrieval.InEuropeanConferenceonInformation                                     Proceedings of the 25th Pacific-Asia Conference on Knowledge Discovery and Data
      Retrieval. 523–528.                                                                                         Mining (PAKDD).
[15]  Will Hamilton, Zhitao Ying,and Jure Leskovec. 2017. Inductive representation                         [40]  XingWeiandCarstenEickhoff.2018. Embeddingelectronichealthrecordsfor
      learningon largegraphs.In Advances in Neural Information Processing Systems.                                clinical information retrieval. arXiv preprint arXiv:1811.05402 (2018).
      1024–1034.                                                                                           [41]  Cheng-HsiungWeng,TonyCheng-KuiHuang,andRuo-PingHan.2016. Disease
[16]  Jiawei Han, Yizhou Sun, Xifeng Yan, and Philip S. Yu. 2010. Mining knowledge                                prediction with different types of neural network classifiers.  Telematics and
      fromdatabases:Aninformationnetworkanalysisapproach.In Proceedings of the                                    Informatics 33 (2016), 277–292.
      ACM SIGMOD International Conference on Management of Data. 1251–1252.                                [42]  Rex Ying, Ruining He, Kaifeng Chen, Pong Eksombatchai, William L. Hamilton,
[17]  Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng                                    and Jure Leskovec. 2018. Graph convolutional neural networks for web-scale
      Chua.2017. Neuralcollaborativefiltering.InProceedings ofthe26thInternational                                recommender systems. In Proceedings of the 24th ACM SIGKDD International
      Conference on World Wide Web. 173–182.                                                                      Conference on Knowledge Discovery and Data Mining. 974–983.
[18]  Emily Herrett, Arlene M. Gallagher, Krishnan Bhaskaran, Harriet Forbes, Rohini                       [43]  JingheZhang,KamranKowsari,JamesH.Harrison,JenniferM.Lobo,andLauraE.
      Mathur, Tjeerd van Staa, and Liam Smeeth. 2015. Data resource profile: Clinical                             Barnes. 2018. Patient2Vec: Apersonalized interpretabledeep representationof
      practice research datalink (CPRD).  International Journal of Epidemiology 44                                the longitudinal electronic health record. IEEE Access 6 (2018), 65333–65346.
      (2015), 827–836.                                                                                     [44]  Xianli Zhang, Buyue Qian, Yang Li, Yang Liu, Xi Chen, Chong Guan, Yefeng
[19]  Bhagya Hettige, Weiqing Wang, Yuan-Fang Li, Suong Le, and Wray Buntine.                                     Zheng,and ChenLi. 2021. LearningRobust PatientRepresentationsfrom Multi-
      2020. MedGraph:StructuralandTemporalRepresentationLearningofElectronic                                      modal Electronic Health Records: A Supervised Deep Learning Approach. In
      Medical Records. In Proceedings of the 24th European Conference on Artificial                               Proceedings of the 2021 SIAM International Conference on Data Mining (SDM).
      Intelligence (ECAI).                                                                                 [45]  ZihaoZhu, ChangchangYin, BuyueQian, Yu Cheng,JishangWei,and FeiWang.
[20]  Anahita Hosseini,Ting Chen,WenjunWu, Yizhou Sun,and MajidSarrafzadeh.                                       2016. Measuringpatientsimilaritiesviaadeeparchitecturewithmedicalconcept
      2018. HeteroMed: Heterogeneous information network for medical diagnosis.                                   embedding. In IEEE 16th International Conference on Data Mining. 749–758.

