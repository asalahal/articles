          ModelingLatentComorbidityforHealthRiskPredictionUsingGraph
                                                      ConvolutionalNetwork
                          RuiWang†           Ming-ChingChang†           MarleenRadigan‡
                                     †University at Albany, State University of New York, NY, USA
                                            ‡New York State Ofﬁce of Mental Health, NY, USA
                      rwang3@albany.edu      mchang2@albany.edu      marleen.radigan@omh.ny.gov
                              Abstract
  We propose to apply deep Graph Convolutional Network
  (GCN) for the analysis and prediction of patient health co-
  morbidity from sparse health records. Patient health data
  are represented in a powerful graph structure. Speciﬁcally,
  healthcare conditions including health diagnosis categories,
  hospitalizations, injury incidents are represented as a type
  of graph nodes, and patient attributes including demograph-
  ics, aid categories are represented as another type of nodes.
  Health records for individuals including diagnostic results,                 Figure 1:  We formulate health risk prediction as a bipartite
  hospital visits are represented as graph links connecting the                graphmatrixcompletionproblem.Theinputtoourapproach
  two node types, such that the whole record forms as a sparse                 isprovidedasaratingmatrixconsistingofthehealthcondi-
  bipartite graph. Our hypothesis is that patient health trend,                tionsofindividualpatientrecords.Eachgraphlinkspanning
  diseaseprognosis,treatment,andtheirlatentcorrelationscan                     a patient and a health condition represents a corresponding
  allbemodeledbyrecoveringthemissinglinksinthisbipartite                       health record with a severity condition level.
  graph (the link prediction problem). Starting with sparse pa-
  tientdataorincompleterecords,graphcompletionandrecord
  fusion via end-to-end GCN modeling can lead to robust pre-
  diction across individual patients and health records. Appli-                morbiditymodelingandhealthriskprediction.First,patient
  cation in estimating health prognosis shows the efﬁcacy of                   attributesincludesdemographics,insuranceconvergelevels,
  the proposed method compared to existing approaches.                         andaidcategoriescanbeorganizedasatypeofgraphnodes
                                                                               (one node per patient). Secondly, the healthcare conditions
  Keywords:healthriskprediction,medicalrecords,health                          including physical/mental health, diagnostic results, hospi-
data, comorbidity, bipartite graph, link prediction, recom-                    talization reasons, injury incidences of each patient can be
mendation system, Graph Convolution Network, GCN.                              represented as another type of graph nodes. This way, each
                                                                               healthrecordorinsuranceeventscanberepresentedasgraph
                       1   Introduction                                        linksconnectingthetwotypesofnodes,suchthatthewhole
In medicine, comorbidity—the presence of one or more                           healthrecordcanbeorganizedasabipartitegraphasshown
mental or physical disorders co-occurring for a patient—                       in Fig.1(a). Graph data can be stored as a sparse adjacency
constitutes a challenge for healthcare professionals and the                   matrix, where edge weight indicates the condition severity
healthcare system (Valderas, Starﬁeld, and Sibbald 2009).                      level as shown in Fig.1(b). Using this representation, co-
Many studies (Piane and Smith 2014;Bhattacharya and                            morbidity relations can be modelled and latent conditions
Shen 2014) suggest that mental disorders correlate with                        can be recovered by predicting the missing links from the
chronic conditions, and mental/medical disorders usually                       sparse bipartite graph.
share  common  risk  factors  (Antonaci,  Nappi,  and  Galli                      Link prediction (Liben-Nowell and Kleinberg 2007) is
2011). From the perspective of public health, understanding                    a fundamental problem in network science. Conventional
the risk factors and latent relations among disorders and co-                  methodsrely onparametric statistics,correlation coefﬁcient
morbidity can lead to early prediction of potential disorders                  amongnodes,logisticregression,cliques(ofthreenodes)as
and timely treatment.                                                          network topology to predict latent structures (see§2).
  Patient health records and person-speciﬁc attributes can                        In this paper, we apply Graph Convolutional Network
beorganizedintoagraphstructureasapowerfulrepresenta-                           (GCN) (Kipf and Welling 2016a) to solve the health link
tionforanalysis.Wefocusontwotypesofattributesforco-                            predictionproblem.GCNhasdrawngrowingattentionsdue
                                                                               to its end-to-end, model-free capability in automatic learn-
Copyright  c⃝   2020, Association for the Advancement of Artiﬁcial             ingofcomplexstatisticalinteractionsbetweenfeaturesfrom
Intelligence (www.aaai.org). All rights reserved.                              high-dimensional data. We present patient characteristics

along with their health conditions and outcome using undi-                   2011).Collaborativeﬁlteringisafamilyofalgorithmswork-
rectedbipartitegraph.Thedataisstoredasmatrixstructure,                       ing with a rating matrix to ﬁnd similar users or items and
thenapplyGraphConvolutionalNetwork(GCN)deeplearn-                            calculate rating based on ratings of similar users. The ma-
ing framework to ﬁll in the sparse matrix of data. Subse-                    trix is typically huge and sparse, with missing values. Data
quentlyweusetheobtainedrelationforindividualemerging                         driven machine learning can be used to learn a function that
or latent health condition prediction, meanwhile predict the                 predicts utility of items for each user. k-Nearest Neighbors
riskoftwohealthoutcomessimultaneously,i.e.hospitaliza-                       (kNN) based on cosine or correlation similarity cannot han-
tion and injury incidents. The aim is to predict the severity                dle sparsity well, as there may not exist enough samples in
risk of each chronic condition for each patient, as well as                  theneighborhood.Matrixfactorizationmethodsworkbyde-
the frequency category of hospitalization and other associ-                  composing the user-item interaction matrix into the product
ated adverse event i.e. injury, given their known conditions                 of two matrices (Koren and Bell 2009), with an aim of di-
forMedicaidinsurers.Fig.1showsanexampleratingmatrix                          mension reduction. The advantage of it over KNN is that
M  and a bipartite graph.                                                    even though two users have not rated any same items, it is
  Contribution of this paper is two-fold. (1) We formulate                   stillpossibletoﬁndthesimilaritybetweenthemiftheyshare
the health record modeling as a bipartite graph matrix com-                  the similar underlying latent features.
pletion problem, and apply the recent deep graph convolu-                       Matrix completion (Cand `es and Recht 2009) is the task
tion methods as an effective solution. (2) The approach is                   of ﬁlling in the missing entries of partially observed ma-
appliedonhealthriskpredictionwithsuperioraccuracyout-                        trix.Therecommendationproblemcanbeposedasamatrix
performing comparison methods.                                               completionproblem, startingwitha sparsematrixof known
  The rest of this paper is organized as follows.§2sur-                      user-item preferences. The underlying assumption is that a
veysfundamentalsofgraphlinkprediction,priorartsofma-                         low-dimensional representation of users and items exists,
trixcompletionandGraphConvolutionsNetwork(GCN)in                             which can be modeled via e.g. a low-rank matrix. Popularly
recommendation systems.§4describes our formulation of                        methods in this category includes Alternative Least Square
patienthealthandmedicalrecordsintoabipartitegraphrep-                        (ALS), spectral regularization with soft threshold, Alternat-
resentation, and how we apply GCN for matrix competition                     ing Direction Method of Multipliers (ADMM),etc.
as a solution for health risk prediction.§5discusses a real-                    The Graph Convolutional Matrix Completion (GC-
world sensitivity analysis evaluation performed on a large                   MC) (Van den Berg, Kipf, and Welling 2017) for recom-
datasetcontainingsamplesofpatientclaimdata.§6describe                        mendationsystemsperformlinkpredictionongraphusinga
performance evaluation of proposed approach and compari-                     graph-based auto-encoder framework, building upon recent
sonwithbaselinemethodsusedincollaborativeﬁlteringfor                         deep GCNs. The auto-encoder extracts latent features from
recommendation systems.§7concludes this paper and dis-                       a user preference dataset through a form of message pass-
cuss future directions.                                                      ing. These latent user-item preference represented on a bi-
                       2   Background                                        partiteinteractiongraphareusedtoderivethedesiredrating
                                                                             through a bi-linear decoder.
Healthcare data analytics. Logistic algorithm is the most
common classiﬁcation method used in healthcare study that                              3   HealthcareProblemStatement
models the relationship between binary outcome and inde-
pendent variables. The importance of each of the explana-                    Ourtaskistopredicttheriskoflatentcomorbidityusingad-
tory variables is assessed by carrying out statistical tests on              ministrative healthcare claim data. The population included
thesigniﬁcanceofcoefﬁcients.Thesemodelsrelyonprede-                          (N = 750 K )arealargesubsetofindividualswhowerecon-
ﬁned heuristics, e.g. assuming little or no multi-collinearity               tinuouslyeligible forNewYorkState Medicaidin2017 and
among  the  independent  variables.  Recently,  deep  learn-                 also had a health condition or event (determined by claim
ing techniques have been applied to clinical applications                    dataincludingphysical,behavioral,inpatientadmission,and
for outcome prediction, including Autoencoder(AE), Long                      injury) in the year. For demonstration purpose, a speciﬁc
Short-Term  Memor  (LSTM),  Restricted  Boltzmann  Ma-                       condition or event is identiﬁed by invoice type and ICD-10-
chine (RBM), and Deep Belief Network (DBN) (Shickel et                       CM diagnostic codes. The severity level (i.e. 1,2,3,4) of a
al. 2018). However, the applied convolutional operation is                   conditionoreventforeachindividualisarbitrarilyattributed
only appropriate for grid structured data. Meanwhile, they                   bytheoccurrenceofcorrespondingdiagnosticcodesonser-
are widely known to be difﬁcult to train and computation-                    viceclaimsintheyear.e.g.forapatient’sheartdiseasecon-
ally heavy.                                                                  dition,theseverityleveliscategorizedas1,2,3and4when
  Recommendation system is an application where link                         ≤2, 3−10 , 11−40 ,≥40  times of visits with such diag-
prediction algorithms can be directly applied on a graph                     nostic codes respectively.
structure, for e.g. social network analysis or movie prefer-                    The goal is to predict individual’s missing condition or
ence prediction (Van den Berg, Kipf, and Welling 2017).                      eventitems,e.g.under-reportedoremergingitems,basedon
Traditional approaches involve the calculation of a heuris-                  their existing conditions and past adverse events, adjusted
tic similarity score for a pair of nodes, such as the number                 bydemographiccharacteristicsandMedicaidenrollmentel-
ofcommonneighborsortheshortestpathlengthconnecting                           igibility category. This task can be cast as a Link Prediction
the nodes, where pairs of nodes with the highest similarity                  problem,consideringbothpersonanditemnodefeatures.A
scores are considered the most likely edges (L ¨u and Zhou                   total of 42 conditions are considered:

             Figure 2: Node aggregation for link prediction based on neighboring nodes with similar connectivities.
•10 physical health conditions: Cancer, Chronic obstruc-                      based side information, also based on neural message pass-
  tive  pulmonary  disease  (COPD),  Cerebral  infection,                     ing directly on the interaction graph and models the rating
  Heart disease, Obesity, Arthritis, Diabetes, Dyslipidemia,                  graph directly in a single encoder-decoder step.
  Epilepsy, Hypertension                                                         Speciﬁcally, we apply GC-MC (Van den Berg, Kipf, and
•23 behavior health conditions: Physiological conditions,                     Welling2017)onthehealthrecordrepresentedasabipartite
  Psychoactive substance use (Alcohol, Opioid, Cannabis                       graph, and cast the link prediction as a matrix completion
  related),  Non-mood  psychotic,  Mood,  Bipolar,  Affec-                    problem.ThehealthrecordbipartitegraphG(U,V,E) inits
  tive,  Depressive,  Anxiety,  Reaction  to  severe  stress                  initialstateconsistsoflinkssuchas(ui,vj)∈E (individual
  (PTSD),Physical,Eating,Personality,Obsessivecompul-                         medical records), connecting a patient (user) nodeui∈U
  sive (OCD), Intellectual disabilities, Pervasive and devel-                 to a health condition (item) nodevj∈V. A link weight
  opmentaldisorders,BehavioralandEmotional,Attention-                         r∈{1,...,R}represents ordinal severity levels. In parallel,
  deﬁcit hyperactivity (ADHD), Conduct, Emotional, Tic.                       a matrix M(|U|×|V|) stores the observed health severity
•9conditionsrelatedtoadverseevents:3typesofinpatient                          data for|U|patients and|V|health conditions as nonzero
  admissionsaswellas6typesofinjuries:riskfactors,self                         entries,i.e.Mij representsanobservedseverityofcondition
  harm,suicideattempt,symptoms,homicidalsuicidalidea,                         itemvj  on patientsui . The matrix completion task is to
  other injury.                                                               predict the unknown or latent entries ofM   .
                                                                                 Theinputgraphforeachweightrisrepresentedbyanad-
  In  our  bipartite  graph  representation,  each  condition                 jacencymatrixM   r whereallentryvaluesarebinary.Avalue
node stores their corresponding category of conditions or                     of 1  or 0  at rowiand columnjindicates whether or not a
event type as the node features. Attributes for each pa-                      weighted ofrlink exists between vertexiand vertexj, re-
tient node includes the age as the end of the year; gender                    spectively.TheﬁnalinputmatrixMconsistsofM   1,...,M   R .
(male/female);  race/ethnicity  (white-non-Hispanic,  black-                     As an option, the node Features of patient or health con-
non-Hispanic, Hispanic or unknown); and Medicaid enroll-                      dition can be conceptualized in the form of vectors x k  for
ment aid category (Foster Care, Supplemental Security In-                     nodekwhere 1≤k≤|U|+|V|= N , such that the in-
come  -  SSI,  Temporary  Assistance  for  Needy  Families-                   put matrixX   = [ x T1,...,x TN ]T  containing the node features
TANF, and other).                                                             for the graph convolution layer is then chosen as an identity
                         4   Methods                                          matrix, with a unique one-hot vector for every node in the
                                                                              graph. In such way, the graph-based node information can
In principle, the objective of matrix completion (MC) is                      be incorporated seamlessly.
to  approximate  the  matrix  with  a  low-rank  matrix.  i.e.                   The detailed system architecture is describe in the fol-
min   x rank(X ) s.t.xij = yij,∀yij∈Ω , whereX∈Rm×n :                         lowing subsections. In summary, ﬁrstly graph convolutional
the matrix we need to learn; Y∈R  m×n : the original ma-                      encoder, a variant of collaborative ﬁltering auto-encoder
trix (including the known entries and the missing data);                      (Salakhutdinov, Mnih, and Hinton 2007), takes the input
Ω   is the set of known entries in Y; where rank(X  ) is the                  graph formulated as matrix M, and an optional node fea-
maximal number of linearly independent columns of X  ,                        ture matrix X, then produces patient and health condition
so that rank(X  ) is minimized when the variables are in a                    itemnodeembeddings(orlatentrepresentations)zui  (zvj)for
smallest subspace. However, rank minimization is an in-                       a single patienti(itemj). They are done through a form
tractable problem. Among a few methods (Ying et al. 2018;                     of message passing on the bipartite interaction graph. In the
Monti,Bronstein,andBresson2017)toapproximatetheso-                            second phase, the patient and health condition item embed-
lution, the Graph Convolutional Matrix Completion (GC-                        dingpairsareusedtoreconstructthethelinksforeachedge
MC) (Van den Berg, Kipf, and Welling 2017) framework is                       type (rating) through a bi-linear decoder  ˜M = g(Zu,Zv).
adoptedinthisstudy,asitfocusesontheinclusionofgraph-                          Fig.2demonstrates GCN model underlying prediction ar-

chitecture.                                                                   each rating level a separate class in the bipartite interaction
                                                                              graph. Speciﬁcally,
4.1   GraphAuto-Encoder
                                                                                                                       i Qr vj
The conventional autoencoder (VE) is an unsupervised arti-                                     p( ˜Mij = r) =   euT∑ R
ﬁcial neural network that learns how to efﬁciently compress                                                         s=1 euTi Qsvj,                (3)
and encode data to a lower-dimensional representation (em-                    where Qr  a trainable parameter matrix of shapeE×E
bedding), then learns how to use the embedding to recon-                      andE  is the dimensionality of hidden user or item rep-
struct the original input. A variational autoencoder (VAE)                    resentations ui(vj). The predicted rating is computed as
embedstheinputtoaregulariseddistribution,thennewdata                          ˜Mij = g(ui,vj) =  E p(˜Mij = r)[r] =∑
or a random sample is generated from the distribution.                                                                     r∈Rp( ˜Mij = r).
  Variational graph autoencoder (VGAE) is a framework
for unsupervised learning on graph-structured data based                      4.4   Modeltrainingandlossfunction
on the variational auto-encoder (VAE). It achieves com-                       The training objective is to minimize loss function, imple-
petitive results on many link prediction tasks. For a non-                    mentedasthenegativeloglikelihoodofthepredictedratings
probabilisticvariantoftheVGAEmodel,theembeddingsZ                              ˜M ij:
and the reconstructed adjacency matrix  ˆA as follows (Kipf                                              R∑
and Welling 2016b):                                                                  L=− ∑                  I[Mij = r]logp( ˜Mij = r).   (4)
                                                                                             i,j;Ω ij =1r=1
               ˆA  = σ(ZZ   T ),withZ  = f(X,A ).             (1)
  The graph encoder model above takes feature matrixX                            Ω   is used to ﬁlter out the unobserved ratings for the op-
and a graph adjacency matrixA, and produce anN×E                              timization. In order to minimize over-ﬁtting, we randomly
                                              TN ]T . In our bipartite        drop out all outgoing messages of a particular node with a
node embedding matrixZ = [zT1,...,z                                           probabilitypdropout    .
graphG  = (W,E,R) setting, the encoder is analogously                            FortheGAEmodelimplementation,weusesparsematrix
formulated as [U,V] = f(X,M1,...,MR ), whereM r∈                              multiplicationswithcomplexityO∥E∥.Thegraphconvolu-
{0,1}N u×N v  is the adjacency matrix for rating or severity                  tional encoder can be vectorized as
typer∈R, such thatM r  contains 1’s for those elements                                  [U  ]                                [HuHv]
for which the original rating matrixM   contains observed                                V    = f(X,M1...,MR ) = σ(                 W T )        (5)
ratings with valuer.U andV are now matrices of patient
(Nu×E) and item (Nv×E) embeddings, respectively.                                    [HuHv]         R∑                                              )
4.2   Graphconvolutionalencoder                                               with         = σ(       D−1MrXW  Tr ),andM =( 0   M  rM  T
                                                                                                                                            r    0
                                                                                                  r=1
For a graph G, convolutional methods represent a node em-
bedding is generated based on local neighborhood and the                                             5   Experiment
node features with aggregation algorithms. Nodes have em-                     The proposed GC-MC approach is applied to a large sam-
beddings at each layer, where the number of layers is arbi-                   ple of New York State Medicaid enrollees who (1) has full
trary.Inatransductivesetting,theconvolutionresultsinlink                      year of coverage in 2017 and (2) has received a behavior
orratingspeciﬁctypemessagesrpassingfromitemjtopa-                             health diagnose or medication. The diagnoses of each con-
tientito be formulated asµj→i,r  =     1cij Wrxj. Here,cij  is                taining 9  physical or 23   behavioral health conditions dur-
a normalization constant√       |N  (ui)||N  (vj)|, andN  (ui) de-            ing the year for each patient are classiﬁed as ratings of
notesthesetofneighborsofnodei,W   r isanlink-typespe-                         {1,2,3,4}. The rating of 4 indicates most frequently di-
ciﬁc parameter matrix, andx j is the initial feature vector of                agnosed. More than one diagnoses can be recorded on one
nodej. The accumulated messages at every nodeifor each                        claimorvisit.Aseventitems,inpatientadmissionsformen-
rating typercan be expressed as                                               tal health, substance use, and physical health reasons, as
                                                                              well as 6  types of injuries are also classiﬁed on the same
 hui  = σ[accum( ∑            µj→i,1,...,∑            µj→i,R )]  (2)          scale set of 1  through 4  according to their frequencies. The
                    j∈N  i(u i)            j∈N  R (u i)                       datasetscontain596 ,475   patientsorusers,and42  items,and
where   accum()   denotes   a   stack   accumulation   opera-                 their respective user-item interaction graphs when applica-
tion, σ()   denotes  an  element-wise  activation  function                   ble. Among all examined physical health conditions, most
ReLU(.)=max(0,.), such that patientiand health condition                      of prevalent is COPD (23.1%   ), then followed by Obesity
itemjembedding are expressed as ui = σ(Wu hi) , vj  =                         (16.3%   ). 14.3%    had a ADHD diagnosis, and 15.2%    had an
σ(Wvhj) respectively.                                                         anxiety issue.
                                                                                 In order to obtain optimized hyper-parameters, we apply
4.3   Bi-lineardecoder                                                        80/20  train/validationmethodandsplittheoriginaltraining
In general GCN settings, decoder model ˜A = g(Z) takes                        set,whereinteractionsofrandomlyselectedtestingusersare
pairsofnodeembeddings(zi,zj) andpredictsrespectiveen-                         cross validated to estimate the performance of recommen-
                                                                              dation on unseen ratings. We compare the performance of
tries ˜A in the adjacency matrix, corresponding to graph link                 the model with and without node features or characteristics,
reconstruction. Here, a bi-linear decoder is applied to treat                 such as age and gender for patient node, health category for

                                                                                              Table2:ComparisonofaverageRMSEandlossscoresover
                                                                                              5runsfor80/20  training/testdatasetsplits,withandwithout
                                                                                              node features.
             Figure3: (a)RMSE(b)Loss(dropout= 0 .7,accum=  stack,
             includes node features), red: training, blue: validation.
             Table 1: Number of patient and rating frequencies used in
             the experiments. There are 42 conditions. Rating levels are
             1,2,3,4.
                              Dataset       Patients      Ratings      Density
                   Full       Training/
                            Validation     596475     1389244       0.055
                              (80:20)                                                         Figure4: (a)RMSE(b)Loss(dropout= 0 .7,accum=  stack,
                              Testing       157877      347311        0.052
                  Sub 1      Training/                                                        without node features), red: training, blue: validation.
                            Validation     228304      277848        0.029
                              (80:20)                                                         6.1   Baselinemethods
                              Testing        65835        69462         0.025
                  Sub 2      Training/                                                        We perform comparison experiments using the following 3
                            Validation     388798      555697        0.034                    baseline methods that are commonly used in recommenda-
                              (80:20)                                                         tion systems
                              Testing       125380      138924        0.026                      AssociationRules(AR):ARhastheabilitytoefﬁciently
                                                                                              identify what items appear together in the same session. It
                                                                                              hasbeenwidelyusedinrecommendationsystem.Itemsthat
                                                                                              are frequently present together are connected with a link in
             item node. Both types of models are trained for 250    full-                     the graph. Rules mined from the interaction matrix should
             batchepochs.Formodelevaluation,wecomputeaverageof                                haveatleastsomeminimalsupportandconﬁdence.Support
             (Root Mean Square Error) RMSE 1 and Cross-entropy loss                           refers frequency of occurrence, and conﬁdence means that
             onthetestingdataresultingfrom5runs.Table1summarizes                              rules are not often violated.
             the dataset statistics.                                                             K-nearest-neighbor(KNN)graph is a standard method
                Parameter settings in our experiments included: 1) Accu-                      of  collaborative  ﬁltering  (CF),  which  can  be  performed
             mulation function, set as stack or sum (as explained in the                      based on the users or the conditions on the bipartite graph.
             methods).2)Dropoutrateistestedatmultiplevalues,butset                            The user-based models the ratings with ann×m  matrix,
             at0.7.3)Learningrateissetto0.01  fortheAdamoptimizer.                            where userui,i= 1 ,...,nand conditionspj,j= 1 ,...,m.
             4)Basisweightmatricsissetto2fordecoder’sweightshar-                              The goal is to predict the ratingrij  if target useridid not
             ing. 5) Layer sizes of500   and75  for the graph convolution.                    rateaconditionj.Theprocessistocalculatethesimilarities
             Fig.3andFig.4showsRMSEandlossscoresateachepoch;                                  between target useriand all other users, and select the top
             training vs. validation (dropout= 0 .7, accum=   stack). The                     X similarusers.Theweightedaverageofratingsfromthese
             only difference is that nodes attributed are included or not.                    X  users are calculated as:∑
                Additionally,weconductsensitivityanalysistoaccessthe                                         rij =      k similarities(ui,uk)rkj
             model stability. The experiments described above are re-                                                         total ratings    ,              (6)
             peated on randomly selected sub-samples, the ﬁnal results                        wheresimilarityscorecanbecalculatedusingPearsonCor-
             are comparable in terms of average RMSE and Loss, which                          relation or Cosine Similarity. Analogously, for condition-
             are displayed and compared in the Table2.                                        based, two items are similar when they received similar rat-
                                      6   Evaluation                                          ings from a same user, so that we can make prediction for
                                                                                              atargetuseronaconditionbycalculatingweightedaverage
             ToevaluatetheproposedGC-MCapproach,wecomparethe                                  of ratings on mostX  similar items from this user.
             averageofRMSEscoresobtainedfromthemodelandother                                     CollectiveMatrixFactorization(MF):  a  method  that
             baselinemethodsusingthesametrainingset.Table3shows                               decompose the original sparse matrix to low-dimensional
             comparison results.                                                              matrices with latent factors/features and less sparsity. The
                                                                                              goal is to ﬁnd a set of latent features, with align with a user
                               √  ∑ Ni=1  (ri− ˜ri)2                                          and an item. The non-negative factorization has a loss func-
                 1RMSE     =                                                                  tion that is non-convex, that can be solved by a few popular
                                        N        .
Sample     Feature     No Feature     Feature     No Feature
             RMSE          RMSE            Loss             Loss
  Full         0.8622          0.8590          1.0634          1.0750
 Sub 1       0.8594          0.8596          1.0577          1.0649
 Sub 2       0.8442          0.8382          1.0282          1.0151

             Table3:ComparisonofaverageRMSEofasampleoftrain-                               ciated with depression and anxiety. BMC Psychiatry 14(1).
             ing set, with and without node features.                                      1
                                                                                         [Cand`es and Recht 2009]Cand         `es, E., and Recht, B.   2009.
                                                                                           Exact matrix completion via convex optimization.   JAMA
                                                                                           Psychiatry9(6):717–772.2
                                                                                         [Kipf and Welling 2016a]Kipf,  T.  N.,  and  Welling,  M.
                                                                                           2016a.  Semi-supervised classiﬁcation with graph convolu-
                                                                                           tional networks. arXiv:1609.02907.1
                                                                                         [Kipf and Welling 2016b]Kipf,  T.  N.,  and  Welling,  M.
             algorithmse.gStochasticGradientDescent(SGD),istotake                          2016b.       Variational  graph  auto-encoders.       In  arXiv
             derivatives of the loss function with respect to each variable                1611.07308.4
             inthemodelandupdatefeatureweightsoneindividualsam-                          [Koren and Bell 2009]Koren, Y., and Bell, R.  2009.  Exact
             ple at a time, until convergence; Alternative Least Square                    matrixcompletionviaconvexoptimization. JAMAPsychia-
             (ALS)isalternativelytoholduseroritemfactormatrixcon-                          try 42(8):30–37.2
             stant, adjust item or user factor matrix by taking derivatives              [Liben-Nowell and Kleinberg 2007]Liben-Nowell,  D.,  and
             of loss function and setting it to 0, and then hold item fac-                 Kleinberg, J.  2007.  The link-prediction problem for social
             tormatrixconstantwhileadjustinguserfactormatrix,repeat                        networks. J.Am.Soc.Inf.Sci. 58(7):1019–1031.1
             until convergence.                                                          [L¨u and Zhou 2011]L         ¨u,L.,andZhou,T. 2011. Linkpredic-
               Table 3 shows that the proposed GC-MC produces better                       tion in complex networks: A survey. Physica A: Statistical
             link predictions of healthcare data with smaller RMSE, in                     MechanicsanditsApplications390(6):1150–1170.2
             comparison with baseline methods. Note that we have also
             applied KNN in our comparison experiment, but no results                    [Monti, Bronstein, and Bresson 2017]Monti, F.; Bronstein,
             are produced due to its limited scalability issue.                            M. M.; and Bresson, X.   2017.   Geometric matrix com-
               Discussion:  This  end-to-end  GCN  framework  has  re-                     pletion with recurrent multi-graph neural networks.  CoRR
             cently emerged as a powerful deep learning-based approach                     abs/1704.06803.3
             for link prediction. It learns a target node’s representation               [Piane and Smith 2014]Piane,  G.,  and  Smith,  T.     2014.
             by propagating neighbor information in an iterative manner                    Building an evidence base for the co-occurrence of chronic
             untilastableﬁxedpointisreached.It’ssupportedbyalarge                          diseaseandpsychiatricdistressandimpairment. Preventing
             bodyofrecentworktoapplythenovelapproachoversimple                             ChronicDisease.1
             patient and health condition network.                                       [Salakhutdinov, Mnih, and Hinton 2007]Salakhutdinov,  R.;
                                     7   Conclusion                                        Mnih, A.; and Hinton, G. 2007. Restricted boltzmann ma-
                                                                                           chines for collaborative ﬁltering.  In ICML, 791–798.  New
             In this paper, we apply a graph link prediction technique on                  York, NY, USA: ACM.3
             health records for health risk prediction, as an application                [Shickel et al. 2018]Shickel, B.; Tighe, P.; Bihorac, A.; and
             in public healthcare. Patient records from NYS Medicaid                       Rashidi,P. 2018. DeepEHR:Asurveyofrecentadvancesin
             public health data are formulated as a bipartite graph, and a                 deeplearningtechniquesforelectronichealthrecord(EHR)
             recent deep Graph Convolutional Matrix Completion (GC-                        analysis. IEEEJournalofBiomedicalandHealthInformat-
             MC) network is applied to generate risk predictions. Perfor-                  ics 22(5):1589–1604.2
             manceareevaluatedandcomparedwiththreebaselinemeth-                          [Singh and Gordon 2008]Singh, A. P., and Gordon, G. J.
             ods, demonstrating the efﬁcacy of the proposed method.                        2008. Relationallearningviacollectivematrixfactorization.
               Future work of this study includes the generation of                        InKDD, 650–658. New York, NY, USA: ACM.6
             moredynamicandreﬁnedpredictionsbyleveragingthetem-                          [Valderas, Starﬁeld, and Sibbald 2009]Valderas,               J.;
             poral or semantic/causal components of the multi-session                      Starﬁeld, B.;  and Sibbald, C.    2009.    Deﬁning comor-
             health conditions and records. Among the others, subject-                     bidity:  Implications  for  understanding  health  and  health
             speciﬁc and explainable predictions are desired properties                    services. TheAnnalsofFamilyMedicine 7(4):357–363.1
             of health risk and prediction and prevention.
                                                                                         [Van den Berg, Kipf, and Welling 2017]Van  den  Berg,  R.;
                                       References                                          Kipf,T.;andWelling,M. 2017. Graphconvolutionalmatrix
           [Agrawal and Srikant 1994]Agrawal,  R.,  and  Srikant,  R.                      completion. arXiv:1706.02263.2,3
             1994.  Fast algorithms for mining association rules in large                [Ying et al. 2018]Ying, R.; He, R.; Chen, K.; and Eksom-
             databases. VLDB 487–499.6                                                     batchai, P.  2018.  Graph convolutional neural networks for
           [Antonaci, Nappi, and Galli 2011]Antonaci, F.; Nappi, G.;                       web-scale recommender systems. KDD50(2):129–135.3
             and Galli, F.  2011.  Migraine and psychiatric comorbidity:
             a review of clinical ﬁndings. The Journal of Headache and
             Pain 12(2):115–125.1
           [Bhattacharya and Shen 2014]Bhattacharya, R., and Shen,
             C.  2014.  Excess risk of chronic physical conditions asso-
Model                                            RMSE       Feature
AR (Agrawal and Srikant 1994)                    1.92       No
MF (Singh and Gordon 2008)                       1.86       Yes
Proposed method (GC-MC)                          0.86       Yes

