Received 16 December 2022, accepted 31 January 2023, date of publication 9 February 2023, date of current version 17 February 2023.
Digital Object Identifier 10.1109/ACCESS.2023.3243722
Multi-Context Mining-Based Graph Neural
Network for Predicting Emerging Health Risks
JI-WON BAEK                    1  AND KYUNGYONG CHUNG                                         2
1DepartmentofComputerScience,KyonggiUniversity,Suwon,Gyeonggi16227,SouthKorea
2DivisionofAIComputerScienceandEngineering,KyonggiUniversity,Suwon,Gyeonggi16227,SouthKorea
Correspondingauthor:KyungyongChung(dragonhci@gmail.com)
ThisresearchwassupportedbyBasicScienceResearchProgramthroughtheNationalResearchFoundationofKorea(NRF)fundedbythe
MinistryofEducation(2020R1A6A1A03040583).Additionally,thisworkwassupportedbyKyonggiUniversity’sGraduateResearch
Assistantship2021.
   ABSTRACT           Patientswithsimilardiseasesareabletohavesimilartreatments,care,symptoms,andcauses.
   Based on these relations, it is possible to predict latent risks. Therefore, this study proposes Graph Neural
   Network-based Multi-Context mining for predicting emerging health risks. The proposed method first,
   collects and pre-processes chronic disease patients’ disease information, behavioral pattern information,
   and mental health information. After that, it performs context mining. This is a multivariate regression
   analysis for predicting multiple dependent variables, it extracts a regression model and generates a feature
   map. Then, the initial graph is created by defining the number of clusters as nodes and constructing edges
   through correlation. By expanding the graph according to the results of context mining, it is possible to
   predictthatauserhasasimilarchronicdisorderandsimilarsymptomsthroughusers’connectionrelations.
   For performance evaluation, the validity of the regression analysis of context mining used in the proposed
   method,andthesuitabilityoftheclusteringtechniqueareevaluated.
   INDEXTERMS             Multi-contextmining,graphneuralnetwork,emerginghealthrisk,healthcare,knowledge,
   recommendation.
I. INTRODUCTION                                                                                                        purpose [2]. For example, the regression analysis method is
Today, inappropriate living habits cause an increase in the                                                            applied to analyze causal relations between dependent and
number of patients with chronic diseases. Various changes                                                              independentvariables.Accordingly,Baeketal.[3]proposed
inlivingenvironmentsinfluencepeople’smentalhealth,and                                                                  themultipleregression-basedContextDNNforpredictingthe
individuals have different living patterns. Also, with the                                                             risk of depression. Aimed at predicting the risk of mental
increase in life expectancy, people are more interested in                                                             health, the proposed method is to design a Context that rep-
healthcare. Chronic disease is a long-term health condition                                                            resents a set of context information including surrounding
that may improve and worsen repeatedly. Unless it is cared                                                             conditions and time, and to apply it to a neural network.
for or prevented, it can cause complications [1]. Therefore,                                                           Furthermore, it establishes neural networks and connects
itisnecessarytourgepatientstopayattentiontosuchhealth                                                                  the individually learned neural networks via the regression
risks.                                                                                                                 formula. Thus, it paves the way for predicting latent situa-
    The development of information technology and artifi-                                                              tionsthatinfluencementalhealth.However,thisapproachis
cial intelligence draws a lot of attention. Along with that,                                                           limited by a dependent variable. In practice, because there
therehasbeenactiveresearchondataanalysisforpredicting                                                                  are multiple dependent variables, it is necessary to consider
results using reinforcement learning and machine learning                                                              them all. Clustering analysis is used to classify similar or
basedondifferentdatacomprisingnumbers,images,videos,                                                                   relateddataintomultiplegroups.Ithasnopre-definedspecial
etc.,andthesubsequentextractionofsignificantinformation.                                                               purposes. Nevertheless, its advantage is that it relies on data
Data analysis methods have differences depending on their                                                              andobtainsmeaningfulinformationforalldata.Jungetal.[4]
                                                                                                                       proposed the social mining-based clustering process for big
    The associate editor coordinating the review of this manuscript and                                                data integration. For a reliable model, the proposed method
approvingitforpublicationwasDianTjondronegoro                                    .                                     isusedtoapplydifferentweightvaluesthroughstaticmodel
VOLUME 11, 2023                    This work is licensed under a Creative Commons Attribution 4.0 License. For more information, see https://creativecommons.org/licenses/by/4.0/                                                                                                         15153

                                                                                               J.-W. Baek, K. Chung: Multi-Context Mining-Based GNN for Predicting Emerging Health Risks
information and information obtained from social networks,                                                        a gate graph neural network. The proposed method learns
dependingonuserrelations.Clusteringforhealthconditions                                                            itemtrendinformationfrominteractionlogsofimplicitusers
of survivors of an illness enables the prediction of health                                                       and integrates recommendations with trend information of
risks,therebyhelpingtoimprovehealthconditionsbasedon                                                              items. Consequently, it improves the accuracy of represen-
theriskofmedicalaccidentsandexpectancy.Insuchacase,                                                               tation through a self-attention mechanism. By integrating a
asufficientamountofdataisrequiredformodeling.Because                                                              user’sshort-termpreferencewithrecommendationitems,itis
of repeated scanning for pattern extraction, it takes longer                                                      possible to improve the accuracy of recommendations and
to draw the analysis results. For this reason, it is necessary                                                    offer custom representations to a user. In other words, the
to devise a method for analyzing continuously growing data                                                        methodisusedtointegrateitemtrendinformationtoimprove
efficiently.                                                                                                      thecurrentrecommendationitem.Becauseitdesignsamodel
    Social network analysis, recommendation systems, and                                                          with the existing data, it performs poorly in predicting new
knowledge graphs have been actively employed in practice.                                                         data. Therefore, it is necessary to provide a solution to the
With the increase in graph applicability, the graph neural                                                        coldstartproblemintherecommendationsystem.
network(GNN)hasbeenactivelyresearched.Agraphisthe                                                                     This study proposes a multi-context mining-based graph
resultofasetofnodesconnectedwithdirectionalorunidirec-                                                            neural network for predicting emerging health risks. The
tional edges. In addition, relationships or interrelationships                                                    proposed method aims to determine the similarity relations
can be structured and presented visually [5]. Because the                                                         between chronic disease patients according to their behav-
nodesize,numberofneighboringnodes,andfeaturesdiffer,                                                              ioralpatternsandmentalhealthtopredicttherisksofchronic
agraph’sstructureisirregular.Tosolvethisproblem,GNNs                                                              disease patients who have similar features and increased
areapplied.GNNisaneuralnetworkforgraphsthatsupport                                                                awareness of health care and prevention. The contributions
nodeclassification,connectionprediction,andgraphclassifi-                                                         ofthemethodproposedinthispaperareasfollows:
cation.Nodeclassificationistheprocessofclassifyinganode                                                               •  Identify the causal relationship between chronic dis-
throughnodeembeddingundertheconditionthatpartofthe                                                                        eases. It is possible to grasp the causal relationship
graph is labeled. Linked prediction is the process of finding                                                             by generating a feature map based on the influencing
relationsbetweennodesandpredictingthedegreeofassoci-                                                                      factorsofchronicdiseases.
ationandcorrelationbetweentwonodes.Itiswidelyapplied                                                                  •  Early graphs were constructed through clustering and
inrecommendationsystems[6].Inaddition,researchisbeing                                                                     correlation of similar diseases through context mining.
conducted in the field of healthcare based on graphs. For                                                                 Therefore, relationships such as similar behavior pat-
example,Dongetal.[7]proposedtoapplyinggraphrepresen-                                                                      terns, diseases, and symptoms may be formed in each
tationstorecognizesimilarsymptomsofinfluenzabasedon                                                                       clustereduserandusersbetweenclusters.
people’sdailymobility,socialinteraction,andphysicalactiv-                                                             •  Graph relationship representations allow us to measure
ity.However,theanalysisofthedynamicinteractionbetween                                                                     potentialrisksinotherusers.
diseasesymptomsandhumanbehaviorisinsufficient.There-                                                                  This  of  paper  is  composed  as  follows:  in  chapter  2,
fore, when GNNs are formed based on social networks, it is                                                        we describe deep learning-based relationship prediction in
possibletoidentifyinteractionsbetweenpeoplewithsimilar                                                            recommendationsystemsandGNN-basedrelationprediction
symptomsandbehavior.Graphclassificationistheprocessof                                                             applications.Inchapter3,theproposedmulti-contextmining-
classifyinggraphsintovariouscategories.Agraphisdefined                                                            basedgraphneuralnetworkisdescribedforpredictingemerg-
as a connection between neighboring nodes. Accordingly,                                                           inghealthrisks.Inchapter4,therecommendationresultsand
if a particular node removes a connection with its neighbor,                                                      performance evaluation are described, and in chapter 5, the
the node is isolated and is meaningless in the graph. GNNs                                                        conclusionsdrawnfromthisstudyaredescribed.
are categorized into recurrent graph neural networks, spatial
convolutionalnetworks,andspectralconvolutionalnetworks.
In RNN, the hidden layer of the past time step and the input                                                      II. RELATEDWORK
layerofthepresenttimestepareconnectedtopredictcurrent                                                             A. RECOMMENDATIONMETHODUSINGRELATIONSHIP
data.Thenodeusedinarecurrentgraphneuralnetworkisan                                                                PREDICTIONBASEDONGRAPHNEURALNETWORK
RNNunit,andthenetworkisdesigneddifferentlydepending                                                               WiththedevelopmentofICT,itispossibletocollectmultiple
on the edge form. Accordingly, all nodes can obtain infor-                                                        forms of data in various ways anytime and anywhere. As a
mationabouttheirneighboringnodes[8].Aspatialconvolu-                                                              result, the amount of data generated is huge and diverse,
tionalnetworkisemployedforimageclassificationorregion                                                             so each data has a variety of features [10]. In various fields
segmentation. Therefore, its structure is similar to that of                                                      includingmedicalserviceandtrafficcontrol,summary,statis-
CNN,usingthefeaturesofthenodesconnectedinthegraph.                                                                tics,decisionmaking,knowledgesearch,andpatternanalysis
Aspectralconvolutionalnetworkisdevelopedbasedongraph                                                              are applied to extract and employ significant information.
signalprocessing,includingmathematicalfactors.Bysharing                                                           However, data missing, bias, contingency, and other prob-
andupdatingnodeinformationeffectively,itexpandsagraph.                                                            lems arise according to real-time data collection methods,
For example, Tao et al. [9] proposed the item trend learning                                                      devices, and collection targets. As a result, a data shortage
method for a sequence-based recommendation system using                                                           problem occurs during analyses [11]. Data augmentation is
15154                                                                                                                                                                                                                                                                                      VOLUME 11, 2023

J.-W. Baek, K. Chung: Multi-Context Mining-Based GNN for Predicting Emerging Health Risks
appliedtosolvethisproblem.Itisatechniqueofaugmenting
small data through diverse algorithms. With the technique,
it is possible to solve the data set shortage problem and to
consider diverse situations that change differently in real-
time. In fact, data for considering these situations have high
dimensionsandarecomplex,sothatitishardtofindrelations
betweendata[12].Tosolvetheproblemagraphisemployed.
A general graph analysis method requires empirical or pre-
liminary knowledge through breadth-first search, depth-first
search, shortest path algorithm, and clustering. Therefore,
if there are multiple graphs, it is hard to extract significant
information. To continue to present massive data in a graph,
itiseasytolosethestructureofthegraph.Forthesereasons,
Graph Neural Network (GNN), in which deep learning is                                                              FIGURE1.        Recommendation method using relationship prediction based
                                                                                                                   on graph neural network process.
applied to graphs, is employed [13]. A graph consists of
nodes and edges. By connecting data that has various types
of relations, it is possible to conduct an analysis. GNN is an                                                     health, and behavioral patterns and suggest knowledge for
effective framework for learning graph representation [14].                                                        preventing latent emerging health risks. The knowledge rec-
In a GNN, a node calculates a new feature vector while                                                             ommendation model for emerging health risks consists of
collectinginformationfromitsneighboringnodesrepeatedly.                                                            three steps: The first step is data preprocessing. In this step,
After K repeated operations, the node expresses the struc-                                                         the data from the National Health and Nutrition Examina-
tural information in its k-hop neighbor as the captured and                                                        tion Survey are collected; information on patients who have
converted functional vector. Accordingly, through pooling,                                                         high blood pressure, diabetes, and dyslipidemia is obtained.
itispossibletoobtaintheentiregraphrepresentation.Anew                                                              Inaddition,thesepatients’informationonmentalhealthand
designofGNNismostlybasedonanempirical,heuristic,and                                                                behavioral patterns is collected. Unnecessary variables are
experimentalresults.Inaddition,thefeaturesandtheoretical                                                           removed. The second step is multi-context mining. In this
resultsofGNNaremerelyfound,andGNNusesalotofdata.                                                                   step, multivariate and multiple regression analysis is con-
Graph embedding [15] of GNN is a process of embedding                                                              ducted to detect the variables influencing high blood pres-
a graph in a vector space in order for easier data analysis.                                                       sure, diabetes, dyslipidemia, mental health, and behavioral
Withthat,itispossibletosolvediversenetworkproblemsin                                                               patternsfrompreprocesseddata.Accordingly,eachoneofthe
avectorspaceandtousethetechniqueinarecommendersys-                                                                 predictionmodelsisgenerated.Patientswithsimilarchronic
tem[16].Forexample,Liuetal.[17]proposedthereal-time                                                                diseaseshavesimilarmentalhealthsymptomsandbehavioral
social recommendation method based on graph embedding                                                              patterns. For this reason, chronic disease patients are clus-
andtemporalcontext.Theproposedmethodisanewdynamic                                                                  teredaccordingtotheirmentalhealthandbehavioralpatterns,
graph-based embedding model for recommending users and                                                             and user relations are analyzed. The last step is a graph
itemsofinterest.Forreal-timerecommendation,itestablishes                                                           representation of relations between chronic disease patients
a heterogeneous user item (HUI) network. Dynamic graph                                                             andtheexpansionofthegraphbasedontheresultsofcontext-
embedding (DGE) shares the HUI network and builds it                                                               mining. This step is aimed at finding not only invisible user
in a low-dimension space. Accordingly, it captures visual                                                          relations but other user risks that appear in specific users.
meaningeffect,socialrelationship,andsequentialpatternof                                                            Figure2showstheprocessofthemulti-contextmining-based
user behavior in an integrated way. Through simple search                                                          GNNforpredictingemerginghealthrisks.
orsimilaritycalculation,itemploysencodedexpressionsand
generates recommendation items. For node representation
learning,it,however,takesintoaccountneighborinformation                                                            A. DATACOLLECTIONANDPREPROCESSING
only,soitisnecessarytodeviseamethodofconsideringthe                                                                Theusedinthisstudyarethe7thrawdataofferedbynational
similaritiesofvarioususers.Giventhatagraphhasmultiple                                                              health and nutrition examination survey. These three-year
features, it is necessary to employ a method for learning                                                          data(2016to2018)arenationallyrepresentativeandreliable
features.Figure1              showstheRecommendationmethodusing                                                    dataforpeople’shealthlevels,healthbehavioralpatterns,and
relationship prediction based on graph neural network pro-                                                         food & nutrition [18]. The raw data of the National Health
cess.                                                                                                              andNutritionExaminationSurveyarecollectedinthehealth
                                                                                                                   questionnairesurveys,healthexaminationsurveys,andother
III. MULTI-CONTEXTMINING-BASEDGRAPHNEURAL                                                                          surveys,sothattherearemissingdata.Therefore,improving
NETWORKFORPREDICTINGEMERGINGHEALTHRISKS                                                                            the accuracy and reliability of the analysis is required to do
The proposed multi-context mining-based graph neural net-                                                          preprocessing. From the raw data, data about chronic dis-
work for predicting emerging health risks aims to iden-                                                            eases(diabetes,highbloodpressure,anddyslipidemia),men-
tify relationships between data on chronic diseases, mental                                                        tal health, and physical activities are collected. From them,
VOLUME 11, 2023                                                                                                                                                                                                                                                      15155

                                                                                                    J.-W. Baek, K. Chung: Multi-Context Mining-Based GNN for Predicting Emerging Health Risks
 FIGURE2.        The process of the multi-context mining-based graph neural network for predicting emerging health risks.
69 variables and 24,269 persons’ data are extracted. Items                                                              regression analysis estimates relations between at least two
thathavetheanswerofnoidea,noanswer,ornoavailability                                                                     independentvariablesandonedependentvariable[19],[20].
are defined as missing data and are preprocessed. Firstly,                                                              However,inreality,anindependentvariableisinfluencedby
intermsofthemissingvalueprocess,allthedataofthepop-                                                                     differentvariables,ortherearemultipledependentvariables.
ulation with missing values are removed. Secondly, if over                                                              Therefore, multivariatemultiple linearregression analysisis
90% of data for a certain variable includes missing values,                                                             appliedtogenerateaprobabilitymodelofusers’healthcon-
the variable is removed due to its unnecessary influence                                                                ditions. This regression model identifies relations between
on the analysis. Thirdly, variables with duplicate meanings                                                             variableswhenthereareatleasttwodependentvariables[21].
are removed. In the end, 50 variables and 189 persons’ data                                                                  Firstly,Diagnosisofeachchronicdiseaseissetasadepen-
areused.Table1showspreprocessedhealthdata.Thistable                                                                     dent variable, and then relations with independent variables
is consisting of variables, a description of variables, and the                                                         analyze. A regression formula is generated with variables
contentofvariables.                                                                                                     that meet the significance level of 0.05. Regression results
                                                                                                                        presentwiththeusesofthedependentvariable,independent
TABLE1.        Preprocessed health data.                                                                                variable,estimatedvalue,standarderror,t-value,andSignif.
                                                                                                                        (significance level). Table2                 shows the regression results for
                                                                                                                        highbloodpressure.
                                                                                                                        TABLE2.        The regression results for diabetes.
B. FEATUREEXTRACTIONUSINGMULTI-CONTEXTMINING                                                                                 Equation (1) shows the regression formula for high blood
People face challenging situations in their daily activities                                                            pressure. HBP (High Blood Pressure) represents high blood
becauseofhealthissues.Thesesituationsincludeinfluential                                                                 pressure,and1.591isavalueofy-intercept.
latent factors that change in real-time. Along with changes
insituationsandcontext,preferencesarechangedinlinewith                                                                  HBP   =  1.591    + (−0.6517       ×  DI1  − pr)+  (2.025     ×  DI1  − pt)
users’statesandsituations.Therefore,itisnecessarytocon-                                                                                                                                                                           (1)
siderusers’contextchangesovertimeandmakerecommen-
dations through context mining. To analyze causal relations                                                                  In Equation (1), variables that meet the significance level
between variables, a probability for users’ health conditions                                                           and influences high blood pressure, whether to have high
isgeneratedinregressionanalysis.Generallinearregression                                                                 blood pressure at present and whether to treat high blood
analysis estimates relations between one dependent variable                                                             pressure is used. Table            3  shows the regression results for
andoneindependentvariableinastraightline.Multiplelinear                                                                 diabetes.
15156                                                                                                                                                                                                                                                                                      VOLUME 11, 2023

J.-W. Baek, K. Chung: Multi-Context Mining-Based GNN for Predicting Emerging Health Risks
TABLE3.        The regression results for diabetes.                                                                    TABLE5.        The regression results for mental health.
    Equation (2) shows the regression formula for diabetes.                                                                Equation (4) presents the regression model for mental
DM  represents  Diabetes  Mellitus,  and  3  is  a  value  of                                                          health.Thevalueofy-interceptis2.901.
y-intercept.                                                                                                                      mental   − health  =  2.901    + (0.050051      ×  HE   − obe  )
               DM   =  3 + ((3.77E     −  17 )×  HE  − glu )                                                                                                     +  (0.27029     ×  DI  2− pr )
                             +  ((4.12E      −  17 )×  Total   − slp_wk       )                                                                                  +  (−0.15524        ×  HE   − HTG   )
                             +  ((−1.66E         −  15 )×  HE   − HbAlc)          (2)                                                                            +  (0.188145      ×  DE   1− 3)
                                                                                                                                                                  +  (−0.41919         ×  BP1)                  (4)
    In Equation (2), variables that meet the significance level                                                            In Equation (4), mental health is predicted with the fol-
and influence diabetes, fasting blood glucose, average daily                                                           lowing variables: obesity, whether to have dyslipidemia at
sleep hours in weeks, and glycated hemoglobin is used.                                                                 present,whethertohavehypertriglyceridemia,bloodglucose
Table    4 showstheregressionresultsfordyslipidemia.                                                                   care treatment for diabetes (non-pharmacological therapy),
                                                                                                                       andbedtimeinweeks.
TABLE4.        The regression results for dyslipidemia.                                                                    Thirdly,  a  model  for  behavioral  patterns  is  extracted.
                                                                                                                       Whethertohavephysicalactivitiesissetasadependentvari-
                                                                                                                       able, and multiple regression analysis is conducted. Table6
                                                                                                                       showstheregressionresultsforbehavioralpatterns.
                                                                                                                       TABLE6.        The regression results for behavioral patterns.
    Equation (3) shows the regression formula for dyslipi-
demia. Dys represents Dyslipidemia, and 0.3858 is a value
ofy-intercept.
Dys   =  0.3858     + (1.009    ×  DI1  − 2)+ (−0.8275       ×  DI2  − pt)
              +  (−0.4076       ×  DI2  − 2)
              +  (0.002468       ×  HE  − chol)                                  (3)
    In  Equation  (3),  variables  that  meet  the  significance                                                           Equation(5)showstheregressionmodelformentalhealth.
level and influence dyslipidemia, high blood pressure treat-                                                           Thevalueofthey-interceptis2.26.
ment,  blood  pressure  control  drug  intake,  dyslipidemia                                                                  Behavior     − pattern   =  2.26    + (−0.03169        ×  BD1   − 11 )
treatment, dyslipidemia drug intake, and total cholesterol                                                                                                          +  (−0.3247       ×  HE  − nARM     )
isused.                                                                                                                                                             +  (−0.02749        ×  HE  − obe )         (5)
    Secondly, a model for mental health is extracted. To do
that, multiple regression analysis (for one dependent vari-                                                                In Equation (5) for behavioral patterns, blood pressure
able and multiple independent variables) is applied. As a                                                              measurement (arm), whether to have obesity and yearly
dependent  variable,  the  prevalence  of  perceived  stress                                                           drinkingfrequencyareusedasvariables.Throughequations
is  used.  With  independent  variables  that  meet  the  sig-                                                         (1) to (5), a matrix is expressed to extract easily relations
nificance  level  of  0.05,  a  regression  model  is  estab-                                                          between users’ chronic disease data, mental health data,
lished.  Table5 shows  the  regression  results  for  mental                                                           and behavioral patterns data. With the matrix, it is possible
health.                                                                                                                to express data relations easily and to execute operations
VOLUME 11, 2023                                                                                                                                                                                                                                                      15157

                                                                                                   J.-W. Baek, K. Chung: Multi-Context Mining-Based GNN for Predicting Emerging Health Risks
conveniently [22], [23]. The values in the matrix are the
values of a regression formula. Figure                   3  shows the user
datamatrix.
                                                                                                                      FIGURE4.        The results of the Elbow method.
FIGURE3.        The user data matrix.
    Also, among users have similar relations. It is necessary
to increase the accuracy of knowledge recommendations by
usingsimilarityrelationsbetweenusers.Tofindusersimilar-
ities, clustering is performed. As for clustering, a K-means
algorithm with low time complexity is employed. However,
acluster’scorepointisrandomlyselectedsothatatestresult
becomes different. In short, it is impossible to draw a con-
sistent result through clustering [24]. To solve the problem,
the K-means++                  algorithm is used. Although it is similar to
theK-meansalgorithm,itsstepofinitializingacorepointis
different. The procedure of the K-means++                               algorithm is as                               FIGURE5.        The results of user clustering based on the K-means++
follows:                                                                                                              algorithm.
    First,acorepointisrandomlyspecified.Next,thedistance
from a core point closest to each one of the remaining data
is calculated. The next core point is specified according to                                                               Table7shows the results of Silhouette according to the
theprobabilityinproportiontothedistancefromtheclosest                                                                 numberofclusters.Inshort,theresultofSilhouetteisdiffer-
core point. In this way, it is possible to prevent a core point                                                       entdependingonK,thenumberofclusters.
from approaching closely the core point already specified.
Therefore,thegeneralK-means++                           algorithmismorestrate-                                        TABLE7.        Silhouette results by number of clustering.
gicthantheK-meansalgorithmwheninitializingthecentral
point,andoptimizedclusteringispossible[25].Todetermine
the most appropriate number of groups for user clustering,
the Elbow method is applied. According to the result of
WithinClusterSumofSquares(WCSS),itpresentsasection
in which the sum of distances between clusters is sharply
reduced. Such a point is used as the number of clusters.
Nevertheless, in the Elbow method, there is still an unclear
part in determining the number of clusters. For this reason,
Silhouette is applied to evaluate the validity of clusters and
determinethenumberofclusterstouse.Silhouetteisamea-
sure of how similar an object is to its own cluster compared
tootherclusters.Itrangesfrom–1to            +1,wherethehighera                                                             AsshowninTable7,aresultofSilhouettescoredthehigh-
valueis,themoreappropriateclusteringoccurs;thelowerthe                                                                estwhenthenumberofclustersis4.Therefore,fourclusters
valueis,thelessappropriateclusteringoccurs[26].Figure                             4                                   are generated. Figure6shows the results of user clustering
showstheresultsofElbow,wheretheverticalaxisrepresents                                                                 based on the K-means++                        algorithm. Persons who suffer
WCSS, and the horizontal axis represents the number of                                                                fromsimilarchronicdiseaseshavesimilarmentalhealthand
clusters.AsshownintheresultsofElbowevaluationinFig.                            4,                                     behavioralpatterns.Forthisreason,chronicdiseasepatients
the distance between clusters reduces when the number of                                                              are clustered on the basis of mental health and behavioral
clustersis2or4.However,inordertodeterminethenumber                                                                    patterns.
ofmostsuitableclusters,itisdeterminedthroughSilhouette                                                                     In  the  Fig.      6,  the  horizontal  axis  represents  mental
results.                                                                                                              health,  and  the  vertical  axis  means  behavioral  patterns.
15158                                                                                                                                                                                                                                                                                      VOLUME 11, 2023

J.-W. Baek, K. Chung: Multi-Context Mining-Based GNN for Predicting Emerging Health Risks
                                                                                                                    is defined through connections of the clusters that are found
                                                                                                                    to be related in cluster correlation analysis. Based on a core
                                                                                                                    pointofusersimilaritycluster,aprimarynodeisused.Based
                                                                                                                    onaweightrepresentingacorrelationbetweenclusters,edges
                                                                                                                    are connected. As shown in Fig.                 6, since the diagonal ele-
                                                                                                                    ments represent clusters, coefficients of correlations are 1.
                                                                                                                    Inanadjacencymatrix,anode’sowninformationisexcluded.
                                                                                                                    For this reason, connections are made when coefficients of
                                                                                                                    correlations are a positive number except for 1. An initial
                                                                                                                    graphisdesigned.Figure             7 showstheinitialgraphandadja-
                                                                                                                    cencymatrix.
FIGURE6.        Coefficients of correlations between clusters for an adjacency
matrix.
The clustering results in Fig.             6 reveal that although clusters
2and3havesimilarmentalhealth,theyhavedifferentbehav-
ioralpatternsandchronicdiseaserisk;thatalthoughclusters
1and3havesimilarbehavioralpatternsandchronicdisease
risk,theyhavedifferentmentalhealth.
C. GRAPHNEURALNETWORKBASEDON                                                                                        FIGURE7.        Coefficients of correlations between clusters for an adjacency
CONTEXT-MINING                                                                                                      matrix.
Persons with similar disorders have latent associations and                                                              In the case of a feature matrix for augmenting a graph
similarities. A graph is employed to find them. A graph is                                                          withtheuseofnodeinformationsharing,auser’sdatamatrix
made up of a set of node (vertex) V and edge E. It helps to                                                         in Fig.4        is used. Algorithm          1  shows a graph augmentation
representrelationsorinteractionsbetweenobjects,toexpress                                                            algorithmbasedoncontextmining.Itsinputvalueisaninitial
a complex issue in a simple way, and to make expressions                                                            graph.Itsoutputvalueisanaugmentedgraph.
frommultipleperspectivestosolveproblems[27],[28].Also,                                                                   The first step in algorithm 1 is to design an initial graph.
the general graph analysis method that needs empirical and                                                          Nodes are established according to the number of clusters.
preliminary knowledge has difficulty extracting information                                                         Edges are connected according to a positive coefficient of
frommultiplegraphs.Accordingly,deeplearningormachine                                                                correlationbetweenclustersandanadjacencymatrixisgen-
learning-based GNN is applied to a graph. It takes into                                                             erated.Thesecondstepistoaugmentagraphthroughmulti-
accountsimilaritieswithaparticularuser’sdistantneighbors                                                            context mining. A feature matrix for graph augmentation
as well as near neighbors and maintains a graph’s structure.                                                        is a user feature matrix based on the regression model that
Graph Convolutional Network (GCN) is a sort of GNN.                                                                 representsusers’chronicdiseaseinformation,mentalhealth,
It extracts abstract features for input data with no use of                                                         and behavioral patterns. In graph convolution operations,
neighboringnodes’information.Accordingly,bymultiplying                                                              featuresareextracted,andaccordingly,agraphisaugmented.
an adjacency matrix (A), it combines neighboring nodes’                                                             However, depending on the degree of connection between
information. An adjacency matrix specifies a graph shape at                                                         nodes, reliability is different. Therefore, weights are newly
the beginning. In the matrix, the presence of a connection                                                          applied to edges. In this way, it is possible to obtain not
between nodes represents ‘1’, and no presence represents                                                            onlytheprimaryneighboringinformationbutk                               th neighboring
‘0’. Accordingly, an adjacency matrix has no edge from a                                                            information. Equation (6) shows an edge weight based on
vertex to itself, so the diagonal elements of the matrix are                                                        context. As a result, with the reliable weight, node and edge
all zeros. Since a weighted kernel, which uses no its own                                                           informationisupdated.
information, is generated, an identity matrix (I) is added to
solve the problem [29]. As for the weight used in a graph,                                                                                Weight    =  update(A×U                    (k)×  F (k))               (6)
the user data attributes extracted from regression analysis
are used. Through relations between users, such as similar                                                               In Equation (6), ‘A’ means an adjacency matrix; U(k)
chronic diseases, disorders, and behaviors, the grounds for                                                         representsk       th userinformation;F(k)meanskthuser’sfeature
recommendation are found. Figure                    6  shows coefficients of                                        matrix. Figure        8  shows the structure of a graph-based on
correlationsbetweenclustersforanadjacencymatrix.                                                                    multi-context mining. The input graph in Fig.                       8 is user data.
    Anadjacencymatrixrepresentstwo-dimensionalarraysof                                                              Context mining is performed through preprocessing of raw
graph connections. The generation of an adjacency matrix                                                            data. Accordingly, the adjacency matrix and feature matrix
VOLUME 11, 2023                                                                                                                                                                                                                                                      15159

                                                                                                    J.-W. Baek, K. Chung: Multi-Context Mining-Based GNN for Predicting Emerging Health Risks
                              FIGURE8.        The structure of a graph-based on multi-context mining.
Algorithm1        GraphAugmentationAlgorithmBasedonCon-                                                                 toobtainadifferentnode’sinformationgradually.Therefore,
textMining                                                                                                              GNN based emerging health risk prediction is a method of
Input:    IG//initialGraph                                                                                              predicting latent risks on the basis of neighboring informa-
Output:AG//ArgmentedGraph                                                                                               tion. Figure9shows the prediction process using Multi-
intAdj_matrix      =  [[0,0,1,0],[0,0,1,1],[1,1,0,1],[0,1,1,0]]                                                         Context-GNN.
intV,E,w
Graph(V,E,w)       ←   0
Step1:PrimeGraphType//nodeisnumberofcluster
     for  eachv   ∈ Vandeache      ∈ E
        for  v to Numberofcluster          do
            if v <=     4
              node++
            else
              stopcreatingvertex
            endif
        end
        for  e to usingAdjacencyMatrix             do                                                                   FIGURE9.        The prediction process using multi-context-GNN.
           if e =  1
        addedgeandconnectionvertex                                                                                           Forexample,inFig.9,users                    a and   b arenotdirectlycon-
           elseif  e =  0
         notaddedgeandnotconnectionvertex                                                                               nectedwitheachother,butitisfoundthattheyareassociated
        end                                                                                                             witheachotheronthebasisofinformationofuser                          c.Inother
end                                                                                                                     words,withgraphaugmentation,userrelationshipcontinues
Step2:GraphArgumentusinggraphconvolution                                                                                to be generated. Accordingly, even if users are not directly
      UserMatrix     =  usingresultofregressionmodel                                                                    associated with each other, it is possible to predict a latent
      for  F to UserMatrixandAdjacencyMatrix                   do                                                       riskwiththeuseofaneighbor’sinformation.
         F[i][j]  =  Convolution(UserMatrix                 i×  AdjacencyMatrix             j)
          G =  F[i][j]//ArgumentGraph                                                                                    IV. PERFORMANCEEVALUATION
      end                                                                                                               RawdatafromtheNationalHealthandNutritionSurvey[18]
      return                                                                                                            were used in the experiments. The purpose of the national
                                                                                                                        healthsurveyistocalculatestatisticswithnationalrepresen-
                                                                                                                        tationandreliabilityregardingthehealthlevel,healthbehav-
of an initial graph is generated. In graph networks, weights                                                            ior,andfoodandnutritionintakeofthepeople.Accordingly,
areupdatedwithgraphconvolutionoperationandtheweight                                                                     itprovidesbasicdataforhealthpolicies,suchasgoalsetting
operationofEquation(6).Agraphisaugmentedinthecourse                                                                     and evaluation of the comprehensive national health promo-
of sharing a particular node’s information with a different                                                             tionplans,andhealthpromotionprogramdevelopments.For
node. In this way, it is possible to classify a node, find                                                              the sampling frame of the National Health and Nutrition
relationsbetweennodes,andpredictadegreeofassociation.                                                                   ExaminationSurvey,themostrecentdatafromthePopulation
Neighboring nodes have similar attributes. Therefore, based                                                             and Housing Census available at the time of sample design
on the node and edge information in layer 1, it is possible                                                             was used as the basic extraction frame. It supplements the
15160                                                                                                                                                                                                                                                                                      VOLUME 11, 2023

J.-W. Baek, K. Chung: Multi-Context Mining-Based GNN for Predicting Emerging Health Risks
basic extraction frame by adding and improving the popula-                                                              As data used in the regression analysis for designing a
tion inclusion rate. In this study, chronic disease, behavioral                                                    featuremapoftheproposedmodel,highbloodpressure,dia-
patterns, and mental health-related data were selected from                                                        betes,anddyslipidemiadataamongrawdataoftheNational
the raw data present in the National Health and Nutrition                                                          Health and Nutrition Examination Survey are used. It is
Examination Survey, and preprocessing was performed on                                                             necessarytoprovethatperformanceisbetterwhenafeature
the data related to 69 variables and 24,269 people. Through                                                        mapisgeneratedviamultivariateregressionanalysisconsid-
preprocessing,missingvaluesareprocessed,andunnecessary                                                             ering all three chronic diseases than that obtained when a
variablesandvariableswithduplicatemeaningsareremoved.                                                              feature map is generated via univariate regression analysis
As a result, preprocessed data are divided into training data                                                      considering each disease individually. Table                      8  shows the
(70%), test data (20%), and validation data (10%). In terms                                                        resultsoftheregressionanalysis.Itshowstheresultsfromthe
of performance, the validity of the regression analysis used                                                       comparison between univariate regression analysis on each
for a user feature matrix is assessed, a clustering method                                                         one of diabetes, high blood pressure, and dyslipidemia) and
is evaluated, and the proposed model is compared with a                                                            multivariateregressionanalysisonallthreechronicdiseases.
conventionalmodel.
    Firstly, the validity of regression analysis is assessed.                                                      TABLE8.        The results of regression analysis evaluation.
Inthisstudy,toestablishafeaturematrixofagraph,aregres-
sionmodelisgeneratedastheresultofmultivariateanalysis.
Tofindthevalidityofthemultivariateanalysis-basedregres-
sion model, univariate analysis is compared with multivari-
ate analysis. As for performance indexes, the coefficient of
determinationdenotedR2,AdjustedR-Squared,andMSEare
used.R2isusedtoevaluatepredictionperformancebasedon
distribution [30]. The larger the coefficient of determination
is,themoretheactualvalueissimilartothepredictedvalue,                                                                   In Table8, the coefficient of determination for diabetes
and the better the data explanation. Equation (7) states the                                                       in the univariate analysis is the highest but has the largest
expressionforR          2.                                                                                         difference from R            2  adj. It means that explanatory variables
                               R 2 =   SSE                                                                         fordiabetesincludeunnecessaryvariables.Therefore,inthe
                                          SST   =  1 −   SSRSST                           (7)                      proposedmethod,multivariateanalysisisthemosteffective.
    InEquation(7),theTotalSumofSquares(SST)isthesum                                                                     Secondly, the clustering method for creating an initial
of the differences between the mean of observed values and                                                         graphisevaluated.Intheproposedmethod,aninitialgraphis
an observed value. Explained Sum of Squares (SSE) is the                                                           generated through K-means++                           algorithm-based clustering.
sumofthedifferencesbetweenthemeanofobservedvalues                                                                  Whenacluster’scorepointisrandomlyselected,itisdifficult
and a predicted value. The Residual Sum of Squares (SSR)                                                           to obtain a consistent result. Therefore, the K-means++
is the sum of the residuals between a predicted value and an                                                       algorithm prevents the problem [33], [34]. In the case of the
observed value. The closer the coefficient of determination                                                        generalK-meansalgorithm,acluster’scorepointischanged
(R  2) is to 1, the better performance occurs. This influences                                                     such that the initial graph loses consistency. To prove that
the number of independent variables. In short, it increases                                                        the K-means++                   algorithm used in the proposed model is
with the number of independent variables. For this reason,                                                         moreappropriatethantheK-meansalgorithm,itisnecessary
it is difficult to evaluate performance accurately. To solve                                                       to compare K-means and K-means++                                algorithms in terms
theproblem,anAdjustedR-Squaredisused.Toolowerthan                                                                  of Precision, Recall, and F-measure. Figure                      10   shows the
Adjusted R-Squared means that unnecessary independent                                                              F-measure values according to the number of clusters of
variablesareincluded[31].Equation(8)statestheexpression                                                            K-meansandK-means++.
fortheadjustedR-squaredvalue.                                                                                           Table     9  shows the precision, recall, and f-measure of
                                                                                                                   K-meansandK-means++                         whenthenumberofclusterswith
                        Adj  − R =  1 −  (n  −  1)(1  −  R 2)n −  p −  1                        (8)                excellentF-measureinFigure12is4.
    In Equation (8), n represents the number of data, and p                                                        TABLE9.        Comparison results of precision, recall, and f-measure of
means the number of independent variables. The closer the                                                          clusters.
AdjustedR-Squaredisto1,themorepredictionisaccurate.A
negativevalueofAdjustedR-Squaredmeansthataregression
model is useless. For the evaluation of a regression analysis
model,MSEisemployed.Itrepresentsthemeanofthesquare
of the difference between actual and predicted values [32].
Equation(9)showstheexpressionofMSE.                                                                                     As  presented  in  Figure             10    and  Table        9,  when  the
                                               ∑N                                                                  K-means++                 algorithmisused,theperformanceisexcellent.
                           MSE    =   1N             i=1    (y  i−ˆyi)2                     (9)                    And in Table        9, the K-means algorithm randomly selects a
VOLUME 11, 2023                                                                                                                                                                                                                                                      15161

                                                                                                 J.-W. Baek, K. Chung: Multi-Context Mining-Based GNN for Predicting Emerging Health Risks
                                                                                                                    the method proposed by Wang et al. [37] does not include
                                                                                                                    the semantic or causal component of the disease and uses
                                                                                                                    sparse data. Therefore, the obtained accuracy of health risk
                                                                                                                    and prediction is low. On the other hand, in the proposed
                                                                                                                    method, a graph is augmented using neighbor information
                                                                                                                    so that it is possible to obtain information according to a
                                                                                                                    2-hoprelationshipaswellasa1-hoprelationship.Therefore,
                                                                                                                    compared to conventional methods, the proposed method
                                                                                                                    attainsexcellentperformance.
FIGURE10.         The F-measure values according to the number of clusters of
K-means and K-means++.
                                                                                                                     V. CONCLUSION
                                                                                                                    A graph has the advantages of being convenient to expand
cluster’s core point, and consequently, recall is low. Given                                                        andbeingabletointuitivelypresenttherelationshipbetween
that, K-means++                   is appropriate for generating an initial                                          nodes. This study proposed a multi-context mining-based
graphoftheproposedmodel.                                                                                            graph neural network for predicting emerging health risks.
    Lastly,toevaluatetheexcellenceoftheproposedmethod,                                                              The proposed method predicts and recommends potential
it is compared with conventional GCN-based recommender                                                              emerging risks through a graph neural network based on
models. For performance evaluation, the MSE and recall                                                              information regarding similar symptoms, causes, and man-
were employed. Wu et al. [35] proposed a graph convo-                                                               agement methods for patients with chronic diseases. It con-
lutional network-based model for social recommendation,                                                             sisted of three steps. The first step was to collect and pre-
which was generated based on the expansion of social net-                                                           process health information, mental health information, and
works and user-item preferences. However, it struggles with                                                         behavioral patterns information of chronic disease patients
predicting a latent factor that is not included in the user                                                         who suffer from high blood pressure, diabetes, and dyslipi-
features.Yangetal.[36]proposedagraphnetworkforsolv-                                                                 demia.Thesecondstep,contextminingwasperformedusing
ingsocialinconsistencyproblems.Intheirproposedmethod,                                                               preprocessed data to generate a feature map for the graph
a sampling probability is associated with the score of the                                                          extension. In multivariate regression analysis, a regression
neighbors’ consistency, and thus, consistent neighbors are                                                          model that has high blood pressure, diabetes, and dyslipi-
sampled.Thismethodwaslimitedowingtoconsideringonly                                                                  demia as dependent variables were extracted. In addition,
the information of consistent neighbors. R. Wang et al. [37]                                                        in linear regression analysis, a regression model for mental
appliedadeepgraphnetworkfortheanalysisandprediction                                                                 health and behavioral patterns was generated, and a feature
of patient health comorbidities from sparse health records.                                                         mapwascreated.Throughclustering,thenodesoftheinitial
Theirapproachrepresentspatientdataincludinghealthexam-                                                              graphwerecreated.Accordingtothecorrelationcoefficients,
ination categories, hospitalization, and injury accidents in a                                                      theedgesoftheinitialgraphweredesigned.Thelaststepwas
graph structure, and models patient health trends, disease                                                          toaugmentthegraphwiththefeaturemapgeneratedthrough
prognosis, and potential correlations by recovering missing                                                         contextminingoftheinitialgraphandtoupdateweightsfor
connections through connection prediction problems [38].                                                            predicting latent risks not only in relations between distant
Table10         showstheperformancecomparisonresultsbetween                                                         neighbors but in relations between close neighbors. In this
theconventionalmodelsandtheproposedmethod.                                                                          way, it was possible to find similar symptoms and causes of
                                                                                                                    all users and to predict emerging risks. Performance eval-
TABLE10.         The results of comparison with conventional models.                                                uation was conducted in three ways. First, the validity of
                                                                                                                    theregressionanalysisfor patientswithchronicdiseasewas
                                                                                                                    evaluated.Asaresult,performancewasbetterinmultivariate
                                                                                                                    analysis than in univariate analysis, in which each chronic
                                                                                                                    diseasewassetasadependentvariable.Second,thecluster-
                                                                                                                    ingmethodfordeterminingthenumberofnodesinaninitial
                                                                                                                    graphwasevaluated.Asaresult,theK-means++                                  algorithm
                                                                                                                    achieved better performance because it overcame the prob-
                                                                                                                    lem of the K-means algorithm, where a cluster’s core point
    As shown in Table            10, the proposed method has excel-                                                 varied. Finally, to evaluate the excellence of the proposed
lent performance. The technique proposed by Wu et al. [35]                                                          model,itwascomparedwithconventionalmodelsintermsof
considers only the neighbors’ information, hence, its per-                                                          MSEandrecall.Asaresult,theproposedmethodsolvedthe
formance is low on the augmented graph. The method pro-                                                             problemsofconventionalmethodssothatitsMSEandRecall
posed by Yang et al. [36] considers the information of the                                                          were about 0.1-0.2 higher than those of conventional ones.
mostrelateduser.Therefore,ifneighborshavenoconsistent                                                               Therefore, it is possible to effectively predict the potential
information or a new user, it is hard to make a graph-based                                                         risk through the proposed model; accordingly, information
analysis, and performance is evaluated as low. In addition,                                                         thatcanpreventhealthriskscanbeprovided.
15162                                                                                                                                                                                                                                                                                      VOLUME 11, 2023

J.-W. Baek, K. Chung: Multi-Context Mining-Based GNN for Predicting Emerging Health Risks
REFERENCES                                                                                                                                           [24]  K. P. Sinaga and M.-S. Yang, ‘‘Unsupervised K-means clustering algo-
  [1]  H. Yoo and K. Chung, ‘‘Deep learning-based evolutionary recommenda-                                                                                     rithm,’’    IEEEAccess,vol.8,pp.80716–80727,2020.
          tion model for heterogeneous big data integration,’’                           KSII Trans. Internet                                        [25]  J.  Hämäläinen,  T.  Kärkkäinen,  and  T.  Rossi,  ‘‘Improving  scalable
          Inf.Syst.,vol.14,no.9,pp.3730–3744,Sep.2020.                                                                                                         K-means++,’’          Algorithms,vol.14,no.1,p.6,Dec.2020.
  [2]  H. Yoo, R. C. Park, and K. Chung, ‘‘IoT-based health big-data process                                                                         [26]  G. Ogbuabor and U. F. N, ‘‘Clustering algorithm for a healthcare dataset
          technologies: A survey,’’              KSII Trans. Internet Inf. Syst., vol. 15, no. 3,                                                              using silhouette score value,’’               Int. J. Comput. Sci. Inf. Technol., vol. 10,
          pp.974–992,2021.                                                                                                                                     no.2,pp.27–37,Apr.2018.
  [3]  J.-W. Baek and K. Chung, ‘‘Context deep neural network model for pre-                                                                         [27]  K.Zhan,C.Niu,C.Chen,F.Nie,C.Zhang,andY.Yang,‘‘Graphstructure
          dicting depression risk using multiple regression,’’                         IEEE Access, vol. 8,                                                    fusionformultiviewclustering,’’                 IEEE Trans. Knowl. Data Eng.,vol.31,
          pp.18171–18181,2020.                                                                                                                                 no.10,pp.1984–1993,Oct.2019.
  [4]  H. Jung and K. Chung, ‘‘Social mining-based clustering process for big-                                                                       [28]  Z.Zhang,J.Bu,M.Ester,J.Zhang,C.Yao,Z.Yu,andC.Wang,‘‘Hierar-
          data integration,’’         J. Ambient Intell. Humanized Comput., vol. 12, no. 1,                                                                    chicalgraphpoolingwithstructurelearning,’’2019,                          arXiv:1911.05954.
          pp.589–600,Jan.2021.                                                                                                                       [29]  L. Yao, C. Mao, and Y. Luo, ‘‘Graph convolutional networks for text
                                                                                                                                                               classification,’’ in        Proc. AAAI Conf. Artif. Intell., vol. 33, Jul. 2019,
  [5]  Z. Guo and H. Wang, ‘‘A deep graph neural network-based mechanism                                                                                       pp.7370–7377.
          for social recommendations,’’                 IEEE Trans. Ind. Informat., vol. 17, no. 4,                                                  [30]  J. D. Rights and S. K. Sterba, ‘‘New recommendations on the use of
          pp.2776–2783,Apr.2021.                                                                                                                               R-Squared differences in multilevel model comparisons,’’                                 Multivariate
  [6]  M. Zhang, P. Li, Y. Xia, K. Wang, and L. Jin, ‘‘Labeling trick: A theory                                                                                Behav.Res.,vol.55,no.4,pp.568–599,Jul.2020.
          of using graph neural networks for multi-node representation learning,’’                                                                   [31]  T. Hayes, ‘‘R-squared change in structural equation models with latent
          2020,    arXiv:2010.16103.                                                                                                                           variables  and  missing  data,’’                Behav.  Res.  Methods,  vol.  53,  no.  5,
  [7]  G. Dong, L. Cai, D. Datta, S. Kumar, L. E. Barnes, and M. Boukhechba,                                                                                   pp.2127–2157,Mar.2021.
          ‘‘Influenza-like symptom recognition using mobile sensing and graph                                                                        [32]  A. de Myttenaere, B. Golden, B. Le Grand, and F. Rossi, ‘‘Mean abso-
          neural networks,’’ in           Proc. Conf. Health, Inference, Learn., Apr. 2021,                                                                    lute percentage error for regression models,’’                       Neurocomputing, vol. 192,
          pp.291–300.                                                                                                                                          pp.38–48,Jun.2016.
  [8]  K. Guo, Y. Hu, Z. Qian, H. Liu, K. Zhang, Y. Sun, J. Gao, and B. Yin,                                                                         [33]  K. Makarychev, A.  Reddy, and L. Shan,  ‘‘Improved Guarantees for
          ‘‘Optimized graph convolution recurrent neural network for traffic pre-                                                                              k-means++ and k-means++ Parallel,’’ in                        Proc. Adv. Neural Inf. Process.
          diction,’’    IEEE Trans. Intell. Transp. Syst., vol. 22, no. 2, pp.1138–1149,                                                                       Syst.,vol.33,2020,pp.16142–16152.
          Feb.2021.                                                                                                                                  [34]  J.  Hämäläinen,  T.  Kärkkäinen,  and  T.  Rossi,  ‘‘Improving  scalable
  [9]  Y. Tao, C. Wang, L. Yao, W. Li, and Y. Yu, ‘‘Item trend learning for                                                                                    K-means++,’’          Algorithms,vol.14,no.1,pp.6–25,Jun.2020.
          sequential recommendation system using gated graph neural network,’’                                                                       [35]  L.Wu,P.Sun,R.Hong,Y.Fu,X.Wang,andM.Wang,‘‘SocialGCN:An
          NeuralComput.Appl.,pp.1–16,Feb.2021.                                                                                                                 efficient graph convolutional network based model for social recommen-
[10]  H. Jung and K. Chung, ‘‘Knowledge-based dietary nutrition recommen-                                                                                      dation,’’2018,        arXiv:1811.02815.
          dation for obese management,’’                   Inf. Technol. Manage., vol. 17, no. 1,                                                    [36]  L.Yang,Z.Liu,Y.Dou,J.Ma,andP.S.Yu,‘‘ConsisRec:EnhancingGNN
          pp.29–42,Mar.2016.                                                                                                                                   forsocialrecommendationviaconsistentneighboraggregation,’’in                                    Proc.
[11]  S. Kobayashi, ‘‘Contextual augmentation: Data augmentation by words                                                                                      44thInt.ACMSIGIRConf.Res.Develop.Inf.Retr.,2021,pp.2141–2145.
          withparadigmaticrelations,’’2018,                  arXiv:1805.06201.                                                                       [37]  R.Wang,M.C.Chang,andM.Radigan,‘‘Modelinglatentcomorbidityfor
[12]  Z. Wu, S. Pan, F. Chen, G. Long, C. Zhang, and S. Y. Philip, ‘‘A com-                                                                                    health risk prediction using graph convolutional network,’’ in                               Proc. 33rd
          prehensive survey on graph neural networks,’’                          IEEE Trans. Neural Netw.                                                      Int.FlairsConf.,2020,pp.341–346.
          Learn.Syst.,vol.32,no.1,pp.4–24,Mar.2020.                                                                                                  [38]  D.-H.Shin,R.C.Park,andK.Chung,‘‘Decisionboundary-basedanomaly
[13]  E.Chien,J.Peng,P.Li,andO.Milenkovic,‘‘Adaptiveuniversalgeneral-                                                                                          detectionmodelusingimprovedAnoGANfromECGdata,’’                                      IEEEAccess,
          izedPageRankgraphneuralnetwork,’’2020,                          arXiv:2006.07988.                                                                    vol.8,pp.108664–108674,2020.
[14]  Y.Jing,J.Wang,W.Wang,L.Wang,andT.Tan,‘‘Relationalgraphneural
          networkforsituationrecognition,’’                 PatternRecognit.,vol.108,Dec.2020,                                                                                                       JI-WON  BAEK             received the B.S. degree from
          Art.no.107544.                                                                                                                                                                             the School of Computer Information Engineer-
[15]  B. Abu-Salih, M. Al-Tawil, I. Aljarah, H. Faris, P. Wongthongtham,                                                                                                                             ing, Sangji University, South Korea, in 2017, and
          K.Y.Chan, and A. Beheshti, ‘‘Relational learning analysis of social pol-                                                                                                                   the master’s degree from the School of Depart-
          itics using knowledge graph embedding,’’                         Data Min. Knowl. Discov.,                                                                                                 ment of Computer Science, Kyonggi University,
          vol.35,no.4,pp.1497–1536,2021.                                                                                                                                                             South Korea, in 2020, where she is currently pur-
[16]  X. Sha, Z. Sun, and J. Zhang, ‘‘Hierarchical attentive knowledge graph                                                                                                                         suing the doctorate degree with the Department
          embeddingforpersonalizedrecommendation,’’2019,                                arXiv:1910.08288.                                                                                            of Computer Science. She was at the Data Man-
[17]  P.Liu,L.Zhang,andJ.A.Gulla,‘‘Real-timesocialrecommendationbased                                                                                                                                agement Department, Infiniq Company Ltd. She
          on graph embedding and temporal context,’’                          Int. J. Hum.-Comput. Stud.,                                                                                            is a Researcher with the Data Mining Laboratory,
          vol.121,pp.58–72,Jan.2019.                                                                                                                 Kyonggi University. Her research interests include data mining, data man-
[18]    The Seventh Korea National Health and Nutrition Examination Survey.                                                                          agement, knowledge systems, automotive testing, deep learning, medical
          Accessed:Jun.27,2020.[Online].Available:https://knhanes.kdca.go.kr/                                                                        datamining,healthcare,andrecommendation.
[19]  C. Wang, B. Zhao, L. Luo, and X. Song, ‘‘Regression analysis of current
          status data with latent variables,’’                Lifetime Data Anal., vol. 27, no. 3,
          pp.413–436,Apr.2021.                                                                                                                                                                       KYUNGYONG CHUNG                      received the B.S., M.S.,
[20]  C. Maheswari, E. B. Priyanka, S. Thangavel, S. V. R. Vignesh, and                                                                                                                              and Ph.D. degrees from the Department of Com-
          C.Poongodi,‘‘Multipleregressionanalysisforthepredictionofextraction                                                                                                                        puter Information Engineering, Inha University,
          efficiency in mining industry with industrial IoT,’’                         Prod. Eng., vol. 14,                                                                                          South Korea, in 2000, 2002, and 2005, respec-
          no.4,pp.457–471,Jun.2020.                                                                                                                                                                  tively.HewasattheSoftwareTechnologyLeading
[21]  S.Ghosal,B.Sinha,M.Majumder,andA.Misra,‘‘Estimationofeffectsof                                                                                                                                 Department, South Korea IT Industry Promotion
          nationwidelockdownforcontainingcoronavirusinfectiononworseningof                                                                                                                           Agency (KIPA). From 2006 to 2016, he was a
          glycosylatedhaemoglobinandincreaseindiabetes-relatedcomplications:                                                                                                                         Professor at the School of Computer Informa-
          A simulation model using multivariate regression analysis,’’                                Diabetes                                                                                       tionEngineering,SangjiUniversity,SouthKorea.
          MetabolicSyndrome,Clin.Res.Rev.,vol.14,no.4,pp.319–323,Jul.2020.                                                                                                                           Since 2017, he has been a Professor with the
[22]  Z.Li,Z.Hu,F.Nie,R.Wang,andX.Li,‘‘Matrixcompletionwithcolumn
          outliersandsparsenoise,’’             Inf.Sci.,vol.573,pp.125–140,Sep.2021.                                                                Division of AI Computer Science and Engineering, Kyonggi University,
[23]  A. Alvarez-Melcon, X. Wu, J. Zang, X. Liu, and J. S. Gomez-Diaz,                                                                               SouthKorea.HewasnamedaHighlyCitedResearcherbyClarivateAnalyt-
          ‘‘Coupling matrix representation of nonreciprocal filters based on time-                                                                   icsin2017.Hisresearchinterestsincludedatamining,artificialintelligence,
          modulated resonators,’’              IEEE Trans. Microw. Theory Techn., vol. 67,                                                           healthcare,knowledgesystems,HCI,andrecommendationsystems.
          no.12,pp.4751–4763,Dec.2019.
VOLUME 11, 2023                                                                                                                                                                                                                                                      15163

