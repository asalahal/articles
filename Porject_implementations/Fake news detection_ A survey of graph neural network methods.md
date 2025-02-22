                                AppliedSoftComputing139(2023)110235
                               Contents lists available atScienceDirect
                               AppliedSoftComputing
                           journal homepage:www.elsevier.com/locate/asoc
Reviewarticle
Fakenewsdetection:Asurveyofgraphneuralnetworkmethods
HuyenTrangPhana,b,NgocThanhNguyenc,∗,DosamHwanga,∗
aDepartment of Computer Engineering, Yeungnam University, Gyeongsan, South Korea
bFaculty of Information Technology, Nguyen Tat Thanh University, Ho Chi Minh, Vietnam
cDepartment of Applied Informatics, Wroclaw University of Science and Technology, Wroclaw, Poland
a r t i c l e    i n f o   a b s t r a c t
Article history:           Theemergenceofvarioussocialnetworkshasgeneratedvastvolumesofdata.Efficientmethods
Received5March2022         forcapturing,distinguishing,andfilteringrealandfakenewsarebecomingincreasinglyimportant,
Receivedinrevisedform3August2022especiallyaftertheoutbreakoftheCOVID-19pandemic.Thisstudyconductsamultiaspectand
Accepted19March2023        systematicreviewofthecurrentstateandchallengesofgraphneuralnetworks(GNNs)forfake
Availableonline24March2023newsdetectionsystemsandoutlinesacomprehensiveapproachtoimplementingfakenewsdetection
Keywords:                  systemsusingGNNs.Furthermore,advancedGNN-basedtechniquesforimplementingpragmaticfake
Fakenews                   newsdetectionsystemsarediscussedfrommultipleperspectives.First,weintroducethebackground
Fakenewscharacteristics    andoverviewrelatedtofakenews,fakenewsdetection,andGNNs.Second,weprovideaGNN
Fakenewsfeatures           taxonomy-basedfakenewsdetectiontaxonomyandreviewandhighlightmodelsincategories.
Fakenewsdetection          Subsequently,wecomparecriticalideas,advantages,anddisadvantagesofthemethodsincategories.
Graphneuralnetwork         Next,wediscussthepossiblechallengesoffakenewsdetectionandGNNs.Finally,wepresentseveral
                           openissuesinthisareaanddiscusspotentialdirectionsforfutureresearch.Webelievethatthisreview
                           canbeutilizedbysystemspractitionersandnewcomersinsurmountingcurrentimpedimentsand
                           navigatingfuturesituationsbydeployingafakenewsdetectionsystemusingGNNs.
                                                                 ©2023ElsevierB.V.Allrightsreserved.
Contents
1. Introduction.........................................................................................................................................................................................................................2
2. Background..........................................................................................................................................................................................................................3
    2.1. Understandingfakenews.....................................................................................................................................................................................3
    2.2. Fakenewsdetection..............................................................................................................................................................................................4
        2.2.1. Whatisfakenewsdetection?..............................................................................................................................................................4
        2.2.2. Fakenewsdetectiondatasets...............................................................................................................................................................4
        2.2.3. Featuresoffakenewsdetection..........................................................................................................................................................6
        2.2.4. Fakenewsdetectiontechniques..........................................................................................................................................................8
    2.3. Understandinggraphneuralnetworks...............................................................................................................................................................10
        2.3.1. Whatisagraph?....................................................................................................................................................................................10
        2.3.2. Whataregraphneuralnetworks?.......................................................................................................................................................10
3. Surveymethodology..........................................................................................................................................................................................................11
4. Quantitativeanalysisofeligiblepapers...........................................................................................................................................................................12
5. Literaturesurvey................................................................................................................................................................................................................13
    5.1. DetectionapproachbasedonGNNs∗..................................................................................................................................................................14
    5.2. DetectionapproachbasedonGCNs....................................................................................................................................................................15
    5.3. DetectionapproachbasedonAGNNs.................................................................................................................................................................17
    5.4. DetectionapproachbasedonGAEs.....................................................................................................................................................................18
6. Discussion............................................................................................................................................................................................................................18
    6.1. DiscussiononGNNs∗-basedmethods.................................................................................................................................................................18
    6.2. DiscussiononGCNs-basedmethods....................................................................................................................................................................18
    6.3. DiscussiononAGNNs-andGAEs-basedmethods.............................................................................................................................................19
7. Challenges............................................................................................................................................................................................................................19
 ∗Correspondingauthors.
  E-mail addresses:    huyentrangtin@ynu.ac.kr,pthtrang@ntt.edu.vn(H.T.Phan),Ngoc-Thanh.Nguyen@pwr.edu.pl(N.T.Nguyen),dshwang@yu.ac.kr(D.Hwang).
https://doi.org/10.1016/j.asoc.2023.110235
1568-4946/©2023ElsevierB.V.Allrightsreserved.

H.T. Phan, N.T. Nguyen and D. Hwang                                                                                                                                                                                            Applied Soft Computing 139 (2023) 110235
   7.1. Fakenewsdetectionchallenges...........................................................................................................................................................................19
   7.2. Challengesrelatedtographneuralnetworks....................................................................................................................................................20
8. Conclusionandopenissues..............................................................................................................................................................................................22
   Declarationofcompetinginterest....................................................................................................................................................................................23
   Dataavailability..................................................................................................................................................................................................................23
   Appendix.Descriptionofdatasets....................................................................................................................................................................................23
   References...........................................................................................................................................................................................................................23
                                        ofnewsposted,anduserinteractions.Consequently,socialnet-
1.Introduction                          worksnaturallybecomecomplexgraphstructuresiftheyare
                                        appliedindependently,whichisproblematicforpreviousma-
 Recently,socialnetworkshavecontributedtoanexplosionchinelearning-basedanddeeplearning-basedfakenewsdetec-
ofinformation.Socialnetworkshavebecomethemaincommu-tionalgorithms.Themainreasonsforthisphenomenonarethe
nicationchannelforpeopleworldwide.However,theveracitydependenceofthegraphsizeonthenumberofnodesandthe
ofnewspostedonsocialnetworksoftencannotbedetermined.differentnumbersofnodeneighbors.Therefore,someimpor-
Thus,usingsocialnetworksisadouble-edgedsword.Therefore,tantoperations(convolutions)aredifficulttocalculateinthe
ifthenewsreceivedfromsocialnetworksisreal,itwillbegraphdomain.Additionally,theprimaryassumptionofprevious
beneficial.Conversely,ifthisnewsisfake,itwillhavemanymachinelearninganddeeplearning-basedfakenewsdetection
harmfulconsequences,andtheextentofdamagewhenfakenewsalgorithmsisthatnewsitemsareindependent.Thisassumption
iswidelydisseminatedisincalculable.cannotapplytographdatabecausenodescanconnecttoother
 Fakenewsisasourceofabsolutelyinventiveinformationtonodesthroughvarioustypesofrelationships,suchascitations,
spreaddeceptivecontentorentirelymisrepresentactualnewsinteractions,andfriendships.GNN-basedfakenewsdetection
articles[1].Numerousexamplesoffakenewsexist.Huntetal.[2]methodshavebeendeveloped.Althoughsomestate-of-the-art
indicatedthatduringthe2016USpresidentialelection,theactiv-resultshavebeenachieved(seeTable1),nocompleteGNN-based
ityofClintonsupporterswasaffectedbythespreadoftraditionalfakenewsdetectionandpreventionsystemexistedwhenwe
centerandleft-leaningnewsfromtopinfluencers,whereastheconductedthisstudy.Fakingnewsonsocialnetworksisstilla
movementofTrumpsupporterswasinfluencedbythedynamicsmajorchallengethatneedstobesolved(thefirstjustification).
oftopfakenewsspreaders.Moreover,publicopinionmanipu-Varioussurveypapersoffakenewsdetectionhavebeenpub-
lationbasedonthespreadoffakenewsrelatedtotheBrexitlished,suchas[18–23].Webrieflysummarizerelatedworkas
voteintheUnitedKingdomwasreported.Mostrecently,thefollows:VitalyKlyuevetal.[20]presentedasurveyofdifferent
prevalenceoffakenewshasbeenwitnessedduringtheCOVID-fakenewsdetectionmethodsbasedonsemanticsusingnatural
19pandemic.Theseexamplesshowthatthespreadoffakenewslanguageprocessing(NLP)andtextminingtechniques.Addition-
onsocialnetworkshasasignificanteffectonmanyfields.Timelyally,theauthorsdiscussedautomaticcheckingandbotdetection
detectionandcontainmentoffakenewsbeforewidespreaddis-onsocialnetworks.Meanwhile,Oshikawaetal.[21]introduceda
seminationisanurgenttask.Therefore,manymethodshavebeensurveyforfakenewsdetection,focusingonlyonreviewingNLP-
implementedtodetectandpreventthespreadoffakenewsbasedapproaches.Collinsetal.[18]presentedvariousvariantsof
overthepastdecade,amongwhichthegraphneuralnetworkfakenewsandreviewedrecenttrendsinpreventingthespread
(GNN)-basedapproachisthemostrecent.offakenewsonsocialnetworks.Shuetal.[22]conducteda
 Basedonpreviousstudies’findingsregardingthebenefitofreviewonvarioustypesofdisinformation,factorinfluences,and
usingGNNsforfakenewsdetection,wesummarizesomemainapproachesthatdecreasetheeffects.Khanetal.[19]presented
justificationsforusingGNNsasfollows.Existingapproachesforfakenewsvariants,suchasmisinformation,rumors,clickbait,
fakenewsdetectionfocusalmostexclusivelyonfeaturesrelatedanddisinformation.Theyprovidedamoredetailedrepresentation
tothecontent,propagation,andsocialcontextseparatelyinofsomefakenewsvariantdetectionmethodswithoutlimiting
theirmodels.GNNspromisetobeapotentiallyunifyingframe-NLP-basedapproaches.Theyalsointroducedtypesofavailable
workforcombiningcontent,propagation,andsocialcontext-detectionmodels,suchasknowledge-based,fact-checking,and
basedapproaches[3].Fakenewsspreaderscanattackmachinehybridapproaches.Moreover,theauthorsintroducedgovern-
learning-basedmodelsbecausethesemodelsdependstronglymentalstrategiestopreventfakenewsanditsvariants.Mahmud
onnewstext.Makingdetectionmodelslessdependentontheetal.[23]presentedacomparativeanalysisbyimplementing
newstextisnecessarytoavoidthisissue.GNN-basedmodelsseveralcommonlyusedmethodsofmachinelearningandGNNs
canachievesimilarorhigherperformancethanmodernmethodsforfakenewsdetectiononsocialmediaandcomparingtheir
withouttextualinformation[4].GNN-basedapproachescanpro-performance.Nosurveypapershaveattemptedtoprovideacom-
videflexibilityindefiningtheinformationpropagationpatternprehensiveandthoroughoverviewoffakenewsdetectionusing
usingparameterizedrandomwalksanditerativeaggregators[5].themostcurrenttechnique,namely,theGNN-basedapproach
 Agraphneuralnetworkisanoveltechniquethatfocuseson
usingdeeplearningalgorithmsovergraphstructures[6].Be-(thesecondjustification).
foretheirapplicationinfakenewsdetectionsystems,GNNshadTheabovetwojustificationsmotivatedustoconductthis
beensuccessfullyappliedinmanymachinelearningandnaturalsurvey.Althoughsomesimilaritiesareunavoidable,oursurvey
languageprocessing-relatedtasks,suchasobjectdetection[7,8],isdifferentfromtheaforementionedworksinthatwefocuson
sentimentanalysis[9,10],andmachinetranslation[11,12].Thedescription,analysis,anddiscussionofthemodelsoffakenews
rapiddevelopmentofnumerousGNNshasbeenachievedbydetectionusingthemostrecentGNN-basedtechniques.Webe-
improvingconvolutionalneuralnetworks,recurrentneuralnet-lievethatthispapercanprovideanessentialandbasicreference
works,andautoencodersthroughdeeplearning[13].Therapidfornewresearchers,newcomers,andsystemspractitionersin
developmentofGNN-basedmethodsforfakenewsdetectionovercomingcurrentbarriersandformingfuturedirectionswhen
systemsonsocialnetworkscanbeattributedtotherapidgrowthimprovingtheperformanceoffakenewsdetectionsystemsusing
ofsocialnetworksintermsofthenumberofusers,theamountGNNs.Thispapermakesthefollowingfourmaincontributions.
                                      2

H.T. Phan, N.T. Nguyen and D. Hwang                                                                                                                                                                                            Applied Soft Computing 139 (2023) 110235
             Table1
             AdescriptionoftheimprovedperformanceofthetraditionalmethodscomparedusingGNN-basedmethods.
             Method   Ref  Improvedmethods              Dataset      Leastimproved
                                                               performance
             GCAN    [14]  DTC,SVM-TS,mGRU,RFC,     Twitter15,Accuracy:18.7%,
                          tCNN,CRNN,CSI,dEFEND      Twitter16  Accuracy:19.9%
             FANG    [5]  FeatureSVM,CSI               Twitter      AUC:6.07%
             SAFER    [15]  HAN,dEFEND,SAFE,CNN,    FakeNewsNet,F1:5.19%,
                          RoBERTa,MajsharingbaselineFakeHealthF1:5.00%
             Bi-GCN   [16]  DTC,SVM-RBF,SVM-TS,RvNNWeibo,      Accuracy:4.5%,
                          PPC_RNN+CNN               Twitter15,Accuracy:13.6%,
                                                    Twitter16  Accuracy:14.3%
             AA-HGNN  [17]  SVM,LIWC,text-CNN,Labelpropagation,PolitiFactAccuracy:2.82%
                          DeepWalk,LINE,GAT,GCN,HANBuzzFeed    Accuracy:9.34%
 •Weprovidethemostcomprehensivesurveyyetoffakeattentionfromresearchers,withdifferingdefinitionsfromvari-
   news,includingsimilarconcepts,characteristics,typesofre-ousviewopinions.In[24],theauthorsdefinedfakenewsas‘‘a
   latedfeatures,typesofapproaches,andbenchmarksdatasets.news article that is intentionally and verifiably false’’        .Alcottand
   WeredefinesimilarconceptsregardingfakenewsbasedonGentzkow[2]providedanarrowdefinitionoffakenewsas‘‘news
   theircharacteristics.Thissurveycanserveasapracticalarticles that are intentionally and verifiably false, and could mislead
   guideforelucidating,improving,andproposingdifferentreaders’’ .Inanotherdefinition,theauthorsconsideredfakenews
   fakenewsdetectionmethods.                 as‘‘fabricated information that mimics news media content in form
 •WeprovideabriefreviewofexistingtypesofGNNmod-butnotinorganizationalprocessorintent’’       [25].In[26],theauthors
   els.Wealsomakenecessarycomparisonsamongtypesofconsideredfakenewsinvariousforms,suchasfalse,misleading,
   modelsandsummarizethecorrespondingalgorithms.orinventivenews,includingseveralcharacteristicsandattributes
 •WeintroducethedetailsofGNNmodelsforfakenewsofthedisseminatedinformation.In[27],theauthorsprovided
   detectionsystems,suchaspipelinesofmodels,benchmarkabroaddefinitionoffakenewsas‘‘false news’’  andanarrow
   datasets,andopensourcecode.Thesedetailsprovideadefinitionoffakenewsas‘‘intentionally false news published by a
   backgroundandguideexperienceddevelopersinproposingnews outlet’’  .Similardefinitionshavebeenemployedinprevious
   differentGNNsforfakenewspreventionapplications.fakenewsdetectionmethods[3,4,28,29].
 •WeintroduceanddiscussopenproblemsforfakenewsCharacteristicsofFakenews:Althoughvariousdefinitions
   detectionandpreventionusingGNNmodels.Weprovideaexist,mostfakenewshasthefollowingcommoncharacteristics.
   thoroughanalysisofeachissueandproposefutureresearch
   directionsregardingmodeldepthandscalabilitytrade-offs.•Echochambereffect:Echochambers[30]canbebroadly
                                                definedasenvironmentsfocusingontheopinionsofusers
  Thissectionjustifiedtheproblemandhighlightedourmoti-whohavethesamepoliticalleaningorbeliefsaboutatopic.
vationsforconductingthissurvey.TheremainingsectionsoftheTheseopinionsarereinforcedbyrepeatedinteractionswith
paperareorderedasfollows.Section2introducesthebackgroundotheruserswithsimilartendenciesandattitudes.Social
andprovidesanoverviewoffakenews,fakenewsdetection,credibility[31]andfrequencyheuristic[31](i.e.,thetrendto
andGNNs.Section3presentsthesurveymethodologyusedtosearchforinformationthatconformstopreexistingreviews)
conductthereview.Generalinformationontheincludedpapersmaybethereasonfortheappearanceofechochamberson
isanalyzedinSection4.InSection5,theselectedpapersaresocialnetworks[24,32–34].Whennewsdoesnotcontain
categorizedandreviewedindetail.Subsequently,wediscusstheenoughinformation,Social credibility   canbeusedtojudge
comparisons,advantages,anddisadvantagesofthemethodsbyitstruthfulness.However,manypeoplestillperceiveitas
categoryinSection6.Next,thepossiblechallengesoffakenewscredibleanddisseminateit,leadingtopopularacceptance
andGNNsarebrieflyevaluatedinSection7.Finally,weidentifyofsuchnewsascredible.AFrequency heuristic    formswhen
severalopenissuesinthisareaanddiscusspotentialdirectionspeoplefrequentlyhearthenews,leadingtonaturalapproval
forfutureresearchinSection8.                    oftheinformation,evenifitisfakenews.
                                               •Intentiontodeceive[35]:Thischaracteristicisidentified
2.Background                                    basedonthehypothesisthat‘‘no  one  inadvertently  pro-
                                                duces inaccurate information in the style of news articles, and
2.1. Understanding fake news                    the fake news genre is created deliberately to deceive’’         [25].
                                                Deceptionispromptedbypolitical/ideologicalorfinancial
  Whatisfakenews?Newsisunderstoodasmeta-informationreasons[2,36–38].However,fakenewsmayalsoappearand
andcanincludethefollowing[24]:                  isspreadtoamuse,toentertain,or,asproposedin[39],‘‘to
 •Source:Publishersofnews,suchasauthors,websites,andprovoke’’ .
   socialnetworks.                             •Maliciousaccount:Currently,newsonsocialnetworks
 •Headline:Descriptionofthemaintopicofthenewswithacomesfrombothrealpeopleandunrealpeople.Although
   shorttexttoattractreaders’attention.fakenewsiscreatedandprimarilyspreadbyaccounts
 •Bodycontent:Detaileddescriptionofthenews,includingthatarenotrealpeople,severalrealpeoplestillspread
   highlightsandpublishercharacteristics.fakenews.Accountscreatedmainlytospreadfakenews
 •Image/Video:Partofthebodycontentthatprovidesavisualarecalledmaliciousaccounts[27].Maliciousaccountsare
   illustrationtosimplifythenewscontent.dividedintothreemaintypes:social bots, trolls, and cyborg
 •Links:Linkstoothernewssources.                users [24]. Social bots  aresocialnetworkaccountscontrolled
                                                bycomputeralgorithms.Asocialbotiscalledamalicious
  ‘‘Fakenews’’wasnamedwordoftheyearbytheMacquarieaccountwhenitisdesignedprimarilytospreadharmful
Dictionaryin2016[24].Fakenewshasreceivedconsiderableinformationandplaysalargeroleincreatingandspreading
                                           3

H.T. Phan, N.T. Nguyen and D. Hwang                                                                                                                                                                                            Applied Soft Computing 139 (2023) 110235
   fakenews[40].Thismaliciousaccountcanalsoautomati-•Fakefacts[50]areundefinedinformation(newsornon-
   callypostnewsandinteractwithothersocialnetworkusers.news)comprisingnonfactualstatementsfrommaliciousac-
   Trolls arerealpeoplewhodisruptonlinecommunitiestocountsthatcancausetheechochambereffect,withthe
   provokeanemotionalresponsefromsocialmediausers[24].intentiontomisleadthepublic.
   Trollsaimtomanipulateinformationtochangetheviewsof•Propaganda[48]isbiasedinformation(newsornon-news)
   others[40]bykindlingnegativeemotionsamongsocialnet-comprisingundefinedstatements(factualornonfactual)re-
   workusers.Consequently,usersdevelopstrongdoubtsandgardingmostlypoliticaleventsfrommaliciousaccountsand
   distrustthem[24];theywillfallintoastateofconfusion,thatcancausetheechochambereffect,withtheintention
   unabletodeterminewhatisrealandwhatisfake.Gradually,tomisleadthepublic.
   userswilldoubtthetruthandbegintobelieveliesandfalse•Sloppyjournalism[19]isunreliableandunverifiedinforma-
   information.Cyborg  users   aremaliciousaccountscreatedtion(newsornon-news)comprisingundefinedstatements
   byrealpeople;however,theymaintainactivitiesbyusingsharedbyjournaliststhatcancausetheechochambereffect,
   programs.Therefore,cyborgsarebetteratspreadingfalsewiththeintentiontomisleadthepublic.
   news[24].
 •Authenticity:Thischaracteristicaimstoidentifywhether2.2. Fake news detection
   newsisfactual[27].Factual statements    canbeproventrue
   orfalse.Subjectiveopinionsarenotconsideredfactualstate-2.2.1. What is fake news detection?
   ments.Onlyobjectiveopinionsareconsideredfactualstate-Unliketraditionalnewsmedia,fakenewsisdetectedusing
   ments.Factualstatementscanneverbeincorrect.Whenamainlycontent-basednewsfeatures;forsocialmedia,social
   statementispublished,itisnotafactualstatementifitcancontext-basedauxiliaryfeaturescanaidindetectingfakenews.
   bedisproved[41].Nonfactual statements    arestatementsthat
   wecanagreeordisagreewith.Inotherwords,thisnewsThus,in[24],theauthorspresentaformaldefinitionoffakenews
   issometimeswrong,sometimesrightorcompletelywrong.detectionbasedonthecontent-basedandcontext-basedfeatures
   Fakenewscontainsmostlynonfactualstatements.ofthenews.Giventhesocialinteractionsεamongnusersfor
 •Theinformationisnews:Thischaracteristic[27]reflectsnewsarticlea,theobjectiveoffakenewsdetectionistopredict
   whethertheinformationisnews.          whetheraisaninstanceoffakenews.Thisobjectiveisdefined
Basedonthecharacteristicsoffakenews,weprovideanewdef-byapredictionfunctionF :ε →{0,1}suchthat,
                                              {1,   ifa is a piece of  fake ne         w s,
initionoffakenewsasfollows.‘‘Fakenews’’isnewscontainingF(a)=
nonfactualstatementswithmaliciousaccountsthatcancausethe0,   other   w ise .                  (1)
echochambereffect,withtheintentiontomisleadthepublic.Herein,ShuandSlivadefinepredictionfunctionFasabinaryclas-
 ConceptsrelatedtoFakenews:Variousconceptsregardingsificationfunctionbecausefakenewsdetectioncomprisesdis-
fakenewsexist.Usingthecharacteristicsoffakenews,wecantortedinformationfrompublishersregardingactualnewstopics
redefinetheseconceptstodistinguishthemasfollows.
                                         (distortionbias).Accordingtomediabiastheory[51],adistortion
 •Falsenews[42,43]isnewscontainingnonfactualstatementsbiasisoftendefinedasabinaryclassification.
   frommaliciousaccountsthatcancausetheechochamberUsingtheabovedefinitionoffakenewsdetection,inthis
   effectwithundefinedintentions.paper,weconsiderfakenewsdetectionasamulticlassification
 •Disinformation[44]isnewsornon-newscontainingnon-task.GivenasetofnnewsN  = {  n1,n2,...,n n}andasetofm
   factualstatementsfrommaliciousaccountsthatcancauselabelsΨ,fakenewsdetectionidentifiesaclassificationfunctionF,
   theechochambereffect,withtheintentiontomisleadthesuchthatF : N  →   Ψ,tomapeachnewsn ∈  Nintothetrueclass
   public.                               withthereliablelabelinΨ.Correspondingtotheconceptsrelated
 •Cherry-picking[45]isnewsornon-newscontainingcom-tofakenews(seeSection2.1)areconceptsrelatedtofakenews
   monfactualstatementsfrommaliciousaccountsandcandetection,suchasrumordetectionandmisinformationdetection
   causetheechochambereffect,withtheintentiontomislead(classification).Theseconceptsaredefinedsimilarlytothefake
   thepublic.                            newsdetectiontask.
 •Rumor[46]isnewsornon-newscontainingfactualornon-
   factualstatementsfrommaliciousaccountsandcancause2.2.2. Fake news detection datasets
   theechochambereffectwithundefinedintentions.Inthissection,weintroducecommondatasetsthathavebeen
 •Fakeinformationisnewsornon-newsofnonfactualstate-recentlyusedforfakenewsdetection.Thesedatasetswerepre-
   mentsfrommaliciousaccountsthatcancausetheechoparedbycombiningtheEnglishdatasetspresentedinprevious
   chambereffect,withtheintentiontomisleadthepublic.papers[19,52,53]andenrichedbyaddingmissingdatasets.In
 •Manipulation[47]isnewsonmarketscontainingnonfactualcontrasttoothersurveysorreviewpapers,wecalculatedthe
   statementsfrommaliciousaccountsthatcancausetheechostatisticson35datasets,whereasD’Uliziaetal.[52],Sharma
   chambereffect,withtheintentiontomisleadthepublic.etal.[53],andKhanetal.[19]consideredonly27,23,and
 •Deceptivenews[2,24,27]isnewscontainingnonfactual10datasets,respectively.Therefore,welistdatasetsbydomain
   statementsfrommaliciousaccountsthatcancausetheechoname,typeofconcept,typeofcontent,andnumberofclasses.A
   chambereffect,withtheintentiontomisleadthepublic.briefcomparisonofthefakenewsdatasetsispresentedinTable2.
 •Satirenews[48]isnewscontainingfactualornonfactualBasedonthecontentpresentedinTable2,thesedatasetscan
   statementsfrommaliciousaccountsthatcancausetheechobefurtherdetailedasfollows:
   chambereffect,withtheintentiontoentertainthepublic.
 •Misinformation[33]isnewsornon-newscontainingnon-•ISOT1:BothfakenewsandrealnewsfromReuters;fake
   factualstatementsfrommaliciousaccountsthatcancausenewsfromwebsitesflaggedbyPolitiFactandWikipedia.
   theechochambereffectwithundefinedintentions.•Fakeddit:Englishmultimodalfakenewsdatasetincluding
 •Clickbait[49]isnewsornon-newscontainingfactualorimages,comments,andmetadatanews.
   nonfactualstatementsfrommaliciousaccountsthatcan
   causetheechochambereffect,withtheintentiontomislead
   thepublic.                             1https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/index.php
                                        4

H.T. Phan, N.T. Nguyen and D. Hwang                                                                                                                                                                                            Applied Soft Computing 139 (2023) 110235
Table2
AcomparisonamongFakenewsdatasets.
Nameofdataset              Domain           Typeofconcept        Typeofcontent       Numberofclasses
1-ISOT[54,55]              Politics,society,Fakenews,realnews      Text             2
                          business,sport,
                          crime,technology,
                          health
2-Fakeddit[56]              Society,politics       Fakenews           Text,image,videos      2,3,6
3-LIAR[57]                Politics           Fakenews           Text             6
4-FakeNewsNet[58]           Society,politics       Fakenews           Text,image         2
5-StanfordFakeNews[59]        Society           Fakenews,satire        Text,image,videos      2
6-FA-KES[60]              Politics           Fakenews           Text             2
7-BREAKING![61]            Society,politics       Fakenews,satire        Text,image         2,3
8-BuzzFeedNews[24]           Politics           Fakenews           Text             4
9-FEVER[62]               Society           Fakenews           Text             3
10-FakeCovid[63]            Health,society        Fakenews           Text             11
11-CredBank[64]             Society           Rumor             Text             2,5
12-Memetracker[65]           Society           Fakenews,realnews      Text             2
13-BuzzFace[66]             Politics,society       Fakenews           Text             4
14-FacebookHoax[67]          Science           Fakenews           Text             2
15-Higgs-Twitter[68]           Science           Fakenews           Text             2
16-TrustandBelieve[69]         Politics           Fakenews           Text             2
17-Yelp[70]               Technology         Fakenews           Text             2
18-PHEME[71]              Society,politics       Rumor             Text             2
19-Factchecking[72]           Politics,society       Fakenews           Text             5
20-EMERGENT[73]            Society,         Rumor             Text             3
                          technology
21-BenjaminPoliticalNews[74]      Politics           Fakenews           Text             3
22-BurfootSatireNews[75]        Politics,economy,Satire             Text             2
                          technology,society
23-MisInfoText[76]            Society           Fakenews           Text             5
24-Ottetal.’sdataset[77]        Tourism           Fakereviews          Text             2
25-FNC-1[78]              Politics,society,Fakenews           Text             4
                          technology
26-Fake_or_real_news[79]        Politics,society       Fakenews           Text             2
27-TSHP-17[80]             Politics           Fakenews           Text             2,6
28-QProp[81]              Politics           Fakenews           Text             2,4
29-NELA-GT-2018[82]           Politics           Fakenews           Text             2,3,5
30-TW_info[83]             Politics           Fakenews           Text             2
31-FCV-2018[84]             Society           Fakenews           Videos,text         2
32-VerificationCorpus[85]        Society           Fakenews           Videos,text,image      2
33-CNN/DailyMail[86]          Politics,society,Fakenews           Text             4
                          business,sport,
                          crime,technology,
                          health
34-Tametal.’sdataset[87]        Politics,Rumor             Text             5
                          technology,
                          science,crime,
                          fraudandscam,
                          fauxtography
35-FakeHealth[88]            Health            Fakenews           Text             2
  •LIAR2:Englishdatasetwith12,836shortstatementsregard-•FakeNewsNet5:Englishdatasetwith422newsarticlesre-
   ingpoliticscollectedfromonlinestreamingandtwosocialgardingsocietyandpoliticscollectedfromonlinestreaming
   networks–TwitterandFacebook–from2007to2016.andTwitter.
  •StanfordFakeNews:Fakenewsandsatirestories,including•FEVER:Englishdatasetwith185,445claimsregardingsoci-
   hyperbolicsupportorcondemnationofafigure,conspiracyetycollectedfromonlinestreaming.
   theories,racistthemes,anddiscreditingofreliablesources.•FakeCovid:Englishdatasetwith5182newsarticlesfor
  •FA-KES:LabeledfakenewsregardingtheSyrianconflict,COVID-19healthandsocietycrawledfrom92fact-checking
   suchascasualties,activities,places,andeventdates.websites,referringtoPoynterandSnopes.
  •BREAKING!:EnglishdatasetcreatedusingtheStanfordFake•CredBank6:Englishdatasetwith60milliontweetsabout
                             3.Thedata,includingover1000eventsregardingsocietycollectedfromTwitter
   NewsdatasetandBSdetectordatasetfromOctober2014toFebruary2015.
   newsregardingthe2016USpresidentialelection,were•Memetracker:Englishdatasetwith90milliondocuments,
   collectedfromwebpages.                          112millionquotes,and22millionvariousphrasesregarding
  •BuzzFeedNews4:Englishdatasetwith2283newsarticlessocietycollectedfrom165millionsites.
   regardingpoliticscollectedfromFacebookfrom2016to•BuzzFace:Englishdatasetwith2263newsarticlesand1.6
   2017.                                           millioncommentsregardingsocietyandpoliticscollected
                                                   fromFacebookfromJuly2016toDecember2016.This
                                                   datasetwasextendedinSeptember2016.
 2https://www.cs.ucsb.edu/william/data/liardataset.zip
 3https://www.kaggle.com/mrisdal/fake-news5https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/
 4https://github.com/BuzzFeedNews/2016-10-facebookfact-check/tree/UEMMHS
master/data                                      6http://compsocial.github.io/CREDBANK-data/
                                              5

H.T. Phan, N.T. Nguyen and D. Hwang                                                                                                                                                                                            Applied Soft Computing 139 (2023) 110235
                        Fig.1.Acomparisonamongdatasetsintermsoffourcriteria.
 •FacebookHoax:Englishdatasetwith15,500hoaxesregard-(54.29%),respectively,whereasonlyonedatasetcontainedecon-
   ingsciencecollectedfromFacebookfromJuly2016toDe-omy,fraud/scam,andfauxtographynews(2.86%).Thesefindings
   cember2016.Additionally,thisdatasetidentifiespostswithcanbeexplainedbythefactthatfakenewsismorepertinent
   over2.3millionlikes.                  andwidespreadinpoliticalandsocietaldomainsthaninother
 •Higgs-Twitter:Englishdatasetwith985,590tweetsposteddomains[89].
   by527,496usersregardingthescienceofthenewHiggsThird,regardingthetypeoffakenewsconcepts,27ofthe35
   bosondetectioncollectedfromTwitter.datasetscontainedthefakenewsconcept(77.14%),followedby
 •TrustandBelieve:Englishdatasetwithinformationfromrumors(11.43%),satire(8.57%),hoaxes,andrealnews(5.71%),
   50,000politicianusersonTwitter.Allinformationwasla-andfinally,fakereviews(2.86%).Therefore,datasetscontaining
   beledmanuallyorusingavailablelearningmethods.thefakenewsconceptaregenerallyusedforfakenewsdetection
 •Yelp:Englishdatasetwith18,912technologyfakereviewsapplicationsbecausefakenewscontainsfalseinformationspread
   collectedfromonlinestreaming.bynewsoutletsforpoliticalorfinancialgains[46].
 •PHEME:EnglishandGermandatasetwith4842tweetsandFinally,regardingthetypeofapplications,themostcom-
   330rumorsconversationsregardingsocietyandpoliticscol-monapplicationobjectiveofthe35datasetswasfakedetection
   lectedfromTwitter.                    (71.43%),followedbyfact-checking(11.43%),veracityclassifica-
Becauseofthelimitednumberofmanuscriptpages,wedonottion,andrumordetection(8.57%)becausefakenewsdetection
                                         applicationscanbeusedtosolvepracticalproblems.Additionally,
describefurtherdatasets.Theremainingdatasetsarepresentedfakenewsdetectionisthemostgeneralapplication,covering
intheAppendixunderDescriptionofDatasets.theentireprocessofclassifyingfalseinformationastrueor
 Basedontheaboveanalysis,wecomparethecriteriaoffakefalse.Thus,fakeinformationdatasetsarethemostrelevantfor
newsdatasetsinFig.1,followedbyadiscussionofobservationscollection[52].
andthemainreasonfortheseobservations.
 First,regardingthetypeofnewscontent,29ofthe35datasets2.2.3. Features of fake news detection
containedtextdata(82.86%);threeofthe35datasetscomprisedThedetailsofextractingandrepresentingusefulcategoriesof
text,image,andvideodata(8.57%),namely,Fakeddit,StanfordfeaturesfromnewscontentandcontextaresummarizedinFig.2.
FakeNews,andVerificationCorpus;twoofthe35datasetscon-Basedonthenewsattributesanddiscriminativecharacter-
tainedtextandimagedata(5.71%),namely,FakeNewsNetandisticsoffakenews,wecanextractdifferentfeaturestobuild
Breaking;andonlyonedatasetcontainedtextandvideodatafakenewsdetectionmodels.Currently,fakenewsdetectionrelies
(2.86%).Nodatasetincludedseparateimagesorvideosbecausemainlyonnewsandcontextinformation.Inthissurvey,we
previousfakenewsdetectionmethodsusedmainlyNLP-basedcategorizefactorsthatcanaidfakenewsdetectionintoseven
techniquesthatwerehighlydependentontextdata.Additionally,categoriesoffeatures:network-,sentiment-,linguistic-,visual-,
labeledimageorvideodataarescarcebecauseannotatingthempost-,user-,andlatent-basedfeatures.
islaborintensiveandcostly.                 Linguistic-basedfeatures:Theseareusedtocaptureinforma-
 Second,regardingthenewsdomain,20and19ofthe35tionregardingtheattributesofthewritingstyleofthenews,
datasetsfocusedonsocietynews(57.14%)andpoliticalnewssuchaswords,phrases,sentences,andparagraphs.Fakenews
                                       6

H.T. Phan, N.T. Nguyen and D. Hwang                                                                                                                                                                                            Applied Soft Computing 139 (2023) 110235
                        Fig.2.Categoriesoffeaturesforfakenewsdetectionmethods.
iscreatedtomisleadorentertainthepublicforfinancialorfakenews,specificallysocialbotsandcyborgusers.User-based
politicalgains.Therefore,basedontheintentionoffakenews,featuresarepropertiesrelatedtouseraccountsthatcreateor
wecaneasilyextractfeaturesrelatedtowritingstylesthatoftenspreadfakenews.Thesefeaturesareclassifiedintotwolevels,
appearonlyinfakenews,suchasusingprovocativewordstonamely,thegrouplevelandtheindividuallevel[27].Theindi-
stimulatethereader’sattentionandsettingsensationalheadlines.vidualfocusesonexploitingfakeorrealfactorsregardingeach
Tobestcapturelinguistic-basedfeatures,wedividethemintofivespecificuser,suchasregistrationage,numberoffollowers,and
commontypes:lexical,syntactic,semantic,domain-specific,andnumberofopinionspostedbyusers[102,104].Meanwhile,the
informality.Lexicalfeaturesrefertowording,suchasthemostgrouplevelfocusesonfactorsregardingthegroupofusers,such
salientcharacters(n-grams)[90,91],frequencyofnegationwords,astheratioofusers,theratiooffollowers,andtheratioof
doubtwords,abbreviationwords,vulgarwords[92],andthenov-followees[95,105].
eltyofwords[93].SyntacticfeaturescapturepropertiesrelatedPost-basedfeatures:Thiscategoryoffeaturesisidentifiedand
tothesentencelevel,suchasthenumberofpunctuations[94],extractedbasedonthemaliciousaccountsandnewscharac-
numberoffunctionwords(nouns,verbs,andadjectives)[93],teristicsoffakenews.Post-basedfeaturesareusedtocapture
frequencyofPOStags[95],andsentencecomplexity[96,97].propertiesrelatedtousers’responsesoropinionsregardingthe
Semanticfeaturescapturepropertiesrelatedtolatentcontent,newsshared.Thesefeaturesareclassifiedintothreecategories:
suchasthenumberoflatenttopics[98]andcontextualclues[99].group,post,andtemporal[27].Thepostlevelfocusesonexploit-
Thesefeaturesareextractedwithstate-of-the-artNLPtechniques,ingfactorsregardingeachpost[28],suchasotherusers’opinions
suchasdistributionsemantics(embeddingtechniques)andtopicregardingthispost(support,deny),maintopic,anddegreeofre-
modeling(LDAtechnique)[100].Domain-specificfeaturescap-liability.Thegrouplevelfocusesonfactorsregardingallopinions
turepropertiesrelatedtodomaintypesinthenews,suchasrelatedtothispost[106],suchastheratioofsupportingopinions,
quotedwords,frequencyofgraphs,andexternallinks[101].ratioofcontradictingopinions,andreliabilitydegree[95,105].
Informalityfeaturescapturepropertiesrelatedtowritingerrors,Thetemporallevelnotesfactorssuchasthechangingnumberof
suchasthenumberoftypos,swearwords,netspeak,andassentpostsandfollowersovertimeandthesensoryratio[105].
words[27].                                 Network-basedfeatures:Network-basedfeaturesareemployed
 Sentiment-basedfeatures:Thiscategoryoffeaturescapturestoextractinformationregardingtheattributesofthemedia
propertiesregardinghumanemotionsorfeelingsappearinginthewherethenewsappearsandisspread[107].Thiscategoryof
news[102,103].Thesefeaturesareidentifiedandextractedbasedfeaturesisidentifiedandextractedbasedonthecharacteristics
ontheintentionsandauthenticitycharacteristicsoffakenews.offakenews,suchastheechochamber,maliciousaccount,
Theyareclassifiedintotwogroups:visualpolarityandtextpo-andintention.Herein,theextractablefeaturesarepropagation
larity.Thecriticalfactorsrelatedtovisualpolarityarethenumber
ofpositive/negativeimages/videos,numberofanxious/angry/sadconstructions,diffusionmethods,andsomefactorsrelatedto
images/videos,andnumberofexclamationmarks[27].Thesethedisseminationofnews,forexample,densityandclustering
factorscaptureinformationsimilartothetextpolarity.coefficient.Therefore,manynetworkpatternscanform,such
 User-basedfeatures:Thiscategoryoffeaturesisidentifiedasoccurrence,stance,friendship,anddiffusion[24].Thestance
andextractedbasedonthemaliciousaccountcharacteristicsofnetwork[106]isagraphwithnodes,edges,nodesshowingallthe
                                       7

H.T. Phan, N.T. Nguyen and D. Hwang                                                                                                                                                                                            Applied Soft Computing 139 (2023) 110235
textrelatedtothenews,andedgesbetweennodesshowsimilar2.2.4. Fake news detection techniques
weightsofstancesintexts.Theco-occurrencenetwork[28]isFig.3showsanoverviewoffakenewsdetectiontechniques.
agraphwithnodesshowingusersandedgesindicatinguseren-Previousrelatedpapers[21,27,42,46,53,79,107]demonstratedthat
gagement,suchasthenumberofuseropinionsonthesamenews.fakenewsdetectiontechniquesareoftenclassifiedintofour
Thefriendshipnetwork[105]isagraphwithnodesshowinguserscategoriesofapproaches:content-basedapproaches,including
whohaveopinionsrelatedtothesamenewsandedgesshowingknowledge-basedandstyle-basedapproaches,context-based
thefollowers/followeesconstructionsoftheseusers.Thediffusionapproaches,propagation-basedapproaches,multilabellearning-
network[105]isanextendedversionofthefriendshipnetworkbasedapproaches,andhybrid-basedfakenewsdetectionap-
withnodesthatindicateuserswhohaveopinionsonthesameproaches.LetΨ   abeoneofthecorrespondingoutputclassesof
news;theedgesshowtheinformationdiffusionpathwaysamongthefakenewsdetectiontask.Forexample,Ψ   a ∈{real,false}or
theseusers.                                Ψ   a ∈{nonrumor,unverifiedrumor,falserumor,truerumor}or
 Data-drivenfeatures:ThiscategoryoffeaturesisidentifiedΨ   a ∈{true,false}.
andextractedbasedonthedatacharacteristicsoffakenews,Knowledge-baseddetection:Givennewsitemawithasetof
suchasthedatadomain,dataconcept,datacontent,andapplica-knowledge denoted by tripleK  =(S,P ,O) [127], where
tion.Thedatadomainexploitsdomain-specificandcross-domainS ={  s1,s2,...,sk}isasetofsubjectsextractedfromnewsitem
knowledgeinthenewstoidentifyfakenewsfromvariousdo-a,P ={  p1,p2,...,p k}isasetofpredicatesextractedfromnews
                                           itema,O  ={  o1,o2,...,o k}isasetofobjectsextractedfrom
mains[108].Thedataconceptfocusesondeterminingwhethernewsitema.Thus,k ai =(si,p i,o i)∈  K ,1≤   i≤   n,iscalleda
conceptdrift[109]existsinthenews.Thedatacontentfocusesknowledge.Forexample:wehaveanewsas‘‘John Smith is a
onconsideringpropertiesrelatedtolatentcontentinthenews,famous doctor at a central hospital"      ;fromthisstatement,wehave
suchasthenumberoflatenttopics[98]andcontextualclues[99].k ai =(JohnSmith      ,Profession     ,Doctor ).Assumethatwehaveasetof
Thesefeaturesareextractedbasedonstate-of-the-artNLPtech-trueknowledgeKt  =(St ,Pt ,Ot),wherekt  al =(st l,pt  l,ot  l)∈  Kt,
niques,suchasdistributionsemantics(embeddingtechniques)1≤   l≤   m.LetG  Kbeatrueknowledgegraphincludingasetof
andtopicmodeling(LDAtechnique)[100].trueknowledge,wherenodesrepresentasetof(St ,Ot)∈   Kt
 Visual-basedfeatures:Fewfakenewsdetectionmethodshaveandedgesrepresentasetof(Pt)∈  Kt,theaimofaknowledge-
beenappliedtovisualnews[24].ThiscategoryoffeaturesisbasedfakenewsdetectionmethodistodefineafunctionFto
identifiedandextractedbasedontheauthenticity,news,andcomparek ai =(si,p i,o i)∈  K withkt  al =(st l,pt  l,ot  l)∈  Kt,such
                                                  G K−→  Ψ   a
intendedcharacteristicsoffakenews.Visual-basedfeaturesarethat:F : k aii.FunctionFisusedtoassignalabelΨ   ai   ∈
usedtocapturepropertiesrelatedtonewscontainingimages,[0,1]toeachtriple(si,p i,o i)bycomparingitwithalltriples
videos,orlinks[27,100].Thefeaturesinthiscategoryareclas-(st l,pt  l,ot  l)ongraphG  K,wherelabels0and1indicatefake
sifiedintotwogroups:visualandstatistical.Thevisuallevelandreal,respectively.FunctionFcanbedefinedasF(k ai,G  K)=
reflectsfactorsregardingeachvideoorimage,suchasclarity,Pr(edge p       i is a link from      ˆsi to  ˆo i on G     K),wherePr istheproba-
coherence,similaritydistribution,diversity,andclusteringscore.bility;ˆsiandˆo iarethematchednodestosiando ionG  K,
Thestatisticallevelcalculatesfactorsregardingallvisualcontent,respectively.ˆsiandˆo iareidentifiedasˆsi =argminst l|J(si,st l)|<ξ
suchastheratioofimagesandtheratioofvideos.andˆo i  = argminot l|J(o i,ot  l)| <  ξ,respectively,whereξisa
 Latentfeatures:Acriticalconceptthatweneedtobeawareofcertainthreshold;J(si,st l)isafunctiontocalculatethedistance
hereinislatentfeaturesthatarenotdirectlyobservable,includingbetweensiandst ianditisthesimilarforJ(o i,ot  l).Forexample,
latenttextualfeaturesandlatentvisualfeatures.Latentfeatureswhen|J(si,st l)|  = 0or|J(si,st l)|  <  ξ,wecanregardsias
areneededtoextractandrepresentlatentsemanticsfromthethesameasst i.Thetechniquesinthiscategoryareproposed
originaldatamoreeffectively.Thiscategoryoffeaturesisiden-basedontheauthenticityandnewscharacteristicsoffakenews.
tifiedandextractedbasedonthecharacteristicsoffakenews,Theobjectiveofknowledge-basedtechniquesistoemployex-
                                           ternalsourcestofact-checknewsstatements.Thefact-checking
suchastheechochamber,authenticity,andnewsinformation.stepaimstoidentifythetruthofastatementcorresponding
Latenttextualfeaturesareoftenextractedbyusingthenewstoaspecificcontext[72].Itcanbeimplementedautomatically
textrepresentationmodelstocreatenewstextvectors.Text(computational-oriented[128])ormanually(expert-based[101,
representationmodelscanbedividedintothreegroups:contex-129,130],crowd-sourced[67,131]).
tualizedtextrepresentations,suchasBERT[110],ELMo[111],Style-baseddetection:Givenanewsitemawithasetoff as
Non-contextualizedtextrepresentation,suchasWord2Vec[112],stylefeatures,wheref asisasetoffeaturesregardingthenews
FastText[113],GloVe[114],andknowledgegraph-basedrep-content.Style-basedfakenewsdetectionisdefinedasbinary
resentation,suchasKoloskietal.method[115],RotatE[116],classificationtoidentifywhethernewsitemaisfakeorreal,
QuatE[117],ComplEx[118].ContextualizedtextrepresentationswhichmeansthatwehavetofindamappingfunctionFsuch
arewordvectorsthatcancapturerichercontextandsemanticthatF : f as  →   Ψ   a.Thetechniquesinthiscategoryareproposed
information.Knowledgegraph-basedrepresentationscanenrichbasedontheintentionandnewscharacteristicsoffakenews.
variouscontextualandnoncontextualrepresentationsbyaddingTheobjectiveofstyle-basedtechniquesistocapturethedistinct
humanknowledgerepresentationsviaconnectionsbetweentwowritingstyleoffakenews.Fakenewsemploysdistinctstylesto
entitieswiththeirrelationshipbasedonknowledgegraphs.Newsattracttheattentionofmanypeopleandstandoutfromordinary
textrepresentationscanbenotonlyusedasinputsfortradi-news.Thecapturingstepofthewritingstyleswasbuiltauto-
tionalmachinelearningmodels[119]butalsointegratedintomatically.However,twotechniquesmustbeobservedascriteria:
deeplearningmodels,suchasneuralnetworks[115],recurrentstylerepresentationtechniques[132–134]andstyleclassification
networks[120],andtransformers[110,121,122],andGNNs-basedtechniques[28,91,135].
models[123–125]forfakenewsdetection.Latentvisualfea-Context-baseddetection:Givennewsitemawithasetof
                                           f accontextfeatures,wheref acincludesnewstext,newssource,
turesareoftenextractedfromvisualnews,suchasimagesandnewspublisher,andnewsinteraction.Context-basedfakenews
videos.Latentvisualfeaturesareextractedbyusingneuralnet-detectionisdefinedasthetaskofbinaryclassificationtoidentify
works[126]tocreatealatentvisualrepresentationcontaininganwhethernewsitemaisfakeorreal,whichmeansthatwehaveto
imagepixeltensorormatrix.                  findamappingfunctionFsuchthatF : f ac  →   Ψ   a.Thetechniques
                                         8

H.T. Phan, N.T. Nguyen and D. Hwang                                                                                                                                                                                            Applied Soft Computing 139 (2023) 110235
                              Fig.3.Categoriesoffakenewsdetection.
inthiscategoryareproposedbasedonthemaliciousaccountclassifiedintofourapproaches:(i)usingstyle-basedrepresen-
andnewscharacteristicsoffakenews.Theobjectiveofsource-tation[17,115,146,147];(ii)usingstyle-basedclassification[15,
basedtechniquesistocapturethecredibilityofsourcesthat29,148–151];(iii)usingnewscascades[140,152];and(iv)using
appear,publish,andspreadthenews[27].Credibilityreferstoself-definedpropagationgraphs[4,16,125,153].
people’semotionalresponsetothequalityandbelievabilityofHybrid-baseddetection:Thismethodisastate-of-the-artap-
news.Thetechniquesinthiscategoryareoftenclassifiedintoproachforfakenewsdetectionthatsimultaneouslycombines
twoapproaches:(i)assessingthereliabilityofsourceswheretwopreviousapproaches,suchascontent-context[154,155],
thenewsappearedandisspreadbasedonnewsauthorsandpropagation-content[147,156],andcontext-propagation[4,14].
publishers[136,137]and(ii)assessingthereliabilityofsourcesThesehybridmethodsarecurrentlyofinterestbecausetheycan
wherethenewsappearedandisspreadbasedonsocialmediacapturemoremeaningfulinformationrelatedtofakenews.Thus,
users[105,138,139].                       theycanimprovetheperformanceoffakenewsdetectionmodels.
 Propagation-baseddetection:GivennewsitemawithasetAcriticalissuethatneedstobediscussedisfakenewsearly
off appropagationpatternsfeaturesfornews.Propagation-baseddetection.Earlydetectionoffakenewsprovidesanearlyalertof
fakenewsdetectionisdefinedasbinaryclassificationtoidentifyfakenewsbyextractingonlythelimitedsocialcontextwitha
whethernewsitemaisfakeorreal,whichmeansthatwesuitabletimedelaycomparedwiththeappearanceoftheoriginal
havetodevelopamappingfunctionFsuchthatF : f ap  →   Ψ   a.newsitem.Knowledge-basedmethodsareslightlyunsuitable
Thetechniquesinthiscategoryareproposedbasedontheechoforfakenewsearlydetectionbecausethesemethodsdepend
chambereffectandnewscharacteristicsoffakenews.Theob-stronglyonknowledgegraphs;meanwhile,newlydisseminated
jectiveofpropagation-basedtechniquesistocaptureandextractnewsoftengeneratesnewinformationandcontainsknowledge
informationregardingthespreadoffakenews.Thatis,themeth-thathasnotappearedinknowledgegraphs.Style-basedmeth-
odsinthiscategoryaimtodetectfakenewsbasedonhowodscanbeusedforfakenewsearlydetectionbecausethey
peopleshareit.Thesetechniquesareoftengroupedintotwodependmainlyonthenewscontentthatallowsustodetect
smallcategories:(i)usingnewscascades[140,141]and(ii)usingfakenewsimmediatelyafternewsappearsandhasnotbeen
self-definedpropagationgraphs[142–145].   spread.However,style-basedfakenewsearlydetectionmethods
 Multilabellearning-baseddetection:Letχ   ∈    R dbethed-areonlysuitableforabriefperiodbecausetheyrelyheavily
dimensioninputfeaturematrix;hence,newsitema = [  a1,...,onthewritingstyle,whichcreatorsandspreaderscanchange.
a d]  ∈   χ;andletΓ   =  {  real ,fake  }lbethelabelmatrix,suchPropagation-basedmethodsareunsuitableforfakenewsearly
thatΨ   =  [ Ψ1,...,Ψ  l]  ∈   Γ,wherelisthenumberofclassdetectionbecausenewsthatisnotyetbeendisseminatedoften
labels.Givenatrainingset{(a,Ψ)},thetaskofmultilabellearningˆΨ   =containsverylittleinformationaboutitsspread.Tothebest
detectionistolearnafunctionF  : χ   →    Γ topredictofourknowledge,context-basedmethodsaremostsuitablefor
F(a).Multilabellearning-baseddetectionisalearningmethodfakenewsearlydetectionbecausetheydependmainlyonthe
whereeachnewsiteminthetrainingsetisassociatedwithanewssurroundings,suchasnewssources,newspublishers,and
setoflabels.Thetechniquesinthiscategoryareproposedbasednewsinteractions.Thisfeatureallowsustodetectfakenews
ontheechochambereffectandnewscharacteristicsoffakeimmediatelyafternewsappearsandhasnotbeenspreadbyusing
news.Theobjectiveofmultilabellearning-basedtechniquesistowebsitespamdetection[157],distrustlinkpruning[158],and
captureandextractinformationregardingthenewscontentanduserbehavioranalysis[159]methods.Ingeneral,earlydetection
thenewslatenttext.Thetechniquesinthiscategoryareoftenoffakenewsisonlysuitableforabriefperiodbecausehuman
                                        9

H.T. Phan, N.T. Nguyen and D. Hwang                                                                                                                                                                                            Applied Soft Computing 139 (2023) 110235
Table3                                         V i∩  V j =  ∅.Meanwhile,edgesmustgenerallysatisfythecondi-
Descriptionsofnotations.                       tionsfollowingthenodetypes.Then,wehavee ij =(v i,τ,v  j)∈  E
Notations          Descriptions                →    e ij =(v i,τh,v j)∈  E,wherev i ∈  V t,v j ∈  V kandt ̸= k.
|.|             Thelengthofaset                  Multiplexgraphs:Here,graphsaredividedintoasetofk
G              Agraph                          layers,whereeachnodebelongstoonelayer,andeachlayerhasa
V              Thesetofnodesinagraph           uniquerelationcalledtheintralayeredgetype.Anotheredgetype
v              Anodeinagraph                   istheinterlayeredgetype.Theinterlayerconnectsthesamenode
E              Thesetofedgesinagraph           acrossthelayers.ThatmeansG  = {  G  i,i ∈ {1,2,...,k}},G  i =
e ij              Anedgebetweentwonodesv i,v jinagraph
A              Thegraphadjacencymatrix         {V i,E i},withV i ={ v1,v2,...,vn},E i =   E intrai    ∪ E interi ,E intrai    ={  e lj =
D              ThedegreematrixofA.D  ii = ∑    nj=1A ij(v l,v j),v l,v j ∈  V i},E interi    ={  e lj =(v l,v j),v l ∈  V i,v j ∈  V h,1≤   h ≤
n              Thenumberofnodes                k,h ̸= i}.
m              Thenumberofedges
r              Thesetofrelationsofedges2.3.2. What are graph neural networks?
d              ThedimensionofnodefeaturevectorGNNsarecreatedusingdeeplearningmodelsovergraph
c              Thedimensionofedgefeaturevector
x evi,vj ∈  R c          Thefeaturevectorofedgee ijstructuredata,whichmeansdeeplearningmodelsdealwith
x nv ∈  R d           Thefeaturevectorofnodev  Euclideanspacedata;incontrast,GNNs[6,161–163]dealwith
X  e ∈  R m ×  c          Theedgefeaturematrixofagraphnon-Euclideandomains.AssumethatwehaveagraphG ={  V ,E}
X  ∈  R n×  d           ThenodefeaturematrixofagraphwithadjacencymatrixAandnodefeaturematrix(oredgefeature
X(t)∈  R n×  d          Thenodefeaturematrixatthetimesteptmatrix)X(orX  e).GivenAandXasinputs,themainobjectiveofa
                                               GNNistofindtheoutput,i.e.,nodeembeddingsandnodeclassifi-
                                               cation,afterthek-thlayeris:H(k)=   F(A,H(k−1);θ(k)),whereFisa
intelligenceislimitless.Whenanearlydetectionmethodoffakepropagationfunction;θistheparameteroffunctionF,andwhen
newsisapplied,itwillnotbelonguntilhumanscreateank =   1,thenH(0)=   X.Thepropagationfunctionhasanumberof
effectivewaytocombatit.Thisissueisstillamajorchallengeforms.Letσ(·)beanon-linearactivationfunction,e.g.,ReLU;W (k)
forthefakenewsdetectionfield.                  istheweightmatrixforlayerˆA =   D −0.5A ⊤ D −0.5withk;ˆAisthenormalizedadjacency
                                               matrixandcalculatedas            D,isthediagonal
2.3. Understanding graph neural networks       degreematrixofA ⊤,thatiscalculatedasD  ii = ∑j A ⊤ij;A ⊤  =   A +  I
                                               withIistheidentitymatrix.Asimpleformofthepropagation
  Inthissection,weprovidethebackgroundanddefinitionofafunctionisoftenused:F(A,H(k))=  σ(AH (k−1)W (k)).Inaddition,
GNN.Thetechniques,challenges,andtypesofGNNsarediscussedthepropagationfunctioncanbeimprovedtobesuitableforGNN
inthefollowingsection.Beforepresentingthecontentofthistasksasfollows:
section,weintroducethenotationsusedinthispaperinTable3.Forthenodeclassificationtask,functionFoftentakesthe
                                               followingform[164]:
2.3.1. What is a graph?                        F(A,H(k))=  σ(ˆAH (k−1)W (k))               (3)
  Beforewediscussdeeplearningmodelsongraphstructures,Forthenodeembeddingstask,functionFoftentakesthe
weprovideamoreformaldescriptionofagraphstructure.For-followingform[165]:
mally,asimplegraphispresentedasG ={  V ,E},where
V  ={ v1,v2,...,vn}isthesetofnodes,andE ={  e11,e12,...,e nn }F(A,H(k))=  σ((Q φ(H(k−1)
isthesetofedgeswheree ij =(v i,v j)∈  E,1≤   i,j≤   n.Inwhich,e       M   e)Q  ⊤ ⊙  ˆA)H(k−1)W (k))     (4)
v iandv jaretwoadjacentnodes.TheadjacencymatrixAisawhereQ isatransformerrepresentingwhetheredgeeiscon-
n ×   nmatrixwith                              nectedtothegivennodeandQ  ⊤  =   T +  I;M   ethelearnablematrix
   {1,   ife ij ∈  E,                          fortheedges;φisthediagonalizationoperator;⊙istheelement-
A ij =                                         wiseproduct;H(k−1)e  isthehiddenfeaturematrixofedgesinthe
    0,   ife ij /∈  E.                   (2)   (k −1)-thlayer,whereH0e  =   X  e(X  eistheedgefeaturematrix).
  WecancreateimprovedgraphswithmoreinformationfromTheQ φ(H(k−1)e       M   e)Q  ⊤istonormalizethefeaturematrixofedges.
simplegraphs,suchasattributedgraphs[6],multi-relationalTheQ φ(H(k−1)e       M   e)Q  ⊤ ⊙ ˆAistofusetheadjacencymatrixbyadding
graphs[160].                                   theinformationfromedges.
  Attributedgraphsaretheextendedversionofsimplegraphs.MorechoicesofthepropagationfunctioninGNNsaredetail
TheyareobtainedbyaddingthenodeattributesXortheedgepresentedinRefs.[13,165].Earlyneuralnetworkswereapplied
attributesX  e,whereX  ∈  R n×  disanodefeaturematrixwithtoacyclicgraphsbySperdutietal.[166]in1997.In2005,Gori
x nv ∈  R dindicatingthefeaturevectorofanodev;X  e ∈  R m ×  cisanetal.[167]introducedthenotionofGNNs,whichwerefurtherde-
edgefeaturematrixwithx evi,vj ∈  R cindicatingthefeaturevectortailedbyScarsellietal.[168]in2009andbyGallicchioetal.[169]
ofanedgee ij.                                  in2010.AccordingtoWuetal.[6],GNNscanbedividedinto
  Spatial–Temporalgraphsarespecialcasesofattributedgraphs,fourmaintaxonomies:conventionalGNNs,graphconvolutional
wherethenodeattributesautomaticallychangeovertime.There-networks,graphautoencoders,andspatial–temporalgraphneu-
fore,letX(t)beafeaturematrixofthenoderepresentationsralnetworks.Inthenextsubsections,weintroducethecategories
attthtimestep,aspatial–temporalgraphisdefinedasG(t)=ofGNNs.
                                                 Conventionalgraphneuralnetworks(GNNs∗)whicharean
{V ,E,X(t)},whereX(t)∈  R n×  d.               extensionofrecurrentneuralnetworks(RNNs),werefirstin-
  Multi-relationalgraphsareanotherextensionversionofsim-troducedbyScarsellietal.[168]byconsideringaninforma-
plegraphsthatincludeedgeswithdifferenttypesofrelationstiondiffusionmechanism,wherestatesofnodesareupdated
τ.Inthesecases,wehavee ij =(v i,v j)∈  E →    e ij =(v i,τ,v  j)∈  E.andinformationisexchangeduntilastableequilibriumisob-
EachedgehasonerelationadjacencymatrixA τ.Theentiregraphtained[167,168].IntheseGNNs,thefunctionFisalsodefined
canbecreatedanadjacencytensorA  ∈  R n×  r×  n.Themulti-asEq.(3).However,thefeaturematrixofthek-thlayerH(k)is
relationalgraphscanbedividedintotwosubtypes:heteroge-updatedusingdifferentequationasfollows:
neousandmultiplexgraphs.
  Heterogeneousgraphs:Here,nodescanbedividedintodif-H(k)vj  =   ∑F(x nvj,x e(vi,vj),x nvi,H(k−1)vi )           (5)
ferenttypes.ThatmeansV  =   V1∪  V2∪  ...∪  V k,wherefori̸= j,vi∈ N(vj)
                                             10

H.T. Phan, N.T. Nguyen and D. Hwang                                                                                                                                                                                            Applied Soft Computing 139 (2023) 110235
whereN(v j)isthesetofneighbornodesofnodev j,FisaAttention-basedgraphneuralnetworks(AGNNs)[178]re-
parametricfunction,H(k)vjisthefeaturevectorofnodev jforthemoveallintermediatefullyconnectedlayersandreplacethe
k-thlayer,andH(0)vjisarandomvector.       propagationlayerswithanattentionmechanismthatmaintains
 Graphconvolutionalnetworks(GCNs)werefirstintroducedthestructureofthegraph[179].Theattentionmechanismal-
byKipfandWelling[164].Theyarecapableofrepresentinglowslearningadynamicandadaptivelocalsummaryofthe
graphsandshowoutstandingperformanceinvarioustasks.Inneighborhoodstoobtainmoreaccuratepredictions[180].The
theseGNNs,afterthegraphisconstructed,thefunctionFisalsopropagationfunctionformoftheAGNNisshowninEq.(3).
definedasEq.(3).However,therecursivepropagationstepofaHowever,theAGNNincludesgraphattentionlayers.Ineachlayer,
GCNatthek-thconvolutionlayerisgivenby:ashared,learnablelineartransformationM  ∈  R th×  d h,wherehis
H(1)=  σ(ˆAH (0)W (1)+  b(1))               (6)thenumberofthet-thhiddenlayer,d histhedimensionalofthe
                                          t-thhiddenlayer,isusedfortheinputfeaturesofeverynodeas
Hence,                                    follows:
H(2)=  σ(ˆAH (1)W (2)+  b(2))               (7)H(t)=  σ(M (t)H(t−1))                  (12)
Thatmeans:                                wheretherow-vectorofnodev idefinedasfollows:
H(k)=  σ(ˆAH (k−1)W (k)+  b(k))              (8)H(t)vi =    ∑M (t−1)ij      H(t−1)j                (13)
whereH(0)=   X.σ(·)isanactivationfunction.W (k)∈  R m ×  d,vj∈ N(vi)∪{ i}
k ={1,2,3,...}isatransitionmatrixcreatedforthek-thlayer.where
b(1)andb(2)arethebiasesoftwolayers.       M (t−1)
 Graphautoencoders(GAEs)aredeepneuralarchitecturesij     =  ϕ([β(t−1)cos(H(t−1)i     ,H(t−1)j )]vj∈ N(vi)∪{ i})     (14)
withtwocomponents:(i)theencoder,whichconvertsnodeswhereβ(t−1)∈  Risanattention-guidedparameterofpropagation
onthegraphintoavectorspaceoflatentfeatures,and(ii)thelayers.Notethatthevalueofβofpropagationlayersischanged
decoder,whichdecodestheinformationonthegraphfromtheoverhiddenstates.ϕ(·)istheactivationfunctionofpropagation
latentfeaturevectors.ThefirstversionofGAEswasintroducedbylayer.
KipfandWelling[170,171].IntheseGNNs,theformoffunction
FisredefinedasthefollowingEquation:3.Surveymethodology
F(˜A,H(k))=  σ(˜AH (k−1)W (k))               (9)
where˜A =  ϕ(ZZ  ⊤)isthereconstructedadjacencymatrixandϕInthisstudy,weconductedasystematicreviewoffakenews
istheactivationfunctionofthedecodercomposition.ZisthedetectionarticlesusingGNNmethods,includingthreeprimary
outputoftheencodercomposition.IntheseGAEs,theGCNssteps:‘‘literaturesearch,’’‘‘selectionofeligiblepapers,’’and‘‘an-
areusedintheencodersteptocreatetheembeddingmatrix;alyzinganddiscussing’’[181].Theresearchmethodologyisillus-
therefore,ZiscalculatedbasedonEq.(3).Thus,Z =   F(ˆA,H(k))tratedinFig.4:
withF(·)correspondstothecaseofGCNs.Z ⊤isthetransposeTheliteraturesearchisusedtoselectpeer-reviewedand
matrixofZ.                                English-languagescientificpaperscontainingthefollowingkey-
 Spatial–temporalgraphneuralnetworks(STGNNs)invari-words:‘‘GNN’’OR‘‘graphneuralnetwork’’OR‘‘GCN’’OR‘‘graph
ousreal-worldtasksaredynamicasbothgraphstructuresandconvolutionalnetwork’’OR‘‘GAE’’OR‘‘graphautoencoder’’OR
graphinputs.Torepresentthesetypesofdata,aspatial–temporal‘‘AGNN’’OR‘‘attention-basedgraphneuralnetwork’’combined
graphisconstructedasintroducedinSectionwith‘‘fakenews’’OR‘‘falsenews’’OR’’rumour’’OR’’rumor’’OR2.3.1.However,
tocapturethedynamicityofthesegraphs,STGNNshavebeen‘‘hoax’’OR‘‘clickbait’’OR‘‘satire’’OR‘‘misinformation’’combined
proposedformodelingtheinputscontainingnodeswithdy-with‘‘detection’’.ThesekeywordswereextractedfromGoogle
namicandinterdependency.STGNNscanbedividedintotwoScholar,Scopus,andDBLPfromJanuary2019totheendofQ2
approaches:RNN-basedandCNN-basedmethods.2021.
 FortheRNN-basedapproach,tocapturethespatial–temporalTheselectionofeligiblepapersisusedtoexcludethenon-
relation,thehiddenstatesofSTGNNsarepassedtoarecurrentexplicitpapersonfakenewsdetectionusingGNNs.Toselect
unitbasedongraphconvolutions[172–174].Thepropagationtheexplicitpapers,wespecifyasetofexclusion/inclusioncri-
functionformofSTGNNsisalsoshowninEq.(3).However,theteria.Theinclusioncriteriawereasfollows:writteninEnglish,
valueofthek-thlayeriscalculatedasfollows:publishedafter2019,peer-reviewed,andretrievedfull-text.The
H(t)=  σ(WX      n(t)+  UH (t−1)+  b)             (10)exclusioncriteriawereasfollows:papersofreviews,surveys,and
                                          comparisonsoronlypresentedmathematicalmodels.
whereX  n(t)isthenodefeaturematrixattimestept.AfterusingAnalysisanddiscussionpapersareusedtocomparethesur-
graphconvolutions,Eq.(10)isrecalculatedasfollows:veyedliteratureandcapturethemainchallengesandinteresting
H(t)=  σ(GCN (X  n(t),ˆA;W)+  GCN (H(t−1),ˆA;U)+  b)   (11)openissuesthataimtoprovidevariousuniquefutureorienta-
                                          tionsforfakenewsdetection.
whereGCN isoneofGCNsmodel.U  ∈  R n×  nistheeigenvectorBytheabovestrategy,afinaltotalof27papers(5papersin
matrixrankedbyeigenvalueswithU  ⊤ U  =   I.2019,16papersin2020,and6papersforthefirst6monthsof
 FortheCNN-basedapproach,RNN-basedapproachesrecur-2021)areselectedforacomprehensivecomparisonandanaly-
sivelyhandlespatial–temporalgraphs.Thus,theymustiteratesis.Theseselectedpapersareclassifiedintofourgroupsbased
thepropagationprocessandthereforetheyhavelimitationsre-onGNNtaxonomies(seeSection2.3.2),includingconventional
gardingthepropagationtimeandgradientexplosionorvanish-GNN-based,GCN-based,AGNN-based,andGAE-basedmethods.
ingproblems[175–177].CNN-basedapproachescansolvetheseInthenextstep,eligiblepapersareanalyzedviathecriteriaof
problemsbyexploitingparallelcomputingtoachievestablegra-themethod’sname,criticalidea,lossfunction,advantage,and
dientsandlowmemory.                       disadvantage.
                                        11

H.T. Phan, N.T. Nguyen and D. Hwang                                                                                                                                                                                            Applied Soft Computing 139 (2023) 110235
                             Fig.4.Flowdiagramofresearchmethodology.
4.QuantitativeanalysisofeligiblepapersoutbreakoffakenewsrelatedtoCOVID-19andthechallengesof
                                          thisproblem.
 PreviousfakenewsdetectionapproacheshavemainlyusedWithregardtothetypeofnewsconceptsemployed(types
machinelearning[74,92,182–184]anddeeplearning[95,120,ofobjectives),14ofthe27surveyedpapersarerelatedtofake
185–189]forclassifyingnewsasfakeorreal,rumorornotanewsdetection(51.85%),followedbyrumorsandspamdetection
rumor,andspamornotspam.Varioussurveysandreviewpapers(29.63%,7.41%),whereasothertypesofdetectionconstituteonly
regardingfakenewsdetectionusingmachinelearninganddeep3.7%.Alikelyreasonfortheseresultsisthecreationandspreadof
learninghavebeenpublished.Inthispaper,wediscussindetailfakenewscorrespondtoactiveeconomicandpoliticalinterests.
themostcurrentGNN-basedfakenewsdetectionapproaches.Thatis,iffakenewsisnotdetectedandpreventedinatimely
UsingtheresearchmethodologyinSection3,afinaltotalof27manner,peoplewillsuffermanydeleteriouseffects.Additionally,
paperspublishedafter2019usingGNNsforfakenewsdetec-asanalyzedabove,anequallyimportantreasonisthatdatasets
tionwereselectedforamoredetailedreviewinthefollowingusedforfakenewsdetectionarenowricherandmorefully
subsections.Table4presentscomparisonsamongpreviousstud-labeledthanotherdatasets(seeSection2.2.2).
iesintermsofmodelname,referralcode(Table5),authors,WithregardtoGNN-basedtechniques,theauthorspredomi-
yearofpublication,typeofGNN,datasets,performance,andnantly(74.07%)usedGCNsforfakenewsdetectionmodels,fol-
approach-basedfakenewsdetection.lowedbyGNN-basedmethods(14.81%),GAE,andAGNN(3.7%).
 UsingtherelationshipsamongtheinformationinTable4,weThischoiceisattributabletothesuitabilityofGCNsforgraph
comparequantitativelysurveyedmethodsintermsoffourdistri-representationsinadditiontohavingachievedstate-of-the-art
butioncriteriaofGNN-basedfakenewsdetectionapproaches,asperformanceinawiderangeoftasksandapplications[13].
showninFig.5.                               Finally,one-thirdofthepropagation-basedandcontent-based
 Thenumberofsurveyedpapersfrom2019to2021(theendapproaches(33.33%)werepublished,followedbyhybrid-based
ofQ2)regardingfakenewsdetectionusingGNNsshowsthatthis(22.22%)andcontext-based(11.11%)approaches.Thisresultis
problemisattractingincreasingattentionfromsystempractition-
ers(increasing40.74%from2019to2020).Althoughin2021,onlyattributabletopropagation-basedandcontext-basedapproaches
22.22%ofarticlesonfakenewsdetectionfocusedonusingGNNs,usingmainlynewsinformationonnetworkstructures,users,and
Q2hasnotyetended,andwebelievethatthelasttwoquartersoflinguisticssimultaneously.Thisinformationismostconsequen-
theyearwillproducemorearticlesinthisfield,consideringthetialforfakenewsdetection.
                                        12

H.T. Phan, N.T. Nguyen and D. Hwang                                                                                                                                                                                            Applied Soft Computing 139 (2023) 110235
Table4
ComparisonofsurveyedmethodsusingGNNsforfakenewsdetection.
Method’sname      Authors  PYandTG    Dataset             Performance           Approach-based
1-Montietal.       [3]    2019,GCN    Tweets              ROCAUC:92.7%          Propagation
2-!GAS          [123]   2019,GCN    Spamdataset     F1:82.17%            Context
3-MGCN          [190]   2019,GCN    Liar               Acc:49.2%            Content
4-!ChangLietal.     [191]   2019,GCN    10,385newsarticles       Acc:67.03–88.89%        Context
5-Benamiraetal.2     [192]   2019,AGNN  Horneetal.[74]         Acc:70.45–84.25%Content
                           2019,GCN                            Acc:72.04–84.94%
6-Marionetal.1      [153]   2020,GCN    FakeNewsNet           Acc:73.3%            Propagation
7-YiHanetal.       [4]    2020,GNN∗    FakeNewsNet:                                  Propagation,context
                                        Politifact             Acc:79.2–80.3%
                                        GossipCop              Acc:82.5–83.3%
8-FakeNews        [193]   2020,GNN∗    Covid-19tweets         ROC:95%            Content
9-GCAN3         [14]    2020,GCN    Twitter15[140],            Acc:87.67%            Context,propagation
                                        Twitter16[140]         Acc:90.84%
10-Nguyenetal.      [155]   2020,GCN    Tweets              Task1:MCC:36.1–41.9%Content
                                                               Task2:MCC:−8.1–1.51%
11-Pehlivanetal.4     [194]   2020,GCN    Covid-19Tweets         GCN:T-MCC:2%Content
                                                               DGCNN:T-MCC:2.3%
                                                               M-FCN:T-MCC:3.5%
12-*Bi-GCN        [16]    2020,GCN    Weibo[99],               Acc:96.8%             Propagation
                                        Twitter15[140],        Acc:88.6%
                                        Twitter16[140]         Acc:88.0%
13-VGCN-ItalianBERT5   [195]   2020,GCN    1600imageswithmetadataF1:84.37%            Content
14-*GCNSI         [196]   2020,GCN    Karate[197],Dolphin[198],Improvethe            Propagation
                                        Powergrid[199],Jazz[200],bestmethod
                                        Ego-Facebook           byabout15%
15-SAFER6         [15]    2020,GNN∗    FakeNewsNet,            F1above92.97%         Context
                                        FakeHealth             F1above58.34%
16-!GCNwithMRF     [201]   2020,GCN    Twitter[202,203]         Acc:79.2–83.9%         Propagation
17-*Linetal.7       [124]   2020,GAE     Weibo[99],            Acc:93.4–94.4%        Propagation
                                        Twitter15[140]         Acc:84–85.6%
                                        Twitter16[140]         Acc:85.2–88.1%
18-*Malhotraetal.     [147]   2020,GCN    Twitter15[99]Acc:86.6%                     Propagation,content
                                        Twitter16[140]         Acc:86.5%
19-!FauxWard       [149]   2020,GCN    CommentsonTwitterAcc:71.09%                   Content
                                        CommentsonReddit       Acc:75.36%
20-KZWANG8       [156]   2020,GCN    Weibo[99],                Acc:95.0%             Propagation
                                        Twitter15[140],        Acc:91.1%
                                        Twitter16[140]         Acc:90.7%
21-FANG9        [5]    2020,GNN∗    FakeNewsNet,PHEME       AUC:75.18%           Context,content
22-*GraphSAGE      [125]   2021,GCN    Twitter[140]            Acc:69.0–77.0%        Propagation
                                        PHEME                  Acc:82.6–84.2%
23-Bert-GCNBert-VGCN  [150]   2021,GCN    Covid-19and5Gtweets     MCC:33.12–47.95%Content
                                                               MCC:39.10–49.75%
24-*Lotfi          [204]   2021,GCN    PHEME                     F1:80%(rumor)       Content,propagation
                                                               F1:79%(non-rumor)
25-*SAGNN        [151]   2021,GCN    Twitter15[140],           Acc:79.2–85.7%        Content
                                        Twitter16[140]         Acc:72.6-86.9%
26-AA-HGNN       [17]    2021,AGNN    Fact-checking,Acc:61.55%                       Content,context
                                        BuzzFeedNews           Acc:73.51%
27-*EGCN         [154]   2021,GCN    PHEME              Acc:63.8–84.1%         Propagation
PYandTG:PublicationyearandTypeofGNNs.MCC:Matthewscorrelationcoefficient.T-MCC:MCCfortestdataset.M-FCN:MALSTM-FCNmodel.
                    Table5
                    Codesource.
                     Refer         Codesource
                     1          https://github.com/MarionMeyers/fake_news_detection_propagation
                     2          https://github.com/bdvllrs/misinformation-detection-tensor-embeddings
                     3          https://github.com/l852888/GCAN
                     4          https://github.com/titu1994/MLSTM-FCN
                     5          https://github.com/dbmdz/berts#italian-bert
                     6          https://github.com/shaanchandra/SAFER
                     7          https://github.com/lhbrichard/rumor-detection
                     8          https://github.com/shanmon110/rumordetection
                     9          https://github.com/nguyenvanhoang7398/FANG
5.Literaturesurvey                                  methodsintoconventionalGNN-based,GCN-based,AGNN-based,
                                                    andGAE-basedmethods,asshowninTable6.
  Inthissection,wesurveypapersusinggraphneuralnet-∗)arepioneering
worksforfakenewsdetection.BasedonGNNtaxonomies(see•ConventionalGNN-basedmethods(GNN
Section2.3.2),wecategorizedGNN-basedfakenewsdetectionGNN-basedfakenewsdetectionmethods.Thesemethods
                                                  13

H.T. Phan, N.T. Nguyen and D. Hwang                                                                                                                                                                                            Applied Soft Computing 139 (2023) 110235
                 Fig.5.AcomparisonoffourdistributioncriteriaofGNN-basedfakenewsdetectionapproaches.
Table6                                     5.1. Detection approach based on GNNs                              ∗
GNN-baseddetectionmethodscategorization.
Category       Publication                  GNNs∗representthefirstversionofGNNsandimprovedthe
ConventionalGNN   [4,5,15,192,193]         performanceoffakenewsdetectionofmachinelearninganddeep
GCN         [3,14,16,123,125,147,149–151,153–156,194,201,204]learningmethodsthatusenon-Euclideandata.
AGNN         [17,192]                                                      ∗usingnon-
GAE          [124]                          Hanetal.[4]exploitedthecapabilityofGNNs
                                           Euclideandatatodetectthedifferencebetweennewspropaga-
                                           tionmethodsonsocialnetworks.Theythenclassifiedthenews
   applyasimilarsetofrecurrentparameterstoallnodesinaintotwolabelsoffakeandrealnewsbytrainingtwoinstances∗.Inthefirstcase,GNNs∗weretrainedoncomplete
                                           ofGNNs
   graphtocreatenoderepresentationswithbetterandhigherdata.ThesecondcaseinvolvedtrainingGNNs∗usingpartialdata.
   levels.                                 Inthesecondcase,unlikeconventionalGNNs,twotechniques
 •GCN-basedmethods(GCN)oftenusetheconvolutionalop-–gradientepisodicmemoryandelasticweightconsolidation–
   erationtocreatenoderepresentationsofagraph.UnlikewereusedtobuildGNNs∗withcontinuallearningaimedatthe
   theconventionalGNN-basedapproach,GCN-basedmethodsearlydetectionoffakepropagationpatterns.Thismethodcan
   allowintegratingmultipleconvolutionallayerstoimproveobtainsuperiorperformancewithoutconsideringanytextinfor-
   thequalityofnoderepresentations.mationcomparedwithstate-of-the-artmodels.Inparticular,time
 •AGNN-basedmethodsareconstructedmainlybyfeedingtheandcostaresavedasthedatasetgrowswhentrainingthenew
   attentionmechanismintographs.Thus,AGNNsareusedtodatabecausetheentiredatasetisnotretrained.However,one
   effectivelycaptureandaggregatesignificantneighborstomajorlimitationisthatthestrongforgettingoccurrenceisnot
   representnodesinthegraph.               solvedbyextractingmorefeatures,includingthe‘‘universal’’fea-
 •GAE-basedmethodsareunsupervisedlearningapproachestures.Hamidetal.[193]introducedamethodtodetectmalicious
   toencodenodesonagraphintoalatentvectoranddecodeuserswhospreadmisinformationbyanalyzingtweetsrelatedto
   theencodedinformationtoreconstructthegraphdatatoconspiracytheoriesbetweenCOVID-19and5Gnetworks.This
   createnoderepresentationsbyintegratinglatentinforma-methodincludestwosubstrategies:(i)content-basedfakenews
   tion.                                   detectionand(ii)context-basedfakenewsdetection.Thesecond
                                           strategyisimplementedbasedonGNNs∗totraintheGNNs∗
 Mostapproachesproposedinthesurveyedpapersfordetect-representationandtoclassify5Gnetworkingcommentsinto
ingfalseinformationareusedtosolveaclassificationproblemthreecategories:nonconspiracy,conspiracy,andotherconspir-
taskthatinvolvesassociatinglabelssuchasrumorornonrumoracies.TheobtainedperformanceintermsofaverageROCisquite
andtrueorfalsewithaparticularpieceoftext.InusingGNNsgood(0.95%)becauseitcapturesmostlyinformationrelatedto
forfakenewsdetection,researchershaveemployedmainlycon-thetextualandstructuralaspectsofnews.However,neithertex-
                                           tualnorstructuralinformationwasusedsimultaneously.Nguyen
ventionalGNNsandGCNstoachievestate-of-the-artresults.Onetal.[5]proposedamodelnamedFANGforfakenewsdetection
theotherhand,someresearchershaveappliedotherapproaches,basedonthenewscontextbyconsideringthefollowingsteps:
suchasGAEandAGNN,topredicttheirconforminglabels.(i)extractingfeaturesregardingthenews,suchasthesource,
                                        14

H.T. Phan, N.T. Nguyen and D. Hwang                                                                                                                                                                                            Applied Soft Computing 139 (2023) 110235
users,andtheirinteractions,andpostingatimeline;(ii)con-usersandotherusersforrumordetection.Unliketheconvolution
structingtwosubhomogeneousgraphs,namely,newssourceandlayeroftheconventionalGCN,theSAGNNdoesnotcontainthe
user;and(iii)usinganunsupervisedmodelovertwosubgraphsweightmatrixW.Moreover,theidentificationoftheadjacency
separatelytomodelneighborrelations.Moreover,theauthorsmatrixwasdifferentfromthatofconventionalGCNs.Thus,the
usedpretraineddetectionnetworkstodetectnewscontentastwolayersintheSAGNNaredefinedasH(1)=  σ(H(0)E),where
extensioninformation.FANGcancaptureanewscontextwithH(1)iscalledtheembeddinglayerandEisthewordembed-
higherfidelitythanrecentgraphicalandnongraphicalmodels.dingmatrix.Hence,H(2)=  σ(˜AH (1)),whereH(2)iscalledthe
Inparticular,FANGstillachievesrobustnessevenwithlimitedaggregationlayer;and˜A =   I +  uB  +  vC,whereu,varelearn-
trainingdata.However,featuressuchasusersandtheirinter-ableparametersofSAGNN.MatrixBiscalculatedasfollows:
actionsareextractedbeforebeingfedintotheFANG.Therefore,ifv i is the  parent of        v j then B      ij =1other   w ise0;whereasmatrix
someerrorsregardingtextualencodingandemotiondetectionCisdefinedasifv i is  a child of      v j then C      ij =1other   w ise0.
canoccur,andtheyareprovidedtoFANG.AnotherlimitationisKeetal.[156]constructedaheterogeneousgraph,namely,
therapidobsolescenceofcontextualdatasetsbecausewecannotKZWANG,forrumordetectionbycapturingthelocalandglobal
retrievehyperlinksandothertracesatthequerytime,astheyrelationshipsonWeibobetweensources,reposts,andusers.This
mightnolongerbeavailable.                   methodcomprisesthreemainstepsasfollows:(i)wordembed-
 UnlikeothercontemporaryworkonGNN∗,thefakenewsdingsconverttextcontentofnewsintovectorsusingamultihead
detectiontaskisintroducedabove.Chandraetal.[15]presentedattentionmechanism,
amethodcalledSAFERwiththreedistinctfeatures.(i)TheyT =   MultiHead  (Q ,K ,V)=   Concat (head 1,...,head      h)W    o  (15)
constructedaGNN∗modelwiththesameheterogeneousinput
graphfortwotypesofedgesandnodes.(ii)Theydeterminedwherehead      i =   attention (QW       Qi ,KW      Ki ,VW      Vi)with
contextfeaturesbyexploitingtheimpactofonlinesocialcom-Q  ∈  R n q×  d, K  ∈  R n k×  d, V  ∈  R n v×  daresentencesofquery,key,
munitieswithoutusinguserprofiles.(iii)Theyonlyusedtheandvalue;n q,n k,n varethenumberofwordsineachsen-
networkinformationofonlineuserstoevaluatethesecommu-tence;andattention (Q ,K ,V)=   Softmax (QK   ⊤√)V;(ii)propaga-
nities’roles,buttheirresultswerestillbetterthanthoseofd k
previousapproaches.TheauthorsproposedarelationalGNN∗andtionandinteractionrepresentationsarelearnedviaGCNs;and
ahyperbolicGNN∗tomodeluserandcommunityrelations.The(iii)graphconstructionbuildsamodelofpotentialinteractions
relationalGNNobtainedbetterresultsthanconventionalGNNs.amongusers:P =   H(k)=  σ(ˆAH (k−1)W (k−1)).Thedifferencebe-
However,theresultsofthehyperbolicGNN∗werecomparabletweenthismodelandconventionalGCNsisthatKZWANGis
onlytotheotherGNN∗.Therefore,modelingusers/communitiesacombinationofthenewstextrepresentationusingamulti-
fortrulyhierarchicalsocialnetworkdatasetsisachallengethatheadattentionmechanismandpropagationrepresentationus-
needstobeaddressedinthefuture.              ingGCNs.Thus,theoutputsoftheGCNlayerandthemul-
                                            tiheadattentionlayeraretheinputsofrumorclassification:
5.2. Detection approach based on GCNs       R =   Softmax (TP  +  b),whereTisthetextrepresentationmatrix.
                                            Pisthepropagationrepresentationmatrix.Ristheoutputofthe
 TheGCN-basedapproachisacategoryofmethodsthatarewholemodel.
usedmostlyforfakenewsdetectionandrelyonGNNs.GCNsLotfietal.[204]introducedamodelthatincludestwoGCNs:
areanextensionofGNNsthatderivethegraphstructureand(i)aGCNoftweets,suchassourceandreplyasT  =   H(k)=  σ
integratenodeinformationfromneighborhoodsbasedonacon-(ˆA T H(k−1)W (k−1));(ii)GCNofusers,suchasinteractionamong
volutionalfunction.GCNscanrepresentgraphsandachievestate-usersasRe  =   H(k)=  σ(ˆA Re H(k−1)W (k−1))whereA Tistheadja-
of-the-artperformanceonvarioustasks,includingfakenewscencymatrixoftheGCNoftweetsanddeterminedasif(tw eet
detection.                                  i replies to t    w eet j)or(i  =    j)then A       ijT  = 1other   w ise0.Mean-
 Luetal.[14]presentedanovelmethodforfakenewsdetectionwhile,A ReistheadjacencymatrixoftheGCNofusersanddefined
oftweetscalledGCAN,whichincludesfivemainstepsasfollows:asfollows:if(user i sent m t      w eets to user i in con        versation )or
(i)extractquantifiedfeaturesrelatedtousers;(ii)convertwords(i=   j)then A       ijRe  =1other   w ise0.AndH(0)=   Xisdeterminedas:
innewstweetsintovectors;(iii)representawarepropagationif(there is high frequency            w ords j in t    w eet i)or(the propagation
methodsoftweetsamongusers;(iv)capturethecorrelationtime is inter     val bet  w een the reply t       w eet i and the source t          w eet)
betweentweetcontextanduserinteractionsandbetweentweetthen X       ij =1other   w ise0.Unlikeothermodels,theauthorscon-
contextanduserpropagation;and(v)classifytweetsasfakestructedtwoindependentGCNsandthenconcatenatedtheminto
orrealnewsbycombiningalllearnedrepresentations.GCANonefullyconnectedlayerforfakenewsdetectionasSoftmax ((T ⊕
exhibitsoutstandingperformancewithreasonableexplainabil-Re)W   +  b),where⊕istheconcatenationfunction.
ity.ThemaincontributionofthisstudyistheintegrationofVuetal.[125]presentedanovelmethodcalledGraphSAGE
dualcoattentionmechanismswithGCNs.Thefirstmechanismforrumordetectionbasedonpropagationdetection.Incontrast
simultaneouslycapturestherelationsbetweenthetweetcontexttootherpropagation-basedapproaches,thismethodproposesa
anduserinteractions.ThesecondmechanismsimultaneouslygraphpropagationembeddingmethodbasedonaGCNtoconvert
capturestherelationsbetweenthetweetcontextanduserprop-thenewspropagationprocedureandtheirfeaturesintovector
agation.Thismethodcanbeconsideredanenrichingversionofspacebyaggregatingthenodefeaturevectorsandfeaturevectors
GCNs.TheformofGCANwasimprovedfromGCNsasfollows:oftheirlocalneighborsintoacombinationvector.Thus,the
H  s =   tanh(W    sS +(W    g G)F ⊤),H  g =   tanh(W    g G +(W    sS)F),whereSdifferencebetweentheGraphSAGEmodelandthetraditional
representstheembeddingsoftherelationsbetweenthetweetGCNmodelsconcernstheaggregatorfunctions,whicharedivided
contextanduserinteractions;Grepresentstheembeddingsofintothefollowingaggregators:(i)Convolutionalaggregator:
therelationsbetweentweetcontextanduserpropagation;W    s
andW    grepresentmatricesoflearnableparameters;FandF ⊤areh kvj =  σ(W    k h k−1vj   + ∑i h k−1vi
thetransformationmatrixanditstranspose,respectively.|N(v j)|+1),∀v i ∈  N(v j)        (16)
 BasedontheinherentaggregationmechanismoftheGNN,(ii)LSTMaggregator:
Zhangetal.[151]proposedasimplifiedGNNmethodcalled
SAGNNtocalculatethedegreeofinteractionbetweenTwitter△  lstmk    =   LSTM ({h k−1vi  ,∀v i ∈  N(v j)})           (17)
                                          15

H.T. Phan, N.T. Nguyen and D. Hwang                                                                                                                                                                                            Applied Soft Computing 139 (2023) 110235
(iii)Poolingaggregator:                    whereA ij =1if  i,j ha ve  the same job        −   title ,other   w ise A    ij =0.
△  pool                                    Then,themulti-depthinformationisaggregatedtocreatefi-m∑
 k    =   max (σ  pool(W    pool  h k−1vi   +  b pool)),∀v i ∈  N(v j)    (18)
(iv)Linearaggregator:                      nalrepresentationP j =α iz iusinganattentionmechanismas
     ∑                                                    i=1
△  linear∑iw  ih k−1vi                     α i =        exp(u i)∑    mu i =   tanh(W    iz i+  b i).
 k     =                                        l=1exp(u l),where
        iw  i   ,∀v i ∈  N(v j)            (19)Nguyenetal.[155]introducedtwomethods,textual-based
wherew  iistheweightvectorofneighborh vi.GraphSAGEeffi-andgraph-based,forfalsenewsdetectionregardingCOVID-19
cientlyintegratesfeaturessuchascontent,social,temporal,andand5Gconspiracy.Forthefirstmethod,theauthorsdetected
propagationstructures.ThesefeaturescanaggregatesignificantfakenewsusingacombinationofapretrainedBERTmodelanda
informationtotrainanalgorithmfordeterminingwhethernewsmultilayerperceptronmodeltocaptureboththetextualfeatures
arerumorsornot.However,thismethodrequiresdataontheandmetadataoftweets.Forthesecondmethod,theauthorused
entirepropagationprocedureofthepostednews.Insomecases,aGCNwithnineextractedfeaturesateachnode,suchaspage
ifthepostednewsdoesnotobtainresponseopinionswhenitisrank,hub,andauthority,forcontent-basedfakenewsdetection.
spread,theaccuracyoftheGraphSAGEmodelcanbereduced.Afterimplementingthetwomethods,theauthorsprovedthatthe
 Bianetal.[16]proposedaBi-GCNmodelwithtwopropagationperformanceofthefirstapproachisbetterthanthatofthesec-
operations,top-down(TD-GCN)andbottom-up(BU-GCN),tode-ond.Thus,metadataplayasignificantroleinfakenewsdetection.
tecttwoessentialcharacteristicsofrumors,dispersion,andprop-Therefore,improvingtheefficiencyoffakenewsdetectionby
agation.Bi-GCNwasconstructedasfollows:(i)High-levelnodeextractingmetadatafeaturesforGCNshouldbeconsideredinthe
representationsasH(k)TD  =  σ(ˆA TD  H(k−1)W (k)TD)andH(k)BU    =   σ(ˆA BUfuture.RegardingCOVID-19and5Gconspiracies,Pehlivan[194]
H(k−1)W (k)BU).(ii)Rootfeatureenhancementasfollows˜H(k)TD    =introducedstructure-basedfakenewsdetectiontoevaluatethe
concat (H(k)                               performanceofexistingmodels.Unlikeothermethods,theauthor
     TD ,(H(k−1)TD )root)and˜H(k)BU  =  concat (H(k)BU ,(H(k−1)BU )root)whereusedonlythetemporalfeaturesofnetworkswithoutconsidering
concat isaconcatenatefunction;root indicatestherootnode;textualfeatures.Twostate-of-the-artmodelswereselectedto
(iii)NoderepresentationsarefedintothepoolingaggregatorasevaluatetheGCNandDGCN[205].Additionally,theauthorsused
S TD  =   mean (˜H(2)TD)andS BU  =   mean (˜H(2)BU),andthenconcatenatedtheirtemporalfeaturestotestthemultivariatelongshort-term
themintoonefullyconnectedlayerforfakenewsdetectionmemoryfullyconvolutionalnetworkmethod[206].Nodefeature
asˆy =   Softmax (concat (S TD ,S BU)).ThismodelcancapturebothmatricesoftheGCNandDGCNarecreatedbasedonthefollowing
thepropagationofrumorpatternsusingtheTD-GCNandthevalues:degreecentrality,closenesscentrality,betweennesscen-
dispersionofrumorstructuresusingtheBU-GCN.Additionally,trality,loadcentrality,harmoniccentrality,#cliques,clustering
hiddeninformationofthenewsisextractedthroughlayersofcoefficient,squareclusteringcoefficient,andaverageneighbor
GCNtoincreasetheinfluenceofrumorroots.However,TD-GCNdegree.Thenodefeaturematrixofthemultivariatelongshort-
andBU-GCNwerestillconstructedindependently.termmemoryfullyconvolutionalnetworkiscreatedwiththe
 Baietal.[154]constructedagraphcalledanSRgraph,whereaverageclusteringcoefficient,#graphcliques,#connectedcom-
thenodefeaturematrixXisdeterminedbywordvectors;andponents,localefficiency,#isolates,andnormalizedtimedistance
theadjacentmatrixAisdefinedasfollows:if  tw eet i   repliestothesourcetweet.Lietal.[191]introducedapropagation-based
to t w eet j ,   then A       ij  = 1other   w ise0.UsingtheSR-graph,themethodfordeterminingpoliticalperspectivebyfocusingonthe
authorsproposedanEGCNmodelforrumordetectionasPG   =socialcontextualinformationofnews.Inthisstudy,theGCNis
H(k)=  σ(ˆAH (k−1)W (k))withanodeproportionallocationmecha-usedwithanadjacencymatrixtocaptureandrepresentthesocial
nismasPT  =   TextCNN  (A,X)whereTextCNN  indicatesaconven-contextviafeatureextractions:sharingactions,followingactions
tionalCNNmodel.Letnandm bethenumberofnodesintheregardingpoliticalnews,andanodefeaturematrixareusedto
currentSR-graphandthemaxSR-graph,respectively,wehavecapturethehiddencontentofnewsviawordembeddings.Meyers
thefeatureoutputoftheEGCNbyY  =   PG  ×    nm  +  PT(1−    nm).etal.[153]showedthesignificantroleofpropagationfeatures
AndtheoutputoftheEGCNisdeterminedbyˆy =   Softmax (FC(Y)).infakenewsdetectionmodels.Theauthorsfirstconstructeda
Thismodelfocusesonexploitingtheimpactofnewscontentpropagationgraphtopresentimportantinformationandthen
onthepropagationprocedureofrumors.However,theEGCNusedarandomforestclassifiertotrainthegraphandcreate
requiresdataontheentireconversationregardingthepostednodeembeddings.Finally,theGCNmodelwasusedtopredict
news.Insomecases,ifthepostednewsdoesnotobtainresponsetheauthenticityoftweets.Unlikeotherpropagationgraphs,the
opinionswhenitisspread,itsaccuracycanbereduced.authorsconstructedthefollowinggraph:LetG ={  V ,E}denote
 MultidepthGCNsintroducedbyHuetal.[190]combinethethepropagationgraph,whereVisasetofnodesincludingtweet
similarityofnewstodistinguishthemasfakeorrealviade-nodesandretweetnodesandEisasetofedgesconnected
greesofdifferences.Thismethodcansolvethesignificantchal-betweenatweetnodeanditsretweetnodewithatimeweight.
lengeoffakenewsdetection,whichisautomaticfakenewsThus,thispropagationgraphincludesasetofsubgraphs,where
detectionforshortnewsitemswithlimitedinformation,foreachsubgraphincludesatweetnodeanditsretweetnodes,and
example,headlines.InsteadofstackingtheGCNlayertomergeitsdepthneverexceeds1.
informationoveralongdistance,theauthorscomputedthedif-Asocialspammerdetectionmodel[201]wasbuiltwithacom-
ferentdistanceproximitymatricestodescribetherelationshipbinationoftheGCNandMarkovrandomfield(MRF)models.First,
betweennodesandexplicitlyprotectthemultigranularityinfor-theauthorsusedconvolutionsondirectedgraphstoexplicitly
mation,thusimprovingthenoderepresentationprocesswithconsidervariousneighbors.Theythenpresentedthreeinfluences
thediversityinformation.Therefore,itperformedk-stepproxim-ofneighborsonauser’slabel(follow,follower,reciprocal)using
itytocreatedifferentdepthproximitymatricesbeforefeedingapairwiseMRF.Significantly,theMRFisformulatedasanRNN
totheGCN.Forthestepk-thproximity,theoutputisdefinedformultistepinference.Finally,MRFlayerswerestackedontopof
asz k =  ˆA kReLU (ˆA kXW (0)              theGCNlayersandtrainedviaanend-to-endprocessoftheentire
             k)W (1)k  ,k =1,2,3,...,wherenodefeaturemodel.UnlikeconventionalGCNs,thismodelusesanimproved
matrixXcontainswordembeddingsandrepresentationofcreditforwardpropagationrule
history.ˆA kisthek-thproximitymatrixasˆA k =  ˆA ×  ˆA ×···×   ˆA,
                                  k        Q  =   H(l+1)=  σ(D −1i   A iH(l)W (l)i  +  D −1o   A oH(l)W (l)o
                                         16

H.T. Phan, N.T. Nguyen and D. Hwang                                                                                                                                                                                            Applied Soft Computing 139 (2023) 110235
        +   ˆD −1/2b    ˆA b ˆD −1/2b      H(l)W (l)b)         (20)theirposters.Eisasetofedgesexpressingoneoffourre-
whereA i,A o,A baretypesofneighbors;ˆA b =   A b +  I;D  i,D  o,andlationsbetweentwonodes:follow,followed,spreading,and
ˆD  baredegreematricesof                     spread.ThisgraphhasnodefeaturematrixX andadjacency
                 A i,A o,andA b,respectively;ThenodematrixA.Xiscreatedbycharacterizinguserfeatures,suchas
featurematrixX  =   H(0)iscreatedbasedonBoWfeatures.Then,profiles,networkstructure,andtweetcontent.However,matrix
theauthorsinitializedtheposteriorprobabilitiesoftheMRFlayerAisdefinedasfollows:if(node    v j spreads t    w eet of  node      v i)or
withtheGCNoutputas                           (node    v i spreads t    w eet of  node      v j)or(node    v i follo w s node    v j)or
                 [− w    w   ′]              (node    v j follo w s node    v i)then A       ij =1other   w ise0;Givenmatrices
R  =   Softmax (logH (k)−  A iQ− w    − w    XandA,similartotraditionalGCNs,theauthorsutilizedafour-
       [− w    − w] [− w    w   ′]           layerGCN:twoconvolutionallayersfornoderepresentation
   −  A oQ      −  A b Q     )      (21)     andtwofullyconnectedlayerstopredictthenewsasfakeor
        w  ′    − w   w  ′    − w            real.However,unlikesomepreviousGCNs,inthisproposal,one
wherew,w   ′≥0aretwolearnableparameterstomeasureho-attentionmechanisminthefilters[178]andthemeanpooling
mophilyandheterophilystrengthofMRFmodel.Thismethodareusedtodecreasethefeaturevectors’dimensionforeach
demonstratedthesuperiorityofthecombinationofGCNandconvolutionallayer.SELU[209]isemployedasanonlinearity
MRFlayers.AmultistepMRFlayerisessentialtoconvergence.activationfunctionfortheentirenetwork.
However,thenodefeaturematrixwascreatedsimplywiththeLietal.[123]presentedaGCN-basedantispammethodfor
bag-of-wordsmethod.Thislimitationcanbeimprovedusinglarge-scaleadvertisementsnamedGAS.UnlikepreviousGCNs,
                                             intheGASmodel,acombinationgraphisconstructedbyin-
state-of-the-artembeddingmodelsinthefuture.tegratingthenodesandedgesoftheheterogeneousgraphand
  AnovelGCNframework,calledFauxWard[149],isproposedahomogeneousgraphtocapturethelocalandglobalcom-
forfauxtographydetectionbyexploitingnewscharacteristics,mentcontexts.TheGASisdefinedinthefollowingsteps:(i)
suchaslinguistic,semantic,andstructuralattributes.TheauthorsGraphsconstruction:Theauthorsconstructedtwotypesofgraphs
modeledfauxtographydetectionasaclassificationproblemandnamedXianyugraphandcommentgraph.Thefirstgraphwas
usedGCNstosolvethisproblem.FauxWardissimilartotra-denotedbyG ={  U ,I,E}whereU ,Iaresetsofnodesrepre-
ditionalGCNmodels;however,unlikethesemodels,itaddsasentingusersandtheiritems,respectively,andEisasetof
cluster-basedpoolinglayerbetweengraphconvolutionallayersedgesrepresentingcomments.Anadjacencyofthisgraphiscre-
tolearnthenoderepresentationmoreefficiently.Thecluster-atedasfollows:if  user i makes comment e to item j                 ,  then A       Xij =
basedpoolinglayerfirstassignsneighbornodesintoclusters1other   −  w ise0.Thesecondgraphisconstructedbyconnect-
basedonthenodevectorsofthepreviousgraphconvolutioningnodesexpressingcommentsthathavethesimilarmeaning.
layersandthenlearnsaclusterrepresentationastheinputofThatmeansif  comment i has similar meaning                    w ith j ,  then  A       Cij =
thebackgraphconvolutionlayer.Itperformsgraphconvolution˜A(k)istheupdatedadjacency1other   w ise0.(ii)GCNonXianyugraph:Leth(l)e ,h(l)U(e)andh(l)I(e)
by˜A(k)=   C(k−1)⊤ ˜A(k−1)C(k−1),where       bethel-thlayernodeembeddingsofedge,user,anditem,re-
matrix;C(k)istheclusteringmatrixobtainedafterthek-thgraphspectively,z e =   h(l)e  =  σ(W (l)E  ·concat (h(l−1)e    ,h(l−1)U(e),h(l−1)I(e)))where
convolutionlayer,suchthatH(k)=   C(k−1)⊤ σ(˜A(k−1)H(k−1)W (k−1)),h(0)U(e),I(e)areusernodeanditem
whereH(0)=   Xbeanodefeaturematrix.Unlikeconventionale   =   TN(w0,w1,...,w  n)and
GCNs,thisXiscreatedbyconcatenatingtextcontent,suchasnodeofedgee.Leth(l)N(u),h(l)N(i)areneighborembeddingsofnode
linguistic,sentiment,endorsement,andimagecontent,suchasu,i.TN standsbyTextCNNmodel[210].w  kisthewordvectorof
metadata.                                    wordkintweet.Hence,
  Malhotraetal.[147]introducedamethodofcombiningh(l)N(u)=  σ(W (l)U  ·att(h(l−1)u    ,concat (h(l−1)e    ,h(l−1)i )))   (22)
RoBERTaandBiLSTM(TD  =   Bi(RoTa (tw eet)),whereRoTa indi-where∀ e =(u,i)∈  E(u)and
catesaRoBERTamodel[207]andBiindicatesaBiLSTMmodel)
andGCNmethods(GD   =   H(k)=  σ(ˆAH (k−1)W (k)),whereH(0)=   Xh(l)N(i)=  σ(W (l)I  ·att(h(l−1)i    ,concat (h(l−1)i    ,h(l−1)e ))),      (23)
isanodefeaturematrixbyconcatenatingelevenfeatures,suchwhere∀ e =(u,i)∈  E(i),andE(u)istheedgeconnectedtou;att
asfriendcount,followercount,followeecount,etc.)forrumorastandsofattentionmechanism.Fromthat,wehave:z u =   h(l)
detectionasˆy =   Softmax  (concat (TD ,GD)).Thismodelisbasedon                      u  =
rumorcharacteristics,suchaspropagationandcontent.Itexploitsconcat (W (l)U  ·h(l)u ,h(l)N(u))andz i =   h(l)i  =   concat (W (l)I  ·h(l)i ,h(l)N(i)).(iii)
featuresregardingthestructure,linguistics,andgraphicsoftweetGCNonthecommentgraph:inthisstep,authorsusedtheGCN
news.                                        modelproposedin[211]torepresentnodesonthecomment
  Vladetal.[195]producedamultimodalmultitasklearninggraphintonodeembeddingsasp e =   GCN (X  C ,A C),whereX  Cis
methodbasedontwomaincomponents:memeidentificationnodefeaturematrix.(iv)GASclassifier:TheoutputofGASmodel
andhatespeechdetection.ThefirstcombinesGCNandanItalianisdefinedasy =   classifier (concat (z i,z u,z e,p e)).
BERTfortextrepresentation,whereasthesecondisanimage5.3. Detection approach based on AGNNs
representationmethod,whichvariesamongdifferentimage-
basedstructures.TheimagecomponentemployedVGG-16withRenetal.[17]introducedanovelapproach,calledAA-HGNN,
fiveCNNstacks[208]torepresentimages.Thetextcomponenttomodeluserandcommunityrelationsasaheterogeneousin-
usedtwomechanismstorepresenttext,namely,ItalianBERTformationnetwork(HIN)forcontent-basedandcontext-based
attentionandconvolution.Thismodelismultimodalbecauseitfakenewsdetection.TheprimarytechniqueusedinAA-HGNN
considersfeaturesrelatedtothetextandimagecontentsimul-involvesimprovingthenoderepresentationprocessbylearn-
taneously.Meanwhile,Montietal.[3]introducedageometricingtheheterogeneousinformationnetwork.Inthisstudy,the
deeplearning-basedfakenewsdetectionmethodbyconstructingAGNNsusetwolevelsofanattentionmechanism:thenode
heterogeneousgraphdatatointegrateinformationrelatedtolearnsthesameneighbors’weightsandthenrepresentsthemby
thenews,suchasuserprofileandinteraction,networkstruc-aggregatingtheneighbors’weightscorrespondingtoeachtype-
ture,propagationpatterns,andcontent.GivenaURLuwithspecificneighborandaschematolearnthenodes’information,
asetoftweetsmentionedu,theauthorsconstructedagraphthusobtainingtheoptimalweightofthetype-specificneigh-
G  u ={  V ,E}.V isasetofnodescorrespondingtotweetsandborrepresentations.AssumethatwehaveanewsHINanda
                                           17

H.T. Phan, N.T. Nguyen and D. Hwang                                                                                                                                                                                            Applied Soft Computing 139 (2023) 110235
newsHINschema,denotedbyG ={  V ,E}andS G ={  V T,E T}.Let(⊙)isusedtoreconstructtheadjacencymatrixas˜A =⊙ (ZZ  ⊤)
V  ={  C ∪  N  ∪  S}withC(creators),N (news),S(subjects);andwhereZisthematrixofdistributionsz.(iii)Detectorcomponent:
E ={  E c,n ∪  E n,s}.LetV T ={ θn,θc,θs}andE T ={ w rite ,belongsto     }Thisstepaimstorepresentthelatentinformationandclassifythe
denotestypesofnodesandtypesoflinks.Node-levelattentionnews.ItisdefinedasS =   MP (Z)whereMP standsforthemean-
isdefinedash ′n i =   M  θn ·h n i,wheren i ∈  N,h n iisthefeaturepoolingoperator.Finally,theoutputlayerofthismodelisdefined
vectorofnoden i.M  θnisthetransformationmatrixfortypeasˆy =   Softmax (SW   +  b)whereW istheparametermatrixofthe
θn.LetT ∈{ C ∪  N  ∪  S},tj ∈  Tbelongstotype-neighborθtandfullyconnectedlayer.
tj ∈  neighbor          n i.Leteθtij =   att(h ′n i,h ′tj;θt)istheimportancedegree
ofnodetjforn i,whereattbeanode-levelattentionmechanism6.Discussion
withtheattentionweightcoefficientasα θtij  =   Softmax         j(eθtij)Then,
theschemanodeiscalculatedbyaggregatingfromtheneighbor’s6.1. Discussion on GNNs                  ∗-based methods
featuresasT n i =  σ( ∑α θtij ·h ′tj).Letω θti   =    schema (WT     n i,
             tj∈ neighbor        n i          Previousstudiesforfakenewsdetectionmodelsbasedon
WN      n i)istheimportancedegreeofschemanodeT n i,whereschemaGNNs∗arecomparedinTable7.
beaschema-levelattentionmechanism,N  n iisaschemanodeWepresentedthemainsteps,advantages,anddisadvantages
correspondingtoneighborsofnoden i.Andthefinalfusionco-ofGNN∗-basedmethodsforfakenewsdetection.Someofour
efficientiscalculatedasβ θti  =   Softmax         t(ω θti).Fromthat,wehaveassessmentsareasfollows:Regardingtheextractedfeatures,[4]
anoderepresentationasrn i =  ∑β θt          usedonlyuser-basedfeatures;[5]usedfeaturesbasedonnet-
                        i  ·T n i.AA-HGNNcanstillworks,users,andlinguistics;and[193]usedlinguistic-based
                    θt∈ V T                 features(textualanalysis).Meanwhile,[15]usedfeaturesrelated
achieveexcellentperformancewithoutusingmuch-labeleddatatonetworksandlinguistics.Regardinggraphstructure,[4,5,193]
becauseitbenefitsfromadversarialactivelearning.Itcanalsoconstructedahomogeneousgraph.However,unlike[4,193]only
beusedforotheractualtasksrelatingtoheterogeneousgraphsonegraphwasconstructed,and[5]createdtwosubgraphsto
becauseofitshighgeneralizability.representnewssourcesandnewsusers.Meanwhile,[15]builta
 Benamiraetal.[192]proposedcontent-basedfakenewsde-heterogeneousgraphwithtwotypesofnodesandedges.How-
tectionmethodsforbinarytextclassificationtasks.Theobjectiveever,althoughthegraphstructureof[15]isbetterthanthatof
wasaGNN-basedsemisupervisedmethodtosolvetheproblemtheotherthreemodels,[192]providesthebestperformance.This
oflabeleddatalimitations.Thismethodcomprisesthefollowingresultmaybebecause[5]canbetterextractmeaningfulfeatures
steps:newsembedding;newsrepresentationbasedonknearest-infakenewsdetection.Therefore,todevelopnewGNN∗-based
neighborgraphinference;andnewsclassificationbasedonGNNs,modelsinthefuture,moreattentionshouldbegiventoextracting
suchasAGNN[179]andGCN[164],whichareconventionalGNNsexcellentfeaturesandbuildinggoodstandarddatainsteadof
withoutimprovementsorupdates.               focusingonimprovingthegraphstructure.
5.4. Detection approach based on GAEs
                                            6.2. Discussion on GCNs-based methods
 Usingtheautoencoderspecialgraphdata,Kipf[170]used
GAEtoencodegraphstorepresentlatentstructureinformationPreviousstudiesforfakenewsdetectionmodelsbasedon
ingraphs.GAEsareusedinvariousfields,suchasrecommen-GCNsarecomparedinTables8and9.
dationsystems[212]andlinkprediction[213],withreasonableWepresentedthemainsteps,advantages,anddisadvantages
performance.Recently,researchershavebeguntoapplyGAEsforofGCN-basedmethodsforfakenewsdetection.Inourassess-
fakenewsdetection.Thepreviousstudiesforfakenewsdetectionments,methodssuchas[3,14,16,123,191],and[196],showthe
modelsbasedonGAEsaresummarizedinTable10.bestefficiency,wheretwomethodsareusedforfakenewsde-
 Linetal.[124]proposedamodeltocapturetextual,propaga-tection,twoforrumordetection,andtwoforspamclassification.
tion,andstructuralinformationfromnewsforrumordetection.Regardingthetwopapersinthefirstcategory,[3]wasthefirstto
Themodelincludesthreeparts:anencoder,adecoder,andaapplyGCNsforfakenewsdetection.Thismethodfocusesonex-
detector.TheencoderusesaGCNtorepresentnewstexttotractinguser-based,network-based,andlinguistic-basedfeatures
learninformation,suchastextcontentandpropagation.Thetobuildpropagation-basedheterogeneousGCNs.Theauthors
decoderusestherepresentationsoftheencodertolearnthedeterminedthatthisproposalcanobtainamorepromisingre-
overallnewsstructure.Thedetectoralsousestherepresentationssultthancontent-basedmethods.Conversely,[14]isanenriched
oftheencodertopredictwhethereventsarerumors.ThedecoderGCNwithadualcoattentionmechanism.Thismethodusesuser-
anddetectoraresimultaneouslyimplemented.Thesepartsarebasedandlinguistic-basedfeaturestoconstructhomogeneous
generallydefinedasfollows:(i)Encodercomponent:TwolayersGCNswithadualcoattentionmechanism.Inourassessment,al-
oftheGCNareusedtoenhancethelearningability:though[14]useddualcoattentionmechanisms,theefficiencywas
H(1)=   GCN (X ,A)=  ˆAσ(ˆAXW  (0)W (1))          (24)stilllowerthanthatin[3].Noticeably,thisresultisattributable
                                            mainlytomorefeaturesbeingextractedby[3]thanby[14].
and                                         Additionally,thegraphstructureusedin[3]wasevaluatedas
H(2)=   GCN (H(1),A)=  ˆAσ(ˆAH (1)W (1)W (2))        (25)betterthanthestructureusedin[14].Movingforward,wehope
                                            toimprovetheperformanceoffakenewsdetectionmethodsby
whereσisReLUfunction.XrepresentswordvectorsthatarebuildingdualcoattentionheterogeneousGCNsusinguser-based,
createdbydeterminingtheTF-IDFvalues,andtheadjacentmatrixnetwork-based,andlinguistic-basedfeaturessimultaneously.For
Aisdefinedasfollows:if  node    v i responds to  node          v j,  then A       ij =thetwopapersinthesecondcategory,bothmethodswerebuilt
1other   w ise0.Then,theGCNisusedtolearnaGaussiandistribu-todetectrumorsbypropagation-basedGCNs.Thedifferenceis
tionforvariationalGAEasz =  µ +  ϵσ,whereµ  =   GCN (H(1),A)that[16]constructedbidirectionalGCNstocapturetherumor
andlog σ =   GCN (H(1),A)(µ,σ,andϵarethemean,standarddispersionstructureandrumorpropagationpatternssimultane-
deviation,andstandardsampleoftheGaussiandistribution,re-ously.Meanwhile,[196]createdunidirectionalGCNsbasedonthe
spectively).(ii)Decodercomponent:Inthisstep,aninnerproductinformationofmultiorderneighborstocapturerumorsources.
                                          18

H.T. Phan, N.T. Nguyen and D. Hwang                                                                                                                                                                                            Applied Soft Computing 139 (2023) 110235
Table7
ComparisonofGNNs∗methods.
Method[Ref]  Criticalidea            Lossfunction        Advantage             Disadvantage
Benamiraetal.–FocusonanalyzingthenewsCross-entropyloss      –Canobtainhighefficacywith–Havenotbeenevaluatedwith
[192]     contentusingsemi-supervised         limitedlabeleddatabigdataandmulti-labeleddata
          learning
          –Binaryclassificationmodel
YiHanetal.–Propagation-basedfakenewsElasticWeightConsolidation–Canimprovetheperformanceof–Ignoretheselectionoffeatures
[4]       detectionmethod     loss            conventionalmethodswithoutorthefindingof‘‘universal’’
          –Focusoncontinuallearningand        usinganytextinformationfeatures
          incrementaltrainingtechniques       –Canhandleunseendataand
          –Usetwotechniques:EWCand            newdata
          GEM
FakeNews[193] –FocusontwotasksaimingtoNA             –Canusetomulti-classification–Tasksimplementseparately
          analysisanddetectfakenewsvia        tasks              correspondingtoinformation
          newstextualandnewsstructure         –Binaryclassificationtaskobtainstextualandstructure
                                              significantlyhigherperformance
                                              thantheternaryones
SAFER[15]   –ContextualfakenewsdetectionNA             –Improvetheperformanceofthe–Sensitivetobottleneckand
          method                              traditionalGNNs    over-smoothingproblems
          –Focusoncombining                   –Canaddmorelayerstoidentify
          information:contentnature,user      moreefficacyneighborhood
          behaviors,anduserssocial
          network
FANG[5]    –ContextualfakenewsdetectionThetotalofunsupervised–Improverepresentationquality–Stancedetectionandtextual
          method              proximityloss,  –Canusetoalimitedtrainingencodehavenotbeenjointly
          –Focusonrepresentationqualityself-super-visedstanceloss,datasetoptimized
          bycapturingsharingpatternsandandsupervisedfakenews–Cancapturetemporalpatterns–Sharingcontentsandhyperlinks
          socialstructure     loss            offakenews         becomeobsoletequickly
Inourview,[16]canoutperform[196]becauserumordetec-representationofgraphs.Thismethodcancapturetheentire
tion,rumorpropagation,anddispersionaremorecriticalthanstructuralinformationefficiency.Itcanthusenrichtraditional
rumorsources.Forthetwopapersinthelastcategory,[123,191]GCNsbyaddingtwomorecomponents,namely,thedecoder
alsoproposedsimilarmethodsforspamdetectionusingsocialanddetector.However,thisstudyfocusedonlyonuser-based
context-basedGCNs.Thedifferentpointsarethat[123]builtaandlinguistic-basedfeatures,ignoringnetwork-basedfeatures;
modelintegratingheterogeneousandhomogeneousgraphstotherefore,thedesiredeffectisnotexpected.
capturebothlocalandglobalnewscontexts.Incontrast,[191]
constructedonlyoneheterogeneousgraphtocapturethegeneral7.Challenges
newscontext.Inouropinion,themodelpresentedin[123]is
morecomprehensible,canbereimplemented,andyieldsslightly7.1. Fake news detection challenges
betterresultsthanthemethodin[191].Thereasonforthisresult
isthatbuildingeachtypeofgraphissuitableforthecaptureBasedonrecentpublicationsinthefieldoffakenewsde-
andintegrationofeachtypeofcontext,whichcancapturethetection,wesummarizedandclassifiedchallengesintofivecat-
newscontextmorecomprehensivelythanconstructingonegraphegories,whereeachcategoryofchallengecorrespondstoone
forallcontexts.Thus,whenbuildingfakenewsdetectionmodelscategoryoffakenewsdetection.Thedetailsofeachtypeof
basedonGNNs,differentgraphsshouldbeconstructedtocapturechallengeareshowninFig.6.Thefollowingpresentssignifi-
eachspecifictypeofinformationandthenperformthefusioncantchallengesthatcanbecomefuturedirectionsinfakenews
step.Thisapproachpromisestoprovidebetterperformancethandetection.
buildingonetypeofgraphtocapturealltypesofinformation.Deepfake[214]isahyperrealistic,digitallycontrolledvideo
Wemaximallylimittheconstructionofageneralgraphandthenthatshowspeoplesayingordoingthingsthatnevertrulyhap-
divideitintospecifictypesbecausethebreakdownofthegraphpenedorcompositedocumentsgeneratedbasedonartificialin-
caneasilyresultinthelossofinformationontherelationshiptelligencetechniques.Giventhesophisticationofthesecounter-
amongedges.                                feitingtechniques,determiningtheveracityofthepublicap-
6.3. Discussion on AGNNs- and GAEs-based methodspearancesorinfluencerclaimsischallengingowingtofabricated
                                           descriptions.Therefore,Deepfakecurrentlyposesasignificant
 Previousstudiesforfakenewsdetectionmodelsbasedonchallengetofakenewsdetection.
AGNNsandGAEsarecomparedinTable10.            Thehackingofinfluencers’accountstospreadfakenewsor
 Wepresentedthemainsteps,advantages,anddisadvantagesdisinformationaboutaspeechbycelebritiesthemselvesisalso
ofthetwomethodsintheAGNNandAGEcategoriesforfakeauniquephenomenoninfakenewsdetection.However,this
newsdetection.Evidently,[17]presentedamoredetailedfakeinformationwillbequicklyremovedwhentheactualownerof
newsdetectionmethodthan[192].Additionally,themethodtheseaccountsdiscoversandcorrectsthem.However,atthetime
in[17]wasproposedafterthatin[192];thus,itisbetterthanofitsspread,thisinformationcausesextremelyharmfuleffects.
[192].Forexample,[192]constructedahomogeneousgraph,Instantlydetectingwhetherthepostsofinfluencersarefakehas
whereas[17]createdaheterogeneousgraph.Theheterogeneousthusbecomeanimportantchallenge.
graphwasevaluatedassuperiortothehomogeneousgraphbe-Newsmaybefakeatonepointintimeandrealatanother.
causeitcancapturemoremeaningfulinformation.Therefore,itThatis,thenewsisrealorfake,dependingonthetimeitissaid
obtainsbetterresultsthan[192].Meanwhile,theLinetal.[124]andspread.Therefore,real-timefakenewsdetectionhasnotyet
methodusesaconventionalGCNvarianttoencodethelatentbeenthoroughlyaddressed.
                                         19

H.T. Phan, N.T. Nguyen and D. Hwang                                                                                                                                                                                            Applied Soft Computing 139 (2023) 110235
Table8
ComparisonofGCNmethods.
Method[Ref]  Criticalidea              Lossfunction       Advantage             Disadvantage
Montietal.[3] –FocusonanalyzingthenewsHingeloss        –Enabletointegrate–Onlyimplementwiththebinary
          content,users,socialstructureandheterogeneousdata         classificationtask
          propagationusinggeometricdeep         –Obtainveryhighperformance
          learning                              withbigrealdata
          –Binaryclassificationtask
!GAS[123]   –FocusoncapturingtheglobalandRegressionloss      –Cansolvespamproblemslike–Onlyimplementwiththebinary
          localcontextsofthenews                adversarialactionsandscalabilityclassificationtask
          –Integrategraphsofhomogeneous         –Obtainhighperformancewith
          andheterogeneous                      large-scaledata
          –Binaryclassificationtask             –Canapplytoonlinenews
Marionetal.–FocusoncapturingthepropagationNA            –Canapplytothe–Useaverybroaddefinition
[153]     featuresofthenewsusinggeometricnon-URL-limitednews–Applytoasingledatasource
          deeplearning                                              –Nothighgeneralizability
          –Binaryclassificationtask
GCAN[14]    –FocusontherelationoforiginalCross-entropyloss    –Canearlydetection–Applytoasingledatasource
          tweetandretweetandthe                 –Candetectatweetstoryasfake–Nothighgeneralization
          co-influenceoftheuserinteractionusingonlyshort-texttweet
          andoriginaltweet                      withoutneedingusercomments
          –Usethedualco-attention               andnetworkstructure
          mechanism                             –Explainableoffakereasons
          –Binaryclassificationtask
Pehlivanetal.–FocusonthetemporalfeaturesofCross-entropyloss    –Canapplytometadata      –Notpromisingperformance
[194]     thenetworkstructurewithout                                –Thedataissplitnotreasonable
          consideringanytextualfeatures                             fortraining,testing,validation
          –Binaryclassificationtask
*Bi-GCN[16]   –FocusonanalyzingfeaturesrelatedCross-entropyloss    –Haveanearlydetection–Nothighgeneralization
          todispersionandpropagationofthemechanism
          news                                  –Candetectrumorsinreal-time
          –Constructatop-downgraphto            –Obtainmuchhigher
          learnrumorspreadandabottom-up         performancethanstate-of-the-art
          graphtocapturerumordispersion         methods
          –Multiclassificationtask
*GCNSI[196]  –FocusonidentifyingmultipleSigmiodcross-entropy–Firstmodelbasedonmultiple–Havetoretrainthemodelifthe
          sourcesofrumorwithoutanyloss          sourcesoftherumorgraphstructureischanged
          knowledgerelatedtonews                –Improvetheperformanceofthe–Takequitemuchtimetotrain
          propagation                           state-of-the-artmethodsbyaboutandobtainsuitableparameters
          –ImprovethepreviousGCNmodels          15%
          bymodifyingtheenhancednode
          representationsandlossfunction
          –Multiclassificationtask
!GCNwithMRF–Thefirstsemi-supervisedmodelCross-entropyloss    –Obtainsuperioreffectiveness–UsesimpleBoWforfeatures
[201]     focusoncontinuouslyintegrating        –Canensureconvergencerepresentation
          bothmethodsoffeature-basedand
          propagation-based
          –Usethedeeplearningmodelwitha
          refinedMRFlayerondirectedgraphs
          toenabletheend-to-endtraining
          –Multiclassificationtask
  Constructingbenchmarkdatasetsanddeterminingthestan-regardingthealgorithmsused,themanyextractedfeatures,and
dardfeaturesetscorrespondingtoeachapproachforfakenewshighfeaturedimensions,theycansimultaneouslycapturevarious
detectionremainchallenges.                   aspectsoffakenews.Therefore,themostefficaciousandleast
  KaiShuetal.[215]constructedthefirstfakenewsdetectioncostlyextractionofcontent,propagationpatterns,andusers’
methodsbyeffectivelyextractingcontent,context,andpropa-stancesimultaneouslyisnotonlyapromisingsolutionbutalso
gationfeaturessimultaneouslythroughfourembeddingcompo-asignificantchallengeforfakenewsdetection.
nents:newscontent,newsusers,user-newsinteractions,and7.2. Challenges related to graph neural networks
publishernewsrelations.Then,thesefourembeddingswerefed
intoasemisupervisedclassificationmethodtolearnaclassifica-Basedonstudyingtherelatedliterature,thissectionsumma-
tionfunctionforunlabelednews.Inaddition,thismethodcanrizessomechallengesofGNN-basedmethodsandthenidentifies
beusedforfakenewsearlydetection.Ruchanskyetal.[28]possiblefuturedirections.
constructedamoreaccuratefakenewspredictionmodelbyex-MostconventionalGNNsutilizeundirectedgraphsandedge
tractingthebehaviorofusers,news,andthegroupbehaviorofweightsasbinaryvalues(1and0)[216]unsuitableformany
fakenewspropagators.Then,threefeatureswerefedintothear-actualtasks.Forexample,ingraphclustering,agraphpartition
chitecture,includingthreemodulesasfollows:(i)usearecurrentissoughtthatsatisfiestwoconditions:(i)thedifferencebetween
neuralnetworktocapturethetemporalactivityofauserongiventheweightsofedgesamongunlikegroupsisaslowaspossible;
newsvianewsandpropagatorbehaviors;(ii)learnthenews(ii)thedifferenceintheweightsofedgesamongsimilargroups
sourceviauserbehavior;and(iii)integratetheprevioustwoisashighaspossible.Here,iftheweightoftheedgesisabinary
modulesforfakenewsdetectionwithhighaccuracy.Fromthisvalue,thegivenproblemcannotbesolvedusingthisgraph.
surveyofliterature,weseethatthemosteffectiveapproachesTherefore,futurestudiescanconstructgraphswiththeweights
combinefeaturesregardingcontent,context,andpropagation.ofedgesastheactualvaluesrepresentingtherelationshipamong
Althoughthesecombinationmethodsmayhavehighcomplexitythenodesasmuchaspossible.
                                           20

H.T. Phan, N.T. Nguyen and D. Hwang                                                                                                                                                                                            Applied Soft Computing 139 (2023) 110235
Table9
ComparisonofGCNmethods(continued).
Method[Ref]   Criticalidea                Lossfunction   Advantage               Disadvantage
*Malhotraetal.–FocusoncombiningfeaturesrelatedCross-entropy–Enableformoreefficientlyfeatures–Evaluatedbyalimited
[147]       totextandusers             loss         extraction               dataset
            –Usethegeometricdeeplearningwith                                 –Overfittingtesterror
            RoBERTa-basedembedding
            –Multiclassificationtask
!FauxWard   –FocusonfeaturesrelatedtotheCross-entropy–Obtainasignificantperformance–Notdirectlyanalyzethe
[149]       linguisticandsemanticoftheuserloss      withinashorttimewindowcontentofthenewscontaining
            commentsandtheusernetwork                                        animage-centric
            structure
            –Usethegeometricdeeplearningona
            usercommentnetwork
            –Binaryclassificationmodel
KZWANG[156]  –FocusondepthintegratingofCross-entropy–Haveanearlydetectionmechanism–Randomsplitforvalidation
            contextualinformationandpropagationloss –Cancreateabetter        dataandmanualsplitfor
            structure                               semantic-integratedrepresentationtraining,testingdata
            –Usemulti-headattentionmechanism–Improveperformancesignificantly
            tocreatecontextualrepresentation
            withoutextractinganyfeatures
            –Multiclassificationmodel
*GraphSAGE–FocusondeterminingpatternsCross-entropy–Highgeneralizationforunseendata–Canreducetheperformance
[125]       propagation-basedcharacteristicsandloss –Reducethedetectionerrorofifnotusefullinformationof
            informationrelatedtothecontent,socialstate-of-the-artmethodsdowntopost(originalandresponse)in
            networkstructure,anddelaytime           10%                      thespreadprocess
            –Useagraphembeddingtechniqueto–Efficientlyintegratefeaturesrelated
            integrateinformationofgraphstructuretothewholepropagatedpost
            andnodefeatures
            –Multiclassificationmodel
Bert-GCN    –FocusonusingfeaturesrelatedtotheNA        –Cancreatebetterword–Nothighgeneralization
Bert-VGCN   contentofnewstext                       representations          –Nosuitableaugmentation
[150]       –ImprovetheotherGCN-basedmodels–Canimprovetheperformanceofdatatoimprovefeatures
            usingBERT-basedembeddings               theconventionalGCNmethodextractionandavoidoverfitting
            –Multiclassificationmodel               significantly
*Lotfi[204]    –Focusoninformationoftextcontent,Cross-entropy–Obtainhighefficacyinearly–Strongdependonthefull
            spreadtime,socialnetworkstructureloss   detection                informationofbothoriginal
            –Constructweightedgraphsbasedon–Canimprovetheperformanceoftweetandresponsetweetsof
            usersinteractioninconversationsthestate-of-the-artmethodsconversations
            –Binaryclassificationmodel              significantly
*SAGNN[151]   –FocusoncapturingtheinformationofCross-entropy–Optimalcaptureofuser’s–Nothigh-performance
            usersinteractions          loss         interactions             generalizationduetoonly
            –ImprovetheconventionalGCNmodels–Capturebetterthedifferentcomparingwithonebaseline
            byaddingoneormoreaggregationlayerfeaturesbetweenrumorsandmethod
                                                    non-rumors
            –Multiclassificationmodel
*EGCN[154]   –FocusonfullyextractingfeaturesNA        –Canobtaincomparable–Nothighgeneralization
            relatedtotextcontentandstructureperformanceorbetterthanmachine
            –Constructweightedgraphsof              learningmethods
            source-repliesrelationforconversations–Canusetheinformationofthe
            –Binaryclassificationmodel              globalandlocalstructure
                                                    simultaneously
Table10
ComparisonofAGNNandGAEmethods.
Method[Ref]   Criticalidea                Lossfunction   Advantage               Disadvantage
Benamira    –FocusonanalyzingthenewscontentCross-entropy–Canobtaingoodefficacywith–Havenotbeenevaluated
etal.[192]  usingsemi-supervisedlearningloss        limitedlabeleddata       withbigdataand
            –Binaryclassificationmodel                                       multi-labeleddata
AA-HGNN[17]  –ThefirstmodelusingadversarialCross-entropy–Supportearlydetectionstage–Notcomparetheefficacy
            activelearningforfakenewsdetectionloss  –Stillobtainhighperformancewithwiththecontext-based
            –ImprovetheconventionalGCNmodelslimitedtrainingdata              methods
            byusinganewhierarchicalattention–Canextractinformationastextand
            mechanismfornoderepresentation          structuresimultaneously
            –Multiclassificationmodel
Linetal.[124]  –FocusonintegratingtheinformationThesumof–ThefirstGAE-basedrumor–Lowperformanceforthe
            relatedtotext,propagation,andCross-entropydetectionmethod        non-rumorclass
            networkstructure           lossandKL–Cancreatebetterandhigh-level–Nothighgeneralizationfor
            –Includethreeparts:encoder,decoder,divergencelossnoderepresentationstheperformance
            anddetector                             –Obtainbetterefficacythanother
            –Multiclassificationmodel               thelatestmethods
  ForNLPtasks,GNNshavenotrepresentednodefeaturesbythis milk tea is not very fragrant.’’     Thissentenceincludesafuzzy
capturingthecontextofaparagraphoranentiresentence.Al-sentimentphrase,namely‘‘not very fragrant’’  .Someapproaches
ternatively,thesemethodshavealsooverlookedthesemanticclassifythissentenceasexpressingapositivesentimentbecause
relationshipsamongphrasesinthesentences.Forexample,fortheyonlyfocuson‘‘fragrant’’ ,ignoringtheroleofboth‘‘not’’
sentimentclassificationtasks,wehavethesentence‘‘The smell ofand‘‘very’’,whereasothermodelsdeterminetheexpressionas
                                               21

H.T. Phan, N.T. Nguyen and D. Hwang                                                                                                                                                                                            Applied Soft Computing 139 (2023) 110235
                           Fig.6.Listofchallengesoffakenewsdetection.
anegativesentimentbecausetheyignoretheimpactof‘‘very’’.sections.Nonetheless,giventhe27surveyedpapers,promising
Therefore,futuredirectionsforimprovingGNN-basedmodelsresultsareexpectedinthefuture.Byaddressingthesechallenges,
shouldfocusondeterminingnodefeaturesbasedonsentencewehopetoimprovetheeffectivenessofGNN-basedfakenews
embeddingsorsignificantphraseembeddings.detection.Thefollowingparagraphsanalyzesomechallengesfor
 Capturingcontext,content,semanticrelations,andsentimentGNN-basedfakenewsdetectionanddiscussfuturedirections.
knowledgesimultaneouslyinsentencesisessentialforGNN-Benchmarkdata:Recently,someresearchershaveargued
basedNLPtasks.Meanwhile,onlyafewstudieshaveincorporatedthatwhentrainingasystem,dataaffectsystemperformance
someofthesefeaturesbyflexibleGNNstoimprovetheefficiencymorethanalgorithmsdo[221].However,inourassessment,
ofNLPtasks,includingfakenewsdetection.Forinstance,in[217],wehadnographbenchmarkdataforfakenewsdetectionin
theauthorsextractedcommonsenseknowledgeandsyntaxviathegraphlearningcommunity.Graph-basedfakenewsdetection
GNNs,whereasin[218],theauthorsconstructedasingletext-benchmarksmaypresentanopportunityanddirectionforfuture
basedGNNbyrepresentingdocument-wordrelationsandwordresearch.
co-occurrence.Tothebestofourknowledge,noGNNhassi-Compatiblehardware:WiththerapidgrowthofDeepfake,
multaneouslyconsideredallthecontent,contexts,commonsensegraphstorepresentthesedatawillbecomemorecomplex.How-
knowledge,andsemanticrelations.Thistaskremainsanexcitingever,themorescalableGNNsare,thehigherthepriceandcom-
challengeforNLPtask-basedGNNs.          plexityofthealgorithmsis.Scientistsoftenusegraphclustering
 Sofar,GCNshavebeenlimitedtoafewlayers(twoorthree)orgraphsamplingtosolvethisproblem,ignoringtheinforma-
owingtothevanishinggradient,whichlimitstheirreal-worldtionlossofthegraphusingthesetechniques.Therefore,inthe
applications.Forexample,GCNsin[217,219,220]stoppedattwofuture,graphscalabilitymaybesolvedbydevelopingdedicated
layersbecauseofthevanishinggradienterror.Therefore,con-hardwarethatfitsthegraphstructure.Forexample,GPUswere
structingdeepfuzzyGCNsofsyntactic,knowledge,andcontextaconsiderableleapforwardinloweringthepriceandincreasing
byusingthedeeplearningalgorithmoverthecombinationgraphthespeedofdeeplearningalgorithms.
ofthefuzzysyntacticgraph,thefuzzyknowledgegraphandtheFakenewsearlydetection:Earlydetectionoffakenewsin-
fuzzycontextgraphcansolvetheaforementionedlimitationsofvolvesdetectingfakenewsatanearlystagebeforeitiswidely
previousmethodsforaspect-levelsentimentanalysis.disseminatedsothatpeoplecaninterveneearly,preventitearly,
                                        andlimititsharm.Earlydetectionoffakenewsmustbeac-
8.Conclusionandopenissues               complishedassoonaspossiblebecausethemorewidespread
                                        fakenewsis,themorelikelyitisthattheauthenticationeffect
 GNN-basedfakenewsdetectionisrelativelynew.Thus,thewilltakehold,meaningthatpeoplewillbelikelytobelieve
numberofpublishedstudiesislimited.Althoughwedidnottheinformation.Currently,forfakenewsearlydetection,people
implementmethodspresentedinthe27studiesonthesameoftenfocusonanalyzingthenewscontentandthenewscontext,
datasetsanddidnotevaluatetheirefficiencyonthesamecom-whichleadstothreechallenges.First,newnewsoftenappearsto
parisoncriteria,the27paperssurveyedhereshowthatthisbringnewknowledge,whichhasnotbeenstoredintheexisting
methodinitiallyobtainedexcellentresults.Additionally,manytrustknowledgegraphandcannotbeupdatedimmediatelyat
challengesneedtobeaddressedtoachievemorecomprehensivethetimethenewsappears.Second,fakenewstendstobewrit-
results,whichwediscussedattheendofthecorrespondingtenwiththesamecontentbutwithdifferentdeceptivewriting
                                      22

H.T. Phan, N.T. Nguyen and D. Hwang                                                                                                                                                                                            Applied Soft Computing 139 (2023) 110235
stylesandtoappearsimultaneouslyinmanyvariousfields.Fi-•TSHP-17:TheEnglishdatasetwith33,063articlesregarding
nally,limitedinformationrelatedtonewscontent,newscontext,politicswascollectedfromonlinestreaming.
newspropagation,andlatentinformationcanadverselyaffectthe•QProp9:TheEnglishdatasetwith51,294articlesregarding
performanceofGNN-baseddetectionmethods.politicswascollectedfromonlinestreaming.
 DynamicGNNs:MostgraphsusedinthecurrentGNN-based•NELA-GT-201810:TheEnglishdatasetwith713,000articles
fakenewsdetectionmethodshaveastaticstructurethatisregardingpoliticswascollectedfromonlinestreamingfrom
difficulttoupdateinrealtime.Incontrast,newsauthenticityFebruary2018toNovember2018.
canchangecontinuouslyovertime.Therefore,itisnecessaryto•TW_info:TheEnglishdatasetwith3472articlesregarding
constructdynamicgraphsthatarespatiotemporallycapableofpoliticswascollectedfromTwitterfromJanuary2015to
changingwithreal-timeinformation.             April2019.
 HeterogeneousGNNs:ThemajorityofcurrentGNN-based•FCV-2018:Thedataset,including8languageswith380
fakenewsdetectionmodelsconstructhomogeneousgraphs.How-videosand77,258tweetsregardingsociety,wascollected
ever,itisdifficulttorepresentallthenewstexts,images,andfromthreesocialnetworks,namelyYouTube,Facebook,and
videossimultaneouslyonthesegraphs.TheuseofheterogeneousTwitterfromApril2017toJuly2017.
graphsthatcontaindifferenttypesofedgesandnodesisthus•VerificationCorpus:Thedatasetincluding4languageswith
afutureresearchdirection.NewGNNsaresuitableforhetero-15,629postsregarding17societyevents(hoaxes)wascol-
geneousgraphs,whicharerequiredinthefakenewsdetectionlectedfromTwitterfrom2012to2015.
field.                                      •CNN/DailyMail:TheEnglishdatasetwith287,000articles
 MultiplexGNNs:AsanalyzedinSection7.2,mostGNN-basedregardingpolitics,society,crime,sport,business,technol-
fakenewsdetectionapproacheshavefocusedonindependentlyogy,andhealthwascollectedfromTwitterfromApril2007
usingpropagation,content,orcontextfeaturesforclassification.toApril2015.
Veryfewmethodshaveusedacombinationoftwoofthethree•Tametal.’sdataset:TheEnglishdatasetwith1022rumors
features.Noapproachusesahybridofpropagation,content,andand4milliontweetsregardingpolitics,science,technology,
contextsimultaneouslyinonemodel.Therefore,thisissueisalsocrime,fauxtography,andfraud/scamwascollectedfrom
acurrentchallengeinfakenewsdetection.Inthefuture,researchTwitterfromMay2017toNovember2017.
shouldbuildGNNmodelsbyconstructingmultiplexgraphsto•FakeHealth11:TheEnglishdatasetwith500,000tweets,
representnewspropagation,content,andcontextinthesame29,000replies,14,000retweets,and27,000userprofiles
structure.                                    withtimelinesandfriendlistsregardinghealthwerecol-
Declarationofcompetinginterest                lectedfromTwitter.
 Theauthorsdeclarethattheyhavenoknowncompetingfinan-References
cialinterestsorpersonalrelationshipsthatcouldhaveappeared[1]A.Bovet,H.A.Makse,InfluenceoffakenewsinTwitterduringthe2016
toinfluencetheworkreportedinthispaper.USpresidentialelection,NatureCommun.10(1)(2019)1–14.
                                            [2]H.Allcott,M.Gentzkow,Socialmediaandfakenewsinthe2016election,
Dataavailability                              J.Econ.Perspect.31(2)(2017)211–236.
                                            [3]F.Monti,F.Frasca,D.Eynard,D.Mannion,M.M.Bronstein,Fakenews
 Nodatawasusedfortheresearchdescribedinthearticledetectiononsocialmediausinggeometricdeeplearning,2019,arXiv
                                              preprintarXiv:1902.06673.
Appendix.Descriptionofdatasets              [4]Y.Han,S.Karunasekera,C.Leckie,Graphneuralnetworkswithcontinual
                                              learningforfakenewsdetectionfromsocialmedia,2020,arXivpreprint
                                              arXiv:2007.03316.
                                            [5]V.-H.Nguyen,K.Sugiyama,P.Nakov,M.-Y.Kan,Fang:Leveragingsocial
 •Fact-checking:TheEnglishdatasetwith221statementscontextforfakenewsdetectionusinggraphrepresentation,in:Pro-
   regardingsocietyandpoliticswascollectedfromonlineceedingsofthe29thACMInternationalConferenceonInformation&
   streaming.                                 KnowledgeManagement,2020,pp.1165–1174.
 •EMERGENT:TheEnglishdatasetwith300claimsand2595[6]Z.Wu,S.Pan,F.Chen,G.Long,C.Zhang,S.Y.Philip,Acomprehensive
   associatedarticleheadlinesregardingsocietyandtechnol-surveyongraphneuralnetworks,IEEETrans.NeuralNetw.Learn.Syst.
                                              32(1)(2020)4–24.
   ogywerecollectedfromonlinestreamingandTwitter.[7]J.Redmon,S.Divvala,R.Girshick,A.Farhadi,Youonlylookonce:Unified,
 •BenjaminPoliticalNews:TheEnglishdatasetwith225sto-real-timeobjectdetection,in:ProceedingsoftheIEEEConferenceon
   riesregardingpoliticswascollectedfromonlinestreamingComputerVisionandPatternRecognition,2016,pp.779–788.
   from2014to2015.                          [8]W.Shi,R.Rajkumar,Point-gnn:Graphneuralnetworkfor3dobject
 •BurfootSatireNews7:TheEnglishdatasetwith4233newsdetectioninapointcloud,in:ProceedingsoftheIEEE/CVFConference
                                              onComputerVisionandPatternRecognition,2020,pp.1711–1719.
   articlesregardingeconomy,politics,society,andtechnology[9]G.Chen,Y.Tian,Y.Song,Jointaspectextractionandsentimentanalysis
   wascollectedfromonlinestreaming.withdirectionalgraphconvolutionalnetworks,in:Proceedingsofthe
 •MisInfoText8:TheEnglishdatasetwith1692newsarticles28thInternationalConferenceonComputationalLinguistics,2020,pp.
   regardingsocietywascollectedfromonlinestreaming.272–279.
                                           [10]C.Zhang,Q.Li,D.Song,Aspect-basedsentimentclassificationwithaspect-
 •Ottetal.’sdataset:TheEnglishdatasetwith800reviewsspecificgraphconvolutionalnetworks,2019,arXivpreprintarXiv:1909.
   regardingtourismwascollectedfromTripAdvisorsocial03477.
   media.                                  [11]J.Bastings,I.Titov,W.Aziz,D.Marcheggiani,K.Sima’an,Graphconvolu-
 •FNC-1:TheEnglishdatasetwith49,972articlesregardingtionalencodersforsyntax-awareneuralmachinetranslation,2017,arXiv
   politicsandsocietywerecollectedfromonlinestreaming.preprintarXiv:1704.04675.
                                           [12]D.Marcheggiani,J.Bastings,I.Titov,Exploitingsemanticsinneural
 •Fake_or_real_news:TheEnglishdatasetwith6337articlesmachinetranslationwithgraphconvolutionalnetworks,2018,arXiv
   regardingpoliticsandsocietywascollectedfromonlinepreprintarXiv:1804.08313.
   streaming.
                                           9http://proppy.qcri.org/about.html
 7http://www.csse.unimelb.edu.au/research/lt/resources/satire/10https://doi.org/10.7910/DVN/ULHLCB
 8https://github.com/sfu-discourse-lab/MisInfoText11https://tinyurl.com/y36h42zu
                                        23

H.T. Phan, N.T. Nguyen and D. Hwang                                                                                                                                                                                            Applied Soft Computing 139 (2023) 110235
[13]S.Zhang,H.Tong,J.Xu,R.Maciejewski,Graphconvolutionalnetworks:[44]N.Kshetri,J.Voas,Theeconomicsof‘‘fakenews’’,ITProf.19(6)(2017)
    Acomprehensivereview,Comput.Soc.Networks6(1)(2019)1–23.8–12.
[14]Y.-J.Lu,C.-T.Li,GCAN:Graph-awareco-attentionnetworksforexplain-[45]E.J.Fox,S.J.Hoch,Cherry-picking,J.Mark.69(1)(2005)46–62.
    ablefakenewsdetectiononsocialmedia,2020,arXivpreprint[46]A.Zubiaga,A.Aker,K.Bontcheva,M.Liakata,R.Procter,DetectionandarXiv:
    2004.11648.                                      resolutionofrumoursinsocialmedia:Asurvey,ACMComput.Surv.51
[15]S.Chandra,P.Mishra,H.Yannakoudakis,M.Nimishakavi,M.Saeidi,E.(2)(2018)1–36.
    Shutova,Graph-basedmodelingofonlinecommunitiesforfakenews[47]F.Allen,D.Gale,Stock-pricemanipulation,Rev.Financ.Stud.5(3)(1992)
    detection,2020,arXivpreprintarXiv:2008.06274.    503–529.
[16]T.Bian,X.Xiao,T.Xu,P.Zhao,W.Huang,Y.Rong,J.Huang,Rumorde-[48]V.L.Rubin,Y.Chen,N.K.Conroy,Deceptiondetectionfornews:Three
    tectiononsocialmediawithbi-directionalgraphconvolutionalnetworks,typesoffakes,Proc.Assoc.Inf.Sci.Technol.52(1)(2015)1–4.
    in:ProceedingsoftheAAAIConferenceonArtificialIntelligence,vol.34,[49]Y.Chen,N.J.Conroy,V.L.Rubin,Misleadingonlinecontent:Recognizing
    (no.01)2020,pp.549–556.                          clickbaitas"falsenews",in:Proceedingsofthe2015ACMonWorkshop
[17]Y.Ren,B.Wang,J.Zhang,Y.Chang,AdversarialactivelearningbasedonMultimodalDeceptionDetection,2015,pp.15–19.
    heterogeneousgraphneuralnetworkforfakenewsdetection,in:2020[50]B.Hofmann,Fakefactsandalternativetruthsinmedicalresearch,BMC
    IEEEInternationalConferenceonDataMining,ICDM,IEEE,2020,pp.Med.Ethics19(1)(2018)1–5.
    452–461.                                      [51]M.Gentzkow,J.M.Shapiro,D.F.Stone,Mediabiasinthemarketplace:
[18]B.Collins,D.T.Hoang,N.T.Nguyen,D.Hwang,TrendsincombatingfakeTheory,in:HandbookofMediaEconomics,vol.1,Elsevier,2015,pp.
    newsonsocialmedia–Asurvey,J.Inf.Telecommun.5(2)(2021)247–266.623–645.
[19]T.Khan,A.Michalas,A.Akhunzada,Fakenewsoutbreak2021:Canwe[52]A.D’Ulizia,M.C.Caschera,F.Ferri,P.Grifoni,Fakenewsdetection:A
    stoptheviralspread?J.Netw.Comput.Appl.(2021)103112.surveyofevaluationdatasets,PeerJComput.Sci.7(2021)e518.
[20]V.Klyuev,Fakenewsfiltering:Semanticapproaches,in:20187thInterna-[53]K.Sharma,F.Qian,H.Jiang,N.Ruchansky,M.Zhang,Y.Liu,Combating
    tionalConferenceonReliability,InfocomTechnologiesandOptimizationfakenews:Asurveyonidentificationandmitigationtechniques,ACM
    (TrendsandFutureDirections),ICRITO,IEEE,2018,pp.9–15.Trans.Intell.Syst.Technol.10(3)(2019)1–42.
[21]R.Oshikawa,J.Qian,W.Y.Wang,Asurveyonnaturallanguageprocessing[54]H.Ahmed,I.Traore,S.Saad,Detectionofonlinefakenewsusingn-gram
    forfakenewsdetection,2018,arXivpreprintarXiv:1811.00770.analysisandmachinelearningtechniques,in:InternationalConference
[22]K.Shu,A.Bhattacharjee,F.Alatawi,T.H.Nazer,K.Ding,M.Karami,H.onIntelligent,Secure,andDependableSystemsinDistributedandCloud
    Liu,Combatingdisinformationinasocialmediaage,WileyInterdiscipl.Environments,Springer,2017,pp.127–138.
    Rev.:DataMin.Knowl.Discov.10(6)(2020)e1385.[55]H.Ahmed,I.Traore,S.Saad,Detectingopinionspamsandfakenews
[23]F.B.Mahmud,M.M.S.Rayhan,M.H.Shuvo,I.Sadia,M.K.Morol,Acom-usingtextclassification,Secur.Privacy1(1)(2018)e9.
    parativeanalysisofgraphneuralnetworksandcommonlyusedmachine[56]K.Nakamura,S.Levy,W.Y.Wang,R/fakeddit:Anewmultimodalbench-
    learningalgorithmsonfakenewsdetection,in:20227thInternationalmarkdatasetforfine-grainedfakenewsdetection,2019,arXivpreprint
    ConferenceonDataScienceandMachineLearningApplications,CDMA,arXiv:1911.03854.
    IEEE,2022,pp.97–102.                          [57]W.Y.Wang,"Liar,liarpantsonfire":Anewbenchmarkdatasetforfake
[24]K.Shu,A.Sliva,S.Wang,J.Tang,H.Liu,Fakenewsdetectiononsocialnewsdetection,2017,arXivpreprintarXiv:1705.00648.
    media:Adataminingperspective,ACMSIGKDDExplor.Newsl.19(1)[58]K.Shu,D.Mahudeswaran,S.Wang,D.Lee,H.Liu,Fakenewsnet:A
    (2017)22–36.                                     datarepositorywithnewscontent,socialcontext,andspatiotemporal
[25]D.M.Lazer,M.A.Baum,Y.Benkler,A.J.Berinsky,K.M.Greenhill,F.informationforstudyingfakenewsonsocialmedia,BigData8(3)(2020)
    Menczer,M.J.Metzger,B.Nyhan,G.Pennycook,D.Rothschild,etal.,The171–188.
    scienceoffakenews,Science359(6380)(2018)1094–1096.[59]J.Golbeck,M.Mauriello,B.Auxier,K.H.Bhanushali,C.Bonk,M.A.
[26]T.Quandt,L.Frischlich,S.Boberg,T.Schatto-Eckrodt,Fakenews,in:TheBouzaghrane,C.Buntain,R.Chanduka,P.Cheakalos,J.B.Everett,etal.,
    InternationalEncyclopediaofJournalismStudies,JohnWiley&Sons,Inc.Fakenewsvssatire:Adatasetandanalysis,in:Proceedingsofthe10th
    Hoboken,NJ,USA,2019,pp.1–6.                      ACMConferenceonWebScience,2018,pp.17–21.
[27]X.Zhou,R.Zafarani,Fakenews:Asurveyofresearch,detectionmethods,[60]F.K.A.Salem,R.AlFeel,S.Elbassuoni,M.Jaber,M.Farah,Fa-kes:Afake
    andopportunities,2,2018,arXivpreprintarXiv:1812.00315.newsdatasetaroundtheSyrianwar,in:ProceedingsoftheInternational
[28]N.Ruchansky,S.Seo,Y.Liu,Csi:AhybriddeepmodelforfakenewsAAAIConferenceonWebandSocialMedia,Vol.13,2019,pp.573–582.
    detection,in:Proceedingsofthe2017ACMonConferenceonInformation[61]A.Pathak,R.K.Srihari,BREAKING!Presentingfakenewscorpusforauto-
    andKnowledgeManagement,2017,pp.797–806.matedfactchecking,in:Proceedingsofthe57thAnnualMeetingofthe
[29]K.Shu,L.Cui,S.Wang,D.Lee,H.Liu,Defend:Explainablefakenewsde-AssociationforComputationalLinguistics:StudentResearchWorkshop,
    tection,in:Proceedingsofthe25thACMSIGKDDInternationalConference2019,pp.357–362.
    onKnowledgeDiscovery&DataMining,2019,pp.395–405.[62]J.Thorne,A.Vlachos,C.Christodoulopoulos,A.Mittal,Fever:Alarge-
[30]M.Cinelli,G.D.F.Morales,A.Galeazzi,W.Quattrociocchi,M.Starnini,Thescaledatasetforfactextractionandverification,2018,arXivpreprint
    echochambereffectonsocialmedia,Proc.Natl.Acad.Sci.118(9)(2021).arXiv:1803.05355.
[31]C.Paul,M.Matthews,TheRussian‘‘firehoseoffalsehood’’propaganda[63]G.K.Shahi,D.Nandini,FakeCovid–Amultilingualcross-domainfactcheck
    model,RandCorporation2(7)(2016)1–10.             newsdatasetforCOVID-19,2020,arXivpreprintarXiv:2006.11343.
[32]M.DelVicario,A.Bessi,F.Zollo,F.Petroni,A.Scala,G.Caldarelli,H.E.[64]T.Mitra,E.Gilbert,Credbank:Alarge-scalesocialmediacorpuswithas-
    Stanley,W.Quattrociocchi,Echochambersintheageofmisinformation,sociatedcredibilityannotations,in:NinthInternationalAAAIConference
    2015,arXivpreprintarXiv:1509.00189.              onWebandSocialMedia,2015.
[33]M.DelVicario,A.Bessi,F.Zollo,F.Petroni,A.Scala,G.Caldarelli,H.E.[65]J.Leskovec,L.Backstrom,J.Kleinberg,Meme-trackingandthedynamics
    Stanley,W.Quattrociocchi,Thespreadingofmisinformationonline,Proc.ofthenewscycle,in:Proceedingsofthe15thACMSIGKDDInternational
                                                     ConferenceonKnowledgeDiscoveryandDataMining,2009,pp.497–506.
    Natl.Acad.Sci.113(3)(2016)554–559.            [66]G.C.Santia,J.R.Williams,Buzzface:AnewsveracitydatasetwithFacebook
[34]M.DelVicario,G.Vivaldo,A.Bessi,F.Zollo,A.Scala,G.Caldarelli,usercommentaryandegos,in:TwelfthInternationalAAAIConferenceon
    W.Quattrociocchi,Echochambers:EmotionalcontagionandgroupWebandSocialMedia,2018.
    polarizationonFacebook,Sci.Rep.6(1)(2016)1–12.[67]E.Tacchini,G.Ballarin,M.L.DellaVedova,S.Moret,L.deAlfaro,Some
[35]J.L.Egelhofer,S.Lecheler,Fakenewsasatwo-dimensionalphenomenon:likeithoax:Automatedfakenewsdetectioninsocialnetworks,2017,
    Aframeworkandresearchagenda,Ann.Int.Commun.Assoc.43(2)arXivpreprintarXiv:1704.07506.
    (2019)97–116.                                 [68]M.DeDomenico,A.Lima,P.Mougel,M.Musolesi,Theanatomyofa
[36]V.Bakir,A.McStay,Fakenewsandtheeconomyofemotions:Problems,scientificrumor,Sci.Rep.3(2013)2980.
    causes,solutions,Digit.Journalism6(2)(2018)154–175.[69]T.Khan,A.Michalas,Trustandbelieve-shouldwe?Evaluatingthe
[37]B.Franklin,B.McNair,FakeNews:Falsehood,FabricationandFantasyintrustworthinessofTwitterusers,in:2020IEEE19thInternationalConfer-
    Journalism,Routledge,2017.                       enceonTrust,SecurityandPrivacyinComputingandCommunications,
[38]E.C.TandocJr.,Z.W.Lim,R.Ling,Defining‘‘fakenews’’atypologyofTrustCom,IEEE,2020,pp.1791–1800.
    scholarlydefinitions,Digit.Journalism6(2)(2018)137–153.[70]R.Barbado,O.Araque,C.A.Iglesias,Aframeworkforfakereviewdetection
[39]C.Wardle,Fakenews.It’scomplicated,FirstDraft16(2017)1–11.inonlineconsumerelectronicsretailers,Inf.Process.Manage.56(4)
[40]K.Stahl,Fakenewsdetectioninsocialmedia,Calif.StateUniv.Stanislaus(2019)1234–1244.
    6(2018)4–15.                                  [71]A.Zubiaga,A.Aker,K.Bontcheva,M.Liakata,R.Procter,Detectionand
[41]N.R.Hanson,Anoteonstatementsoffact,Analysis13(1)(1952)24.resolutionofrumoursinsocialmedia:Asurvey,2017,Preprint,arXiv
[42]F.Pierri,S.Ceri,Falsenewsonsocialmedia:Adata-drivensurvey,ACM1704.
    SigmodRecord48(2)(2019)18–27.                 [72]A.Vlachos,S.Riedel,Factchecking:Taskdefinitionanddatasetcon-
[43]S.Vosoughi,D.Roy,S.Aral,Thespreadoftrueandfalsenewsonline,struction,in:ProceedingsoftheACL2014WorkshoponLanguage
    Science359(6380)(2018)1146–1151.                 TechnologiesandComputationalSocialScience,2014,pp.18–22.
                                               24

H.T. Phan, N.T. Nguyen and D. Hwang                                                                                                                                                                                            Applied Soft Computing 139 (2023) 110235
[73]W.Ferreira,A.Vlachos,Emergent:Anoveldata-setforstanceclassifica-[100]Z.Jin,J.Cao,Y.Zhang,J.Zhou,Q.Tian,Novelvisualandstatisticalimage
    tion,in:Proceedingsofthe2016ConferenceoftheNorthAmericanChap-featuresformicroblogsnewsverification,IEEETrans.Multimed.19(3)
    teroftheAssociationforComputationalLinguistics:HumanLanguage(2016)598–608.
    Technologies,2016,pp.1163–1168.              [101]M.Potthast,J.Kiesel,K.Reinartz,J.Bevendorff,B.Stein,Astylometric
[74]B.Horne,S.Adali,Thisjustin:Fakenewspacksalotintitle,usessimpler,inquiryintohyperpartisanandfakenews,2017,arXivpreprintarXiv:
    repetitivecontentintextbody,moresimilartosatirethanrealnews,1702.05638.
    in:ProceedingsoftheInternationalAAAIConferenceonWebandSocial[102]C.Castillo,M.Mendoza,B.Poblete,Informationcredibilityontwitter,in:
    Media,vol.11,(no.1)2017.                         Proceedingsofthe20thInternationalConferenceonWorldWideWeb,
[75]C.Burfoot,T.Baldwin,Automaticsatiredetection:Areyouhavinga2011,pp.675–684.
    laugh?in:ProceedingsoftheACL-IJCNLP2009ConferenceShortPapers,[103]S.Hamidian,M.T.Diab,RumordetectionandclassificationforTwitter
    2009,pp.161–164.                                 data,2019,arXivpreprintarXiv:1912.08926.
[76]F.TorabiAsr,M.Taboada,Bigdataandqualitydataforfakenewsand[104]X.Hu,J.Tang,H.Liu,Onlinesocialspammerdetection,in:Proceedings
    misinformationdetection,BigDataSoc.6(1)(2019)2053951719843310.oftheAAAIConferenceonArtificialIntelligence,vol.28,(no.1)2014.
[77]M.Ott,Y.Choi,C.Cardie,J.T.Hancock,Findingdeceptiveopinionspam[105]S.Kwon,M.Cha,K.Jung,W.Chen,Y.Wang,Prominentfeaturesofrumor
    byanystretchoftheimagination,2011,arXivpreprintpropagationinonlinesocialmedia,in:2013IEEE13thInternationalarXiv:1107.4557.
[78]B.Riedel,I.Augenstein,G.P.Spithourakis,S.Riedel,Asimplebuttough-ConferenceonDataMining,IEEE,2013,pp.1103–1108.
    to-beatbaselineforthefakenewschallengestancedetectiontask,2017,[106]Z.Jin,J.Cao,Y.Zhang,J.Luo,Newsverificationbyexploitingconflicting
    arXivpreprintarXiv:1707.03264.                   socialviewpointsinmicroblogs,in:ProceedingsoftheAAAIConference
[79]P.S.Dutta,M.Das,S.Biswas,M.Bora,S.S.Saikia,Fakenewsprediction:onArtificialIntelligence,vol.30,(no.1)2016.
    Asurvey,Int.J.Sci.Eng.Sci.3(3)(2019)1–3.[107]A.Bondielli,F.Marcelloni,Asurveyonfakenewsandrumourdetection
[80]H.Rashkin,E.Choi,J.Y.Jang,S.Volkova,Y.Choi,Truthofvaryingtechniques,Inform.Sci.497(2019)38–55.
    shades:Analyzinglanguageinfakenewsandpoliticalfact-checking,in:[108]A.Silva,L.Luo,S.Karunasekera,C.Leckie,Embracingdomaindifferences
    Proceedingsofthe2017ConferenceonEmpiricalMethodsinNaturalinfakenews:Cross-domainfakenewsdetectionusingmulti-modaldata,
    LanguageProcessing,2017,pp.2931–2937.            in:ProceedingsoftheAAAIConferenceonArtificialIntelligence,vol.35,
[81]A.Barrón-Cedeno,I.Jaradat,G.DaSanMartino,P.Nakov,Proppy:(no.1)2021,pp.557–565.
    Organizingthenewsbasedontheirpropagandisticcontent,Inf.Process.[109]T.R.Hoens,R.Polikar,N.V.Chawla,Learningfromstreamingdatawith
    Manage.56(5)(2019)1849–1864.                     conceptdriftandimbalance:Anoverview,ProgressArtif.Intell.1(1)
[82]J.Nørregaard,B.D.Horne,S.Adalı,Nela-gt-2018:Alargemulti-labelled(2012)89–101.
    newsdatasetforthestudyofmisinformationinnewsarticles,in:[110]J.Devlin,M.-W.Chang,K.Lee,K.Toutanova,Bert:Pre-trainingof
    ProceedingsoftheInternationalAAAIConferenceonWebandSocialdeepbidirectionaltransformersforlanguageunderstanding,2018,arXiv
    Media,vol.13,2019,pp.630–638.                    preprintarXiv:1810.04805.
[83]Y.Jang,C.-H.Park,Y.-S.Seo,Fakenewsanalysismodelingusingquote[111]M.E.Peters,M.Neumann,M.Iyyer,M.Gardner,C.Clark,K.Lee,
    retweet,Electronics8(12)(2019)1377.              L.Zettlemoyer,Deepcontextualizedwordrepresentations,2018,CoRR
[84]O.Papadopoulou,M.Zampoglou,S.Papadopoulos,I.Kompatsiaris,Aabs/1802.05365,arXivpreprintarXiv:1802.05365,1802.
    corpusofdebunkedandverifieduser-generatedvideos,OnlineInf.Rev.[112]T.Mikolov,E.Grave,P.Bojanowski,C.Puhrsch,A.Joulin,Advances
    (2019).                                          inpre-trainingdistributedwordrepresentations,2017,arXivpreprint
[85]C.Boididou,S.Papadopoulos,M.Zampoglou,L.Apostolidis,O.Pa-arXiv:1712.09405.
    padopoulou,Y.Kompatsiaris,Detectionandvisualizationofmisleading[113]E.Grave,P.Bojanowski,P.Gupta,A.Joulin,T.Mikolov,Learningword
    contentonTwitter,Int.J.MultimediaInf.Retr.7(1)(2018)71–86.vectorsfor157languages,2018,arXivpreprintarXiv:1802.06893.
[86]H.Jwa,D.Oh,K.Park,J.M.Kang,H.Lim,Exbake:Automaticfakenews[114]J.Pennington,R.Socher,C.D.Manning,Glove:Globalvectorsforword
    detectionmodelbasedonbidirectionalencoderrepresentationsfromrepresentation,in:Proceedingsofthe2014ConferenceonEmpirical
    transformers(bert),Appl.Sci.9(19)(2019)4062.MethodsinNaturalLanguageProcessing,EMNLP,2014,pp.1532–1543.
[87]N.T.Tam,M.Weidlich,B.Zheng,H.Yin,N.Q.V.Hung,B.Stantic,From[115]B.Koloski,T.S.Perdih,M.Robnik-Šikonja,S.Pollak,B.Škrlj,Knowledge
    anomalydetectiontorumourdetectionusingdatastreamsofsocialgraphinformedfakenewsclassificationviaheterogeneousrepresentation
    platforms,Proc.VLDBEndow.12(9)(2019)1016–1029.ensembles,Neurocomputing(2022).
[88]E.Dai,Y.Sun,S.Wang,Gingercannotcurecancer:Battlingfakehealth[116]Z.Sun,Z.-H.Deng,J.-Y.Nie,J.Tang,Rotate:Knowledgegraphembedding
    newswithacomprehensivedatarepository,in:Proceedingsofthebyrelationalrotationincomplexspace,2019,arXivpreprintarXiv:
    InternationalAAAIConferenceonWebandSocialMedia,vol.14,2020,1902.10197.
    pp.853–862.                                  [117]S.Zhang,Y.Tay,L.Yao,Q.Liu,Quaternionknowledgegraphembeddings,
[89]J.A.Hill,S.Agewall,A.Baranchuk,G.W.Booz,J.S.Borer,P.G.Camici,P.-S.Adv.NeuralInf.Process.Syst.32(2019).
    Chen,A.F.Dominiczak,Ç.Erol,C.L.Grines,etal.,Medicalmisinformation:[118]T.Trouillon,J.Welbl,S.Riedel,É.Gaussier,G.Bouchard,Complex
    Vetthemessage!2019.                              embeddingsforsimplelinkprediction,in:InternationalConferenceon
[90]M.Potthast,S.Köpsel,B.Stein,M.Hagen,Clickbaitdetection,in:EuropeanMachineLearning,PMLR,2016,pp.2071–2080.
    ConferenceonInformationRetrieval,Springer,2016,pp.810–817.[119]V.Pérez-Rosas,B.Kleinberg,A.Lefevre,R.Mihalcea,Automaticdetection
[91]V.L.Rubin,N.Conroy,Y.Chen,S.Cornwell,Fakenewsortruth?Usingoffakenews,2017,arXivpreprintarXiv:1708.07104.
    satiricalcuestodetectpotentiallymisleadingnews,in:Proceedingsofthe[120]K.Cho,B.VanMerriënboer,C.Gulcehre,D.Bahdanau,F.Bougares,H.
    SecondWorkshoponComputationalApproachesToDeceptionDetection,Schwenk,Y.Bengio,LearningphraserepresentationsusingRNNencoder-
    2016,pp.7–17.                                    decoderforstatisticalmachinetranslation,2014,arXivpreprintarXiv:
[92]S.Vosoughi,M.N.Mohsenvand,D.Roy,Rumorgauge:Predictingthe1406.1078.
                                                 [121]A.Vaswani,N.Shazeer,N.Parmar,J.Uszkoreit,L.Jones,A.N.Gomez,Ł.
    veracityofrumorsonTwitter,ACMTrans.Knowl.Discov.Data(TKDD)Kaiser,I.Polosukhin,Attentionisallyouneed,Adv.NeuralInf.Process.
    11(4)(2017)1–36.                                 Syst.30(2017).
[93]Y.Qin,D.Wurzer,V.Lavrenko,C.Tang,Spottingrumorsvianovelty[122]A.Glazkova,M.Glazkov,T.Trifonov,g2tmnatConstraint@Aaai2021:
    detection,2016,arXivpreprintarXiv:1611.06322.    ExploitingCT-BERTandEnsemblingLearningforCOVID-19FakeNews
[94]V.Qazvinian,E.Rosengren,D.Radev,Q.Mei,Rumorhasit:IdentifyingDetection,Springer,2021,pp.116–127.
    misinformationinmicroblogs,in:Proceedingsofthe2011Conferenceon[123]A.Li,Z.Qin,R.Liu,Y.Yang,D.Li,Spamreviewdetectionwithgraph
    EmpiricalMethodsinNaturalLanguageProcessing,2011,pp.1589–1599.convolutionalnetworks,in:Proceedingsofthe28thACMInternational
[95]A.Zubiaga,M.Liakata,R.Procter,LearningreportingdynamicsduringConferenceonInformationandKnowledgeManagement,2019,pp.
    breakingnewsforrumourdetectioninsocialmedia,2016,arXivpreprint2703–2711.
    arXiv:1610.07363.                            [124]H.Lin,X.Zhang,X.Fu,Agraphconvolutionalencoderanddecodermodel
[96]E.J.Briscoe,D.S.Appling,H.Hayes,Cuestodeceptioninsocialmediacom-forrumordetection,in:2020IEEE7thInternationalConferenceonData
    munications,in:201447thHawaiiInternationalConferenceonSystemScienceandAdvancedAnalytics,DSAA,IEEE,2020,pp.300–306.
    Sciences,IEEE,2014,pp.1435–1443.             [125]D.T.Vu,J.J.Jung,Rumordetectionbypropagationembeddingbasedon
[97]A.Y.Chua,S.Banerjee,Linguisticpredictorsofrumorveracityonthein-graphconvolutionalnetwork,Int.J.Comput.Intell.Syst.14(1)(2021)
    ternet,in:ProceedingsoftheInternationalMultiConferenceofEngineers1053–1065.
    andComputerScientists,vol.1,2016,pp.387–391.[126]K.Simonyan,A.Zisserman,Verydeepconvolutionalnetworksfor
[98]J.Ito,J.Song,H.Toda,Y.Koike,S.Oyama,Assessmentoftweetcredibilitylarge-scaleimagerecognition,2014,arXivpreprintarXiv:1409.1556.
    withLDAfeatures,in:Proceedingsofthe24thInternationalConference[127]M.Nickel,K.Murphy,V.Tresp,E.Gabrilovich,Areviewofrelational
    onWorldWideWeb,2015,pp.953–958.                  machinelearningforknowledgegraphs,Proc.IEEE104(1)(2015)11–33.
[99]J.Ma,W.Gao,P.Mitra,S.Kwon,B.J.Jansen,K.-F.Wong,M.Cha,Detecting[128]A.L.Ginsca,A.Popescu,M.Lupu,Credibilityininformationretrieval,
    rumorsfrommicroblogswithrecurrentneuralnetworks,2016.Found.TrendsInf.Retr.9(5)(2015)355–475.
                                               25

H.T. Phan, N.T. Nguyen and D. Hwang                                                                                                                                                                                            Applied Soft Computing 139 (2023) 110235
[129]O.Etzioni,M.Banko,S.Soderland,D.S.Weld,Openinformationextraction[155]N.Tuan,P.Minh,FakeNewsdetectionusingpre-trainedlanguagemodels
    fromtheweb,Commun.ACM51(12)(2008)68–74.andgraphconvolutionalnetworks,in:MultimediaEvaluationBenchmark
[130]A.Magdy,N.Wanas,Web-basedstatisticalfactcheckingoftextualWorkshop2020,MediaEval2020,2020.
    documents,in:Proceedingsofthe2ndInternationalWorkshoponSearch[156]Z.Ke,Z.Li,C.Zhou,J.Sheng,W.Silamu,Q.Guo,Rumordetectiononsocial
    andMiningUser-GeneratedContents,2010,pp.103–110.mediaviafusedsemanticinformationandapropagationheterogeneous
[131]L.DeAlfaro,V.Polychronopoulos,M.Shavlovsky,Reliableaggregationofgraph,Symmetry12(11)(2020)1806.
    Booleancrowdsourcedtasks,in:ProceedingsoftheAAAIConferenceon[157]B.D.Horne,J.Nørregaard,S.Adalı,Differentspiralsofsameness:Astudy
    HumanComputationandCrowdsourcing,vol.3,(no.1)2015.ofcontentsharinginmainstreamandalternativemedia,in:Proceedings
[132]Y.Chen,N.K.Conroy,V.L.Rubin,Newsinanonlineworld:TheneedforoftheInternationalAAAIConferenceonWebandSocialMedia,vol.13,
    an‘‘automaticcrapdetector’’,Proc.Assoc.Inf.Sci.Technol.52(1)(2015)2019,pp.257–266.
    1–4.                                          [158]G.Bachi,M.Coscia,A.Monreale,F.Giannotti,Classifyingtrust/distrust
[133]V.L.Rubin,N.J.Conroy,Y.Chen,Towardsnewsverification:Deceptionde-relationshipsinonlinesocialnetworks,in:2012InternationalConference
    tectionmethodsfornewsdiscourse,in:HawaiiInternationalConferenceonPrivacy,Security,RiskandTrustand2012InternationalConferneceon
    onSystemSciences,2015,pp.5–8.                    SocialComputing,IEEE,2012,pp.552–557.
[134]Z.Wei,J.Chen,W.Gao,B.Li,L.Zhou,Y.He,K.-F.Wong,Anempirical[159]G.Wang,X.Zhang,S.Tang,H.Zheng,B.Y.Zhao,Unsupervisedclickstream
    studyonuncertaintyidentificationinsocialmediacontext,in:Socialclusteringforuserbehavioranalysis,in:Proceedingsofthe2016CHI
    MediaContentAnalysis:NaturalLanguageProcessingandbeyond,WorldConferenceonHumanFactorsinComputingSystems,2016,pp.225–236.
    Scientific,2018,pp.79–88.                     [160]W.L.Hamilton,Graphrepresentationlearning,SynthesisLect.Artif.Intell.
[135]B.Shi,T.Weninger,Factcheckinginheterogeneousinformationnetworks,Mach.Learn.14(3)(2020)1–159.
    in:Proceedingsofthe25thInternationalConferenceCompanionon[161]J.Bruna,W.Zaremba,A.Szlam,Y.LeCun,Spectralnetworksandlocally
    WorldWideWeb,2016,pp.101–102.                    connectednetworksongraphs,2013,arXivpreprintarXiv:1312.6203.
[136]K.Shu,D.Mahudeswaran,S.Wang,D.Lee,H.Liu,Fakenewsnet:A[162]M.Niepert,M.Ahmed,K.Kutzkov,Learningconvolutionalneuralnet-
    datarepositorywithnewscontent,socialcontextandspatialtemporalworksforgraphs,in:InternationalConferenceonMachineLearning,
    informationforstudyingfakenewsonsocialmedia,2018,arXivpreprintPMLR,2016,pp.2014–2023.
    arXiv:1809.01286.                             [163]R.Ying,J.You,C.Morris,X.Ren,W.L.Hamilton,J.Leskovec,Hierarchical
[137]N.Sitaula,C.K.Mohan,J.Grygiel,X.Zhou,R.Zafarani,Credibility-basedgraphrepresentationlearningwithdifferentiablepooling,2018,arXiv
    fakenewsdetection,in:Disinformation,Misinformation,andFakeNewspreprintarXiv:1806.08804.
    inSocialMedia,Springer,2020,pp.163–182.[164]T.N. Kipf, M. Welling, Semi-supervised classification with graph
[138]D.Mocanu,L.Rossi,Q.Zhang,M.Karsai,W.Quattrociocchi,Collectiveconvolutionalnetworks,2016,arXivpreprintarXiv:1609.02907.
    attentionintheageof(mis)information,Comput.Hum.Behav.51(2015)[165]X.Jiang,R.Zhu,S.Li,P.Ji,Co-embeddingofnodesandedgeswithgraph
    1198–1204.                                       neuralnetworks,IEEETrans.PatternAnal.Mach.Intell.(2020).
[139]M.Tambuscio,G.Ruffo,A.Flammini,F.Menczer,Fact-checkingeffecton[166]A.Sperduti,A.Starita,Supervisedneuralnetworksfortheclassification
    viralhoaxes:Amodelofmisinformationspreadinsocialnetworks,in:ofstructures,IEEETrans.NeuralNetw.8(3)(1997)714–735.
    Proceedingsofthe24thInternationalConferenceonWorldWideWeb,[167]M.Gori,G.Monfardini,F.Scarselli,Anewmodelforlearningingraph
    2015,pp.977–982.                                 domains,in:Proceedings.2005IEEEInternationalJointConferenceon
[140]J.Ma,W.Gao,K.-F.Wong,DetectRumorsinMicroblogPostsUsingNeuralNetworks,2005,Vol.2,IEEE,2005,pp.729–734.
    PropagationStructureViaKernelLearning,AssociationforComputational[168]F.Scarselli,M.Gori,A.C.Tsoi,M.Hagenbuchner,G.Monfardini,Thegraph
    Linguistics,2017.                                neuralnetworkmodel,IEEETrans.NeuralNetw.20(1)(2008)61–80.
[141]L.Wu,H.Liu,Tracingfake-newsfootprints:Characterizingsocialmedia[169]C.Gallicchio,A.Micheli,Graphechostatenetworks,in:The2010
    messagesbyhowtheypropagate,in:ProceedingsoftheEleventhACMInternationalJointConferenceonNeuralNetworks,IJCNN,IEEE,2010,
    InternationalConferenceonWebSearchandDataMining,2018,pp.pp.1–8.
    637–645.                                      [170]T.N.Kipf,M.Welling,Variationalgraphauto-encoders,2016,arXiv
[142]Y.Liu,Y.-F.B.Wu,EarlydetectionoffakenewsonsocialmediapreprintarXiv:1611.07308.
    throughpropagationpathclassificationwithrecurrentandconvolutional[171]Y.Wang,B.Xu,M.Kwak,X.Zeng,Asimpletrainingstrategyforgraph
    networks,in:Thirty-SecondAAAIConferenceonArtificialIntelligence,autoencoder,in:Proceedingsofthe202012thInternationalConference
    2018.                                            onMachineLearningandComputing,2020,pp.341–345.
[143]K.Shu,D.Mahudeswaran,S.Wang,H.Liu,Hierarchicalpropagation[172]Y.Li,R.Yu,C.Shahabi,Y.Liu,Diffusionconvolutionalrecurrentneural
    networksforfakenewsdetection:Investigationandexploitation,in:network:Data-driventrafficforecasting,2017,arXivpreprintarXiv:1707.
    ProceedingsoftheInternationalAAAIConferenceonWebandSocial01926.
    Media,vol.14,2020,pp.626–637.                 [173]Y.Seo,M.Defferrard,P.Vandergheynst,X.Bresson,Structuredsequence
[144]K.Wu,S.Yang,K.Q.Zhu,Falserumorsdetectiononsinaweibobymodelingwithgraphconvolutionalrecurrentnetworks,in:Interna-
    propagationstructures,in:2015IEEE31stInternationalConferenceontionalConferenceonNeuralInformationProcessing,Springer,2018,pp.
    DataEngineering,IEEE,2015,pp.651–662.            362–373.
[145]X.Zhou,R.Zafarani,Network-basedfakenewsdetection:Apattern-[174]J.Zhang,X.Shi,J.Xie,H.Ma,I.King,D.-Y.Yeung,Gaan:Gatedattention
    drivenapproach,ACMSIGKDDExplor.Newsl.21(2)(2019)48–60.networksforlearningonlargeandspatiotemporalgraphs,2018,arXiv
[146]S.Alonso-Bartolome,I.Segura-Bedmar,Multimodalfakenewsdetection,preprintarXiv:1803.07294.
    2021,arXivpreprintarXiv:2112.04831.           [175]Z.Wu,S.Pan,G.Long,J.Jiang,C.Zhang,Graphwavenetfordeep
[147]B.Malhotra,D.K.Vishwakarma,Classificationofpropagationpathand
    tweetsforrumordetectionusinggraphicalconvolutionalnetworksspatial-temporalgraphmodeling,2019,arXivpreprintarXiv:1906.00121.
    andtransformerbasedencodings,in:2020IEEESixthInternational[176]S.Yan,Y.Xiong,D.Lin,Spatialtemporalgraphconvolutionalnetworks
    ConferenceonMultimediaBigData,BigMM,IEEE,2020,pp.183–190.forskeleton-basedactionrecognition,in:Thirty-SecondAAAIConference
[148]X.Zhou,J.Wu,R.Zafarani,SAFE:Similarity-awaremulti-modalfakenewsonArtificialIntelligence,2018.
    detection,in:Pacific-AsiaConferenceonKnowledgeDiscoveryandData[177]B.Yu,H.Yin,Z.Zhu,Spatio-temporalgraphconvolutionalnetworks:
    Mining,Springer,2020,pp.354–367.                 Adeeplearningframeworkfortrafficforecasting,2017,arXivpreprint
[149]L.Shang,Y.Zhang,D.Zhang,D.Wang,Fauxward:AgraphneuralnetworkarXiv:1709.04875.
    approachtofauxtographydetectionusingsocialmediacomments,Soc.[178]P.Veličković,G.Cucurull,A.Casanova,A.Romero,P.Lio,Y.Bengio,Graph
    Netw.Anal.Min.10(1)(2020)1–16.                   attentionnetworks,2017,arXivpreprintarXiv:1710.10903.
[150]A.Paraschiv,G.-E.Zaharia,D.-C.Cercel,M.Dascalu,Graphconvolutional[179]K.K.Thekumparampil,C.Wang,S.Oh,L.-J.Li,Attention-basedgraph
    networksappliedtofakenews:Coronavirusand5Gconspiracy,Univ.neuralnetworkforsemi-supervisedlearning,2018,arXivpreprintarXiv:
    Politeh.BucharestSci.Bull.Ser.C-Electr.Eng.Comput.Sci.(2021)71–82.1803.03735.
[151]L.Zhang,J.Li,B.Zhou,Y.Jia,RumordetectionbasedonSAGNN:[180]J.B.Lee,R.Rossi,X.Kong,Graphclassificationusingstructuralattention,
    Simplifiedaggregationgraphneuralnetworks,Mach.Learn.Knowl.Extr.in:Proceedingsofthe24thACMSIGKDDInternationalConferenceon
    3(1)(2021)84–94.                                 KnowledgeDiscovery&DataMining,2018,pp.1666–1674.
[152]J.Ma,W.Gao,K.-F.Wong,RumorDetectiononTwitterwithTree-[181]A.Liberati,D.G.Altman,J.Tetzlaff,C.Mulrow,P.C.Gøtzsche,J.P.Ioannidis,
    StructuredRecursiveNeuralNetworks,AssociationforComputationalM.Clarke,P.J.Devereaux,J.Kleijnen,D.Moher,ThePRISMAstatementfor
    Linguistics,2018.                                reportingsystematicreviewsandmeta-analysesofstudiesthatevaluate
[153]M.Meyers,G.Weiss,G.Spanakis,FakenewsdetectiononTwitterusinghealthcareinterventions:Explanationandelaboration,J.Clin.Epidemiol.
    propagationstructures,in:MultidisciplinaryInternationalSymposiumon62(10)(2009)e1–e34.
    DisinformationinOpenOnlineMedia,Springer,2020,pp.138–158.[182]S.Afroz,M.Brennan,R.Greenstadt,Detectinghoaxes,frauds,anddecep-
[154]N.Bai,F.Meng,X.Rui,Z.Wang,Rumourdetectionbasedongraphtioninwritingstyleonline,in:2012IEEESymposiumonSecurityand
    convolutionalneuralnet,IEEEAccess9(2021)21686–21693.Privacy,IEEE,2012,pp.461–475.
                                               26

H.T. Phan, N.T. Nguyen and D. Hwang                                                                                                                                                                                            Applied Soft Computing 139 (2023) 110235
[183]J.Lafferty,A.McCallum,F.C.Pereira,Conditionalrandomfields:[202]K.Lee,B.D.Eoff,J.Caverlee,Sevenmonthswiththedevils:Along-
    Probabilisticmodelsforsegmentingandlabelingsequencedata,2001.termstudyofcontentpollutersontwitter,in:FifthInternationalAAAI
[184]S.Vosoughi,AutomaticDetectionandVerificationofRumorsonTwitterConferenceonWeblogsandSocialMedia,2011.
    (Ph.D.thesis),MassachusettsInstituteofTechnology,2015.[203]C.Yang,R.Harkreader,J.Zhang,S.Shin,G.Gu,Analyzingspammers’social
[185]O.Ajao,D.Bhowmik,S.Zargari,Fakenewsidentificationontwitterwithnetworksforfunandprofit:Acasestudyofcybercriminalecosystemon
    hybridCNNandRNNmodels,in:Proceedingsofthe9thInternationalTwitter,in:Proceedingsofthe21stInternationalConferenceonWorld
    ConferenceonSocialMediaandSociety,2018,pp.226–230.WideWeb,2012,pp.71–80.
[186]A.Jacovi,O.S.Shalom,Y.Goldberg,Understandingconvolutionalneural[204]S.Lotfi,M.Mirzarezaee,M.Hosseinzadeh,V.Seydi,Detectionofrumor
    networksfortextclassification,2018,arXivpreprintconversationsinTwitterusinggraphconvolutionalnetworks,Appl.Intell.arXiv:1809.08037.
[187]S.Volkova,K.Shaffer,J.Y.Jang,N.Hodas,Separatingfactsfromfiction:51(7)(2021)4774–4787.
    Linguisticmodelstoclassifysuspiciousandtrustednewspostsontwitter,[205]C.Data61,Stellargraphmachinelearninglibrary,2018,PublicationTitle:
    in:Proceedingsofthe55thAnnualMeetingoftheAssociationforGitHubRepository.GitHub.
    ComputationalLinguistics(Volume2:ShortPapers),2017,pp.647–653.[206]N.J.Vickers,Animalcommunication:Wheni’mcallingyou,willyou
[188]F.Yu,Q.Liu,S.Wu,L.Wang,T.Tan,etal.,Aconvolutionalapproachforanswertoo?Curr.Biol.27(14)(2017)R713–R715.
    misinformationidentification,in:IJCAI,2017,pp.3901–3907.
[189]A.Zubiaga,M.Liakata,R.Procter,G.WongSakHoi,P.Tolmie,Analysing[207]Y.Liu,M.Ott,N.Goyal,J.Du,M.Joshi,D.Chen,O.Levy,M.Lewis,L.
    howpeopleorienttoandspreadrumoursinsocialmediabylookingatZettlemoyer,V.Stoyanov,Roberta:Arobustlyoptimizedbertpretraining
    conversationalthreads,PLoSOne11(3)(2016)e0150989.approach,2019,arXivpreprintarXiv:1907.11692.
[190]G.Hu,Y.Ding,S.Qi,X.Wang,Q.Liao,Multi-depthgraphconvolutional[208]T.Pires,E.Schlinger,D.Garrette,Howmultilingualismultilingual
    networksforfakenewsdetection,in:CCFInternationalConferenceonBERT?2019,arXivpreprintarXiv:1906.01502.
    NaturalLanguageProcessingandChineseComputing,Springer,2019,pp.[209]G.Klambauer,T.Unterthiner,A.Mayr,S.Hochreiter,Self-normalizing
    698–710.                                          neuralnetworks,in:Proceedingsofthe31stInternationalConferenceon
[191]C.Li,D.Goldwasser,Encodingsocialinformationwithgraphcon-NeuralInformationProcessingSystems,2017,pp.972–981.
    volutionalnetworksforpoliticalperspectivedetectioninnewsmedia,[210]Y.Zhang,B.Wallace,Asensitivityanalysisof(andpractitioners’guide
    in:Proceedingsofthe57thAnnualMeetingoftheAssociationforto)convolutionalneuralnetworksforsentenceclassification,2015,arXiv
    ComputationalLinguistics,2019,pp.2594–2604.preprintarXiv:1510.03820.
[192]A.Benamira,B.Devillers,E.Lesot,A.K.Ray,M.Saadi,F.D.Malliaros,[211]W.L.Hamilton,R.Ying,J.Leskovec,Inductiverepresentationlearningon
    Semi-supervisedlearningandgraphneuralnetworksforfakenewslargegraphs,in:Proceedingsofthe31stInternationalConferenceon
    detection,in:2019IEEE/ACMInternationalConferenceonAdvancesinNeuralInformationProcessingSystems,2017,pp.1025–1035.
    SocialNetworksAnalysisandMining,ASONAM,IEEE,2019,pp.568–569.[212]Q.Xu,F.Shen,L.Liu,H.T.Shen,Graphcar:Content-awaremultimedia
[193]A.Hamid,N.Shiekh,N.Said,K.Ahmad,A.Gul,L.Hassan,A.Al-Fuqaha,recommendationwithgraphautoencoder,in:The41stInternationalACM
    FakenewsdetectioninsocialmediausinggraphneuralnetworksandnlpSIGIRConferenceonResearch&DevelopmentinInformationRetrieval,
    techniques:ACOVID-19use-case,2020,arXivpreprintarXiv:2012.07517.
[194]Z.Pehlivan,Onthepursuitoffakenews:Fromgraphconvolutional2018,pp.981–984.
    networkstotimeseries,in:MultimediaEvaluationBenchmarkWorkshop[213]R.v.d.Berg,T.N.Kipf,M.Welling,Graphconvolutionalmatrixcompletion,
    2020,MediaEval2020,2020.                          2017,arXivpreprintarXiv:1706.02263.
[195]G.-A.Vlad,G.-E.Zaharia,D.-C.Cercel,M.Dascalu,UPB@DANKMEMES:[214]F.J.García-Ull,DeepFakes:Thenextchallengeinfakenewsdetection,
    Italianmemesanalysis-employingvisualmodelsandgraphconvolutionalAnàlisi:QuadernsdeComunicacióICultura(64)(2021)0103–120.
    networksformemeidentificationandhatespeechdetection(short[215]K.Shu,S.Wang,H.Liu,Beyondnewscontents:Theroleofsocialcontext
    paper),in:EVALITA,2020.                           forfakenewsdetection,in:ProceedingsoftheTwelfthACMInternational
[196]M.Dong,B.Zheng,N.QuocVietHung,H.Su,G.Li,MultiplerumorConferenceonWebSearchandDataMining,2019,pp.312–320.
    sourcedetectionwithgraphconvolutionalnetworks,in:Proceedingsof[216]U.VonLuxburg,Atutorialonspectralclustering,Stat.Comput.17(4)
    the28thACMInternationalConferenceonInformationandKnowledge(2007)395–416.
    Management,2019,pp.569–578.                   [217]J.Zhou,J.X.Huang,Q.V.Hu,L.He,SK-GCN:Modelingsyntaxand
[197]W.W.Zachary,Aninformationflowmodelforconflictandfissioninsmallknowledgeviagraphconvolutionalnetworkforaspect-levelsentiment
    groups,J.Anthropol.Res.33(4)(1977)452–473.classification,Knowl.-BasedSyst.205(2020)106292.
[198]D.Lusseau,K.Schneider,O.J.Boisseau,P.Haase,E.Slooten,S.M.Dawson,[218]L.Yao,C.Mao,Y.Luo,Graphconvolutionalnetworksfortextclassifica-
    Thebottlenosedolphincommunityofdoubtfulsoundfeaturesalargetion,in:ProceedingsoftheAAAIConferenceonArtificialIntelligence,vol.
    proportionoflong-lastingassociations,Behav.Ecol.Sociobiol.54(4)33,(no.01)2019,pp.7370–7377.
    (2003)396–405.
[199]D.J.Watts,S.H.Strogatz,Collectivedynamicsof‘small-world’networks,[219]K.Bijari,H.Zare,E.Kebriaei,H.Veisi,Leveragingdeepgraph-basedtext
    Nature393(6684)(1998)440–442.                     representationforsentimentpolarityapplications,ExpertSyst.Appl.144
[200]P.M.Gleiser,L.Danon,Communitystructureinjazz,Adv.ComplexSyst.(2020)113090.
    6(04)(2003)565–573.                           [220]M.Zhang,T.Qian,Convolutionoverhierarchicalsyntacticandlexical
[201]Y.Wu,D.Lian,Y.Xu,L.Wu,E.Chen,Graphconvolutionalnetworksgraphsforaspectlevelsentimentanalysis,in:Proceedingsofthe2020
    withmarkovrandomfieldreasoningforsocialspammerdetection,in:ConferenceonEmpiricalMethodsinNaturalLanguageProcessing,EMNLP,
    ProceedingsoftheAAAIConferenceonArtificialIntelligence,vol.34,(no.2020,pp.3540–3549.
    01)2020,pp.1054–1061.                         [221]A.Wissner-Gross,Datasetsoveralgorithms,Edge.Com.Retr.8(2016).
                                               27

