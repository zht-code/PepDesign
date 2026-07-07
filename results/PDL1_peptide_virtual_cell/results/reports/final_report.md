# PD-L1 多肽虚拟细胞五层验证 — 总结报告

## 参考扰动（阳性对照）

在缺少「PD-L1 多肽直接处理」单细胞数据时，可使用 **anti-PD-L1、anti-PD-1 或 PD-L1/TGFβ 双抗（如 Bintrafusp alfa）**
治疗前后样本作为 **功能等效阳性扰动**：其共同生物学后果包含削弱 PD-1/PD-L1 轴、恢复 T 细胞效应程序。
本 pipeline 将该类数据用于构建 **reference transcriptional signature**，并与候选多肽的理化/对接特征在虚拟细胞层做类比评分。

## 本次 global pathway blockade 参考分 (0–1)：0.500

## 候选排序

```
peptide_id                      sequence  binding_score  pathway_blockade_score  blockade_similarity_score  immune_activation_score  toxicity_risk_score  safety_score  final_score     recommendation
    pep_01 PVTVAKKAPVVVPKKKAAVPVVAAPKKKA       0.890226                     0.5                   0.466656                 0.592613             0.439642      0.560358     0.595907    Not recommended
    pep_02 ARRAAARASARPAATPCAGSSASTTTRAS       0.736841                     0.5                   0.324299                 0.490500             0.346997      0.653003     0.353174    Not recommended
    pep_03 VVFGAAGAVAVASEGVATGAGFFAAADGF       0.966518                     0.5                   0.547908                 0.703308             0.264188      0.735812     0.854171   Strong candidate
    pep_04 AALAAEDAAFAAALLATDDAALAAAELLA       0.747704                     0.5                   0.326020                 0.508239             0.352060      0.647940     0.368865    Not recommended
    pep_05 AARRSRRRGASRASGPTRTRRPASGLRSR       0.860531                     0.5                   0.455484                 0.540666             0.585684      0.414316     0.478494    Not recommended
    pep_06 SMTLPPAGPSPAASASPPASSAPSRRSCT       0.750549                     0.5                   0.326818                 0.462242             0.496984      0.503016     0.283923    Not recommended
    pep_07 PVVLDVVLLDVVLVDVVALLDLVVALPDE       0.913908                     0.5                   0.522984                 0.663775             0.363377      0.636623     0.738106 Moderate candidate
    pep_08 TVASTASAVATTAAKHHTKKKPHTKKKHH       0.871434                     0.5                   0.453259                 0.620731             0.289845      0.710155     0.651530 Moderate candidate
    pep_09 GGGLSGTAGAAATTFGSASGVGGSSGSFA       0.956531                     0.5                   0.479751                 0.631690             0.226472      0.773528     0.753843   Strong candidate
    pep_10 TLSVSGATVATAAGATALLSDAASLLTDA       0.838162                     0.5                   0.494809                 0.647939             0.258911      0.741089     0.705221 Moderate candidate
    pep_11 APSVAEVTVADVVAAVTVAEELMVAADEV       0.660508                     0.5                   0.309906                 0.488178             0.363495      0.636505     0.291901    Not recommended
    pep_12 SMLATLAGSEAAKSEPAAKKPSKRAAASK       0.873507                     0.5                   0.431928                 0.568246             0.338297      0.661703     0.580004    Not recommended
    pep_13 VVAGGGAVGAVVAVASPGAGTTTAGGAGV       0.993436                     0.5                   0.559634                 0.697798             0.247683      0.752317     0.883082   Strong candidate
    pep_14 AAALVVPPAGAVLVLVVPVLAAVAVAPAA       0.941485                     0.5                   0.537328                 0.640112             0.354877      0.645123     0.755869 Moderate candidate
    pep_15 PVAAPTSPPAPTASPAAAPTTTPPAAPPT       0.945812                     0.5                   0.485118                 0.570704             0.413909      0.586091     0.640139 Moderate candidate
    pep_16 RMMRLASTLLLLRPRLLLLLLLLLLLLLL       0.883628                     0.5                   0.513280                 0.651887             0.330163      0.669837     0.719687 Moderate candidate
    pep_17 GGGAGGGGAGGGGAGGGEGAAAGAAGAAA       0.500000                     0.5                   0.312500                 0.475558             0.258656      0.741344     0.248941    Not recommended
    pep_18 TMMITTMTMMTMTMVMTMIMIVTITIMTM       0.939737                     0.5                   0.536235                 0.691966             0.230463      0.769537     0.836089   Strong candidate
    pep_19 SMMTAMPMMMMAMMMMVMMMLMMMKMMMI       0.898562                     0.5                   0.467233                 0.614244             0.303251      0.696749     0.669871 Moderate candidate
    pep_20 TMVARTATVVMMMMMMMMTMVMMVMMMTM       0.884577                     0.5                   0.437468                 0.609937             0.247278      0.752722     0.653529 Moderate candidate
    pep_21 VMMVTTMMMMLTMVVMTMMMMMVMMTLMM       0.940339                     0.5                   0.536745                 0.691244             0.245240      0.754760     0.830376   Strong candidate
    pep_22 SMTMMMMTMMMTMMMMMMMMMMMMMMMMM       0.978632                     0.5                   0.553207                 0.709866             0.240050      0.759950     0.879478   Strong candidate
    pep_23 LMMVMMSTLRMMMTVVAMAMMMMLMLMLL       0.855088                     0.5                   0.501726                 0.630448             0.309571      0.690429     0.689407 Moderate candidate
    pep_24 VMMLRMVVMMMMMIMMMMMMMTMMTMMMM       0.925291                     0.5                   0.530512                 0.685040             0.269171      0.730829     0.802944   Strong candidate
    pep_25 EMMLLLLLLLLLLLLLLLLLLLLLLLLLL       0.927810                     0.5                   0.531543                 0.676005             0.299808      0.700192     0.787232   Strong candidate
    pep_26 LMVASSARLVLVLLMLLWWWWWWWWWWWW       0.859189                     0.5                   0.455960                 0.603541             0.309034      0.690966     0.629884 Moderate candidate
    pep_27 RMVAATATTMMAMLMLPVARMLLMLLLLM       0.889236                     0.5                   0.459997                 0.605856             0.318210      0.681790     0.646738 Moderate candidate
    pep_28 PMKPPLRAVSSTTMMRAPPLPLPPPLLPL       0.827843                     0.5                   0.367624                 0.512302             0.392454      0.607546     0.436860    Not recommended
    pep_29 APDAPTALDDAAATDDDGAAAIPAAPAAT       0.742049                     0.5                   0.324884                 0.497818             0.380795      0.619205     0.346802    Not recommended
    pep_30 LLARLVPDAAAEGSGAAAVVVTAGASAGA       0.868658                     0.5                   0.400026                 0.580213             0.248543      0.751457     0.589776    Not recommended
    pep_31 TMMTMMTTMMMMTMMMMTMVTTMMMMMTM       0.948239                     0.5                   0.540189                 0.697414             0.224087      0.775913     0.850227   Strong candidate
    pep_32 EMKNEDDDDDEDDGDEDDDEDKDDKDGND       0.858707                     0.5                   0.391400                 0.551901             0.470977      0.529023     0.467446    Not recommended
    pep_33 EMMAKEMVMTMMVTVTKMMMMVLVRTMTM       0.902752                     0.5                   0.466769                 0.625772             0.258584      0.741416     0.696824 Moderate candidate
    pep_34 LMLLMLLMMLVPLLLLARLVVLLMLLVMM       0.898428                     0.5                   0.469151                 0.621063             0.302355      0.697645     0.676150 Moderate candidate
    pep_35 VMKQKMKMAVAMMMVVVEVTVMVMMIMKE       0.871234                     0.5                   0.357856                 0.525552             0.323532      0.676468     0.485397    Not recommended
    pep_36 LMMMMMMMLMMLMMMMMLMTLLLLLLLLL       0.916003                     0.5                   0.475194                 0.639286             0.268060      0.731940     0.716100 Moderate candidate
    pep_37 VGSLGAGGSGLGSGLLGGSAGFLGGFFFF       0.954148                     0.5                   0.536651                 0.679578             0.228298      0.771702     0.837332   Strong candidate
    pep_38 AAAITATGAVAVSPTTLATGTSYASAGLA       0.952407                     0.5                   0.541961                 0.679508             0.253356      0.746644     0.831380   Strong candidate
    pep_39 VLVGVGVAFLVFGGGAEVAVATVAFEGAA       0.734967                     0.5                   0.323691                 0.510345             0.276371      0.723629     0.392573    Not recommended
    pep_40 TMMMMLMLRMPMTLMPMRTLMMPIMMMML       0.850963                     0.5                   0.446387                 0.578243             0.343706      0.656294     0.586772    Not recommended
    pep_41 LMMLMMMMPMMMLMMMLLLLLLLLLLLLL       0.909888                     0.5                   0.473101                 0.625117             0.298741      0.701259     0.689813 Moderate candidate
    pep_42 VRRRRLLLLLLLLLLLLLLLLLLLLLLLL       0.858413                     0.5                   0.455992                 0.563479             0.421223      0.578777     0.559384    Not recommended
    pep_43 AALLFPAFVLAFRDLAARFDAFFRALAAA       0.861005                     0.5                   0.456188                 0.600905             0.304623      0.695377     0.631284 Moderate candidate
    pep_44 TMMMMMMMMMMMMVMMVMMLLVVVLVLVV       0.933601                     0.5                   0.476074                 0.641657             0.272196      0.727804     0.725600 Moderate candidate
    pep_45 SMMINIMRMMIMVMTMMIMIRMMLMIMMT       0.860333                     0.5                   0.456363                 0.609591             0.291593      0.708407     0.641665 Moderate candidate
    pep_46 VMVRTRTMRVTMLMVAVTVMVMVMMVMMV       0.883057                     0.5                   0.463327                 0.629223             0.281735      0.718265     0.675897 Moderate candidate
    pep_47 TMMMMMMMMMMMMMMMMMMMMMMMMMMMM       0.921965                     0.5                   0.363919                 0.557168             0.250298      0.749702     0.566204    Not recommended
    pep_48 EMEKEGEEKEEKKMEKMEEDKKKEMMMMM       0.890656                     0.5                   0.447873                 0.567275             0.339207      0.660793     0.603702 Moderate candidate
    pep_49 RMMTIAAPMPAMMPLASVRPPRISAMKPP       0.828859                     0.5                   0.490778                 0.595232             0.397377      0.602623     0.607966 Moderate candidate
    pep_50 TMMRMTMMMMTIMRMMTMMMMTRMRRMRM       0.844686                     0.5                   0.421732                 0.592782             0.317480      0.682520     0.578384    Not recommended
    pep_51 LMLRLVLLMLLLVLLLLLLMLLLLLLVLL       0.891172                     0.5                   0.466872                 0.617051             0.305483      0.694517     0.666514 Moderate candidate
    pep_52 TMMTTMTMTLMTTMMKMTMTRMTMMTTMK       0.736394                     0.5                   0.324113                 0.511686             0.256119      0.743881     0.402876    Not recommended
    pep_53 LMMLMIMSMKMLMVMMTLMIMMLMMMVMM       0.910541                     0.5                   0.472742                 0.613365             0.316317      0.683683     0.675539 Moderate candidate
    pep_54 EMGAGAAGSAALTVGGVVGAAVAVAVSAV       0.949012                     0.5                   0.452962                 0.627917             0.243122      0.756878     0.714097 Moderate candidate
    pep_55 AAAAAAAVAAAVAVAAVAAVAAVVAAVAV       0.962865                     0.5                   0.546426                 0.690228             0.290922      0.709078     0.832021   Strong candidate
    pep_56 GGGGGGAGLGAGLGGAGGGGGGADGGDAG       0.967991                     0.5                   0.548623                 0.665058             0.277904      0.722096     0.827217   Strong candidate
    pep_57 TMMMMMATAATKTLVTITLTMRMLMSATM       0.909713                     0.5                   0.472859                 0.624670             0.274109      0.725891     0.699383 Moderate candidate
    pep_58 PMMKLVKIVMMKRMMLTMRMMMLLLILLL       0.838499                     0.5                   0.447958                 0.608015             0.334583      0.665417     0.603506 Moderate candidate
    pep_59 LMMSMMMLSMMTMMMMALMTMAMVLMPTL       0.927843                     0.5                   0.530167                 0.648371             0.315264      0.684736     0.763053 Moderate candidate
    pep_60 LVLSESSLGAVLASVSSVSDSFSVSVVDV       0.929218                     0.5                   0.469752                 0.610908             0.313010      0.686990     0.681912 Moderate candidate
    pep_61 AAAVAAAAAGATVAVAAVAVAVAVAVVTV       0.981659                     0.5                   0.431820                 0.609489             0.267734      0.732266     0.688352 Moderate candidate
    pep_62 PPLPLPPLPPLPPLLPPLLPPLLLLLLLL       0.891011                     0.5                   0.392797                 0.515012             0.413882      0.586118     0.486833    Not recommended
    pep_63 LMMLLLLLLLLLLLLLLPLLLLLLLLLLL       0.931095                     0.5                   0.480721                 0.631941             0.300381      0.699619     0.711569 Moderate candidate
    pep_64 VMMMMMMMMMMMMMMMMMMMMMMMMMMMM       0.913977                     0.5                   0.474591                 0.640111             0.249665      0.750335     0.722553 Moderate candidate
    pep_65 LMMMMMMMMMMMMMMMMMMMMMMMMMMMM       0.918233                     0.5                   0.527561                 0.681764             0.249254      0.750746     0.802686   Strong candidate
    pep_66 SMMMTVITMMAIMRMVVITTRMAMAMTMT       0.859066                     0.5                   0.348948                 0.533666             0.275433      0.724567     0.494993    Not recommended
    pep_67 PMALGSVLPLLRLLVLRVSLLQLLLLALL       0.855986                     0.5                   0.429520                 0.568075             0.342315      0.657685     0.566957    Not recommended
    pep_68 AALAADEDDDDEDEDEDDEEDEDEEDEDD       0.846662                     0.5                   0.446159                 0.545567             0.587500      0.412500     0.464299    Not recommended
    pep_69 TGTSLTITPDSNGYYYWYSVYYWSDPATA       0.721067                     0.5                   0.321050                 0.508358             0.263148      0.736852     0.387163    Not recommended
    pep_70 AAGGGAAGGGGAVAALGARARALLRASDE       0.837410                     0.5                   0.494302                 0.627416             0.265350      0.734650     0.689466 Moderate candidate
    pep_71 TVVTILTTSTPVATAIAPLVLATAVPTTT       0.937436                     0.5                   0.418439                 0.576195             0.283208      0.716792     0.626357 Moderate candidate
    pep_72 AAAVAAAVTAVAVVAPVVTAAVAVLAMLA       0.964303                     0.5                   0.538945                 0.677828             0.293310      0.706690     0.816898   Strong candidate
    pep_73 SMIMMTMPMMMMVMMVMMIVMVMMIMVLM       0.922753                     0.5                   0.356450                 0.543370             0.289544      0.710456     0.534719    Not recommended
    pep_74 EMEEEEDEDEEDDAEEDEDEEDEDDEEEE       0.856941                     0.5                   0.395415                 0.503605             0.587500      0.412500     0.393745    Not recommended
    pep_75 VMVVEVVVMEEVVVVEVVMMVEMVVVMVM       0.921192                     0.5                   0.475956                 0.624077             0.363290      0.636710     0.671137 Moderate candidate
    pep_76 VMMVAVLMMMMVMVVMVTLMVMLVMVMLV       0.810395                     0.5                   0.433390                 0.595902             0.276374      0.723626     0.591504    Not recommended
    pep_77 PMKMMIIKMIIMKMMIIMKIVMMKMMMMM       0.859257                     0.5                   0.425013                 0.571239             0.384393      0.615607     0.548616    Not recommended
    pep_78 RMTRTMKRTMRTETMMMRMMMMMVMTLKM       0.734023                     0.5                   0.323793                 0.504145             0.333825      0.666175     0.364795    Not recommended
    pep_79 ALAADEERALGDGGAGGDDGARVGLALEV       0.933893                     0.5                   0.534124                 0.678057             0.307651      0.692349     0.790881 Moderate candidate
    pep_80 LMMMLMLMMLMMMLLMMMLMMLMPMMMLM       0.923477                     0.5                   0.472543                 0.628039             0.295613      0.704387     0.699169 Moderate candidate
    pep_81 SMMMMIMMMMIMMVMMMMIMMMMMMMMMM       0.948142                     0.5                   0.540148                 0.694186             0.261836      0.738164     0.832636   Strong candidate
    pep_82 DMDLSDDDEGMDMDGMDDEDEDLVDEDSL       0.739152                     0.5                   0.323951                 0.497581             0.494748      0.505252     0.297225    Not recommended
    pep_83 LMKVNVVNVLSVALVLAVALLESLVVLVV       0.924005                     0.5                   0.472192                 0.626989             0.269375      0.730625     0.709291 Moderate candidate
    pep_84 DMVDDLVADDMHVAPDDAVFPDDDRLPRD       0.617983                     0.5                   0.374980                 0.523943             0.399782      0.600218     0.341799    Not recommended
    pep_85 DVADDDDDVDGDDVDDADADDADDADDAA       0.897277                     0.5                   0.446740                 0.618969             0.446839      0.553161     0.592249    Not recommended
    pep_86 LMMMLMMMLMMMLMMMLMMMLMMMMLMML       0.913767                     0.5                   0.523952                 0.677037             0.269587      0.730413     0.785605   Strong candidate
    pep_87 LMSRKKKEKKRRMKMRMRKMRMKRTMWKR       0.827002                     0.5                   0.436256                 0.525616             0.584400      0.415600     0.433835    Not recommended
    pep_88 VMMLMLMMLMLMMLMMMLMMMLMMMLMMM       0.925503                     0.5                   0.477369                 0.641873             0.274691      0.725309     0.721892 Moderate candidate
    pep_89 VMKEATVAEAVMEEAGIIAVTAMAIVVEA       0.909623                     0.5                   0.467138                 0.622492             0.323369      0.676631     0.671983 Moderate candidate
    pep_90 SMIRASASARARSARSSSSRAVTTSRASG       0.898362                     0.5                   0.432259                 0.556303             0.380673      0.619327     0.568333    Not recommended
    pep_91 DMVADDDGDGAGEDVVIVIVLIVAVVAVV       0.833913                     0.5                   0.440225                 0.607901             0.338344      0.661656     0.591821    Not recommended
    pep_92 GGGEDGDGGDGCGEGGDEGDGGCGEGDGE       0.914523                     0.5                   0.408175                 0.571238             0.357227      0.642773     0.570973    Not recommended
    pep_93 APGAASSGSGEGPSAPTAVAEDEDEDDDE       0.747142                     0.5                   0.325833                 0.488620             0.424097      0.575903     0.326987    Not recommended
    pep_94 DDEDEDEDDEDDDEDDEDEAEDDEDDEDE       0.882137                     0.5                   0.462175                 0.569177             0.587500      0.412500     0.512351    Not recommended
    pep_95 RLPAAASGPARGPRGRARRGGRRGGRSRR       0.851719                     0.5                   0.453377                 0.538690             0.513970      0.486030     0.500346    Not recommended
    pep_96 GGEAVVGEVVVVAVVVAVVSVVVLVVVVA       0.889537                     0.5                   0.514840                 0.661979             0.288867      0.711133     0.747292 Moderate candidate
    pep_97 LMPRVLMLRAVLMMLGLAVALRSMLMMLL       0.862017                     0.5                   0.437760                 0.569794             0.369327      0.630673     0.568135    Not recommended
    pep_98 AALRSALGAELAAQPDSAPAAAAAVAPAV       0.876062                     0.5                   0.452876                 0.570263             0.362403      0.637597     0.593520    Not recommended
    pep_99 SMMVALSSAGAMSPSARAMTSSGALMSSP       0.932318                     0.5                   0.417039                 0.535438             0.359383      0.640617     0.566675    Not recommended
   pep_100 PLPLQSQLELEPLQLQLPLQEPLQLLQLL       0.723532                     0.5                   0.321465                 0.457719             0.477126      0.522874     0.270382    Not recommended
```
